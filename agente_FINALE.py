#!/usr/bin/env python3
"""
parkinson_agent_crnn.py
=======================

Agente LangGraph per lo screening preliminare del Parkinson da audio vocale,
basato sul modello CRNN selezionato per massimizzare la sensibilità.

Pipeline:
  Fase 0  →  Conversione audio in WAV mono 16 kHz
  Fase 1  →  Preprocessing coerente con CRNN_FINALE.py:
              normalize, trim top_db=35, log-Mel, delta, delta-delta,
              padding/truncation a [3, 64, 1024]
  Fase 2  →  Inferenza CRNNClassifier + sigmoid
  Fase 3  →  Referto bilingue ibrido:
              template deterministico + LLM locale solo per suggerimenti clinici

Dipendenze:
  pip install langgraph librosa torch matplotlib numpy soundfile requests

Nota:
  Questo agente è uno strumento di screening preliminare. Non produce diagnosi
  medica e non sostituisce una valutazione neurologica clinica.
"""
from __future__ import annotations

import json
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# 0. COSTANTI — coerenti con CRNN_FINALE.py
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
N_MELS = 64
HOP_LENGTH = 160
WIN_LENGTH = 400
FMIN = 50
FMAX = 4_000
MAX_FRAMES = 1_024
SPEC_PAD_VALUE = -80.0

FALLBACK_THRESHOLD = 0.5

# Qualità audio minima accettabile dopo trim.
MIN_DURATION_S = 0.5

# Livelli di rischio visualizzati nel referto.
RISK_LOW = 0.35
RISK_HIGH = 0.65

CHECKPOINT_DIR = Path("milestone2/checkpoints")
OUTPUT_DIR = Path("output_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".m4a", ".mp4", ".aac",
    ".flac", ".ogg", ".opus", ".aiff", ".aif",
    ".wma", ".webm", ".3gp", ".amr",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATO CONDIVISO
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    audio_path: str
    original_path: str
    patient_name: str

    # Preprocessing
    audio_quality_ok: bool
    quality_reason: str
    duration_s: float
    spectrogram_path: Optional[str]
    tensor_path: Optional[str]

    # Inferenza
    prob_score: float
    threshold: float
    is_parkinson: bool

    # Referto
    report_it: str
    report_en: str
    report_path: Optional[str]

    # Controllo flusso
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELLO CRNN — stessa architettura del training
# ─────────────────────────────────────────────────────────────────────────────

class CRNNClassifier(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden_size: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
        )
        self.lstm = nn.LSTM(
            input_size=64 * (n_mels // 8),
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                    # (B, C, F, T)
        x = x.permute(0, 3, 1, 2)          # (B, T, C, F)
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)

        out, _ = self.lstm(x)              # (B, T, 2H)
        out_max, _ = torch.max(out, dim=1)
        out_avg = torch.mean(out, dim=1)

        out = torch.cat([out_max, out_avg], dim=1)  # (B, 4H)
        out = self.dropout(out)

        return self.classifier(out).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARICAMENTO CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict = {}


def _load_best_checkpoint() -> tuple[CRNNClassifier, float, torch.device]:
    """
    Carica il modello CRNN finale.

    Priorità:
      1. milestone2/checkpoints/best_model_final.pt
         contiene model_state_dict + threshold + metriche.
      2. milestone2/checkpoints/best_crnn.pt
         contiene model_state_dict, ma potrebbe non contenere threshold.
         In quel caso viene usata FALLBACK_THRESHOLD = 0.416.
    """
    if _model_cache:
        return _model_cache["model"], _model_cache["threshold"], _model_cache["device"]

    ckpt_path = CHECKPOINT_DIR / "best_model_final.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "best_crnn.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Nessun checkpoint trovato in {CHECKPOINT_DIR}. "
            "Esegui prima il training con CRNN_FINALE.py."
        )

    print(f"[Agent] Carico checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Il checkpoint {ckpt_path} non contiene 'model_state_dict'. "
            "Controlla il file salvato dal training."
        )

    model = CRNNClassifier()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    if "threshold" in ckpt:
        threshold = float(ckpt["threshold"])
    else:
        threshold = FALLBACK_THRESHOLD
        print(
            f"[Agent] [WARN] Threshold non trovata nel checkpoint. "
            f"Uso fallback {threshold:.4f}."
        )

    val_auc = ckpt.get("best_val_auc", float("nan"))

    print(f"[Agent] Device: {device}")
    print(f"[Agent] Soglia decisionale caricata: {threshold:.4f}")
    print(f"[Agent] ROC-AUC validation checkpoint: {val_auc:.3f}")

    _model_cache["model"] = model
    _model_cache["threshold"] = threshold
    _model_cache["device"] = device

    return model, threshold, device


# ─────────────────────────────────────────────────────────────────────────────
# 4. NODI LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────

def node_convert_to_wav(state: AgentState) -> AgentState:
    """Converte qualunque formato supportato in WAV mono 16 kHz."""
    import shutil as _shutil
    import subprocess

    src = Path(state["audio_path"])
    ext = src.suffix.lower()

    print(f"[Convert] File ricevuto: {src.name}  (estensione: {ext or 'nessuna'})")

    if not src.exists():
        return {
            **state,
            "audio_quality_ok": False,
            "quality_reason": f"File non trovato: {src}",
            "original_path": state["audio_path"],
        }

    if ext == ".wav":
        print("[Convert] Formato già WAV — conversione non necessaria.")
        return {
            **state,
            "audio_path": str(src),
            "original_path": state["audio_path"],
        }

    if ext and ext not in SUPPORTED_FORMATS:
        supported_str = ", ".join(sorted(SUPPORTED_FORMATS))
        return {
            **state,
            "audio_quality_ok": False,
            "quality_reason": (
                f"Formato '{ext}' non supportato. "
                f"Formati accettati: {supported_str}."
            ),
            "original_path": state["audio_path"],
        }

    tmp_dir = Path(tempfile.mkdtemp())
    wav_dst = tmp_dir / f"{src.stem}_converted.wav"

    ffmpeg_bin = _shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(src),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-sample_fmt", "s16",
            str(wav_dst),
        ]
        print(f"[Convert] ffmpeg → {wav_dst.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and wav_dst.exists():
            print("[Convert] ✔ Conversione ffmpeg riuscita.")
            return {
                **state,
                "audio_path": str(wav_dst),
                "original_path": state["audio_path"],
            }

        print(f"[Convert] ✘ ffmpeg fallito:\n{result.stderr[-400:]}")
    else:
        print("[Convert] ffmpeg non trovato nel PATH — uso fallback librosa.")

    try:
        import soundfile as sf

        print(f"[Convert] librosa+soundfile → {wav_dst.name}")
        y, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
        sf.write(str(wav_dst), y, SAMPLE_RATE, subtype="PCM_16")
        print("[Convert] ✔ Conversione librosa riuscita.")
        return {
            **state,
            "audio_path": str(wav_dst),
            "original_path": state["audio_path"],
        }

    except Exception as exc:
        print(f"[Convert] ✘ librosa fallito: {exc}")

    return {
        **state,
        "audio_quality_ok": False,
        "quality_reason": (
            f"Impossibile convertire '{src.name}' in WAV mono a 16 kHz. "
            "Installa ffmpeg oppure converti manualmente il file."
        ),
        "original_path": state["audio_path"],
    }


def _build_crnn_tensor(y: np.ndarray) -> np.ndarray:
    """
    Costruisce il tensore [3, 64, 1024] coerente con il training CRNN:
      canale 0: log-Mel
      canale 1: delta
      canale 2: delta-delta
    """
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    logmel = librosa.power_to_db(melspec, ref=np.max).astype(np.float32)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    full = np.stack([logmel, delta, delta2], axis=0).astype(np.float32)

    T = full.shape[2]
    if T >= MAX_FRAMES:
        full = full[:, :, :MAX_FRAMES]
    else:
        pad = MAX_FRAMES - T
        full = np.pad(
            full,
            ((0, 0), (0, 0), (0, pad)),
            constant_values=SPEC_PAD_VALUE,
        )

    return full.astype(np.float32)


def node_preprocess(state: AgentState) -> AgentState:
    """
    Esegue preprocessing coerente con CRNN_FINALE.py:
      - load mono 16 kHz
      - normalize
      - trim top_db=35
      - log-Mel + delta + delta-delta
      - padding/truncation a [3, 64, 1024]
      - spettrogramma PNG per visualizzazione/referto
    """
    wav_path = Path(state["audio_path"])
    print(f"[Preprocess] File: {wav_path.name}")

    try:
        y, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        return {
            **state,
            "audio_quality_ok": False,
            "quality_reason": f"Impossibile leggere il file audio: {exc}",
        }

    if y.size == 0:
        return {
            **state,
            "audio_quality_ok": False,
            "quality_reason": "Audio vuoto o non leggibile.",
        }

    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=35)

    duration_s = len(y) / SAMPLE_RATE

    if duration_s < MIN_DURATION_S:
        print(f"[Preprocess] RIFIUTO — durata {duration_s:.2f}s < {MIN_DURATION_S}s")
        return {
            **state,
            "audio_quality_ok": False,
            "quality_reason": (
                f"Audio troppo breve ({duration_s:.2f}s). "
                f"Minimo richiesto: {MIN_DURATION_S}s. "
                "Pronunciare la vocale /a/ sostenuta per almeno 0.5 secondi."
            ),
            "duration_s": duration_s,
        }

    full = _build_crnn_tensor(y)

    tmp_dir = Path(tempfile.mkdtemp())
    tensor_npy = tmp_dir / "crnn_tensor.npy"
    np.save(tensor_npy, full)

    # PNG solo per visualizzazione/referto.
    # Usiamo il primo canale del tensore già costruito.
    logmel_for_plot = full[0]
    stem = wav_path.stem
    spec_png = OUTPUT_DIR / f"{stem}_spectrogram.png"

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        logmel_for_plot,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram — {wav_path.name}")
    plt.tight_layout()
    plt.savefig(spec_png, dpi=100)
    plt.close()

    print(f"[Preprocess] Durata utile: {duration_s:.2f}s")
    print(f"[Preprocess] Tensore CRNN salvato: {tensor_npy}")
    print(f"[Preprocess] Spettrogramma salvato: {spec_png}")

    return {
        **state,
        "audio_quality_ok": True,
        "quality_reason": "ok",
        "duration_s": duration_s,
        "spectrogram_path": str(spec_png),
        "tensor_path": str(tensor_npy),
    }


def node_inference(state: AgentState) -> AgentState:
    """Esegue inferenza CRNN e restituisce score PD e decisione binaria."""
    print("[Inference] Carico modello CRNN e checkpoint…")
    model, threshold, device = _load_best_checkpoint()

    if not state.get("tensor_path"):
        return {
            **state,
            "error": "Tensore CRNN mancante: preprocessing non completato.",
        }

    spec = np.load(state["tensor_path"]).astype(np.float32)
    x = torch.from_numpy(spec).unsqueeze(0).to(device)  # [1, 3, 64, 1024]

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    is_pd = prob > threshold

    print(
        f"[Inference] score_PD={prob:.4f}  "
        f"threshold={threshold:.4f}  →  {'POSITIVE' if is_pd else 'NEGATIVE'}"
    )

    return {
        **state,
        "prob_score": prob,
        "threshold": threshold,
        "is_parkinson": is_pd,
    }


def _risk_label(prob: float) -> tuple[str, str]:
    if prob < RISK_LOW:
        return "Basso / Low", "LOW"
    if prob < 0.5:
        return "Medio-basso / Medium-low", "MEDIUM-LOW"
    if prob < RISK_HIGH:
        return "Medio-alto / Medium-high", "MEDIUM-HIGH"
    return "Alto / High", "HIGH"


def _prob_bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {prob:.1%}"


def _fallback_suggestions() -> tuple[str, str]:
    it = (
        "Il risultato suggerisce la presenza di segnali vocali che meritano "
        "un approfondimento clinico. Non si tratta di una diagnosi: il passo "
        "più corretto è parlarne con il medico di base, che potrà valutare "
        "se indirizzare il paziente a una visita neurologica. È utile portare "
        "con sé questo referto e, se possibile, ripetere la registrazione in "
        "condizioni ambientali silenziose."
    )
    en = (
        "The result suggests the presence of vocal signs that deserve clinical "
        "follow-up. This is not a diagnosis: the most appropriate next step is "
        "to discuss it with a general practitioner, who can evaluate whether a "
        "neurological consultation is needed. It may be useful to bring this "
        "report and, if possible, repeat the recording in a quiet environment."
    )
    return it, en


def _llm_clinical_suggestions(duration_s: float) -> tuple[str, str]:
    """
    Usa un LLM locale via Ollama per generare suggerimenti clinici.
    In caso di errore, ritorna un template sicuro.
    """
    import requests

    prompt = textwrap.dedent(f"""
        Sei un medico neurologo che scrive una breve nota di orientamento
        per un paziente non esperto.

        Un sistema automatico di screening vocale ha rilevato segnali vocali
        compatibili con un possibile rischio neurologico da approfondire.
        La registrazione vocale analizzata aveva una durata di {duration_s:.1f}
        secondi.

        Scrivi una sezione "Suggerimenti clinici" chiara e accessibile.

        Regole:
        - NON usare parole come CRNN, modello, soglia, F1, algoritmo,
          intelligenza artificiale, probabilità, score o percentuale.
        - NON formulare una diagnosi.
        - Tono rassicurante ma serio, mai allarmistico.
        - Indica come passo successivo il medico di base e, se opportuno,
          una valutazione neurologica.
        - 4-5 frasi massimo.

        Rispondi SOLO con un oggetto JSON valido:
        {{
          "it": "testo in italiano",
          "en": "same content in English"
        }}
    """)

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "medgemma",
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()
        data = json.loads(response.json()["response"])
        return data["it"], data["en"]

    except Exception as exc:
        print(f"[Report] [WARN] LLM non disponibile o risposta non valida: {exc}")
        return _fallback_suggestions()


def _build_report(
    lang: str,
    audio_path: str,
    patient_name: str,
    prob: float,
    threshold: float,
    duration_s: float,
    is_pd: bool,
    risk_label: str,
    suggestions: str,
    spectrogram_path: Optional[str],
    timestamp: str,
) -> str:
    audio_name = Path(audio_path).name
    spec_line = spectrogram_path or "non disponibile"
    patient_line = patient_name.strip() if patient_name and patient_name.strip() else "N/D"

    if lang == "it":
        result_str = (
            "POSITIVO — Segnali vocali compatibili con possibile rischio Parkinson"
            if is_pd
            else "NEGATIVO — Nessun segnale vocale rilevante nel campione analizzato"
        )
        note_neg = (
            "L'analisi automatica del segnale vocale non ha evidenziato pattern "
            "acustici rilevanti nel campione analizzato. Questo risultato non "
            "esclude clinicamente la presenza della malattia, soprattutto in caso "
            "di sintomi o familiarità. Si raccomanda di ripetere lo screening "
            "periodicamente e di rivolgersi a un medico in caso di dubbi."
        )
        disclaimer = (
            "⚠  AVVERTENZA: Questo referto è prodotto da un sistema automatico "
            "di screening vocale. Non costituisce diagnosi medica e non sostituisce "
            "una valutazione neurologica clinica."
        )
        t = {
            "header": "REFERTO DI SCREENING VOCALE — MORBO DI PARKINSON",
            "s1": "DATI CAMPIONE AUDIO",
            "s2": "RISULTATO DELLO SCREENING",
            "s3": "SUGGERIMENTI CLINICI" if is_pd else "NOTA CLINICA",
            "patient": "Paziente           ",
            "file": "File audio         ",
            "dur": "Durata utile       ",
            "spec": "Spettrogramma      ",
            "date": "Data analisi       ",
            "score": "Indice vocale PD   ",
            "thresh": "Soglia decisionale ",
            "risk": "Livello di rischio ",
            "result": "Esito              ",
        }
    else:
        result_str = (
            "POSITIVE — Vocal patterns compatible with possible Parkinson's risk"
            if is_pd
            else "NEGATIVE — No relevant vocal anomaly detected in the analyzed sample"
        )
        note_neg = (
            "The automated analysis of the vocal signal did not detect relevant "
            "acoustic patterns in the analyzed sample. This result does not "
            "clinically exclude the disease, especially in the presence of symptoms "
            "or family history. Periodic repetition of the screening is recommended, "
            "as well as medical consultation in case of concerns."
        )
        disclaimer = (
            "⚠  WARNING: This report is produced by an automated vocal screening "
            "system. It is not a medical diagnosis and does not replace a clinical "
            "neurological evaluation."
        )
        t = {
            "header": "VOCAL SCREENING REPORT — PARKINSON'S DISEASE",
            "s1": "AUDIO SAMPLE DATA",
            "s2": "SCREENING RESULT",
            "s3": "CLINICAL SUGGESTIONS" if is_pd else "CLINICAL NOTE",
            "patient": "Patient            ",
            "file": "Audio file         ",
            "dur": "Useful duration    ",
            "spec": "Spectrogram        ",
            "date": "Analysis date      ",
            "score": "PD vocal index     ",
            "thresh": "Decision threshold ",
            "risk": "Risk level         ",
            "result": "Result             ",
        }

    sep = "─" * 62
    body = suggestions if is_pd else note_neg

    return (
        f"{t['header']}\n{sep}\n\n"
        f"{t['s1']}\n"
        f"  {t['patient']}: {patient_line}\n"
        f"  {t['file']}: {audio_name}\n"
        f"  {t['dur']}: {duration_s:.1f} s\n"
        f"  {t['spec']}: {spec_line}\n"
        f"  {t['date']}: {timestamp}\n\n"
        f"{t['s2']}\n"
        f"  {t['score']}: {_prob_bar(prob)}\n"
        f"  {t['thresh']}: {threshold:.1%}\n"
        f"  {t['risk']}: {risk_label}\n"
        f"  {t['result']}: {result_str}\n\n"
        f"{t['s3']}\n"
        f"{body}\n\n"
        f"{sep}\n"
        f"{disclaimer}\n"
    )


def node_report(state: AgentState) -> AgentState:
    """Compone il referto bilingue."""
    prob = state["prob_score"]
    threshold = state["threshold"]
    is_pd = state["is_parkinson"]
    duration = state["duration_s"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    risk_it, risk_en = _risk_label(prob)

    if is_pd:
        print("[Report] Esito positivo — genero suggerimenti clinici…")
        sugg_it, sugg_en = _llm_clinical_suggestions(duration)
    else:
        print("[Report] Esito negativo — uso nota standard.")
        sugg_it = ""
        sugg_en = ""

    report_it = _build_report(
        lang="it",
        audio_path=state["audio_path"],
        patient_name=state.get("patient_name", ""),
        prob=prob,
        threshold=threshold,
        duration_s=duration,
        is_pd=is_pd,
        risk_label=risk_it,
        suggestions=sugg_it,
        spectrogram_path=state.get("spectrogram_path"),
        timestamp=timestamp,
    )
    report_en = _build_report(
        lang="en",
        audio_path=state["audio_path"],
        patient_name=state.get("patient_name", ""),
        prob=prob,
        threshold=threshold,
        duration_s=duration,
        is_pd=is_pd,
        risk_label=risk_en,
        suggestions=sugg_en,
        spectrogram_path=state.get("spectrogram_path"),
        timestamp=timestamp,
    )

    stem = Path(state["audio_path"]).stem
    safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{stem}_report_{safe_ts}.txt"

    separator = "\n" + "═" * 62 + "\n\n"
    out_path.write_text(report_it + separator + report_en, encoding="utf-8")

    print(f"[Report] Referto salvato: {out_path}")

    return {
        **state,
        "report_it": report_it,
        "report_en": report_en,
        "report_path": str(out_path),
    }


def node_reject(state: AgentState) -> AgentState:
    """Termina il flusso quando l'audio non è idoneo."""
    reason = state.get("quality_reason") or state.get("error") or "Motivo non specificato."
    print(f"\n[Reject] Audio rifiutato.\nMotivo: {reason}\n")
    return {
        **state,
        "error": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def route_convert(state: AgentState) -> str:
    if not state.get("audio_quality_ok", True) and state.get("quality_reason"):
        return "reject"
    return "preprocess"


def route_quality(state: AgentState) -> str:
    return "inference" if state.get("audio_quality_ok") else "reject"


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRAFO
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("convert", node_convert_to_wav)
    graph.add_node("preprocess", node_preprocess)
    graph.add_node("inference", node_inference)
    graph.add_node("report", node_report)
    graph.add_node("reject", node_reject)

    graph.set_entry_point("convert")

    graph.add_conditional_edges(
        "convert",
        route_convert,
        {
            "preprocess": "preprocess",
            "reject": "reject",
        },
    )

    graph.add_conditional_edges(
        "preprocess",
        route_quality,
        {
            "inference": "inference",
            "reject": "reject",
        },
    )

    graph.add_edge("inference", "report")
    graph.add_edge("report", END)
    graph.add_edge("reject", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(audio_path: str, patient_name: str = "") -> AgentState:
    """Esegue l'agente su un singolo file audio."""
    app = build_graph()

    initial_state: AgentState = {
        "audio_path": audio_path,
        "original_path": audio_path,
        "patient_name": patient_name,
        "audio_quality_ok": False,
        "quality_reason": "",
        "duration_s": 0.0,
        "spectrogram_path": None,
        "tensor_path": None,
        "prob_score": 0.0,
        "threshold": FALLBACK_THRESHOLD,
        "is_parkinson": False,
        "report_it": "",
        "report_en": "",
        "report_path": None,
        "error": None,
    }

    print(f"\n{'=' * 62}")
    print("  Parkinson Screening Agent — CRNN")
    print(f"  File: {Path(audio_path).name}")
    print(f"{'=' * 62}\n")

    final_state = app.invoke(initial_state)

    print(f"\n{'=' * 62}")
    if final_state.get("error"):
        print("  AGENTE TERMINATO — audio non idoneo")
        print(f"  {final_state['error']}")
    else:
        print("  AGENTE COMPLETATO")
        print(f"  Referto: {final_state['report_path']}")
        print(f"\n{final_state['report_it']}")
    print(f"{'=' * 62}\n")

    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Agente LangGraph — screening Parkinson da audio vocale con CRNN"
    )
    parser.add_argument(
        "audio",
        help="Path al file audio da analizzare, preferibilmente vocalizzazione /a/ sostenuta.",
    )
    parser.add_argument(
        "--patient-name",
        default="",
        help="Nome del paziente da riportare nel referto.",
    )

    args = parser.parse_args()
    run_agent(args.audio, patient_name=args.patient_name)