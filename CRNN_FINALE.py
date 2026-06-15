#!/usr/bin/env python3
"""
CRNN.py — Rilevamento Parkinson da voce (versione corretta, augmentation-aware)
================================================================================

Struttura dei dati attesa:
    data/Training_augmented/HC/<SUBJECT_ID>.wav                  (originale)
                              <SUBJECT_ID>_noise.wav             (augmentazioni)
                              <SUBJECT_ID>_pitch_up.wav
                              <SUBJECT_ID>_pitch_down.wav
                              <SUBJECT_ID>_speed_up.wav
                              <SUBJECT_ID>_speed_down.wav
                              <SUBJECT_ID>.mask.npy              (mask, ignorate)
                              <SUBJECT_ID>.npy                   (spettrogramma)
                              ...
    data/Training_augmented/PD/...
    data/Test/HC/<SUBJECT_ID>.wav    (solo originali, niente augmentation)
    data/Test/PD/...

Protocollo metodologico:
  1. Split TRAIN / VAL a livello di SOGGETTO (le 6 augmentation di uno stesso
     soggetto restano tutte insieme, in train OPPURE in val).
  2. Selezione del miglior checkpoint su VAL AUC (subject-level: le 6
     probabilità di ciascun soggetto vengono mediate prima di calcolare l'AUC).
  3. Soglia ottima F1 cercata sul VAL set, applicata fissa sul TEST.
  4. TEST set valutato UNA SOLA VOLTA, alla fine.
  5. Bootstrap (1000 iter., CI 95%) sul TEST.

Esecuzione:
    python CRNN.py                  # training completo + test + bootstrap
    python CRNN.py --eval-only      # solo test + bootstrap (richiede checkpoint)
    python CRNN.py -e 30 --plot     # 30 epoche + salva spettrogrammi mel
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# ───────────────────────── CONFIG ─────────────────────────

TRAIN_DIR      = Path("Training_augmented")
TEST_DIR       = Path("Test")

PLOT_DIR       = Path("milestone2/plots")
CHECKPOINT_DIR = Path("milestone2/checkpoints")
STATS_DIR      = Path("milestone2/stats")
for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Audio / spettrogramma (identici alla versione originale per confronto fair con AST)
SAMPLE_RATE    = 16_000
N_MELS         = 64
HOP_LENGTH     = 160
WIN_LENGTH     = 400
FMIN           = 50
FMAX           = 4_000
MAX_FRAMES     = 1_024
SPEC_PAD_VALUE = -80.0

# Training
RANDOM_SEED   = 42
EPOCHS        = 15
BATCH_SIZE    = 8
NUM_WORKERS   = os.cpu_count() or 2
EPSILON       = 0.1            # label smoothing
LR            = 1e-5
VAL_SIZE      = 0.20           # frazione di SOGGETTI usati come val

# Bootstrap (stesso formato di AST.py)
BOOTSTRAP_N    = 1000
BOOTSTRAP_SEED = 42

LABELS = ('HC', 'PD')          # 0 = HC, 1 = PD

# Suffissi di augmentation effettivamente presenti nei tuoi file.
# IMPORTANTE: l'ordine non conta, ma DEVONO essere prefissati da underscore
# come nel naming reale ("..._noise", "..._pitch_up", ecc.).
AUG_SUFFIXES = ('_noise', '_pitch_up', '_pitch_down', '_speed_up', '_speed_down')

BEST_CKPT_PATH         = CHECKPOINT_DIR / "best_crnn.pt"
TEST_PREDICTIONS_PATH  = CHECKPOINT_DIR / "test_predictions_crnn.csv"
BOOTSTRAP_RESULTS_PATH = CHECKPOINT_DIR / "bootstrap_test_crnn.txt"


# ───────────────────────── UTILS ──────────────────────────

def seed_everything(seed: int = RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_subject_id(file_path: Path) -> str:
    """
    Estrae l'ID del soggetto rimuovendo il suffisso di augmentation.

    Esempi:
      AH_064F_7AB...C5.npy             →  AH_064F_7AB...C5
      AH_064F_7AB...C5_noise.npy       →  AH_064F_7AB...C5
      AH_064F_7AB...C5_pitch_down.npy  →  AH_064F_7AB...C5
      AH_064F_7AB...C5_speed_up.npy    →  AH_064F_7AB...C5

    Così tutte le augmentation dello stesso paziente finiscono insieme
    nello stesso split (train OPPURE val, mai entrambi).
    """
    stem = file_path.stem  # nome file senza ".npy"
    for suffix in AUG_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def is_original(file_path: Path) -> bool:
    """True se il file non ha suffisso di augmentation (versione originale)."""
    stem = file_path.stem
    return not any(stem.endswith(s) for s in AUG_SUFFIXES)


# ──────────────────── PREPROCESSING ───────────────────────

def load_and_preprocess(
    wav_path: Path,
    spec_path: Path,
    plot_path: Path | None,
    do_plot: bool,
) -> None:
    already_done = (
        spec_path.exists() and
        (not do_plot or (plot_path and plot_path.exists()))
    )
    if already_done:
        return

    y, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=35)

    melspec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    logmel = librosa.power_to_db(melspec, ref=np.max).astype(np.float32)
    delta  = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    full   = np.stack([logmel, delta, delta2], axis=0)   # (3, N_MELS, T)

    T = full.shape[2]
    if T >= MAX_FRAMES:
        full = full[:, :, :MAX_FRAMES]
    else:
        pad  = MAX_FRAMES - T
        full = np.pad(full, ((0, 0), (0, 0), (0, pad)), constant_values=SPEC_PAD_VALUE)

    spec_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(spec_path, full)

    if do_plot and plot_path and not plot_path.exists():
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            logmel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
            x_axis='time', y_axis='mel', fmin=FMIN, fmax=FMAX,
        )
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


def preprocess_split(wav_root: Path, plot_root: Path | None, do_plot: bool) -> None:
    """Preprocessa tutti i .wav in wav_root/{HC,PD} e salva .npy accanto ai .wav."""
    for lbl in LABELS:
        wav_dir = wav_root / lbl
        if not wav_dir.exists():
            print(f"[WARN] Directory non trovata: {wav_dir}")
            continue
        plot_base = (plot_root / lbl) if (do_plot and plot_root) else None
        for wav in sorted(wav_dir.glob('*.wav')):
            spec = wav.with_suffix('.npy')
            plot = (plot_base / f"{wav.stem}.png") if plot_base else None
            load_and_preprocess(wav, spec, plot, do_plot)


# ───────────────────────── DATASET ────────────────────────

def label_from_parent(path: Path) -> int:
    return 0 if path.parent.name == 'HC' else 1


class MelSpecDataset(Dataset):
    def __init__(self, files: List[Path]):
        self.files  = files
        self.labels = [label_from_parent(f) for f in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = np.load(str(self.files[idx]))            # (3, N_MELS, MAX_FRAMES)
        x = torch.from_numpy(spec)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def gather_npy_files(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Raccoglie i .npy in root/{HC,PD}, ESCLUDENDO le maschere (*.mask.npy)
    che non sono spettrogrammi.
    """
    files: List[Path] = []
    for lbl in LABELS:
        lbl_dir = root / lbl
        if lbl_dir.exists():
            for p in sorted(lbl_dir.glob('*.npy')):
                if p.name.endswith('.mask.npy'):
                    continue
                files.append(p)
    labels = np.array([label_from_parent(f) for f in files])
    return np.array(files), labels


def stratified_subject_split(
    files: np.ndarray,
    labels: np.ndarray,
    val_size: float = VAL_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split stratificato a livello di SOGGETTO. Tutte le augmentation di uno
    stesso paziente finiscono nello stesso split.
    """
    subject_ids = np.array([extract_subject_id(Path(f)) for f in files])

    # Mappa soggetto → etichetta (verifica coerenza)
    subj_to_label: Dict[str, int] = {}
    for sid, lbl in zip(subject_ids, labels):
        if sid in subj_to_label and subj_to_label[sid] != int(lbl):
            raise ValueError(
                f"Il soggetto '{sid}' compare con etichette diverse "
                f"({subj_to_label[sid]} e {int(lbl)}). Controlla `extract_subject_id`."
            )
        subj_to_label[sid] = int(lbl)

    unique_subjects = np.array(sorted(subj_to_label.keys()))
    subject_labels  = np.array([subj_to_label[s] for s in unique_subjects])

    train_subj, val_subj = train_test_split(
        unique_subjects,
        test_size=val_size,
        stratify=subject_labels,
        random_state=seed,
    )

    train_set = set(train_subj)
    val_set   = set(val_subj)

    train_idx = np.array([i for i, sid in enumerate(subject_ids) if sid in train_set])
    val_idx   = np.array([i for i, sid in enumerate(subject_ids) if sid in val_set])

    return train_idx, val_idx


def check_leakage(
    train_files: np.ndarray, val_files: np.ndarray, test_files: np.ndarray,
) -> None:
    """Verifica che non ci siano soggetti in comune tra gli split."""
    tr = {extract_subject_id(Path(f)) for f in train_files}
    va = {extract_subject_id(Path(f)) for f in val_files}
    te = {extract_subject_id(Path(f)) for f in test_files}

    print("\nLeakage check (subject-level)")
    print("-" * 50)
    print(f"  Train soggetti  : {len(tr)}")
    print(f"  Val   soggetti  : {len(va)}")
    print(f"  Test  soggetti  : {len(te)}")
    print(f"  Train ∩ Val     : {len(tr & va)} soggetti in comune")
    print(f"  Train ∩ Test    : {len(tr & te)} soggetti in comune")
    print(f"  Val   ∩ Test    : {len(va & te)} soggetti in comune")
    if (tr & va) or (tr & te) or (va & te):
        print("  [WARN] LEAKAGE rilevato. Controlla `extract_subject_id` "
              "o la struttura dei file.")


# ───────────────────────── MODEL ──────────────────────────

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
            # Bidirezionale: cattura pattern sia forward che backward nel tempo.
            # Es. un tremore a fine vocalizzazione influenza l'interpretazione
            # anche dell'inizio.
        )
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                    # (B, C, F, T)
        x = x.permute(0, 3, 1, 2)          # (B, T, C, F)
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)
        out, _ = self.lstm(x)
        out_max, _ = torch.max(out, dim=1)
        out_avg    = torch.mean(out, dim=1)
        out = torch.cat([out_max, out_avg], dim=1)
        out = self.dropout(out)
        return self.classifier(out).squeeze(1)


# ─────────────────── PREDICTIONS / METRICS ────────────────

def get_predictions_in_order(
    model: nn.Module, loader: DataLoader, device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restituisce (y_true, y_prob) nello stesso ordine dei file del dataset.
    REQUISITO: il loader deve avere shuffle=False.
    """
    probs_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x))
            probs_list.append(prob.cpu().numpy())
            labels_list.append(y.numpy())
    return np.concatenate(labels_list), np.concatenate(probs_list)


def aggregate_per_subject(
    files: List[Path], y_true: np.ndarray, y_prob: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Aggrega le probabilità a livello di soggetto, mediando sulle 6 augmentation
    (1 originale + 5 augmentate). I file devono essere nello stesso ordine
    di y_true e y_prob.

    Returns:
        subjects:    lista di subject_id (ordinati)
        y_true_subj: una etichetta per soggetto
        y_prob_subj: probabilità media per soggetto
    """
    subject_probs: Dict[str, List[float]] = defaultdict(list)
    subject_labels: Dict[str, int]        = {}

    for f, yt, yp in zip(files, y_true, y_prob):
        sid = extract_subject_id(Path(f))
        subject_probs[sid].append(float(yp))
        subject_labels[sid] = int(yt)

    subjects   = sorted(subject_probs.keys())
    y_true_s   = np.array([subject_labels[s] for s in subjects])
    y_prob_s   = np.array([float(np.mean(subject_probs[s])) for s in subjects])
    return subjects, y_true_s, y_prob_s


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trova la soglia che massimizza F1 (uso esclusivo sul VAL set)."""
    prec_c, rec_c, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-8)
    if len(f1_scores) == 0 or np.all(np.isnan(f1_scores)):
        return 0.5
    return float(thresholds[np.argmax(f1_scores)])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob > threshold).astype(np.float32)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0,
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    return {
        'acc':         accuracy_score(y_true, y_pred),
        'precision':   prec,
        'recall':      rec,
        'specificity': specificity,
        'f1':          f1,
        'roc_auc':     auc,
        'cm':          cm,
        'y_true':      y_true,
        'probs':       y_prob,
        'threshold':   threshold,
    }


# ─────────────────────── BOOTSTRAP ────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap con resampling con rimpiazzo. Restituisce mean, std e CI 95%."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    keys = ('acc', 'precision', 'recall', 'specificity', 'f1', 'roc_auc')
    vals: Dict[str, List[float]] = {k: [] for k in keys}

    skipped = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]

        if len(np.unique(yt)) < 2:
            skipped += 1
            continue

        m = compute_metrics(yt, yp, threshold)

        for k in keys:
            v = m[k]
            if not np.isnan(v):
                vals[k].append(float(v))

    print(f"[DEBUG] Bootstrap valid samples: {len(vals['acc'])}/{n_bootstrap}")
    print(f"[DEBUG] Bootstrap skipped samples: {skipped}")

    summary: Dict[str, dict] = {}
    for k, arr_list in vals.items():
        arr = np.array(arr_list, dtype=float)

        if arr.size == 0:
            summary[k] = {
                'mean': float('nan'),
                'std': float('nan'),
                'ci_low': float('nan'),
                'ci_high': float('nan'),
            }
        else:
            summary[k] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'ci_low': float(np.percentile(arr, 2.5)),
                'ci_high': float(np.percentile(arr, 97.5)),
            }

    return summary


def save_bootstrap_summary(summary: dict, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        f.write("BOOTSTRAP TEST RESULTS — CRNN — SUBJECT LEVEL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<12}{'Mean':>10}{'Std':>10}{'CI lower':>12}{'CI upper':>12}\n")
        f.write("-" * 56 + "\n")
        for k, v in summary.items():
            f.write(
                f"{k:<12}{v['mean']:>10.4f}{v['std']:>10.4f}"
                f"{v['ci_low']:>12.4f}{v['ci_high']:>12.4f}\n"
            )


def save_test_predictions(
    subjects: List[str], y_true: np.ndarray, y_prob: np.ndarray,
    threshold: float, path: Path,
) -> None:
    with path.open('w', encoding='utf-8') as f:
        f.write("subject_id,y_true,y_prob_pd,y_pred\n")
        for sid, yt, yp in zip(subjects, y_true, y_prob):
            pred = int(yp > threshold)
            f.write(f"{sid},{int(yt)},{float(yp):.8f},{pred}\n")


# ─────────────────────── PLOTS ────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str = 'Confusion Matrix') -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()
    ticks = ['HC', 'PD']
    plt.xticks([0, 1], ticks)
    plt.yticks([0, 1], ticks)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(path); plt.close()


def plot_roc_curve(y_true: np.ndarray, probs: np.ndarray, path: Path,
                   title: str = 'ROC Curve') -> None:
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        return
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{title} (AUC={auc:.3f})')
    plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()


def plot_training_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss per Epoca')
    axes[0].set_xlabel('Epoca'); axes[0].set_ylabel('Loss'); axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   label='Val Acc (file-level)')
    axes[1].plot(epochs, history['val_auc'],   label='Val AUC (subject-level)',
                 linestyle='--')
    axes[1].set_title('Accuracy / AUC per Epoca')
    axes[1].set_xlabel('Epoca'); axes[1].set_ylabel('Score'); axes[1].legend()

    plt.tight_layout(); plt.savefig(path); plt.close()


# ──────────────────── TRAINING HELPER ─────────────────────

def step_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str = 'cpu',
    epsilon: float = 0.0,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad()
        y_smooth = y * (1 - epsilon) + 0.5 * epsilon
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss   = criterion(logits, y_smooth)
            if is_train:
                loss.backward()
                optimizer.step()
        preds       = (torch.sigmoid(logits) > 0.5).float()
        correct    += (preds == y).sum().item()
        samples    += y.size(0)
        total_loss += loss.item() * y.size(0)

    return total_loss / samples, correct / samples


# ───────────────────────── MAIN ───────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('--plot',      action='store_true', help='Salva spettrogrammi Mel')
    parser.add_argument('--eval-only', action='store_true',
                        help='Salta il training, carica best checkpoint e fa solo test+bootstrap')
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)
    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # ── 1) PREPROCESSING ──────────────────────────────────
    print("\n[1/4] Preprocessing Training_augmented ...")
    preprocess_split(TRAIN_DIR, PLOT_DIR / 'train' if args.plot else None, args.plot)
    print("[2/4] Preprocessing Test ...")
    preprocess_split(TEST_DIR,  PLOT_DIR / 'test'  if args.plot else None, args.plot)

    # ── 2) RACCOLTA FILE E SPLIT ──────────────────────────
    train_full_files, train_full_labels = gather_npy_files(TRAIN_DIR)
    test_files,       test_labels       = gather_npy_files(TEST_DIR)

    if len(train_full_files) == 0:
        raise RuntimeError(f"Nessun .npy trovato in {TRAIN_DIR}")
    if len(test_files) == 0:
        raise RuntimeError(f"Nessun .npy trovato in {TEST_DIR}")

    # Split train/val a livello di SOGGETTO (le 6 augmentation di un soggetto
    # finiscono tutte nello stesso split).
    train_idx, val_idx = stratified_subject_split(
        train_full_files, train_full_labels,
        val_size=VAL_SIZE, seed=RANDOM_SEED,
    )
    train_files,  train_labels  = train_full_files[train_idx], train_full_labels[train_idx]
    val_files,    val_labels    = train_full_files[val_idx],   train_full_labels[val_idx]

    n_train_subj = len({extract_subject_id(Path(f)) for f in train_files})
    n_val_subj   = len({extract_subject_id(Path(f)) for f in val_files})
    n_test_subj  = len({extract_subject_id(Path(f)) for f in test_files})

    print("\nSplit summary")
    print("-" * 60)
    print(f"  TRAIN : {len(train_files):4d} file   /  {n_train_subj:3d} soggetti  "
          f"(HC={int((train_labels==0).sum())}, PD={int((train_labels==1).sum())})")
    print(f"  VAL   : {len(val_files):4d} file   /  {n_val_subj:3d} soggetti  "
          f"(HC={int((val_labels==0).sum())}, PD={int((val_labels==1).sum())})")
    print(f"  TEST  : {len(test_files):4d} file   /  {n_test_subj:3d} soggetti  "
          f"(HC={int((test_labels==0).sum())}, PD={int((test_labels==1).sum())})")

    check_leakage(train_files, val_files, test_files)

    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\nDevice: {device}")

    train_dl = DataLoader(MelSpecDataset(train_files.tolist()),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_dl   = DataLoader(MelSpecDataset(val_files.tolist()),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl  = DataLoader(MelSpecDataset(test_files.tolist()),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = CRNNClassifier().to(device)

    # ── 3) TRAINING (selezione su VAL AUC subject-level) ─
    if not args.eval_only:
        print(f"\n[3/4] Training per {args.epochs} epoche "
              f"(selezione su VAL AUC subject-level) ...\n")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        history    = {'train_loss': [], 'train_acc': [],
                      'val_loss':   [], 'val_acc':   [], 'val_auc': []}
        best_auc   = -1.0
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device, EPSILON)
            vl_loss, vl_acc = step_epoch(model, val_dl,   criterion, None,      device, 0.0)
            scheduler.step(vl_loss)

            # AUC subject-level su val: medio le 6 probabilità per soggetto
            y_true_v, y_prob_v = get_predictions_in_order(model, val_dl, device)
            _, y_true_vs, y_prob_vs = aggregate_per_subject(
                val_files.tolist(), y_true_v, y_prob_v,
            )
            try:
                val_auc = roc_auc_score(y_true_vs, y_prob_vs)
            except ValueError:
                val_auc = float('nan')

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)
            history['val_auc'].append(val_auc if not np.isnan(val_auc) else 0.0)

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val_loss {vl_loss:.4f} acc {vl_acc:.3f} | "
                f"val_AUC_subj {val_auc:.3f}"
            )

            if not np.isnan(val_auc) and val_auc > best_auc:
                best_auc   = val_auc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     best_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_auc':         best_auc,
                }, BEST_CKPT_PATH)
                print(f"  [CHECKPOINT] Salvato: {BEST_CKPT_PATH.name} "
                      f"(val_AUC_subj={best_auc:.3f})")

        if not BEST_CKPT_PATH.exists():
            raise RuntimeError("Nessun checkpoint valido salvato.")

        plot_training_curves(history, STATS_DIR / 'training_curves-CRNN.png')
        print(f"\n[INFO] Migliore epoch: {best_epoch} | best val_AUC_subj: {best_auc:.3f}")

    # ── 4) VALUTAZIONE FINALE SUL TEST ────────────────────
    print("\n[4/4] Carico best checkpoint e valuto su TEST set ...")
    if not BEST_CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato: {BEST_CKPT_PATH}. "
            f"Esegui il training senza --eval-only."
        )
    ckpt = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Caricato: {BEST_CKPT_PATH.name} | "
          f"epoch={ckpt['epoch']} | val_AUC_subj={ckpt['best_val_auc']:.3f}")

    # 4a. Soglia ottima su VAL (subject-level)
    print("\n[INFO] Calcolo soglia ottima F1 su VAL (subject-level) ...")
    y_true_v, y_prob_v = get_predictions_in_order(model, val_dl, device)
    _, y_true_vs, y_prob_vs = aggregate_per_subject(
        val_files.tolist(), y_true_v, y_prob_v,
    )
    best_threshold = find_optimal_threshold(y_true_vs, y_prob_vs)
    print(f"  Soglia ottima (da VAL): {best_threshold:.4f}")

    # 4b. Predizioni su TEST (subject-level: nel test c'è già 1 file/soggetto,
    #     l'aggregazione è un no-op ma la facciamo per coerenza di pipeline)
    print("\n[INFO] Predizioni su TEST con soglia fissa ...")
    y_true_t, y_prob_t = get_predictions_in_order(model, test_dl, device)
    test_subjects, y_true_ts, y_prob_ts = aggregate_per_subject(
        test_files.tolist(), y_true_t, y_prob_t,
    )

    test_metrics = compute_metrics(y_true_ts, y_prob_ts, best_threshold)
    tn, fp, fn, tp = test_metrics['cm'].ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS — CRNN — SUBJECT LEVEL")
    print("=" * 60)
    print(f"  Subjects     : {len(y_true_ts)}")
    print(f"  Threshold    : {best_threshold:.4f}  (selezionata su VAL)")
    print(f"  Accuracy     : {test_metrics['acc']:.4f}")
    print(f"  Precision    : {test_metrics['precision']:.4f}")
    print(f"  Recall/Sens. : {test_metrics['recall']:.4f}")
    print(f"  Specificity  : {specificity:.4f}")
    print(f"  F1           : {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC      : {test_metrics['roc_auc']:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 60)

    plot_confusion_matrix(test_metrics['cm'],
                          STATS_DIR / 'test_confusion_matrix-CRNN.png',
                          'Test Set - Confusion Matrix')
    plot_roc_curve(y_true_ts, y_prob_ts,
                   STATS_DIR / 'test_roc_curve-CRNN.png',
                   'Test Set - ROC Curve')

    save_test_predictions(test_subjects, y_true_ts, y_prob_ts,
                          best_threshold, TEST_PREDICTIONS_PATH)
    print(f"\n[INFO] Predizioni TEST salvate in: {TEST_PREDICTIONS_PATH}")

    # 4c. Bootstrap (1000 iter., CI 95%)
    print(f"\n[INFO] Bootstrap su TEST ({BOOTSTRAP_N} iter.) ...")
    bootstrap_summary = bootstrap_ci(
        y_true_ts, y_prob_ts,
        threshold=best_threshold,
        n_bootstrap=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
    )
    save_bootstrap_summary(bootstrap_summary, BOOTSTRAP_RESULTS_PATH)

    print("\n" + "=" * 60)
    print(f"BOOTSTRAP ({BOOTSTRAP_N} iterazioni, CI 95%)")
    print("=" * 60)
    print(f"{'Metric':<12}{'Mean':>10}{'Std':>10}{'CI lower':>12}{'CI upper':>12}")
    print("-" * 56)
    for k, v in bootstrap_summary.items():
        print(f"{k:<12}{v['mean']:>10.4f}{v['std']:>10.4f}"
              f"{v['ci_low']:>12.4f}{v['ci_high']:>12.4f}")
    print("=" * 60)
    print(f"[INFO] Bootstrap salvato in: {BOOTSTRAP_RESULTS_PATH}")

    # Checkpoint finale con threshold incluso (utile per l'agente)
    torch.save({
        'model_state_dict': ckpt['model_state_dict'],
        'best_val_auc':     ckpt['best_val_auc'],
        'threshold':        best_threshold,
        'test_metrics':     {k: float(test_metrics[k])
                     for k in ('acc', 'precision', 'recall', 'specificity', 'f1', 'roc_auc')},
        'bootstrap':        bootstrap_summary,
    }, CHECKPOINT_DIR / 'best_model_final.pt')

    print(f"\nDone. Output in milestone2/")
    print(f"  → {STATS_DIR / 'training_curves-CRNN.png'}")
    print(f"  → {STATS_DIR / 'test_confusion_matrix-CRNN.png'}")
    print(f"  → {STATS_DIR / 'test_roc_curve-CRNN.png'}")
    print(f"  → {BEST_CKPT_PATH}")
    print(f"  → {CHECKPOINT_DIR / 'best_model_final.pt'}")
    print(f"  → {TEST_PREDICTIONS_PATH}")
    print(f"  → {BOOTSTRAP_RESULTS_PATH}")


if __name__ == '__main__':
    main()