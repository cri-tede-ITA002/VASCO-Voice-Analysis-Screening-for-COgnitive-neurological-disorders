"""
LSTM.py — Early Parkinson's Detection — LSTM baseline (versione corretta)
=========================================================================

  1. `extract_subject_id` rimuove sia `_winXXX` SIA i suffissi di augmentation
     (_noise, _pitch_up, _pitch_down, _speed_up, _speed_down): le 6 versioni
     augmentate di uno stesso paziente collassano sullo stesso subject_id.
  2. `check_leakage` esplicito stampato a inizio run.
  3. Soglia ottima F1 cercata sul VAL (subject-level), applicata fissa sul
     TEST anziché 0.5 hardcoded.
  4. Bootstrap (1000 iter., CI 95%) sul TEST, stesso formato di CRNN/AST.
  5. Argparse `--eval-only`.

"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# -------------------- CONFIG --------------------
TRAIN_ROOT       = Path("data/Training_augmented")
TEST_ROOT        = Path("data/Test")

TRAIN_CACHE_DIR  = Path("artifacts/mel_specs_train")
TEST_CACHE_DIR   = Path("artifacts/mel_specs_test")
PLOT_DIR         = Path("artifacts/plots")
CHECKPOINT_DIR   = Path("artifacts/checkpoints")
STATS_DIR        = Path("artifacts/stats")

SAMPLE_RATE = 16_000
N_MELS      = 64
HOP_LENGTH  = 160      # 10 ms → 100 frames/s
WIN_LENGTH  = 400      # 25 ms
FMIN, FMAX  = 50, 4_000

# ---- Window params ----
WINDOW_SEC   = 2.0
FRAMES_PER_S = SAMPLE_RATE // HOP_LENGTH       # 100
WIN_FRAMES   = int(WINDOW_SEC * FRAMES_PER_S)  # 200 frames / window
HOP_FRAMES   = WIN_FRAMES // 2                  # 50 % overlap (1 s)
PAD_VALUE_DB = -80.0
# -------------------------------------------------

RANDOM_SEED = 42
BATCH_SIZE  = 8
NUM_WORKERS = os.cpu_count() or 2
VAL_SIZE    = 0.20

# Bootstrap (allineato con CRNN.py / AST.py)
BOOTSTRAP_N    = 1000
BOOTSTRAP_SEED = 42

# Suffissi di augmentation (stessi di CRNN.py e AST.py)
AUG_SUFFIXES = ("_noise", "_pitch_up", "_pitch_down", "_speed_up", "_speed_down")

TEST_PREDICTIONS_PATH  = CHECKPOINT_DIR / "test_predictions_lstm.csv"
BOOTSTRAP_RESULTS_PATH = CHECKPOINT_DIR / "bootstrap_test_lstm.txt"


# ---------- Utility helpers ----------

def list_wav_files(data_root: Path) -> List[Tuple[Path, int]]:
    out: List[Tuple[Path, int]] = []
    for label_name, label in [("HC", 0), ("PD", 1)]:
        for wav in (data_root / label_name).glob("*.wav"):
            out.append((wav, label))
    return out


def extract_subject_id(cache_file: Path) -> str:
    """
    Estrae l'ID del soggetto dal nome file di cache, rimuovendo:
      1) il suffisso `_winXXX` (numero finestra)
      2) il suffisso di augmentation (_noise, _pitch_up, ecc.)

    Esempi:
      HC_AH_064F_..._C5_win000.npy             →  HC_AH_064F_..._C5
      HC_AH_064F_..._C5_noise_win003.npy       →  HC_AH_064F_..._C5
      HC_AH_064F_..._C5_pitch_down_win005.npy  →  HC_AH_064F_..._C5
    """
    stem = cache_file.stem.rsplit("_win", 1)[0]
    for suffix in AUG_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def plot_spectrogram(spec: np.ndarray, wav: Path, label_prefix: str):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.T, aspect="auto", origin="lower")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-mel spectrogram - {wav.name} ({label_prefix})")
    plt.xlabel("Time (frames)")
    plt.ylabel("Mel bands")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{label_prefix}_{wav.stem}.png")
    plt.close()


def cache_all_windows(data_root: Path, cache_dir: Path, plot: bool = False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for wav, _ in list_wav_files(data_root):
        y, _ = librosa.load(wav, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y, top_db=35)

        melspec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            fmin=FMIN, fmax=FMAX, power=2.0,
        )
        full_db = librosa.power_to_db(melspec, ref=np.max).T.astype(np.float32)
        total_frames = full_db.shape[0]

        for start in range(0, total_frames, HOP_FRAMES):
            end = start + WIN_FRAMES
            window = full_db[start:end]

            if window.shape[0] < WIN_FRAMES:
                pad = WIN_FRAMES - window.shape[0]
                window = np.pad(
                    window,
                    ((0, pad), (0, 0)),
                    mode="constant",
                    constant_values=PAD_VALUE_DB,
                )

            win_idx = start // HOP_FRAMES
            label_pf = wav.parent.name.split("_", 1)[0]
            out_name = f"{label_pf}_{wav.stem}_win{win_idx:03d}.npy"
            np.save(cache_dir / out_name, window)

            if plot and win_idx == 0:
                plot_spectrogram(window, wav, label_pf)

    print(f"[cache_all_windows] DONE - cached windows under {cache_dir}")


# ---------- Dataset ----------
class ParkinsonDataset(Dataset):
    """Window-level dataset. Each item → (T=200, M=64)."""
    def __init__(self, files: List[Path]):
        self.files  = files
        self.labels = [0 if f.name.startswith("HC_") else 1 for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])  # (T, M)
        return torch.from_numpy(spec), torch.tensor(self.labels[idx], dtype=torch.float32)


# ---------- Model ----------
class LSTMAudioClassifier(nn.Module):
    def __init__(self, n_mels=N_MELS, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_mels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out  = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x):  # x: (B, T, M)
        out, _ = self.lstm(x)
        return self.out(out[:, -1, :]).squeeze(1)


# ---------- Window-level evaluate (debug / monitoring) ----------
def evaluate_window_level(model, loader: DataLoader, device="cpu", threshold: float = 0.5):
    """Valutazione a livello di finestra — solo per debug, NON metrica finale."""
    model.eval()
    logits_lst, labels_lst = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits_lst.append(logits.cpu())
            labels_lst.append(y.cpu())
    logits = torch.cat(logits_lst).numpy()
    y_true = torch.cat(labels_lst).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > threshold).astype(np.float32)

    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, preds)
    return {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc_auc, "cm": cm, "probs": probs, "y_true": y_true,
    }


# ---------- Speaker-level evaluate (metrica FINALE) ----------
def evaluate_speaker_level(
    model, files: List[Path], device="cpu", threshold: float = 0.5,
):
    """
    Per ogni soggetto (estratto da `extract_subject_id`):
      - raccoglie le probabilità di TUTTE le finestre di TUTTE le sue
        versioni (originale + 5 augmentate) durante train/val,
      - oppure solo le finestre del singolo file originale durante test,
      - media → probabilità del soggetto,
      - applica `threshold` → predizione finale.
    """
    model.eval()

    speaker_probs:  Dict[str, List[float]] = defaultdict(list)
    speaker_labels: Dict[str, int]         = {}

    with torch.no_grad():
        for f in files:
            sid = extract_subject_id(f)
            label = 0 if sid.startswith("HC_") else 1
            spec = torch.from_numpy(np.load(f)).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(spec)).item()
            speaker_probs[sid].append(prob)
            speaker_labels[sid] = label

    subjects = sorted(speaker_probs.keys())
    y_true = np.array([speaker_labels[s] for s in subjects])
    avg_probs = np.array([np.mean(speaker_probs[s]) for s in subjects])
    preds = (avg_probs > threshold).astype(int)

    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0,
    )
    try:
        roc_auc = roc_auc_score(y_true, avg_probs)
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    n_windows = [len(speaker_probs[s]) for s in subjects]
    print(f"  [speaker-level] Soggetti: {len(subjects)} | "
          f"Finestre/sogg: min={min(n_windows)} max={max(n_windows)} "
          f"mean={np.mean(n_windows):.1f}")

    return {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": roc_auc, "cm": cm,
        "probs": avg_probs, "y_true": y_true,
        "subjects": subjects, "n_subjects": len(subjects),
        "threshold": threshold,
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trova la soglia che massimizza F1 (uso esclusivo sul VAL set)."""
    prec_c, rec_c, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-8)
    if len(f1_scores) == 0 or np.all(np.isnan(f1_scores)):
        return 0.5
    return float(thresholds[np.argmax(f1_scores)])


# ---------- Bootstrap ----------
def compute_metrics_from_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "f1":          f1,
        "roc_auc":     auc,
    }


def bootstrap_subject_level_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")
    values: Dict[str, List[float]] = {k: [] for k in keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics_from_predictions(yt, yp, threshold)
        for k in keys:
            v = m[k]
            if not (isinstance(v, float) and np.isnan(v)):
                values[k].append(v)

    summary: Dict[str, dict] = {}
    for k, arr_list in values.items():
        arr = np.array(arr_list, dtype=float)
        if arr.size == 0:
            summary[k] = {"mean": float("nan"), "std": float("nan"),
                          "ci_low": float("nan"), "ci_high": float("nan")}
        else:
            summary[k] = {
                "mean":    float(np.mean(arr)),
                "std":     float(np.std(arr)),
                "ci_low":  float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }
    return summary


def save_bootstrap_summary(summary: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("BOOTSTRAP TEST RESULTS — LSTM — SUBJECT LEVEL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<12}{'Mean':>10}{'Std':>10}{'CI lower':>12}{'CI upper':>12}\n")
        f.write("-" * 56 + "\n")
        for k, v in summary.items():
            f.write(
                f"{k:<12}{v['mean']:>10.4f}{v['std']:>10.4f}"
                f"{v['ci_low']:>12.4f}{v['ci_high']:>12.4f}\n"
            )


def save_subject_predictions(
    subjects: List[str], y_true: np.ndarray, y_prob: np.ndarray,
    threshold: float, output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("subject_id,y_true,y_prob_pd,y_pred\n")
        for sid, yt, yp in zip(subjects, y_true, y_prob):
            pred = int(yp > threshold)
            f.write(f"{sid},{int(yt)},{float(yp):.8f},{pred}\n")


# ---------- Plots ----------
def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str = "Confusion Matrix - LSTM"):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["HC", "PD"])
    plt.yticks(tick_marks, ["HC", "PD"])
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, probs: np.ndarray, path: Path,
                   title: str = "ROC curve"):
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        return
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ---------- Speaker-level split ----------
def make_splits(cache_dir: Path) -> Tuple[List[Path], List[Path], List[str]]:
    """
    Split a livello di SOGGETTO. Le 6 augmentation di uno stesso paziente
    finiscono nello stesso fold grazie a `extract_subject_id`.
    """
    all_npys = sorted(cache_dir.glob("*.npy"))
    if not all_npys:
        raise RuntimeError(f"No .npy files found in {cache_dir}")

    subjects = sorted({extract_subject_id(p) for p in all_npys})
    labels_per_subj = [0 if s.startswith("HC_") else 1 for s in subjects]

    train_subj, val_subj = train_test_split(
        subjects,
        test_size=VAL_SIZE,
        stratify=labels_per_subj,
        random_state=RANDOM_SEED,
    )

    train_set = set(train_subj)
    val_set   = set(val_subj)

    train_files = [p for p in all_npys if extract_subject_id(p) in train_set]
    val_files   = [p for p in all_npys if extract_subject_id(p) in val_set]

    return train_files, val_files, sorted(val_set)


def load_test_files(cache_dir: Path):
    test_files = sorted(cache_dir.glob("*.npy"))
    if not test_files:
        raise RuntimeError(f"No test .npy files found in {cache_dir}")
    return test_files


def check_leakage(train_files, val_files, test_files) -> None:
    tr = {extract_subject_id(p) for p in train_files}
    va = {extract_subject_id(p) for p in val_files}
    te = {extract_subject_id(p) for p in test_files}

    print("\nLeakage check (subject-level)")
    print("-" * 50)
    print(f"  Train soggetti  : {len(tr)}")
    print(f"  Val   soggetti  : {len(va)}")
    print(f"  Test  soggetti  : {len(te)}")
    print(f"  Train ∩ Val     : {len(tr & va)} soggetti in comune")
    print(f"  Train ∩ Test    : {len(tr & te)} soggetti in comune")
    print(f"  Val   ∩ Test    : {len(va & te)} soggetti in comune")
    if (tr & va) or (tr & te) or (va & te):
        print("  [WARN] LEAKAGE rilevato. Controlla `AUG_SUFFIXES`.")


def step_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


# ---------- Main ----------
def main():
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument("-e", "--epochs", type=int, default=15, help="Number of epochs")
    argp.add_argument("--plot", action="store_true", help="Save one spectrogram per speaker")
    argp.add_argument("--recache", action="store_true", help="Force re-compute windows")
    argp.add_argument("--eval-only", action="store_true",
                      help="Salta training, carica best checkpoint e fa solo test+bootstrap")
    args = argp.parse_args()

    print("EXECUTION TIME:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if args.recache or not any(TRAIN_CACHE_DIR.glob("*.npy")):
        cache_all_windows(TRAIN_ROOT, TRAIN_CACHE_DIR, args.plot)
    else:
        print("[info] Found cached TRAIN windows - skipping preprocessing.")

    if args.recache or not any(TEST_CACHE_DIR.glob("*.npy")):
        cache_all_windows(TEST_ROOT, TEST_CACHE_DIR, args.plot)
    else:
        print("[info] Found cached TEST windows - skipping preprocessing.")

    # Split a livello di soggetto (no leakage anche tra augmentation)
    train_files, val_files, val_subjects = make_splits(TRAIN_CACHE_DIR)
    test_files = load_test_files(TEST_CACHE_DIR)

    train_subjects = sorted({extract_subject_id(f) for f in train_files})
    test_subjects  = sorted({extract_subject_id(f) for f in test_files})

    print(f"\nSplit summary (soggetti):")
    print(f"  Train soggetti : {len(train_subjects)}")
    print(f"  Val   soggetti : {len(val_subjects)}")
    print(f"  Test  soggetti : {len(test_subjects)}")
    print(f"\nSplit summary (finestre):")
    print(f"  Train finestre : {len(train_files)}")
    print(f"  Val   finestre : {len(val_files)}")
    print(f"  Test  finestre : {len(test_files)}")

    check_leakage(train_files, val_files, test_files)
    print()

    if len(train_files) == 0:
        raise RuntimeError("Zero train examples - check TRAIN cache content and file naming!")

    train_dl = DataLoader(
        ParkinsonDataset(train_files),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    val_dl = DataLoader(
        ParkinsonDataset(val_files),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")
    model = LSTMAudioClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"

    # ---- Training oppure caricamento checkpoint ----
    if args.eval_only:
        print("\n[INFO] --eval-only: salto il training e carico il checkpoint salvato...")
        if not best_ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Caricato checkpoint dall'epoca {ckpt['epoch']}")
    else:
        best_roc = -1.0
        best_epoch = -1

        print("=" * 60)
        print("  TRAINING (window-level loss | speaker-level val ROC)")
        print("=" * 60)

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device)
            val_loss, val_acc = step_epoch(model, val_dl, criterion, None, device)

            # Valutazione speaker-level sulla validation (per checkpoint)
            val_sp = evaluate_speaker_level(model, val_files, device, threshold=0.5)

            print(
                f"Epoch {epoch:02d} | "
                f"tr_loss {tr_loss:.3f} tr_acc {tr_acc:.3f} | "
                f"val_loss {val_loss:.3f} val_acc(win) {val_acc:.3f} | "
                f"val_ROC(spk) {val_sp['roc_auc']:.3f}"
            )

            if not np.isnan(val_sp["roc_auc"]) and val_sp["roc_auc"] > best_roc:
                best_roc = val_sp["roc_auc"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_roc,
                    "val_metrics_speaker": {
                        k: v for k, v in val_sp.items() if k != "cm"
                    },
                }, best_ckpt_path)
                print(f"  → [CHECKPOINT] Nuovo best model (val speaker ROC={best_roc:.3f})")

        print(f"\n[INFO] Migliore epoch: {best_epoch} | best val ROC: {best_roc:.3f}")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # ---- Soglia ottima F1 calcolata sul VAL (subject-level) ----
    print("\n[INFO] Calcolo soglia ottima F1 su VAL (subject-level)...")
    val_sp = evaluate_speaker_level(model, val_files, device, threshold=0.5)
    best_threshold = find_optimal_threshold(val_sp["y_true"], val_sp["probs"])
    print(f"  Soglia ottima (da VAL): {best_threshold:.4f}")

    # ---- Valutazione finale TEST con soglia FISSA da val ----
    print("\n[INFO] Valutazione TEST a livello di SOGGETTO (soglia da VAL)...")
    test_sp = evaluate_speaker_level(model, test_files, device, threshold=best_threshold)
    cm = test_sp["cm"]
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print("\n" + "=" * 60)
    print(f"   FINAL TEST METRICS — LSTM — SUBJECT LEVEL")
    print(f"   (N soggetti = {test_sp['n_subjects']})")
    print("=" * 60)
    print(f"  Threshold        : {best_threshold:.4f}  (selezionata su VAL)")
    print(f"  Accuracy         : {test_sp['acc']:.4f}")
    print(f"  Precision        : {test_sp['precision']:.4f}")
    print(f"  Recall (Sens.)   : {test_sp['recall']:.4f}")
    print(f"  Specificity      : {specificity:.4f}")
    print(f"  F1 Score         : {test_sp['f1']:.4f}")
    print(f"  ROC-AUC          : {test_sp['roc_auc']:.4f}")
    print("-" * 60)
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 60)
    print(f"\nConfusion Matrix (soggetti):")
    print(f"              Pred HC   Pred PD")
    print(f"  True HC  :    {tn:4d}      {fp:4d}")
    print(f"  True PD  :    {fn:4d}      {tp:4d}")

    # Plots speaker-level
    plot_confusion_matrix(
        cm,
        PLOT_DIR / "test_confusion_matrix_speaker-LSTM.png",
        title="Confusion Matrix (Speaker-level) - LSTM",
    )
    plot_roc_curve(
        test_sp["y_true"], test_sp["probs"],
        PLOT_DIR / "test_roc_curve_speaker-LSTM.png",
        title="ROC curve (Speaker-level) - LSTM",
    )

    # Salva predizioni per soggetto
    save_subject_predictions(
        test_sp["subjects"], test_sp["y_true"], test_sp["probs"],
        best_threshold, TEST_PREDICTIONS_PATH,
    )
    print(f"\n[INFO] Predizioni TEST salvate in: {TEST_PREDICTIONS_PATH}")

    # ---- Bootstrap (1000 iter., CI 95%) sul TEST ----
    print(f"\n[INFO] Bootstrap su TEST ({BOOTSTRAP_N} iter.)...")
    bootstrap_summary = bootstrap_subject_level_ci(
        y_true=test_sp["y_true"],
        y_prob=test_sp["probs"],
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

    # ---- Window-level metrics solo per confronto/debug ----
    print("\n[INFO] Window-level metrics (debug, NON metrica finale)...")
    test_dl = DataLoader(
        ParkinsonDataset(test_files),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    test_win = evaluate_window_level(model, test_dl, device, threshold=best_threshold)
    plot_confusion_matrix(
        test_win["cm"],
        PLOT_DIR / "test_confusion_matrix_window-LSTM.png",
        title=f"Confusion Matrix (Window-level, N={len(test_files)})",
    )
    plot_roc_curve(
        test_win["y_true"], test_win["probs"],
        PLOT_DIR / "test_roc_curve_window-LSTM.png",
        title="ROC curve (Window-level) - LSTM",
    )
    print(f"  Window-level acc={test_win['acc']:.4f} ROC={test_win['roc_auc']:.4f} "
          f"(su {len(test_files)} finestre — NON metrica finale)")

    print("\nPlots salvati in artifacts/plots/")
    print("[INFO] Finished\n")


if __name__ == "__main__":
    main()