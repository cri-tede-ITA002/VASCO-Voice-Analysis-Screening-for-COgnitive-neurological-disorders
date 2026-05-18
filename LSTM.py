"""
Early Parkinson's Detection Using Speech Analysis
**Windowed data pipeline + LSTM baseline**
--------------------------------------------------------------
Valutazione a livello di SOGGETTO (speaker-level) per evitare
data leakage e ottenere metriche clinicamente significative.

Usage examples
```
python milestone1_windowed.py            # quick 10-epoch smoke test
python milestone1_windowed.py -e 50 --plot   # longer run + plots
```
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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

SAMPLE_RATE = 16_000  # 16 kHz
N_MELS      = 64
HOP_LENGTH  = 160      # 10 ms → 100 frames/s
WIN_LENGTH  = 400      # 25 ms
FMIN, FMAX  = 50, 4_000

# ---- Window params ----
WINDOW_SEC   = 2.0
FRAMES_PER_S = SAMPLE_RATE // HOP_LENGTH  # 100
WIN_FRAMES   = int(WINDOW_SEC * FRAMES_PER_S)  # 200 frames / window
HOP_FRAMES   = WIN_FRAMES // 2  # 50 % overlap (1 s)
PAD_VALUE_DB = -80.0
# -------------------------------------------------

RANDOM_SEED = 42
BATCH_SIZE  = 8
NUM_WORKERS = os.cpu_count() or 2

# ---------- Utility helpers ----------

def list_wav_files(data_root: Path) -> List[Tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for label_name, label in [("HC", 0), ("PD", 1)]:
        for wav in (data_root / label_name).glob("*.wav"):
            out.append((wav, label))
    return out


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
                    constant_values=PAD_VALUE_DB
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


# ---------- Window-level evaluate (usato internamente durante il training) ----------
def evaluate_window_level(model, loader: DataLoader, device="cpu"):
    """
    Valutazione a livello di finestra — usata solo per monitorare il training.
    NON è la metrica finale.
    """
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
    preds = (probs > 0.5).astype(np.float32)

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
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
        "probs": probs,
        "y_true": y_true,
    }


# ---------- Speaker-level evaluate (metrica FINALE corretta) ----------
def evaluate_speaker_level(model, files: List[Path], device="cpu"):
    """
    Valutazione a livello di SOGGETTO.

    Per ogni soggetto (identificato dallo stem senza _winXXX):
      - raccoglie le probabilità di tutte le sue finestre
      - fa la media → probabilità del soggetto
      - soglia 0.5 → predizione finale

    Questo è l'unico modo corretto per riportare metriche cliniche:
    la confusion matrix avrà tanti sample quanti sono i soggetti,
    non le finestre.
    """
    model.eval()

    # stem → lista di probs scalari
    speaker_probs: dict[str, list[float]] = defaultdict(list)
    speaker_labels: dict[str, int] = {}

    with torch.no_grad():
        for f in files:
            stem = f.stem.rsplit("_win", 1)[0]          # es. "PD_subject01"
            label = 0 if stem.startswith("HC_") else 1
            spec = torch.from_numpy(np.load(f)).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(spec)).item()
            speaker_probs[stem].append(prob)
            speaker_labels[stem] = label

    stems_sorted = sorted(speaker_probs.keys())
    y_true = np.array([speaker_labels[s] for s in stems_sorted])
    avg_probs = np.array([np.mean(speaker_probs[s]) for s in stems_sorted])
    preds = (avg_probs > 0.5).astype(int)

    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, avg_probs)
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, preds)

    # Stampa debug: quante finestre per soggetto
    n_windows = [len(speaker_probs[s]) for s in stems_sorted]
    print(f"  [speaker-level] Soggetti: {len(stems_sorted)} | "
          f"Finestre/sogg: min={min(n_windows)} max={max(n_windows)} "
          f"mean={np.mean(n_windows):.1f}")

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
        "probs": avg_probs,
        "y_true": y_true,
        "n_subjects": len(stems_sorted),
    }


# ---------- Plots ----------
def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str = "Confusion Matrix - LSTM"):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["HC", "PD"])
    plt.yticks(tick_marks, ["HC", "PD"])
    thresh = cm.max() / 2.0
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


def plot_roc_curve(y_true: np.ndarray, probs: np.ndarray, path: Path, title: str = "ROC curve"):
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


# ---------- Helper: speaker-level split (train/val senza leakage) ----------
def make_splits(cache_dir: Path):
    """
    Split a livello di SOGGETTO: tutte le finestre dello stesso soggetto
    vanno nello stesso fold. Evita data leakage train→val.
    """
    all_npys = sorted(cache_dir.glob("*.npy"))
    if not all_npys:
        raise RuntimeError(f"No .npy files found in {cache_dir}")

    stems = sorted({p.stem.rsplit("_win", 1)[0] for p in all_npys})
    labels_per_stem = [0 if s.startswith("HC_") else 1 for s in stems]

    train_stems, val_stems = train_test_split(
        stems,
        test_size=0.3,
        stratify=labels_per_stem,
        random_state=RANDOM_SEED
    )

    def stems_to_windows(stem_set):
        stem_set = set(stem_set)
        return [p for p in all_npys if p.stem.rsplit("_win", 1)[0] in stem_set]

    return stems_to_windows(train_stems), stems_to_windows(val_stems), list(val_stems)


def load_test_files(cache_dir: Path):
    test_files = sorted(cache_dir.glob("*.npy"))
    if not test_files:
        raise RuntimeError(f"No test .npy files found in {cache_dir}")
    return test_files


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
    argp.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs")
    argp.add_argument("--plot", action="store_true", help="Save one spectrogram per speaker")
    argp.add_argument("--recache", action="store_true", help="Force re-compute windows")
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

    # Split a livello di soggetto (no leakage)
    train_files, val_files, val_stems = make_splits(TRAIN_CACHE_DIR)
    test_files = load_test_files(TEST_CACHE_DIR)

    # Conta soggetti unici per split
    train_stems_set = {f.stem.rsplit("_win", 1)[0] for f in train_files}
    test_stems_set  = {f.stem.rsplit("_win", 1)[0] for f in test_files}

    print(f"\nSplit summary (soggetti):")
    print(f"  Train soggetti : {len(train_stems_set)}")
    print(f"  Val   soggetti : {len(val_stems)}")
    print(f"  Test  soggetti : {len(test_stems_set)}")
    print(f"\nSplit summary (finestre):")
    print(f"  Train finestre : {len(train_files)}")
    print(f"  Val   finestre : {len(val_files)}")
    print(f"  Test  finestre : {len(test_files)}")
    print()

    if len(train_files) == 0:
        raise RuntimeError("Zero train examples - check TRAIN cache content and file naming!")

    train_dl = DataLoader(
        ParkinsonDataset(train_files),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_dl = DataLoader(
        ParkinsonDataset(val_files),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")
    model = LSTMAudioClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    best_roc = -1.0
    best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"

    print("=" * 55)
    print("  TRAINING (window-level loss | speaker-level val ROC)")
    print("=" * 55)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = step_epoch(model, val_dl, criterion, None, device)

        # Valutazione speaker-level sulla validation (per checkpoint)
        val_sp = evaluate_speaker_level(model, val_files, device)

        print(
            f"Epoch {epoch:02d} | "
            f"tr_loss {tr_loss:.3f} tr_acc {tr_acc:.3f} | "
            f"val_loss {val_loss:.3f} val_acc(win) {val_acc:.3f} | "
            f"val_ROC(spk) {val_sp['roc_auc']:.3f}"
        )

        if not np.isnan(val_sp["roc_auc"]) and val_sp["roc_auc"] > best_roc:
            best_roc = val_sp["roc_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics_speaker": val_sp,
                },
                best_ckpt_path
            )
            print(f"  → [CHECKPOINT] Nuovo best model (val speaker ROC={best_roc:.3f})")

    # ---- Valutazione finale sul TEST (speaker-level) ----
    print("\n[INFO] Loading best checkpoint for final test...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Checkpoint dall'epoca {ckpt['epoch']} (val ROC={best_roc:.3f})")

    print("\n[INFO] Valutazione TEST a livello di SOGGETTO...")
    test_sp = evaluate_speaker_level(model, test_files, device)
    cm = test_sp["cm"]
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 55)
    print("   FINAL TEST METRICS  — livello SOGGETTO")
    print(f"   (N soggetti = {test_sp['n_subjects']})")
    print("=" * 55)
    print(f"  Accuracy          : {test_sp['acc']:.4f}")
    print(f"  Precision         : {test_sp['precision']:.4f}")
    print(f"  Recall (Sensi.)   : {test_sp['recall']:.4f}")
    print(f"  F1 Score          : {test_sp['f1']:.4f}")
    print(f"  ROC-AUC           : {test_sp['roc_auc']:.4f}")
    print(f"  Specificity       : {tn / (tn + fp) if (tn + fp) > 0 else float('nan'):.4f}")
    print("-" * 55)
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 55)
    print(f"\nConfusion Matrix (soggetti):")
    print(f"              Pred HC   Pred PD")
    print(f"  True HC  :    {tn:4d}      {fp:4d}")
    print(f"  True PD  :    {fn:4d}      {tp:4d}")

    # Salva plots speaker-level
    plot_confusion_matrix(
        cm,
        PLOT_DIR / "test_confusion_matrix_speaker-LSTM.png",
        title="Confusion Matrix (Speaker-level) - LSTM"
    )
    plot_roc_curve(
        test_sp["y_true"],
        test_sp["probs"],
        PLOT_DIR / "test_roc_curve_speaker-LSTM.png",
        title="ROC curve (Speaker-level) - LSTM"
    )

    # Salva anche window-level per confronto/debug
    print("\n[INFO] Salvataggio window-level metrics per confronto...")
    test_dl = DataLoader(
        ParkinsonDataset(test_files),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_win = evaluate_window_level(model, test_dl, device)
    plot_confusion_matrix(
        test_win["cm"],
        PLOT_DIR / "test_confusion_matrix_window-LSTM.png",
        title=f"Confusion Matrix (Window-level, N={len(test_files)})"
    )
    plot_roc_curve(
        test_win["y_true"],
        test_win["probs"],
        PLOT_DIR / "test_roc_curve_window-LSTM.png",
        title="ROC curve (Window-level) - LSTM"
    )
    print(f"  Window-level acc={test_win['acc']:.4f} ROC={test_win['roc_auc']:.4f} "
          f"(su {len(test_files)} finestre — non usare come metrica finale!)")

    print("\nPlots salvati in artifacts/plots/")
    print("[INFO] Finished\n")


if __name__ == "__main__":
    main()