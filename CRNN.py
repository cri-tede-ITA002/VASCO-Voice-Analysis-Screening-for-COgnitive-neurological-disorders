#!/usr/bin/env python3
"""
Early Parkinson's Detection Using Speech Analysis
Training: Training_augmented/{HC, PD}  — 20 epoche
Test:     Test/{HC, PD}
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# -------------------- CONFIG --------------------

TRAIN_DIR      = Path("data/Training_augmented")   
TEST_DIR       = Path("data/Test")                 

PLOT_DIR       = Path("milestone2/plots")
CHECKPOINT_DIR = Path("checkpoints/checkpoints_crnn")
STATS_DIR      = Path("milestone2/stats")
for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE    = 16_000
N_MELS         = 64
HOP_LENGTH     = 160
WIN_LENGTH     = 400
FMIN           = 50
FMAX           = 4_000
MAX_FRAMES     = 1_024
SPEC_PAD_VALUE = -80.0

RANDOM_SEED = 42
EPOCHS      = 20
BATCH_SIZE  = 8
NUM_WORKERS = os.cpu_count() or 2
EPSILON     = 0.1

LABELS = ('HC', 'PD')   # 0 = HC, 1 = PD

# -------------------- PREPROCESSING --------------------

def load_and_preprocess(
    wav_path: Path,
    spec_path: Path,
    plot_path: Path | None,
    do_plot: bool
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
        power=2.0
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
        full = np.pad(full, ((0,0),(0,0),(0,pad)), constant_values=SPEC_PAD_VALUE)

    spec_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(spec_path, full)

    if do_plot and plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        if not plot_path.exists():
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                logmel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                x_axis='time', y_axis='mel', fmin=FMIN, fmax=FMAX
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

# -------------------- DATASET --------------------

def label_from_parent(path: Path) -> int:
    return 0 if path.parent.name == 'HC' else 1


class MelSpecDataset(Dataset):
    def __init__(self, files: List[Path]):
        self.files  = files
        self.labels = [label_from_parent(f) for f in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = np.load(str(self.files[idx]))   # (3, N_MELS, MAX_FRAMES)
        x = torch.from_numpy(spec)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def gather_npy_files(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Raccoglie tutti i .npy in root/{HC,PD}."""
    files = []
    for lbl in LABELS:
        lbl_dir = root / lbl
        if lbl_dir.exists():
            files += list(lbl_dir.glob('*.npy'))
    labels = np.array([label_from_parent(f) for f in files])
    return np.array(files), labels

# -------------------- MODEL --------------------

class CRNNClassifier(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden_size: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.lstm = nn.LSTM(
            input_size=64 * (n_mels // 8),
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True # per catturare pattern sia forward che backward nel tempo
            # un tremore a fine frase influenza l'interpretazione anche dell'inizio 
        )
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                    # (B, C, F, T)
        x = x.permute(0, 3, 1, 2)         # (B, T, C, F)
        B, T, C, F = x.shape
        x = x.reshape(B, T, C * F)
        out, _ = self.lstm(x)
        out_max, _ = torch.max(out, dim=1)
        out_avg    = torch.mean(out, dim=1)
        out = torch.cat([out_max, out_avg], dim=1)
        out = self.dropout(out)
        return self.classifier(out).squeeze(1)

# -------------------- METRICS --------------------

def evaluate(model: nn.Module, loader: DataLoader, device: str = 'cpu') -> dict:
    probs_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x))
            probs_list.append(prob.cpu().numpy())
            labels_list.append(y.numpy())

    y_true = np.concatenate(labels_list)
    y_prob = np.concatenate(probs_list)

    prec_c, rec_c, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores   = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]

    y_pred = (y_prob > best_thresh).astype(np.float32)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return {
        'acc':       accuracy_score(y_true, y_pred),
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'roc_auc':   roc_auc_score(y_true, y_prob),
        'cm':        confusion_matrix(y_true, y_pred),
        'y_true':    y_true,
        'probs':     y_prob,
        'threshold': best_thresh,
    }


def bootstrap_evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Bootstrap sul test set: campiona con rimpiazzo n_iterations volte
    e calcola media, std e intervallo di confidenza per acc, precision,
    recall, f1 e roc_auc.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)

    metrics_boot: dict[str, list] = {
        'acc': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []
    }

    for _ in range(n_iterations):
        idx   = rng.integers(0, n, size=n)
        yt    = y_true[idx]
        yp    = y_prob[idx]
        ypred = (yp > threshold).astype(np.float32)

        metrics_boot['acc'].append(accuracy_score(yt, ypred))
        p, r, f, _ = precision_recall_fscore_support(
            yt, ypred, average='binary', zero_division=0
        )
        metrics_boot['precision'].append(p)
        metrics_boot['recall'].append(r)
        metrics_boot['f1'].append(f)
        try:
            metrics_boot['roc_auc'].append(roc_auc_score(yt, yp))
        except ValueError:          # campione con una sola classe
            pass

    alpha   = 1.0 - ci
    results = {}
    for name, values in metrics_boot.items():
        arr = np.array(values)
        results[name] = {
            'mean':  float(np.mean(arr)),
            'std':   float(np.std(arr)),
            'lower': float(np.percentile(arr, 100 * alpha / 2)),
            'upper': float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return results


def print_bootstrap_results(boot: dict, ci: float = 0.95, n_iterations: int = 1000) -> None:
    ci_pct = int(ci * 100)
    print(f"\n{'='*55}")
    print(f"BOOTSTRAP ({n_iterations} iterazioni, CI {ci_pct}%)")
    print(f"{'='*55}")
    header = f"  {'Metric':<12}  {'Mean':>7}  {'Std':>7}  {'CI lower':>9}  {'CI upper':>9}"
    print(header)
    print(f"  {'-'*56}")
    for name, vals in boot.items():
        print(
            f"  {name:<12}  {vals['mean']:>7.4f}  {vals['std']:>7.4f}"
            f"  {vals['lower']:>9.4f}  {vals['upper']:>9.4f}"
        )


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str = 'Confusion Matrix') -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()
    ticks = ['HC', 'PD']
    plt.xticks([0, 1], ticks)
    plt.yticks([0, 1], ticks)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, probs: np.ndarray, path: Path, title: str = 'ROC Curve') -> None:
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        return
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} (AUC={auc:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['test_loss'],  label='Test Loss')
    axes[0].set_title('Loss per Epoca')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['test_acc'],  label='Test Acc')
    axes[1].set_title('Accuracy per Epoca')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# -------------------- TRAINING HELPER --------------------

def step_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str = 'cpu',
    epsilon: float = 0.0
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad()
        y_smooth = y * (1 - epsilon) + 0.5 * epsilon
        logits   = model(x)
        loss     = criterion(logits, y_smooth)
        if is_train:
            loss.backward()
            optimizer.step()
        preds      = (torch.sigmoid(logits) > 0.5).float()
        correct    += (preds == y).sum().item()
        samples    += y.size(0)
        total_loss += loss.item() * y.size(0)

    return total_loss / samples, correct / samples

# -------------------- MAIN --------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('--plot', action='store_true', help='Salva spettrogrammi Mel')
    args = parser.parse_args()

    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Epoche: {args.epochs}")

    # ------------------------------------------------------------------
    # 1) PREPROCESSING
    # ------------------------------------------------------------------
    print("\n[1/3] Preprocessing Training_augmented ...")
    preprocess_split(TRAIN_DIR, PLOT_DIR / 'train' if args.plot else None, args.plot)

    print("[2/3] Preprocessing Test ...")
    preprocess_split(TEST_DIR, PLOT_DIR / 'test' if args.plot else None, args.plot)

    # ------------------------------------------------------------------
    # 2) RACCOLTA FILE E DATALOADER
    # ------------------------------------------------------------------
    train_files, train_labels = gather_npy_files(TRAIN_DIR)
    test_files,  test_labels  = gather_npy_files(TEST_DIR)

    print(f"\nFile training: {len(train_files)}  (HC={np.sum(train_labels==0)}, PD={np.sum(train_labels==1)})")
    print(f"File test:     {len(test_files)}   (HC={np.sum(test_labels==0)},  PD={np.sum(test_labels==1)})")

    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}\n")

    train_dl = DataLoader(
        MelSpecDataset(train_files.tolist()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_dl = DataLoader(
        MelSpecDataset(test_files.tolist()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # ------------------------------------------------------------------
    # 3) TRAINING
    # ------------------------------------------------------------------
    print(f"[3/3] Training su Training_augmented per {args.epochs} epoche ...\n")

    model     = CRNNClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history    = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_auc   = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = step_epoch(model, train_dl, criterion, optimizer, device, EPSILON)
        te_loss, te_acc = step_epoch(model, test_dl,  criterion, None,      device, 0.0)
        scheduler.step(te_loss)

        metrics = evaluate(model, test_dl, device)
        auc     = metrics['roc_auc']

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {tr_loss:.4f} | train_acc {tr_acc:.3f} | "
            f"test_loss {te_loss:.4f} | test_acc {te_acc:.3f} | "
            f"test_AUC {auc:.3f}"
        )

        if not np.isnan(auc) and auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt_path  = CHECKPOINT_DIR / f"best_auc_{best_auc:.3f}_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
            }, ckpt_path)
            print(f"  [CHECKPOINT] Salvato: {ckpt_path.name}")

    # ------------------------------------------------------------------
    # VALUTAZIONE FINALE CON IL MIGLIOR MODELLO
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("VALUTAZIONE FINALE sul TEST SET (miglior modello per AUC)")
    print(f"{'='*55}\n")

    best_model = CRNNClassifier().to(device)
    best_model.load_state_dict(best_state)
    final = evaluate(best_model, test_dl, device)

    scalar_keys = [k for k, v in final.items() if not isinstance(v, np.ndarray)]
    for k in scalar_keys:
        v = final[k]
        print(f"  {k:12}: {v:.4f}" if isinstance(v, float) else f"  {k:12}: {v}")

    # Bootstrap sul test set
    boot = bootstrap_evaluate(
        final['y_true'], final['probs'], final['threshold'],
        n_iterations=1000, ci=0.95
    )
    print_bootstrap_results(boot, ci=0.95, n_iterations=1000)

    # Salva plots
    plot_confusion_matrix(final['cm'],   STATS_DIR / 'test_confusion_matrix-CRNN.png', 'Test Set - Confusion Matrix')
    plot_roc_curve(final['y_true'], final['probs'], STATS_DIR / 'test_roc_curve-CRNN.png', 'Test Set - ROC Curve')
    plot_training_curves(history,        STATS_DIR / 'training_curves-CRNN.png')

    # Salva checkpoint finale
    torch.save(
        {'model_state_dict': best_state,
         'best_auc': best_auc,
         'test_metrics': {k: final[k] for k in scalar_keys},
         'bootstrap': boot},
        CHECKPOINT_DIR / 'best_model_final.pt'
    )

    print(f"\nDone. Output salvati in milestone2/")
    print(f"  → milestone2/stats/training_curves-CRNN.png")
    print(f"  → milestone2/stats/test_confusion_matrix-CRNN.png")
    print(f"  → milestone2/stats/test_roc_curve-CRNN.png")
    print(f"  → checkpoints/checkpoints_crnn/best_model_final.pt")


if __name__ == '__main__':
    main()