#!/usr/bin/env python3
"""
AST.py — Rilevamento Parkinson da voce con Audio Spectrogram Transformer
=========================================================================

Esecuzione:
    python AST.py                 # training + test + bootstrap + salva threshold
    python AST.py --eval-only     # solo test + bootstrap + salva threshold
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification


# =========================
# CONFIG
# =========================

TRAIN_ROOT = Path("Training_augmented")
TEST_ROOT  = Path("Test")

LABELS = ("HC", "PD")
LABEL_TO_ID = {"HC": 0, "PD": 1}
ID_TO_LABEL = {0: "HC", 1: "PD"}

SAMPLE_RATE = 16_000
WINDOW_SEC  = 5.0
HOP_SEC     = 2.5

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

BATCH_SIZE  = 4
NUM_WORKERS = 0

EPOCHS       = 15
LR           = 1e-4
WEIGHT_DECAY = 1e-4

VAL_SIZE    = 0.20
RANDOM_SEED = 42

# Suffissi di augmentation
AUG_SUFFIXES = ("_noise", "_pitch_up", "_pitch_down", "_speed_up", "_speed_down")

CHECKPOINT_DIR = Path("checkpoints_ast")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT_PATH         = CHECKPOINT_DIR / "best_ast_frozen.pt"
TEST_PREDICTIONS_PATH  = CHECKPOINT_DIR / "test_subject_predictions_ast.csv"
BOOTSTRAP_RESULTS_PATH = CHECKPOINT_DIR / "bootstrap_test_ast.txt"

BOOTSTRAP_N    = 1000
BOOTSTRAP_SEED = 42


# =========================
# UTILS
# =========================

def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def strip_aug_suffix(stem: str) -> str:
    """Rimuove il suffisso di augmentation dal nome file (vedi AUG_SUFFIXES)."""
    for suffix in AUG_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# =========================
# FILE DISCOVERY
# =========================

def list_audio_files(root: Path) -> List[Tuple[Path, int, str]]:
    rows: List[Tuple[Path, int, str]] = []
    for label_name in LABELS:
        label_dir = root / label_name
        if not label_dir.exists():
            print(f"[WARN] Directory non trovata: {label_dir}")
            continue
        for wav_path in sorted(label_dir.glob("*.wav")):
            rows.append((wav_path, LABEL_TO_ID[label_name], label_name))
    return rows


def subject_id_from_path(path: Path, label_name: str) -> str:
    """
    Ritorna l'ID del soggetto, rimuovendo i suffissi di augmentation in modo
    che TUTTE le versioni augmentate dello stesso paziente abbiano lo stesso
    subject_id.

    Esempi:
        HC/AH_064F_..._C5.wav             →  HC_AH_064F_..._C5
        HC/AH_064F_..._C5_noise.wav       →  HC_AH_064F_..._C5
        HC/AH_064F_..._C5_pitch_down.wav  →  HC_AH_064F_..._C5
    """
    return f"{label_name}_{strip_aug_suffix(path.stem)}"


# =========================
# DATASET
# =========================

class ASTWindowDataset(Dataset):
    def __init__(
        self,
        root: Path,
        feature_extractor: ASTFeatureExtractor,
        sample_rate: int = SAMPLE_RATE,
        window_sec: float = WINDOW_SEC,
        hop_sec: float = HOP_SEC,
    ):
        self.root = root
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self.hop_samples = int(hop_sec * sample_rate)

        self.items = []

        audio_files = list_audio_files(root)

        for wav_path, label_id, label_name in audio_files:
            # Calcoliamo le finestre sull'audio già normalizzato e trimmato,
            # cioè lo stesso segnale che verrà usato in __getitem__.
            y_tmp, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
            y_tmp = librosa.util.normalize(y_tmp)
            y_tmp, _ = librosa.effects.trim(y_tmp, top_db=35)
            total_samples = len(y_tmp)

            subject_id = subject_id_from_path(wav_path, label_name)

            if total_samples <= 0:
                continue

            if total_samples <= self.window_samples:
                self.items.append({
                    "path": wav_path,
                    "label": label_id,
                    "label_name": label_name,
                    "subject_id": subject_id,
                    "start_sample": 0,
                })
            else:
                for start in range(
                    0,
                    total_samples - self.window_samples + 1,
                    self.hop_samples,
                ):
                    self.items.append({
                        "path": wav_path,
                        "label": label_id,
                        "label_name": label_name,
                        "subject_id": subject_id,
                        "start_sample": start,
                    })

                last_start = max(0, total_samples - self.window_samples)
                if (
                    self.items[-1]["path"] != wav_path
                    or self.items[-1]["start_sample"] != last_start
                ):
                    self.items.append({
                        "path": wav_path,
                        "label": label_id,
                        "label_name": label_name,
                        "subject_id": subject_id,
                        "start_sample": last_start,
                    })

        if not self.items:
            raise RuntimeError(f"Nessun item trovato in {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        wav_path = item["path"]
        start_sample = item["start_sample"]

        y, _ = librosa.load(str(wav_path), sr=self.sample_rate, mono=True)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y, top_db=35)

        start = start_sample
        end = start + self.window_samples
        chunk = y[start:end]

        if len(chunk) < self.window_samples:
            pad = self.window_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad), mode="constant")

        inputs = self.feature_extractor(
            chunk,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        input_values = inputs["input_values"].squeeze(0)
        label = torch.tensor(item["label"], dtype=torch.long)

        return {
            "input_values": input_values,
            "label": label,
            "subject_id": item["subject_id"],
            "path": str(wav_path),
        }


# =========================
# COLLATE
# =========================

def ast_collate_fn(batch):
    input_values = torch.stack([b["input_values"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    subject_ids = [b["subject_id"] for b in batch]
    paths = [b["path"] for b in batch]

    return {
        "input_values": input_values,
        "labels": labels,
        "subject_ids": subject_ids,
        "paths": paths,
    }


def make_dataset_from_items(items, feature_extractor):
    ds = ASTWindowDataset.__new__(ASTWindowDataset)
    ds.root = None
    ds.feature_extractor = feature_extractor
    ds.sample_rate = SAMPLE_RATE
    ds.window_samples = int(WINDOW_SEC * SAMPLE_RATE)
    ds.hop_samples = int(HOP_SEC * SAMPLE_RATE)
    ds.items = items
    return ds


def split_items_by_subject(items, val_size: float = VAL_SIZE, seed: int = RANDOM_SEED):
    subject_to_label = {}
    for item in items:
        subject_to_label[item["subject_id"]] = item["label"]

    subjects = sorted(subject_to_label.keys())
    labels = [subject_to_label[s] for s in subjects]

    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=val_size,
        stratify=labels,
        random_state=seed,
    )

    train_subjects = set(train_subjects)
    val_subjects = set(val_subjects)

    train_items = [item for item in items if item["subject_id"] in train_subjects]
    val_items   = [item for item in items if item["subject_id"] in val_subjects]

    return train_items, val_items


def check_leakage(train_subjects, val_subjects, test_subjects) -> None:
    tr = set(train_subjects)
    va = set(val_subjects)
    te = set(test_subjects)

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


# =========================
# MODEL
# =========================

def build_frozen_ast_model(device: torch.device):
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)
    return model


# =========================
# TRAIN / EVAL
# =========================

def step_epoch(model, loader, optimizer, device):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    samples = 0

    for batch in tqdm(loader, leave=False):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            if is_train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        samples += labels.size(0)

    return total_loss / samples, correct / samples


def evaluate_speaker_level(model, loader, device, threshold: float = 0.5):
    """
    Aggrega le probabilità delle finestre per subject_id (media) e calcola
    metriche con la soglia indicata.
    """
    model.eval()

    speaker_probs = defaultdict(list)
    speaker_labels = {}

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].cpu().numpy()
            subject_ids = batch["subject_ids"]

            logits = model(input_values=input_values).logits
            probs_pd = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for subject_id, label, prob in zip(subject_ids, labels, probs_pd):
                speaker_probs[subject_id].append(float(prob))
                speaker_labels[subject_id] = int(label)

    subjects = sorted(speaker_probs.keys())

    y_true = np.array([speaker_labels[s] for s in subjects])
    y_prob = np.array([np.mean(speaker_probs[s]) for s in subjects])
    y_pred = (y_prob > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "cm": cm,
        "y_true": y_true,
        "y_prob": y_prob,
        "subjects": subjects,
        "threshold": threshold,
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trova la soglia che massimizza F1 (uso esclusivo sul VAL set)."""
    prec_c, rec_c, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-8)
    if len(f1_scores) == 0 or np.all(np.isnan(f1_scores)):
        return 0.5
    return float(thresholds[np.argmax(f1_scores)])


# =========================
# BOOTSTRAP
# =========================

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
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc,
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

    values = {
        "accuracy":    [],
        "precision":   [],
        "recall":      [],
        "specificity": [],
        "f1":          [],
        "roc_auc":     [],
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_b = y_true[idx]
        y_prob_b = y_prob[idx]

        # roc_auc richiede entrambe le classi nel campione
        if len(np.unique(y_true_b)) < 2:
            continue

        metrics = compute_metrics_from_predictions(y_true_b, y_prob_b, threshold)
        for key in values:
            if not (isinstance(metrics[key], float) and np.isnan(metrics[key])):
                values[key].append(metrics[key])

    summary = {}
    for key, metric_values in values.items():
        arr = np.array(metric_values, dtype=float)
        if arr.size == 0:
            summary[key] = {
                "mean":    float("nan"),
                "std":     float("nan"),
                "ci_low":  float("nan"),
                "ci_high": float("nan"),
            }
        else:
            summary[key] = {
                "mean":    float(np.mean(arr)),
                "std":     float(np.std(arr)),
                "ci_low":  float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }

    return summary


def save_subject_predictions(metrics: dict, output_path: Path, threshold: float) -> None:
    subjects = metrics["subjects"]
    y_true = metrics["y_true"]
    y_prob = metrics["y_prob"]

    with output_path.open("w", encoding="utf-8") as f:
        f.write("subject_id,y_true,y_prob_pd,y_pred\n")
        for subject_id, true_label, prob in zip(subjects, y_true, y_prob):
            pred = int(prob > threshold)
            f.write(f"{subject_id},{int(true_label)},{float(prob):.8f},{pred}\n")


def save_bootstrap_summary(summary: dict, output_path: Path) -> None:
    """Stessa tabulazione usata da CRNN.py per confronto diretto."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("BOOTSTRAP TEST RESULTS — AST FROZEN — SUBJECT LEVEL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<12}{'Mean':>10}{'Std':>10}{'CI lower':>12}{'CI upper':>12}\n")
        f.write("-" * 56 + "\n")
        for key, values in summary.items():
            f.write(
                f"{key:<12}{values['mean']:>10.4f}{values['std']:>10.4f}"
                f"{values['ci_low']:>12.4f}{values['ci_high']:>12.4f}\n"
            )


def update_checkpoint_with_threshold(
    ckpt_path: Path,
    device: torch.device,
    best_threshold: float,
    test_metrics: dict,
    bootstrap_summary: dict,
) -> None:
    """
    [PATCH] Riapre il best checkpoint e vi aggiunge:
      - threshold: soglia ottima F1 calcolata sul val set
      - test_metrics: metriche finali sul test set
      - bootstrap: summary del bootstrap

    Lascia inalterati tutti gli altri campi (epoch, model_state_dict, ecc.).
    L'agente (agente.py) usa `ckpt["threshold"]` per la decisione PD/HC.
    """
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint non trovato per la patch: {ckpt_path}")
        return

    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_data["threshold"] = float(best_threshold)
    ckpt_data["test_metrics"] = {
        "acc":       float(test_metrics["acc"]),
        "precision": float(test_metrics["precision"]),
        "recall":    float(test_metrics["recall"]),
        "f1":        float(test_metrics["f1"]),
        "roc_auc":   float(test_metrics["roc_auc"]),
    }
    ckpt_data["bootstrap"] = bootstrap_summary

    torch.save(ckpt_data, ckpt_path)
    print(f"[INFO] Checkpoint aggiornato con threshold={best_threshold:.4f} "
          f"e metriche di test: {ckpt_path}")


# =========================
# RICALIBRAZIONE SOGLIA (criteri alternativi a F1)  [AGGIUNTA]
# =========================

def _candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    """Griglia di soglie candidate: prob uniche + punti intermedi + estremi."""
    u = np.unique(y_prob)
    mids = (u[:-1] + u[1:]) / 2.0 if len(u) > 1 else u
    return np.unique(np.concatenate([[0.0], u, mids, [1.0]]))


def threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Soglia che massimizza Youden J = recall + specificity - 1 (sul VAL)."""
    best_j, best_t = -2.0, 0.5
    for t in _candidate_thresholds(y_prob):
        m = compute_metrics_from_predictions(y_true, y_prob, float(t))
        spec = m["specificity"]
        spec = 0.0 if (isinstance(spec, float) and np.isnan(spec)) else spec
        j = m["recall"] + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def threshold_target_sensitivity(y_true: np.ndarray, y_prob: np.ndarray,
                                 min_recall: float = 0.90):
    """Soglia PIÙ ALTA con recall >= min_recall sul VAL (max specificità sotto
    il vincolo di sensibilità). Ritorna (soglia, raggiunto?)."""
    feasible = [
        float(t) for t in _candidate_thresholds(y_prob)
        if compute_metrics_from_predictions(y_true, y_prob, float(t))["recall"] >= min_recall
    ]
    if feasible:
        return max(feasible), True
    # ripiego: nessuna soglia raggiunge il target -> massimizza il recall
    best_r, best_t = -1.0, 0.5
    for t in _candidate_thresholds(y_prob):
        r = compute_metrics_from_predictions(y_true, y_prob, float(t))["recall"]
        if r > best_r:
            best_r, best_t = r, float(t)
    return best_t, False


def derive_threshold_by_criterion(criterion: str, y_true: np.ndarray,
                                  y_prob: np.ndarray, min_recall: float = 0.90):
    """Ritorna (soglia, feasible). 'f1' riusa la vostra find_optimal_threshold."""
    if criterion == "f1":
        return find_optimal_threshold(y_true, y_prob), True
    if criterion == "youden":
        return threshold_youden(y_true, y_prob), True
    if criterion == "target_sens":
        return threshold_target_sensitivity(y_true, y_prob, min_recall)
    raise ValueError(f"Criterio sconosciuto: {criterion}")


def run_recalibration(model, val_dl, test_dl, device, stage: str,
                      criterion: str = "target_sens", min_recall: float = 0.90) -> None:
    """
    Ricalibrazione soglia in due stadi, anti-leakage:
      stage='val'  -> deriva e MOSTRA le soglie (F1/Youden/target_sens) SUL VAL.
                      Non tocca il test. Serve a scegliere il criterio a priori.
      stage='test' -> applica al TEST la soglia del criterio scelto a priori,
                      con metriche + bootstrap. AUC test = canary (~0.85).
    Il VAL è quello esatto del training (stesso split, include augmentation):
    l'unica variabile che cambia è il CRITERIO di scelta della soglia.
    """
    val_metrics = evaluate_speaker_level(model, val_dl, device, threshold=0.5)
    yv, pv = val_metrics["y_true"], val_metrics["y_prob"]

    if stage == "val":
        print("\n" + "=" * 64)
        print("RICALIBRAZIONE — SOGLIE DERIVATE SUL VAL (subject-level)")
        print("=" * 64)
        print(f"VAL soggetti : {len(yv)}  (PD={int(yv.sum())}  HC={int((1 - yv).sum())})")
        print(f"VAL ROC-AUC  : {val_metrics['roc_auc']:.4f}  (indipendente dalla soglia)\n")
        print(f"{'criterio':<26}{'thr':>9}{'recall':>9}{'spec':>9}{'f1':>9}{'acc':>9}")
        print("-" * 71)
        for name, crit in [("F1 (baseline)", "f1"),
                           ("Youden J", "youden"),
                           (f"Target-sens>={min_recall:.2f}", "target_sens")]:
            thr, feasible = derive_threshold_by_criterion(crit, yv, pv, min_recall)
            m = compute_metrics_from_predictions(yv, pv, thr)
            tag = "" if feasible else "  [target non raggiunto->max recall]"
            print(f"{name:<26}{thr:>9.3f}{m['recall']:>9.3f}"
                  f"{m['specificity']:>9.3f}{m['f1']:>9.3f}{m['accuracy']:>9.3f}{tag}")
        print("\n[CHECK] La riga F1 dovrebbe riprodurre ~0.749 (la vostra soglia attuale).")
        print("Scegli UN criterio a priori (obiettivo clinico), poi lancia:")
        print("  python AST_FINALE.py --recalibrate test --criterion <f1|youden|target_sens>")
        return

    # stage == 'test'
    thr, feasible = derive_threshold_by_criterion(criterion, yv, pv, min_recall)
    if not feasible:
        print(f"[ATTENZIONE] recall>={min_recall} non raggiungibile sul val; "
              f"uso soglia di max-recall = {thr:.4f}")
    print(f"\n[VAL] soglia '{criterion}' = {thr:.4f}  (derivata sul val, fissata)\n")

    test_metrics = evaluate_speaker_level(model, test_dl, device, threshold=thr)
    cm = test_metrics["cm"]
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print("=" * 64)
    print(f"TEST — criterio '{criterion}' — soglia {thr:.4f}")
    print("=" * 64)
    print(f"Subjects     : {len(test_metrics['subjects'])}")
    print(f"Accuracy     : {test_metrics['acc']:.4f}")
    print(f"Precision    : {test_metrics['precision']:.4f}")
    print(f"Recall/Sens. : {test_metrics['recall']:.4f}")
    print(f"Specificity  : {spec:.4f}")
    print(f"F1           : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC      : {test_metrics['roc_auc']:.4f}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    if not (0.78 <= test_metrics["roc_auc"] <= 0.92):
        print(f"[CANARY] AUC test = {test_metrics['roc_auc']:.3f} fuori da ~0.85: "
              "l'inferenza non combacia, verifica prima di usare i numeri.")
    else:
        print(f"[CANARY] AUC test = {test_metrics['roc_auc']:.3f} ~ 0.85: coerente.")

    bs = bootstrap_subject_level_ci(test_metrics["y_true"], test_metrics["y_prob"],
                                    threshold=thr)
    print("-" * 64)
    print(f"BOOTSTRAP ({BOOTSTRAP_N} iter, CI 95%)")
    print(f"{'Metric':<12}{'Mean':>10}{'Std':>10}{'CI low':>12}{'CI high':>12}")
    print("-" * 56)
    for k, v in bs.items():
        print(f"{k:<12}{v['mean']:>10.4f}{v['std']:>10.4f}"
              f"{v['ci_low']:>12.4f}{v['ci_high']:>12.4f}")
    print("\n[NB] Il checkpoint NON è stato modificato. Se adotti questa soglia "
          "per l'agente VASCO, aggiorna ckpt['threshold'].")


def run_test_and_bootstrap(model, val_dl, test_dl, device) -> None:
    # 1) Soglia ottima F1 calcolata sul VAL (subject-level)
    print()
    print("[INFO] Calcolo soglia ottima F1 su VAL (subject-level)...")
    val_metrics = evaluate_speaker_level(model, val_dl, device, threshold=0.5)
    best_threshold = find_optimal_threshold(val_metrics["y_true"], val_metrics["y_prob"])
    print(f"[INFO] Soglia ottima da VAL: {best_threshold:.4f}")

    # 2) Valutazione finale TEST con soglia FISSA da val
    print()
    print("[INFO] Valutazione finale TEST speaker-level (soglia da VAL)...")
    test_metrics = evaluate_speaker_level(model, test_dl, device, threshold=best_threshold)

    cm = test_metrics["cm"]
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print()
    print("=" * 60)
    print("FINAL TEST METRICS — AST FROZEN — SPEAKER LEVEL")
    print("=" * 60)
    print(f"Subjects     : {len(test_metrics['subjects'])}")
    print(f"Threshold    : {best_threshold:.4f}  (selezionata su VAL)")
    print(f"Accuracy     : {test_metrics['acc']:.4f}")
    print(f"Precision    : {test_metrics['precision']:.4f}")
    print(f"Recall/Sens. : {test_metrics['recall']:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"F1           : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC      : {test_metrics['roc_auc']:.4f}")
    print("-" * 60)
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("=" * 60)

    print()
    print("[INFO] Salvataggio predizioni subject-level TEST...")
    save_subject_predictions(test_metrics, TEST_PREDICTIONS_PATH, best_threshold)
    print(f"[INFO] Predizioni salvate in: {TEST_PREDICTIONS_PATH}")

    print()
    print(f"[INFO] Bootstrap subject-level sul TEST ({BOOTSTRAP_N} iter.)...")
    bootstrap_summary = bootstrap_subject_level_ci(
        y_true=test_metrics["y_true"],
        y_prob=test_metrics["y_prob"],
        threshold=best_threshold,
        n_bootstrap=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
    )
    save_bootstrap_summary(bootstrap_summary, BOOTSTRAP_RESULTS_PATH)

    print()
    print("=" * 60)
    print(f"BOOTSTRAP ({BOOTSTRAP_N} iterazioni, CI 95%)")
    print("=" * 60)
    print(f"{'Metric':<12}{'Mean':>10}{'Std':>10}{'CI lower':>12}{'CI upper':>12}")
    print("-" * 56)
    for k, v in bootstrap_summary.items():
        print(f"{k:<12}{v['mean']:>10.4f}{v['std']:>10.4f}"
              f"{v['ci_low']:>12.4f}{v['ci_high']:>12.4f}")
    print("=" * 60)
    print(f"[INFO] Bootstrap salvato in: {BOOTSTRAP_RESULTS_PATH}")

    # ── [PATCH] Aggiorna il checkpoint con threshold + metriche di test ──────
    print()
    print("[INFO] Aggiorno il checkpoint con threshold e metriche di test...")
    update_checkpoint_with_threshold(
        ckpt_path=BEST_CKPT_PATH,
        device=device,
        best_threshold=best_threshold,
        test_metrics=test_metrics,
        bootstrap_summary=bootstrap_summary,
    )


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS)
    parser.add_argument('--eval-only', action='store_true',
                        help='Salta training, carica best checkpoint e fa solo test+bootstrap')
    parser.add_argument('--recalibrate', choices=['val', 'test'], default=None,
                        help="Ricalibra la soglia: 'val' mostra i criteri sul val, "
                             "'test' applica al test il criterio scelto")
    parser.add_argument('--criterion', choices=['f1', 'youden', 'target_sens'],
                        default='target_sens',
                        help='Criterio soglia (solo con --recalibrate test); deciso a priori')
    parser.add_argument('--min-recall', type=float, default=0.90,
                        help='Vincolo di sensibilità per target_sens')
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)

    device = get_device()
    print(f"[INFO] Device: {device}")

    print("[INFO] Carico ASTFeatureExtractor...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)

    # ── Creo SEMPRE i dataset train+val+test (anche in eval-only),
    #    perché il val serve per calcolare la soglia ottima F1.
    print("[INFO] Creo dataset training completo...")
    full_train_ds = ASTWindowDataset(TRAIN_ROOT, feature_extractor)
    print("[INFO] Creo dataset test...")
    test_ds = ASTWindowDataset(TEST_ROOT, feature_extractor)

    # Split train/val a livello di SOGGETTO (subject_id già normalizzato
    # dalle augmentation).
    train_items, val_items = split_items_by_subject(
        full_train_ds.items,
        val_size=VAL_SIZE,
        seed=RANDOM_SEED,
    )
    train_ds = make_dataset_from_items(train_items, feature_extractor)
    val_ds   = make_dataset_from_items(val_items,   feature_extractor)

    train_subjects = sorted({item["subject_id"] for item in train_ds.items})
    val_subjects   = sorted({item["subject_id"] for item in val_ds.items})
    test_subjects  = sorted({item["subject_id"] for item in test_ds.items})

    print()
    print("Split summary")
    print("-" * 60)
    print(f"  TRAIN : {len(train_ds):5d} finestre   /  {len(train_subjects):3d} soggetti")
    print(f"  VAL   : {len(val_ds):5d} finestre   /  {len(val_subjects):3d} soggetti")
    print(f"  TEST  : {len(test_ds):5d} finestre   /  {len(test_subjects):3d} soggetti")

    check_leakage(train_subjects, val_subjects, test_subjects)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=ast_collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=ast_collate_fn,
    )
    test_dl = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=ast_collate_fn,
    )

    print()
    print("[INFO] Carico ASTForAudioClassification frozen...")
    model = build_frozen_ast_model(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali      : {total:,}")
    print(f"Parametri addestrabili: {trainable:,}")

    # ── Ricalibrazione soglia (carica checkpoint, NON addestra, NON sovrascrive) ──
    if args.recalibrate is not None:
        if not BEST_CKPT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {BEST_CKPT_PATH}")
        ckpt = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Checkpoint caricato per ricalibrazione: {BEST_CKPT_PATH} | "
              f"epoch={ckpt.get('epoch','?')} | val_auc_spk={ckpt.get('best_val_auc', float('nan')):.3f}")
        run_recalibration(model, val_dl, test_dl, device,
                          stage=args.recalibrate,
                          criterion=args.criterion,
                          min_recall=args.min_recall)
        return

    # ── Training oppure caricamento checkpoint ──────────────────────────
    if args.eval_only:
        print()
        print("[INFO] --eval-only: salto il training e carico il checkpoint salvato...")
        if not BEST_CKPT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {BEST_CKPT_PATH}")

        ckpt = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"[INFO] Caricato checkpoint: {BEST_CKPT_PATH} | "
            f"epoch={ckpt['epoch']} | val_auc_spk={ckpt['best_val_auc']:.3f}"
        )

        run_test_and_bootstrap(model, val_dl, test_dl, device)
        return

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_auc = -1.0
    best_state = None

    print()
    print("=" * 60)
    print("TRAINING AST FROZEN")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = step_epoch(model, train_dl, optimizer, device)
        val_loss, val_acc     = step_epoch(model, val_dl,   None,      device)
        val_metrics = evaluate_speaker_level(model, val_dl, device, threshold=0.5)
        val_auc = val_metrics["roc_auc"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc_win={val_acc:.3f} | "
            f"val_auc_spk={val_auc:.3f}"
        )

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_auc": best_auc,
                    "val_metrics": {
                        "acc": val_metrics["acc"],
                        "precision": val_metrics["precision"],
                        "recall": val_metrics["recall"],
                        "f1": val_metrics["f1"],
                        "roc_auc": val_metrics["roc_auc"],
                        "cm": val_metrics["cm"],
                    },
                    "model_name": MODEL_NAME,
                    "sample_rate": SAMPLE_RATE,
                    "window_sec": WINDOW_SEC,
                    "hop_sec": HOP_SEC,
                    "label_to_id": LABEL_TO_ID,
                },
                BEST_CKPT_PATH,
            )

            print(
                f"  [CHECKPOINT] Salvato: {BEST_CKPT_PATH} | "
                f"epoch={epoch} | val_auc_spk={best_auc:.3f}"
            )

    if best_state is None:
        raise RuntimeError("Nessun checkpoint valido salvato.")

    print()
    print("[INFO] Carico il miglior checkpoint per il test finale...")
    ckpt = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[INFO] Caricato checkpoint: {BEST_CKPT_PATH} | "
        f"epoch={ckpt['epoch']} | val_auc_spk={ckpt['best_val_auc']:.3f}"
    )

    run_test_and_bootstrap(model, val_dl, test_dl, device)


if __name__ == "__main__":
    main()