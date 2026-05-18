import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from torchvggish import vggish, vggish_input

# =========================================================
# FOLDER CONFIG
# =========================================================
DATA_ROOT = Path("data")
CHECKPOINT_DIR = Path("artifacts/checkpoints_vggish_lr_128")
PLOT_DIR = Path("artifacts/plots")
STATS_DIR = Path("artifacts/stats")

for d in (CHECKPOINT_DIR, PLOT_DIR, STATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================
SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0
RANDOM_SEED = 42
FRAMES_PER_AUDIO = 4
FRAME_EMBED_SIZE = 128
EMBEDDING_SIZE = FRAMES_PER_AUDIO * FRAME_EMBED_SIZE  # 512

HC_LABEL = 0
PD_LABEL = 1

# =========================================================
# LOAD VGGish MODEL
# =========================================================
def load_vggish_model():
    model = vggish()
    model.eval()
    return model

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_vggish_embeddings(file_path, model):
    """
    Estrae embedding VGGish da un file audio.
    Output finale: vettore 512D = 4 frame x 128D
    """
    try:
        wav, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        wav = wav[: int(MAX_SECONDS * SAMPLE_RATE)]

        # se audio troppo corto, padding minimo fino a 1 secondo
        if len(wav) < SAMPLE_RATE:
            wav = np.pad(wav, (0, SAMPLE_RATE - len(wav)))

        example = vggish_input.waveform_to_examples(wav, SAMPLE_RATE)

        if not isinstance(example, torch.Tensor):
            example = torch.from_numpy(example)

        with torch.no_grad():
            embedding = model(example)

        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)


        if embedding.shape[0] < FRAMES_PER_AUDIO:
            pad = torch.zeros(FRAMES_PER_AUDIO - embedding.shape[0], FRAME_EMBED_SIZE)
            embedding = torch.cat([embedding, pad], dim=0)
        else:
            embedding = embedding[:FRAMES_PER_AUDIO]

        return embedding.flatten().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None

# =========================================================
# DATASET
# =========================================================
def infer_label_from_path(path: Path) -> int:
    """
    HC -> 0
    PD -> 1
    """
    parent_name = path.parent.name.upper()

    if parent_name == "HC":
        return HC_LABEL
    if parent_name == "PD":
        return PD_LABEL

    raise ValueError(f"Cartella non riconosciuta: {path.parent.name}")

def collect_all_audio_files(data_root: Path):
    """
    Raccoglie tutti i file wav da HC e PD.
    """
    files = []
    for sub in ("HC", "PD"):
        files.extend((data_root / sub).glob("*.wav"))
    return sorted(files, key=lambda p: p.name)

def load_dataset(files, model):
    X, y, file_paths = [], [], []

    for path in files:
        emb = extract_vggish_embeddings(str(path), model)
        if emb is not None:
            X.append(emb)
            y.append(infer_label_from_path(path))
            file_paths.append(str(path))

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(file_paths),
    )

# =========================================================
# METRICS
# =========================================================
def compute_metrics_dict(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }

# =========================================================
# FINAL PLOTS
# =========================================================
def save_final_metrics_and_plots(y_true, y_prob, out_prefix="vggish_cv_final", threshold=0.5):
    metrics = compute_metrics_dict(y_true, y_prob, threshold=threshold)
    cm = np.array(metrics["confusion_matrix"])

    with open(CHECKPOINT_DIR / f"{out_prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Confusion Matrix finale
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Final OOF")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    plt.xticks([0, 1], ["HC", "PD"])
    plt.yticks([0, 1], ["HC", "PD"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{out_prefix}_confusion_matrix.png", dpi=200)
    plt.close()

    # ROC finale
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Final OOF")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{out_prefix}_roc_curve.png", dpi=200)
        plt.close()
    except Exception:
        pass

    return metrics

# =========================================================
# STRATIFIED K-FOLD CV (AUDIO-LEVEL)
# =========================================================
def run_stratified_kfold_cv_only(X, y, file_paths, n_splits=5):
    print(f"\n[INFO] Avvio StratifiedKFold Cross Validation ({n_splits} fold)...")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    cv_results = []

    # out-of-fold probabilities
    oof_prob = np.full(shape=len(y), fill_value=np.nan, dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n[INFO] Fold {fold}/{n_splits}")

        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        print(f"  Train fold -> HC: {np.sum(y_train_fold == 0)}, PD: {np.sum(y_train_fold == 1)}")
        print(f"  Test  fold -> HC: {np.sum(y_test_fold == 0)}, PD: {np.sum(y_test_fold == 1)}")

        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
        clf.fit(X_train_fold, y_train_fold)

        y_prob = clf.predict_proba(X_test_fold)[:, 1]
        oof_prob[test_idx] = y_prob

        metrics = compute_metrics_dict(y_test_fold, y_prob, threshold=0.5)
        metrics["fold"] = fold
        metrics["n_train_samples"] = int(len(train_idx))
        metrics["n_test_samples"] = int(len(test_idx))
        metrics["n_test_hc"] = int(np.sum(y_test_fold == 0))
        metrics["n_test_pd"] = int(np.sum(y_test_fold == 1))

        cv_results.append(metrics)

        print(f"  Accuracy          : {metrics['accuracy']:.4f}")
        print(f"  Precision         : {metrics['precision']:.4f}")
        print(f"  Recall (Sensi.)   : {metrics['recall']:.4f}")
        print(f"  F1 Score          : {metrics['f1']:.4f}")
        print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
        print(f"  Specificity       : {metrics['specificity']:.4f}")
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

    if np.isnan(oof_prob).any():
        missing = int(np.isnan(oof_prob).sum())
        raise RuntimeError(f"OOF prediction mancanti per {missing} campioni.")

    # summary media sui fold
    summary = {
        "n_samples": int(len(y)),
        "n_hc": int(np.sum(y == 0)),
        "n_pd": int(np.sum(y == 1)),
        "n_splits": int(n_splits),
        "accuracy_mean": float(np.mean([m["accuracy"] for m in cv_results])),
        "accuracy_std": float(np.std([m["accuracy"] for m in cv_results])),
        "precision_mean": float(np.mean([m["precision"] for m in cv_results])),
        "precision_std": float(np.std([m["precision"] for m in cv_results])),
        "recall_mean": float(np.mean([m["recall"] for m in cv_results])),
        "recall_std": float(np.std([m["recall"] for m in cv_results])),
        "f1_mean": float(np.mean([m["f1"] for m in cv_results])),
        "f1_std": float(np.std([m["f1"] for m in cv_results])),
        "roc_auc_mean": float(np.mean([m["roc_auc"] for m in cv_results])),
        "roc_auc_std": float(np.std([m["roc_auc"] for m in cv_results])),
        "specificity_mean": float(np.mean([m["specificity"] for m in cv_results])),
        "specificity_std": float(np.std([m["specificity"] for m in cv_results])),
    }

    # metriche finali aggregate OOF
    final_oof_metrics = save_final_metrics_and_plots(
        y_true=y,
        y_prob=oof_prob,
        out_prefix="vggish_stratified_kfold_cv_final",
        threshold=0.5,
    )

    print("\n" + "=" * 60)
    print("          STRATIFIED K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Samples        : {summary['n_samples']}")
    print(f"  HC             : {summary['n_hc']}")
    print(f"  PD             : {summary['n_pd']}")
    print("-" * 60)
    print(f"  Accuracy mean  : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  Precision mean : {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    print(f"  Recall mean    : {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    print(f"  F1 mean        : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  ROC-AUC mean   : {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")
    print(f"  Spec mean      : {summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}")
    print("-" * 60)
    print("  FINAL OOF METRICS")
    print(f"  Accuracy       : {final_oof_metrics['accuracy']:.4f}")
    print(f"  Precision      : {final_oof_metrics['precision']:.4f}")
    print(f"  Recall         : {final_oof_metrics['recall']:.4f}")
    print(f"  F1 Score       : {final_oof_metrics['f1']:.4f}")
    print(f"  ROC-AUC        : {final_oof_metrics['roc_auc']:.4f}")
    print(f"  Specificity    : {final_oof_metrics['specificity']:.4f}")
    print(f"  TP={final_oof_metrics['tp']}  FP={final_oof_metrics['fp']}  "
          f"TN={final_oof_metrics['tn']}  FN={final_oof_metrics['fn']}")
    print("=" * 60)


    with open(CHECKPOINT_DIR / "vggish_stratified_kfold_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "folds": cv_results,
                "summary": summary,
                "final_oof_metrics": final_oof_metrics,
                "oof_predictions": [
                    {
                        "file_path": str(file_paths[i]),
                        "y_true": int(y[i]),
                        "y_prob": float(oof_prob[i]),
                    }
                    for i in range(len(y))
                ],
            },
            f,
            indent=2,
        )

    return cv_results, summary, final_oof_metrics

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--comments", type=str, default="")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"EXECUTION TIME: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("[INFO] Starting VGGish embedding classification.")
    if args.comments:
        print("[COMMENT]", args.comments)

    all_files = collect_all_audio_files(DATA_ROOT)

    if len(all_files) == 0:
        raise RuntimeError("Nessun file trovato in data/HC o data/PD.")

    print(f"[INFO] Totale file trovati: {len(all_files)}")

    model = load_vggish_model()

    print("\n[INFO] Extracting features from all files...")
    X, y, file_paths = load_dataset(all_files, model)

    if len(X) == 0:
        raise RuntimeError("Nessun embedding estratto correttamente.")

    print(f"[INFO] Totale campioni: {len(X)}")
    print(f"[INFO] HC (0): {np.sum(y == 0)}")
    print(f"[INFO] PD (1): {np.sum(y == 1)}")
    print(f"[INFO] Shape embeddings: {X.shape}")

    run_stratified_kfold_cv_only(
        X=X,
        y=y,
        file_paths=file_paths,
        n_splits=args.n_splits,
    )

    print("\n[INFO] Done.")

if __name__ == "__main__":
    main()