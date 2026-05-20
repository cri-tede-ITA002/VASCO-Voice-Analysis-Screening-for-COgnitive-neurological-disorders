#!/usr/bin/env python3
"""
Bootstrap standalone sul Test Set — senza rieseguire il training.

Uso:
    python bootstrap_eval.py
    python bootstrap_eval.py --checkpoint milestone2/checkpoints/best_model_final.pt
    python bootstrap_eval.py --checkpoint milestone2/checkpoints/best_auc_0.XXX_epochN.pt
    python bootstrap_eval.py --n_boot 2000 --ci 0.99

Il checkpoint deve contenere 'model_state_dict'.
I file .npy del test set devono essere già presenti in data/Test/{HC,PD}/
(vengono generati al primo run di CRNN.py).
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

# ── importa tutto dal modulo principale ─────────────────────────────────────
from CRNN import (
    CRNNClassifier,
    MelSpecDataset,
    gather_npy_files,
    evaluate,
    bootstrap_evaluate,
    print_bootstrap_results,
    TEST_DIR,
    CHECKPOINT_DIR,
    STATS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)
from torch.utils.data import DataLoader


# ────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(ckpt_dir: Path) -> Path:
    """Cerca best_model_final.pt in checkpoints/checkpoints_crnn/."""
    final = ckpt_dir / 'best_model_final.pt'
    if final.exists():
        return final
    # fallback: prende il .pt con AUC più alta nel nome (best_auc_X.XXX_epochN.pt)
    candidates = sorted(ckpt_dir.glob('best_auc_*.pt'))
    if not candidates:
        raise FileNotFoundError(
            f"Nessun checkpoint trovato in {ckpt_dir}.\n"
            "Assicurati che esista checkpoints/checkpoints_crnn/best_model_final.pt "
            "oppure almeno un best_auc_*.pt nella stessa cartella."
        )
    def auc_from_name(p: Path) -> float:
        try:
            return float(p.stem.split('_')[2])
        except (IndexError, ValueError):
            return 0.0
    return max(candidates, key=auc_from_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap evaluation sul Test Set CRNN")
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help="Percorso del checkpoint .pt (default: auto-detect in milestone2/checkpoints/)"
    )
    parser.add_argument('--n_boot', type=int, default=1000, help="Numero iterazioni bootstrap (default 1000)")
    parser.add_argument('--ci',     type=float, default=0.95, help="Livello CI (default 0.95)")
    parser.add_argument('--seed',   type=int,   default=42,   help="Seed casuale (default 42)")
    args = parser.parse_args()

    # ── 1. Trova checkpoint ──────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_best_checkpoint(CHECKPOINT_DIR)
    print(f"Checkpoint caricato: {ckpt_path}")

    # ── 2. Carica modello ────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = CRNNClassifier().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── 3. Carica test set ───────────────────────────────────────────────────
    test_files, test_labels = gather_npy_files(TEST_DIR)
    if len(test_files) == 0:
        raise RuntimeError(
            f"Nessun file .npy trovato in {TEST_DIR}. "
            "Esegui CRNN.py almeno una volta per generare i file preprocessati."
        )
    print(f"File test: {len(test_files)}  "
          f"(HC={np.sum(test_labels==0)}, PD={np.sum(test_labels==1)})")

    test_dl = DataLoader(
        MelSpecDataset(test_files.tolist()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # ── 4. Inferenza sul test set ─────────────────────────────────────────────
    print("\nInferenza sul test set ...")
    final = evaluate(model, test_dl, device)

    print("\nMetriche puntuali (soglia ottimale F1):")
    for k, v in final.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k:<12}: {v:.4f}" if isinstance(v, float) else f"  {k:<12}: {v}")

    # ── 5. Bootstrap ─────────────────────────────────────────────────────────
    print(f"\nEseguo bootstrap ({args.n_boot} iterazioni, CI {int(args.ci*100)}%) ...")
    boot = bootstrap_evaluate(
        final['y_true'], final['probs'], final['threshold'],
        n_iterations=args.n_boot, ci=args.ci, seed=args.seed
    )
    print_bootstrap_results(boot, ci=args.ci, n_iterations=args.n_boot)

    # ── 6. Salva risultati ────────────────────────────────────────────────────
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / 'bootstrap_results.npz'
    np.savez(
        out_path,
        y_true=final['y_true'],
        y_prob=final['probs'],
        **{f"boot_{k}_{stat}": v
           for k, vals in boot.items()
           for stat, v in vals.items()}
    )
    print(f"\nRisultati salvati in: {out_path}")


if __name__ == '__main__':
    main()
