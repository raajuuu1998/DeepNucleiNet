"""
evaluate.py — Inter-patient cross-validation evaluation with bootstrap CIs.

Runs all 4 folds and reports mean ± SD metrics plus 95% bootstrap confidence
intervals (2000 tile-level resamples per fold) for F1-score and MCC.

Usage:
    python evaluate.py \
        --he_necrosis     data/necrosis \
        --he_non_necrosis data/non_necrosis \
        --mask_necrosis   data/masks/necrosis \
        --mask_non_necrosis data/masks/non_necrosis \
        --ckpt_dir        checkpoints
"""

import argparse
import numpy as np
from pathlib import Path

from sklearn.metrics import f1_score, matthews_corrcoef

from dataset import load_records, get_fold_split
from model import (build_feature_extractor, extract_features,
                   concatenate_features, load_svm, compute_metrics)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 n_resamples: int = 2000,
                 ci: float = 0.95) -> dict:
    """
    Compute bootstrap confidence intervals for F1 and MCC.

    Resampling is performed at the tile level within the test fold.

    Parameters
    ----------
    y_true      : np.ndarray
    y_pred      : np.ndarray
    n_resamples : int
    ci          : float — confidence level

    Returns
    -------
    dict with keys: f1_mean, f1_lower, f1_upper, mcc_mean, mcc_lower, mcc_upper
    """
    rng = np.random.default_rng(42)
    n   = len(y_true)

    f1s, mccs = [], []
    for _ in range(n_resamples):
        idx  = rng.integers(0, n, size=n)
        yt   = y_true[idx]
        yp   = y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        f1s.append(f1_score(yt, yp, zero_division=0))
        mccs.append(matthews_corrcoef(yt, yp))

    alpha = (1 - ci) / 2
    return {
        'f1_mean':  np.mean(f1s),
        'f1_lower': np.quantile(f1s, alpha),
        'f1_upper': np.quantile(f1s, 1 - alpha),
        'mcc_mean':  np.mean(mccs),
        'mcc_lower': np.quantile(mccs, alpha),
        'mcc_upper': np.quantile(mccs, 1 - alpha),
    }


# ---------------------------------------------------------------------------
# Evaluate all folds
# ---------------------------------------------------------------------------

def evaluate_all_folds(records, ckpt_dir: str):
    """
    Load trained SVM and encoders for each fold and evaluate.

    Parameters
    ----------
    records   : list of dict (from load_records)
    ckpt_dir  : str — base checkpoint directory (contains fold_1/, fold_2/, etc.)
    """
    ckpt_dir = Path(ckpt_dir)
    all_metrics = []
    all_ci      = []

    print(f"\n{'='*60}")
    print(f"  DeepNucleiNet — Inter-Patient 4-Fold Evaluation")
    print(f"{'='*60}")

    for fold in [1, 2, 3, 4]:
        fold_dir = ckpt_dir / f'fold_{fold}'
        _, test_records = get_fold_split(records, fold)

        # Load SVM
        svm = load_svm(str(fold_dir / 'svm.pkl'))

        # Rebuild feature extractors from saved weights
        from model import build_encoder_with_head
        he_model   = build_encoder_with_head()
        mask_model = build_encoder_with_head()
        he_model.load_weights(str(fold_dir / 'xception_he.h5'))
        mask_model.load_weights(str(fold_dir / 'xception_mask.h5'))
        he_extractor   = build_feature_extractor(he_model)
        mask_extractor = build_feature_extractor(mask_model)

        # Extract test features
        test_he_paths   = [r['he_path']   for r in test_records]
        test_mask_paths = [r['mask_path'] for r in test_records]
        test_labels     = np.array([r['label'] for r in test_records])

        he_feats   = extract_features(test_he_paths,   he_extractor,   is_mask=False)
        mask_feats = extract_features(test_mask_paths, mask_extractor, is_mask=True)
        X_test     = concatenate_features(he_feats, mask_feats)

        y_pred  = svm.predict(X_test)
        metrics = compute_metrics(test_labels, y_pred)
        ci      = bootstrap_ci(test_labels, y_pred)

        all_metrics.append(metrics)
        all_ci.append(ci)

        print(f"\nFold {fold}:")
        print(f"  F1-score    : {metrics['f1']:.3f}  [{ci['f1_lower']:.3f}, {ci['f1_upper']:.3f}]")
        print(f"  Balanced Acc: {metrics['bal_acc']:.3f}")
        print(f"  MCC         : {metrics['mcc']:.3f}  [{ci['mcc_lower']:.3f}, {ci['mcc_upper']:.3f}]")
        print(f"  Sensitivity : {metrics['sens']:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Mean ± SD across 4 folds")
    print(f"{'='*60}")
    for key, label in [('f1','F1-score'), ('bal_acc','Balanced Acc'),
                        ('mcc','MCC'), ('sens','Sensitivity')]:
        vals = [m[key] for m in all_metrics]
        print(f"  {label:<15}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DeepNucleiNet across all folds.')
    parser.add_argument('--he_necrosis',       required=True)
    parser.add_argument('--he_non_necrosis',   required=True)
    parser.add_argument('--mask_necrosis',     required=True)
    parser.add_argument('--mask_non_necrosis', required=True)
    parser.add_argument('--ckpt_dir',          default='checkpoints')
    args = parser.parse_args()

    records = load_records(
        args.he_necrosis, args.he_non_necrosis,
        args.mask_necrosis, args.mask_non_necrosis,
    )
    evaluate_all_folds(records, ckpt_dir=args.ckpt_dir)
