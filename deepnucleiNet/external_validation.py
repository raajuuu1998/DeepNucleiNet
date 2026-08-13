"""
external_validation.py — Cross-tumor generalization without retraining.

Evaluates the meningioma-trained DeepNucleiNet on external datasets:
    - TCGA GBM   (Grade 4 Glioma)
    - TCGA LGG   (Low-Grade Glioma — necrosis-free control)
    - DeepHisto  (Mixed Glioma)
    - TiGER      (Breast TNBC)

The model is applied directly without any retraining or fine-tuning.

Usage:
    python external_validation.py \
        --he_dir     external/TCGA_GBM/he \
        --mask_dir   external/TCGA_GBM/masks \
        --labels_csv external/TCGA_GBM/labels.csv \
        --ckpt_dir   checkpoints/fold_1 \
        --dataset    TCGA_GBM
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from model import (build_encoder_with_head, build_feature_extractor,
                   extract_features, concatenate_features, load_svm,
                   compute_metrics)


def validate_external(he_dir: str,
                       mask_dir: str,
                       labels_csv: str,
                       ckpt_dir: str,
                       dataset_name: str):
    """
    Run external validation for one dataset.

    Parameters
    ----------
    he_dir       : str — directory of H&E tiles
    mask_dir     : str — directory of binarized nuclei masks
    labels_csv   : str — CSV with columns: filename, label (1=necrosis, 0=non-necrosis)
    ckpt_dir     : str — fold checkpoint directory (e.g. checkpoints/fold_1)
    dataset_name : str — display name for logging
    """
    ckpt_dir = Path(ckpt_dir)
    he_dir   = Path(he_dir)
    mask_dir = Path(mask_dir)

    # Load labels
    df        = pd.read_csv(labels_csv)
    filenames = df['filename'].tolist()
    labels    = np.array(df['label'].tolist())

    he_paths   = [str(he_dir   / f) for f in filenames]
    mask_paths = [str(mask_dir / f) for f in filenames]

    # Load encoders
    he_model   = build_encoder_with_head()
    mask_model = build_encoder_with_head()
    he_model.load_weights(str(ckpt_dir / 'xception_he.h5'))
    mask_model.load_weights(str(ckpt_dir / 'xception_mask.h5'))
    he_extractor   = build_feature_extractor(he_model)
    mask_extractor = build_feature_extractor(mask_model)

    # Extract features
    print(f"\nExtracting features for {dataset_name}...")
    he_feats   = extract_features(he_paths,   he_extractor,   is_mask=False)
    mask_feats = extract_features(mask_paths, mask_extractor, is_mask=True)
    X          = concatenate_features(he_feats, mask_feats)

    # Load SVM and predict
    svm    = load_svm(str(ckpt_dir / 'svm.pkl'))
    y_pred = svm.predict(X)

    # Handle necrosis-free datasets (e.g. TCGA LGG)
    n_necrosis = int(np.sum(labels == 1))
    if n_necrosis == 0:
        specificity = float(np.mean(y_pred == 0))
        print(f"\n{dataset_name} (necrosis-free control):")
        print(f"  Tiles          : {len(labels)}")
        print(f"  Necrosis preds : {int(np.sum(y_pred == 1))}")
        print(f"  Specificity    : {specificity:.3f}")
        return

    metrics = compute_metrics(labels, y_pred)
    acc     = float(np.mean(y_pred == labels))

    print(f"\n{dataset_name}:")
    print(f"  Tiles        : {len(labels)}  (N={n_necrosis}, NN={len(labels)-n_necrosis})")
    print(f"  Accuracy     : {acc:.3f}")
    print(f"  Sensitivity  : {metrics['sens']:.3f}")
    print(f"  F1-score     : {metrics['f1']:.3f}")
    print(f"  MCC          : {metrics['mcc']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='External cross-tumor validation for DeepNucleiNet.'
    )
    parser.add_argument('--he_dir',     required=True)
    parser.add_argument('--mask_dir',   required=True)
    parser.add_argument('--labels_csv', required=True,
                        help='CSV with columns: filename, label')
    parser.add_argument('--ckpt_dir',   required=True,
                        help='Fold checkpoint directory (e.g. checkpoints/fold_1)')
    parser.add_argument('--dataset',    required=True,
                        help='Dataset display name (e.g. TCGA_GBM)')
    args = parser.parse_args()

    validate_external(
        he_dir=args.he_dir,
        mask_dir=args.mask_dir,
        labels_csv=args.labels_csv,
        ckpt_dir=args.ckpt_dir,
        dataset_name=args.dataset,
    )
