"""
dataset.py — Tile loading and inter-patient fold assignments for DeepNucleiNet.

Data is expected in the following flat directory structure:
    data/
        necrosis/       <- H&E tiles, filenames prefixed with SID (e.g. S2_tile.png)
        non_necrosis/   <- H&E tiles, filenames prefixed with SID
        masks/
            necrosis/       <- binarized nuclei masks (same filenames as H&E)
            non_necrosis/

Inter-patient fold assignments (as reported in the paper):
    Fold 1 — Test: S2
    Fold 2 — Test: S5, S7
    Fold 3 — Test: S6
    Fold 4 — Test: S4
    S1, S3, S8 — train-only (no necrosis tiles; cannot serve as test partitions)
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Fold configuration
# ---------------------------------------------------------------------------

FOLD_MAP = {
    'S2': 1,
    'S5': 2, 'S7': 2,
    'S6': 3,
    'S4': 4,
}

FOLDS = {
    1: {'test': ['S2'],       'train': ['S1', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']},
    2: {'test': ['S5', 'S7'], 'train': ['S1', 'S2', 'S3', 'S4', 'S6', 'S8']},
    3: {'test': ['S6'],       'train': ['S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S8']},
    4: {'test': ['S4'],       'train': ['S1', 'S2', 'S3', 'S5', 'S6', 'S7', 'S8']},
}


def get_sid(filename: str) -> str:
    """Extract sample ID from tile filename (e.g. 'S2_tile-x123.png' -> 'S2')."""
    return Path(filename).name.split('_')[0]


def load_records(he_necrosis_dir: str,
                 he_non_necrosis_dir: str,
                 mask_necrosis_dir: str,
                 mask_non_necrosis_dir: str):
    """
    Build a list of (he_path, mask_path, label, sid) tuples for all tiles.

    Returns
    -------
    records : list of dict
        Each dict has keys: he_path, mask_path, label (1=necrosis, 0=non-necrosis), sid, fold
    """
    records = []

    for label, he_dir, mask_dir in [
        (1, he_necrosis_dir,     mask_necrosis_dir),
        (0, he_non_necrosis_dir, mask_non_necrosis_dir),
    ]:
        he_dir   = Path(he_dir)
        mask_dir = Path(mask_dir)

        for he_path in sorted(he_dir.glob('*.png')):
            mask_path = mask_dir / he_path.name
            if not mask_path.exists():
                continue  # skip tiles without a corresponding mask
            sid  = get_sid(he_path.name)
            fold = FOLD_MAP.get(sid, 0)   # fold 0 = train-only patients
            records.append({
                'he_path':   str(he_path),
                'mask_path': str(mask_path),
                'label':     label,
                'sid':       sid,
                'fold':      fold,
            })

    return records


def get_fold_split(records, fold: int):
    """
    Split records into train and test for a given fold.

    Parameters
    ----------
    records : list of dict  (output of load_records)
    fold    : int           (1, 2, 3, or 4)

    Returns
    -------
    train_records, test_records : list of dict
    """
    test_sids  = set(FOLDS[fold]['test'])
    train_records = [r for r in records if r['sid'] not in test_sids]
    test_records  = [r for r in records if r['sid'] in test_sids]
    return train_records, test_records


def load_image_pair(record, xception_preprocess=True):
    """
    Load an H&E tile and its corresponding binarized nuclei mask as numpy arrays.

    The mask is loaded as grayscale and replicated to 3 channels so it can
    be passed through the same XceptionNet encoder as the H&E tile.

    Parameters
    ----------
    record              : dict (from load_records)
    xception_preprocess : bool — apply Keras xception preprocess_input scaling

    Returns
    -------
    he_img   : np.ndarray, shape (256, 256, 3), float32
    mask_img : np.ndarray, shape (256, 256, 3), float32
    label    : int
    """
    he_img   = np.array(Image.open(record['he_path']).convert('RGB'),   dtype=np.float32)
    mask_img = np.array(Image.open(record['mask_path']).convert('L'),   dtype=np.float32)
    mask_img = np.stack([mask_img] * 3, axis=-1)   # replicate to 3 channels

    if xception_preprocess:
        from tensorflow.keras.applications.xception import preprocess_input
        he_img   = preprocess_input(he_img)
        mask_img = preprocess_input(mask_img)

    return he_img, mask_img, record['label']
