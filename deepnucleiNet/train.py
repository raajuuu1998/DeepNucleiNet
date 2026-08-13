"""
train.py — Fine-tune dual XceptionNet encoders and train RBF-SVM for DeepNucleiNet.

Training procedure (per fold):
    1. Fine-tune histology XceptionNet (H&E stream) for 50 epochs with FC head
    2. Fine-tune nuclei XceptionNet (mask stream) for 50 epochs with FC head
    3. Discard FC heads, extract frozen 2048-d features from both streams
    4. Concatenate to 4096-d, train RBF-SVM

Usage:
    python train.py \
        --he_necrosis     data/necrosis \
        --he_non_necrosis data/non_necrosis \
        --mask_necrosis   data/masks/necrosis \
        --mask_non_necrosis data/masks/non_necrosis \
        --out_dir         checkpoints \
        --fold            1
"""

import argparse
import os
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from dataset import load_records, get_fold_split
from model import (build_encoder_with_head, build_feature_extractor,
                   extract_features, concatenate_features,
                   build_svm, train_svm, save_svm, compute_metrics)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LR         = 1e-4
EPOCHS     = 50
BATCH_SIZE = 32
SVM_C      = 10.0
SVM_GAMMA  = 0.001


# ---------------------------------------------------------------------------
# Data generator (simple tf.data pipeline)
# ---------------------------------------------------------------------------

def make_tf_dataset(records, batch_size, shuffle=True):
    """
    Build a tf.data.Dataset from a list of records.
    Returns (he_img, mask_img), label batches.
    """
    from tensorflow.keras.applications.xception import preprocess_input
    from PIL import Image

    def generator():
        idxs = list(range(len(records)))
        if shuffle:
            np.random.shuffle(idxs)
        for i in idxs:
            r = records[i]
            he   = np.array(Image.open(r['he_path']).convert('RGB'),  dtype=np.float32)
            mask = np.array(Image.open(r['mask_path']).convert('L'),  dtype=np.float32)
            mask = np.stack([mask] * 3, axis=-1)
            he   = preprocess_input(he)
            mask = preprocess_input(mask)
            yield (he, mask), r['label']

    output_sig = (
        (tf.TensorSpec(shape=(256,256,3), dtype=tf.float32),
         tf.TensorSpec(shape=(256,256,3), dtype=tf.float32)),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Single-stream fine-tuning
# ---------------------------------------------------------------------------

def finetune_encoder(train_records, stream: str, epochs: int, lr: float,
                     batch_size: int, out_dir: Path) -> tf.keras.Model:
    """
    Fine-tune one XceptionNet stream (histology or nuclei).

    Parameters
    ----------
    train_records : list of dict
    stream        : 'he' or 'mask'
    epochs        : int
    lr            : float
    batch_size    : int
    out_dir       : Path — checkpoint directory

    Returns
    -------
    extractor : frozen feature extractor (2048-d output)
    """
    from PIL import Image
    from tensorflow.keras.applications.xception import preprocess_input

    print(f"\nFine-tuning {stream.upper()} encoder for {epochs} epochs...")

    model = build_encoder_with_head(num_classes=2)
    model.compile(
        optimizer=Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Build simple numpy arrays for fit() — feasible for tile-level training
    imgs, labels = [], []
    for r in train_records:
        if stream == 'he':
            img = np.array(Image.open(r['he_path']).convert('RGB'),  dtype=np.float32)
        else:
            img = np.array(Image.open(r['mask_path']).convert('L'),  dtype=np.float32)
            img = np.stack([img] * 3, axis=-1)
        imgs.append(preprocess_input(img))
        labels.append(r['label'])

    imgs   = np.stack(imgs)
    labels = np.array(labels)

    cb = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(
        imgs, labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb],
        verbose=1,
    )

    ckpt_path = out_dir / f'xception_{stream}.h5'
    model.save_weights(str(ckpt_path))
    print(f"Checkpoint saved -> {ckpt_path}")

    extractor = build_feature_extractor(model)
    extractor.trainable = False
    return extractor


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_fold(records, fold: int, out_dir: str):
    """
    Full training pipeline for one inter-patient fold.

    Parameters
    ----------
    records  : list of dict (from load_records)
    fold     : int (1-4)
    out_dir  : str — directory to save SVM and encoder checkpoints
    """
    out_dir = Path(out_dir) / f'fold_{fold}'
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records, test_records = get_fold_split(records, fold)
    print(f"\nFold {fold} | Train: {len(train_records)} | Test: {len(test_records)}")

    # --- Fine-tune both encoders ---
    he_extractor   = finetune_encoder(train_records, 'he',   EPOCHS, LR, BATCH_SIZE, out_dir)
    mask_extractor = finetune_encoder(train_records, 'mask', EPOCHS, LR, BATCH_SIZE, out_dir)

    # --- Extract features ---
    print("\nExtracting train features...")
    train_he_paths   = [r['he_path']   for r in train_records]
    train_mask_paths = [r['mask_path'] for r in train_records]
    train_labels     = np.array([r['label'] for r in train_records])

    he_train   = extract_features(train_he_paths,   he_extractor,   is_mask=False)
    mask_train = extract_features(train_mask_paths, mask_extractor, is_mask=True)
    X_train    = concatenate_features(he_train, mask_train)

    print("Extracting test features...")
    test_he_paths   = [r['he_path']   for r in test_records]
    test_mask_paths = [r['mask_path'] for r in test_records]
    test_labels     = np.array([r['label'] for r in test_records])

    he_test   = extract_features(test_he_paths,   he_extractor,   is_mask=False)
    mask_test = extract_features(test_mask_paths, mask_extractor, is_mask=True)
    X_test    = concatenate_features(he_test, mask_test)

    # --- Train SVM ---
    print("\nTraining SVM...")
    svm = build_svm(C=SVM_C, gamma=SVM_GAMMA)
    svm = train_svm(svm, X_train, train_labels)
    save_svm(svm, str(out_dir / 'svm.pkl'))

    # --- Evaluate ---
    y_pred  = svm.predict(X_test)
    metrics = compute_metrics(test_labels, y_pred)

    print(f"\nFold {fold} Results:")
    print(f"  F1-score        : {metrics['f1']:.3f}")
    print(f"  Balanced Acc    : {metrics['bal_acc']:.3f}")
    print(f"  MCC             : {metrics['mcc']:.3f}")
    print(f"  Sensitivity     : {metrics['sens']:.3f}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepNucleiNet for one fold.')
    parser.add_argument('--he_necrosis',       required=True)
    parser.add_argument('--he_non_necrosis',   required=True)
    parser.add_argument('--mask_necrosis',     required=True)
    parser.add_argument('--mask_non_necrosis', required=True)
    parser.add_argument('--out_dir',           default='checkpoints')
    parser.add_argument('--fold', type=int,    choices=[1,2,3,4], required=True)
    args = parser.parse_args()

    from dataset import load_records
    records = load_records(
        args.he_necrosis, args.he_non_necrosis,
        args.mask_necrosis, args.mask_non_necrosis,
    )
    train_fold(records, fold=args.fold, out_dir=args.out_dir)
