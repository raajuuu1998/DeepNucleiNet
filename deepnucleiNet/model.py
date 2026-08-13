"""
model.py — DeepNucleiNet dual-stream XceptionNet + RBF-SVM classifier.

Architecture:
    - Histology stream  : XceptionNet (ImageNet pretrained) -> 2048-d features
    - Nuclei-map stream : XceptionNet (ImageNet pretrained) -> 2048-d features
    - Concatenation     : [histology | nuclei] -> 4096-d
    - Classifier        : RBF-kernel SVM (C=10, gamma=0.001)

Both XceptionNet encoders are fine-tuned end-to-end with a fully connected
classification head for 50 epochs, then the FC heads are discarded and the
frozen encoder features are passed to the SVM.
"""

import numpy as np
import joblib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.svm import SVC
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                              matthews_corrcoef, recall_score)


# ---------------------------------------------------------------------------
# Encoder construction
# ---------------------------------------------------------------------------

def build_encoder_with_head(num_classes: int = 2,
                              dropout: float = 0.3) -> Model:
    """
    Build XceptionNet with a fully connected classification head for fine-tuning.

    Parameters
    ----------
    num_classes : int
    dropout     : float

    Returns
    -------
    model : tf.keras.Model  (input -> class logits)
    """
    base = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3),
    )
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)


def build_feature_extractor(trained_model: Model) -> Model:
    """
    Strip the FC head from a fine-tuned encoder to expose 2048-d features.

    Parameters
    ----------
    trained_model : fine-tuned Keras model (output = class logits)

    Returns
    -------
    extractor : tf.keras.Model  (input -> 2048-d GlobalAveragePooling output)
    """
    gap_layer = trained_model.get_layer('global_average_pooling2d')
    return Model(inputs=trained_model.input, outputs=gap_layer.output)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(paths,
                     extractor: Model,
                     batch_size: int = 128,
                     is_mask: bool = False) -> np.ndarray:
    """
    Extract 2048-d features from a list of image paths using a frozen extractor.

    Parameters
    ----------
    paths      : list of str or Path — image file paths
    extractor  : frozen Keras feature extractor
    batch_size : int
    is_mask    : bool — if True, replicate single channel to 3 channels

    Returns
    -------
    features : np.ndarray, shape (N, 2048)
    """
    from PIL import Image
    from tqdm import tqdm

    features = []
    for i in tqdm(range(0, len(paths), batch_size), desc='Extracting'):
        batch_paths = paths[i:i + batch_size]
        batch_imgs  = []
        for p in batch_paths:
            img = np.array(Image.open(p).convert('RGB'), dtype=np.float32)
            if is_mask:
                img = np.stack([img[:, :, 0]] * 3, axis=-1)
            batch_imgs.append(img)
        batch_imgs = preprocess_input(np.stack(batch_imgs))
        feats      = extractor.predict(batch_imgs, verbose=0)
        features.append(feats)
    return np.vstack(features)


def concatenate_features(he_features: np.ndarray,
                          mask_features: np.ndarray) -> np.ndarray:
    """Concatenate H&E and nuclei-map features to 4096-d."""
    return np.concatenate([he_features, mask_features], axis=1)


# ---------------------------------------------------------------------------
# SVM classifier
# ---------------------------------------------------------------------------

def build_svm(C: float = 10.0, gamma: float = 0.001) -> SVC:
    """
    Build RBF-kernel SVM with class-balanced weighting.

    Parameters
    ----------
    C     : float — regularisation parameter
    gamma : float — RBF kernel coefficient

    Returns
    -------
    svm : sklearn.svm.SVC
    """
    return SVC(
        kernel='rbf',
        C=C,
        gamma=gamma,
        class_weight='balanced',
        probability=False,
    )


def train_svm(svm: SVC,
              X_train: np.ndarray,
              y_train: np.ndarray) -> SVC:
    """Fit the SVM on concatenated 4096-d training features."""
    svm.fit(X_train, y_train)
    return svm


def save_svm(svm: SVC, path: str):
    """Save trained SVM to disk."""
    joblib.dump(svm, path)
    print(f"SVM saved -> {path}")


def load_svm(path: str) -> SVC:
    """Load trained SVM from disk."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray) -> dict:
    """
    Compute F1, balanced accuracy, MCC, and sensitivity.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    metrics : dict
    """
    return {
        'f1':       f1_score(y_true, y_pred, zero_division=0),
        'bal_acc':  balanced_accuracy_score(y_true, y_pred),
        'mcc':      matthews_corrcoef(y_true, y_pred),
        'sens':     recall_score(y_true, y_pred, zero_division=0),
    }
