"""
nuclei_map.py — Nuclei map generation using HoVerNet (PanNuke checkpoint)
                via TIA Toolbox, followed by binarization.

Each H&E tile is processed by HoVerNet to produce instance segmentation maps.
The instance map is then binarized: nuclei pixels -> white (255), background -> black (0).
The resulting binary mask is the input to the nuclei-map encoder stream in DeepNucleiNet.

Requirements:
    tiatoolbox==1.6.0
    numpy==1.26.4

Usage:
    python nuclei_map.py --he_dir data/necrosis --out_dir data/masks/necrosis
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def load_segmentor(batch_size: int = 32, num_workers: int = 4):
    """
    Load HoVerNet (hovernet_fast-pannuke) via TIA Toolbox.

    Parameters
    ----------
    batch_size  : int
    num_workers : int

    Returns
    -------
    segmentor : NucleusInstanceSegmentor
    """
    from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

    segmentor = NucleusInstanceSegmentor(
        pretrained_model='hovernet_fast-pannuke',
        num_loader_workers=num_workers,
        num_postproc_workers=num_workers,
        batch_size=batch_size,
        verbose=False,
    )
    return segmentor


def binarize_instance_map(instance_map: np.ndarray) -> np.ndarray:
    """
    Convert a HoVerNet instance map to a binary nuclei mask.

    In HoVerNet output, background pixels have value 0.
    Any pixel belonging to a nucleus instance has value > 0.

    Parameters
    ----------
    instance_map : np.ndarray, shape (H, W), int

    Returns
    -------
    binary_mask : np.ndarray, shape (H, W), uint8
                  255 = nucleus, 0 = background
    """
    binary_mask = (instance_map > 0).astype(np.uint8) * 255
    return binary_mask


def generate_nuclei_maps(he_dir: str,
                          out_dir: str,
                          batch_size: int = 32,
                          num_workers: int = 4):
    """
    Process all H&E tiles in he_dir, generate binarized nuclei masks,
    and save them to out_dir with the same filenames.

    Parameters
    ----------
    he_dir      : str — directory containing H&E tiles (.png)
    out_dir     : str — output directory for binary masks
    batch_size  : int
    num_workers : int
    """
    he_dir  = Path(he_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = sorted(he_dir.glob('*.png'))
    print(f"Found {len(tile_paths)} tiles in {he_dir}")

    segmentor = load_segmentor(batch_size=batch_size, num_workers=num_workers)

    for tile_path in tile_paths:
        out_path = out_dir / tile_path.name

        if out_path.exists():
            continue   # skip already processed tiles

        # HoVerNet expects a list of image paths
        output = segmentor.predict(
            imgs=[str(tile_path)],
            save_dir=None,
            mode='tile',
            on_gpu=True,
            crash_on_exception=False,
        )

        # output is a dict: {filename: {'inst_map': np.ndarray, ...}}
        key          = list(output.keys())[0]
        instance_map = output[key]['inst_map']
        binary_mask  = binarize_instance_map(instance_map)

        Image.fromarray(binary_mask).save(out_path)

    print(f"Done. Masks saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate binarized nuclei maps using HoVerNet PanNuke.'
    )
    parser.add_argument('--he_dir',     required=True, help='Directory of H&E tiles (.png)')
    parser.add_argument('--out_dir',    required=True, help='Output directory for binary masks')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers',type=int, default=4)
    args = parser.parse_args()

    generate_nuclei_maps(
        he_dir=args.he_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
