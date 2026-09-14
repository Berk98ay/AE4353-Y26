import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

# Preprocessing settings. These are the numbers used to build FFHQ_FACES.npz,
# and they are kept here so the shipped file and the on-the-fly image mode
# always agree.
CROP_FRAC = 0.8  # center crop, as a fraction of the shorter side
IMG_SIZE = 32  # output resolution (IMG_SIZE * IMG_SIZE = x_dim)
SPLIT_SEED = 4353  # fixed so the train/test split is reproducible
N_TEST = 10_000  # matches MNIST's 60k / 10k split (70000 // 7 == 10000)


def preprocess_face(img, crop_frac=CROP_FRAC, size=IMG_SIZE):
    """Turn one FFHQ thumbnail into a small grayscale array.

    FFHQ images are already aligned and centered, but they leave a generous
    margin of background around the face. We crop that away, drop the color
    and shrink to `size` x `size`.

    Parameters:
    - img (PIL.Image): a single FFHQ thumbnail.
    - crop_frac (float): center crop size, as a fraction of the shorter side.
    - size (int): output width and height in pixels.

    Returns a (size, size) uint8 array.
    """
    from PIL import Image

    w, h = img.size
    side = int(min(w, h) * crop_frac)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img = img.convert("L")
    img = img.resize((size, size), Image.LANCZOS)

    return np.asarray(img, dtype=np.uint8)


def _load_npz(source, split):
    """Read one split out of the prepared FFHQ_FACES.npz."""
    with np.load(source) as data:
        if split not in data:
            available = ", ".join(sorted(data.files))
            raise KeyError(f"'{split}' not in {source} (has: {available})")
        return data[split]


def _load_images(source, split, crop_frac, size):
    """Preprocess a folder of raw FFHQ images into the same array layout.

    This is the optional path: it lets you point the dataset straight at the
    public FFHQ Kaggle dataset instead of the prepared file. It is slower,
    since every image is decoded and resized, but it produces identical
    tensors.
    """
    from PIL import Image
    from tqdm import tqdm

    paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(glob.glob(os.path.join(source, "**", pattern), recursive=True))
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(f"No images found under {source}")

    # Same shuffle and split as the build script, so both sources line up.
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(paths)

    # Scale the test split down if this folder holds only part of FFHQ, so
    # pointing at a subset still gives you a usable train set.
    n_test = min(N_TEST, max(1, len(paths) // 7))

    if split == "train":
        paths = paths[:-n_test]
    elif split == "test":
        paths = paths[-n_test:]
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    faces = np.empty((len(paths), size, size), dtype=np.uint8)
    for i, path in enumerate(tqdm(paths, desc=f"Preprocessing {split}")):
        with Image.open(path) as img:
            faces[i] = preprocess_face(img, crop_frac=crop_frac, size=size)

    return faces


class FaceDataset(Dataset):
    """FFHQ faces, grayscale and downsampled, ready for the VAE.

    Accepts either source:

    - a path to the prepared `FFHQ_FACES.npz` (what both notebooks use), or
    - a directory of raw FFHQ images, preprocessed on the fly.

    Parameters:
    - source (str): path to the .npz file, or to a folder of images.
    - split (str): "train" or "test".
    - flatten (bool): True gives a flat (size*size,) vector for the MLP;
      False gives a (1, size, size) image, ready for Conv2d if you swap the
      encoder and decoder for convolutional ones.
    - crop_frac, size: only used for the image-folder path.

    __getitem__ returns `(x, 0)`. The second element is a placeholder so the
    dataset unpacks the same way MNIST did — there are no labels here.
    """

    def __init__(
        self,
        source,
        split="train",
        flatten=True,
        crop_frac=CROP_FRAC,
        size=IMG_SIZE,
    ):
        self.source = source
        self.split = split
        self.flatten = flatten

        if os.path.isdir(source):
            self.faces = _load_images(source, split, crop_frac, size)
        else:
            self.faces = _load_npz(source, split)

        self.img_shape = self.faces.shape[1:]

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        face = torch.from_numpy(self.faces[idx].astype(np.float32) / 255.0)

        # Flat vector for the MLP, or a single-channel image for a conv net.
        face = face.reshape(-1) if self.flatten else face.unsqueeze(0)

        return face, 0


if __name__ == "__main__":
    # Smoke test. Point this at your own copy of the dataset.
    path = "../data/AE4353-Datasets-2026/FFHQ_FACES.npz"

    for split in ("train", "test"):
        dataset = FaceDataset(path, split=split)
        x, _ = dataset[0]
        print(
            f"{split:>5}: {len(dataset)} samples, "
            f"x {tuple(x.shape)} {x.dtype}, "
            f"range [{x.min():.3f}, {x.max():.3f}]"
        )
