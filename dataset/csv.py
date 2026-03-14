from copy import deepcopy
import h5py
import math
import json
import numpy as np
from PIL import Image
import random
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box
from typing import Tuple, Any, Dict, Union, Optional


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] if max > 1 (assumes uint8-like range)."""
    if arr.size == 0:
        return arr
    if arr.max() > 1.0:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def map_mask_values(mask: np.ndarray) -> np.ndarray:
    """Map {0, 128, 255} → {0, 1, 2} robustly."""
    mask = mask.astype(np.int64)
    # Only remap if needed (avoid double-mapping)
    if mask.max() > 2:
        mask = np.where(mask == 128, 1, mask)
        mask = np.where(mask == 255, 2, mask)
    return mask


class CSVSemiDataset(Dataset):
    def __init__(
        self,
        json_file_path: str,
        mode: str,
        size: Optional[int] = None,
        n_sample: Optional[int] = None
    ):
        self.json_file_path = json_file_path
        self.mode = mode
        self.size = size
        self.n_sample = n_sample

        with open(self.json_file_path, 'r') as f:
            self.case_list = json.load(f)

        if mode == 'train_l' and n_sample is not None:
            repeat = math.ceil(n_sample / len(self.case_list))
            self.case_list = (self.case_list * repeat)[:n_sample]

    def _read_pair(self, image_h5_file: str) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(image_h5_file, 'r') as f:
            long_img = normalize_to_01(f['long_img'][:])
            trans_img = normalize_to_01(f['trans_img'][:])
        return long_img, trans_img

    def _read_label(self, label_h5_file: str) -> Tuple[np.ndarray, np.ndarray, int]:
        with h5py.File(label_h5_file, 'r') as f:
            long_mask = map_mask_values(f['long_mask'][:])
            trans_mask = map_mask_values(f['trans_mask'][:])
            cls = int(f['cls'][()])  # ensure scalar int
        return long_mask, trans_mask, cls

    def _resize(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Resize image (and optionally mask) to target size using nearest neighbor for masks."""
        h, w = img.shape[-2:] if img.ndim == 3 else img.shape
        scale_h = self.size / h
        scale_w = self.size / w

        if mask is not None:
            # Use order=0 (nearest) for masks to preserve label integrity
            resized_img = zoom(img, (scale_h, scale_w), order=1)  # bilinear for image
            resized_mask = zoom(mask, (scale_h, scale_w), order=0)  # nearest for mask
            return resized_img, resized_mask
        else:
            # For unlabeled: just resize image (order=1 is fine)
            return zoom(img, (scale_h, scale_w), order=1)

    def _augment_labeled(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply same geometric augmentation to image and mask."""
        if random.random() > 0.5:
            return random_rot_flip(img, mask)
        elif random.random() > 0.5:
            return random_rotate(img, mask)
        return img, mask

    def _process_unlabeled(self, img: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate weak and two strong augmentations for unlabeled data."""
        # Geometric aug (same as labeled)
        if random.random() > 0.5:
            img = random_rot_flip(img)
        elif random.random() > 0.5:
            img = random_rotate(img)

        # Resize
        img = self._resize(img)

        # Convert to PIL for color jitter & blur
        pil_img = Image.fromarray((img * 255).astype(np.uint8))

        # Weak augmentation: just tensor
        img_w = torch.from_numpy(np.array(pil_img)).unsqueeze(0).float() / 255.0

        # Strong augmentation 1
        img_s1 = deepcopy(pil_img)
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0
        box1 = obtain_cutmix_box(self.size, p=0.5)

        # Strong augmentation 2
        img_s2 = deepcopy(pil_img)
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0
        box2 = obtain_cutmix_box(self.size, p=0.5)

        return img_w, img_s1, img_s2, box1, box2

    def __getitem__(self, item: int):
        case = self.case_list[item]

        if self.mode == 'valid':
            long_img, trans_img = self._read_pair(case['image'])
            long_mask, trans_mask, cls = self._read_label(case['label'])

            return (
                torch.from_numpy(long_img).unsqueeze(0),
                torch.from_numpy(trans_img).unsqueeze(0),
                torch.from_numpy(long_mask).long(),
                torch.from_numpy(trans_mask).long(),
                torch.tensor(cls).long()
            )

        elif self.mode == 'train_l':
            long_img, trans_img = self._read_pair(case['image'])
            long_mask, trans_mask, cls = self._read_label(case['label'])

            # Apply same geometric aug to both views
            long_img, long_mask = self._augment_labeled(long_img, long_mask)
            trans_img, trans_mask = self._augment_labeled(trans_img, trans_mask)

            # Resize
            long_img, long_mask = self._resize(long_img, long_mask)
            trans_img, trans_mask = self._resize(trans_img, trans_mask)

            return (
                torch.from_numpy(long_img).unsqueeze(0),
                torch.from_numpy(trans_img).unsqueeze(0),
                torch.from_numpy(long_mask).long(),
                torch.from_numpy(trans_mask).long(),
                torch.tensor(cls).long()
            )

        elif self.mode == 'train_u':
            long_img, trans_img = self._read_pair(case['image'])

            long_w, long_s1, long_s2, box_l1, box_l2 = self._process_unlabeled(long_img)
            trans_w, trans_s1, trans_s2, box_t1, box_t2 = self._process_unlabeled(trans_img)

            return long_w, long_s1, long_s2, box_l1, box_l2, trans_w, trans_s1, trans_s2, box_t1, box_t2

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __len__(self) -> int:
        return len(self.case_list)