"""
Created by: Mohammed Abdelrahim
"""
import os
from torch.utils.data import Dataset
import cv2
import torch

class CrackDataSets(Dataset):
    def __init__(self, base_dir, split="train", image_file=None, mask_file=None, file_dir=None, transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        
        def _read_list(list_path):
            list_path = list_path if os.path.isabs(list_path) else os.path.join(base_dir, list_path)
            with open(list_path, 'r') as f:
                return [ln.strip() for ln in f if ln.strip()]
        
        def _resolve_path(p, is_mask=False):
            if os.path.isabs(p):
                return p
            # If path already includes split/images or split/masks, join to base_dir
            if any(seg in p for seg in ["train/", "val/", "test/"]):
                return os.path.join(base_dir, p)
            sub = "masks" if is_mask else "images"
            return os.path.join(base_dir, split, sub, p)
        
        def _derive_mask_from_image(img_path):
            # Replace images->masks and _image->_mask
            mp = img_path.replace("/images/", "/masks/").replace("\\images\\", "\\masks\\")
            mp = mp.replace("_image", "_mask")
            return mp
        
        if image_file is not None:
            img_lines = _read_list(image_file)
            self.image_paths = [_resolve_path(p, is_mask=False) for p in img_lines]
            if mask_file is not None:
                mask_lines = _read_list(mask_file)
                self.mask_paths = [_resolve_path(p, is_mask=True) for p in mask_lines]
                if len(self.image_paths) != len(self.mask_paths):
                    raise ValueError(f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}")
            else:
                self.mask_paths = [_derive_mask_from_image(p) for p in self.image_paths]
        elif file_dir is not None:
            # Legacy: a single file possibly mixing image and mask paths
            lines = _read_list(file_dir)
            img_lines = [ln for ln in lines if ('/images/' in ln or '\\images\\' in ln)]
            mask_lines = [ln for ln in lines if ('/masks/' in ln or '\\masks\\' in ln)]
            if img_lines and mask_lines:
                self.image_paths = [_resolve_path(p, is_mask=False) for p in img_lines]
                self.mask_paths = [_resolve_path(p, is_mask=True) for p in mask_lines]
                if len(self.image_paths) != len(self.mask_paths):
                    raise ValueError(f"Image/mask count mismatch in {file_dir}")
            else:
                # File contains only filenames; derive mask paths from images
                self.image_paths = [_resolve_path(p, is_mask=False) for p in lines]
                self.mask_paths = [_derive_mask_from_image(p) for p in self.image_paths]
        else:
            raise ValueError("Provide either (image_file[, mask_file]) or legacy file_dir.")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Robust thresholding to binary [0,1] mask before any tensor ops
        # Many legacy datasets contain non-binary mask values due to interpolation.
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if label is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        
        # Ensure binary 0/1 mask after augmentations
        import torch
        if isinstance(label, torch.Tensor):
            # ToTensorV2 returns tensor in [0,1] already; binarize robustly
            label = (label > 0.5).to(torch.float32)
            if label.dim() == 2:
                label = label.unsqueeze(0)
        else:
            # Numpy path (unlikely when using ToTensorV2, but safe)
            if label.max() > 1.0:
                label = (label > 127).astype('uint8')
                label = label / 255.0
            import numpy as np
            if label.ndim == 2:
                label = np.expand_dims(label, axis=0)
            label = torch.from_numpy(label).to(torch.float32)

        return {"image": image, "label": label, "idx": idx, "img_path": image_path, "mask_path": mask_path}

