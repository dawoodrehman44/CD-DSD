import os
import logging
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_diffusion_transform(image_size, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ============================================================================
# BASE DATASET
# ============================================================================

class BaseXRayDataset(Dataset):

    def __init__(self, label_cols, transform=None):
        self.label_cols = label_cols
        self.transform = transform
        self.dataframe = None
        self.valid_indices = []

    def _validate_dataset(self, path_fn):
        logger.info(f"Validating {self.__class__.__name__}...")
        for idx in range(len(self.dataframe)):
            row = self.dataframe.iloc[idx]
            path = path_fn(row)
            if os.path.exists(path):
                try:
                    with Image.open(path) as img:
                        img.verify()
                    self.valid_indices.append(idx)
                except Exception:
                    pass

        total = len(self.dataframe)
        valid = len(self.valid_indices)
        logger.info(f"  Valid: {valid}/{total} images ({100*valid/total:.1f}%)")

        if valid == 0:
            raise ValueError(
                f"No valid images found for {self.__class__.__name__}. "
                "Check your image root paths in config.py."
            )

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def _get_label(self, row):
        label = row[self.label_cols].values.astype(np.float32)
        label = np.clip(label, 0, 1)
        return torch.tensor(label)

    def _fallback(self):
        h = w = 224
        if self.transform is not None:
            for t in self.transform.transforms:
                if hasattr(t, 'size'):
                    h = w = t.size if isinstance(t.size, int) else t.size[0]
                    break
        return torch.zeros(3, h, w), torch.zeros(len(self.label_cols)), -1

    def __len__(self):
        return len(self.valid_indices)


# ============================================================================
# CHEXPERT DATASET
# ============================================================================

class CheXpertDataset(BaseXRayDataset):

    def __init__(self, csv_path, image_root, label_cols, transform=None):
        super().__init__(label_cols, transform)
        self.image_root = image_root
        self.dataframe = pd.read_csv(csv_path)

        for col in self.label_cols:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].fillna(0).clip(lower=0)
            else:
                self.dataframe[col] = 0.0

        self._validate_dataset(self._path_fn)

    def _path_fn(self, row):
        raw = row.get('Path', row.get('path', ''))
        relative = str(raw).replace("CheXpert-v1.0/train/", "")
        return os.path.join(self.image_root, relative)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.dataframe.iloc[actual_idx]
        path = self._path_fn(row)
        try:
            image = self._load_image(path)
            label = self._get_label(row)
            return image, label, actual_idx
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            return self._fallback()


# ============================================================================
# MIMIC-CXR DATASET
# ============================================================================

class MIMICCXRDataset(BaseXRayDataset):

    def __init__(self, csv_path, image_root, label_cols, transform=None, max_samples=None):
        super().__init__(label_cols, transform)
        self.image_root = image_root
        self.dataframe = pd.read_csv(csv_path)

        if max_samples is not None:
            self.dataframe = self.dataframe.head(max_samples)

        for col in self.label_cols:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].fillna(0).clip(lower=0)
            else:
                self.dataframe[col] = 0.0

        self._validate_dataset(self._path_fn)

    def _path_fn(self, row):
        if 'path' in row:
            return os.path.join(self.image_root, str(row['path']))
        if 'Path' in row:
            return os.path.join(self.image_root, str(row['Path']))
        subj = str(row['subject_id'])
        subject_folder = f"p{subj[:2]}/p{subj}"
        study_folder   = f"s{row['study_id']}"
        dicom_file     = f"{row['dicom_id']}.jpg"
        return os.path.join(self.image_root, subject_folder, study_folder, dicom_file)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.dataframe.iloc[actual_idx]
        path = self._path_fn(row)
        try:
            image = self._load_image(path)
            label = self._get_label(row)
            return image, label, actual_idx
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            return self._fallback()


# ============================================================================
# NIH CHESTX-RAY14 DATASET
# ============================================================================

class NIHChestXrayDataset(BaseXRayDataset):

    NIH_TO_CHEXPERT = {
        'Atelectasis':  'Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Consolidation':'Consolidation',
        'Edema':        'Edema',
        'Effusion':     'Pleural Effusion',
        'Pneumonia':    'Pneumonia',
        'Pneumothorax': 'Pneumothorax',
    }

    def __init__(self, csv_path, image_root, label_cols, transform=None, max_samples=None):
        super().__init__(label_cols, transform)
        self.image_root = image_root
        self.dataframe = pd.read_csv(csv_path)

        if max_samples is not None:
            self.dataframe = self.dataframe.head(max_samples)

        self._remap_labels()
        self._validate_dataset(self._path_fn)

    def _remap_labels(self):
        for col in self.label_cols:
            self.dataframe[col] = 0.0

        for idx, row in self.dataframe.iterrows():
            findings = str(row.get('Finding Labels', '')).split('|')
            findings = [f.strip() for f in findings]

            if 'No Finding' in findings:
                self.dataframe.at[idx, 'No Finding'] = 1.0
            else:
                for nih_label, chex_label in self.NIH_TO_CHEXPERT.items():
                    if nih_label in findings and chex_label in self.label_cols:
                        self.dataframe.at[idx, chex_label] = 1.0

    def _path_fn(self, row):
        return os.path.join(self.image_root, str(row['Image Index']))

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.dataframe.iloc[actual_idx]
        path = self._path_fn(row)
        try:
            image = self._load_image(path)
            label = self._get_label(row)
            return image, label, actual_idx
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            return self._fallback()


# ============================================================================
# COLLATE + DATALOADER
# ============================================================================

def collate_fn(batch):
    valid = [item for item in batch if item is not None and item[2] != -1]
    if len(valid) == 0:
        n = len(batch)
        return torch.zeros(n, 3, 224, 224), torch.zeros(n, 8), [-1] * n

    images, labels, indices = zip(*valid)
    return (
        torch.stack([torch.as_tensor(img) for img in images]),
        torch.stack([torch.as_tensor(lbl) for lbl in labels]),
        list(indices)
    )


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
