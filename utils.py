# utils.py
import os
import cv2
import math
import random
import warnings
import numpy as np
import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def audio2melspec(audio_data, cfg):
    """Convert audio data to a normalized mel spectrogram."""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm

def process_audio_file(audio_path, cfg):
    """Load an audio file and return the corresponding mel spectrogram."""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)
        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            audio_data = np.concatenate([audio_data] * n_copy)
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]
        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, (0, target_samples - len(center_audio)), mode='constant')
        mel_spec = audio2melspec(center_audio, cfg)
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        return mel_spec.astype(np.float32)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

class BirdCLEFDataset(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df.copy()
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms

        # Build label encoding from taxonomy.
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.df.filename.apply(lambda x: os.path.join(cfg.train_datadir, x))
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)
        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":
                print(f"Warning: Could not generate spectrogram for {samplename}")
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # add channel dimension

        # Optionally, you can add augmentations for training here.
        target = np.zeros(self.num_classes)
        if row['primary_label'] in self.label_to_idx:
            target[self.label_to_idx[row['primary_label']]] = 1.0
        return {'melspec': spec, 'target': torch.tensor(target, dtype=torch.float32), 'filename': row['filename']}

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
    result = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    for key in result:
        if key in ['melspec', 'target'] and isinstance(result[key][0], torch.Tensor):
            try:
                result[key] = torch.stack(result[key])
            except Exception:
                pass
    return result

def get_optimizer(model, cfg):
    if cfg.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {cfg.optimizer}")

def get_scheduler(optimizer, cfg):
    if cfg.scheduler == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=cfg.min_lr, verbose=True)
    elif cfg.scheduler == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)
    elif cfg.scheduler == 'OneCycleLR':
        return None
    else:
        return None

def get_criterion(cfg):
    if cfg.criterion == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Unsupported criterion: {cfg.criterion}")

def calculate_auc(targets, outputs):
    from sklearn.metrics import roc_auc_score
    num_classes = targets.shape[1]
    aucs = []
    probs = 1 / (1 + np.exp(-outputs))
    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            aucs.append(roc_auc_score(targets[:, i], probs[:, i]))
    return np.mean(aucs) if aucs else 0.0

def run_preprocessing(df, cfg, save_path):
    """
    Generate and save pre-computed mel spectrograms for the entire dataframe.
    """
    print("Generating pre-computed spectrograms...")
    spectrograms = {}
    for _, row in df.iterrows():
        filepath = os.path.join(cfg.train_datadir, row['filename'])
        spec = process_audio_file(filepath, cfg)
        if spec is not None:
            samplename = row['filename'].split('/')[0] + '-' + row['filename'].split('/')[-1].split('.')[0]
            spectrograms[samplename] = spec
    np.save(save_path, spectrograms)
    print(f"Preprocessing complete. Saved to {save_path}")
