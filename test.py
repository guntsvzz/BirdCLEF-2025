# valid.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import BirdCLEFModel
from utils import BirdCLEFDataset, collate_fn, get_criterion, calculate_auc

def run_test(cfg, checkpoint_path, val_df, args):
    device = cfg.device
    # Load pre-computed spectrograms if desired.
    spectrograms = None
    if cfg.LOAD_DATA:
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Loaded {len(spectrograms)} pre-computed spectrograms")
        except Exception as e:
            print("Error loading spectrograms, switching to on-the-fly processing.", e)
            cfg.LOAD_DATA = False
    dataset = BirdCLEFDataset(val_df, cfg, spectrograms=spectrograms, mode="test")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_fn)
    model = BirdCLEFModel(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = get_criterion(cfg)
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            all_outputs.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    print(f"Test Loss: {np.mean(losses):.4f}, Test AUC: {auc:.4f}")
