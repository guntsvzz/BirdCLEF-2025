# train.py
import os
import gc
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from utils import (
    BirdCLEFDataset, 
    collate_fn, 
    set_seed, 
    get_optimizer, 
    get_scheduler,
    get_criterion, 
    calculate_auc
    )

from model import BirdCLEFModel

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    for batch in tqdm(loader, desc="Training"):
        inputs = batch['melspec'].to(device)
        targets = batch['target'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs, loss = outputs
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler.__class__.__name__ == 'OneCycleLR':
            scheduler.step()
        losses.append(loss.item())
        all_outputs.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    return np.mean(losses), auc

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
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
    return np.mean(losses), auc

def run_train(train_df, cfg):
    # Make sure to update num_classes by loading taxonomy.
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    cfg.num_classes = len(taxonomy_df)
    if cfg.debug:
        cfg.epochs = 2

    # Optionally load pre-computed spectrograms.
    spectrograms = None
    if cfg.LOAD_DATA:
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Loaded {len(spectrograms)} pre-computed spectrograms")
        except Exception as e:
            print("Error loading spectrograms, switching to on-the-fly processing.", e)
            cfg.LOAD_DATA = False

    # Stratified KFold split.
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    best_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue
        print(f"\n{'='*30} Fold {fold} {'='*30}")
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = BirdCLEFDataset(train_fold, cfg, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDataset(val_fold, cfg, spectrograms=spectrograms, mode='valid')
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, collate_fn=collate_fn)
        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        scheduler = None
        if cfg.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.lr, steps_per_epoch=len(train_loader), epochs=cfg.epochs, pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)
        best_auc = 0
        best_epoch = 0
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg.device, 
                scheduler
            )
            val_loss, val_auc = validate(
                model, 
                val_loader, 
                criterion, 
                cfg.device
            )
            
            if scheduler is not None and scheduler.__class__.__name__ != 'OneCycleLR':
                if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg
                }
                save_path = os.path.join(cfg.OUTPUT_DIR, f"model_{cfg.model_name}.pth") #fold{fold}
                torch.save(checkpoint, save_path)
                print(f"Checkpoint saved to {save_path}")
                
        best_scores.append(best_auc)
        
        # Clear memory.
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
        
    print("\nCross-Validation Results:")
    for idx, score in enumerate(best_scores):
        print(f"Fold {cfg.selected_folds[idx]}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")
