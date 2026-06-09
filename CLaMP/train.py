
import os
import time
import argparse
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import CFG
from utils import (
    cosine_lr, linear_decay_weight, info_nce_loss, masked_mse_loss,
    save_norm_stats, save_checkpoint,
)
from models import build_model
from dataset import MaterialDataset, collate_fn, compute_label_stats



def train_one_epoch(model, model_type, loader, optimizer,
                    global_mean, global_std, epoch, lambda_nce):
    model.train()
    metrics = {'loss': 0., 'nce': 0., 'mse': 0.}
    n = 0

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False)
    for batch in pbar:
        bg = batch['graph'].to(CFG.DEVICE)
        lb = batch['labels'].to(CFG.DEVICE)
        mk = batch['label_mask'].to(CFG.DEVICE)
        labels_norm = (lb - global_mean) / global_std

        optimizer.zero_grad()

        if model_type == 'graph_only':
            pred = model(bg)
            loss_mse = masked_mse_loss(
                pred, labels_norm, mk, labels=lb,
                stable_threshold=CFG.EHULL_STABLE_THRESHOLD,
                stable_weight=CFG.EHULL_STABLE_WEIGHT,
            )
            loss_nce = torch.tensor(0.0, device=CFG.DEVICE)
            loss = loss_mse
        else:  # multimodal
            te = batch['text_emb'].to(CFG.DEVICE)
            g_norm, t_norm, pred = model(bg, te)
            loss_nce = info_nce_loss(g_norm, t_norm, CFG.TEMPERATURE)
            loss_mse = masked_mse_loss(
                pred, labels_norm, mk, labels=lb,
                stable_threshold=CFG.EHULL_STABLE_THRESHOLD,
                stable_weight=CFG.EHULL_STABLE_WEIGHT,
            )
            loss = lambda_nce * loss_nce + CFG.LAMBDA_REG * loss_mse

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        optimizer.step()

        metrics['loss'] += loss.item()
        metrics['nce']  += loss_nce.item()
        metrics['mse']  += loss_mse.item()
        n += 1
        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'mse':  f"{loss_mse.item():.4f}",
        })

    return {k: v / max(n, 1) for k, v in metrics.items()}


@torch.no_grad()
def validate(model, model_type, loader, global_mean, global_std, lambda_nce):
    model.eval()
    metrics = {'loss': 0., 'nce': 0., 'mse': 0.}
    n = 0
    all_pred = [[] for _ in range(3)]
    all_true = [[] for _ in range(3)]

    for batch in loader:
        bg = batch['graph'].to(CFG.DEVICE)
        lb = batch['labels'].to(CFG.DEVICE)
        mk = batch['label_mask'].to(CFG.DEVICE)
        labels_norm = (lb - global_mean) / global_std

        if model_type == 'graph_only':
            pred = model(bg)
            loss_mse = masked_mse_loss(pred, labels_norm, mk)
            loss_nce = torch.tensor(0.0, device=CFG.DEVICE)
            loss = loss_mse
        else:
            te = batch['text_emb'].to(CFG.DEVICE)
            g_norm, t_norm, pred = model(bg, te)
            loss_nce = info_nce_loss(g_norm, t_norm, CFG.TEMPERATURE)
            loss_mse = masked_mse_loss(pred, labels_norm, mk)
            loss = lambda_nce * loss_nce + CFG.LAMBDA_REG * loss_mse

        metrics['loss'] += loss.item()
        metrics['nce']  += loss_nce.item()
        metrics['mse']  += loss_mse.item()
        n += 1

        pred_real = pred * global_std + global_mean
        for i in range(3):
            mi = mk[:, i]
            if mi.sum() > 0:
                all_pred[i].extend(pred_real[mi, i].cpu().numpy())
                all_true[i].extend(lb[mi, i].cpu().numpy())

    out = {k: v / max(n, 1) for k, v in metrics.items()}
    out['mae'] = {}
    for i, name in enumerate(CFG.PROP_NAMES):
        if all_true[i]:
            out['mae'][name] = float(np.mean(
                np.abs(np.array(all_pred[i]) - np.array(all_true[i]))
            ))
        else:
            out['mae'][name] = float('nan')
    return out



def plot_curves(log_df, model_type, save_path):
    if model_type == 'graph_only':
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(log_df['epoch'], log_df['train_mse'], label='Train')
        axes[0].plot(log_df['epoch'], log_df['val_mse'],   label='Val')
        axes[0].set_title('MSE'); axes[0].legend(); axes[0].set_xlabel('Epoch')

        for col, lbl in [('val_mae_bandgap', 'Bandgap'),
                         ('val_mae_formation_e', 'Formation E'),
                         ('val_mae_ehull', 'Ehull')]:
            axes[1].plot(log_df['epoch'], log_df[col], label=lbl)
        axes[1].set_title('Val MAE'); axes[1].legend(); axes[1].set_xlabel('Epoch')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes[0,0].plot(log_df['epoch'], log_df['train_loss'], label='Train')
        axes[0,0].plot(log_df['epoch'], log_df['val_loss'],   label='Val')
        axes[0,0].set_title('Total Loss'); axes[0,0].legend()

        axes[0,1].plot(log_df['epoch'], log_df['train_nce'], label='Train NCE')
        axes[0,1].plot(log_df['epoch'], log_df['val_nce'],   label='Val NCE')
        axes[0,1].set_title('InfoNCE'); axes[0,1].legend()

        axes[1,0].plot(log_df['epoch'], log_df['train_mse'], label='Train MSE')
        axes[1,0].plot(log_df['epoch'], log_df['val_mse'],   label='Val MSE')
        axes[1,0].set_title('MSE'); axes[1,0].legend()

        for col, lbl in [('val_mae_bandgap', 'Bandgap'),
                         ('val_mae_formation_e', 'Formation E'),
                         ('val_mae_ehull', 'Ehull')]:
            axes[1,1].plot(log_df['epoch'], log_df[col], label=lbl)
        axes[1,1].set_title('Val MAE'); axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True,
                        choices=['graph_only', 'multimodal'])
    parser.add_argument('--epochs', type=int, default=None,
                        help='See README.')
    parser.add_argument('--ckpt_dir', default=None,
                        help='See README.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    ckpt_dir = args.ckpt_dir or (
        CFG.CKPT_GRAPH_ONLY if args.model_type == 'graph_only'
        else CFG.CKPT_MULTIMODAL
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    epochs = args.epochs or CFG.EPOCHS

    print("=" * 65)
    print(f"Training model: {args.model_type}")
    print(f"  Device:  {CFG.DEVICE}")
    print(f"  Epochs:  {epochs}")
    print(f"  Output:  {ckpt_dir}")
    print(f"  Hidden={CFG.HIDDEN_DIM}, Embed={CFG.EMBED_DIM}, "
          f"Layers={CFG.N_LAYERS}, Dropout={CFG.DROPOUT}")
    print(f"  LR={CFG.LEARNING_RATE} → {CFG.MIN_LR}, "
          f"WD={CFG.WEIGHT_DECAY}")
    if args.model_type == 'multimodal':
        print(f"  λ_nce: {CFG.LAMBDA_NCE_INIT} → {CFG.LAMBDA_NCE_FINAL} "
              f"(decay {CFG.NCE_DECAY_START*100:.0f}%-"
              f"{CFG.NCE_DECAY_END*100:.0f}%)")
    print("=" * 65)

    
    train_path = CFG.TRAIN_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)
    val_path   = CFG.VAL_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)
    train_ds = MaterialDataset(train_path)
    val_ds   = MaterialDataset(val_path)

    mean_np, std_np = compute_label_stats(train_ds)
    save_norm_stats(mean_np, std_np, os.path.join(ckpt_dir, 'norm_stats.json'))

    global_mean = torch.tensor(mean_np, dtype=torch.float32, device=CFG.DEVICE)
    global_std  = torch.tensor(std_np,  dtype=torch.float32, device=CFG.DEVICE)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
    )

    
    model = build_model(args.model_type, CFG).to(CFG.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Processing{n_params/1e6:.2f} M\n")

    optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE,
                             weight_decay=CFG.WEIGHT_DECAY)

    best_val_bg = float('inf')
    log_records = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = cosine_lr(epoch, CFG.LEARNING_RATE,
                            CFG.WARMUP_EPOCHS, epochs, CFG.MIN_LR)
        for g in optimizer.param_groups:
            g['lr'] = lr_now

        if args.model_type == 'multimodal':
            lambda_nce = linear_decay_weight(
                epoch, epochs,
                CFG.LAMBDA_NCE_INIT, CFG.LAMBDA_NCE_FINAL,
                CFG.NCE_DECAY_START, CFG.NCE_DECAY_END,
            )
        else:
            lambda_nce = 0.0

        tr = train_one_epoch(
            model, args.model_type, train_loader, optimizer,
            global_mean, global_std, epoch, lambda_nce,
        )
        va = validate(
            model, args.model_type, val_loader,
            global_mean, global_std, lambda_nce,
        )
        elapsed = time.time() - t0

        
        rec = {
            'epoch':    epoch,
            'lr':       lr_now,
            'train_loss': tr['loss'],
            'train_nce':  tr['nce'],
            'train_mse':  tr['mse'],
            'val_loss':   va['loss'],
            'val_nce':    va['nce'],
            'val_mse':    va['mse'],
            'val_mae_bandgap':     va['mae']['bandgap'],
            'val_mae_formation_e': va['mae']['formation_energy'],
            'val_mae_ehull':       va['mae']['ehull'],
            'time_s':     elapsed,
        }
        if args.model_type == 'multimodal':
            rec['lambda_nce'] = lambda_nce
        log_records.append(rec)

        
        msg = (
            f"E{epoch:>3}/{epochs} | "
            f"Tr {tr['loss']:.4f} (mse={tr['mse']:.4f}) | "
            f"Va {va['loss']:.4f} (mse={va['mse']:.4f}) | "
            f"MAE Bg={va['mae']['bandgap']:.4f} "
            f"FE={va['mae']['formation_energy']:.4f} "
            f"Eh={va['mae']['ehull']:.4f} | "
            f"lr={lr_now:.2e}"
        )
        if args.model_type == 'multimodal':
            msg += f" | λnce={lambda_nce:.2f}"
        msg += f" | {elapsed:.1f}s"
        print(msg)

        
        if va['mae']['bandgap'] < best_val_bg:
            best_val_bg = va['mae']['bandgap']
            save_checkpoint({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_mae':     va['mae'],
                'val_loss':    va['loss'],
                'model_type':  args.model_type,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"Best validation bandgap MAE: {best_val_bg:.4f}")

    
    save_checkpoint({
        'epoch':       epochs,
        'model_state': model.state_dict(),
        'model_type':  args.model_type,
    }, os.path.join(ckpt_dir, 'last_model.pt'))

    
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(os.path.join(ckpt_dir, 'training_log.csv'), index=False)
    plot_curves(log_df, args.model_type,
                os.path.join(ckpt_dir, 'loss_curves.png'))

    print('Done.')
    print(f"Best validation bandgap MAE: {best_val_bg:.4f}")
    print(f"Training model: {args.model_type}")


if __name__ == "__main__":
    main()
