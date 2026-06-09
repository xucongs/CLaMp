
import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix



def cosine_lr(epoch, base_lr, warmup_epochs, total_epochs, min_lr):
    """Linear warmup + cosine annealing。"""
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos


def linear_decay_weight(epoch, total_epochs, w_init, w_final, decay_start_frac, decay_end_frac):
    frac = (epoch - 1) / max(total_epochs - 1, 1)
    if frac <= decay_start_frac:
        return w_init
    elif frac >= decay_end_frac:
        return w_final
    else:
        t = (frac - decay_start_frac) / (decay_end_frac - decay_start_frac)
        return w_init + t * (w_final - w_init)



def info_nce_loss(graph_emb, text_emb, temperature):
    B = graph_emb.size(0)
    logits = (graph_emb @ text_emb.T) / temperature
    labels = torch.arange(B, device=graph_emb.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2.0


def masked_mse_loss(pred, target, mask, labels=None,
                     stable_threshold=0.05, stable_weight=3.0):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    weight = torch.ones_like(pred)
    if labels is not None:
        ehull_mask = mask[:, 2]
        is_stable  = (labels[:, 2] <= stable_threshold) & ehull_mask
        weight[is_stable, 2] = stable_weight
    diff_sq = (pred - target) ** 2
    return (diff_sq * weight)[mask].mean()



def regression_metrics(pred, true, mask):
    valid = mask.astype(bool)
    if valid.sum() < 2:
        return {'mae': float('nan'), 'rmse': float('nan'),
                'r2': float('nan'), 'n': 0}
    p = pred[valid]
    t = true[valid]
    return {
        'mae':  float(mean_absolute_error(t, p)),
        'rmse': float(np.sqrt(np.mean((p - t) ** 2))),
        'r2':   float(r2_score(t, p)),
        'n':    int(valid.sum()),
    }


def stability_classification_metrics(pred_eh, true_eh, mask, threshold=0.05):
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return {}
    p = pred_eh[valid]
    t = true_eh[valid]
    y_t = (t <= threshold).astype(int)
    y_p = (p <= threshold).astype(int)
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    acc  = (tp + tn) / (tp + tn + fp + fn)
    return {
        'threshold':  threshold,
        'accuracy':   float(acc),
        'precision':  float(prec),
        'recall':     float(rec),
        'f1':         float(f1),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'n_total':    int(valid.sum()),
    }


def recall_at_k(graph_embs, text_embs, ks=(1, 5, 10)):
    sim = graph_embs @ text_embs.T
    ranks = np.argsort(-sim, axis=1)
    N = sim.shape[0]
    out = {}
    for k in ks:
        hits = sum(1 for i in range(N) if i in ranks[i, :k])
        out[f'recall@{k}'] = hits / N
    return out



def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    return obj


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(obj), f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Checkpoint
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=False)



def save_norm_stats(mean, std, path):
    save_json({'mean': mean.tolist() if hasattr(mean, 'tolist') else list(mean),
               'std':  std.tolist()  if hasattr(std, 'tolist')  else list(std)},
              path)


def load_norm_stats(path, device='cpu'):
    s = load_json(path)
    return (torch.tensor(s['mean'], dtype=torch.float32, device=device),
            torch.tensor(s['std'],  dtype=torch.float32, device=device))



def print_table(rows, headers, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]
    sep = '  '.join('-' * w for w in col_widths)
    print(sep)
    print('  '.join(f'{h:<{w}}' for h, w in zip(headers, col_widths)))
    print(sep)
    for r in rows:
        print('  '.join(f'{str(c):<{w}}' for c, w in zip(r, col_widths)))
    print(sep)
