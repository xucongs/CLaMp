
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from utils import (
    load_norm_stats, load_checkpoint, save_json,
    regression_metrics, stability_classification_metrics, recall_at_k,
)
from models import build_model
from dataset import MaterialDataset, collate_fn



@torch.no_grad()
def run_inference(model, model_type, loader, global_mean, global_std):
    all_pred, all_true, all_mask = [], [], []
    all_g, all_t = [], []

    for batch in tqdm(loader, desc='Inference'):
        bg = batch['graph'].to(CFG.DEVICE)
        lb = batch['labels']
        mk = batch['label_mask']

        if model_type == 'graph_only':
            pred = model(bg)
            pred_real = pred * global_std + global_mean
        else:
            te = batch['text_emb'].to(CFG.DEVICE)
            g_norm, t_norm, pred = model(bg, te)
            pred_real = pred * global_std + global_mean
            all_g.append(g_norm.cpu().numpy())
            all_t.append(t_norm.cpu().numpy())

        all_pred.append(pred_real.cpu().numpy())
        all_true.append(lb.numpy())
        all_mask.append(mk.numpy())

    pred_arr = np.concatenate(all_pred)
    true_arr = np.concatenate(all_true)
    mask_arr = np.concatenate(all_mask)
    graph_embs = np.concatenate(all_g) if all_g else None
    text_embs  = np.concatenate(all_t) if all_t else None
    return pred_arr, true_arr, mask_arr, graph_embs, text_embs



def rag_predict(query_graph_embs, kb_text_embs, kb_labels, kb_masks,
                top_k=10, temperature=0.05):
    N = query_graph_embs.shape[0]
    sim = query_graph_embs @ kb_text_embs.T

    rag_pred = np.zeros((N, 3), dtype=np.float32)
    rag_valid = np.zeros((N, 3), dtype=bool)

    for i in range(N):
        top_idx = np.argsort(-sim[i])[:top_k]
        weights = np.exp(sim[i][top_idx] / temperature)
        weights = weights / (weights.sum() + 1e-8)

        for prop in range(3):
            valid = kb_masks[top_idx, prop].astype(bool)
            if valid.sum() == 0:
                continue
            w = weights[valid]; w = w / (w.sum() + 1e-8)
            rag_pred[i, prop]  = np.sum(kb_labels[top_idx[valid], prop] * w)
            rag_valid[i, prop] = True

    return rag_pred, rag_valid


@torch.no_grad()
def build_kb(model, train_loader, global_mean, global_std):
    kb_t, kb_lb, kb_mk = [], [], []
    for batch in tqdm(train_loader, desc='Inference'):
        bg = batch['graph'].to(CFG.DEVICE)
        te = batch['text_emb'].to(CFG.DEVICE)
        _, t_norm, _ = model(bg, te)
        kb_t.append(t_norm.cpu().numpy())
        kb_lb.append(batch['labels'].numpy())
        kb_mk.append(batch['label_mask'].numpy())
    return (np.concatenate(kb_t),
            np.concatenate(kb_lb),
            np.concatenate(kb_mk))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True,
                        choices=['graph_only', 'multimodal'])
    parser.add_argument('--ckpt_dir', default=None)
    parser.add_argument('--ckpt_name', default='best_model.pt')
    parser.add_argument('--max_samples', type=int, default=CFG.MAX_TEST_SAMPLES)
    parser.add_argument('--tag', default='best',
                        help='See README.')
    parser.add_argument('--skip_rag', action='store_true',
                        help='See README.')
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir or (
        CFG.CKPT_GRAPH_ONLY if args.model_type == 'graph_only'
        else CFG.CKPT_MULTIMODAL
    )
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
    stats_path = os.path.join(ckpt_dir, 'norm_stats.json')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{stats_path}")

    print("=" * 65)
    print(f"Evaluating model: {args.model_type}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Tag:        {args.tag}")
    print("=" * 65)

    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

    
    global_mean, global_std = load_norm_stats(stats_path, device=CFG.DEVICE)

    model = build_model(args.model_type, CFG).to(CFG.DEVICE)
    ckpt = load_checkpoint(ckpt_path, device=CFG.DEVICE)
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    if isinstance(ckpt, dict) and 'epoch' in ckpt:
        bg_mae = ckpt.get('val_mae', {}).get('bandgap', 'N/A')
        print(f"Processing{ckpt['epoch']}, val_bg_mae={bg_mae}")

    
    test_path = CFG.TEST_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)
    test_ds = MaterialDataset(test_path, max_samples=args.max_samples)
    test_loader = DataLoader(
        test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=CFG.NUM_WORKERS,
    )

    
    pred_real, true_labels, masks, graph_embs, text_embs = run_inference(
        model, args.model_type, test_loader, global_mean, global_std,
    )

    results = {
        'model_type':     args.model_type,
        'checkpoint':     ckpt_path,
        'n_test_samples': int(len(pred_real)),
        'tag':            args.tag,
    }

    
    print('Done.')
    print(f"  {'Processing':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'N':>8}")
    print(f"  {'-'*56}")
    reg = {}
    for i, (name, unit) in enumerate(zip(CFG.PROP_NAMES, CFG.PROP_UNITS)):
        m = regression_metrics(pred_real[:, i], true_labels[:, i], masks[:, i])
        reg[name] = m
        print(f"  {name:<22} {m['mae']:>7.4f}  {m['rmse']:>7.4f}  "
              f"{m['r2']:>7.4f}  {m['n']:>7}  ({unit})")
    results['regression'] = reg

    
    print(f"Processing{CFG.EHULL_STABLE_THRESHOLD})")
    eh_idx = CFG.PROP_NAMES.index('ehull')
    stab = stability_classification_metrics(
        pred_real[:, eh_idx], true_labels[:, eh_idx], masks[:, eh_idx],
        threshold=CFG.EHULL_STABLE_THRESHOLD,
    )
    results['stability'] = stab
    if stab:
        print(f"  Accuracy : {stab['accuracy']:.4f}")
        print(f"  Precision: {stab['precision']:.4f}")
        print(f"  Recall   : {stab['recall']:.4f}")
        print(f"  F1 Score : {stab['f1']:.4f}")
        print(f"  CM: TP={stab['tp']} FP={stab['fp']} TN={stab['tn']} FN={stab['fn']}")

    
    if args.model_type == 'multimodal':
        print(f"Processing{len(pred_real)})")
        recall = recall_at_k(graph_embs, text_embs, ks=[1, 5, 10])
        results['retrieval'] = recall
        for k, v in recall.items():
            print(f"  {k:<12}: {v*100:.2f}%")

        if not args.skip_rag:
            print(f"Processing{CFG.TOP_K_RAG})")
            train_path = CFG.TRAIN_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)
            train_ds = MaterialDataset(train_path)
            train_loader = DataLoader(
                train_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=CFG.NUM_WORKERS,
            )
            kb_t, kb_lb, kb_mk = build_kb(model, train_loader,
                                           global_mean, global_std)
            rag_pred, rag_valid = rag_predict(
                graph_embs, kb_t, kb_lb, kb_mk,
                top_k=CFG.TOP_K_RAG, temperature=CFG.TEMPERATURE_RAG,
            )
            print(f"  {'Processing':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'N':>8}")
            print(f"  {'-'*56}")
            rag_results = {}
            for i, (name, unit) in enumerate(zip(CFG.PROP_NAMES, CFG.PROP_UNITS)):
                vm = masks[:, i].astype(bool) & rag_valid[:, i]
                m = regression_metrics(rag_pred[:, i], true_labels[:, i], vm)
                rag_results[name] = m
                print(f"  {name:<22} {m['mae']:>7.4f}  {m['rmse']:>7.4f}  "
                      f"{m['r2']:>7.4f}  {m['n']:>7}  ({unit})")
            results['rag'] = rag_results

    
    out_json = os.path.join(CFG.RESULTS_DIR,
                             f"eval_{args.model_type}_{args.tag}.json")
    save_json(results, out_json)
    print(f"Saved metrics: {out_json}")

    np.savez(
        os.path.join(CFG.RESULTS_DIR,
                     f"predictions_{args.model_type}_{args.tag}.npz"),
        pred=pred_real, true=true_labels, mask=masks,
    )
    if graph_embs is not None:
        np.savez(
            os.path.join(CFG.RESULTS_DIR,
                         f"embeddings_{args.model_type}_{args.tag}.npz"),
            graph=graph_embs, text=text_embs,
        )

    print('Done.')


if __name__ == "__main__":
    main()
