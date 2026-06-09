
import os
import json
import argparse
import warnings
import tempfile

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import CFG

warnings.filterwarnings("ignore")


def import_build_graphs():
    try:
        import build_graphs
        print('Done.')
        print(f"{len(build_graphs.ELEMENT_CACHE)}")
        print(f"   sample Z=3 (Li): {build_graphs.ELEMENT_CACHE[3]}")
        return build_graphs.ELEMENT_CACHE, build_graphs.cif_to_graph
    except ImportError as e:
        print(f"{e}")
        print('Done.')
        raise


def atoms_dict_to_cif(atoms_dict, output_path):
    from pymatgen.core import Structure, Lattice
    structure = Structure(
        lattice=Lattice(np.asarray(atoms_dict['lattice_mat'])),
        species=atoms_dict['elements'],
        coords=np.asarray(atoms_dict['coords']),
        coords_are_cartesian=True,
    )
    structure.to(filename=output_path)


def verify_against_train(cif_to_graph_fn):
    print("\n" + "=" * 65)
    print('Done.')
    print("=" * 65)

    train_meta = os.path.join(CFG.SPLIT_DIR, 'train', 'train_metadata.jsonl')
    with open(train_meta) as f:
        first = json.loads(f.readline())
    jid = first['jid']

    
    graph_path = os.path.join(first['graph_dir'], first['pt_file'])
    g_saved = torch.load(graph_path, weights_only=False)
    print(f"{graph_path}")
    print(f"     n_atoms: {g_saved.x.shape[0]}")
    print(f"     n_edges: {g_saved.edge_index.shape[1]}")

    
    cif_path = os.path.join('cif_files_jarvis', f"{jid}.cif")
    if not os.path.exists(cif_path):
        print(f"{cif_path}")
        print('Done.')
        return True

    g_new = cif_to_graph_fn(jid, cif_path)
    print('Done.')
    print(f"     n_atoms: {g_new.x.shape[0]}")
    print(f"     n_edges: {g_new.edge_index.shape[1]}")

    if g_saved.x.shape != g_new.x.shape:
        print('Done.')
        return False

    diff = (g_saved.x - g_new.x).abs().max().item()
    print(f"   x[0] diff:    {(g_saved.x[0] - g_new.x[0]).abs().max():.6f}")
    print(f"{diff:.6f}")

    if diff < 1e-4:
        print('Done.')
        return True
    else:
        print('Done.')
        return False


def load_jarvis_2d():
    from jarvis.db.figshare import data
    print('Done.')
    jdft_2d = data('dft_2d')
    out = []
    for d in jdft_2d:
        if 'atoms' not in d:
            continue
        out.append({
            'jid':       d.get('jid', 'unknown'),
            'formula':   d.get('formula', d.get('reduced_formula', '')),
            'atoms':     d['atoms'],
            'true_props': {
                'bandgap':          d.get('optb88vdw_bandgap'),
                'formation_energy': d.get('formation_energy_peratom'),
                'ehull':            d.get('ehull'),
            },
        })
    return out


def encode_with(graphs):
    from torch_geometric.data import Batch
    from utils import load_norm_stats, load_checkpoint
    from models import build_model

    print('Done.')
    model = build_model('multimodal', CFG).to(CFG.DEVICE)
    ckpt_path = os.path.join(CFG.CKPT_MULTIMODAL, 'best_model.pt')
    ckpt = load_checkpoint(ckpt_path, device=CFG.DEVICE)
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state); model.eval()

    global_mean, global_std = load_norm_stats(
        os.path.join(CFG.CKPT_MULTIMODAL, 'norm_stats.json'),
        device=CFG.DEVICE,
    )

    embs, preds = [], []
    bs = 32
    first_done = False
    print(f"{len(graphs)}")
    with torch.no_grad():
        for s in tqdm(range(0, len(graphs), bs), desc='Processing'):
            batch_g = graphs[s:s + bs]
            try:
                batch = Batch.from_data_list(batch_g).to(CFG.DEVICE)
                B = len(batch_g)
                dummy_text = torch.zeros(B, CFG.LLM_DIM,
                                          dtype=torch.float32,
                                          device=CFG.DEVICE)
                g_norm, _, pred_norm = model(batch, dummy_text)
                pred_real = pred_norm * global_std + global_mean
                embs.append(g_norm.cpu().numpy())
                preds.append(pred_real.cpu().numpy())
                if not first_done:
                    print('Done.')
                    print(f"      embedding: {g_norm.shape}")
                    print(f"      sample 0: bg={pred_real[0,0].item():.3f}, "
                           f"fe={pred_real[0,1].item():.3f}, "
                           f"eh={pred_real[0,2].item():.3f}")
                    first_done = True
            except Exception as e:
                if not first_done:
                    print(f"Processing{type(e).__name__}: {e}")
                    raise
                bad = len(batch_g)
                embs.append(np.zeros((bad, CFG.EMBED_DIM), dtype=np.float32))
                preds.append(np.full((bad, 3), np.nan, dtype=np.float32))
    return np.concatenate(embs), np.concatenate(preds)


def load_train_jids():
    seen = set()
    for split in ['train', 'val']:
        path = os.path.join(CFG.SPLIT_DIR, split, f"{split}_metadata.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(json.loads(line)['jid'])
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',  default='2d_candidates')
    parser.add_argument('--max_n',    type=int, default=None)
    parser.add_argument('--verify_only', action='store_true')
    args = parser.parse_args()

    
    print("=" * 65)
    print("  Step 1: import build_graphs.py")
    print("=" * 65)
    ELEMENT_CACHE, cif_to_graph = import_build_graphs()

    
    is_consistent = verify_against_train(cif_to_graph)
    if not is_consistent:
        print('Done.')

    if args.verify_only:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'graphs'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'atoms'),  exist_ok=True)
    cif_tmp_dir = os.path.join(args.out_dir, 'cif_tmp')
    os.makedirs(cif_tmp_dir, exist_ok=True)

    candidates = load_jarvis_2d()
    if args.max_n:
        candidates = candidates[:args.max_n]
    print(f"{len(candidates)}")
    seen = load_train_jids()
    print(f"{len(seen)}")

    
    print('Done.')
    graphs, kept_meta = [], []
    fail_reasons = {}
    for c in tqdm(candidates, desc='Processing'):
        try:
            
            cif_path = os.path.join(cif_tmp_dir, f"{c['jid']}.cif")
            from pymatgen.core import Structure, Lattice
            structure = Structure(
                lattice=Lattice(np.asarray(c['atoms']['lattice_mat'])),
                species=c['atoms']['elements'],
                coords=np.asarray(c['atoms']['coords']),
                coords_are_cartesian=True,
            )
            structure.to(filename=cif_path)

            
            g = cif_to_graph(c['jid'], cif_path)
            graphs.append(g)

            
            atoms_path = os.path.join(args.out_dir, 'atoms', f"{c['jid']}.json")
            with open(atoms_path, 'w') as f:
                json.dump({
                    'lattice_mat': np.asarray(c['atoms']['lattice_mat']).tolist(),
                    'coords':      np.asarray(c['atoms']['coords']).tolist(),
                    'elements':    list(c['atoms']['elements']),
                }, f)

            kept_meta.append({
                'jid':       c['jid'],
                'formula':   c['formula'],
                'source':    'jarvis-2d',
                'in_train':  c['jid'] in seen,
                'true_bandgap':          c['true_props']['bandgap'],
                'true_formation_energy': c['true_props']['formation_energy'],
                'true_ehull':            c['true_props']['ehull'],
            })
        except Exception as e:
            et = type(e).__name__
            fail_reasons[et] = fail_reasons.get(et, '') + ' / ' + str(e)[:60]

    print(f"{len(graphs)}")
    if fail_reasons:
        for et, msg in fail_reasons.items():
            print(f"Processing{et}: {msg[:120]}")

    
    import shutil
    try:
        shutil.rmtree(cif_tmp_dir)
    except Exception:
        pass

    if not graphs:
        return

    print('Done.')
    for g, m in tqdm(zip(graphs, kept_meta), total=len(graphs)):
        torch.save(g, os.path.join(args.out_dir, 'graphs', f"{m['jid']}.pt"))

    embs, preds = encode_with(graphs)

    print('Done.')
    df_chk = pd.DataFrame({
        'jid':     [m['jid']     for m in kept_meta],
        'formula': [m['formula'] for m in kept_meta],
        'pred_bg': preds[:, 0],
        'true_bg': [m['true_bandgap']  for m in kept_meta],
        'pred_eh': preds[:, 2],
        'true_eh': [m['true_ehull']    for m in kept_meta],
    }).dropna()

    if len(df_chk) > 10:
        bg_mae = (df_chk['pred_bg'] - df_chk['true_bg']).abs().mean()
        eh_mae = (df_chk['pred_eh'] - df_chk['true_eh']).abs().mean()
        bg_r2 = np.corrcoef(df_chk['pred_bg'], df_chk['true_bg'])[0,1]**2

        print(f"   bandgap MAE: {bg_mae:.3f}Processing")
        print(f"   bandgap R²:  {bg_r2:.3f}Processing")
        print(f"   ehull MAE:   {eh_mae:.3f} eV/atom")

        
        print('Done.')
        for lo, hi in [(0, 0.1), (0.1, 1), (1, 2), (2, 4), (4, 10)]:
            sub = df_chk[(df_chk['true_bg'] >= lo) & (df_chk['true_bg'] < hi)]
            if len(sub) > 0:
                mae = (sub['pred_bg'] - sub['true_bg']).abs().mean()
                pred_mean = sub['pred_bg'].mean()
                true_mean = sub['true_bg'].mean()
                print(f"     true∈[{lo},{hi}) eV (n={len(sub):>3}): "
                       f"MAE={mae:.3f}, pred={pred_mean:.2f}, true={true_mean:.2f}")

    for i, m in enumerate(kept_meta):
        m['pred_bandgap']         = (float(preds[i,0]) if not np.isnan(preds[i,0]) else None)
        m['pred_formation_energy']= (float(preds[i,1]) if not np.isnan(preds[i,1]) else None)
        m['pred_ehull']           = (float(preds[i,2]) if not np.isnan(preds[i,2]) else None)

    df = pd.DataFrame(kept_meta)
    df.to_csv(os.path.join(args.out_dir, 'metadata.csv'), index=False)
    np.savez(os.path.join(args.out_dir, 'embeddings.npz'),
             embeddings=embs, jids=np.array([m['jid'] for m in kept_meta]))

    print('Done.')
    print(f"   {args.out_dir}/metadata.csv: {len(df)}Processing")


if __name__ == "__main__":
    main()
