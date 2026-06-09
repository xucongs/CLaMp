
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

from config import CFG
from utils import load_norm_stats, load_checkpoint
from models import build_model


APPLICATION_TEMPLATES = {
    'heterojunction': {
        'queries': [
             "Two-dimensional semiconductor materials as components of type-II heterojunctions suitable for solar cells",

        ],
        'filter': {'true_bandgap': [1.0, 3.5]},
    },
    'solar': {
        'queries': [
            "a 2D semiconductor with bandgap optimal for thin-film photovoltaic absorber",
            "a 2D direct-gap semiconductor with bandgap near 1.4 eV for single-junction solar cells",
        ],
        'filter': {'true_bandgap': [0.9, 1.8]},
    },
    'thermo': {
        'queries': [
            "a 2D narrow-gap thermoelectric semiconductor containing heavy elements",
            "a 2D semiconductor with bandgap below 0.5 eV suitable for thermoelectric applications",
        ],
        'filter': {'true_bandgap': [0.05, 0.8]},
    },
    'topo': {
        'queries': [
            "a 2D topological insulator candidate with narrow gap and strong spin-orbit coupling",
            "a 2D narrow-gap semiconductor containing Bi, Sb, or other heavy elements",
        ],
        'filter': {'true_bandgap': [0.0, 0.5]},
    },
    'photocatalysis': {
        'queries': [
            "a 2D photocatalyst for visible-light water splitting",
            "a 2D semiconductor with bandgap between 1.5 and 3 eV for photocatalytic CO2 reduction",
            "a stable 2D oxide or chalcogenide photocatalyst",
        ],
        'filter': {'true_bandgap': [1.5, 3.0]},
    },
    'tco': {
        'queries': [
            "a 2D wide-gap transparent conductor",
            "a 2D wide-gap semiconductor with bandgap above 3 eV for transparent electronics",
        ],
        'filter': {'true_bandgap': [3.0, 6.0]},
    },
}


def load_text_encoder():
    from sentence_transformers import SentenceTransformer
    print('Done.')
    model = build_model('multimodal', CFG).to(CFG.DEVICE)
    ckpt = load_checkpoint(
        os.path.join(CFG.CKPT_MULTIMODAL, 'best_model.pt'),
        device=CFG.DEVICE,
    )
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state); model.eval()
    print('Done.')
    qwen = SentenceTransformer('Qwen/Qwen3-Embedding-8B', device=str(CFG.DEVICE),local_files_only=True)
    return model, qwen


def encode_queries(model, qwen, queries):
    raw = qwen.encode(queries, convert_to_tensor=True,
                       normalize_embeddings=False).float().to(CFG.DEVICE)
    with torch.no_grad():
        proj = model.encode_text(raw).cpu().numpy()
    return proj


def apply_filter(df, filter_cfg):
    out = df.copy()
    for col, range_ in filter_cfg.items():
        if col not in out.columns:
            continue
        lo, hi = range_
        if lo is not None:
            out = out[out[col] >= lo]
        if hi is not None:
            out = out[out[col] <= hi]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates_dir', default='2d_candidates')
    parser.add_argument('--application', default='heterojunction',
                        choices=list(APPLICATION_TEMPLATES.keys()))
    parser.add_argument('--queries', default=None)
    parser.add_argument('--top_k',  type=int, default=50)
    parser.add_argument('--n_dft',  type=int, default=30)
    parser.add_argument('--exclude_train', action='store_true', default=False)
    parser.add_argument('--tag', default=None,
                        help='See README.')
    parser.add_argument('--overwrite', action='store_true',
                        help='See README.')
    args = parser.parse_args()

    meta = pd.read_csv(os.path.join(args.candidates_dir, 'metadata.csv'))
    embs_data = np.load(os.path.join(args.candidates_dir, 'embeddings.npz'),
                         allow_pickle=True)
    embs = embs_data['embeddings']
    jids = list(embs_data['jids'])

    if args.exclude_train:
        n_before = len(meta)
        keep_idx = [i for i, m in enumerate(meta['in_train']) if not m]
        meta = meta.iloc[keep_idx].reset_index(drop=True)
        embs = embs[keep_idx]
        jids = [jids[i] for i in keep_idx]
        print(f"Processing{len(meta)} (was {n_before})")

    if args.queries:
        with open(args.queries) as f:
            queries_obj = [json.loads(line.strip()) for line in f if line.strip()]
        query_texts = [q['query'] for q in queries_obj]
        filter_cfg = {}
    else:
        tpl = APPLICATION_TEMPLATES[args.application]
        query_texts = tpl['queries']
        filter_cfg = tpl['filter']
    print(f"{len(query_texts)}")
    print(f"{filter_cfg}")

    model, qwen = load_text_encoder()
    query_proj = encode_queries(model, qwen, query_texts)

    print(f"Processing{args.top_k}...")
    sim = query_proj @ embs.T
    top_idx_all = np.argsort(-sim, axis=1)[:, :args.top_k]

    rows = []
    for q_i, q_text in enumerate(query_texts):
        for rank, cand_i in enumerate(top_idx_all[q_i]):
            row = meta.iloc[cand_i].to_dict()
            row['query']      = q_text
            row['query_id']   = q_i
            row['rank']       = rank + 1
            row['similarity'] = float(sim[q_i, cand_i])
            rows.append(row)
    df_all = pd.DataFrame(rows)

    if filter_cfg:
        n_before = len(df_all)
        df_filtered = apply_filter(df_all, filter_cfg)
        print(f"Processing{n_before} → {len(df_filtered)}")
    else:
        df_filtered = df_all

    if len(df_filtered) == 0:
        print('Done.')
        df_filtered = df_all

    
    suffix = args.application + (f'_{args.tag}' if args.tag else '')
    out1 = os.path.join(args.candidates_dir, f'retrieval_{suffix}.csv')
    if os.path.exists(out1) and not args.overwrite:
        raise FileExistsError(
            f"{out1}"
        )
    df_filtered.to_csv(out1, index=False)
    print(f"{out1}")

    
    df_dft = (df_filtered.sort_values('similarity', ascending=False)
                          .drop_duplicates('formula')
                          .head(args.n_dft)
                          .reset_index(drop=True))
    if 'pred_ehull' in df_dft.columns:
        df_dft['ehull_rank'] = df_dft['pred_ehull'].rank(ascending=True).astype(int)

    cols = ['jid', 'formula', 'similarity', 'query',
            'pred_bandgap', 'pred_formation_energy', 'pred_ehull']
    if 'ehull_rank' in df_dft.columns:
        cols.append('ehull_rank')
    out2 = os.path.join(args.candidates_dir, f'dft_jobs_{suffix}.csv')
    if os.path.exists(out2) and not args.overwrite:
        raise FileExistsError(
            f"{out2}"
        )
    df_dft[cols].to_csv(out2, index=False)
    print(f"Processing{out2}  (top-{args.n_dft} unique formulas)")

    print(f"\n{'='*70}")
    print(f"Processing{args.application}）")
    print(f"{'='*70}")
    for i, r in df_dft.head(30).iterrows():
        eh_rank = (f"  eh_rank={int(r['ehull_rank'])}/{len(df_dft)}"
                   if 'ehull_rank' in df_dft.columns else "")
        print(f"  #{i+1:>2}  {r['formula']:<15} jid={r['jid']:<15} "
              f"sim={r['similarity']:.3f}  Eg={r['pred_bandgap']:.2f}  "
              f"FE={r['pred_formation_energy']:.2f}{eh_rank}")


if __name__ == "__main__":
    main()
