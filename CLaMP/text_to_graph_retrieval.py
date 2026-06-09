
import os
import re
import json
import math
import argparse
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from utils import load_norm_stats, load_checkpoint
from models import build_model
from dataset import MaterialDataset, collate_fn




def parse_top_ks(s: str) -> List[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


def to_float_or_none(x):
    if x is None:
        return None
    try:
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None


def parse_elements(formula: str) -> set:
    if not formula:
        return set()
    return set(re.findall(r"([A-Z][a-z]?)", formula))


def get_split_path(split: str) -> str:
    split = split.lower()
    if split == "train":
        return CFG.TRAIN_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)
    if split == "val":
        return CFG.VAL_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)
    if split == "test":
        return CFG.TEST_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)
    raise ValueError(f"Unknown split: {split}")


def load_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if not txt:
        return []

    # JSON object
    if txt[0] in "[{":
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and "queries" in obj:
                return obj["queries"]
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "query" in obj:
                return [obj]
        except json.JSONDecodeError:
            pass

    # JSONL fallback
    queries = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        queries.append(json.loads(line))
    return queries


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)




def load_multimodal_model(ckpt_dir: str, ckpt_name: str = "best_model.pt"):
    stats_path = os.path.join(ckpt_dir, "norm_stats.json")
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"{stats_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path}")

    global_mean, global_std = load_norm_stats(stats_path, device=CFG.DEVICE)

    model = build_model("multimodal", CFG).to(CFG.DEVICE)
    ckpt = load_checkpoint(ckpt_path, device=CFG.DEVICE)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    print(f"{ckpt_path}")
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"   epoch={ckpt.get('epoch')}")

    return model, global_mean, global_std


@torch.no_grad()
def encode_candidate_pool(
    model,
    dataset: MaterialDataset,
    batch_size: int,
    num_workers: int,
    global_mean: torch.Tensor,
    global_std: torch.Tensor,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    all_g = []
    all_t = []
    all_pred = []
    all_true = []
    all_mask = []
    all_jids = []

    for batch in tqdm(loader, desc='Processing'):
        bg = batch["graph"].to(CFG.DEVICE)
        te = batch["text_emb"].to(CFG.DEVICE)

        
        g_norm = model.encode_graph(bg)

        
        t_norm = model.encode_text(te)

        
        pred_norm = model.predict_properties(bg)
        pred_real = pred_norm * global_std + global_mean

        all_g.append(g_norm.cpu().numpy())
        all_t.append(t_norm.cpu().numpy())
        all_pred.append(pred_real.cpu().numpy())
        all_true.append(batch["labels"].numpy())
        all_mask.append(batch["label_mask"].numpy())
        all_jids.extend(batch["jids"])

    graph_embs = np.concatenate(all_g, axis=0)
    text_embs = np.concatenate(all_t, axis=0)
    pred_props = np.concatenate(all_pred, axis=0)
    true_props = np.concatenate(all_true, axis=0)
    masks = np.concatenate(all_mask, axis=0)

    records = dataset.samples

    
    for i, r in enumerate(records):
        r.setdefault("jid", all_jids[i])
        r.setdefault("formula", "")

    return graph_embs, text_embs, pred_props, true_props, masks, records, all_jids


# Paired Text -> Graph Retrieval

def paired_text_to_graph_recall(
    text_embs: np.ndarray,
    graph_embs: np.ndarray,
    top_ks: List[int],
) -> Dict[str, float]:
    sim = text_embs @ graph_embs.T
    n = sim.shape[0]
    max_k = max(top_ks)

    # top indices for each text query
    top_idx = np.argsort(-sim, axis=1)[:, :max_k]

    results = {}
    ranks = []

    for i in range(n):
        hit_positions = np.where(top_idx[i] == i)[0]
        if len(hit_positions) > 0:
            rank = int(hit_positions[0]) + 1
        else:
            
            order = np.argsort(-sim[i])
            rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    for k in top_ks:
        results[f"text_to_graph_recall@{k}"] = float(np.mean(ranks <= k))

    results["text_to_graph_mrr"] = float(np.mean(1.0 / ranks))
    results["n_queries"] = int(n)
    results["pool_size"] = int(graph_embs.shape[0])

    return results


# Natural Language -> Graph Semantic Retrieval

def soft_window_score(value, lo, hi, soft_margin_ratio=0.3) -> float:
    value = to_float_or_none(value)
    if value is None:
        return 0.0

    lo = float(lo)
    hi = float(hi)

    if lo <= value <= hi:
        return 1.0

    width = max(hi - lo, 0.5)
    margin = width * soft_margin_ratio

    if value < lo:
        d = lo - value
    else:
        d = value - hi

    return max(0.0, 1.0 - d / (margin + 1e-12))


def score_record_against_expected(
    record: Dict[str, Any],
    props: Dict[str, Optional[float]],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    formula = record.get("formula", "")
    elems = parse_elements(formula)

    components = {}
    violations = []
    hard_ok = True

    
    if "must_contain" in expected:
        miss = [e for e in expected["must_contain"] if e not in elems]
        ok = len(miss) == 0
        components["must_contain"] = 1.0 if ok else 0.0
        if not ok:
            hard_ok = False
            violations.append(f"missing {miss}")

    if "must_contain_any" in expected:
        need = expected["must_contain_any"]
        ok = any(e in elems for e in need)
        components["must_contain_any"] = 1.0 if ok else 0.0
        if not ok:
            hard_ok = False
            violations.append(f"none of {need}")

    if "must_contain_any_2" in expected:
        need = expected["must_contain_any_2"]
        ok = any(e in elems for e in need)
        components["must_contain_any_2"] = 1.0 if ok else 0.0
        if not ok:
            hard_ok = False
            violations.append(f"none of {need}")

    if "must_not_contain" in expected:
        bad = [e for e in expected["must_not_contain"] if e in elems]
        ok = len(bad) == 0
        components["must_not_contain"] = 1.0 if ok else 0.0
        if not ok:
            hard_ok = False
            violations.append(f"contains forbidden {bad}")

    if expected.get("transition_metal_free", False):
        tm = {
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        }
        bad = elems & tm
        ok = len(bad) == 0
        components["transition_metal_free"] = 1.0 if ok else 0.0
        if not ok:
            hard_ok = False
            violations.append(f"has transition metals {sorted(bad)}")

    if "n_unique_elements" in expected:
        target = int(expected["n_unique_elements"])
        diff = abs(len(elems) - target)
        if diff == 0:
            components["n_unique_elements"] = 1.0
        elif diff == 1:
            components["n_unique_elements"] = 0.5
        else:
            components["n_unique_elements"] = 0.0
        if diff != 0:
            hard_ok = False
            violations.append(f"n_elem={len(elems)} vs expected={target}")

    # ---------- bandgap ----------
    bg = props.get("bandgap", None)

    if "bandgap_range" in expected:
        lo, hi = expected["bandgap_range"]
        s = soft_window_score(bg, lo, hi, soft_margin_ratio=0.4)
        components["bandgap_range"] = s
        if s < 1.0:
            
            hard_ok = False
            violations.append(f"bandgap={bg} not in [{lo}, {hi}]")

    if expected.get("bandgap") == "metallic":
        bg_val = to_float_or_none(bg)
        ok = bg_val is not None and bg_val <= 0.3
        components["metallic"] = 1.0 if ok else 0.0
        if not ok:
            hard_ok = False
            violations.append(f"expected metallic, got bandgap={bg}")

    # ---------- ehull / stability ----------
    eh = props.get("ehull", None)
    eh_val = to_float_or_none(eh)

    if "stability" in expected:
        stab = expected["stability"]
        if stab == "stable":
            ok = eh_val is not None and eh_val <= CFG.EHULL_STABLE_THRESHOLD
            # soft score
            if eh_val is None:
                s = 0.0
            elif eh_val <= CFG.EHULL_STABLE_THRESHOLD:
                s = 1.0
            elif eh_val <= 0.1:
                s = 0.7
            elif eh_val <= 0.2:
                s = 0.4
            else:
                s = 0.0
            components["stability"] = s
            if not ok:
                hard_ok = False
                violations.append(f"not stable, ehull={eh}")

        elif stab == "stable_or_metastable":
            ok = eh_val is not None and eh_val <= 0.2
            if eh_val is None:
                s = 0.0
            elif eh_val <= 0.2:
                s = 1.0
            elif eh_val <= 0.5:
                s = 0.5
            else:
                s = 0.0
            components["stability"] = s
            if not ok:
                hard_ok = False
                violations.append(f"not stable_or_metastable, ehull={eh}")

    if "ehull_range" in expected:
        lo, hi = expected["ehull_range"]
        s = soft_window_score(eh, lo, hi, soft_margin_ratio=0.5)
        components["ehull_range"] = s
        if s < 1.0:
            hard_ok = False
            violations.append(f"ehull={eh} not in [{lo}, {hi}]")

    # ---------- formation energy ----------
    fe = props.get("formation_energy", None)
    fe_val = to_float_or_none(fe)

    if expected.get("formation_energy") == "negative":
        ok = fe_val is not None and fe_val < 0
        components["formation_energy_negative"] = 1.0 if ok else 0.0
        if not ok:
            hard_ok = False
            violations.append(f"formation_energy not negative: {fe}")

    elif expected.get("formation_energy") == "very_negative":
        ok = fe_val is not None and fe_val < -1.0
        if fe_val is None:
            s = 0.0
        elif fe_val < -1.0:
            s = 1.0
        elif fe_val < -0.3:
            s = 0.6
        else:
            s = 0.0
        components["formation_energy_very_negative"] = s
        if not ok:
            hard_ok = False
            violations.append(f"formation_energy not very negative: {fe}")

    elif expected.get("formation_energy") == "positive":
        ok = fe_val is not None and fe_val > 0
        components["formation_energy_positive"] = 1.0 if ok else 0.0
        if not ok:
            hard_ok = False
            violations.append(f"formation_energy not positive: {fe}")

    
    if len(components) == 0:
        semantic_score = 0.0
        relevant_strict = False
    else:
        semantic_score = float(np.mean(list(components.values())))
        relevant_strict = bool(hard_ok)

    return {
        "semantic_score": semantic_score,
        "relevant_strict": relevant_strict,
        "components": {k: float(v) for k, v in components.items()},
        "violations": violations,
    }


def record_to_props(
    record: Dict[str, Any],
    pred_row: Optional[np.ndarray] = None,
    use_pred: bool = False,
) -> Dict[str, Optional[float]]:
    if use_pred:
        assert pred_row is not None
        return {
            "bandgap": float(pred_row[0]),
            "formation_energy": float(pred_row[1]),
            "ehull": float(pred_row[2]),
        }

    return {
        "bandgap": to_float_or_none(record.get("bandgap")),
        "formation_energy": to_float_or_none(record.get("formation_energy")),
        "ehull": to_float_or_none(record.get("ehull")),
    }


def build_relevance_sets(
    queries: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    pred_props: np.ndarray,
    use_pred_for_relevance: bool = False,
) -> List[Dict[str, Any]]:
    out = []

    for q in queries:
        expected = q.get("expected_properties", {}) or {}
        relevant = []
        scores = []

        for i, r in enumerate(records):
            props = record_to_props(
                r,
                pred_row=pred_props[i],
                use_pred=use_pred_for_relevance,
            )
            ev = score_record_against_expected(r, props, expected)
            scores.append(ev["semantic_score"])
            if ev["relevant_strict"]:
                relevant.append(i)

        out.append({
            "query_id": q.get("id", q.get("query_id", None)),
            "query": q.get("query", ""),
            "n_relevant": len(relevant),
            "relevant_indices": set(relevant),
            "all_rule_scores": np.array(scores, dtype=np.float32),
        })

    return out


def compute_retrieval_metrics_for_one_query(
    retrieved: List[int],
    relevant_set: set,
    top_ks: List[int],
) -> Dict[str, Optional[float]]:
    metrics = {}
    n_rel = len(relevant_set)

    first_hit_rank = None

    for k in top_ks:
        topk = retrieved[:k]
        hit_count = len(set(topk) & relevant_set)

        metrics[f"precision@{k}"] = float(hit_count / max(len(topk), 1))

        if n_rel > 0:
            metrics[f"recall@{k}"] = float(hit_count / n_rel)
        else:
            metrics[f"recall@{k}"] = None

        metrics[f"hit@{k}"] = float(hit_count > 0)

    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant_set:
            first_hit_rank = rank
            break

    metrics["mrr"] = float(1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
    metrics["n_relevant"] = int(n_rel)

    return metrics


def load_sentence_transformer(qwen_path: str, local_files_only: bool):
    from sentence_transformers import SentenceTransformer

    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    kwargs = {
        "device": str(CFG.DEVICE),
    }

    
    
    if local_files_only:
        kwargs["local_files_only"] = True

    try:
        return SentenceTransformer(qwen_path, **kwargs)
    except TypeError:
        
        return SentenceTransformer(qwen_path, device=str(CFG.DEVICE))


@torch.no_grad()
def encode_nl_queries(
    queries: List[Dict[str, Any]],
    model,
    qwen_path: str,
    local_files_only: bool,
) -> np.ndarray:
    print(f"{qwen_path}")
    qwen = load_sentence_transformer(qwen_path, local_files_only=local_files_only)

    texts = [q.get("query", "") for q in queries]
    query_raw = qwen.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=False,
    ).float().to(CFG.DEVICE)

    print(f"   raw query embedding shape: {tuple(query_raw.shape)}")
    print(f"   CFG.LLM_DIM: {CFG.LLM_DIM}")

    if query_raw.shape[-1] != CFG.LLM_DIM:
        raise ValueError(
            f"Query embedding dim={query_raw.shape[-1]} != CFG.LLM_DIM={CFG.LLM_DIM}. "
            f"Processing"
        )

    query_proj = model.encode_text(query_raw)
    return query_proj.cpu().numpy()


def natural_language_text_to_graph_retrieval(
    queries: List[Dict[str, Any]],
    query_embs: np.ndarray,
    graph_embs: np.ndarray,
    records: List[Dict[str, Any]],
    pred_props: np.ndarray,
    relevance_info: List[Dict[str, Any]],
    top_ks: List[int],
    hard_filter: bool = False,
    rerank_weight: float = 0.0,
    use_pred_for_display_score: bool = False,
) -> Dict[str, Any]:
    sim = query_embs @ graph_embs.T
    max_k = max(top_ks)

    all_query_results = []

    for qi, q in enumerate(queries):
        rel_set = relevance_info[qi]["relevant_indices"]
        rule_scores = relevance_info[qi]["all_rule_scores"]

        candidate_indices = np.arange(graph_embs.shape[0])

        if hard_filter:
            
            
            if len(rel_set) > 0:
                candidate_indices = np.array(sorted(rel_set), dtype=np.int64)
            else:
                candidate_indices = np.arange(graph_embs.shape[0])

        base_scores = sim[qi, candidate_indices]

        if rerank_weight > 0:
            final_scores = base_scores + rerank_weight * rule_scores[candidate_indices]
        else:
            final_scores = base_scores

        
        order_local = np.argsort(-final_scores)
        ranked_indices = candidate_indices[order_local].tolist()

        top_for_output = ranked_indices[:max_k]

        metrics = compute_retrieval_metrics_for_one_query(
            retrieved=ranked_indices,
            relevant_set=rel_set,
            top_ks=top_ks,
        )

        items = []
        for rank, idx in enumerate(top_for_output, start=1):
            r = records[idx]

            props_true = record_to_props(r, use_pred=False)
            props_pred = record_to_props(r, pred_row=pred_props[idx], use_pred=True)

            
            props_for_score = props_pred if use_pred_for_display_score else props_true
            ev = score_record_against_expected(
                r,
                props_for_score,
                q.get("expected_properties", {}) or {},
            )

            items.append({
                "rank": rank,
                "index": int(idx),
                "jid": r.get("jid"),
                "formula": r.get("formula", ""),
                "similarity": float(sim[qi, idx]),
                "rule_score": float(rule_scores[idx]),
                "final_score": float(
                    sim[qi, idx] + rerank_weight * rule_scores[idx]
                ),
                "relevant_strict": bool(idx in rel_set),
                "true": props_true,
                "predicted": props_pred,
                "semantic_eval": ev,
            })

        all_query_results.append({
            "query_id": q.get("id", q.get("query_id", qi)),
            "query": q.get("query", ""),
            "category": q.get("category", ""),
            "intent": q.get("intent", ""),
            "expected_properties": q.get("expected_properties", {}) or {},
            "n_relevant_in_pool": int(len(rel_set)),
            "metrics": metrics,
            "top_results": items,
        })

    
    summary_metrics = {}
    for k in top_ks:
        p_vals = [r["metrics"][f"precision@{k}"] for r in all_query_results]
        h_vals = [r["metrics"][f"hit@{k}"] for r in all_query_results]

        
        r_vals = [
            r["metrics"][f"recall@{k}"]
            for r in all_query_results
            if r["metrics"][f"recall@{k}"] is not None
        ]

        summary_metrics[f"mean_precision@{k}"] = float(np.mean(p_vals)) if p_vals else None
        summary_metrics[f"mean_recall@{k}"] = float(np.mean(r_vals)) if r_vals else None
        summary_metrics[f"mean_hit@{k}"] = float(np.mean(h_vals)) if h_vals else None

    mrr_vals = [r["metrics"]["mrr"] for r in all_query_results]
    summary_metrics["mean_mrr"] = float(np.mean(mrr_vals)) if mrr_vals else None

    return {
        "mode": "natural_language_text_to_graph",
        "top_ks": top_ks,
        "hard_filter": hard_filter,
        "rerank_weight": rerank_weight,
        "n_queries": len(queries),
        "pool_size": int(graph_embs.shape[0]),
        "summary": summary_metrics,
        "queries": all_query_results,
    }




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=["pair", "nl"],
        help="pair = test text_emb -> graph paired recall; nl = natural language query -> graph semantic retrieval",
    )

    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt_dir", default=CFG.CKPT_MULTIMODAL)
    parser.add_argument("--ckpt_name", default="best_model.pt")
    parser.add_argument("--max_samples", type=int, default=CFG.MAX_TEST_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    parser.add_argument("--top_ks", default="1,5,10")

    # NL mode args
    parser.add_argument("--queries", default="nl_queries.jsonl")
    parser.add_argument("--qwen_path", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--hard_filter",
        action="store_true",
        help='See README.',
    )
    parser.add_argument(
        "--rerank_weight",
        type=float,
        default=0.0,
        help='See README.',
    )
    parser.add_argument(
        "--use_pred_for_relevance",
        action="store_true",
        help='See README.',
    )
    parser.add_argument(
        "--use_pred_for_display_score",
        action="store_true",
        help='See README.',
    )

    args = parser.parse_args()

    top_ks = parse_top_ks(args.top_ks)

    print("=" * 80)
    print(f"Text -> Graph Retrieval")
    print(f"mode       : {args.mode}")
    print(f"split      : {args.split}")
    print(f"ckpt_dir   : {args.ckpt_dir}")
    print(f"top_ks     : {top_ks}")
    print(f"device     : {CFG.DEVICE}")
    print("=" * 80)

    model, global_mean, global_std = load_multimodal_model(
        args.ckpt_dir,
        ckpt_name=args.ckpt_name,
    )

    data_path = get_split_path(args.split)
    dataset = MaterialDataset(data_path, max_samples=args.max_samples)

    graph_embs, text_embs, pred_props, true_props, masks, records, jids = encode_candidate_pool(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        global_mean=global_mean,
        global_std=global_std,
    )

    print(f"{len(records)}")
    print(f"graph_embs: {graph_embs.shape}")
    print(f"text_embs : {text_embs.shape}")

    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

    if args.mode == "pair":
        results = paired_text_to_graph_recall(
            text_embs=text_embs,
            graph_embs=graph_embs,
            top_ks=top_ks,
        )

        print("\n[Paired Text -> Graph Retrieval]")
        for k in top_ks:
            key = f"text_to_graph_recall@{k}"
            print(f"  Recall@{k:<3}: {results[key] * 100:.2f}%")
        print(f"  MRR      : {results['text_to_graph_mrr']:.4f}")

        out_path = os.path.join(CFG.RESULTS_DIR, "text_to_graph_pair_results.json")
        save_json(results, out_path)
        print(f"{out_path}")
        return

    if args.mode == "nl":
        queries = load_queries(args.queries)
        if len(queries) == 0:
            raise ValueError(f"{args.queries}")

        print(f"{len(queries)}")

        query_embs = encode_nl_queries(
            queries=queries,
            model=model,
            qwen_path=args.qwen_path,
            local_files_only=args.local_files_only,
        )

        relevance_info = build_relevance_sets(
            queries=queries,
            records=records,
            pred_props=pred_props,
            use_pred_for_relevance=args.use_pred_for_relevance,
        )

        for q, rel in zip(queries, relevance_info):
            print(
                f"   query_id={q.get('id', q.get('query_id', 'NA'))} "
                f"n_relevant={rel['n_relevant']} "
                f"query={q.get('query', '')[:80]}"
            )

        results = natural_language_text_to_graph_retrieval(
            queries=queries,
            query_embs=query_embs,
            graph_embs=graph_embs,
            records=records,
            pred_props=pred_props,
            relevance_info=relevance_info,
            top_ks=top_ks,
            hard_filter=args.hard_filter,
            rerank_weight=args.rerank_weight,
            use_pred_for_display_score=args.use_pred_for_display_score,
        )

        print("\n[Natural Language -> Graph Semantic Retrieval]")
        for k in top_ks:
            print(
                f"  K={k:<3} "
                f"P@K={results['summary'][f'mean_precision@{k}']:.4f}  "
                f"R@K={results['summary'][f'mean_recall@{k}'] if results['summary'][f'mean_recall@{k}'] is not None else None}  "
                f"Hit@K={results['summary'][f'mean_hit@{k}']:.4f}"
            )
        print(f"  MRR={results['summary']['mean_mrr']:.4f}")

        out_path = os.path.join(CFG.RESULTS_DIR, "text_to_graph_nl_results.json")
        save_json(results, out_path)
        print(f"{out_path}")
        return


if __name__ == "__main__":
    main()
