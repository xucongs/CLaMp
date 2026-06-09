import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
import os
import json
import re
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score

from config import CFG


SYSTEM_PROMPT = """You are a materials science expert. Given a qualitative description of a crystalline material, predict three numerical properties:

1. Bandgap (eV): non-negative real number. Metals = 0. Wide-gap insulators > 3.
2. Formation energy per atom (eV/atom): can be negative (stable) or positive.
3. Ehull (eV/atom): non-negative. Stable phases ≤ 0.05.

Output ONLY a JSON object with three numbers and nothing else:
{"bandgap": <float>, "formation_energy": <float>, "ehull": <float>}"""


def parse_llm_output(text: str) -> dict:
    
    text_clean = re.sub(r'<think>.*?</think>', '', text,
                          flags=re.DOTALL)
    
    if '<think>' in text_clean and '</think>' not in text_clean:
        
        pass

    
    
    json_matches = re.findall(r'\{[^{}]*\}', text_clean, re.DOTALL)
    for json_str in reversed(json_matches):
        try:
            obj = json.loads(json_str)
            if 'bandgap' in obj or 'formation_energy' in obj or 'ehull' in obj:
                return {
                    'bandgap': float(obj.get('bandgap', np.nan)),
                    'formation_energy': float(obj.get('formation_energy', np.nan)),
                    'ehull': float(obj.get('ehull', np.nan)),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    
    out = {}
    for k in ['bandgap', 'formation_energy', 'ehull']:
        m = re.search(rf'"?{k}"?\s*[:=]\s*([-+]?\d*\.?\d+)',
                       text_clean, re.IGNORECASE)
        out[k] = float(m.group(1)) if m else np.nan
    return out


def build_messages(description, few_shot_examples=None):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({
                'role': 'user',
                'content': f"Material: {ex['desc']}"
            })
            messages.append({
                'role': 'assistant',
                'content': (f'{{"bandgap": {ex["bg"]:.4f}, '
                            f'"formation_energy": {ex["fe"]:.4f}, '
                            f'"ehull": {ex["eh"]:.4f}}}')
            })
    messages.append({
        'role': 'user',
        'content': f"Material: {description}"
    })
    return messages


def load_clean_text_map(clean_jsonl):
    out = {}
    with open(clean_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('leaked', False):
                continue
            out[r['jid']] = r['text_description']
    return out


def load_test_samples(test_jsonl, desc_map, max_samples=None):
    samples = []
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            jid = r['jid']
            if jid not in desc_map:
                continue
            samples.append({
                'jid': jid,
                'description': desc_map[jid],
                'true': {
                    'bandgap': r.get('bandgap'),
                    'formation_energy': r.get('formation_energy'),
                    'ehull': r.get('ehull'),
                },
            })
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def load_few_shot_examples(train_jsonl, desc_map, n_shots, seed=42):
    if n_shots <= 0:
        return []
    candidates = []
    with open(train_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            jid = r['jid']
            if jid not in desc_map:
                continue
            if any(r.get(k) is None for k in
                   ['bandgap', 'formation_energy', 'ehull']):
                continue
            candidates.append({
                'desc': desc_map[jid],
                'bg':   r['bandgap'],
                'fe':   r['formation_energy'],
                'eh':   r['ehull'],
            })
    random.seed(seed)
    random.shuffle(candidates)
    metals    = [c for c in candidates if c['bg'] == 0.0]
    semi      = [c for c in candidates if 0 < c['bg'] <= 2.5]
    insulator = [c for c in candidates if c['bg'] > 2.5]
    n_per = max(n_shots // 3, 1)
    return (metals[:n_per] + semi[:n_per] + insulator[:n_per])[:n_shots]


def evaluate(samples, predictions):
    out = {}
    for prop in ['bandgap', 'formation_energy', 'ehull']:
        t_list, p_list, n_failed = [], [], 0
        for s, p in zip(samples, predictions):
            t = s['true'].get(prop)
            v = p.get(prop)
            if t is None or v is None or (isinstance(v, float) and np.isnan(v)):
                n_failed += 1
                continue
            t_list.append(t)
            p_list.append(v)
        if len(t_list) < 2:
            out[prop] = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
                         'n': len(t_list), 'n_failed': n_failed}
            continue
        t_arr = np.array(t_list); p_arr = np.array(p_list)
        out[prop] = {
            'mae':  float(mean_absolute_error(t_arr, p_arr)),
            'rmse': float(np.sqrt(np.mean((p_arr - t_arr) ** 2))),
            'r2':   float(r2_score(t_arr, p_arr)),
            'n':    len(t_list),
            'n_failed': n_failed,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--clean_texts', default=CFG.TEXTS_CLEAN_JSONL)
    parser.add_argument('--test_jsonl',  default=None)
    parser.add_argument('--train_jsonl', default=None)
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--shots',       type=int, default=0)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--gpu_mem_util', type=float, default=0.85)
    parser.add_argument('--max_model_len', type=int, default=8192,
                        help='See README.')
    parser.add_argument('--max_tokens',  type=int, default=8192,
                        help='See README.')
    parser.add_argument('--disable_thinking', action='store_true',
                        help='See README.')
    args = parser.parse_args()

    test_jsonl  = args.test_jsonl  or CFG.TEST_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)
    train_jsonl = args.train_jsonl or CFG.TRAIN_JSONL_TPL.format(split_dir=CFG.SPLIT_DIR)

    print("=" * 65)
    print(f"  LLM Baseline ({args.shots}-shot) - v2 fixed")
    print(f"  Model: {args.model}")
    print(f"  N samples: {args.max_samples}")
    print(f"  Thinking: {'OFF' if args.disable_thinking else 'ON'}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 65)

    desc_map = load_clean_text_map(args.clean_texts)
    print(f"{len(desc_map)}")

    samples = load_test_samples(test_jsonl, desc_map, args.max_samples)
    print(f"{len(samples)}")

    few_shot = load_few_shot_examples(train_jsonl, desc_map, args.shots)
    if few_shot:
        print(f"{len(few_shot)}")

    print('Done.')
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype='bfloat16',
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        
    )
    print('Done.')

    
    chat_template_kwargs = {}
    if args.disable_thinking:
        
        chat_template_kwargs['enable_thinking'] = False

    predictions = []
    raw_logs = []

    for batch_start in tqdm(range(0, len(samples), args.batch_size),
                             desc='Processing'):
        batch = samples[batch_start: batch_start + args.batch_size]
        batch_messages = [build_messages(s['description'], few_shot)
                          for s in batch]

        
        outputs = llm.chat(
            batch_messages,
            sp,
            use_tqdm=False,
            chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None,
        )

        for s, o in zip(batch, outputs):
            txt = o.outputs[0].text.strip()
            parsed = parse_llm_output(txt)
            predictions.append(parsed)
            raw_logs.append({
                'jid': s['jid'],
                'raw': txt,
                'parsed': parsed,
                'true': s['true'],
            })

        
        if batch_start == 0:
            print('Done.')
            print(f"   JID: {raw_logs[0]['jid']}")
            print('Done.')
            print(raw_logs[0]['raw'][:500])
            print(f"   --- PARSED ---")
            print(f"   {raw_logs[0]['parsed']}")
            print(f"   --- TRUE ---")
            print(f"   {raw_logs[0]['true']}")
            print(f"   ====================================\n")

    
    metrics = evaluate(samples, predictions)
    n_total_failed = sum(metrics[p]['n_failed']
                          for p in ['bandgap', 'formation_energy', 'ehull']) / 3
    success_rate = 1 - n_total_failed / len(samples)

    print(f"\n{'='*65}")
    print(f"Processing{args.shots}-shot）")
    print(f"{'='*65}")
    print(f"Processing{success_rate*100:.1f}% "
          f"({len(samples) - int(n_total_failed)}/{len(samples)})")
    print()
    print(f"  {'Processing':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'N':>5} {'Processing':>5}")
    print(f"  {'-'*60}")
    for prop in ['bandgap', 'formation_energy', 'ehull']:
        m = metrics[prop]
        mae_str = f"{m['mae']:>7.4f}" if not np.isnan(m['mae']) else "    nan"
        rmse_str = f"{m['rmse']:>7.4f}" if not np.isnan(m['rmse']) else "    nan"
        r2_str = f"{m['r2']:>7.4f}" if not np.isnan(m['r2']) else "    nan"
        print(f"  {prop:<22} {mae_str}  {rmse_str}  {r2_str}  "
              f"{m['n']:>4}  {m['n_failed']:>4}")

    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
    tag = f"llm_{args.shots}shot"
    if args.disable_thinking:
        tag += "_nothink"
    out_path = os.path.join(CFG.RESULTS_DIR, f"eval_{tag}.json")
    save_obj = {
        'model_type': tag,
        'tag':        tag,
        'llm_model':  args.model,
        'shots':      args.shots,
        'disable_thinking': args.disable_thinking,
        'n_test_samples':   len(samples),
        'parse_success_rate': success_rate,
        'regression': metrics,
        'raw_outputs': raw_logs[:30],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_obj, f, indent=2, ensure_ascii=False, default=float)
    print(f"{out_path}")

    print(f"\n{'='*65}")
    print('Done.')
    print(f"{'='*65}")
    for path, label in [
        (f"{CFG.RESULTS_DIR}/eval_graph_only_clean.json", 'Graph-only'),
        (f"{CFG.RESULTS_DIR}/eval_multimodal_clean.json", 'Multimodal'),
        (out_path, f'LLM ({args.shots}-shot)'),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            mae = d.get('regression', {}).get('bandgap', {}).get('mae')
            if mae is not None and not (isinstance(mae, float) and np.isnan(mae)):
                print(f"  {label:<25} {mae:>8.4f} eV")
            else:
                print(f"  {label:<25}      nan eV")


if __name__ == "__main__":
    main()
