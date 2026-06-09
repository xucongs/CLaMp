
import os
import json
import argparse

import torch
from tqdm import tqdm

from config import CFG


DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def load_whitelist_jids(split_dir):
    needed = set()
    for split in ['train', 'val', 'test']:
        path = os.path.join(split_dir, split, f"{split}_metadata.jsonl")
        if not os.path.exists(path):
            print(f"Split file not found, skipping: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    needed.add(json.loads(line)['jid'])
    return needed


def load_tasks(input_jsonl, output_dir, whitelist=None, skip_leaked=True):
    tasks = []
    n_already = n_leaked = n_filtered = n_total = 0
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            data = json.loads(line)
            jid, text = data['jid'], data['text_description']
            if skip_leaked and data.get('leaked', False):
                n_leaked += 1
                continue
            if whitelist is not None and jid not in whitelist:
                n_filtered += 1
                continue
            if os.path.exists(os.path.join(output_dir, f"{jid}.pt")):
                n_already += 1
                continue
            tasks.append({'jid': jid, 'text': text})

    print(f"Processing{input_jsonl}):")
    print(f"Total records: {n_total}")
    print(f"Already embedded: {n_already}")
    print(f"Skipped leaked records: {n_leaked}")
    if whitelist is not None:
        print(f"Filtered by split whitelist: {n_filtered}")
    print(f"Pending records: {len(tasks)}")
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default=CFG.TEXTS_CLEAN_JSONL)
    parser.add_argument('--output', default=CFG.EMBED_DIR_CLEAN)
    parser.add_argument('--model',  default=DEFAULT_MODEL)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_len',    type=int, default=512)
    parser.add_argument('--whitelist_split', action='store_true')
    parser.add_argument('--allow_leaked', action='store_true')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    print("=" * 65)
    print('Text embedding extraction')
    print("=" * 65)
    print(f"Input:       {args.input}")
    print(f"Output dir:  {args.output}")
    print(f"  Model:       {args.model}")
    print(f"  Batch / Len: {args.batch_size} / {args.max_len}")
    print(f"Skip leaked: {not args.allow_leaked}")
    print(f"Whitelist:   {args.whitelist_split}")
    print("=" * 65)

    os.makedirs(args.output, exist_ok=True)
    err_path = "embedding_errors.txt"

    whitelist = load_whitelist_jids(CFG.SPLIT_DIR) if args.whitelist_split else None
    if whitelist is not None:
        print(f"Whitelist JIDs: {len(whitelist)}")

    tasks = load_tasks(
        args.input, args.output,
        whitelist=whitelist,
        skip_leaked=not args.allow_leaked,
    )
    if not tasks:
        print('Text embedding extraction')
        return

    print('Text embedding extraction')
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    backend = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model, device=device)
        backend = 'sentence_transformers'
        print(f"Processing{device})")
    except ImportError:
        print('Text embedding extraction')
        print('Text embedding extraction')
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if device != 'cpu' else torch.float32,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        backend = 'transformers'
        print(f"Processing{device})")

    
    n_done = n_failed = 0
    embed_dim = None
    batches = [tasks[i:i + args.batch_size]
               for i in range(0, len(tasks), args.batch_size)]

    with torch.no_grad():
        for batch in tqdm(batches, desc='Processing'):
            texts = [b['text'] for b in batch]
            jids  = [b['jid']  for b in batch]
            try:
                if backend == 'sentence_transformers':
                    emb = model.encode(
                        texts,
                        batch_size=len(texts),
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        normalize_embeddings=False,
                    )
                else:
                    inputs = tokenizer(
                        texts, padding=True, truncation=True,
                        max_length=args.max_len, return_tensors="pt",
                    ).to(device)
                    outputs = model(**inputs)
                    last_hidden = outputs.last_hidden_state
                    mask = inputs.attention_mask.unsqueeze(-1).float()
                    summed = (last_hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1)
                    emb = summed / counts

                emb = emb.cpu().float()

                if embed_dim is None:
                    embed_dim = emb.shape[-1]
                    print(f"Embedding dimension: {embed_dim}")
                    print(f"Embedding dimension: {embed_dim}")

                for i, jid in enumerate(jids):
                    save_path = os.path.join(args.output, f"{jid}.pt")
                    torch.save(emb[i].clone(), save_path)
                n_done += len(batch)

            except Exception as e:
                n_failed += len(batch)
                with open(err_path, 'a', encoding='utf-8') as f:
                    for jid in jids:
                        f.write(f"{jid}\t{type(e).__name__}: {e}\n")

    print(f"\n{'='*65}")
    print('Text embedding extraction')
    print(f"Completed: {n_done}")
    print(f"Failed: {n_failed}")
    if n_failed > 0:
        print(f"Error log: {err_path}")
    print(f"Saved embeddings under: {os.path.abspath(args.output)}")
    if embed_dim:
        print(f"Embedding dimension: {embed_dim}")
        print(f"Embedding dimension: {embed_dim}")
        print('Text embedding extraction')


if __name__ == "__main__":
    main()
