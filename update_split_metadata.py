
import os
import json
import argparse

from config import CFG


def update_split(jsonl_path, new_embed_dir, filter_missing=False):
    if not os.path.exists(jsonl_path):
        print(f"{jsonl_path}")
        return

    backup_path = jsonl_path + '.bak'
    if not os.path.exists(backup_path):
        os.system(f'cp "{jsonl_path}" "{backup_path}"')
        print(f"{backup_path}")

    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_kept = 0
    n_dropped = 0
    out_lines = []
    for r in records:
        new_path = os.path.join(new_embed_dir, f"{r['jid']}.pt")
        if filter_missing and not os.path.exists(new_path):
            n_dropped += 1
            continue
        r['text_emb_path'] = new_path
        out_lines.append(json.dumps(r, ensure_ascii=False))
        n_kept += 1

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')

    print(f"   {os.path.basename(jsonl_path)}: kept={n_kept}, "
          f"dropped={n_dropped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_embed_dir', default=CFG.EMBED_DIR_CLEAN)
    parser.add_argument('--filter_missing', action='store_true',
                        help='See README.')
    args = parser.parse_args()

    print("=" * 65)
    print('Done.')
    print(f"{args.new_embed_dir}")
    print(f"{args.filter_missing}")
    print("=" * 65)

    for split in ['train', 'val', 'test']:
        path = os.path.join(CFG.SPLIT_DIR, split, f"{split}_metadata.jsonl")
        update_split(path, args.new_embed_dir, args.filter_missing)

    print('Done.')


if __name__ == "__main__":
    main()
