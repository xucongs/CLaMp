
import os
import json
import random
import pandas as pd
from tqdm import tqdm


GRAPH_DIR        = "pyg_graphs"                  
TEXT_EMB_DIR     = "text_embeddings_clean"             
TEXTS_JSONL      = "jarvis_clean_texts.jsonl"  
PROPS_CSV        = "jarvis_properties.csv"       

OUTPUT_BASE      = "dataset_split"               

TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO       = 0.15   # = 1 - TRAIN_RATIO - VAL_RATIO

RANDOM_SEED      = 42


def load_props(props_csv: str) -> dict:
    df = pd.read_csv(props_csv, sep=',', dtype={'jid': str}, low_memory=False)
    df = df.drop_duplicates(subset=['jid']).reset_index(drop=True)

    props = {}
    for _, row in df.iterrows():
        jid = str(row['jid'])

        def safe_float(val):
            try:
                v = float(val)
                return None if pd.isna(v) else v
            except (TypeError, ValueError):
                return None

        props[jid] = {
            'bandgap':          safe_float(row.get('optb88vdw_bandgap')),
            'formation_energy': safe_float(row.get('formation_energy_peratom')),
            'ehull':            safe_float(row.get('ehull')),
        }

    return props


def load_text_meta(texts_jsonl: str) -> dict:
    meta = {}
    with open(texts_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            jid  = str(item['jid'])
            meta[jid] = {
                'formula':          item.get('formula', ''),
                'text_description': item.get('text_description', ''),
            }
    return meta


def main():
    print("=" * 60)
    print('Creating dataset split')
    print("=" * 60)

    
    map_path = os.path.join(GRAPH_DIR, 'processed_map.csv')
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"Missing graph map: {map_path}"
        )
    df_map = pd.read_csv(map_path, dtype={'jid': str})
    graph_jids = set(df_map['jid'].tolist())
    print(f"Valid graph files: {len(graph_jids)}")

    
    print(f"Reading property table: {PROPS_CSV}")
    props_dict = load_props(PROPS_CSV)

    
    print(f"Reading text descriptions: {TEXTS_JSONL}")
    text_meta = load_text_meta(TEXTS_JSONL)

    
    
    valid_samples = []
    skipped_no_text_emb  = 0
    skipped_no_any_label = 0

    for jid in tqdm(graph_jids, desc='Processing'):
        
        emb_path = os.path.join(TEXT_EMB_DIR, f"{jid}.pt")
        if not os.path.exists(emb_path):
            skipped_no_text_emb += 1
            continue

        
        props = props_dict.get(jid, {})
        has_any_label = any(v is not None for v in props.values())
        if not has_any_label:
            skipped_no_any_label += 1
            continue

        valid_samples.append(jid)

    print(f"Valid samples: {len(valid_samples)}")
    print(f"Skipped without text embeddings: {skipped_no_text_emb}")
    print(f"Skipped without labels: {skipped_no_any_label}")

    
    bg_count = sum(1 for j in valid_samples
                   if props_dict.get(j, {}).get('bandgap') is not None)
    fe_count = sum(1 for j in valid_samples
                   if props_dict.get(j, {}).get('formation_energy') is not None)
    eh_count = sum(1 for j in valid_samples
                   if props_dict.get(j, {}).get('ehull') is not None)
    all_3    = sum(1 for j in valid_samples if all(
        props_dict.get(j, {}).get(k) is not None
        for k in ['bandgap', 'formation_energy', 'ehull']
    ))
    print(f"Valid samples: {len(valid_samples)}")
    print(f"   bandgap:          {bg_count:>6} ({bg_count/len(valid_samples)*100:.1f}%)")
    print(f"   formation_energy: {fe_count:>6} ({fe_count/len(valid_samples)*100:.1f}%)")
    print(f"   ehull:            {eh_count:>6} ({eh_count/len(valid_samples)*100:.1f}%)")
    print(f"Processing{all_3:>6} ({all_3/len(valid_samples)*100:.1f}%)")

    
    random.seed(RANDOM_SEED)
    random.shuffle(valid_samples)

    n_total = len(valid_samples)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    
    n_test  = n_total - n_train - n_val

    train_jids = valid_samples[:n_train]
    val_jids   = valid_samples[n_train: n_train + n_val]
    test_jids  = valid_samples[n_train + n_val:]

    print('Creating dataset split')
    print(f"Train: {len(train_jids)}")
    print(f"Validation: {len(val_jids)}")
    print(f"Test: {len(test_jids)}")

    
    jid_to_pt = {}
    for _, row in df_map.iterrows():
        jid_to_pt[str(row['jid'])] = row['pt_file']

    
    def write_jsonl(jid_list, output_path, split_name):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for jid in tqdm(jid_list, desc=f"{split_name}"):
                props  = props_dict.get(jid, {})
                tmeta  = text_meta.get(jid, {})
                record = {
                    'jid':              jid,
                    'formula':          tmeta.get('formula', ''),
                    'text_description': tmeta.get('text_description', ''),
                    'pt_file':          jid_to_pt[jid],         
                    'graph_dir':        GRAPH_DIR,               
                    'text_emb_path':    os.path.join(
                        TEXT_EMB_DIR, f"{jid}.pt"
                    ),
                    
                    'bandgap':          props.get('bandgap'),
                    'formation_energy': props.get('formation_energy'),
                    'ehull':            props.get('ehull'),
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    write_jsonl(
        train_jids,
        os.path.join(OUTPUT_BASE, 'train', 'train_metadata.jsonl'),
        'train'
    )
    write_jsonl(
        val_jids,
        os.path.join(OUTPUT_BASE, 'val', 'val_metadata.jsonl'),
        'val'
    )
    write_jsonl(
        test_jids,
        os.path.join(OUTPUT_BASE, 'test', 'test_metadata.jsonl'),
        'test'
    )

    print(f"Processing{OUTPUT_BASE}/")
    print(f"   {OUTPUT_BASE}/train/train_metadata.jsonl")
    print(f"   {OUTPUT_BASE}/val/val_metadata.jsonl")
    print(f"   {OUTPUT_BASE}/test/test_metadata.jsonl")
    print('Creating dataset split')


if __name__ == "__main__":
    main()
