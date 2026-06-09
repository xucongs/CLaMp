
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch



class MaterialDataset(Dataset):

    
    PROP_NAMES = ['bandgap', 'formation_energy', 'ehull']

    def __init__(self, jsonl_path: str, max_samples: int = None):
        super().__init__()
        self.samples = []   

        print(f"Loading dataset: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.samples.append(record)

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        print(f"Loaded {len(self.samples)} samples")
        self._print_label_stats()

    def _print_label_stats(self):
        for prop in self.PROP_NAMES:
            count = sum(
                1 for s in self.samples if s.get(prop) is not None
            )
            pct = count / len(self.samples) * 100
            print(f"   {prop:<20}: {count:>6} / {len(self.samples)}  ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        record = self.samples[idx]

        
        graph_path = os.path.join(record['graph_dir'], record['pt_file'])
        graph = torch.load(graph_path, weights_only=False)
        graph.jid = record['jid']   

        
        text_emb = torch.load(record['text_emb_path'], weights_only=False)
        
        text_emb = text_emb.squeeze().float()

        
        label_values = []
        label_mask   = []
        for prop in self.PROP_NAMES:
            val = record.get(prop)
            if val is not None:
                label_values.append(float(val))
                label_mask.append(True)
            else:
                label_values.append(0.0)   
                label_mask.append(False)

        labels     = torch.tensor(label_values, dtype=torch.float32)
        label_mask = torch.tensor(label_mask,   dtype=torch.bool)

        return {
            'graph':      graph,
            'text_emb':   text_emb,
            'labels':     labels,
            'label_mask': label_mask,
            'jid':        record['jid'],
        }



def collate_fn(batch: list) -> dict:
    graphs     = [item['graph']      for item in batch]
    text_embs  = [item['text_emb']   for item in batch]
    labels     = [item['labels']     for item in batch]
    label_masks= [item['label_mask'] for item in batch]
    jids       = [item['jid']        for item in batch]

    return {
        'graph':      Batch.from_data_list(graphs),
        'text_emb':   torch.stack(text_embs),            # [B, 2048]
        'labels':     torch.stack(labels),               # [B, 3]
        'label_mask': torch.stack(label_masks),          # [B, 3]
        'jids':       jids,
    }



def compute_label_stats(train_dataset: MaterialDataset):
    all_vals = [[] for _ in range(3)]

    for sample in train_dataset.samples:
        for i, prop in enumerate(MaterialDataset.PROP_NAMES):
            val = sample.get(prop)
            if val is not None:
                all_vals[i].append(float(val))

    mean = np.array([
        np.mean(v) if v else 0.0
        for v in all_vals
    ], dtype=np.float32)

    std = np.array([
        np.std(v)  if v else 1.0
        for v in all_vals
    ], dtype=np.float32)

    
    std[std < 1e-6] = 1.0

    print('Training-label statistics:')
    for i, prop in enumerate(MaterialDataset.PROP_NAMES):
        print(f"   {prop:<20}: mean={mean[i]:.4f}, std={std[i]:.4f}")

    return mean, std



if __name__ == "__main__":
    import sys

    jsonl = sys.argv[1] if len(sys.argv) > 1 \
        else "dataset_split/train/train_metadata.jsonl"

    ds = MaterialDataset(jsonl, max_samples=10)
    sample = ds[0]

    print(f"Processing{sample['jid']}):")
    print(f"   graph.x.shape      : {sample['graph'].x.shape}")
    print(f"   graph.edge_index   : {sample['graph'].edge_index.shape}")
    print(f"   text_emb.shape     : {sample['text_emb'].shape}")
    print(f"   labels             : {sample['labels']}")
    print(f"   label_mask         : {sample['label_mask']}")
