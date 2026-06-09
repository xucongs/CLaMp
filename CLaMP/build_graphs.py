
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
from mendeleev import element
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

warnings.filterwarnings("ignore")


CIF_DIR        = "cif_files_jarvis"          
PROPS_CSV      = "jarvis_properties.csv"     
OUTPUT_DIR     = "pyg_graphs"                
CUTOFF_RADIUS  = 8.0                         
MAX_ATOMS      = 500                         
NUM_WORKERS    = max(1, cpu_count() - 2)     



def _build_element_cache():
    def block_to_int(b):
        return {'s': 0, 'p': 1, 'd': 2, 'f': 3}.get(b, 4)

    cache = {}
    print('Building element feature cache')
    for z in range(1, 101):
        try:
            e = element(z)
            cache[z] = [
                e.atomic_number,
                getattr(e, 'period', 0) or 0,
                getattr(e, 'group_id', 0) or 0,
                getattr(e, 'mass', 0) or 0,
                getattr(e, 'atomic_radius', 0) or 0,
                getattr(e, 'covalent_radius_pyykko', 0) or 0,
                getattr(e, 'en_pauling', 0) or 0,
                getattr(e, 'ionenergies', {}).get(1, 0) or 0,
                getattr(e, 'electron_affinity', 0) or 0,
                block_to_int(getattr(e, 'block', 's')),
            ]
        except Exception:
            cache[z] = [0.0] * 10
    print('Building element feature cache')
    return cache

ELEMENT_CACHE = _build_element_cache()



def cif_to_graph(jid: str, cif_path: str, radius: float = 8.0):
    structure = Structure.from_file(cif_path)

    if len(structure) > MAX_ATOMS:
        raise ValueError(f"Processing{len(structure)}Processing{MAX_ATOMS}")

    
    node_feats = [ELEMENT_CACHE.get(site.specie.Z, [0.0]*10)
                  for site in structure]
    x   = torch.tensor(node_feats, dtype=torch.float32)
    z   = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float32)

    
    lattice = torch.tensor(
        structure.lattice.matrix, dtype=torch.float32
    ).unsqueeze(0)   # [1, 3, 3]

    
    center_idx, neighbor_idx, image_offsets, _ = \
        structure.get_neighbor_list(r=radius)

    if len(center_idx) == 0:
        
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_vec   = torch.empty((0, 3), dtype=torch.float32)
        edge_shift = torch.empty((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor(
            np.stack([center_idx, neighbor_idx], axis=0), dtype=torch.long
        )
        edge_shift = torch.tensor(image_offsets, dtype=torch.float32)

        
        pos_i      = pos[edge_index[0]]
        pos_j      = pos[edge_index[1]]
        offsets_cart = torch.matmul(edge_shift, lattice.squeeze(0))
        edge_vec   = pos_j - pos_i + offsets_cart

    return Data(
        x=x,
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_shift=edge_shift,
        lattice=lattice,
        jid=jid,
        num_atoms=len(structure),
    )



def _worker(args, cif_dir: str, radius: float, output_dir: str):
    processed_idx, jid = args
    cif_path = os.path.join(cif_dir, f"{jid}.cif")

    if not os.path.exists(cif_path):
        return {'jid': jid, 'status': 'missing_cif'}

    try:
        data = cif_to_graph(jid, cif_path, radius)
        save_path = os.path.join(output_dir, f"data_{processed_idx}.pt")
        torch.save(data, save_path)
        return {
            'jid': jid,
            'processed_idx': processed_idx,
            'pt_file': f"data_{processed_idx}.pt",
            'status': 'success'
        }
    except Exception as e:
        return {'jid': jid, 'status': 'failed', 'error': str(e)}



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    print(f"Reading property table: {PROPS_CSV}")
    df = pd.read_csv(PROPS_CSV, sep=',', dtype={'jid': str}, low_memory=False)
    df = df.drop_duplicates(subset=['jid']).reset_index(drop=True)
    all_jids = df['jid'].tolist()
    print(f"Found {len(all_jids)} entries")

    
    tasks = list(enumerate(all_jids))

    
    worker_fn = partial(
        _worker,
        cif_dir=CIF_DIR,
        radius=CUTOFF_RADIUS,
        output_dir=OUTPUT_DIR
    )

    
    print(f"Processing CIF files with {NUM_WORKERS} workers")
    results = []
    with Pool(processes=NUM_WORKERS) as pool:
        for res in tqdm(
            pool.imap_unordered(worker_fn, tasks),
            total=len(tasks),
            desc='Processing'
        ):
            results.append(res)

    
    success   = [r for r in results if r['status'] == 'success']
    failed    = [r for r in results if r['status'] == 'failed']
    missing   = [r for r in results if r['status'] == 'missing_cif']

    print(f"Succeeded: {len(success)}")
    print(f"Failed: {len(failed)}")
    print(f"Missing CIF files: {len(missing)}")

    
    map_df = pd.DataFrame([
        {'jid': r['jid'], 'processed_idx': r['processed_idx'], 'pt_file': r['pt_file']}
        for r in success
    ]).sort_values('processed_idx').reset_index(drop=True)

    map_path = os.path.join(OUTPUT_DIR, 'processed_map.csv')
    map_df.to_csv(map_path, index=False)
    print(f"Saved processed map: {map_path}")
    print(f"Valid graph files: {len(map_df)}")

    
    if failed:
        print('Building element feature cache')
        for r in failed[:5]:
            print(f"   {r['jid']}: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
