
import os
import glob
import json
import numpy as np
import pandas as pd

from config import CFG


def load_all_results():
    pattern = os.path.join(CFG.RESULTS_DIR, "eval_*.json")
    paths = sorted(glob.glob(pattern))
    out = []
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            r = json.load(f)
        r['_filename'] = os.path.basename(p)
        out.append(r)
    return out


def build_table(results):
    rows = []
    for r in results:
        row = {
            'Model':   r.get('model_type', '?'),
            'Tag':     r.get('tag', '?'),
            'N test':  r.get('n_test_samples', '?'),
        }
        reg = r.get('regression', {})
        for prop in CFG.PROP_NAMES:
            row[f'{prop}_MAE']  = reg.get(prop, {}).get('mae')
            row[f'{prop}_R2']   = reg.get(prop, {}).get('r2')
        stab = r.get('stability', {})
        row['stab_F1']  = stab.get('f1')
        row['stab_Acc'] = stab.get('accuracy')
        ret = r.get('retrieval', {})
        row['Recall@1']  = ret.get('recall@1')
        row['Recall@5']  = ret.get('recall@5')
        row['Recall@10'] = ret.get('recall@10')
        rag = r.get('rag', {})
        for prop in CFG.PROP_NAMES:
            row[f'rag_{prop}_MAE'] = rag.get(prop, {}).get('mae')
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(v, digits=4, percent=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    if percent and isinstance(v, (int, float)):
        return f"{v*100:.2f}%"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_markdown(df, path):
    cols = [
        ('Model', 'Model'), ('Tag', 'Tag'), ('N test', 'N'),
        ('bandgap_MAE',          'Bg MAE'),
        ('formation_energy_MAE', 'FE MAE'),
        ('ehull_MAE',            'Eh MAE'),
        ('bandgap_R2',           'Bg R²'),
        ('stab_F1',              'Stab F1'),
        ('Recall@1',             'R@1'),
        ('Recall@5',             'R@5'),
        ('rag_bandgap_MAE',      'RAG Bg'),
    ]
    lines = ['| ' + ' | '.join(label for _, label in cols) + ' |',
             '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        cells = []
        for col, _ in cols:
            v = row.get(col)
            if col in ('Recall@1', 'Recall@5', 'Recall@10'):
                cells.append(fmt(v, percent=True))
            else:
                cells.append(fmt(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    results = load_all_results()
    if not results:
        print(f"Processing{CFG.RESULTS_DIR}/eval_*.json")
        return

    df = build_table(results)

    
    print("\n" + "=" * 95)
    print(f"{len(df)}")
    print("=" * 95)

    fmt_cols = {
        'Bg MAE':  'bandgap_MAE',
        'FE MAE':  'formation_energy_MAE',
        'Eh MAE':  'ehull_MAE',
        'Bg R²':   'bandgap_R2',
        'Stab F1': 'stab_F1',
        'R@1':     'Recall@1',
        'R@5':     'Recall@5',
        'RAG Bg':  'rag_bandgap_MAE',
    }
    header = ['Model', 'Tag'] + list(fmt_cols.keys())
    widths = [14, 12] + [9] * len(fmt_cols)
    print('  '.join(f"{h:<{w}}" for h, w in zip(header, widths)))
    print('-' * sum(widths) + '-' * (len(widths) * 2))

    for _, row in df.iterrows():
        cells = [str(row['Model']), str(row['Tag'])]
        for label, col in fmt_cols.items():
            v = row.get(col)
            if col in ('Recall@1', 'Recall@5', 'Recall@10'):
                cells.append(fmt(v, percent=True))
            else:
                cells.append(fmt(v))
        print('  '.join(f"{c:<{w}}" for c, w in zip(cells, widths)))

    
    md_path = os.path.join(CFG.RESULTS_DIR, 'comparison_table.md')
    csv_path = os.path.join(CFG.RESULTS_DIR, 'comparison_table.csv')
    write_markdown(df, md_path)
    df.to_csv(csv_path, index=False)
    print(f"{md_path}")
    print(f"{csv_path}")


if __name__ == "__main__":
    main()
