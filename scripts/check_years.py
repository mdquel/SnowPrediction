import pandas as pd, numpy as np, os

d = pd.read_csv('dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv')
md = 'dataset_v4_ms_sx200/masks'

for y, g in d.groupby('year'):
    vals = []
    for t in g['tile_id'].sample(min(150, len(g)), random_state=0):
        m = np.load(os.path.join(md, t)).ravel()
        m = m[np.isfinite(m) & (m > 0.01)]
        if m.size:
            vals.append(m.mean())
    print(f'{y}: HS media = {np.mean(vals):.3f} m   (n={len(vals)} tiles)')