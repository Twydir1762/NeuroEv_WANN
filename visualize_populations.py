from wann_neuroev.evolution import load_ind
from wann_neuroev.visualizer import Visualizer
import os
import re

os.makedirs('ind_checkpoints', exist_ok=True)
inds = sorted(os.listdir('ind_checkpoints'), key=lambda x: int(re.search(r'\d+', x).group()))

for ind in inds:
    best_ind = load_ind(os.path.join('ind_checkpoints', ind))
    vis = Visualizer((1900, 1060), 30, n_radius=30)
    vis.draw(best_ind, column_size=10)

