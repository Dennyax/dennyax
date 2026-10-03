import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# 12 цветов: [стрелка вниз] + 10 интервалов (от синего к красному) + [стрелка вверх]
colors_01 = [
    '#19238c',  # < -1.0  (стрелка)
    '#2936b5',  # -1.0 … -0.8
    '#546be0',  # -0.8 … -0.6
    '#7e9cf1',  # -0.6 … -0.4
    '#aac2f5',  # -0.4 … -0.2
    '#ccd8e9',  # -0.2 …  0.0
    '#e7d5c8',  #  0.0 …  0.2
    '#edb59a',  #  0.2 …  0.4
    '#e68a6a',  #  0.4 …  0.6
    '#c95240',  #  0.6 …  0.8
    '#a1201a',  #  0.8 …  1.0
    '#7f190f',  # > 1.0   (стрелка)
]
cmap_01 = LinearSegmentedColormap.from_list('red_blue', colors_01, 12)

# границы уровней: -1.0 … 1.0 с шагом 0.2 (10 интервалов)
levels = np.round(np.arange(-1.0, 1.0 + 0.2, 0.2), 1)
norm_01 = BoundaryNorm(levels, ncolors=cmap_01.N, extend='both')


def colorbar_01(path='cmap_01_preview.png'):
    fig, ax = plt.subplots(figsize=(1.0, 3.2))
    fig.subplots_adjust(left=0.05, right=0.35, top=0.95, bottom=0.05)
    sm = plt.cm.ScalarMappable(norm=norm_01, cmap=cmap_01)
    cb = fig.colorbar(sm, cax=ax, extend='both', ticks=[-0.8, -0.4, 0.0, 0.4, 0.8])
    cb.ax.tick_params(labelsize=11)
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    colorbar_01()
