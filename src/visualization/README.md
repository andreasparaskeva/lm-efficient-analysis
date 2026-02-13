# Visualization Module

Lean plotting/data utilities used by this project.

## Kept modules

- `learning_curves.py`: unified BLiMP/GLUE learning curves
- `heatmaps.py`: unified BLiMP/GLUE heatmaps
- `grouped_plots.py`: grouped BLiMP heatmaps and summary heatmaps
- `tokenizer_plots.py`: tokenizer-size comparison curves
- `data_simple.py`: consolidated BLiMP data loading/fetching
- `blimp_tasks.py`, `config.py`, `utils.py`: shared definitions/utilities

## Quick usage

```python
from src.visualization import (
    plot_tokenizer_comparison,
    plot_grouped_summary_heatmap,
)
from src.visualization.learning_curves import load_learning_curve_data, plot_learning_curves
from src.visualization.heatmaps import load_heatmap_data, plot_heatmap
```

These are the functions currently used by `plots.py`.
