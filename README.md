# t2v-eval

**t2v-eval** is a Python library for evaluating *Text-to-Visualisation (T2V)* models. Given a Python script that generates a chart, it logs the plotted data by monkey-patching matplotlib, seaborn, and geopandas, then compares the logged data against a ground-truth reference to report two metrics:

| Metric | Range | Description |
|---|---|---|
| **Displayed Data Similarity** | [0, 1] | How closely the predicted chart's data matches the ground truth (Hungarian-algorithm optimal matching) |
| **Chart Type Accuracy** | True / False | Whether the predicted chart uses a superset of the chart types present in the ground truth |

---

## How it works

```
Your script                 T2V logging layer              Evaluation
──────────────             ─────────────────────          ─────────────────────
ax.scatter(x, y)  ──────►  monkey-patch captures x, y ──► JSON log files
ax.plot(x, y)     ──────►  writes to log directory    ──► calculate_similarity_score()
sns.barplot(...)  ──────►  (no code changes needed)   ──► calculate_chart_type_accuracy()
gdf.plot(...)     ──────►                             ──► score in [0, 1]
```

The logging layer captures **what was actually rendered** — not what was passed as arguments — so transforms, aggregations, and computed statistics (e.g. KDE curves, box-plot quantiles, regression lines) are all recorded faithfully.

---

## Installation

```bash
pip install -e .
```

**Requires Python 3.10 – 3.12.**  
Core dependencies: `matplotlib`, `seaborn`, `geopandas`, `scipy`, `numpy`, `pandas`.

---

## Quick start

### 1. Activate logging inside a script

Add one line at the top of the visualisation block:

```python
# your_script.py

##START VISUALISATION CODE
from t2v_eval.logging import activate_t2v_logging
activate_t2v_logging("/path/to/log_dir")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [4, 5, 6])
plt.show()
```

Or run an existing script programmatically (the library injects the activation line automatically):

```python
from t2v_eval.logging import execute_and_log_data_py_path, LoggingState

state, log_dir = execute_and_log_data_py_path(
    py_path   = "your_script.py",
    py_env    = "python3",
    timeout   = 300,
    use_cache = False,
    verbose   = False,
)
print(state, log_dir)   # LoggingState.HAVING_DATA  /path/to/log_dir
```

### 2. Compute metrics

```python
from t2v_eval.evaluation import (
    calculate_similarity_score,
    calculate_chart_type_accuracy,
)

similarity = calculate_similarity_score(
    pred_log_dir = "/path/to/pred_log",
    gt_log_dir   = "/path/to/gt_log",
)
print(f"Similarity: {similarity:.4f}")   # e.g. 0.8732

correct = calculate_chart_type_accuracy(
    pred_log_dir = "/path/to/pred_log",
    gt_log_dir   = "/path/to/gt_log",
)
print(f"Chart type correct: {correct}")   # True / False
```

---

## Supported visualisation libraries

| Library | Chart types logged |
|---|---|
| **matplotlib** | plot, scatter, bar, barh, hist, pie, boxplot, violin, stem, stairs, fill_between, fill_betweenx, errorbar, hexbin, hist2d, imshow, pcolormesh, contour, contourf, scatter3d, bar3d, plot_surface, eventplot, and more |
| **seaborn** | relplot, lmplot, catplot, displot, FacetGrid, PairGrid, pairplot, and all underlying axes-level functions |
| **geopandas** | GeoDataFrame.plot (choropleth / heatmap) |
| **pandas** | DataFrame.plot (line, bar, scatter, box, hist, kde, pie, hexbin, area) |

---

## API reference

### Logging

```python
from t2v_eval.logging import activate_t2v_logging, execute_and_log_data_py_path
```

| Function | Description |
|---|---|
| `activate_t2v_logging(log_dir)` | Activate logging and write JSON files to `log_dir`. Call this once at the start of the visualisation block. |
| `execute_and_log_data_py_path(py_path, ...)` | Run a `.py` file in a subprocess and capture its visualisation data automatically. Returns `(LoggingState, log_dir)`. |

### Evaluation

```python
from t2v_eval.evaluation import (
    calculate_similarity_score,
    calculate_chart_type_accuracy,
    calculate_accessibility_score,
)
```

| Function | Returns | Description |
|---|---|---|
| `calculate_similarity_score(pred_log_dir, gt_log_dir, ...)` | `float` in [0, 1] | Optimal matching between predicted and ground-truth data series using the Hungarian algorithm. 1.0 = perfect match. |
| `calculate_chart_type_accuracy(pred_log_dir, gt_log_dir, ...)` | `bool` | `True` if the predicted chart's type set is a superset of the ground truth's type set. |
| `calculate_accessibility_score(image_path, ...)` | `float` | WCAG-based accessibility score (colour contrast + word spacing). |

### Key parameters for `calculate_similarity_score`

| Parameter | Default | Description |
|---|---|---|
| `down_sampling` | `True` | Sub-sample long series before comparison |
| `down_sampling_max_length` | `5000` | Max points per series after downsampling |
| `use_cache` | `False` | Return cached result from `evaluation.json` if it exists |
| `save_results` | `False` | Write result to `evaluation.json` in `pred_log_dir` |
| `verbose` | `False` | Print debug information |

---

## Log directory structure

Each call to `activate_t2v_logging(log_dir)` creates a directory like:

```
log_dir/
  variables.json                          ← internal T2V_ISLOG flag
  {fig_id}_{ax_id}_{timestamp}.json       ← one file per plot call
  {fig_id}_{ax_id}_{timestamp}.json
  ...
  evaluation.json                         ← written by calculate_similarity_score (if save_results=True)
```

Each JSON file contains:

```json
{
  "t2v_eval_version": "0.2.0",
  "plt_func": "scatter",
  "args": [],
  "kargs": { "ax_col": 0, "ax_row": 0 },
  "data_series": [
    { "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "s": [5, 5, 5] }
  ]
}
```

---

## Typical workflow for Batch evaluation

### Step 1 — Execute scripts and capture logs

Run prediction and ground-truth scripts, inject T2V logging, and record outcomes:

```bash
python t2v_batch_log.py \
  --experiment_dir /data/experiment_name \
  --data_rel_dir   batch_01 \
  --py_env         python3 \
  --timeout        300 \
  --use_cache
```

This discovers all `*.vis-request.txt` files under
`experiment_dir/batch_01/*/t2v_pred/`, runs each paired prediction + GT
script, and saves a per-sample outcome table to:

```
logs/t2v_logging/batch_01_logging.csv
```

Expected directory layout:

```
experiment_dir/
  batch_01/
    sample_001/
      t2v_pred/  sample_001.py          ← prediction script
                 sample_001.vis-request.txt
                 sample_001_log/        ← written by t2v_batch_log.py
      t2v_gt/    sample_001.py          ← ground-truth script
                 sample_001_log/        ← written by t2v_batch_log.py
    sample_002/
      ...
```

Optional flags:

| Flag | Description |
|---|---|
| `--use_cache` | Skip scripts whose log directory already exists |
| `--replot` | Render a merged PNG overlay after each script |
| `--verbose` | Print subprocess stdout / stderr |

### Step 2 — Compute metrics

```bash
# Both metrics (default)
python t2v_batch_evaluate.py \
  --experiment_dir /data/experiment_name \
  --data_rel_dir   batch_01 \
  --use_cache

# Similarity only
python t2v_batch_evaluate.py ... --metric similarity

# Chart type accuracy only
python t2v_batch_evaluate.py ... --metric chart_type
```

Results are written to:

```
logs/metrics-scores/ddata_similarity/batch_01_sim_score.csv
logs/metrics-scores/chart_type_accuracy/batch_01_chart_type_accuracy.csv
```

---

## Running the test suite

This is to ensure the logging layer captures data correctly across a wide range of chart types and libraries. Tests are located in `tests/t2v_logging/` and can be run with:

```bash
pytest tests/t2v_logging/ -v
```

195 tests covering matplotlib (pairwise, distribution, gridded, 3D), pandas, seaborn (relational, distribution, categorical, regression, axis grids), and geopandas.

To run a specific subset:

```bash
pytest tests/t2v_logging/test_matplotlib_pairwise.py -v
pytest tests/t2v_logging/test_seaborn_relational.py  -v
pytest tests/t2v_logging/test_geopandas.py            -v
```

---

## Docker build

```bash
docker buildx build \
  --platform linux/amd64 \
  -t your-image-name:tag \
  --push \
  .
```

---

## License

See [LICENSE](LICENSE).
