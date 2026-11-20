# Can 200 labeled clients be enough to predict credit default?

With the same 200 labeled clients, NVIDIA Kumo Small ranked default risk better than regularized XGBoost in all three training draws (mean ROC-AUC 0.7575 vs. 0.7325). With 800 additional clients, regularized XGBoost reached 0.7771 and beat the 200-label Kumo in two draws.

ROC-AUC chart ([English](results/regularization/kumo-vs-xgboost-regularized-en.png) | [Português](results/regularization/kumo-vs-xgboost-regularizado.png)) · [UCI dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) · [Results by draw](results/regularization/metrics.csv) · [Reproduce the benchmark](#reproduce-the-benchmark)

I-Cheng Yeh's dataset follows 30,000 Taiwanese credit card clients from April through September 2005. The target is default the following month. All three draws share one test set of 6,000 clients from that period. Kumo used 200 context examples throughout.

## Reproduce the benchmark

Clone or download this repository, open its root directory and create an isolated environment with Python 3.11. The experiments use the CPU. The first run needs internet access for the UCI archive and public Kumo Small checkpoint.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m kumo_benchmark all
.\.venv\Scripts\python.exe -m kumo_benchmark verify
```

If Python 3.11 is installed outside the Windows launcher, use its full executable path for the first command.

**Linux or macOS shell**

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m kumo_benchmark all
.venv/bin/python -m kumo_benchmark verify
```

The dataset is verified by SHA-256. New outputs are written to `runs/`. The verification command compares them with the reference CSVs, excluding elapsed time.

<details>
<summary>Full results, holdout design and hyperparameter selection</summary>

| Model | Labels | Mean training ROC-AUC | Mean test ROC-AUC | Mean test average precision |
| --- | ---: | ---: | ---: | ---: |
| Kumo Small | 200 | Not measured | 0.7575 | 0.5197 |
| XGBoost default | 200 | 1.0000 | 0.7175 | 0.4490 |
| XGBoost regularized | 200 | 0.9209 | 0.7325 | 0.4783 |
| XGBoost default | 1,000 | 1.0000 | 0.7424 | 0.4914 |
| XGBoost regularized | 1,000 | 0.8385 | 0.7771 | 0.5539 |

The three nested training draws are the same as in the original experiment. The new stratified test set (6,000 clients, seed 2026) excludes every previously trained or tested client and every profile matching their 23 input features. It represents the same historical period. Compare models within this table since the original table uses different test clients.

Within each training draw and budget, eight regularized XGBoost configurations competed on mean validation ROC-AUC in three stratified folds, grouping identical feature profiles. The grid combined depths 2 and 3, minimum child weights 1 and 5, and L2 penalties 5 and 20, with learning rate 0.05. Each winner was refit on its full training sample. Kumo used its original one-estimator context protocol.

[Experiment code](kumo_benchmark/regularization.py) · [Validation scores and selected parameters](results/regularization/selection.csv)

</details>

<details>
<summary>Original benchmark with default XGBoost</summary>

| Model | Labeled clients | Mean ROC-AUC | Mean average precision |
| --- | ---: | ---: | ---: |
| Kumo Tabular Small | 200 | 0.736 | 0.499 |
| XGBoost | 200 | 0.695 | 0.430 |
| XGBoost | 1,000 | 0.723 | 0.471 |

In the original experiment, Kumo with 200 labels beat default XGBoost with 1,000 in each draw. The paired ROC-AUC advantage ranged from 0.004 to 0.026. This prompted the regularization follow-up on a fresh holdout.

Baseline ROC-AUC chart ([English](results/baseline/kumo-vs-xgboost-default-en.png) | [Português](results/baseline/kumo-vs-xgboost-padrao.png))

[Data source: Default of Credit Card Clients (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

A stratified split with seed 42 holds out 6,000 clients. Each of three seeds (12, 36 and 53) draws 1,000 stratified training rows and a nested subset of 200. Models share the same test clients. The client ID and target are excluded from the 23 numeric input columns. Both models use the original numeric codes. Kumo caches its context with one estimator and predicts in batches of 125. XGBoost uses package defaults with four threads and a fixed seed. There is no hyperparameter search.

Elapsed wall time covers fitting or context preparation and predicting all test rows, excluding disk reads, Kumo checkpoint loading and evaluation. Measurements are single runs on an Intel Core i5-1135G7. Repetitions vary training clients while reusing one test set. The comparison is descriptive and specific to this dataset and these settings.

[UCI dataset and attribution](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) · [NVIDIA Kumo Tabular](https://docs.nvidia.com/sdgm/structured-data-models/models/kumo-tabular)

[Original metrics](results/baseline/metrics.csv) · [Experiment code](kumo_benchmark/baseline.py)

</details>

<details>
<summary>Command reference and repository layout</summary>

| Command after `python -m kumo_benchmark` | Purpose | Output |
| --- | --- | --- |
| `data` | Fetch or verify the UCI archive | `data/uci.zip` |
| `baseline` | Run Kumo and default XGBoost on the original split | `runs/baseline/` |
| `regularization` | Select regularization on training folds and score the fresh holdout | `runs/regularization/` |
| `all` | Run both experiments in order | Both run directories |
| `verify` | Compare both local runs against the reference CSVs (absolute tolerance 0.000001, timings excluded) | Terminal report |
| `figures` | Redraw the English and Portuguese reference PNGs from their saved CSVs | `results/baseline/` and `results/regularization/` |

**Repository layout**

| Location | Contents |
| --- | --- |
| [kumo_benchmark/](kumo_benchmark/) | Command line, verified data download, experiment implementations and figures |
| [results/baseline/](results/baseline/) | Nine original measurements and the original chart |
| [results/regularization/](results/regularization/) | Fifteen follow-up measurements, 48 validation scores and the main chart |
| [requirements.txt](requirements.txt) | Pinned Python dependencies |
| `runs/` | Metrics and figures generated on your machine. Repeating an experiment replaces its local outputs. |
| `data/`, `.hf/` | Local dataset and model cache |

</details>
