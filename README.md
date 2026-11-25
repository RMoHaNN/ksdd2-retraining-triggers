# Evaluating Retraining Triggers and Model Acceptance Rules in Industrial MLOps

This repository contains the code and the recorded run behind the paper.
One command loads the run into MLflow and redraws the figures.

## Paper

Mohan Rajashekarappa, Ebru Turanoglu Bekar, Jon Bokrantz and Anders Skoogh. IFAC AMEST 2026.
The citation below will be updated after publication.

```bibtex
@inproceedings{rajashekarappa2026retraining,
  title = {Evaluating Retraining Triggers and Model Acceptance Rules in Industrial MLOps},
  author = {Rajashekarappa, Mohan and Turanoglu Bekar, Ebru and
            Bokrantz, Jon and Skoogh, Anders},
  booktitle = {IFAC AMEST},
  year = {2026},
  note = {Publication details to be updated}
}
```

## Data

[KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) is licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Please cite Božič, Tabernik and Skočaj (2021), *Mixed supervision for surface-defect detection:
from weakly to fully supervised learning*, Computers in Industry, 129, 103459,
[doi:10.1016/j.compind.2021.103459](https://doi.org/10.1016/j.compind.2021.103459).
The dataset is only needed for the defect-crop figure. Keep its official folders and masks:

```text
data/KolektorSDD2/
  train/  10000.png, 10000_GT.png, ...
  test/   20000.png, 20000_GT.png, ...
```

## Running

Use Python 3.10 in a virtual environment.

```console
pip install -r requirements.txt
python run.py
```

The command downloads the [recorded paper run](https://github.com/RMoHaNN/ksdd2-retraining-triggers/releases/tag/paper-run),
loads MLflow, saves PNG and PDF figures in `figures/`, and opens http://127.0.0.1:5000.
Local files are kept in `results/`. Press Ctrl+C to stop the dashboard.

| Flag | Purpose |
|---|---|
| `--bundle PATH` | Use a local `paper_run.zip`. |
| `--data PATH` | Set the KolektorSDD2 folder; default: `data/KolektorSDD2`. |
| `--no-ui` | Load the results and draw the figures without opening the dashboard. |
| `--force` | Rebuild the local MLflow store. |

## What you will see in MLflow

The `training` experiment contains the six plain and managed runs, with test metrics and ten
epochs of loss and validation AP per run. `condition-test` contains the seven condition summaries.
`replay` contains the five settings, PB1 to PB8 metrics and a child run for each candidate decision,
with its gate scores. The relevant figures are attached to these runs. The initial model, v1,
is registered as `ksdd2-classifier` with alias `paper`.

## Results

Training means across seeds 0, 1 and 2:

| Regime | AP (%) | Recall (%) |
|---|---:|---:|
| Plain | 92.9 | 84.2 |
| Managed | 95.4 | 91.8 |

Table 1. Initial model under seven camera conditions. Flags are the mean out of 13 features.

| Condition | Flags | AP (%) | Recall (%) |
|---|---:|---:|---:|
| Nominal | 0.2 | 95.9 | 92.7 |
| Lighting change | 9.8 | 95.4 | 92.7 |
| Darker scene | 13.0 | 86.5 | 80.9 |
| Lighting + noise | 12.6 | 90.4 | 88.2 |
| JPEG compression | 12.1 | 86.3 | 76.4 |
| Mild defocus | 12.1 | 76.3 | 54.5 |
| Strong defocus | 12.0 | 52.7 | 32.7 |

Table 2. Eight-batch replay with 110 defects. G2 was exploratory.

| Trigger | Gate | Retrains | Rejected | Promoted | Missed defects | False alarms | GPU min |
|---|---|---:|---:|---:|---:|---:|---:|
| None | - | 0 | 0 | 0 | 27 | 11 | 0.0 |
| Data drift | G1 | 5 | 4 | 1 | 25 | 11 | 29.8 |
| Quality alert | G1 | 3 | 2 | 1 | 25 | 11 | 17.9 |
| Data drift | G2 | 5 | 1 | 4 | 16 | 18 | 29.9 |
| Quality alert | G2 | 2 | 0 | 2 | 16 | 14 | 11.9 |

## Code layout

- `run.py`: bundle loading, summary tables, figures and dashboard.
- `config.yaml`: image, monitor, gate and figure settings.
- `requirements.txt`: pinned Python dependencies.
- `src/data.py`: labels, validation split and camera conditions.
- `src/model.py`: ResNet-18 loading and prediction.
- `src/monitor.py`: feature drift and quality alerts.
- `src/experiments.py`: gate and replay decision rules, and MLflow records.
- `src/figures.py`: workflow, architecture, defect crops, replay and gate figures.

The training script is not included. `src/data.py`, `src/model.py` and `src/monitor.py` hold the
data handling, model and monitor used in the study; `run.py` reads their recorded outputs from the release.

## Licence

The code is licensed under MIT. The dataset and model weights fall under the dataset's
[CC BY-NC-SA 4.0 licence](https://www.vicos.si/resources/kolektorsdd2/) for non-commercial research use.
