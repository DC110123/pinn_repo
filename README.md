# Universal PINN for Solar Panel I-V Curve Prediction

A physics-informed neural network (PINN) that predicts the current-voltage
(I-V) characteristic curve and Maximum Power Point (MPP) of **any** silicon
solar panel, at arbitrary irradiance and temperature, from its datasheet-level
physical parameters — without needing a separate model per panel.

## Overview

Traditional single-diode models (e.g. De Soto) compute a panel's I-V curve
exactly via `pvlib`, but solving the underlying equation is comparatively
expensive at scale and in real time. This project trains a single neural
network on synthetic curves generated from `pvlib`'s CEC module database,
using each panel's own parameters as input features. The result is one model
that generalizes across panel types and conditions, with a physics-informed
loss term that penalizes non-physical (non-monotonic) I-V curves — and that
runs fast enough to act as a real-time "digital twin" for live panel
monitoring.

**Key components:**
- **Data generation** — synthetic I-V points produced via `pvlib.pvsystem.calcparams_desoto`
  and `i_from_v` across randomized irradiance (200–1100 W/m²) and cell temperature (15–75°C)
- **Physics-informed loss** — penalizes any predicted region where `dI/dV > 0`,
  which is physically impossible for a real I-V curve
- **Bayesian hyperparameter optimization** — [Optuna](https://optuna.org/) (TPE
  sampler) searches network architecture, learning rate schedule, batch size,
  and the physics-loss warmup/ramp schedule, with mid-training pruning of
  unpromising trials
- **Held-out panel validation** — validation panels are excluded from training
  data generation and scaler fitting entirely, so reported metrics reflect
  genuine generalization to unseen panels
- **Three-tier validation suite** (see `simulation/`) — shape checks on
  individual panels, population-level accuracy statistics, and real-time
  digital-twin simulation under dynamic weather conditions

## Repository Structure

```
.
├── Notebook/
│   └── final-dynamic-pinn.ipynb               # End-to-end exploratory / development notebook
├── model/
│   └── universal_pinn_weights.weights.h5      # Trained model weights
├── simulation/
│   ├── Test_A/                                # Individual I-V curve shape validation
│   │   ├── A1.png, A2.png, A3.png
│   │   ├── panel_shape_validation.py
│   │   └── README.md
│   ├── Test_B/                                # Population-level accuracy validation (1000 panels)
│   │   ├── B1.png
│   │   ├── panel_accuracy_validation.py
│   │   └── README.md
│   └── Test_C/                                # Real-time "digital twin" time-series simulation
│       ├── C1.png, C2.png, C3.png
│       ├── panel_simulation.py
│       └── README.md
├── src/
│   ├── __init__.py
│   ├── network_and_train_step.py              # Model architecture, training step, PINN loss, train_model()
│   ├── bayesian_optimization_and_training.py  # Optuna hyperparameter search + final training orchestration
│   ├── data_loader.py                         # CEC database loading, panel train/val split, synthetic data generation
│   ├── eda_cell_count.py                      # Exploratory analysis: cell count distribution
│   └── eda_panel_materials.py                 # Exploratory analysis: panel material/technology breakdown
├── main.py                                    # Entry point
├── pyproject.toml
├── requirements.txt
├── .python-version
├── CITATION.cff
└── README.md
```

## Module Responsibilities

The `src/` package is split so that the training engine has no dependency on
the hyperparameter search, and can be used standalone:

| Module | Responsibility |
|---|---|
| `data_loader.py` | Loads and filters the CEC panel database, performs a panel-level train/val split (so validation panels are never seen during training or scaler fitting), and generates synthetic I-V training points via `pvlib`'s De Soto single-diode model |
| `network_and_train_step.py` | Defines the PINN architecture (`build_model`), the physics-informed training step (`make_train_step`, using `dI/dV` to penalize non-monotonic curves), the adaptive physics-weight schedule, and `train_model()` — trains one model given a fixed set of hyperparameters. Has no dependency on Optuna. |
| `bayesian_optimization_and_training.py` | Defines the Optuna search space, runs the TPE-sampled hyperparameter search with mid-training pruning, and retrains a final model using the winning hyperparameters. Imports `train_model()` from `network_and_train_step.py`. |

This separation means `network_and_train_step.py` can be used directly for a
one-off training run with hand-picked hyperparameters, while
`bayesian_optimization_and_training.py` is only needed when you want Optuna to
search for good hyperparameters automatically.

## Validation Suite

The `simulation/` directory contains three complementary validation tiers,
moving from individual-panel spot checks to population-level statistics to
real-time dynamic behavior:

| Test | Question it answers | Method |
|---|---|---|
| **Test_A** | Does the predicted I-V curve look physically correct for a given panel? | Plots individual predicted curves against known reference points (Isc, Voc, MPP) |
| **Test_B** | Is the model's accuracy consistent and reliable across a large, varied population? | Compares predicted energy yield vs. `pvlib` ground truth across 1,000 sampled panels; reports MAE and error distribution |
| **Test_C** | Can the model track fast-changing conditions in real time, like a live digital twin? | Simulates 10 panels through a 12-hour weather profile (including a rapid "cloud notch" transient) and tracks MPP in real time vs. a traditional iterative solver |

See each `Test_*/README.md` for details specific to that suite.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Usage

### Run the Bayesian hyperparameter search + final training
```bash
python main.py
```
This runs `src/bayesian_optimization_and_training.py`, which searches network
width, learning rate schedule, batch size, and the physics-loss schedule via
Optuna (TPE sampler with mid-training pruning), then retrains a final model
using the best hyperparameters found.

### Train a single model directly (no hyperparameter search)
```python
from src.network_and_train_step import train_model

model, val_metrics = train_model(
    params={...},   # hand-picked hyperparameters
    epochs=150,
    X_tr=X_train, y_tr=y_train,
    X_va=X_val, y_va=y_val,
    verbose=True,
)
```

### Run the validation suite
```bash
python simulation/Test_A/panel_shape_validation.py
python simulation/Test_B/panel_accuracy_validation.py
python simulation/Test_C/panel_simulation.py
```

### Explore interactively
Open `Notebook/final-dynamic-pinn.ipynb` for the full development workflow,
including data generation, training, hyperparameter search, and validation
plots.

## Methodology

1. **Panel database** — `pvlib`'s CEC module database is filtered to valid
   silicon panels with physically sane parameter ranges.
2. **Train/validation split** — performed at the *panel* level (not the
   data-point level) so validation panels are never seen during training,
   scaler fitting, or hyperparameter search.
3. **Model input** — 12 features: normalized irradiance, normalized
   temperature, normalized voltage, 7 scaled panel "DNA" parameters
   (`alpha_sc`, `a_ref`, `I_L_ref`, `I_o_ref` [log], `R_s`, `R_sh_ref` [log],
   `N_s`), and normalized `V_oc`/`I_sc`.
4. **Model output** — predicted current, normalized by the panel's `I_sc`.
5. **Loss** — data MSE plus a physics-informed monotonicity penalty
   (`relu(dI/dV)²`), weighted on an adaptive warmup/ramp schedule.
6. **Hyperparameter optimization** — Optuna TPE sampler with median pruning,
   scored on a combined validation metric (data MSE + weighted monotonicity
   violation) to select configurations that are both accurate and physically
   valid.
7. **Validation** — individual curve shape (Test_A), population-level
   accuracy statistics across 1,000 panels (Test_B), and real-time dynamic
   tracking under a simulated weather profile (Test_C).

## Requirements

See `requirements.txt` / `pyproject.toml`. Core dependencies include:
- `tensorflow`
- `pvlib`
- `optuna`
- `scikit-learn`
- `numpy`, `pandas`, `matplotlib`

## Citation

If you use this project, please cite it — see [`CITATION.cff`](./CITATION.cff).

## License

Add a license (e.g. MIT, Apache-2.0) and reference it here if this repo is
intended to be shared or open-sourced.