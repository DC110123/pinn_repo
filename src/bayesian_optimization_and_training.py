!pip install pvlib optuna

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pvlib
from pvlib import pvsystem
from sklearn.preprocessing import RobustScaler
import optuna
from optuna.samplers import TPESampler

pd.options.mode.chained_assignment = None


def phys_weight_for_epoch(epoch, warmup_end, ramp_end, final_weight):
    """Adaptive physics weight schedule, now tunable instead of hardcoded."""
    if epoch < warmup_end:
        return 0.0
    elif epoch < ramp_end:
        return final_weight * 0.2
    else:
        return final_weight


def _compute_val_metrics(model, X_va, y_va, monotonic_weight=0.05):
    """Returns (data_mse, monotonic_violation, combined_score) on validation data.

    data_mse: plain MSE between predicted and true normalized current.
    monotonic_violation: mean squared magnitude of any positive dI/dV slope
        (should be ~0 for a physically valid I-V curve; I always decreases
        with V, so dI/dV should be <= 0 everywhere).
    combined_score: what Optuna actually optimizes/prunes on, so a model
        that fits well but has visible non-monotonic wiggles is not treated
        as "the best" over one that fits almost as well with a clean curve.
    """
    X_va_tf = tf.convert_to_tensor(X_va, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_va_tf)
        y_pred = model(X_va_tf, training=False)
    grads = tape.gradient(y_pred, X_va_tf)
    dI_dV = grads[:, 2:3]  # voltage is feature index 2

    monotonic_violation = float(tf.reduce_mean(tf.square(tf.nn.relu(dI_dV))))
    data_mse = float(np.mean(np.square(y_va - y_pred.numpy())))
    combined_score = data_mse + monotonic_weight * monotonic_violation

    return data_mse, monotonic_violation, combined_score


# How heavily monotonicity violations count against the validation score.
# Scaled so that a "clean" model (near-zero violation) barely differs from
# pure data MSE, but a model with visible non-monotonic wiggles is penalized.
VAL_MONOTONIC_WEIGHT = 0.05


def train_model(params, epochs, X_tr, y_tr, X_va=None, y_va=None, verbose=False,
                 trial=None, report_every=5):
    """Builds, trains, and returns (model, val_metrics).

    val_metrics is a dict {'data_mse', 'monotonic_violation', 'combined_score'}
    (or None if no validation data is provided).

    If `trial` (an optuna.Trial) is passed, the combined score is computed and
    reported every `report_every` epochs so Optuna's pruner can actually cut
    unpromising trials short instead of only seeing one score at the very end.
    """
    model = build_model(params['units1'], params['units2'], params['units3'])

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params['learning_rate'],
        decay_steps=params['decay_steps'],
        decay_rate=params['decay_rate'],
        staircase=True)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    train_step = make_train_step(model, optimizer)

    dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)) \
        .shuffle(100000).batch(params['batch_size'])

    for epoch in range(epochs):
        p_w = phys_weight_for_epoch(
            epoch, params['warmup_end'], params['ramp_end'], params['phys_weight_final']
        )
        l_d, l_p = [], []
        for xb, yb in dataset:
            d, p = train_step(xb, yb, tf.constant(p_w, dtype=tf.float32))
            l_d.append(d); l_p.append(p)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:<4} | data_loss={np.mean(l_d):.6f} "
                  f"| phys_loss={np.mean(l_p):.6f} | phys_w={p_w:.3f}")

        # --- Mid-training pruning check ---
        is_last_epoch = (epoch == epochs - 1)
        if trial is not None and X_va is not None:
            if ((epoch + 1) % report_every == 0) or is_last_epoch:
                _, _, interim_score = _compute_val_metrics(
                    model, X_va, y_va, monotonic_weight=VAL_MONOTONIC_WEIGHT
                )
                trial.report(interim_score, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch+1} (combined_score={interim_score:.6f})"
                    )

    val_metrics = None
    if X_va is not None:
        data_mse, monotonic_violation, combined_score = _compute_val_metrics(
            model, X_va, y_va, monotonic_weight=VAL_MONOTONIC_WEIGHT
        )
        val_metrics = {
            'data_mse': data_mse,
            'monotonic_violation': monotonic_violation,
            'combined_score': combined_score,
        }
        if verbose:
            print(f"  val: data_mse={data_mse:.6f} | monotonic_violation={monotonic_violation:.6f} "
                  f"| combined_score={combined_score:.6f}")

    return model, val_metrics

# =============================================================================
# 4. BAYESIAN OPTIMIZATION (Optuna, TPE sampler)
# =============================================================================
SEARCH_EPOCHS = 35   # short budget per trial to keep the search cheap
N_TRIALS = 25         # number of Bayesian-optimization trials

def objective(trial):
    params = {
        'units1': trial.suggest_int('units1', 64, 512, step=32),          # was 384 max -> widened
        'units2': trial.suggest_int('units2', 32, 256, step=32),
        'units3': trial.suggest_int('units3', 16, 128, step=16),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),  # was 5e-3 max -> widened
        'decay_rate': trial.suggest_float('decay_rate', 0.85, 0.99),
        'decay_steps': trial.suggest_int('decay_steps', 100, 2000, step=100),  # was 300 min -> widened
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096]),
        'phys_weight_final': trial.suggest_float('phys_weight_final', 0.005, 0.4, log=True),  # was 0.2 max -> widened
        'warmup_end': trial.suggest_int('warmup_end', 5, 20),
        'ramp_end': trial.suggest_int('ramp_end', 20, 40),
    }
    # keep the schedule sane: ramp_end must exceed warmup_end
    if params['ramp_end'] <= params['warmup_end']:
        params['ramp_end'] = params['warmup_end'] + 5

    _, val_metrics = train_model(
        params, epochs=SEARCH_EPOCHS,
        X_tr=X_train, y_tr=y_train, X_va=X_val, y_va=y_val,
        verbose=False,
        trial=trial,          # enables mid-training reporting + pruning
        report_every=5,       # check every 5 epochs instead of only at the end
    )

    return val_metrics['combined_score']


print("\nStarting Bayesian hyperparameter search (Optuna / TPE)...")
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # don't prune until 5 trials have fully completed
        n_warmup_steps=10,     # don't prune before epoch 10 within a trial
        interval_steps=5,      # matches report_every in train_model
    )
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\nBest trial:")
print(f"  Validation combined score: {study.best_value:.6f}  "
      f"(= data_mse + {VAL_MONOTONIC_WEIGHT} * monotonic_violation)")
print("  Params:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

# =============================================================================
# 5. FINAL TRAINING WITH BEST HYPERPARAMETERS (full epoch budget)
# =============================================================================
best_params = dict(study.best_params)
if best_params['ramp_end'] <= best_params['warmup_end']:
    best_params['ramp_end'] = best_params['warmup_end'] + 5

print("\nRetraining final model with best hyperparameters (150 epochs)...")
model, final_val_metrics = train_model(
    best_params, epochs=150,
    X_tr=X_train, y_tr=y_train, X_va=X_val, y_va=y_val,
    verbose=True
)
print(f"\nFinal held-out validation:")
print(f"  data_mse             = {final_val_metrics['data_mse']:.6f}")
print(f"  monotonic_violation  = {final_val_metrics['monotonic_violation']:.6f}")
print(f"  combined_score       = {final_val_metrics['combined_score']:.6f}")

# =============================================================================
# 6. TEST / VISUALIZE
# =============================================================================
def test_panel(panel_name=None):
    if panel_name is not None and panel_name in df_db.index:
        row = df_db.loc[panel_name]
    else:
        row = df_db.sample(1, random_state=None).iloc[0]
        panel_name = row.name

    dna_vals = [row[c] for c in fit_cols]
    dna_df = pd.DataFrame([dna_vals], columns=fit_cols)
    dna_norm = dna_scaler.transform(dna_df)[0]

    v_sweep = np.linspace(0, 1.0, 100)
    X_test = []
    for v in v_sweep:
        X_test.append([1.0, 0.0, v] + dna_norm.tolist() +
                       [row['V_oc_ref'] / 50.0, row['I_sc_ref'] / 10.0])

    y_pred = model.predict(np.array(X_test, dtype='float32'), verbose=0)
    I_pred = y_pred.flatten() * row['I_sc_ref']

    plt.figure(figsize=(10, 6))
    plt.plot(v_sweep * row['V_oc_ref'], I_pred, label='Universal PINN (Bayes-tuned)',
              color='blue', lw=3)
    plt.plot(row['V_mp_ref'], row['I_mp_ref'], 'ro', label='True MPP')
    plt.axhline(row['I_sc_ref'], color='gray', ls=':')
    plt.axvline(row['V_oc_ref'], color='gray', ls=':')
    plt.title(f"{panel_name}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()


print("\nPlotting results on a random held-out panel...")
test_panel("Ablytek_5MN6C180_A0")  # picks a random panel; pass a specific index name if you want a fixed one

# Optional: inspect the optimization history
try:
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.show()
except Exception as e:
    print(f"(Skipping Optuna plots: {e})")