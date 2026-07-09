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

# =============================================================================
# 1. DATABASE
# =============================================================================
all_modules = pvlib.pvsystem.retrieve_sam('CECMod').T
df_db = all_modules[all_modules['Technology'].str.contains('Si', na=False)].copy()

dna_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'N_s']
for col in dna_cols + ['V_oc_ref', 'I_sc_ref', 'V_mp_ref', 'I_mp_ref']:
    df_db[col] = pd.to_numeric(df_db[col], errors='coerce')

df_db = df_db[
    (df_db['I_o_ref'] > 1e-15) & (df_db['a_ref'] > 0.5) &
    (df_db['R_sh_ref'] > 10.0) & (df_db['I_sc_ref'] > 0.5) &
    (df_db['V_oc_ref'] > 10.0)
].dropna(subset=dna_cols)

df_db['I_o_ref_log'] = np.log10(df_db['I_o_ref'])
df_db['R_sh_ref_log'] = np.log10(df_db['R_sh_ref'])

fit_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref_log', 'R_s', 'R_sh_ref_log', 'N_s']

# --- Panel-level split BEFORE fitting the scaler, so validation panels are truly held out ---
rng = np.random.default_rng(42)
shuffled_idx = rng.permutation(len(df_db))
n_val_panels = max(20, int(0.15 * len(df_db)))
val_panel_idx = shuffled_idx[:n_val_panels]
train_panel_idx = shuffled_idx[n_val_panels:]

df_train_panels = df_db.iloc[train_panel_idx]
df_val_panels = df_db.iloc[val_panel_idx]

dna_scaler = RobustScaler()
dna_scaler.fit(df_train_panels[fit_cols])  # fit only on training panels, avoid leakage

print(f"Database ready: {len(df_db)} panels "
      f"({len(df_train_panels)} train / {len(df_val_panels)} val).")

# =============================================================================
# 2. DATA GENERATOR
# =============================================================================
def generate_pinn_data(source_df, n_panels=2500, pts=50, seed=None):
    X, y = [], []
    local_rng = np.random.default_rng(seed)
    indices = local_rng.choice(len(source_df), n_panels, replace=True)

    for idx in indices:
        row = source_df.iloc[idx]

        dna_vals = [row[c] for c in fit_cols]
        dna_df = pd.DataFrame([dna_vals], columns=fit_cols)
        dna_norm = dna_scaler.transform(dna_df)[0]

        G = local_rng.uniform(200, 1100, pts)
        T_c = local_rng.uniform(15, 75, pts)

        try:
            IL, Io, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                G, T_c, row['alpha_sc'], row['a_ref'], row['I_L_ref'],
                row['I_o_ref'], row['R_sh_ref'], row['R_s']
            )
            V_norm_axis = np.linspace(0, 1.0, pts)
            I_raw = pvsystem.i_from_v(V_norm_axis * row['V_oc_ref'], IL, Io, Rs, Rsh, nNsVth)

            for j in range(pts):
                if not np.isnan(I_raw[j]):
                    X.append([
                        G[j] / 1000.0, (T_c[j] - 25.0) / 100.0, V_norm_axis[j]
                    ] + dna_norm.tolist() + [
                        row['V_oc_ref'] / 50.0, row['I_sc_ref'] / 10.0
                    ])
                    y.append(max(0, I_raw[j] / row['I_sc_ref']))
        except Exception:
            continue

    return np.array(X, 'float32'), np.array(y, 'float32').reshape(-1, 1)


print("Generating train and validation datasets...")
X_train, y_train = generate_pinn_data(df_train_panels, n_panels=2500, pts=40, seed=1)
X_val, y_val = generate_pinn_data(df_val_panels, n_panels=400, pts=40, seed=2)
print(f"Train points: {len(X_train)} | Val points: {len(X_val)}")