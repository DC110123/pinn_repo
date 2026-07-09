import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# --- 1. THE TRICK: Connect this subfolder to the project root ---
# This allows you to import from 'src' even though you are in a subfolder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_loader import prepare_pv_database
# No need to import PINNModel if using keras.models.load_model

# --- 2. LOAD SAVED ASSETS ---
print("Loading model and database...")
# Note the path points back to the root 'model' folder
model = keras.models.load_model("../../model/universal_pinn_weights.weights.h5", compile=False)
dna_scaler = joblib.load("../../model/dna_scaler.pkl")
df_db = prepare_pv_database()

# Define the columns used during training
fit_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref_log', 'R_s', 'R_sh_ref_log', 'N_s']

# --- 3. YOUR TEST FUNCTIONS (Copy-pasted from your prompt) ---

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

# --- 4. EXECUTION ---
if __name__ == "__main__":
    print("Starting sampling test.......")
    all_module_names = df_db.index.tolist()
    for i in range(3): # Testing first 3 for brevity
        test_panel(all_module_names[i])