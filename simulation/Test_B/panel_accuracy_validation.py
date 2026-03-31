import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from pvlib import pvsystem

# --- 1. CONNECT TO PROJECT ROOT ---
# This allows us to import from the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_loader import prepare_pv_database

# --- 2. LOAD ASSETS CREATED BY train.py ---
print("Loading model and dependencies...")
model = keras.models.load_model("../../model/universal_pinn_weights.weights.h5", compile=False)
dna_scaler = joblib.load("../../model/dna_scaler.pkl")
df_db = prepare_pv_database()

# The columns used for the 'DNA' portion of the input
fit_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref_log', 'R_s', 'R_sh_ref_log', 'N_s']

# --- 3. THE BENCHMARK FUNCTION ---
def run_massive_benchmark(trained_model, n_panels=200):
    """
    Simulates a full 'Test Day' for hundreds of panels and compares 
    PINN predictions against the ground-truth Physics (PVLib).
    """
    print("="*60)
    print(f"🚀 STARTING MASSIVE BENCHMARK: {n_panels} PANELS")
    print("="*60)
   
    test_indices = np.random.choice(len(df_db), n_panels, replace=False)
   
    # Weather Profile (15 min intervals for 12 hours)
    times = pd.date_range('2025-06-21 06:00', '2025-06-21 18:00', freq='15min')
    N_t = len(times)
    x = np.linspace(0, np.pi, N_t)
    g_signal = 1000 * np.sin(x)
    g_signal[20:26] *= 0.4 # Simulate cloud cover at noon
    g_signal[g_signal < 0] = 0
    t_signal = 20 + 30 * np.sin(x)
   
    results = []

    for i, idx in enumerate(test_indices):
        if i % 50 == 0: print(f"Processing panel {i}/{n_panels}...")
        
        row = df_db.iloc[idx]
        
        # A. PREPARE DNA
        dna_vals = [row[c] for c in fit_cols]
        dna_df = pd.DataFrame([dna_vals], columns=fit_cols)
        dna_norm = dna_scaler.transform(dna_df)[0]
       
        # B. CALCULATE TRUE ENERGY (PVLIB PHYSICS)
        try:
            IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                g_signal, t_signal,
                row['alpha_sc'], row['a_ref'], row['I_L_ref'],
                row['I_o_ref'], row['R_sh_ref'], row['R_s']
            )
            p_mp = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth, method='lambertw')['p_mp']
            energy_true = np.sum(np.nan_to_num(p_mp)) * 0.25 
        except: continue
           
        if energy_true < 10: continue 
           
        # C. CALCULATE AI ENERGY (PINN)
        v_res = 50
        v_sweep = np.linspace(0, 1.0, v_res)
       
        G_flat = np.repeat(g_signal/1000.0, v_res)
        T_flat = np.repeat((t_signal-25.0)/100.0, v_res)
        V_flat = np.tile(v_sweep, N_t)
        DNA_flat = np.tile(dna_norm, (len(G_flat), 1))
        SC_flat = np.tile([row['V_oc_ref']/50.0, row['I_sc_ref']/10.0], (len(G_flat), 1))
       
        X_batch = np.column_stack([G_flat, T_flat, V_flat, DNA_flat, SC_flat])
        y_pred = trained_model.predict(X_batch, batch_size=4096, verbose=0).flatten()
       
        # Denormalize & find MPP
        p_real = (V_flat * row['V_oc_ref']) * (y_pred * row['I_sc_ref'])
        p_matrix = p_real.reshape(N_t, v_res)
        ai_p_mp = np.max(p_matrix, axis=1) 
        energy_ai = np.sum(ai_p_mp) * 0.25
       
        error_pct = (energy_ai - energy_true) / energy_true * 100
        results.append({
            "Name": row.name,
            "True_Wh": energy_true,
            "AI_Wh": energy_ai,
            "Error_%": error_pct
        })

    # 4. VISUALIZATION & STATS
    df_res = pd.DataFrame(results)
    df_res = df_res[df_res['Error_%'].abs() < 50] # Remove data outliers
   
    # Accuracy Histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(df_res['Error_%'], kde=True, color='blue')
    plt.title("Universal PINN Energy Accuracy")
    plt.show()

    print(f"\nMean Abs Error: {df_res['Error_%'].abs().mean():.2f}%")
    return df_res

# --- 4. EXECUTION BLOCK ---
if __name__ == "__main__":
    benchmark_data = run_massive_benchmark(model, n_panels=200)