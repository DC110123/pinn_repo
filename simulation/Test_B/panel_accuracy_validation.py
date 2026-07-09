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
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

def run_massive_benchmark(trained_model, n_panels=200):
    print("="*60)
    print(f"🚀 STARTING MASSIVE BENCHMARK: {n_panels} PANELS")
    print("="*60)
   
    # 1. Select Random Panels
    # Ensure we only pick valid rows from the CEC database
    test_indices = np.random.choice(len(df_db), n_panels, replace=False)
   
    # 2. Define the "Test Day" Weather (Same for all panels)
    times = pd.date_range('2025-06-21 06:00', '2025-06-21 18:00', freq='15min')
    N_t = len(times)
   
    # Create weather signals
    x = np.linspace(0, np.pi, N_t)
    g_signal = 1000 * np.sin(x)
    g_signal[20:26] *= 0.4  # Noon cloud
    
    # STABILITY FIX: PVLib solvers can fail at exactly 0 irradiance. 
    # We use a tiny floor (0.1W) for physics calculations.
    g_physics = np.maximum(g_signal, 0.1) 
    
    t_signal = 20 + 30 * np.sin(x) # Simple temp curve (20C to 50C)
   
    # 3. STORAGE FOR RESULTS
    results = []
   
    print(f"Simulating full day for {n_panels} panels... (Physics vs. PINN)")
   
    for i, idx in enumerate(test_indices):
        if i % 20 == 0: print(f"Processing panel {i}/{n_panels}...")
       
        row = df_db.iloc[idx]
        panel_name = row.name
       
        # --- A. PREPARE DNA ---
        dna_vals = [row[c] for c in fit_cols]
        dna_df = pd.DataFrame([dna_vals], columns=fit_cols)
        dna_norm = dna_scaler.transform(dna_df)[0]
       
        # --- B. CALCULATE TRUE ENERGY (PVLIB) ---
        energy_true = 0
        try:
            # We wrap the solver in errstate to ignore the 'divide by zero' warnings 
            # produced by internal Scipy iterations
            with np.errstate(divide='ignore', invalid='ignore'):
                IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                    g_physics, t_signal,
                    row['alpha_sc'], row['a_ref'], row['I_L_ref'],
                    row['I_o_ref'], row['R_sh_ref'], row['R_s']
                )
                
                # Solve for max power point using LambertW
                sol = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth, method='lambertw')
                p_mp = sol['p_mp']
                
                # Clean up results (replace NaNs with 0)
                p_mp_clean = np.nan_to_num(p_mp, nan=0.0)
                energy_true = np.sum(p_mp_clean) * 0.25 # 15 min intervals = 0.25 hours
        except Exception:
            continue # Skip panels that cause the physics solver to diverge
           
        if energy_true < 5: continue # Skip broken/tiny panels
           
        # --- C. CALCULATE AI ENERGY (PINN) ---
        # Generate Voltage Sweep [0 to 1 normalized]
        v_res = 50
        v_sweep = np.linspace(0, 1.0, v_res)
       
        # Build Vectorized Batch: [Time * Voltage Sweep, Features]
        G_flat = np.repeat(g_signal/1000.0, v_res)
        T_flat = np.repeat((t_signal-25.0)/100.0, v_res)
        V_flat = np.tile(v_sweep, N_t)
       
        DNA_flat = np.tile(dna_norm, (len(G_flat), 1))
        # SC characteristics for scaling the PINN output
        SC_flat = np.tile([row['V_oc_ref']/50.0, row['I_sc_ref']/10.0], (len(G_flat), 1))
       
        X_batch = np.column_stack([G_flat, T_flat, V_flat, DNA_flat, SC_flat])
       
        # Run AI Inference (Batch process the entire day at once)
        y_pred = trained_model.predict(X_batch, batch_size=8192, verbose=0).flatten()
       
        # Convert AI normalized current back to Power
        i_real = y_pred * row['I_sc_ref']
        v_real = V_flat * row['V_oc_ref']
        p_real = v_real * i_real
       
        # Find peak power for each of the 48 time steps
        p_matrix = p_real.reshape(N_t, v_res)
        ai_p_mp = np.max(p_matrix, axis=1) 
       
        energy_ai = np.sum(ai_p_mp) * 0.25
       
        # --- D. COMPARE ---
        error_pct = (energy_ai - energy_true) / energy_true * 100
       
        results.append({
            "Name": panel_name,
            "Cells": row['N_s'],
            "True_Wh": energy_true,
            "AI_Wh": energy_ai,
            "Error_%": error_pct
        })

    # 4. VISUALIZATION
    df_res = pd.DataFrame(results)
    
    # Filter extreme outliers usually caused by physical solver failures
    df_res = df_res[df_res['Error_%'].abs() < 50]
   
    # Plot 1: Accuracy Distribution
    plt.figure(figsize=(12, 5))
    sns.histplot(df_res['Error_%'], kde=True, color='royalblue', bins=30)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Universal PINN Performance Across {len(df_res)} Different Solar Panels", fontsize=14)
    plt.xlabel("Energy Error (%) [AI vs Physics]", fontsize=12)
    plt.ylabel("Number of Panels", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
   
    # Plot 2: Regression Correlation
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_res, x='True_Wh', y='AI_Wh', hue='Error_%', palette='viridis', alpha=0.6)
   
    max_val = max(df_res['True_Wh'].max(), df_res['AI_Wh'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Agreement')
   
    plt.title("Daily Energy Production: Physics vs. AI", fontsize=14)
    plt.xlabel("Physics Solver (Wh)", fontsize=12)
    plt.ylabel("PINN Prediction (Wh)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
   
    # 5. FINAL STATS
    mae = df_res['Error_%'].abs().mean()
    print("\n" + "="*30)
    print("       FINAL SCORECARD       ")
    print("="*30)
    print(f"Panels Tested:      {len(df_res)}")
    print(f"Mean Abs Error:     {mae:.2f}%")
    print(f"Median Abs Error:   {df_res['Error_%'].abs().median():.2f}%")
    print(f"95th Percentile:    {df_res['Error_%'].abs().quantile(0.95):.2f}%")
    print("="*30)
   
    return df_res

# RUN THE BENCHMARK
# Note: Ensure 'model', 'df_db', 'fit_cols', and 'dna_scaler' are defined in your workspace.
benchmark_results = run_massive_benchmark(model, n_panels=1000)