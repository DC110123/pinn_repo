import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow import keras
from pvlib import pvsystem

# --- 1. CONNECT TO PROJECT ROOT ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data_loader import prepare_pv_database

# --- 2. LOAD ASSETS ---
print("Loading model and database...")
model = keras.models.load_model("../../model/universal_pinn_weights.weights.h5", compile=False)
dna_scaler = joblib.load("../../model/dna_scaler.pkl")
df_db = prepare_pv_database()
fit_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref_log', 'R_s', 'R_sh_ref_log', 'N_s']

# --- 3. YOUR SIMULATOR FUNCTION (Copy-pasted) ---
def run_10_panel_simulation(trained_model, df_db, fit_cols, dna_scaler):
    # 1. Select 10 Random Panels
    # We try to get a mix of different cell counts if possible
    random_indices = np.random.choice(len(df_db), 10, replace=False)
    
    print(f"🚀 STARTING BATCH SIMULATION FOR 10 PANELS")
    
    for i, idx in enumerate(random_indices):
        row = df_db.iloc[idx]
        panel_name = row.name
        
        print(f"\n[{i+1}/10] Simulating: {panel_name}")
        print("-" * 60)

        # =====================================================================
        # A. PREPARE DNA
        # =====================================================================
        dna_vals = [row[c] for c in fit_cols]
        dna_df = pd.DataFrame([dna_vals], columns=fit_cols)
        dna_norm = dna_scaler.transform(dna_df)[0]

        # =====================================================================
        # B. WEATHER PROFILE (THE "NOTCH")
        # =====================================================================
        times = pd.date_range('2025-07-01 07:00', '2025-07-01 19:00', freq='10min')
        N_t = len(times)

        x = np.linspace(0, np.pi, N_t)
        g_signal = 1000 * np.sin(x)
        
        # Create the visual "Notch" (Cloud)
        start_notch = int(N_t * 0.40)
        end_notch = int(N_t * 0.60)
        g_signal[start_notch:end_notch] *= 0.6
        g_signal = np.clip(g_signal, 0, None)
        
        t_signal = 25 + 20 * np.sin(x)

        # =====================================================================
        # C. AI PREDICTION
        # =====================================================================
        v_res = 100
        v_sweep = np.linspace(0, 1.0, v_res)

        G_flat = np.repeat(g_signal / 1000.0, v_res)
        T_flat = np.repeat((t_signal - 25.0) / 100.0, v_res)
        V_flat = np.tile(v_sweep, N_t)
        DNA_flat = np.tile(dna_norm, (len(G_flat), 1))
        SC_flat = np.tile([row['V_oc_ref']/50.0, row['I_sc_ref']/10.0], (len(G_flat), 1))

        X_batch = np.column_stack([G_flat, T_flat, V_flat, DNA_flat, SC_flat])

        print(f"PVLIB DYNAMIC SIMULATION — FULLY VECTORIZED")
        print(f"Batch predicting {len(X_batch)} points...")

        # Predict
        y_pred = trained_model.predict(X_batch, batch_size=8192, verbose=0).flatten()

        # Denormalize
        i_pred = y_pred * row['I_sc_ref']
        v_real = V_flat * row['V_oc_ref']
        p_pred = v_real * i_pred
        
        # Get Max Power
        p_matrix = p_pred.reshape(N_t, v_res)
        ai_p_mp = np.max(p_matrix, axis=1)

        # =====================================================================
        # D. PHYSICS VERIFICATION
        # =====================================================================
        true_p_mp = []
        
        # Calculate params (Vectorized)
        IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
            g_signal, t_signal, 
            row['alpha_sc'], row['a_ref'], row['I_L_ref'], 
            row['I_o_ref'], row['R_sh_ref'], row['R_s']
        )
        
        # Solve Singlediode (Loop for safety)
        for k in range(N_t):
            if g_signal[k] < 10: 
                true_p_mp.append(0)
                continue
            try:
                res = pvsystem.singlediode(
                    IL[k], I0[k], Rs[k], Rsh[k], nNsVth[k], method='lambertw'
                )
                true_p_mp.append(res['p_mp'])
            except:
                true_p_mp.append(0)

        # Efficiency
        total_ai = np.sum(ai_p_mp)
        total_true = np.sum(true_p_mp)
        eff = (total_ai / total_true) * 100

        print(f"Efficiency: \033[1m{eff:.2f}%\033[0m") 

        # =====================================================================
        # E. PLOTTING
        # =====================================================================
        fig, ax = plt.subplots(figsize=(10, 3))
        
        ax.plot(times, true_p_mp, label='Ground Truth', color='#1f77b4', lw=2)
        ax.plot(times, ai_p_mp, label='PINN AI', color='#ff7f0e', linestyle='--', lw=2)
        
        ax.set_title(f"{panel_name} ({int(row['N_s'])} Cells)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel("Power (W)")
        
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

# Run the Batch
run_10_panel_simulation(model, df_db, fit_cols, dna_scaler)



# --- 4. EXECUTION ---
if __name__ == "__main__":
    run_10_panel_simulation(model, df_db, fit_cols, dna_scaler)