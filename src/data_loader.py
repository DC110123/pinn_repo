import numpy as np
import pandas as pd
import pvlib
from pvlib import pvsystem
from sklearn.preprocessing import RobustScaler

def prepare_pv_database():
    """Retrieves and cleans the CEC database."""
    all_modules = pvlib.pvsystem.retrieve_sam('CECMod').T
    df = all_modules[all_modules['Technology'].str.contains('Si', na=False)].copy()
    
    dna_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'N_s']
    for col in dna_cols + ['V_oc_ref', 'I_sc_ref', 'V_mp_ref', 'I_mp_ref']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[
        (df['I_o_ref'] > 1e-15) & (df['a_ref'] > 0.5) &
        (df['R_sh_ref'] > 10.0) & (df['I_sc_ref'] > 0.5) &
        (df['V_oc_ref'] > 10.0)
    ].dropna(subset=dna_cols)

    df['I_o_ref_log'] = np.log10(df['I_o_ref'])
    df['R_sh_ref_log'] = np.log10(df['R_sh_ref'])
    return df

def generate_pinn_data(df_db, dna_scaler, fit_cols, n_panels=2500, pts=50):
    """Generates the synthetic dataset."""
    X, y = [], []
    indices = np.random.choice(len(df_db), n_panels, replace=True)

    for idx in indices:
        row = df_db.iloc[idx]
        dna_vals = [row[c] for c in fit_cols]
        dna_df = pd.DataFrame([dna_vals], columns=fit_cols)
        dna_norm = dna_scaler.transform(dna_df)[0]

        G = np.random.uniform(200, 1100, pts)
        T_c = np.random.uniform(15, 75, pts)

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
                        G[j]/1000.0, (T_c[j]-25.0)/100.0, V_norm_axis[j]
                    ] + dna_norm.tolist() + [
                        row['V_oc_ref'] / 50.0, row['I_sc_ref'] / 10.0
                    ])
                    y.append(max(0, I_raw[j] / row['I_sc_ref']))
        except Exception: continue

    return np.array(X, 'float32'), np.array(y, 'float32').reshape(-1, 1)