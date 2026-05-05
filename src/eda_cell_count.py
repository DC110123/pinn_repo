import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pvlib

# 1. Load and Transpose the Database
print("Loading Database...")
df_raw = pvlib.pvsystem.retrieve_sam('CECMod').T
def perform_panel_eda_by_size(df):
    dna_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'V_oc_ref', 'N_s']
    df_numeric = df.copy()
    
    for col in dna_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # --- STEP 1: CATEGORIZE BY CELL COUNT (N_s) ---
    # Most panels are 60 or 72 cells. We will group them to clean up the plot.
    def classify_size(ns):
        if ns <= 60: return '60-Cell Class'
        if 60 < ns <= 72: return '72-Cell Class'
        if ns > 72: return 'High-Voltage/Industrial'
        return 'Small/Other'

    df_numeric['Module_Size'] = df_numeric['N_s'].apply(classify_size)
    
    # Feature Engineering
    df_numeric['I_o_ref_log'] = np.log10(df_numeric['I_o_ref'].astype(float) + 1e-20)
    
    # Filter for the main two classes to make the plot readable
    df_plot = df_numeric[df_numeric['Module_Size'].isin(['60-Cell Class', '72-Cell Class'])]
    df_plot = df_plot.dropna(subset=['alpha_sc', 'a_ref', 'V_oc_ref', 'I_L_ref'])
    
    # Remove extreme outliers for better axis scaling
    df_plot = df_plot[df_plot['V_oc_ref'] < 100]

    # --- STEP 2: PAIRPLOT COLORED BY SIZE ---
    sns.set_theme(style="ticks")
    plot_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref_log', 'R_s', 'V_oc_ref']
    
    g = sns.pairplot(
        df_plot[plot_cols + ['Module_Size']].sample(800, random_state=42), 
        hue='Module_Size',
        palette='magma', # 'magma' or 'rocket' provides good contrast for sizes
        diag_kind='kde', 
        plot_kws={'alpha': 0.4, 's': 20}
    )
    g.fig.suptitle("Solar Panel DNA: Distinguishing by Cell Count (Ns)", y=1.02, fontsize=16)
    plt.show()

perform_panel_eda_by_size(df_raw)
