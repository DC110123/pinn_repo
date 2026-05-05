import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pvlib

# 1. Load and Transpose the entire database
print("Loading Database...")
df_raw = pvlib.pvsystem.retrieve_sam('CECMod').T
df_raw.to_csv('modules.csv')

def perform_panel_eda(df):
    # --- STEP 1: CATEGORIZATION & INITIAL ANALYSIS ---
    print(f"Original database has {len(df)} entries.")
    print("Top 5 technologies in the original database:")
    print(df['Technology'].value_counts().head())

    ### MODIFICATION: Create a new column to group technologies.
    # We use np.where to create our two categories: 'Mono-c-Si' and 'Other'.
    df['Tech_Group'] = np.where(df['Technology'] == 'Mono-c-Si', 'Mono-c-Si', 'Other')
    
    print("\nNew category distribution:")
    print(df['Tech_Group'].value_counts())

    # --- STEP 2: CLEANING & CONVERSION ---
    # Define columns we want to analyze, now including our new group
    dna_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'V_oc_ref', 'N_s', 'Tech_Group']
    
    df_numeric = df.copy()
    for col in dna_cols:
        if col != 'Tech_Group': # Don't convert the group label
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # --- STEP 3: FEATURE ENGINEERING & CLEANING ---
    df_numeric['I_o_ref_log'] = np.log10(df_numeric['I_o_ref'].astype(float) + 1e-20)
    
    # Define plot columns, using Tech_Group instead of Technology
    plot_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref_log', 'R_s', 'R_sh_ref', 'V_oc_ref', 'Tech_Group']
    
    print("\nChecking for NaN values after data type conversion:")
    print(df_numeric[[col for col in plot_cols if col != 'Tech_Group']].isnull().sum())
    
    df_clean = df_numeric.dropna(subset=plot_cols)
    print(f"\n{len(df_clean)} rows remain after dropping NaNs.")

    # Filter for reasonable physical values
    df_clean = df_clean[df_clean['R_sh_ref'] < 10000]
    df_clean = df_clean[df_clean['R_s'] < 5]
    print(f"{len(df_clean)} rows remain after filtering outliers.")

    if len(df_clean) == 0:
        print("Error: All data was removed during cleaning. No plots can be generated.")
        return

    # --- STEP 4: COLOR-CODED PAIRPLOT ---
    sns.set_theme(style="whitegrid")
    print("\nGenerating Pairplot...")

    # Robustness check for sample size
    sample_size = min(1000, len(df_clean))
    print(f"(Using {sample_size} sample points for speed)...")

    ### MODIFICATION: Hue is now based on 'Tech_Group'
    g = sns.pairplot(
        df_clean.sample(sample_size, random_state=42),
        vars=[col for col in plot_cols if col != 'Tech_Group'],
        hue='Tech_Group', # This is the key change!
        palette='magma',   # A different palette for a new look
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 15}
    )
    g.fig.suptitle("Panel DNA: Mono-c-Si vs. Other Technologies", y=1.02, fontsize=16)
    plt.show()

    # --- STEP 5: SEPARATE CORRELATION HEATMAPS ---
    print("\nGenerating separate correlation heatmaps for each group...")
    ### MODIFICATION: Loop through the new 'Tech_Group' categories
    for group_type in df_clean['Tech_Group'].unique():
        plt.figure(figsize=(10, 8))
        subset_df = df_clean[df_clean['Tech_Group'] == group_type]
        
        if not subset_df.empty:
            corr = subset_df[[col for col in plot_cols if col != 'Tech_Group']].corr()
            sns.heatmap(corr, annot=True, cmap='magma', fmt=".2f")
            plt.title(f"DNA Feature Correlation Matrix for {group_type} Panels", fontsize=14)
            plt.show()

# Run the EDA function
perform_panel_eda(df_raw)
