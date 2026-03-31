import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
import joblib # To save the scaler later

# Import our custom modules
from src.data_loader import prepare_pv_database, generate_pinn_data
from src.network import build_pinn_model, train_step

# 1. Setup Data
df_db = prepare_pv_database()
fit_cols = ['alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref_log', 'R_s', 'R_sh_ref_log', 'N_s']

dna_scaler = RobustScaler()
dna_scaler.fit(df_db[fit_cols])

print(f"✅ Database Ready: {len(df_db)} panels.")
X_train, y_train = generate_pinn_data(df_db, dna_scaler, fit_cols, n_panels=2500, pts=40)

# 2. Setup Model & Optimizer
model = build_pinn_model()
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.92, staircase=True)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# 3. Training Loop
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100000).batch(2048)

for epoch in range(400):
    # Adaptive weighting logic
    p_w = 0.0 if epoch < 20 else (0.01 if epoch < 50 else 0.05)

    l_d, l_p = [], []
    for xb, yb in dataset:
        d, p = train_step(model, optimizer, xb, yb, tf.constant(p_w, dtype=tf.float32))
        l_d.append(d)
        l_p.append(p)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3} | Data Loss: {np.mean(l_d):.6f} | Phys Loss: {np.mean(l_p):.6f} | W: {p_w}")

# 4. Save your work
model.save("models/pinn_pv_model.h5")
joblib.dump(dna_scaler, "models/dna_scaler.pkl")