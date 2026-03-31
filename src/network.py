import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_pinn_model(input_shape=(12,)):
    model = keras.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Dense(256, activation='swish'),
        layers.Dense(128, activation='swish'),
        layers.Dense(64, activation='swish'),
        layers.Dense(1, activation='linear')
    ])
    return model

@tf.function
def train_step(model, optimizer, x, y_true, phys_weight):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y_pred = model(x, training=True)

        # Physics: Calculate dI/dV (Voltage is index 2)
        grads_input = tape.gradient(y_pred, x)
        dI_dV = grads_input[:, 2:3] 

        # Losses
        loss_data = tf.reduce_mean(tf.square(y_true - y_pred))
        # Monotonicity: Penalize positive slope
        loss_monotonic = tf.reduce_mean(tf.square(tf.nn.relu(dI_dV)))

        total_loss = loss_data + (phys_weight * loss_monotonic)

    grads_weights = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads_weights, model.trainable_variables))
    
    return loss_data, loss_monotonic