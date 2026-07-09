import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(units1, units2, units3):
    return keras.Sequential([
        layers.InputLayer(shape=(12,)),
        layers.Dense(units1, activation='swish'),
        layers.Dense(units2, activation='swish'),
        layers.Dense(units3, activation='swish'),
        layers.Dense(1, activation='linear')  # linear output is crucial for PINNs
    ])


def make_train_step(model, optimizer):
    @tf.function
    def train_step(x, y_true, phys_weight):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_pred = model(x, training=True)

            grads_input = tape.gradient(y_pred, x)
            dI_dV = grads_input[:, 2:3]  # voltage is feature index 2

            loss_data = tf.reduce_mean(tf.square(y_true - y_pred))
            loss_monotonic = tf.reduce_mean(tf.square(tf.nn.relu(dI_dV)))

            total_loss = loss_data + (phys_weight * loss_monotonic)

        grads_weights = tape.gradient(total_loss, model.trainable_variables)
        del tape
        optimizer.apply_gradients(zip(grads_weights, model.trainable_variables))
        return loss_data, loss_monotonic

    return train_step
