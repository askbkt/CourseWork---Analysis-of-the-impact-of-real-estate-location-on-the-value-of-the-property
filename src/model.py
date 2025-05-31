import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

def build_tabular_model(
    num_input_dim,
    dense_units=[128, 64],
    dropout_rate=0.3
):
    """
    Простая MLP для табличных признаков.
    """
    inp = Input(shape=(num_input_dim,), name="tab_input")
    x = inp
    for units in dense_units:
        x = Dense(units, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    out = Dense(1, activation="linear", name="price_output")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
