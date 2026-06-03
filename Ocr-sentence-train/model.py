import tensorflow as tf
from tensorflow.keras import layers, Model

from mltu.tensorflow.model_utils import residual_block


@tf.keras.utils.register_keras_serializable(package="Cognilabel")
class ReverseTimeAxis(layers.Layer):
    """
    Reverses OCR sequence time axis.

    Input/output shape:
      [batch, time_steps, vocab + blank]

    Used for RTL logical labels with CRNN/CTC so that CTC sees:
      label[0] -> label[-1]
    """

    def call(self, inputs):
        return tf.reverse(inputs, axis=[1])

    def get_config(self):
        return super().get_config()


def train_model(
    input_dim,
    output_dim,
    activation="leaky_relu",
    dropout=0.2,
    reverse_time_axis=False,
):
    inputs = layers.Input(shape=input_dim, name="input")

    # Normalize images here instead of preprocessing step.
    x = layers.Lambda(lambda t: t / 255.0, name="image_normalize")(inputs)

    x1 = residual_block(x, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 128, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x8 = residual_block(x7, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x9 = residual_block(x8, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = layers.Reshape(
        (x9.shape[-3] * x9.shape[-2], x9.shape[-1]),
        name="image_to_sequence",
    )(x9)

    blstm = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True),
        name="bilstm_1",
    )(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True),
        name="bilstm_2",
    )(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(
        output_dim + 1,
        activation="softmax",
        name="output",
    )(blstm)

    if reverse_time_axis:
        output = ReverseTimeAxis(name="rtl_reverse_time_axis")(output)

    model = Model(inputs=inputs, outputs=output, name="cognilabel_ocr_crnn_ctc")
    return model