from keras import layers
from keras.models import Model
import tensorflow as tf
from mltu.tensorflow.model_utils import residual_block
from keras.saving import register_keras_serializable

@register_keras_serializable(package="cgl_ml")
class HWCToSeq(layers.Layer):
    def call(self, t):
        t = tf.transpose(t, perm=[0, 2, 1, 3])
        b = tf.shape(t)[0]
        w = tf.shape(t)[1]
        h = tf.shape(t)[2]
        c = tf.shape(t)[3]
        return tf.reshape(t, [b, w, h * c])

    def get_config(self):
        config = super().get_config()
        return config

    

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")

    # ✅ explicit dtype + normalization
    # x = layers.Lambda(lambda t: tf.cast(t, tf.float32) / 255.0, name="norm_255")(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="norm_255")(inputs)

    x1 = residual_block(x, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x2 = residual_block(x1, 32, activation=activation, skip_conv=True, strides=(2,1), dropout=dropout)
    x3 = residual_block(x2, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3, 64, activation=activation, skip_conv=True, strides=(2,1), dropout=dropout)
    x5 = residual_block(x4, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # ✅ preserve width in deeper layers (Arabic-friendly)
    x6 = residual_block(x5, 128, activation=activation, skip_conv=True, strides=(2,1), dropout=dropout)
    x7 = residual_block(x6, 128, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    x8 = residual_block(x7, 128, activation=activation, skip_conv=True,  strides=(2,2), dropout=dropout)
    x9 = residual_block(x8, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    # 🔻 REQUIRED: channel compression before sequence
    x9 = layers.Conv2D(
        filters=256,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="pre_seq_compress"
    )(x9)
    x9 = layers.BatchNormalization()(x9)
    x9 = layers.ReLU()(x9)

    # ✅ width as time axis: (B,H,W,C) -> (B,W,H*C)
    # def hwc_to_seq(t):
    #     t = tf.transpose(t, perm=[0, 2, 1, 3])   # (B,W,H,C)
    #     b = tf.shape(t)[0]
    #     w = tf.shape(t)[1]
    #     h = tf.shape(t)[2]
    #     c = tf.shape(t)[3]
    #     return tf.reshape(t, [b, w, h * c])

    # squeezed = layers.Lambda(hwc_to_seq, name="hwc_to_seq")(x9)
    
    squeezed = HWCToSeq(name="hwc_to_seq")(x9)

    blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name="bilstm_256")(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    # ✅ a bit more capacity for Arabic
    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="bilstm_128")(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(output_dim, activation="softmax", dtype="float32", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model
