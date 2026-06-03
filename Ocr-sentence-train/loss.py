import tensorflow as tf


class CTCloss(tf.keras.losses.Loss):
    """CTC loss for OCR training with correct per-sample label lengths."""

    def __init__(self, name: str = "CTCloss", reduction: str = tf.keras.losses.Reduction.NONE):
        super().__init__(name=name, reduction=reduction)
        self.loss_fn = tf.keras.backend.ctc_batch_cost
        self._debug_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int64)

        batch_len = tf.shape(y_true)[0]
        time_steps = tf.shape(y_pred)[1]

        # Since your model output is vocab + 1, last index is blank/padding.
        blank_index = tf.cast(tf.shape(y_pred)[-1] - 1, tf.int64)

        input_length = tf.fill(
            dims=(batch_len, 1),
            value=tf.cast(time_steps, tf.int64),
        )

        # ✅ Correct real label lengths: count non-padding entries per sample.
        label_length = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, blank_index), tf.int64),
            axis=1,
            keepdims=True,
        )

        # 🔍 Debug only first 5 batches
        def _print_debug():
            tf.print(
                "[ctc-debug]",
                "padded_width=", tf.shape(y_true)[1],
                "blank_index=", blank_index,
                "real_lengths=", tf.squeeze(label_length, axis=1)[:8],
                "old_wrong_length_would_be=", tf.shape(y_true)[1],
                summarize=-1,
            )
            self._debug_counter.assign_add(1)
            return 0

        tf.cond(
            self._debug_counter < 5,
            _print_debug,
            lambda: 0,
        )

        return self.loss_fn(y_true, y_pred, input_length, label_length)