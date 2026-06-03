import tensorflow as tf
import numpy as np


class SafeCERMetric(tf.keras.metrics.Metric):
    def __init__(self, vocabulary, name="CER", **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocab = tf.constant(list(vocabulary))
        self.total_cer = self.add_weight(name="total_cer", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = tf.shape(y_pred)
        batch_size = input_shape[0]
        time_steps = input_shape[1]

        input_length = tf.ones(shape=(batch_size,), dtype="int32") * time_steps

        decoded, _ = tf.keras.backend.ctc_decode(
            y_pred,
            input_length=input_length,
            greedy=True,
        )

        pred = tf.cast(decoded[0], tf.int64)
        y_true = tf.cast(y_true, tf.int64)

        vocab_len = tf.cast(tf.shape(self.vocab)[0], tf.int64)

        # Any index >= vocab_len is padding/blank.
        pred = tf.where(pred < vocab_len, pred, tf.cast(-1, tf.int64))
        true = tf.where(y_true < vocab_len, y_true, tf.cast(-1, tf.int64))

        sparse_pred = tf.RaggedTensor.from_tensor(pred, padding=-1).to_sparse()
        sparse_true = tf.RaggedTensor.from_tensor(true, padding=-1).to_sparse()

        distance = tf.edit_distance(sparse_pred, sparse_true, normalize=True)

        self.total_cer.assign_add(tf.reduce_sum(distance))
        self.count.assign_add(tf.cast(batch_size, tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total_cer, self.count)

    def reset_state(self):
        self.total_cer.assign(0.0)
        self.count.assign(0.0)
        
        



def _word_edit_distance_py(ref_text, pred_text):
    ref_words = ref_text.strip().split()
    pred_words = pred_text.strip().split()

    n, m = len(ref_words), len(pred_words)

    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == pred_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return float(dp[n][m]) / max(1, n)


class SafeWERMetric(tf.keras.metrics.Metric):
    def __init__(self, vocabulary, name="WER", **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocabulary = list(vocabulary)
        self.total_wer = self.add_weight(name="total_wer", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def _decode_batch_py(self, y_true_np, y_pred_np):
        batch_size = y_pred_np.shape[0]
        time_steps = y_pred_np.shape[1]
        vocab_len = len(self.vocabulary)
        blank_index = vocab_len

        input_len = [time_steps] * batch_size

        decoded, _ = tf.keras.backend.ctc_decode(
            y_pred_np,
            input_length=input_len,
            greedy=True,
        )

        decoded_np = decoded[0].numpy()

        refs = []
        preds = []

        for true_row, pred_row in zip(y_true_np, decoded_np):
            ref_chars = []
            for k in true_row:
                k = int(k)
                if 0 <= k < vocab_len:
                    ref_chars.append(self.vocabulary[k])

            pred_chars = []
            for k in pred_row:
                k = int(k)
                if 0 <= k < vocab_len:
                    pred_chars.append(self.vocabulary[k])

            refs.append("".join(ref_chars))
            preds.append("".join(pred_chars))

        wers = [
            _word_edit_distance_py(ref, pred)
            for ref, pred in zip(refs, preds)
        ]

        return np.float32(sum(wers)), np.float32(len(wers))

    def update_state(self, y_true, y_pred, sample_weight=None):
        total_wer, count = tf.py_function(
            func=self._decode_batch_py,
            inp=[y_true, y_pred],
            Tout=[tf.float32, tf.float32],
        )

        # ✅ Required because tf.py_function returns unknown TensorShape
        total_wer.set_shape([])
        count.set_shape([])

        self.total_wer.assign_add(total_wer)
        self.count.assign_add(count)

    def result(self):
        return tf.math.divide_no_nan(self.total_wer, self.count)

    def reset_state(self):
        self.total_wer.assign(0.0)
        self.count.assign(0.0)