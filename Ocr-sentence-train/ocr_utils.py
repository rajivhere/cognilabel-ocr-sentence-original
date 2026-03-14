# ocr_utils.py
import os, io, re, json, hashlib
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

try:
    import boto3
except ImportError:
    boto3 = None

# ---------- I/O & JSONL → cache ----------

def _safe_name(s: str, ext=".png") -> str:
    return f"{hashlib.sha1(s.encode('utf-8')).hexdigest()}{ext}"

def _resolve_local_path(p: str, *, jsonl_dir: Path, image_base: Optional[Path]) -> str:
    p = p.replace("\\", "/")
    if re.match(r"^[a-zA-Z]:/", p):
        cand = Path(p)
    else:
        if p.startswith("/") and not p.startswith("//"):
            p = p.lstrip("/")
        cand = Path(p)

    if cand.is_absolute() and cand.exists():
        return str(cand)
    cand2 = (jsonl_dir / p).resolve()
    if cand2.exists():
        return str(cand2)
    if image_base:
        cand3 = (image_base / p).resolve()
        if cand3.exists():
            return str(cand3)
        cand4 = (image_base / Path(p).name).resolve()
        if cand4.exists():
            return str(cand4)
    raise FileNotFoundError(f"Could not find image '{p}'")

def _open_image(uri: str, *, jsonl_dir: Path, image_base: Optional[Path]) -> Image.Image:
    if uri.startswith("s3://"):
        if not boto3:
            raise RuntimeError("Install boto3 for s3:// support.")
        from urllib.parse import urlparse
        p = urlparse(uri)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
    return Image.open(_resolve_local_path(uri, jsonl_dir=jsonl_dir, image_base=image_base)).convert("RGB")

def _char_filter(s: str) -> str:
    import unicodedata as ud
    return "".join([c for c in s if not ud.category(c).startswith("C")]).strip()

def _should_rtl_for_record(rec, policy):
    if policy == "rtl":
        return True
    if policy == "ltr":
        return False
    # auto: respect JSONL per-record if present, else fallback to language heuristics (optional)
    d = (rec.get("direction") or rec.get("dir") or "").strip().lower()
    if d in ("rtl", "ltr"):
        return d == "rtl"
    return False

def build_cache(
    jsonl_path: str,
    out_dir: str,
    *,
    image_base: Optional[str] = None,
    min_w: int = 8,
    min_h: int = 8,
    rtl_policy
) -> Tuple[List[List[str]], str, int]:
    """
    Read OCR_JSONL (file- or region-scope), write cached crops, return:
      items: [[png_path, label], ...], vocab_str, max_text_len
    """
    jd = Path(jsonl_path).resolve().parent
    ib = Path(image_base).resolve() if image_base else None
    os.makedirs(out_dir, exist_ok=True)

    items: List[List[str]] = []
    vocab, max_len = set(), 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            src = rec.get("image") or rec.get("file") or rec.get("path") or rec.get("uri")
            if not src:
                continue

            try:
                img = _open_image(src, jsonl_dir=jd, image_base=ib)
            except Exception as e:
                print("[warn]", e)
                continue

            # file-scope
            if "text" in rec and not rec.get("regions"):
                lab = _char_filter(rec.get("text") or "")
                if not lab:
                    continue
                
                is_rtl = _should_rtl_for_record(rec, rtl_policy)
                if is_rtl:
                    lab = lab[::-1]     # <— reverse for RTL
                path = os.path.join(out_dir, _safe_name(src + lab))
                img.save(path)
                items.append([path, lab])
                vocab.update(lab)
                max_len = max(max_len, len(lab))
                continue

            # region-scope
            for r in (rec.get("regions") or []):
                lab = _char_filter(r.get("text") or "")
                if not lab:
                    continue
                is_rtl = _should_rtl_for_record(rec, rtl_policy)
                if is_rtl:
                    lab = lab[::-1]     # <— reverse for RTL              
                    
                x, y, w, h = (r.get("bbox") or [0, 0, 0, 0])
                if w < min_w or h < min_h:
                    continue
                crop = img.crop((x, y, x + w, y + h))
                path = os.path.join(out_dir, _safe_name(f"{src}_{x}_{y}_{w}_{h}_{lab}"))
                crop.save(path)
                items.append([path, lab])
                vocab.update(lab)
                max_len = max(max_len, len(lab))

    return items, "".join(sorted(vocab)), max_len

# ---------- vocab/encoding & tf.data ----------

def build_vocab(items: List[List[str]]) -> Tuple[str, dict]:
    chars = set()
    for _, l in items:
        chars.update(l)
    vocab = "".join(sorted(chars))
    c2i = {c: i for i, c in enumerate(vocab)}
    return vocab, c2i

def encode_label_py(text: str, c2i: dict, pad_value: int, max_len: int) -> np.ndarray:
    ids = [c2i.get(c, None) for c in text]
    ids = [i for i in ids if i is not None]
    ids = (ids + [pad_value] * (max_len - len(ids)))[:max_len]
    return np.array(ids, dtype=np.int32)

def letterbox(img: tf.Tensor, H: int, W: int) -> tf.Tensor:
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    s = tf.minimum(W / tf.cast(w, tf.float32), H / tf.cast(h, tf.float32))
    nh = tf.cast(tf.round(tf.cast(h, tf.float32) * s), tf.int32)
    nw = tf.cast(tf.round(tf.cast(w, tf.float32) * s), tf.int32)
    resized = tf.image.resize(img, [nh, nw], method="bilinear")
    pad_h = H - nh
    pad_w = W - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return tf.pad(resized, [[top, bottom], [left, right], [0, 0]], constant_values=0.0)

def make_dataset(
    items: List[List[str]],
    vocab: str,
    H: int,
    W: int,
    batch: int,
    max_len: int,
    pad_value: int,
    *,
    training: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    c2i = {c: i for i, c in enumerate(vocab)}
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(4096, len(items)), seed=seed)

    def _map(p, l):
        img = tf.io.read_file(p)
        img = tf.image.decode_png(img, channels=3, dtype=tf.uint8)
        img = tf.image.convert_image_dtype(img, tf.float32)
        # if training:
        #     img = tf.image.random_brightness(img, max_delta=0.1)
        #     img = tf.image.random_contrast(img, 0.9, 1.1)
        img = letterbox(img, H, W)
        ids = tf.numpy_function(
            lambda s: encode_label_py(s.decode("utf-8"), c2i, pad_value, max_len),
            [l],
            Tout=tf.int32,
        )
        ids.set_shape([max_len])
        return img, ids

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------- model / loss ----------

def residual_block(x, filters, *, activation="relu", skip_conv=False, strides=1, dropout=0.0):
    """
    Basic 2x Conv BN + residual add, optional projection on the skip.
    If strides=2 → downsample (both H and W).
    """
    y = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activation)(y)
    if dropout and dropout > 0:
        y = layers.Dropout(dropout)(y)

    y = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    # project the shortcut if shape/stride mismatch
    if skip_conv or (strides != 1) or (x.shape[-1] != filters):
        s = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(x)
        s = layers.BatchNormalization()(s)
    else:
        s = x

    z = layers.Add()([y, s])
    z = layers.Activation(activation)(z)
    if dropout and dropout > 0:
        z = layers.Dropout(dropout)(z)
    return z


# ocr_utils.py
def build_resnet_lstm(
    input_hw=(48, 1024),                # (H, W) — mltu-style default
    channels=3,
    vocab_size=100,
    activation="leaky_relu",            # mltu default
    dropout=0.2,                        # mltu default
):
    """
    Residual CNN backbone (same stride pattern as mltu example)
    → flatten spatial to time
    → BiLSTM(256) + Dropout
    → BiLSTM(64)  + Dropout
    → Dense(vocab+1) + Softmax   (+1 = CTC blank)
    """
    H, W = input_hw
    inputs = layers.Input(shape=(H, W, channels), name="input")

    # Images from cgl-ml pipeline are already [0,1]; if you ever bypass it, you can uncomment:
    # x = layers.Lambda(lambda t: t / 255.0)(inputs)
    x = inputs

    x1 = residual_block(x,   32, activation=activation, skip_conv=True,  strides=1, dropout=dropout)
    x2 = residual_block(x1,  32, activation=activation, skip_conv=True,  strides=2, dropout=dropout)
    x3 = residual_block(x2,  32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x4 = residual_block(x3,  64, activation=activation, skip_conv=True,  strides=2, dropout=dropout)
    x5 = residual_block(x4,  64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = residual_block(x5, 128, activation=activation, skip_conv=True,  strides=2, dropout=dropout)
    x7 = residual_block(x6, 128, activation=activation, skip_conv=True,  strides=1, dropout=dropout)

    x8 = residual_block(x7, 128, activation=activation, skip_conv=True,  strides=2, dropout=dropout)
    x9 = residual_block(x8, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # (B, H', W', C) → (B, T, C) where T = H'*W'
    squeezed = layers.Reshape((-1, x9.shape[-1]))(x9)

    blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    logits = layers.Dense(vocab_size + 1)(blstm)  # +1 for CTC blank
    output = layers.Activation("softmax", name="softmax")(logits)

    return tf.keras.Model(inputs=inputs, outputs=output, name="resnet_lstm_ctc")





def build_crnn(input_hw=(48, 1024), channels=3, vocab_size=100, lstm_units=256) -> tf.keras.Model:
    H, W = input_hw
    inp = tf.keras.Input(shape=(H, W, channels), name="image")
    x = inp
    # keep width downsampling modest: /2, /2, /1 => total /4
    for f, k, pw in [(64, 3, 2), (128, 3, 2), (256, 3, 1)]:
        x = tf.keras.layers.Conv2D(f, k, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, pw))(x)
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)  # (B, W', C)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(x)
    logits = tf.keras.layers.Dense(vocab_size + 1)(x)  # +1 for CTC blank
    out = tf.keras.layers.Activation("softmax", name="softmax")(logits)
    return tf.keras.Model(inp, out, name="crnn_ctc")

def make_ctc_loss(pad_value: int):
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        label_len = tf.math.count_nonzero(tf.not_equal(y_true, pad_value), axis=1, dtype=tf.int32)
        in_len = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        return tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, tf.expand_dims(in_len, -1), tf.expand_dims(label_len, -1)
        )
    return _loss

# def time_steps(width: int, pooling=(2, 2, 1)) -> int:
#     div = 1
#     for p in pooling:
#         div *= p
#     return width // div  # matches build_crnn pooling


def time_steps(width: int, height: int = None, arch: str = "crnn", pooling=(2, 2, 1)) -> int:
    """
    Estimate sequence length (T) for CTC safety filtering.
    - crnn: time along width after pooling (e.g., / (2*2*1) = /4)
    - resnet_lstm: four stride-2 stages on BOTH H and W → T = (H/16) * (W/16)
    """
    arch = (arch or "crnn").lower()
    if arch == "resnet_lstm":
        if height is None:
            raise ValueError("time_steps(...): height is required for arch='resnet_lstm'")
        return max(1, (height // 16) * (width // 16))
    # default: crnn
    div = 1
    for p in pooling:
        div *= p
    return max(1, width // div)


# ---------- metrics / callbacks ----------

# def levenshtein(a: str, b: str) -> int:
#     if a == b:
#         return 0
#     if not a:
#         return len(b)
#     if not b:
#         return len(a)
#     m, n = len(a), len(b)
#     dp = list(range(n + 1))
#     for i, ca in enumerate(a, 1):
#         prev, dp[0] = dp[0], i
#         for j, cb in enumerate(b, 1):
#             prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
#     return dp[n]

class EvalCER(tf.keras.callbacks.Callback):
    """Lightweight CER on a small validation sample; prints CER=... per epoch."""
    def __init__(self, ds: Optional[tf.data.Dataset], vocab: str, max_batches=10):
        super().__init__()
        self.ds = ds.take(max_batches) if ds is not None else None
        self.vocab = list(vocab)

    def on_epoch_end(self, epoch, logs=None):
        if self.ds is None:
            return
        total_err, total_len = 0, 0
        for xb, yb in self.ds:
            preds = self.model.predict(xb, verbose=0)
            in_len = np.ones(preds.shape[0]) * preds.shape[1]
            dec, _ = tf.keras.backend.ctc_decode(preds, input_length=in_len, greedy=True)
            pred = dec[0]
            for i in range(pred.shape[0]):
                gt_ids = yb[i].numpy()
                gt = "".join([self.vocab[idx] for idx in gt_ids if idx < len(self.vocab)])
                pr = "".join([self.vocab[idx] for idx in pred[i].numpy() if idx < len(self.vocab)])
                total_err += levenshtein(gt, pr)
                total_len += max(1, len(gt))
        cer = total_err / total_len if total_len > 0 else 0.0
        print(f"CER={cer:.6f}", flush=True)


def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[n]

def _cer_batch_py(y_true_np, y_pred_np, vocab_bytes, pad_value):
    vocab = list(vocab_bytes.decode("utf-8"))
    err = 0
    total = 0
    B = y_true_np.shape[0]
    for i in range(B):
        # strip padding from y_true
        gt_ids = [int(t) for t in y_true_np[i].tolist() if int(t) != int(pad_value)]
        gt = "".join(vocab[j] for j in gt_ids if 0 <= j < len(vocab))
        # y_pred from ctc_decode is padded with -1
        pr_ids = [int(t) for t in y_pred_np[i].tolist() if int(t) >= 0]
        pr = "".join(vocab[j] for j in pr_ids if 0 <= j < len(vocab))
        err += _levenshtein(gt, pr)
        total += max(1, len(gt))
    return np.array(err, np.float64), np.array(total, np.float64)

def _wer_batch_py(y_true_np, y_pred_np, vocab_bytes, pad_value):
    vocab = list(vocab_bytes.decode("utf-8"))
    err = 0
    total = 0
    B = y_true_np.shape[0]
    for i in range(B):
        gt_ids = [int(t) for t in y_true_np[i].tolist() if int(t) != int(pad_value)]
        gt = "".join(vocab[j] for j in gt_ids if 0 <= j < len(vocab))
        pr_ids = [int(t) for t in y_pred_np[i].tolist() if int(t) >= 0]
        pr = "".join(vocab[j] for j in pr_ids if 0 <= j < len(vocab))
        gt_words = gt.split()
        pr_words = pr.split()
        err += _levenshtein(" ".join(gt_words), " ".join(pr_words))  # simple proxy
        total += max(1, len(gt_words))
    return np.array(err, np.float64), np.array(total, np.float64)

class StreamingCER(tf.keras.metrics.Metric):
    def __init__(self, vocab: str, pad_value: int, name="CER", **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocab_bytes = tf.convert_to_tensor(vocab.encode("utf-8"))
        self.pad_value = tf.convert_to_tensor(pad_value, dtype=tf.int32)
        self.err = self.add_weight(name="err", initializer="zeros", dtype=tf.float64)
        self.tot = self.add_weight(name="tot", initializer="zeros", dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        T = tf.shape(y_pred)[1]
        in_len = tf.fill([tf.shape(y_pred)[0]], T)
        decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=in_len, greedy=True)
        pred = tf.cast(decoded[0], tf.int32)

        err, tot = tf.numpy_function(
            _cer_batch_py,
            [y_true, pred, self.vocab_bytes, self.pad_value],
            Tout=[tf.float64, tf.float64],
        )
        # 👇 ensure scalar shape & dtype
        err = tf.reshape(tf.cast(err, tf.float64), [])
        tot = tf.reshape(tf.cast(tot, tf.float64), [])
        self.err.assign_add(err)
        self.tot.assign_add(tot)

    def result(self):
        return tf.math.divide_no_nan(self.err, self.tot)

    def reset_state(self):
        self.err.assign(0.0); self.tot.assign(0.0)

class StreamingWER(tf.keras.metrics.Metric):
    def __init__(self, vocab: str, pad_value: int, name="WER", **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocab_bytes = tf.convert_to_tensor(vocab.encode("utf-8"))
        self.pad_value = tf.convert_to_tensor(pad_value, dtype=tf.int32)
        self.err = self.add_weight(name="err", initializer="zeros", dtype=tf.float64)
        self.tot = self.add_weight(name="tot", initializer="zeros", dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        T = tf.shape(y_pred)[1]
        in_len = tf.fill([tf.shape(y_pred)[0]], T)
        decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=in_len, greedy=True)
        pred = tf.cast(decoded[0], tf.int32)

        err, tot = tf.numpy_function(
            _wer_batch_py,
            [y_true, pred, self.vocab_bytes, self.pad_value],
            Tout=[tf.float64, tf.float64],
        )
        # 👇 ensure scalar shape & dtype
        err = tf.reshape(tf.cast(err, tf.float64), [])
        tot = tf.reshape(tf.cast(tot, tf.float64), [])
        self.err.assign_add(err)
        self.tot.assign_add(tot)

    def result(self):
        return tf.math.divide_no_nan(self.err, self.tot)

    def reset_state(self):
        self.err.assign(0.0); self.tot.assign(0.0)