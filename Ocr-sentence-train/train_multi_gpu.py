#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cognilabel OCR Training (CGL-DATA compliant)

- Uses cgl_data.resolver for dataset access
- Uses OutputManager for outputs
- Compatible with local / S3 / future providers
"""

import os, json, random, time, platform
import threading
import subprocess
from pathlib import Path

import signal



import numpy as np
import tensorflow as tf
import re

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.tensorflow.dataProvider import DataProvider
# from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from metrics import SafeCERMetric, SafeWERMetric
from mltu.annotations.images import CVImage

from loss import CTCloss

from model import train_model
from last_state_writer import LastStateWriter

# ✅ NEW: Cognilabel data + outputs
from cgl_data.resolver import resolve_split, resolve_resume_weights
from cgl_data.outputs import OutputManager
from uploadUtil import UploadBestOnImprove, UploadLastEveryEpoch
from tensorflow.keras import mixed_precision
from cgl_data.logging.keras.logger import CGLKerasLogger, CGLEarlyStopping
from cgl_data.logging.emitter import emit
from cgl_data.logging.events import CGL_EVAL
from preprocessors import NormalizePolarity, NormalizeContrast, ToGrayscale, BinarizeNormalize
import hashlib
import cv2
from val_diagnostics import run_val_diagnostics


def handle_sigterm(signum, frame):
    print("[SPOT] interruption signal received - checkpointing...")
    emit("CGL_SPOT_INTERRUPT", {
        "timestamp": int(time.time())
    })

signal.signal(signal.SIGTERM, handle_sigterm)

# mixed_precision.set_global_policy("mixed_float16")



# --------------------------------------------------
# helpers
# --------------------------------------------------
def start_gpu_monitor(interval=10):
    def monitor():
        while True:
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
                ).decode("utf-8").strip()

                lines = result.split("\n")
                for i, line in enumerate(lines):
                    util, mem_used, mem_total = line.split(", ")
                    print(f"[GPU {i}] util={util}% mem={mem_used}/{mem_total} MB")
            except Exception as e:
                print(f"[gpu-monitor] error: {e}")

            time.sleep(interval)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

def setup_tf():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

def setup_distribution():
    gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(gpus)

    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[dist] MirroredStrategy enabled with {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()
        print(f"[dist] Single-device strategy with {strategy.num_replicas_in_sync} replica(s)")

    return strategy


def build_vocab(samples):
    chars = set()
    for _, txt in samples:
        chars.update(txt)
    return sorted(chars)



SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,،؛:!؟?\…])")

def normalize_ocr_label(txt: str) -> str:
    """
    Keep OCR labels in logical Unicode order.

    This must NOT convert Arabic labels to visual order.
    This must NOT move trailing punctuation to the front.
    This must NOT reverse text.

    Safe cleanup only:
    - trim
    - collapse whitespace
    - remove space before punctuation
    """
    txt = txt or ""
    txt = " ".join(txt.strip().split())
    txt = SPACE_BEFORE_PUNCT_RE.sub(r"\1", txt)
    return txt

def env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)

    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def apply_output_time_axis_policy(model, reverse_time_axis: bool):
    """
    For CRNN/CTC OCR.

    Normal model output shape:
    [batch, time_steps, vocab + blank]

    For RTL logical labels, CTC needs model time order to match:
    label[0] -> label[-1]

    Arabic image x-axis is usually:
    sentence end -> sentence start when scanned left-to-right

    So for RTL we reverse the model output time axis once here.

    By wrapping the model output, loss, CER/WER metrics, diagnostics,
    saved final.keras, and inference all see the same corrected order.
    """
    if not reverse_time_axis:
        return model

    y = tf.keras.layers.Lambda(
        lambda t: tf.reverse(t, axis=[1]),
        name="rtl_reverse_time_axis",
    )(model.output)

    return tf.keras.Model(
        inputs=model.input,
        outputs=y,
        name=f"{model.name}_rtl_time_corrected",
    )


def load_jsonl(path, text_dir="ltr", max_reasonable_len=1000):
    items = []
    max_len = 0
    debug_label_examples = 0
    
    def normalize_label(txt: str) -> str:
        return normalize_ocr_label(txt)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            rec = json.loads(line)  

            img = rec.get("file") or rec.get("image") or rec.get("path")
            txt = rec.get("text") if rec.get("text") is not None else rec.get("transcription")

            if not img or txt is None:
                continue

            if not isinstance(txt, str):
                print(f"[data] skipping non-string label at line {line_no}: {type(txt).__name__}")
                continue

            txt = normalize_label(txt)
            
            if text_dir == "rtl" and debug_label_examples < 5:
                print(
                    "[label][logical]",
                    "first=", repr(txt[0] if txt else ""),
                    "last=", repr(txt[-1] if txt else ""),
                    "label=", repr(txt),
                )
                debug_label_examples += 1

            if not txt:
                print(f"[data] skipping empty/whitespace-only label at line {line_no}")
                continue
            

            if len(txt) > max_reasonable_len:
                print(f"[data] skipping suspiciously long label at line {line_no}: len={len(txt)} img={img}")
                continue

            items.append((img, txt))
            max_len = max(max_len, len(txt))

    return items, max_len


def absolutize_items(items, local_root: Path):
    out = []
    for img, txt in items:
        p = Path(img)

        # If JSONL has relative paths like "images/xxx" or "xxx"
        if not p.is_absolute():
            p = (local_root / p).resolve()

        out.append((str(p), txt))
    return out

def cache_deterministic_preprocessed_items(items, preprocessors, cache_dir: Path, split_name="unknown"):
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []

    # skip ImageReader because we will load manually here
    effective_pre = preprocessors[1:] if preprocessors else []

    cache_hits = 0
    cache_misses = 0
    failures = 0

    for img_path, txt in items:
        src = str(img_path)
        digest = hashlib.md5(src.encode("utf-8")).hexdigest()
        cached_path = cache_dir / f"{digest}.png"

        try:
            if not cached_path.exists():
                img = CVImage(src).numpy()
                data = (img, txt)

                for proc in effective_pre:
                    data = proc(data)

                processed_img, _ = data

                if processed_img is None:
                    raise ValueError("Processed image is None")

                if len(processed_img.shape) not in (2, 3):
                    raise ValueError(f"Unexpected processed image shape: {processed_img.shape}")

                ok = cv2.imwrite(str(cached_path), processed_img)
                if not ok:
                    raise RuntimeError(f"cv2.imwrite failed for {cached_path}")

                cache_misses += 1
            else:
                cache_hits += 1

            out.append((str(cached_path), txt))

        except Exception as e:
            failures += 1
            print(f"[pipeline][{split_name}] FAILED src={src} err={e}")
            raise

    print(
        f"[pipeline][{split_name}] cache total={len(items)} "
        f"hits={cache_hits} misses={cache_misses} failures={failures}"
    )

    return out, {
        "split": split_name,
        "total": len(items),
        "hits": cache_hits,
        "misses": cache_misses,
        "failures": failures,
        "cache_dir": str(cache_dir),
    }

def inspect_sample_image_shape(items, split_name="unknown"):
    if not items:
        return None

    sample_path, _ = items[0]
    img = CVImage(sample_path).numpy()
    print(f"[preprocess][{split_name}] sample cached image shape={img.shape} dtype={img.dtype} path={sample_path}")
    return {
        "split": split_name,
        "sample_path": sample_path,
        "shape": list(img.shape),
        "dtype": str(img.dtype)
    }
        
def build_preprocessors_from_env(preprocess_json):
    pre = [ImageReader(CVImage)]

    pjson = (preprocess_json or "").strip()
    if not pjson:
        return pre

    try:
        spec = json.loads(pjson)

        if not isinstance(spec, list):
            raise ValueError("CGL_PREPROCESS_JSON must decode to a list")

        for item in spec:
            if not isinstance(item, dict):
                continue

            key = item.get("key")
            config = item.get("config") or {}

            if key == "GrayscaleNormalize":
                pre.append(
                    ToGrayscale(
                        strength=float(config.get("strength", 1.0)),
                        keep_3ch=bool(config.get("keep3Channels", True)),
                    )
                )

            elif key == "NormalizePolarity":
                pre.append(
                    NormalizePolarity(
                        target=config.get("target", "dark_on_light"),
                        mode=config.get("mode", "auto"),
                    )
                )

            elif key == "ContrastNormalize":
                tgs = config.get("tileGridSize", 8)
                if isinstance(tgs, int):
                    tgs = (tgs, tgs)
                elif isinstance(tgs, list):
                    tgs = tuple(tgs)

                pre.append(
                    NormalizeContrast(
                        method=config.get("method", "clahe"),
                        clip_limit=float(config.get("clipLimit", 2.0)),
                        tile_grid_size=tgs,
                    )
                )
            elif key == "BinarizeNormalize":
                pre.append(
                    BinarizeNormalize(
                        method=config.get("method", "adaptive_gaussian"),
                        mode=config.get("mode", "soft"),
                        block_size=int(config.get("blockSize", 21)),
                        C=float(config.get("C", 10)),
                        threshold=int(config.get("threshold", 160)),
                        keep_3ch=bool(config.get("keep3Channels", True)),
                    )
                )

            else:
                raise ValueError(f"Unknown preprocessor key: {key}")
                # print(f"[CGL] unknown preprocessor skipped: {key}")

    except Exception as e:
        print(f"[CGL] bad CGL_PREPROCESS_JSON ({e}); using NO preprocessors")

    return pre

def build_augmentors_from_env(aug_json):
    aug = []
    ajson = (aug_json or "").strip()

    if not ajson:
        return aug

    try:
        spec = json.loads(ajson)

        if not isinstance(spec, list):
            raise ValueError("CGL_AUG_JSON must decode to a list")

        for item in spec:
            if not isinstance(item, dict):
                continue

            name = item.get("key")
            config = item.get("config") or {}

            if name == "RandomBrightness":
                aug.append(RandomBrightness())

            elif name == "RandomErodeDilate":
                aug.append(RandomErodeDilate())

            elif name == "RandomSharpen":
                aug.append(RandomSharpen())

            elif name == "RandomRotate":
                p = float(config.get("p", 0.5))
                degrees = float(config.get("degrees", 3.0))
                aug.append(RandomRotate(p, degrees))

            else:
                raise ValueError(f"Unknown augmenter key: {name}")
                # print(f"[CGL] unknown augmenter skipped: {name}")

    except Exception as e:
        print(f"[CGL] bad CGL_AUG_JSON ({e}); using NO augmenters")

    return aug
# --------------------------- pipeline from ENV ---------------------------
def build_pipeline_from_env(aug_json, trf_json, vocab, max_label_len, H, W):
    pre = [ImageReader(CVImage)]
    
    # if any(str(ENV.get(k, "")).startswith("s3://")
    #     for k in ("IMAGE_BASE","IMAGE_BASE_VAL","IMAGE_BASE_TEST")):
    #         pre = [_S3FetchBeforeRead(ENV["CACHE_DIR"])] + pre

    # ---- Augmenters: only from JSON; otherwise default to NONE ----
    aug = []
    ajson = aug_json.strip()
    if ajson:
        try:
            spec = json.loads(ajson)
            for item in spec:
                name = item.get("name")
                p = float(item.get("p", 0.0))
                if name == "RandomRotate":
                    aug.append(RandomRotate(p, float(item.get("degrees", 3.0))))
                elif name == "RandomGaussianBlur":
                    aug.append(RandomGaussianBlur(p, float(item.get("sigma", 1.2))))
                elif name == "RandomElasticTransform":
                    alpha = item.get("alpha", (0.0, 0.06)); sigma = item.get("sigma", (0.004, 0.012))
                    aug.append(RandomElasticTransform(p, tuple(alpha), tuple(sigma)))
                elif name == "RandomBrightness":
                    # aug.append(RandomBrightness(p, int(item.get("delta", 25))))
                    aug.append(RandomBrightness())                
                elif name == "RandomErodeDilate":
                    # aug.append(RandomErodeDilate(p, int(item.get("ksize", 3)), int(item.get("iters", 1))))
                    aug.append(RandomErodeDilate())
                elif name == "RandomSharpen":
                    # aug.append(RandomSharpen(p, float(item.get("radius", 1.0)), int(item.get("percent", 150)), int(item.get("threshold", 3))))
                    aug.append(RandomSharpen())
        except Exception as e:
            print(f"[CGL] bad CGL_AUG_JSON ({e}); using NO augmenters")

    # ---- Transformers: from JSON; otherwise use EXACT default set ----
    trf = []
    # tjson = trf_json.strip()
    # if tjson:
    #     try:
    #         spec = json.loads(tjson)
    #         has_index = has_pad = False
    #         for item in spec:
    #             name = item.get("name")
    #             if name == "ImageResizer":
    #                 trf.append(ImageResizer(int(item.get("width", W)), int(item.get("height", H)),
    #                                         keep_aspect_ratio=bool(item.get("keep_aspect", True))))
    #             elif name == "LabelIndexer":
    #                 trf.append(LabelIndexer(list(vocab))); has_index = True
    #             elif name == "LabelPadding":
    #                 trf.append(LabelPadding(max_word_length=int(item.get("maxLen", max_label_len)),
    #                                         padding_value=len(vocab))); has_pad = True
    #             elif name == "ImageNormalizer":
    #                 # allowed if explicitly requested in JSON
    #                 trf.append(ImageNormalizer(bool(item.get("transpose_axis", False))))
    #         if not has_index:
    #             trf.append(LabelIndexer(list(vocab)))
    #         if not has_pad:
    #             trf.append(LabelPadding(max_word_length=max_label_len, padding_value=len(vocab)))
    #     except Exception as e:
    #         print(f"[CGL] bad CGL_TRF_JSON ({e}); falling back to defaults")

    # if not trf:
    #     # EXACT defaults you asked for (no ImageNormalizer)
    #     trf = [
    #         ImageResizer(int(W), int(H), keep_aspect_ratio=True),
    #         LabelIndexer(list(vocab)),
    #         LabelPadding(max_word_length=max_label_len, padding_value=len(vocab)),
    #     ]

    return pre, aug, trf


def get_image_padding_color_from_preprocess_ops(preprocess_ops):
    """
    Choose ImageResizer padding color to match deterministic polarity normalization.
    Default: black, to preserve existing behavior if no polarity op is present.
    """
    for op in preprocess_ops or []:
        if not isinstance(op, dict):
            continue
        if op.get("key") == "NormalizePolarity":
            cfg = op.get("config") or {}
            target = cfg.get("target", "dark_on_light")

            if target == "dark_on_light":
                return (255, 255, 255)

            if target == "light_on_dark":
                return (0, 0, 0)

    return (0, 0, 0)

def env(name, default=None, cast=str):
    val = os.getenv(name, default)
    if val is None:
        return None
    try:
        return cast(val)
    except Exception:
        raise ValueError(f"Invalid value for {name}: {val}")

def _as_tensor_spec(value, name="tensor"):
    """
    Build a TensorSpec from the actual DataProvider output.
    Keeps this compatible with TF 2.18 / Keras 3 without changing mltu.DataProvider.
    """
    arr = np.asarray(value)
    if arr.dtype == np.dtype("O"):
        raise RuntimeError(
            f"{name} has object dtype. DataProvider must return dense numeric arrays, "
            "not ragged/object arrays, for MirroredStrategy."
        )
    return tf.TensorSpec(shape=arr.shape, dtype=tf.as_dtype(arr.dtype))


def repeat_provider(dp, steps, name="train"):
    """
    Infinite epoch-safe wrapper around mltu.DataProvider.

    This avoids Keras 3 / MirroredStrategy exhausting the PyDataset-style provider
    after the first epoch, while keeping the original DataProvider pipeline intact.
    """
    epoch = 0

    while True:
        for i in range(steps):
            batch = dp[i]

            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise RuntimeError(
                    f"[repeat_provider][{name}] Expected (x, y) from provider, "
                    f"got {type(batch)}"
                )

            x, y = batch

            # Convert once here so tf.data sees stable dense arrays.
            x = np.asarray(x)
            y = np.asarray(y)

            if x.shape[0] <= 0:
                raise RuntimeError(f"[repeat_provider][{name}] Empty x batch at index={i}")

            if y.shape[0] <= 0:
                raise RuntimeError(f"[repeat_provider][{name}] Empty y batch at index={i}")

            yield x, y

        epoch += 1

        # Preserve mltu shuffle/epoch behavior if present.
        if hasattr(dp, "on_epoch_end"):
            try:
                dp.on_epoch_end()
            except Exception as e:
                print(f"[repeat_provider][{name}] on_epoch_end failed: {e}")


def make_repeating_tf_dataset(dp, steps, name="train"):
    """
    Convert the existing mltu DataProvider into a repeatable tf.data.Dataset.
    Does not replace your preprocessing/augmentation/transformer logic.
    """
    sample_x, sample_y = dp[0]

    sample_x = np.asarray(sample_x)
    sample_y = np.asarray(sample_y)

    print(f"[tfdata][{name}] sample_x.shape={sample_x.shape} dtype={sample_x.dtype}")
    print(f"[tfdata][{name}] sample_y.shape={sample_y.shape} dtype={sample_y.dtype}")

    if sample_x.shape[0] <= 0 or sample_y.shape[0] <= 0:
        raise RuntimeError(f"[tfdata][{name}] Provider returned an empty first batch")

    output_signature = (
        _as_tensor_spec(sample_x, f"{name}.x"),
        _as_tensor_spec(sample_y, f"{name}.y"),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: repeat_provider(dp, steps, name),
        output_signature=output_signature,
    )

    # Important: let MirroredStrategy shard by batch, not by file.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    return ds.with_options(options).prefetch(tf.data.AUTOTUNE)


def decode_batch_greedy_np(y_pred, vocab, blank_index):
    """
    Simple greedy CTC-style decode for diagnostics.
    Assumes y_pred is already probabilities from softmax:
    [batch, time, vocab + blank]
    """
    idxs = np.argmax(y_pred, axis=-1)

    texts = []
    for row in idxs:
        prev = None
        chars = []

        for k in row:
            k = int(k)

            # CTC collapse repeats, remove blank
            if k != blank_index and k != prev:
                if 0 <= k < len(vocab):
                    chars.append(vocab[k])

            prev = k

        texts.append("".join(chars))

    return texts


def label_row_to_text(y_row, vocab, blank_index):
    chars = []
    for k in np.asarray(y_row).tolist():
        k = int(k)
        if 0 <= k < len(vocab) and k != blank_index:
            chars.append(vocab[k])
    return "".join(chars)


def simple_cer(ref, hyp):
    """
    Tiny dependency-free CER for diagnostics only.
    """
    ref = ref or ""
    hyp = hyp or ""

    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur

    return prev[m] / max(1, n)


def inspect_model_direction(model, text_dir, reverse_time_axis):
    """
    Confirms whether the model architecture contains the RTL reversal layer.
    """
    layer_names = [layer.name for layer in model.layers]
    has_rtl_layer = "rtl_reverse_time_axis" in layer_names

    print("[direction-check] text_dir=", text_dir)
    print("[direction-check] reverse_time_axis=", reverse_time_axis)
    print("[direction-check] has_rtl_reverse_time_axis_layer=", has_rtl_layer)

    if reverse_time_axis and not has_rtl_layer:
        print(
            "[direction-check][WARNING] reverse_time_axis=True but "
            "rtl_reverse_time_axis layer was not found in model."
        )

    if not reverse_time_axis and has_rtl_layer:
        print(
            "[direction-check][WARNING] reverse_time_axis=False but "
            "rtl_reverse_time_axis layer exists in model."
        )

    # Print the last few layers so you can verify final order in logs
    print("[direction-check] last layers:")
    for layer in model.layers[-8:]:
        print(f"  - {layer.name}: {layer.__class__.__name__}")


def run_rtl_direction_batch_diagnostic(
    model,
    data_provider,
    vocab,
    blank_index,
    text_dir,
    reverse_time_axis,
    name="train",
    max_samples=3,
):
    """
    Shows whether decoded output is being read in the intended direction.

    For RTL fixed model:
    - normal decode should be the corrected/logical direction
    - manually reversed decode is shown only as a contrast
    """
    try:
        x_batch, y_batch = data_provider[0]
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)

        y_pred = model.predict(x_batch, verbose=0)

        # Normal model output decode
        pred_normal = decode_batch_greedy_np(y_pred, vocab, blank_index)

        # Diagnostic-only opposite direction decode
        pred_opposite = decode_batch_greedy_np(y_pred[:, ::-1, :], vocab, blank_index)

        print(
            f"[rtl-diag][{name}] text_dir={text_dir} "
            f"reverse_time_axis={reverse_time_axis} "
            f"x_shape={x_batch.shape} y_shape={y_batch.shape} "
            f"y_pred_shape={y_pred.shape}"
        )

        for i in range(min(max_samples, x_batch.shape[0])):
            ref = label_row_to_text(y_batch[i], vocab, blank_index)
            normal = pred_normal[i]
            opposite = pred_opposite[i]

            cer_normal = simple_cer(ref, normal)
            cer_opposite = simple_cer(ref, opposite)

            print(f"[rtl-diag][{name}][sample {i}]")
            print(f"  REF first/last: {repr(ref[:1])} / {repr(ref[-1:] if ref else '')}")
            print(f"  REF: {repr(ref)}")
            print(f"  NORMAL_DECODE first/last: {repr(normal[:1])} / {repr(normal[-1:] if normal else '')}")
            print(f"  NORMAL_DECODE: {repr(normal)}")
            print(f"  NORMAL_CER: {cer_normal:.4f}")
            print(f"  OPPOSITE_TIME_DECODE first/last: {repr(opposite[:1])} / {repr(opposite[-1:] if opposite else '')}")
            print(f"  OPPOSITE_TIME_DECODE: {repr(opposite)}")
            print(f"  OPPOSITE_CER: {cer_opposite:.4f}")

    except Exception as e:
        print(f"[rtl-diag][{name}] failed: {e}")
        
class DirectionDecodeProbe(tf.keras.callbacks.Callback):
    """
    Periodically decodes the same fixed batch in normal output order and opposite
    time order. For the fixed RTL model, NORMAL should become better than OPPOSITE
    as training starts learning.
    """

    def __init__(
        self,
        data_provider,
        vocab,
        blank_index,
        text_dir,
        reverse_time_axis,
        name="train",
        every_n_epochs=5,
        max_samples=3,
    ):
        super().__init__()
        self.data_provider = data_provider
        self.vocab = vocab
        self.blank_index = blank_index
        self.text_dir = text_dir
        self.reverse_time_axis = reverse_time_axis
        self.name = name
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_samples = max_samples

        x_batch, y_batch = self.data_provider[0]
        self.x_batch = np.asarray(x_batch)
        self.y_batch = np.asarray(y_batch)

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1

        if epoch_num != 1 and epoch_num % self.every_n_epochs != 0:
            return

        y_pred = self.model.predict(self.x_batch, verbose=0)

        pred_normal = decode_batch_greedy_np(y_pred, self.vocab, self.blank_index)
        pred_opposite = decode_batch_greedy_np(y_pred[:, ::-1, :], self.vocab, self.blank_index)

        normal_cers = []
        opposite_cers = []

        print(
            f"[direction-probe][{self.name}] epoch={epoch_num} "
            f"text_dir={self.text_dir} reverse_time_axis={self.reverse_time_axis}"
        )

        for i in range(min(self.max_samples, self.x_batch.shape[0])):
            ref = label_row_to_text(self.y_batch[i], self.vocab, self.blank_index)
            normal = pred_normal[i]
            opposite = pred_opposite[i]

            cer_n = simple_cer(ref, normal)
            cer_o = simple_cer(ref, opposite)

            normal_cers.append(cer_n)
            opposite_cers.append(cer_o)

            print(f"[direction-probe][{self.name}][sample {i}]")
            print(f"  REF: {repr(ref)}")
            print(f"  NORMAL: {repr(normal)} CER={cer_n:.4f}")
            print(f"  OPPOSITE: {repr(opposite)} CER={cer_o:.4f}")

        print(
            f"[direction-probe][{self.name}] "
            f"mean_normal_cer={float(np.mean(normal_cers)):.4f} "
            f"mean_opposite_cer={float(np.mean(opposite_cers)):.4f}"
        )


# --------------------------------------------------
# main
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # 🔹 ENV-ONLY CONFIG (Cognilabel contract)
    # --------------------------------------------------

    # Generic Cognilabel arguments
    dataset_uri = env("CGL_DATASET_URI", cast=str)
    outputs_uri = env("CGL_OUTPUTS_URI", cast=str)
    job_id      = env("CGL_JOB_ID", cast=str)
    job_name    = env("CGL_JOB_NAME", "ocr-train", str)
    cache_dir   = env("CACHE_DIR", "/tmp/cgl_cache", str)

    if not dataset_uri or not outputs_uri or not job_id:
        raise RuntimeError("CGL_DATASET_URI, CGL_OUTPUTS_URI, and CGL_JOB_ID are required")

    # 🔹 Hyperparams
    epochs      = env("CGL_EPOCHS", 50, int)
    # batch_size = env("CGL_BATCH_SIZE", 8, int)
    client_batch_size = env("CGL_BATCH_SIZE", 8, int)
    lr         = env("CGL_LR", 1e-3, float)
    width      = env("CGL_WIDTH", 1048, int)
    height     = env("CGL_HEIGHT", 96, int)
    dropout    = env("CGL_DROPOUT", 0.2, float)
    activation = env("CGL_ACTIVATION", "leaky_relu", str)

    # 🔹 Behavior / flags
    text_dir = env("CGL_TEXT_DIR", "ltr", str).strip().lower()

    if text_dir not in ("ltr", "rtl"):
        raise ValueError(f"Invalid CGL_TEXT_DIR={text_dir}. Expected 'ltr' or 'rtl'.")
    
    early_patience = env("CGL_EARLY_STOP_PATIENCE", 10, int)
    reduce_lr_patience = env("CGL_REDUCE_LR_PATIENCE", 10, int)

    resume_mode  = env("CGL_RESUME_MODE", "none", str)
    resume_which = env("CGL_RESUME_WHICH", "best", str)
    resume_path  = env("CGL_RESUME_PATH", "", str)

    ft_lr = env("CGL_FT_LR", None, float)
    freeze_cnn_epochs = env("CGL_FREEZE_CNN_EPOCHS", 0, int)

    augment_json = env("CGL_AUG_JSON", "", str)
    preprocess_json = env("CGL_PREPROCESS_JSON", "", str)
    tranformer_json = env("CGL_TRF_JSON", "", str)  # typo preserved intentionally
    spot_enabled = os.getenv("CGL_ENABLE_SPOT_RESUME", "false") == "true"
    spot_ckpt = Path("/opt/ml/checkpoints/ckpt.weights.h5")
    instance_type = os.getenv("CGL_INSTANCE_TYPE", "unknown")
    
    label_order_policy = "logical"

    # Default: auto-enable for RTL, off for LTR.
    # You can override with CGL_REVERSE_TIME_AXIS=true/false.
    reverse_time_axis = env_bool(
        "CGL_REVERSE_TIME_AXIS",
        default=(text_dir == "rtl"),
    )

    sequence_axis_policy = (
        "rtl_reverse_time_axis" if reverse_time_axis else "ltr_native_time_axis"
    )
    
    

    # --------------------------------------------------
    # 🔹 Setup
    # --------------------------------------------------

    setup_tf()
    strategy = setup_distribution()
    
    start_gpu_monitor(interval=60)
    
    num_gpus = strategy.num_replicas_in_sync
    cache_root = Path(cache_dir)
    
    print("[TF] physical GPUs:", tf.config.list_physical_devices("GPU"))
    print("[TF] logical GPUs:", tf.config.list_logical_devices("GPU"))
    print("[TF] num_replicas:", strategy.num_replicas_in_sync)
    
    emit("CGL_ENV", {
        "spot_enabled": spot_enabled,
        "instance_type": instance_type,
        "num_gpus": num_gpus,
        "preprocess_json": preprocess_json,
        "augment_json": augment_json,
        "preprocess_enabled": bool((preprocess_json or "").strip()),
        "augment_enabled": bool((augment_json or "").strip()),
        "text_dir": text_dir,
        "label_order_policy": label_order_policy,
        "reverse_time_axis": reverse_time_axis,
        "sequence_axis_policy": sequence_axis_policy,
            })
        
    
    # per_gpu_batch_size = max(1, client_batch_size // max(1, num_gpus))
    # effective_global_batch_size = per_gpu_batch_size * max(1, num_gpus)
    per_gpu_batch_size = client_batch_size
    effective_global_batch_size = client_batch_size * max(1, num_gpus)

    if effective_global_batch_size != client_batch_size:
        print(
            f"[batch] requested global batch={client_batch_size}, "
            f"adjusted effective global batch={effective_global_batch_size} "
            f"(per_gpu_batch_size={per_gpu_batch_size}, gpus={num_gpus})"
        )
    else:
        print(
            f"[batch] global batch={effective_global_batch_size}, "
            f"per_gpu_batch_size={per_gpu_batch_size}, gpus={num_gpus}"
        )

    outputs = OutputManager(
        outputs_uri=outputs_uri,
        cache_root=cache_root,
        job_id=job_id,
        job_name=job_name,
        async_uploads=True,
        max_workers=2,
    )

    refs = outputs.refs()
    refs.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Snapshot of env-driven config
    train_args_snapshot = {
        "dataset_uri": dataset_uri,
        "outputs_uri": outputs_uri,
        "job_id": job_id,
        "job_name": job_name,
        "epochs": epochs,
        "client_batch_size": client_batch_size,
        "per_gpu_batch_size": per_gpu_batch_size,
        "effective_global_batch_size": effective_global_batch_size,
        "num_gpus": num_gpus,
        "lr": lr,
        "width": width,
        "height": height,
        "dropout": dropout,
        "activation": activation,
        "text_dir": text_dir,
        "label_order_policy": label_order_policy,
        "reverse_time_axis": reverse_time_axis,
        "sequence_axis_policy": sequence_axis_policy,
        "early_patience": early_patience,
        "reduce_lr_patience": reduce_lr_patience,
        "resume_mode": resume_mode,
        "resume_which": resume_which,
        "resume_path": resume_path,
        "ft_lr": ft_lr,
        "freeze_cnn_epochs": freeze_cnn_epochs,
        "augment_json": augment_json,
        "tranformer_json": tranformer_json,
    }

    with open(refs.artifacts_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(train_args_snapshot, f, indent=2)

    config_snapshot = {
        "width": width,
        "height": height,
        "client_batch_size": client_batch_size,
        "per_gpu_batch_size": per_gpu_batch_size,
        "effective_global_batch_size": effective_global_batch_size,
        "num_gpus": num_gpus,
        "lr": lr,
        "dropout": dropout,
        "activation": activation,
        "text_dir": text_dir,
        "early_patience": early_patience,
        "reduce_lr_patience": reduce_lr_patience,
        "label_order_policy": label_order_policy,
        "reverse_time_axis": reverse_time_axis,
        "sequence_axis_policy": sequence_axis_policy,
    }   
    

    with open(refs.artifacts_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)
        
    

    # --------------------------------------------------
    # 🔹 Dataset
    # --------------------------------------------------

    train_ref = resolve_split(dataset_uri, "train", cache_root)
    val_ref = None
    try:
        val_ref = resolve_split(dataset_uri, "val", cache_root)
    except Exception as e:
        print(f"[dataset] validation split not found; training without validation. reason={e}")


    test_ref = None
    
    try:
        test_ref = resolve_split(dataset_uri, "test", cache_root)
    except Exception:
        pass

    dataset_manifest = {
        "dataset_uri": dataset_uri,
        "splits": {
            "train": {
                "annotations": str(train_ref.local_annotations_path),
                "images_dir": str(train_ref.local_images_dir),
            },
        
        },
    }
    
    if val_ref:
        dataset_manifest["splits"]["val"] = {
            "annotations": str(val_ref.local_annotations_path),
            "images_dir": str(val_ref.local_images_dir),
        }

    if test_ref:
        dataset_manifest["splits"]["test"] = {
            "annotations": str(test_ref.local_annotations_path),
            "images_dir": str(test_ref.local_images_dir),
        }

    with open(refs.artifacts_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    # OCR expects JSONL annotations
    train_items, train_max = load_jsonl(train_ref.local_annotations_path, text_dir or "ltr")
    train_items = absolutize_items(train_items, train_ref.local_root)
    
    
    val_items = []
    val_max = 0

    if val_ref:
        val_items, val_max = load_jsonl(val_ref.local_annotations_path, text_dir or "ltr")
        val_items = absolutize_items(val_items, val_ref.local_root)
        
    test_items = []
    test_max = 0
    if test_ref:
        test_items,  test_max  = load_jsonl(test_ref.local_annotations_path, text_dir or "ltr") if test_ref else ([], 0)
        test_items = absolutize_items(test_items, test_ref.local_root)
    else:
        test_items = []

    all_label_items = train_items + val_items + test_items
    vocab = build_vocab(all_label_items)
    
    blank_index = len(vocab)

    max_len = max(
        [x for x in [train_max, val_max, test_max] if x and x > 0],
        default=0,
    )

    if max_len <= 0:
        raise RuntimeError("No valid OCR labels found in train/val/test splits")
    
    print(f"[dataset] train samples: {len(train_items)}")
    print(f"[dataset] val samples: {len(val_items)}")
    print(f"[dataset] test samples: {len(test_items)}")
    print(f"[dataset] vocab size: {len(vocab)}")
    print(f"[dataset] max label length: {max_len}")

    # --------------------------------------------------
    # 🔹 Data providers
    # --------------------------------------------------

    train_augmentors = build_augmentors_from_env(augment_json)
    shared_preprocessors = build_preprocessors_from_env(preprocess_json)
    
    
    
    def describe_ops(spec_json):
        try:
            spec = json.loads(spec_json) if (spec_json or "").strip() else []
            if not isinstance(spec, list):
                return []
            return [
                {
                    "key": item.get("key"),
                    "config": item.get("config") or {}
                }
                for item in spec if isinstance(item, dict)
            ]
        except Exception:
            return []

    preprocess_ops = describe_ops(preprocess_json)
    augment_ops = describe_ops(augment_json)
    
    image_padding_color = get_image_padding_color_from_preprocess_ops(preprocess_ops)
    print(f"[preprocess] ImageResizer padding_color={image_padding_color}")

    print(f"[preprocess] deterministic ops requested: {json.dumps(preprocess_ops, ensure_ascii=False)}")
    print(f"[augment] stochastic ops requested: {json.dumps(augment_ops, ensure_ascii=False)}")
    print(f"[preprocess] provider preprocessors count: {len(shared_preprocessors)}")
    print(f"[augment] train augmentors count: {len(train_augmentors)}")
    
    use_preprocess_cache = bool((preprocess_json or "").strip())

    preprocess_cache_stats = []
    preprocess_sample_inspection = []

    if use_preprocess_cache:
        cache_base = cache_root / "preprocessed"

        train_items, train_cache_stats = cache_deterministic_preprocessed_items(
            train_items,
            shared_preprocessors,
            cache_base / "train",
            split_name="train"
        )
        preprocess_cache_stats.append(train_cache_stats)
        preprocess_sample_inspection.append(inspect_sample_image_shape(train_items, "train"))

        if val_items:
            val_items, val_cache_stats = cache_deterministic_preprocessed_items(
                val_items,
                shared_preprocessors,
                cache_base / "val",
                split_name="val"
            )
            preprocess_cache_stats.append(val_cache_stats)
            preprocess_sample_inspection.append(inspect_sample_image_shape(val_items, "val"))            

        if test_items:
            test_items, test_cache_stats = cache_deterministic_preprocessed_items(
                test_items,
                shared_preprocessors,
                cache_base / "test",
                split_name="test"
            )
            preprocess_cache_stats.append(test_cache_stats)
            preprocess_sample_inspection.append(inspect_sample_image_shape(test_items, "test"))

        provider_preprocessors = [ImageReader(CVImage)]
        print("[preprocess] using cached deterministic preprocessing")
    else:
        provider_preprocessors = shared_preprocessors
        print("[preprocess] no deterministic cache; preprocessors will run in provider")
        
    preprocess_summary = {
        "preprocess_json": preprocess_json,
        "augment_json": augment_json,
        "use_preprocess_cache": bool((preprocess_json or "").strip()),
        "preprocess_ops": preprocess_ops,
        "augment_ops": augment_ops,
    }
    
    preprocess_summary["cache_stats"] = preprocess_cache_stats
    preprocess_summary["sample_inspection"] = preprocess_sample_inspection
    
    
    
    train_dp = DataProvider(
        dataset=train_items,
        batch_size=effective_global_batch_size,
        data_preprocessors=provider_preprocessors,
        transformers=[
            ImageResizer(width, height, keep_aspect_ratio=True, padding_color=image_padding_color,),
            LabelIndexer(vocab),
            LabelPadding(max_word_length=max_len, padding_value=blank_index),
        ],
    )
    
    train_dp.augmentors = train_augmentors

    val_dp = None

    if val_items:
        val_dp = DataProvider(
            dataset=val_items,
            skip_validation=True,
            batch_size=effective_global_batch_size,
            data_preprocessors=provider_preprocessors,
            transformers=[
                ImageResizer(width, height, keep_aspect_ratio=True, padding_color=image_padding_color),
                LabelIndexer(vocab),
                LabelPadding(max_word_length=max_len, padding_value=blank_index),
            ],
        )

    test_dp = None
    if test_items:
        test_dp = DataProvider(
            dataset=test_items,
            skip_validation=True,
            batch_size=effective_global_batch_size,
            data_preprocessors=provider_preprocessors,
            transformers=[
                ImageResizer(width, height, keep_aspect_ratio=True, padding_color=image_padding_color,),
                LabelIndexer(vocab),
                LabelPadding(max_word_length=max_len, padding_value=len(vocab)),
            ],
        )

    # --------------------------------------------------
    # 🔹 Model
    # --------------------------------------------------

    print(f"[Model] Input w={width}, h={height}")

    with strategy.scope():
        model = train_model(
            input_dim=(height, width, 3),
            output_dim=len(vocab),
            dropout=dropout,
            activation=activation,
            reverse_time_axis=reverse_time_axis,
        )
        
        inspect_model_direction(
            model=model,
            text_dir=text_dir,
            reverse_time_axis=reverse_time_axis,
        )
        
        if text_dir == "rtl" and not reverse_time_axis:
            print(
                "[direction-check][WARNING] text_dir=rtl but reverse_time_axis=False. "
                "This is probably wrong for CRNN/CTC Arabic logical labels."
            )

        if text_dir == "rtl":
            has_rtl_layer = any(layer.name == "rtl_reverse_time_axis" for layer in model.layers)
            if reverse_time_axis and not has_rtl_layer:
                raise RuntimeError(
                    "RTL direction policy error: reverse_time_axis=True but "
                    "model does not contain rtl_reverse_time_axis layer."
                )

        print(f"[direction] text_dir={text_dir}")
        print(f"[direction] label_order_policy={label_order_policy}")
        print(f"[direction] reverse_time_axis={reverse_time_axis}")
        print(f"[direction] sequence_axis_policy={sequence_axis_policy}")
        
        loaded_from_spot = False
        effective_lr = lr
        checkpoint_dir = Path("/opt/ml/checkpoints")
        last_state_path = checkpoint_dir / "last_state.json"

        if spot_enabled and spot_ckpt.exists():
            print("🔥 Spot resume detected")     
            model.load_weights(str(spot_ckpt))           
            loaded_from_spot = True
            try:
                st = json.loads(last_state_path.read_text(encoding="utf-8"))
                effective_lr = float(st.get("last_metrics", {}).get("lr", 0.0))
                print(f"[resume] Resuming from learning rate {effective_lr}")
            except Exception as e:
                print(f"[resume] Failed to read last_state.json: {e}")

        else:           
            
            # Resume / finetune
            if resume_mode in ("resume", "finetune") and resume_path:
                print(f"[resume] mode={resume_mode}, which={resume_which}")

                ckpt_file = "best.weights.h5" if resume_which == "best" else "last.weights.h5"
                resume_path = resume_path.rstrip("/")
                resume_uri = f"{resume_path}/{ckpt_file}"

                print(f"[weights] loading weights from {resume_uri}")
                local_weights = resolve_resume_weights(resume_uri, refs)
                model.load_weights(local_weights)  
                print("[weights] initialized from checkpoint")            
            
            else:
                print("🚀 Fresh training")

        print("[model] output_shape:", model.output_shape)
        print("[labels] max_label_len:", max_len)

        T = model.output_shape[1]
        if T is None:
            raise RuntimeError("CTC time dimension (T) is None; model output shape invalid")

        if max_len >= T:
            raise RuntimeError(
                f"CTC invalid: max_label_len={max_len} >= time_steps={T}. "
                "This should have been validated client-side."
            )

        print(f"[ctc] using max_label_len={max_len}, time_steps={T}")

        PAD = "\u25A1"
        vocab_for_metrics = "".join(vocab) + PAD

        # Conservative LR scaling:
        # scale from the original single-GPU baseline in proportion to effective global batch
        # For resume / finetune, do NOT scale LR yet.
        # CTC training is numerically sensitive, especially after restoring weights.  
        
                        
        
        optimizer=tf.keras.optimizers.Adam(learning_rate=effective_lr)

        model.compile(
            optimizer=optimizer,
            loss=CTCloss(),
            metrics=[
                SafeCERMetric(vocabulary=vocab),
                SafeWERMetric(vocabulary=vocab),
            ],
        )
        # else:
        #     print("[opt] using optimizer state restored from spot checkpoint")
        
    

        
    # ---------------- callbacks ----------------
    
    
    train_steps = len(train_items) // effective_global_batch_size
    val_steps = len(val_items) // effective_global_batch_size if val_items else 0

    if train_steps < 1:
        raise RuntimeError(
            f"Not enough training samples for one full global batch: "
            f"train={len(train_items)}, global_batch={effective_global_batch_size}"
        )

    validation_enabled = val_dp is not None and val_steps >= 1

    if not validation_enabled:
        print(
            f"[fit] validation disabled: val={len(val_items)}, "
            f"global_batch={effective_global_batch_size}"
        )
        validation_steps = None
    else:
        validation_steps = val_steps



    print(
        f"[fit] train_steps={train_steps}, val_steps={val_steps}, "
        f"train_remainder={len(train_items) % effective_global_batch_size}, "
        f"val_remainder={len(val_items) % effective_global_batch_size}"
    )
    
    monitor_metric = "val_CER" if validation_enabled else "CER"
    monitor_mode = "min"
    
    print(f"[monitor] monitor_metric={monitor_metric}, validation_enabled={validation_enabled}")
    ckpt_dir = refs.models_dir
    
    
    callbacks = [
        CGLKerasLogger(monitor=monitor_metric, mode=monitor_mode),
        # 🔹 Full model – best
        ModelCheckpoint(
            ckpt_dir / "best.keras",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_best_only=True,
            verbose=0,
        ),

        # 🔹 Full model – last
        ModelCheckpoint(
            ckpt_dir / "last.keras",
            save_best_only=False,
            verbose=0,
        ),

        # 🔹 Weights only – best (used for resume / finetune)
        ModelCheckpoint(
            ckpt_dir / "best.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor=monitor_metric,
            mode=monitor_mode,
            verbose=0,
        ),

        # 🔹 Weights only – last
        ModelCheckpoint(
            ckpt_dir / "last.weights.h5",
            save_weights_only=True,
            save_best_only=False,
            verbose=0,
        ),
        
        # Spot checkpoint
        ModelCheckpoint(
            "/opt/ml/checkpoints/ckpt.weights.h5",
            save_weights_only=True,
            save_best_only=False,
            verbose=0,
        ),

        # 🔹 Early stopping (env-driven)
        CGLEarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=early_patience,
            restore_best_weights=True,
            verbose=0,
        ),

        # 🔹 LR schedule
        ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.8,
            patience=reduce_lr_patience,
            min_delta=1e-4,
            cooldown=10,
            mode=monitor_mode,
            min_lr=1e-5,
            verbose=1,
        ),
        # ---- Upload policies via outputs.py ----
        UploadBestOnImprove(outputs, monitor=monitor_metric, mode=monitor_mode),
        UploadLastEveryEpoch(outputs),

    ]

    callbacks.append(
        LastStateWriter(
            out_dir=str("/opt/ml/checkpoints"),
            hyp={
                "epochs": epochs,
                "client_batch_size": client_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "effective_global_batch_size": effective_global_batch_size,
                "num_gpus": num_gpus,
                "lr": effective_lr,
                "base_lr": lr,
                "width": width,
                "height": height,
            },
            data_counts={
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
            },
            rtl_policy={
                "text_dir": text_dir,
                "label_order_policy": label_order_policy,
                "reverse_time_axis": reverse_time_axis,
                "sequence_axis_policy": sequence_axis_policy,
            },
            blank_index=blank_index,
            vocab=vocab,
            resume={},
        )
    )
    
    callbacks.append(
        DirectionDecodeProbe(
            data_provider=train_dp,
            vocab=vocab,
            blank_index=blank_index,
            text_dir=text_dir,
            reverse_time_axis=reverse_time_axis,
            name="train_fixed_batch",
            every_n_epochs=int(os.getenv("CGL_DIRECTION_PROBE_EVERY", "5")),
            max_samples=int(os.getenv("CGL_DIRECTION_PROBE_SAMPLES", "3")),
        )
    )

    if val_dp is not None:
        callbacks.append(
            DirectionDecodeProbe(
                data_provider=val_dp,
                vocab=vocab,
                blank_index=blank_index,
                text_dir=text_dir,
                reverse_time_axis=reverse_time_axis,
                name="val_fixed_batch",
                every_n_epochs=int(os.getenv("CGL_DIRECTION_PROBE_EVERY", "5")),
                max_samples=int(os.getenv("CGL_DIRECTION_PROBE_SAMPLES", "3")),
            )
        )
            

    initial_epoch = 0

    if loaded_from_spot and last_state_path.exists():
        try:
            st = json.loads(last_state_path.read_text(encoding="utf-8"))
            initial_epoch = int(st.get("epoch", 0))
            print(f"[resume] Resuming from epoch {initial_epoch}")
        except Exception as e:
            print(f"[resume] Failed to read last_state.json: {e}")

    # ---------------- training ----------------

    # Keep one DataProvider path, but make it repeatable for Keras 3 + MirroredStrategy.
    train_fit_data = make_repeating_tf_dataset(train_dp, train_steps, "train")

    if validation_steps is not None:
        val_fit_data = make_repeating_tf_dataset(val_dp, validation_steps, "val")
    else:
        val_fit_data = None

    model.fit(
        train_fit_data,
        validation_data=val_fit_data,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=0,
    )
    
    print("[rtl-diag] running pre-training direction diagnostic...")

    run_rtl_direction_batch_diagnostic(
        model=model,
        data_provider=train_dp,
        vocab=vocab,
        blank_index=blank_index,
        text_dir=text_dir,
        reverse_time_axis=reverse_time_axis,
        name="train_prefit",
        max_samples=3,
    )

    if val_dp is not None:
        run_rtl_direction_batch_diagnostic(
            model=model,
            data_provider=val_dp,
            vocab=vocab,
            blank_index=blank_index,
            text_dir=text_dir,
            reverse_time_axis=reverse_time_axis,
            name="val_prefit",
            max_samples=3,
        )
        
    try:
        print("\n[eval-check] Running explicit Keras evaluate on train and val datasets...")

        train_eval = model.evaluate(train_fit_data, steps=train_steps, verbose=0)
        val_eval = (
            model.evaluate(val_fit_data, steps=validation_steps, verbose=0)
            if val_fit_data is not None
            else None
        )

        print("[eval-check] metrics_names:", model.metrics_names)
        print("[eval-check] raw train_eval:", train_eval)
        print("[eval-check] raw val_eval:", val_eval)

        try:
            train_eval_dict = model.evaluate(
                train_fit_data,
                steps=train_steps,
                verbose=0,
                return_dict=True,
            )

            val_eval_dict = (
                model.evaluate(
                    val_fit_data,
                    steps=validation_steps,
                    verbose=0,
                    return_dict=True,
                )
                if val_fit_data is not None
                else {}
            )

            train_eval_dict = {
                k: float(v) if hasattr(v, "__float__") else v
                for k, v in train_eval_dict.items()
            }

            val_eval_dict = {
                k: float(v) if hasattr(v, "__float__") else v
                for k, v in val_eval_dict.items()
            }

            print("[eval-check] return_dict train:", json.dumps(train_eval_dict, indent=2))
            print("[eval-check] return_dict val:", json.dumps(val_eval_dict, indent=2))

        except Exception as e:
            print(f"[eval-check] return_dict evaluate failed: {e}")

    except Exception as e:
        print(f"[eval-check] model.evaluate failed: {e}")
            
    train_diag_report = None

    try:
        print("[diag] running training diagnostics on fixed subset...")

        # Keep this small to avoid long runtime
        train_diag_sample_count = min(100, len(train_items))
        train_diag_items = train_items[:train_diag_sample_count]

        train_diag_dp = DataProvider(
            dataset=train_diag_items,
            skip_validation=True,
            batch_size=min(16, client_batch_size),
            data_preprocessors=provider_preprocessors,
            transformers=[
                ImageResizer(
                    width,
                    height,
                    keep_aspect_ratio=True,
                    padding_color=image_padding_color,
                ),
                LabelIndexer(vocab),
                LabelPadding(max_word_length=max_len, padding_value=blank_index),
            ],
        )

        train_diag_report = run_val_diagnostics(
            model=model,
            data_provider=train_diag_dp,
            dataset_items=train_diag_items,
            vocab=vocab,
            blank_index=blank_index,
        )

        print(f"[diag] train subset mean CER={train_diag_report['mean_cer']:.4f}")
        print(f"[diag] train subset mean WER={train_diag_report.get('mean_wer'):.4f}")
        print(f"[diag] train subset sample_count={train_diag_report.get('sample_count')}")        
        

    except Exception as e:
        print(f"[diag] training diagnostics failed: {e}")
        train_diag_report = {"error": str(e)}
        
    val_diag_report = None

    if val_items:
        try:
            print("[diag] running validation diagnostics...")

            diag_global_batch_size = min(16, client_batch_size)

            val_diag_dp = DataProvider(
                dataset=val_items,
                skip_validation=True,
                batch_size=diag_global_batch_size,
                data_preprocessors=provider_preprocessors,
                transformers=[
                    ImageResizer(width, height, keep_aspect_ratio=True, padding_color=image_padding_color),
                    LabelIndexer(vocab),
                    LabelPadding(max_word_length=max_len, padding_value=blank_index),
                ],
            )

            val_diag_report = run_val_diagnostics(
                model=model,
                data_provider=val_diag_dp,
                dataset_items=val_items,
                vocab=vocab,
                blank_index=blank_index,
            )

            print(f"[diag] val mean CER={val_diag_report['mean_cer']:.4f}")
            print(f"[diag] val mean WER={val_diag_report.get('mean_wer'):.4f}")
            print(f"[diag] top error chars={val_diag_report['top_error_chars'][:20]}")

        except Exception as e:
            print(f"[diag] validation diagnostics failed: {e}")
            val_diag_report = {"error": str(e)}
    else:
        print("[diag] validation diagnostics skipped: no validation split")
        val_diag_report = {
            "skipped": True,
            "reason": "no_validation_split"
        }
            
    # ---------------- test evaluation ----------------
    test_metrics = None
    test_steps = len(test_items) // effective_global_batch_size

    if test_dp and test_steps >= 1:
        emit(CGL_EVAL, {
            "phase": "begin",
            "split": "test",
            "samples": len(test_items),
            "steps": test_steps,
        })

        test_fit_data = make_repeating_tf_dataset(test_dp, test_steps, "test")
        results = model.evaluate(test_fit_data, steps=test_steps, verbose=0)

        test_metrics = dict(zip(model.metrics_names, map(float, results)))

        emit(CGL_EVAL, {
            "phase": "end",
            "split": "test",
            "samples": len(test_items),
            "steps": test_steps,
            "metrics": test_metrics,
        })
    elif test_dp:
        print(
            f"[test] test evaluation skipped: test={len(test_items)} is smaller than "
            f"global_batch={effective_global_batch_size}"
        )


    # ---------------- save artifacts ----------------
    # model.save(refs.models_dir / "model.keras")
    final_model = refs.models_dir / "final.keras"
    final_weights = refs.models_dir / "final.weights.h5"

    model.save(final_model)
    model.save_weights(final_weights)
    # Optional but useful for resume compatibility   
    

    outputs.upload_model(final_model, "final.keras")
    outputs.upload_model(final_weights, "final.weights.h5")
    


    with open(refs.artifacts_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    outputs.upload_artifact(refs.artifacts_dir / "vocab.json", "vocab.json")
    
    with open(refs.artifacts_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_summary, f, indent=2)
    outputs.upload_artifact(refs.artifacts_dir / "preprocess_summary.json", "preprocess_summary.json")

    with open(refs.artifacts_dir / "val_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(val_diag_report, f, ensure_ascii=False, indent=2)

    outputs.upload_artifact(refs.artifacts_dir / "val_diagnostics.json", "val_diagnostics.json")

    metrics_payload = {
        "status": "completed",
        "framework": "ocr_ctc",
        "task_type": "ocr",
        "training": {
            "epochs_requested": epochs,
            "base_learning_rate": lr,
            "effective_learning_rate": effective_lr,
            "client_batch_size": client_batch_size,
            "per_gpu_batch_size": per_gpu_batch_size,
            "effective_global_batch_size": effective_global_batch_size,
            "num_gpus": num_gpus,
        },
        "dataset": {
            "train_samples": len(train_items),
            "val_samples": len(val_items),
            "test_samples": len(test_items),
            "vocab_size": len(vocab),
            "max_label_len": max_len,
        },
        "preprocessing": {
            "enabled": bool((preprocess_json or "").strip()),
            "cached": use_preprocess_cache,
            "ops": preprocess_ops,
            "cache_stats": preprocess_cache_stats
        },
        "direction": {
            "text_dir": text_dir,
            "label_order_policy": label_order_policy,
            "reverse_time_axis": reverse_time_axis,
            "sequence_axis_policy": sequence_axis_policy,
        },
        }
    
    metrics_payload["val_diagnostics"] = {
        "mean_cer": val_diag_report.get("mean_cer") if val_diag_report else None,
        "top_error_chars": (val_diag_report.get("top_error_chars", [])[:20] if val_diag_report else []),
        "cer_by_label_length_bucket": (val_diag_report.get("cer_by_label_length_bucket", {}) if val_diag_report else {}),
        "cer_by_orig_width_bucket": (val_diag_report.get("cer_by_orig_width_bucket", {}) if val_diag_report else {}),
        "cer_by_orig_height_bucket": (val_diag_report.get("cer_by_orig_height_bucket", {}) if val_diag_report else {}),
        "cer_by_orig_aspect_bucket": (val_diag_report.get("cer_by_orig_aspect_bucket", {}) if val_diag_report else {}),
    }
    
    if test_metrics:
        metrics_payload["test_samples"] = len(test_items)
        metrics_payload["test_metrics"] = test_metrics
        
    # last_state_path = Path("/opt/ml/checkpoints/last_state.json")

    if last_state_path.exists():
        st = json.loads(last_state_path.read_text(encoding="utf-8"))

        metrics_payload["best"] = st.get("best_so_far")
        metrics_payload["last"] = {
            "epoch": st.get("epoch"),
            **(st.get("last_metrics") or {})
        }


    outputs.write_metrics(metrics_payload) 
    outputs.upload_metrics()   
    
    outputs.upload_artifact(refs.artifacts_dir / "dataset_manifest.json", "dataset_manifest.json")
    outputs.upload_artifact(refs.artifacts_dir / "train_args.json", "train_args.json")
    outputs.upload_artifact(refs.artifacts_dir / "config.json", "config.json")

    

    outputs.finalize()

if __name__ == "__main__":
    main()
