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
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.annotations.images import CVImage

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

# BOUNDARY_NEUTRALS_RE = re.compile(
#     r"""([\s\.\,\،\؛\:\!\؟\?\…\"\'\)\]\}\»\”\“\%]+)$"""
# )
BOUNDARY_NEUTRALS_RE = re.compile(
    r"""([\s\.\,\،\؛\:\!\؟\?\…]+)$"""
)

def rtl_boundary_neutrals(txt: str) -> str:
    """
    For RTL OCR line labels, move only a trailing neutral run to the front.

    Example:
      'الآخر واحترام الذات .' -> '. الآخر واحترام الذات'

    This does NOT reverse Arabic text.
    This does NOT move punctuation in the middle of the sentence.
    """
    txt = " ".join(txt.strip().split())

    m = BOUNDARY_NEUTRALS_RE.search(txt)
    if not m:
        return txt

    tail = m.group(1).strip()
    body = txt[:m.start()].rstrip()

    if not tail or not body:
        return txt

    return f"{tail} {body}"


def load_jsonl(path, text_dir="ltr", max_reasonable_len=1000):
    items = []
    max_len = 0
    debug_label_examples = 0
    
    def normalize_label(txt: str) -> str:
        txt = txt.strip()
        txt = " ".join(txt.split())
        if text_dir == "rtl":
            txt = rtl_boundary_neutrals(txt)
        return txt

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
            
            # Optional: show only first few RTL label transformations
            if text_dir == "rtl" and debug_label_examples < 5:
                raw_txt = rec.get("text") if rec.get("text") is not None else rec.get("transcription")
                raw_norm = " ".join(str(raw_txt).strip().split())
                if raw_norm != txt:
                    print(f"[label][rtl_boundary_neutrals] {repr(raw_norm)} -> {repr(txt)}")
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

def env(name, default=None, cast=str):
    val = os.getenv(name, default)
    if val is None:
        return None
    try:
        return cast(val)
    except Exception:
        raise ValueError(f"Invalid value for {name}: {val}")


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
    
    label_order_policy = "rtl_boundary_neutrals" if text_dir == "rtl" else "logical"
    
    

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
        "early_patience": early_patience,
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
    }   
    

    with open(refs.artifacts_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)
        
    

    # --------------------------------------------------
    # 🔹 Dataset
    # --------------------------------------------------

    train_ref = resolve_split(dataset_uri, "train", cache_root)
    val_ref   = resolve_split(dataset_uri, "val", cache_root)

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
            "val": {
                "annotations": str(val_ref.local_annotations_path),
                "images_dir": str(val_ref.local_images_dir),
            },
        },
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
    val_items,   val_max   = load_jsonl(val_ref.local_annotations_path, text_dir or "ltr" )
   
    
    train_items = absolutize_items(train_items, train_ref.local_root)
    val_items   = absolutize_items(val_items,   val_ref.local_root)
    
    test_items = []
    test_max = 0
    if test_ref:
        test_items,  test_max  = load_jsonl(test_ref.local_annotations_path, text_dir or "ltr") if test_ref else ([], 0)
        test_items = absolutize_items(test_items, test_ref.local_root)
    else:
        test_items = []

    vocab = build_vocab(train_items + val_items)
    blank_index = len(vocab)
    if(test_max):
        max_len = max(train_max, val_max, test_max)
    else:
        max_len = max(train_max, val_max)
    
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
            ImageResizer(width, height, keep_aspect_ratio=True),
            LabelIndexer(vocab),
            LabelPadding(max_word_length=max_len, padding_value=blank_index),
        ],
    )
    
    train_dp.augmentors = train_augmentors

    val_dp = DataProvider(
        dataset=val_items,
        skip_validation=True,
        batch_size=effective_global_batch_size,
        data_preprocessors=provider_preprocessors,
        transformers=[
            ImageResizer(width, height, keep_aspect_ratio=True),
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
                ImageResizer(width, height, keep_aspect_ratio=True),
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
            )
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
                CERMetric(vocabulary=vocab),
                WERMetric(vocabulary=vocab),
            ],
        )
        # else:
        #     print("[opt] using optimizer state restored from spot checkpoint")
        
       

        
    # ---------------- callbacks ----------------
    
    
    ckpt_dir = refs.models_dir
    
    
    callbacks = [
        CGLKerasLogger(monitor="val_CER", mode="min"),
        # 🔹 Full model – best
        ModelCheckpoint(
            ckpt_dir / "best.keras",
            monitor="val_CER",
            mode="min",
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
            monitor="val_CER",
            mode="min",
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
            monitor="val_CER",
            mode="min",
            patience=early_patience,
            restore_best_weights=True,
            verbose=0,
        ),

        # 🔹 LR schedule
        ReduceLROnPlateau(
            monitor="val_CER",
            factor=0.9,
            patience=10,
            min_delta=1e-10,           
            cooldown=2,
            mode="min",
            verbose=1,
        ),
        # ---- Upload policies via outputs.py ----
        UploadBestOnImprove(outputs, monitor="val_CER", mode="min"),
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
            rtl_policy="auto",
            blank_index=blank_index,
            vocab=vocab,
            resume={},
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
    model.fit(
        train_dp,
        validation_data=val_dp,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=0,
        # workers=4,
        # use_multiprocessing=True,
        # max_queue_size=16,
    )
    
    val_diag_report = None

    try:
        print("[diag] running validation diagnostics...")
        # ---------------- diagnostics provider ----------------
        # Use the same validation path, but with a smaller GLOBAL batch size to avoid OOM
        diag_global_batch_size = min(16, client_batch_size)

        val_diag_dp = DataProvider(
            dataset=val_items,
            batch_size=diag_global_batch_size,
            data_preprocessors=provider_preprocessors,
            transformers=[
                ImageResizer(width, height, keep_aspect_ratio=True),
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
        print(f"[diag] top error chars={val_diag_report['top_error_chars'][:20]}")
        print(f"[diag] CER by label length bucket={val_diag_report['cer_by_label_length_bucket']}")
        print(f"[diag] CER by orig width bucket={val_diag_report['cer_by_orig_width_bucket']}")
        print(f"[diag] CER by orig height bucket={val_diag_report['cer_by_orig_height_bucket']}")
        print(f"[diag] CER by orig aspect bucket={val_diag_report['cer_by_orig_aspect_bucket']}")

    except Exception as e:
        print(f"[diag] validation diagnostics failed: {e}")
        val_diag_report = {
            "error": str(e)
        }
        
    # ---------------- test evaluation ----------------
    test_metrics = None
    if test_dp:
        emit(CGL_EVAL, {
            "phase": "begin",
            "split": "test",
            "samples": len(test_items),
        })

        results = model.evaluate(test_dp, verbose=0)

        # Keras returns list aligned with model.metrics_names
        test_metrics = dict(zip(model.metrics_names, map(float, results)))

        emit(CGL_EVAL, {
            "phase": "end",
            "split": "test",
            "samples": len(test_items),
            "metrics": test_metrics,
        })


    # ---------------- save artifacts ----------------
    # model.save(refs.models_dir / "model.keras")
    final_model = refs.models_dir / "final.keras"
    final_weights = refs.models_dir / "final.weights.h5"

    model.save(final_model)
    model.save_weights(final_weights)

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
