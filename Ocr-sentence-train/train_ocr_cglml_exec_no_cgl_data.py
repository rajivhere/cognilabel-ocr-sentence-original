#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executable training script derived from your notebook:
- Uses ONLY cgl_ml.* for data/pipeline
- Builds model via: from model import train_model
- Reads env vars (CL_*, CGL_*) for I/O + hyperparams + behavior
- No imports from your ocr_utils/train_utils
"""

import os, json, random, math, time, platform
from pathlib import Path

import numpy as np
import tensorflow as tf
import itertools
# try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
# except: pass

import pandas as pd
import io
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric

# from model import train_model
from configs import ModelConfigs
import unicodedata
import numpy as np
from pathlib import Path
# prefer library helper if available
try:
    from mltu.tensorflow.callbacks import set_memory_growth as _lib_set_mem
except Exception:
    _lib_set_mem = None
# Model comes from your local Ocr-sentence-train/model.py
# from model import train_model
from model1 import train_model
import re
import hashlib
from urllib.parse import urlparse
from last_state_writer import LastStateWriter  # NEW

def _available_cpus():
    # SageMaker sets this
    v = os.getenv("SM_NUM_CPUS")
    if v:
        try: return int(float(v))
        except: pass
    # cgroups-aware on Linux
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

def _available_gpus():
    v = os.getenv("SM_NUM_GPUS")
    if v:
        try: return int(float(v))
        except: pass
    try:
        return len(tf.config.list_physical_devices("GPU"))
    except Exception:
        return 0

def _tf_runtime_setup():
    # 1) GPU memory growth
    try:
        if _lib_set_mem:
            _lib_set_mem()
        else:
            for gpu in tf.config.experimental.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[tf] memory growth not set: {e}")

    # 2) Thread heuristics (auto unless overridden)
    intra_env = os.getenv("CGL_TF_INTRA_OP")
    inter_env = os.getenv("CGL_TF_INTER_OP")

    if intra_env or inter_env:
        # respect overrides
        try:
            if intra_env: tf.config.threading.set_intra_op_parallelism_threads(int(intra_env))
            if inter_env: tf.config.threading.set_inter_op_parallelism_threads(int(inter_env))
            print(f"[tf] threads override: intra={intra_env or 'auto'}, inter={inter_env or 'auto'}")
        except Exception as e:
            print(f"[tf] threading override failed: {e}")
        return

    cpus = _available_cpus()
    gpus = _available_gpus()

    # Heuristics:
    # - CPU-only: use most cores for intra-op; keep a few for inter/op & input threads
    # - GPU: CPU threads primarily feed the GPU; keep inter-op modest
    if gpus > 0:
        intra = max(1, cpus - 1)                # leave 1 for OS/IO
        inter = min(4, max(1, cpus // 4))       # small cross-op parallelism
    else:
        intra = max(1, cpus - 1)
        inter = min(4, max(1, cpus // 2))

    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
        print(f"[tf] threads auto: cpus={cpus}, gpus={gpus}, intra={intra}, inter={inter}")
    except Exception as e:
        print(f"[tf] threading auto failed: {e}")

# def _tf_threads():
#     try:
#         if ENV["CGL_TF_INTRA_OP"]:
#             tf.config.threading.set_intra_op_parallelism_threads(int(ENV["CGL_TF_INTRA_OP"]))
#         if ENV["CGL_TF_INTER_OP"]:
#             tf.config.threading.set_inter_op_parallelism_threads(int(ENV["CGL_TF_INTER_OP"]))
#     except Exception as e:
#         print(f"[tf] threading not applied: {e}")
        
# ------------------------------- ENV -------------------------------
ENV = {
    # JSONL inputs (local path or s3://); base dirs for images
    "JSONL_TRAIN": os.getenv("CL_JSONL_LOCAL") or os.getenv("CL_JSONL_TRAIN") or "D:/User/dataset/train/annotations/labels.jsonl",
    "JSONL_VAL":   os.getenv("CL_JSONL_LOCAL_VAL") or os.getenv("CL_JSONL_VAL") or "D:/User/dataset/val/annotations/labels.jsonl",
    "JSONL_TEST":  os.getenv("CL_JSONL_LOCAL_TEST") or os.getenv("CL_JSONL_TEST") or "D:/User/dataset/test/annotations/labels.jsonl",
    "IMAGE_BASE":       os.getenv("CL_IMAGE_BASE") or "D:/User",
    "IMAGE_BASE_VAL":   os.getenv("CL_IMAGE_BASE_VAL") or "D:/User",
    "IMAGE_BASE_TEST":  os.getenv("CL_IMAGE_BASE_TEST") or "D:/User",

    # I/O
    "CACHE_DIR":   os.getenv("CL_CACHE_DIR", "cache/lines"),
    "OUTPUT_DIR":  os.getenv("CL_OUTPUT", "./outputs"),

    # Hyperparams (JSON string)
    "HYP_JSON":    os.getenv("HYPERPARAMS_JSON") or os.getenv("HYP_JSON") or "{}",

    # Seeding
    "CGL_TRAIN_SEED": os.getenv("CGL_TRAIN_SEED", "42"),
    "CL_SEED":        os.getenv("CL_SEED", ""),

    # RTL policy
    "CL_RTL":        os.getenv("CL_RTL", ""),                     # legacy boolean
    "CL_RTL_POLICY": os.getenv("CL_RTL_POLICY", "auto"),          # "auto"|"rtl"|"ltr"

    # Resume / fine-tune
    "CL_RESUME_MODE":  os.getenv("CL_RESUME_MODE", "none"),       # "none"|"resume"|"finetune"
    "CL_RESUME_WHICH": os.getenv("CL_RESUME_WHICH", "best"),      # "best"|"last"
    "CL_RESUME_S3":    os.getenv("CL_RESUME_S3", ""),             # s3://.../best.keras
    "CL_FT_LR":        os.getenv("CL_FT_LR", ""),                 # optional ft LR
    "CL_FREEZE_CNN_EPOCHS": os.getenv("CL_FREEZE_CNN_EPOCHS", "0"),

    # Checkpoints & exports
    "CGL_SAVE_EVERY_N_EPOCHS": os.getenv("CGL_SAVE_EVERY_N_EPOCHS", ""),
    "CGL_EARLYSTOP_PATIENCE":  os.getenv("CGL_EARLYSTOP_PATIENCE", "10"),
    "CGL_EXPORT_ONNX":         os.getenv("CGL_EXPORT_ONNX", "0"),
    "CL_OUTPUT_S3_PREFIX":     os.getenv("CL_OUTPUT_S3_PREFIX", ""),

    # Pipeline JSON specs (server-provided)
    "CGL_TRF_JSON": os.getenv("CGL_TRF_JSON", ""),
    "CGL_AUG_JSON": os.getenv("CGL_AUG_JSON", ""),
    
    "CL_FORCE_IMAGE_BASE": os.getenv("CL_FORCE_IMAGE_BASE", "1"),  # 1 = ignore any s3:// in JSONL image fields
    "CGL_TF_INTRA_OP": os.getenv("CGL_TF_INTRA_OP", ""),           # optional threading caps
    "CGL_TF_INTER_OP": os.getenv("CGL_TF_INTER_OP", ""),
}

# Defaults (overridable by HYP_JSON)
DEFAULT_HYP = dict(
    arch="resnet_lstm",
    width=1048,
    height=96,
    batch_size=8,
    epochs=50,
    learning_rate=1e-3,
    dropout=0.2,
    activation="leaky_relu",
    val_split=0.1,
    max_label_len=64,  # used when building LabelPadding if not inferred
)
HYP = {**DEFAULT_HYP, **json.loads(ENV["HYP_JSON"] or "{}")}

# ------------------------------ helpers ------------------------------
def _ensure_dirs():
    Path(ENV["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(ENV["CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

def _train_seed():
    v = (ENV["CGL_TRAIN_SEED"] or ENV["CL_SEED"]).strip()
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return 42

def _rtl_policy():
    pol = (ENV["CL_RTL_POLICY"] or "auto").strip().lower()
    if pol not in ("auto","rtl","ltr"):
        pol = "auto"
    legacy = ENV["CL_RTL"].strip().lower()
    if legacy in ("1","true","yes"):
        pol = "rtl"
    elif legacy in ("0","false","no"):
        pol = "ltr"
    return pol

def _abs_path(base_dir: str, file_field: str) -> str:
    if not file_field:
        return ""
    # If env base is S3, always construct from base (ignore JSONL s3://)
    if base_dir and str(base_dir).startswith("s3://"):
        return base_dir.rstrip("/") + "/" + file_field.lstrip("/\\")
    # Windows absolute? (C:\ or \\server\share)
    if re.match(r"^[a-zA-Z]:[\\/]", file_field) or file_field.startswith("\\\\"):
        return file_field
    # POSIX absolute (leading /) — treat as relative if base is provided
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, file_field.lstrip("/\\")))
    return file_field

def _load_jsonl(jsonl_path: str, image_base: str, rtl_policy: str):
    """Read OCR JSONL lines into list[(img_path, text)], applying rtl policy per record when policy='auto'."""
    items = []
    max_len = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = str(rec.get("text") or rec.get("transcription") or "")
            if not text:
                continue

            # direction handling
            direction = (rec.get("direction") or rec.get("dir") or "").strip().lower()
            is_rtl = (rtl_policy == "rtl") or (rtl_policy == "auto" and direction == "rtl")
            if is_rtl:
                text = text[::-1]

            img_path = _abs_path(image_base, rec.get("file") or rec.get("image") or rec.get("path") or "")
            if not img_path:
                continue

            max_len = max(max_len, len(text))
            items.append((img_path, text))
    return items, max_len

def _build_vocab(samples):
    # simple character-level vocab
    seen = set()
    for _, t in samples:
        for ch in t:
            seen.add(ch)
    vocab = sorted(seen)
    return vocab

def _download_if_s3(uri: str, dst: Path) -> str:
    """If uri is s3://bucket/key, download to dst and return str(dst).
    If uri is not s3://, return it unchanged."""
    if not uri or not uri.startswith("s3://"):
        return uri

    # Validate & prepare
    u = urlparse(uri)
    if not u.netloc or not u.path or u.path.endswith("/"):
        raise ValueError(f"[s3] Expected a file URI, got: {uri}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Idempotent: skip if already present and non-empty
    try:
        if dst.exists() and dst.stat().st_size > 0:
            return str(dst)
    except Exception:
        pass  # if stat fails, just re-download

    # Download
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
        boto3.client("s3").download_file(u.netloc, u.path.lstrip("/"), str(dst))
        return str(dst)
    except (ImportError, NoCredentialsError) as e:
        raise SystemExit(f"[s3] boto3/credentials required for {uri}: {e}") from e
    except (BotoCoreError, ClientError, OSError) as e:
        # Fail fast with a clear message instead of returning "".
        raise RuntimeError(f"[s3] download failed {uri} -> {dst}: {e}") from e
    
def _ensure_h5_weights_from_checkpoint(ckpt_path: Path) -> Path:
    """
    Given a checkpoint path (.keras or .h5),
    ensure a .weights.h5 file exists and return its path.
    """
    ckpt_path = Path(ckpt_path)

    # Case 1: already .h5 → done
    if ckpt_path.suffix in (".h5", ".hdf5"):
        return ckpt_path

    # Case 2: .keras → convert ONCE
    if ckpt_path.suffix == ".keras":
        h5_path = ckpt_path.with_suffix(".weights.h5")

        if h5_path.exists() and h5_path.stat().st_size > 0:
            print(f"[weights] found existing {h5_path.name}")
            return h5_path

        print(f"[weights] migrating {ckpt_path.name} → {h5_path.name}")
        old = tf.keras.models.load_model(str(ckpt_path), compile=False, safe_mode=False)
        old.save_weights(str(h5_path))
        print("[weights] migration complete")

        return h5_path

    raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

def _resolve_resume_weights(ckpt_path: Path) -> Path:
    """
    Resolve a resume checkpoint.
    Only supports .weights.h5 for training resume.
    """
    ckpt_path = Path(ckpt_path)

    if ckpt_path.name.endswith(".weights.h5"):
        return ckpt_path

    raise RuntimeError(
        f"Unsupported resume checkpoint: {ckpt_path.name}. "
        "Resume requires a .weights.h5 file."
    )



class _S3FetchBeforeRead:
    def __init__(self, cache_root):
        self.cache = Path(cache_root) / "s3_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        try:
            import boto3  # noqa
            self._ok = True
        except Exception:
            self._ok = False
            print("[s3] boto3 not available; cannot fetch s3:// images")

    def __call__(self, data, annotation):
        # data is image path
        if isinstance(data, str) and data.startswith("s3://") and self._ok:
            u = urlparse(data)
            key = u.path.lstrip("/")
            ext = Path(key).suffix or ".img"
            h = hashlib.md5(data.encode("utf-8")).hexdigest()[:16]
            local = self.cache / f"{h}{ext}"
            if not local.exists():
                import boto3
                boto3.client("s3").download_file(u.netloc, key, str(local))
            return str(local), annotation
        return data, annotation

# --------------------------- pipeline from ENV ---------------------------
def build_pipeline_from_env(vocab, max_label_len, H, W):
    pre = [ImageReader(CVImage)]
    
    if any(str(ENV.get(k, "")).startswith("s3://")
        for k in ("IMAGE_BASE","IMAGE_BASE_VAL","IMAGE_BASE_TEST")):
            pre = [_S3FetchBeforeRead(ENV["CACHE_DIR"])] + pre

    # ---- Augmenters: only from JSON; otherwise default to NONE ----
    aug = []
    ajson = ENV["CGL_AUG_JSON"].strip()
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
                elif name == "RandomBrightness":
                    aug.append(RandomBrightness(p, int(item.get("delta", 25))))
                elif name == "RandomElasticTransform":
                    alpha = item.get("alpha", (0.0, 0.06)); sigma = item.get("sigma", (0.004, 0.012))
                    aug.append(RandomElasticTransform(p, tuple(alpha), tuple(sigma)))
                elif name == "RandomErodeDilate":
                    aug.append(RandomErodeDilate(p, int(item.get("ksize", 3)), int(item.get("iters", 1))))
                elif name == "RandomSharpen":
                    aug.append(RandomSharpen(p, float(item.get("radius", 1.0)), int(item.get("percent", 150)), int(item.get("threshold", 3))))
        except Exception as e:
            print(f"[CGL] bad CGL_AUG_JSON ({e}); using NO augmenters")

    # ---- Transformers: from JSON; otherwise use EXACT default set ----
    trf = []
    tjson = ENV["CGL_TRF_JSON"].strip()
    if tjson:
        try:
            spec = json.loads(tjson)
            has_index = has_pad = False
            for item in spec:
                name = item.get("name")
                if name == "ImageResizer":
                    trf.append(ImageResizer(int(item.get("width", W)), int(item.get("height", H)),
                                            keep_aspect_ratio=bool(item.get("keep_aspect", True))))
                elif name == "LabelIndexer":
                    trf.append(LabelIndexer(list(vocab))); has_index = True
                elif name == "LabelPadding":
                    trf.append(LabelPadding(max_word_length=int(item.get("maxLen", max_label_len)),
                                            padding_value=len(vocab))); has_pad = True
                elif name == "ImageNormalizer":
                    # allowed if explicitly requested in JSON
                    trf.append(ImageNormalizer(bool(item.get("transpose_axis", False))))
            if not has_index:
                trf.append(LabelIndexer(list(vocab)))
            if not has_pad:
                trf.append(LabelPadding(max_word_length=max_label_len, padding_value=len(vocab)))
        except Exception as e:
            print(f"[CGL] bad CGL_TRF_JSON ({e}); falling back to defaults")

    if not trf:
        # EXACT defaults you asked for (no ImageNormalizer)
        trf = [
            ImageResizer(int(W), int(H), keep_aspect_ratio=True),
            LabelIndexer(list(vocab)),
            LabelPadding(max_word_length=max_label_len, padding_value=len(vocab)),
        ]

    return pre, aug, trf





# ------------------------------ main ------------------------------
def main():
    # set_memory_growth()
    _tf_runtime_setup()
    _ensure_dirs()

    # Seed
    seed = _train_seed()
    if seed is not None:
        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    # Load JSONL
    if not ENV["JSONL_TRAIN"]:
        raise SystemExit("Set CL_JSONL_LOCAL or CL_JSONL_TRAIN to a JSONL file")
    rtl_pol = _rtl_policy()

    tr_path = ENV["JSONL_TRAIN"]
    va_path = ENV["JSONL_VAL"]
    te_path = ENV["JSONL_TEST"]
    
    if tr_path.startswith("s3://"):
        tr_path = _download_if_s3(tr_path, Path(ENV["CACHE_DIR"]) / "jsonl_train.jsonl")
    if va_path and va_path.startswith("s3://"):
        va_path = _download_if_s3(va_path, Path(ENV["CACHE_DIR"]) / "jsonl_val.jsonl")
    if te_path and te_path.startswith("s3://"):
        te_path = _download_if_s3(te_path, Path(ENV["CACHE_DIR"]) / "jsonl_test.jsonl")

    train_items, train_max = _load_jsonl(tr_path, ENV["IMAGE_BASE"], rtl_pol)
    rev_train = sum(1 for _,t in train_items if rtl_pol=='rtl')  # crude visibility
    print(f"[dbg] rtl_policy={_rtl_policy()} (train reversed≈{rev_train})")
    
    if va_path:
        val_items, _ = _load_jsonl(va_path, ENV["IMAGE_BASE_VAL"] or ENV["IMAGE_BASE"], rtl_pol)
    else:
        # fallback split from train
        random.shuffle(train_items)
        k = int(len(train_items) * (1.0 - float(HYP["val_split"])))
        val_items = train_items[k:]
        train_items = train_items[:k]

    test_items = []
    if te_path:
        try:
            test_items, _ = _load_jsonl(te_path, ENV["IMAGE_BASE_TEST"] or ENV["IMAGE_BASE_VAL"] or ENV["IMAGE_BASE"], rtl_pol)
        except Exception as e:
            print(f"[test] skip: {e}")

    print(f"[data] train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    
    print(f"[dbg] rtl_policy={_rtl_policy()}")
    print("[dbg] train[0]:", train_items[0][1][:40])
    print("[dbg]  val [0]:",  val_items[0][1][:40])

    # Vocab & max length
    vocab = _build_vocab(train_items + val_items)
    val_chars = {ch for _, t in (val_items or []) for ch in t}
    oov = val_chars - set(vocab)
    print(f"[data] vocab_len={len(vocab)}  val_OOV_chars={len(oov)}")
    # blank index like configs.blank_index; overridable via env
    blank_index = len(vocab)
    num_classes = blank_index + 1

    # max_len = max(train_max, int(HYP.get("max_label_len", 64)))
    val_max  = max((len(t) for _, t in (val_items or [])), default=0)
    test_max = max((len(t) for _, t in (test_items or [])), default=0)
    max_len  = max(train_max, val_max, test_max, int(HYP.get("max_label_len", 64)))

    # Pipeline
    H, W = int(HYP["height"]), int(HYP["width"])
    pre, aug, trf = build_pipeline_from_env(vocab, max_len, H, W)

    # Providers
    B = int(HYP["batch_size"])
     
    train_dp = DataProvider(
        dataset=train_items,
        skip_validation=True,
        batch_size=B,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(W, H, keep_aspect_ratio=True),
            LabelIndexer(vocab),
            LabelPadding(max_word_length=max_len, padding_value=len(vocab)),
        ],
    )

    val_dp = DataProvider(
        dataset=val_items,
        skip_validation=True,
        batch_size=B,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(W, H, keep_aspect_ratio=True),
            LabelIndexer(vocab),
            LabelPadding(max_word_length=max_len, padding_value=len(vocab)),
        ],
    )

    test_dp = None
    if test_items:
        test_dp = DataProvider(dataset=test_items,
                          skip_validation=True,
                          batch_size=B,
                          data_preprocessors=[ImageReader(CVImage)],                         
                          transformers=[
                            ImageResizer(W, H, keep_aspect_ratio=True),
                            LabelIndexer(vocab),
                            LabelPadding(max_word_length=max_len, padding_value=len(vocab)),
                        ])
        
    # 1) Inspect pixel range after transformers
    try:
        batch_x, batch_y = next(iter(train_dp))       
        m, M = float(np.min(batch_x)), float(np.max(batch_x))
        print(f"[data] train batch pixel range: [{m:.1f}, {M:.1f}]")
    except Exception as e:
        print(f"[data] pixel-range check skipped: {e}")

    # 2) Inspect model output time dimension vs label length
    print("[model] (will print after build)")


    # Build or load model
    arch = (HYP.get("arch") or "resnet_lstm").lower()
    out = Path(ENV["OUTPUT_DIR"]); (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    mode  = (ENV["CL_RESUME_MODE"] or "none").lower()       # "none"|"resume"|"finetune"
    which = (ENV["CL_RESUME_WHICH"] or "best").lower()      # "best"|"last"
    resume_uri = ENV["CL_RESUME_S3"] or ""
    resume_path = None

    # ---------------------------------------------------------
    # 1) Download checkpoint if s3:// (UNCHANGED)
    # ---------------------------------------------------------
    if mode in ("resume", "finetune") and resume_uri.startswith("s3://"):
        # dst = out / "checkpoints" / (
        #     f"{which}.keras" if resume_uri.endswith("/") else Path(resume_uri).name
        # )
        # Always download weights for resume / finetune
        dst = out / "checkpoints" / "best.weights.h5"
        rp = _download_if_s3(resume_uri, dst)
        if rp:
            resume_path = Path(rp)

    # Normalize resume_path
    if isinstance(resume_path, (str, Path)):
        resume_path = Path(resume_path)
    else:
        resume_path = None

    # ---------------------------------------------------------
    # 2) ALWAYS build the model (NO graph resume anymore)
    # ---------------------------------------------------------
    print(f"[model] building '{arch}' via model.train_model(...)")

    input_dim = (H, W, 3)
    output_dim = num_classes

    model = train_model(
        input_dim,
        output_dim,
        activation=HYP.get("activation", "leaky_relu"),
        dropout=float(HYP.get("dropout", 0.2)),
    )

        # ---------------------------------------------------------
    # 3) Load weights for resume / finetune (H5-first, auto-migrate)
    # ---------------------------------------------------------
    if mode in ("resume", "finetune") and resume_path and resume_path.exists():
        try:
            resume_path = Path(resume_path)

            # 🔹 If user passed best.keras, switch to best.weights.h5
            # if resume_path.name.endswith(".keras"):
            #     candidate = resume_path.with_name("best.weights.h5")
            #     if not candidate.exists():
            #         raise RuntimeError(
            #             f"Resume requested from {resume_path.name}, "
            #             f"but {candidate.name} was not found."
            #         )
            #     print(f"[weights] switching resume from {resume_path.name} → {candidate.name}")
            #     weights_path = candidate

            # # 🔹 If user already passed .weights.h5, use it directly
            # elif resume_path.name.endswith(".weights.h5"):
            #     weights_path = resume_path

            # else:
            #     raise RuntimeError(
            #         f"Unsupported resume checkpoint: {resume_path.name}. "
            #         "Expected best.weights.h5 or best.keras."
            #     )

            # weights_path = _ensure_h5_weights_from_checkpoint(resume_path)
            # weights_path = _resolve_resume_weights(resume_path)

            print(f"[weights] loading weights from {resume_path.name}")
            # model.load_weights(
            #     str(weights_path),
            #     by_name=True,
            #     skip_mismatch=True,
            # )
            model.load_weights(str(resume_path))   # ✅ FIXED
            print("[weights] initialized from checkpoint (graph-safe)")

        except Exception as e:
            raise RuntimeError(f"[weights] failed to prepare/load weights: {e}")


    # ---------------------------------------------------------
    # 4) Debug prints (UNCHANGED)
    # ---------------------------------------------------------
    print("[model] output_shape:", model.output_shape)
    print("[labels] max_label_len:", max_len)

    
    # --- CTC SAFETY CLAMP (MANDATORY) ---
    T = model.output_shape[1]

    if T is None:
        raise RuntimeError("CTC time dimension (T) is None; model output shape invalid")

    if max_len >= T:
        print(f"[ctc] WARNING: max_label_len ({max_len}) >= time_steps ({T}), clamping")
        max_len = T - 1

    print(f"[ctc] using max_label_len={max_len}, time_steps={T}")


    # Compile
    lr = float(ENV["CL_FT_LR"] or HYP["learning_rate"])
    pad_value = len(vocab);
    PAD = "\u25A1"  # any dummy char
    # you already have 'vocab' as a Python list of chars
    vocab_for_metrics = "".join(vocab) + PAD   # <-- string + one extra char
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                loss=CTCloss(),
                metrics=[
                CERMetric(vocabulary=vocab_for_metrics),
                WERMetric(vocabulary=vocab_for_metrics)
                ],
                run_eagerly=False)

    # Callbacks
    def _patience(default_val=10):
        v = ENV["CGL_EARLYSTOP_PATIENCE"].strip()
        if not v:
            return int(default_val)
        try:
            n = int(float(v)); return max(1, n)
        except Exception:
            return int(default_val)

    cbs = [
        ModelCheckpoint(str(out / "checkpoints" / "best.keras"),
                                           monitor="val_CER", mode="min", save_best_only=True, verbose=1),
        ModelCheckpoint(str(out / "checkpoints" / "last.keras"),
                                           save_best_only=False, verbose=0),
        ModelCheckpoint(str(out / "checkpoints" / "best.weights.h5"),
                save_weights_only=True,
                save_best_only=True,   # <-- IMPORTANT
                verbose=1,
            ),
        ModelCheckpoint(str(out / "checkpoints" / "last.weights.h5"),
                save_weights_only=True,
                save_best_only=False,   # <-- IMPORTANT
                verbose=1,
            ),
        EarlyStopping(monitor="val_CER", mode="min",
                                         patience=_patience(10), restore_best_weights=True, verbose=1),
        # ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode="min"),
        ReduceLROnPlateau(
                monitor="val_CER",
                factor=0.5,
                patience=6,
                min_delta=0.002,
                min_lr=1e-6,
                mode="min",
                verbose=1
            )
        # CSVLogger(str(out / "train.csv"), append=True),
    ]
    
    lsw = LastStateWriter(
    out_dir=str(out),
    hyp=HYP,
    data_counts={"train": len(train_items), "val": len(val_items), "test": len(test_items)},
    rtl_policy=_rtl_policy(),
    blank_index=blank_index,
    vocab=vocab,
    resume={"mode": mode, "which": which, "resume_uri": resume_uri, "resume_path": (str(resume_path) if resume_path else "")}
    )
    cbs.append(lsw)  # NEW

    
    def _peek_decode_cb(dp, every=1):
        def _cb(epoch, logs=None):
            if (epoch+1) % every: return
            try:
                batch = next(iter(dp))
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    print("[peek] unexpected batch structure; skip")
                    return
                x, y = batch
                # y can be a tensor of indices OR a (y_idx, seq_len, ...) container
                if isinstance(y, (tuple, list)) and len(y) >= 1:
                    y_idx = y[0]
                else:
                    y_idx = y
                # Get logits for sample 0
                pred = model.predict(x, verbose=0)
                blank = len(vocab)
                ids = np.argmax(pred[0], axis=-1)
                # CTC collapse  remove blanks
                ids = [k for k, g in itertools.groupby(ids) if k != blank]
                PAD = "\u25A1"
                dec = "".join(( "".join(vocab) + PAD )[i] if i < len(vocab) else "" for i in ids)
                # Ground truth for sample 0 (strip pads/blanks)
                if hasattr(y_idx, "numpy"):
                    y0 = y_idx[0].numpy().tolist()
                else:
                    y0 = np.array(y_idx)[0].tolist()
                gt_ids = [int(i) for i in y0 if int(i) != blank]
                ref = "".join(vocab[i] for i in gt_ids if 0 <= i < len(vocab))
                print(f"[peek] ref='{ref[:60]}'")
                print(f"[peek] hyp='{dec[:60]}'  (len={len(dec)})")
            except Exception as e:
                print(f"[peek] skipped: {e}")
        return tf.keras.callbacks.LambdaCallback(on_epoch_end=_cb)

    cbs.append(_peek_decode_cb(val_dp, every=1))
    # Optional: periodic checkpoints
    save_every = ENV["CGL_SAVE_EVERY_N_EPOCHS"].strip()
    if save_every:
        try: save_every = int(save_every)
        except: save_every = 0
    if save_every and save_every > 0:
        ep_dir = out / "checkpoints"; ep_dir.mkdir(parents=True, exist_ok=True)
        def _save_every_cb(epoch, logs=None):
            e = epoch + 1
            if e % save_every == 0:
                path = ep_dir / f"ep{e:03d}.keras"
                try:
                    model.save(str(path))
                    print(f"[ckpt] saved {path}")
                except Exception as _e:
                    print(f"[ckpt] skip ep{e}: {_e}")
        cbs.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=_save_every_cb))

    # Optional two-stage fine-tune with CNN freeze
    total_epochs = int(HYP["epochs"])
    freeze_epochs = int(ENV["CL_FREEZE_CNN_EPOCHS"] or 0)
    stage2 = total_epochs

    if mode == "finetune" and freeze_epochs > 0:
        print(f"[finetune] stage1: freeze CNN for {freeze_epochs} epoch(s)")
        for L in model.layers:
            lname = L.name.lower()
            if "lstm" in lname or "bidirectional" in lname or "dense" in lname or "softmax" in lname:
                break
            L.trainable = False

        model.fit(train_dp, validation_data=val_dp, epochs=freeze_epochs, callbacks=cbs, verbose=1)

        # unfreeze & recompile
        for L in model.layers: L.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss=CTCloss(),
                      metrics=[])

        stage2 = max(0, total_epochs - freeze_epochs)

    if stage2 > 0:
        print(f"[train] full model for {stage2} epoch(s)")
        model.fit(train_dp, validation_data=val_dp, epochs=stage2, callbacks=cbs, verbose=1)

    # Save final
    model.save(str(out / "model.keras"))
    
    export_dir = out / "export"  # will contain saved_model.pb, variables/, assets/
    try:
        # Keras 3 preferred export for inference
        model.export(str(export_dir))
        print(f"[export] SavedModel exported to {export_dir}")
    except Exception as e1:
        try:
            # Fallback that still works for serving
            tf.saved_model.save(model, str(export_dir))
            print(f"[export] SavedModel exported via tf.saved_model.save to {export_dir}")
        except Exception as e2:
            print(f"[export] skipped: {e1} / {e2}")
    
    lsw.finalize(final_model_path=str(out / "model.keras"), export_dir=str(export_dir))  # NEW
            
    # 1) Save the vocabulary (ordered list) and mapping
    char_to_index = {ch: i for i, ch in enumerate(vocab)}
    index_to_char = {i: ch for ch, i in char_to_index.items()}
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab": vocab,                 # ordered list of characters
                "blank_index": len(vocab),      # matches training
                "char_to_index": char_to_index, # convenient for clients
                "index_to_char": index_to_char,
            },
            f, ensure_ascii=False, indent=2
        )

    # 2) Save minimal runtime config your inference service will need
    configs = {
        "width": int(HYP["width"]),
        "height": int(HYP["height"]),
        "channels": 3,
        "max_label_len": int(max_len),
        "arch": str(HYP.get("arch", "")),
        "activation": str(HYP.get("activation", "")),
        "dropout": float(HYP.get("dropout", 0.2)),
        "vocab_size": len(vocab) + 1,          # includes blank
        "blank_index": len(vocab),
        "rtl_policy": _rtl_policy(),           # how text was preprocessed
        "normalize_inputs": False,             # flip to True if you add ImageNormalizer
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tf_version": tf.__version__,
        "python": platform.python_version(),
    }
    
    with open(out / "configs.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)

    # ONNX (optional)
    if ENV["CGL_EXPORT_ONNX"].strip().lower() in ("1","true","yes"):
        try:
            import tf2onnx, os, tensorflow as _tf
            # force CPU so Keras maps LSTMs to standard kernels
            prev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            _tf.config.set_visible_devices([], 'GPU')
            
            input_spec = (tf.TensorSpec((None, H, W, 3), tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec, opset=13)
            with open(out / "model.onnx", "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"[onnx] exported to {out/'model.onnx'}")
        except Exception as e:
            print(f"[onnx] export skipped: {e}")
        finally:
            # restore
            if prev is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev

    # Optional test
    if test_dp is not None:
        print("[test] evaluating on test split...")
        _ = model.evaluate(test_dp, verbose=1)

    # Optional upload to S3
    s3_prefix = ENV["CL_OUTPUT_S3_PREFIX"].strip()
    if s3_prefix.startswith("s3://"):
        try:
            import boto3
            from urllib.parse import urlparse
            u = urlparse(s3_prefix)
            bucket = u.netloc
            key_prefix = u.path.lstrip("/").rstrip("/") + "/"

            s3 = boto3.client("s3")
            def _put(local_path, rel_key):
                if os.path.exists(local_path):
                    s3.upload_file(str(local_path), bucket, key_prefix + rel_key)
                    
            # NEW: recursive upload for directories (SavedModel export/)
            def _put_dir(local_dir, rel_prefix):
                local_dir = str(local_dir)
                if not os.path.isdir(local_dir):
                    return
                for root, _, files in os.walk(local_dir):
                    for name in files:
                        lp = os.path.join(root, name)
                        rel = os.path.relpath(lp, local_dir).replace("\\", "/")
                        s3.upload_file(lp, bucket, key_prefix + rel_prefix.rstrip("/") + "/" + rel)

            _put(out / "checkpoints" / "best.keras", "checkpoints/best.keras")
            _put(out / "checkpoints" / "last.keras", "checkpoints/last.keras")
            _put(out / "checkpoints" / "best.weights.h5", "checkpoints/best.weights.h5")
            _put(out / "checkpoints" / "last.weights.h5", "checkpoints/last.weights.h5")
            _put(out / "train.csv", "train.csv")
            _put(out / "model.keras", "model.keras")
            _put(out / "model.onnx", "model.onnx")
            
            # NEW uploads
            _put(out / "vocab.json",   "vocab.json")
            _put(out / "configs.json", "configs.json")
            _put(out / "last_state.json", "last_state.json")
            _put_dir(out / "export",   "export")
            
            
            print(f"[upload] pushed to {s3_prefix}")
        except Exception as e:
            print(f"[upload] skipped: {e}")

if __name__ == "__main__":
    main()
