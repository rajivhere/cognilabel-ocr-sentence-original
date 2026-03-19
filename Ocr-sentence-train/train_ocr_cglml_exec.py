#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cognilabel OCR Training (CGL-DATA compliant)

- Uses cgl_data.resolver for dataset access
- Uses OutputManager for outputs
- Compatible with local / S3 / future providers
"""

import os, json, random, time, platform
from pathlib import Path

import numpy as np
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.annotations.images import CVImage

from model1 import train_model
from last_state_writer import LastStateWriter

# ✅ NEW: Cognilabel data + outputs
from cgl_data.resolver import resolve_split, resolve_resume_weights
from cgl_data.outputs import OutputManager
from uploadUtil import UploadBestOnImprove, UploadLastEveryEpoch
from tensorflow.keras import mixed_precision
from cgl_data.logging.keras.logger import CGLKerasLogger, CGLEarlyStopping
from cgl_data.logging.emitter import emit
from cgl_data.logging.events import CGL_EVAL

mixed_precision.set_global_policy("mixed_float16")



# --------------------------------------------------
# helpers
# --------------------------------------------------
def setup_tf():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass


def build_vocab(samples):
    chars = set()
    for _, txt in samples:
        chars.update(txt)
    return sorted(chars)


def load_jsonl(path, text_dir="ltr", max_reasonable_len=1000):
    items = []
    max_len = 0

    def normalize_label(txt: str) -> str:
        txt = txt.strip()
        txt = " ".join(txt.split())
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

            if not txt:
                print(f"[data] skipping empty/whitespace-only label at line {line_no}")
                continue

            if text_dir == "rtl":
                txt = txt[::-1]

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
    tjson = trf_json.strip()
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
    batch_size = env("CGL_BATCH_SIZE", 8, int)
    lr         = env("CGL_LR", 1e-3, float)
    width      = env("CGL_WIDTH", 1048, int)
    height     = env("CGL_HEIGHT", 96, int)
    dropout    = env("CGL_DROPOUT", 0.2, float)
    activation = env("CGL_ACTIVATION", "leaky_relu", str)

    # 🔹 Behavior / flags
    text_dir = env("CGL_TEXT_DIR", "ltr", str)
    early_patience = env("CGL_EARLY_STOP_PATIENCE", 10, int)

    resume_mode  = env("CGL_RESUME_MODE", "none", str)
    resume_which = env("CGL_RESUME_WHICH", "best", str)
    resume_path  = env("CGL_RESUME_PATH", "", str)

    ft_lr = env("CGL_FT_LR", None, float)
    freeze_cnn_epochs = env("CGL_FREEZE_CNN_EPOCHS", 0, int)

    augment_json = env("CGL_AUG_JSON", "", str)
    tranformer_json = env("CGL_TRF_JSON", "", str)  # typo preserved intentionally

    # --------------------------------------------------
    # 🔹 Setup
    # --------------------------------------------------

    setup_tf()
    cache_root = Path(cache_dir)

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
        "batch_size": batch_size,
        "lr": lr,
        "width": width,
        "height": height,
        "dropout": dropout,
        "activation": activation,
        "text_dir": text_dir,
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
        "batch_size": batch_size,
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
    if test_ref:
        test_items,  test_max  = load_jsonl(test_ref.local_annotations_path, text_dir or "ltr") if test_ref else ([], 0)
        test_items = absolutize_items(test_items, test_ref.local_root)
    else:
        test_items = []

    vocab = build_vocab(train_items + val_items)
    blank_index = len(vocab)
    max_len = max(train_max, val_max, test_max)
    
    print(f"[dataset] train samples: {len(train_items)}")
    print(f"[dataset] val samples: {len(val_items)}")
    print(f"[dataset] test samples: {len(test_items)}")
    print(f"[dataset] vocab size: {len(vocab)}")
    print(f"[dataset] max label length: {max_len}")

    # --------------------------------------------------
    # 🔹 Data providers
    # --------------------------------------------------

    train_dp = DataProvider(
        dataset=train_items,
        batch_size=batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(width, height, keep_aspect_ratio=True),
            LabelIndexer(vocab),
            LabelPadding(max_word_length=max_len, padding_value=blank_index),
        ],
    )

    val_dp = DataProvider(
        dataset=val_items,
        skip_validation=True,
        batch_size=batch_size,
        data_preprocessors=[ImageReader(CVImage)],
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
            batch_size=batch_size,
            data_preprocessors=[ImageReader(CVImage)],
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
    
    model = train_model(
        input_dim=(height, width, 3),
        output_dim=len(vocab) + 1,
        dropout=dropout,
        activation=activation,
    )

    # Resume / finetune
    if resume_mode in ("resume", "finetune") and resume_path:
        print(f"[resume] mode={resume_mode}, which={resume_which}")

        # ---------------------------------------------------
        # Fix resume path
        # ---------------------------------------------------
        ckpt_file = "best.keras" if resume_which == "best" else "last.keras"

        # Ensure no trailing slash
        resume_path = resume_path.rstrip("/")

        resume_uri = f"{resume_path}/{ckpt_file}"

        print(f"[weights] loading weights from {resume_uri}")

        # ---------------------------------------------------
        # Resolve weights through cgl-data
        # ---------------------------------------------------
        local_weights = resolve_resume_weights(resume_uri, refs)

        model.load_weights(local_weights)

        print("[weights] initialized from checkpoint")
    # ---------------------------------------------------------
    # 4) Debug prints (UNCHANGED)
    # ---------------------------------------------------------
    print("[model] output_shape:", model.output_shape)
    print("[labels] max_label_len:", max_len)

    
    # --- CTC SAFETY CLAMP (MANDATORY) ---
    T = model.output_shape[1]

    if T is None:
        raise RuntimeError("CTC time dimension (T) is None; model output shape invalid")

    # if max_len >= T:
    #     print(f"[ctc] WARNING: max_label_len ({max_len}) >= time_steps ({T}), clamping")
    #     max_len = T - 1
    if max_len >= T:
        raise RuntimeError(
            f"CTC invalid: max_label_len={max_len} >= time_steps={T}. "
            "This should have been validated client-side."
        )

    print(f"[ctc] using max_label_len={max_len}, time_steps={T}")


    # Compile
   
    pad_value = len(vocab);
    PAD = "\u25A1"  # any dummy char
    # you already have 'vocab' as a Python list of chars
    vocab_for_metrics = "".join(vocab) + PAD   # <-- string + one extra char

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=CTCloss(),
        metrics=[
            CERMetric(vocabulary=vocab_for_metrics),
            WERMetric(vocabulary=vocab_for_metrics),
        ],
    )
    
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
            factor=0.5,
            patience=6,
            min_delta=0.002,
            min_lr=1e-6,
            mode="min",
            verbose=0,
        ),
        # ---- Upload policies via outputs.py ----
        UploadBestOnImprove(outputs, monitor="val_CER", mode="min"),
        UploadLastEveryEpoch(outputs),

    ]

    callbacks.append(
        LastStateWriter(
            out_dir=str(refs.local_root),
            hyp={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
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

    # ---------------- training ----------------
    model.fit(
        train_dp,
        validation_data=val_dp,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )
    
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


    metrics_payload = {
    "status": "completed",
    "framework": "ocr_ctc",
    "task_type": "ocr",
    "training": {
        "epochs_requested": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
    },
    "dataset": {
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "test_samples": len(test_items),
        "vocab_size": len(vocab),
        "max_label_len": max_len,
    },
}
    if test_metrics:
        metrics_payload["test_samples"] = len(test_items)
        metrics_payload["test_metrics"] = test_metrics
        
    last_state_path = Path(refs.local_root) / "last_state.json"

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
