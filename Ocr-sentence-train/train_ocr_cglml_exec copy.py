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


# --------------------------------------------------
# helpers
# --------------------------------------------------
def setup_tf():
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def build_vocab(samples):
    chars = set()
    for _, txt in samples:
        chars.update(txt)
    return sorted(chars)


def load_jsonl(path, text_dir='ltr'):
    items = []
    max_len = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            img = rec.get("file") or rec.get("image") or rec.get("path")
            txt = rec.get("text") or rec.get("transcription")
            if not img or not txt:
                continue
            is_rtl = (text_dir == "rtl") 
            if is_rtl:
                txt = txt[::-1]
            items.append((img, txt))
            max_len = max(max_len, len(txt))
    return items, max_len




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
    import argparse

    parser = argparse.ArgumentParser()

    # 🔹 Generic Cognilabel arguments
    parser.add_argument("--dataset_uri", required=True)
    parser.add_argument("--outputs_uri", required=True)
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--job_name", default="ocr-train")
    parser.add_argument("--cache_dir", default=os.getenv("CACHE_DIR", "/tmp/cgl_cache"))

    # 🔹 Hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=1048)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default='leaky_relu')
   
    
    
    # Env variables
    parser.add_argument("--text-dir", type=str, default= os.getenv("CGL_TEXT_DIR", 'ltr'))
    parser.add_argument("--early_patience",type=int, default=os.getenv("CGL_EARLY_STOP_PATIENCE", 10))
    parser.add_argument("--resume_mode",type=str, default=os.getenv("CGL_RESUME_MODE", "none"))
    parser.add_argument("--resume_which",type=str, default=os.getenv("CGL_RESUME_WHICH", "best"))
    parser.add_argument("--resume_path",type=str, default=os.getenv("CGL_RESUME_PATH", ""))
    parser.add_argument("--ft_lr",type=float, default=os.getenv("CGL_FT_LR", ""))
    parser.add_argument("--freeze_cnn_epochs",type=int, default=os.getenv("CGL_FREEZE_CNN_EPOCHS", 0))    
    parser.add_argument("--augment_json",default=os.getenv("CGL_AUG_JSON", ""), help="JSON list of augmentations (Cognilabel pipeline)")
    parser.add_argument("--tranformer_json",default=os.getenv("CGL_TRF_JSON", ""), help="JSON list of transformers (Cognilabel pipeline)")  
        

    args = parser.parse_args()

    setup_tf()

    cache_root = Path(args.cache_dir)

    # ---------------- outputs ----------------
    outputs = OutputManager(
        outputs_uri=args.outputs_uri,
        cache_root=cache_root,
        job_id=args.job_id,
        job_name=args.job_name,
        async_uploads=True,
        max_workers=2,
    )

    refs = outputs.refs()
    
        # ---- artifacts (local) ----
    refs.artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(refs.artifacts_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    config_snapshot = {
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "activation": args.activation,
    }
    with open(refs.artifacts_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    


    # ---------------- dataset ----------------
    train_ref = resolve_split(args.dataset_uri, "train", cache_root)
    val_ref   = resolve_split(args.dataset_uri, "val", cache_root)

    test_ref = None
    try:
        test_ref = resolve_split(args.dataset_uri, "test", cache_root)
    except Exception:
        pass
    
    dataset_manifest = {
        "dataset_uri": args.dataset_uri,
        "splits": {
            "train": {"annotations": str(train_ref.local_annotations_path), "images_dir": str(train_ref.local_images_dir)},
            "val":   {"annotations": str(val_ref.local_annotations_path),   "images_dir": str(val_ref.local_images_dir)},
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
    train_items, train_max = load_jsonl(train_ref.local_annotations_path,  args.text_dir or "ltr")
    val_items,   val_max   = load_jsonl(val_ref.local_annotations_path, args.text_dir or "ltr" )
    test_items,  test_max  = load_jsonl(test_ref.local_annotations_path,  args.text_dir or "ltr") if test_ref else ([], 0)

    vocab = build_vocab(train_items + val_items)
    blank_index = len(vocab)
    max_len = max(train_max, val_max, test_max)

    # ---------------- providers ----------------
    train_dp = DataProvider(
        dataset=train_items,
        batch_size=args.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(args.width, args.height, keep_aspect_ratio=True),
            LabelIndexer(vocab),
            LabelPadding(max_word_length=max_len, padding_value=blank_index),
        ],
    )

    val_dp = DataProvider(
        dataset=val_items,
        skip_validation=True,
        batch_size=args.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(args.width, args.height, keep_aspect_ratio=True),
            LabelIndexer(vocab),
            LabelPadding(max_word_length=max_len, padding_value=blank_index),
        ],
    )

    test_dp = None
    if test_items:
        test_dp = DataProvider(dataset=test_items,
                          skip_validation=True,
                          batch_size=args.batch_size,
                          data_preprocessors=[ImageReader(CVImage)],                         
                          transformers=[
                            ImageResizer(args.width, args.height, keep_aspect_ratio=True),
                            LabelIndexer(vocab),
                            LabelPadding(max_word_length=max_len, padding_value=len(vocab)),
                        ])
        
        
    
    
    # ---------------- model ----------------
    model = train_model(
        input_dim=(args.height, args.width, 3),
        output_dim=len(vocab) + 1, #num of classes
        dropout=args.dropout,
        activation=args.activation,
    )
    
    # --------------------------------------------------
    # Resume / finetune config (Cognilabel contract)
    # --------------------------------------------------
    
    # ENV = os.environ
    resume_mode  =  args.resume_mode # (ENV.get("CL_RESUME_MODE") or "none").lower()   # none | resume | finetune
    resume_which = args.resume_which #(ENV.get("CL_RESUME_WHICH") or "best").lower() # best | last
    resume_uri   = args.resume_path #ENV.get("CL_RESUME_PATH") or ""

    ft_lr = args.ft_lr   #ENV.get("CL_FT_LR")
    freeze_cnn_epochs = args.freeze_cnn_epochs #ENV.get("CL_FREEZE_CNN_EPOCHS")

    ft_lr = float(ft_lr) if ft_lr is not None else None
    freeze_cnn_epochs = int(freeze_cnn_epochs) if freeze_cnn_epochs else 0
    
   
    if resume_mode in ("resume", "finetune") and resume_uri:
        print(f"[resume] mode={resume_mode}, which={resume_which}")
                
        local_weights = resolve_resume_weights(resume_uri, refs)
        
        print(f"[weights] loading weights from {Path(resume_uri).name}")
        model.load_weights(local_weights, by_name=True)
        print("[weights] initialized from checkpoint (graph-safe)")
        # if rp:
        #     resume_path = Path(rp)
        
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
   
    pad_value = len(vocab);
    PAD = "\u25A1"  # any dummy char
    # you already have 'vocab' as a Python list of chars
    vocab_for_metrics = "".join(vocab) + PAD   # <-- string + one extra char

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=CTCloss(),
        metrics=[
            CERMetric(vocabulary=vocab_for_metrics),
            WERMetric(vocabulary=vocab_for_metrics),
        ],
    )
    
    # def _patience(default=10):
    #     try:
    #         return int(os.getenv("CGL_EARLYSTOP_PATIENCE", default))
    #     except Exception:
    #         return default


    # ---------------- callbacks ----------------
    
    
    ckpt_dir = refs.models_dir
    
    
    callbacks = [
        # 🔹 Full model – best
        ModelCheckpoint(
            ckpt_dir / "best.keras",
            monitor="val_CER",
            mode="min",
            save_best_only=True,
            verbose=1,
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
            verbose=1,
        ),

        # 🔹 Weights only – last
        ModelCheckpoint(
            ckpt_dir / "last.weights.h5",
            save_weights_only=True,
            save_best_only=False,
            verbose=0,
        ),

        # 🔹 Early stopping (env-driven)
        EarlyStopping(
            monitor="val_CER",
            mode="min",
            patience=args.early_patience,
            restore_best_weights=True,
            verbose=1,
        ),

        # 🔹 LR schedule
        ReduceLROnPlateau(
            monitor="val_CER",
            factor=0.5,
            patience=6,
            min_delta=0.002,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
        # ---- Upload policies via outputs.py ----
        UploadBestOnImprove(outputs, monitor="val_CER", mode="min"),
        UploadLastEveryEpoch(outputs),

    ]

    callbacks.append(
        LastStateWriter(
            out_dir=str(refs.local_root),
            hyp={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "width": args.width,
                "height": args.height,
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
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # ---------------- test evaluation ----------------
    test_metrics = None
    if test_dp:
        print("Running test evaluation...")
        results = model.evaluate(test_dp, verbose=1)

        # Keras returns list aligned with model.metrics_names
        test_metrics = dict(zip(model.metrics_names, map(float, results)))


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
        "epochs_requested": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
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

    outputs.write_metrics(metrics_payload) 
    outputs.upload_metrics()   
    
    outputs.upload_artifact(refs.artifacts_dir / "dataset_manifest.json", "dataset_manifest.json")
    outputs.upload_artifact(refs.artifacts_dir / "train_args.json", "train_args.json")
    outputs.upload_artifact(refs.artifacts_dir / "config.json", "config.json")

    

    outputs.finalize()

if __name__ == "__main__":
    main()
