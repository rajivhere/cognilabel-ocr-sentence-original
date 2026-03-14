# Ocr-sentence-train/last_state_writer.py
import json, time, os
from pathlib import Path
import tensorflow as tf

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _safe_float(x):
    try:
        return float(tf.keras.backend.get_value(x))
    except Exception:
        try:
            return float(x)
        except Exception:
            return None

def _current_lr(model):
    # Try common places for LR in tf.keras
    try:
        opt = model.optimizer
        # Keras 2/TF 2.15 style
        if hasattr(opt, "learning_rate"):
            lr = opt.learning_rate
            # If it's a schedule, try calling with step
            try:
                if callable(lr):
                    step = tf.cast(opt.iterations, tf.float32)
                    return float(lr(step).numpy())
            except Exception:
                pass
            return _safe_float(lr)
        # Legacy alias
        if hasattr(opt, "lr"):
            return _safe_float(opt.lr)
    except Exception:
        pass
    return None

class LastStateWriter(tf.keras.callbacks.Callback):
    def __init__(
        self,
        out_dir: str,
        *,
        hyp: dict,
        data_counts: dict,
        rtl_policy: str,
        blank_index: int,
        vocab=None,                 # optional list (small; ok to store)
        resume: dict = None,        # {mode, which, resume_uri, resume_path}
    ):
        super().__init__()
        self.out = Path(out_dir)
        self.path = self.out / "last_state.json"
        self.hyp = dict(hyp or {})
        self.data_counts = dict(data_counts or {})
        self.rtl_policy = str(rtl_policy)
        self.blank_index = int(blank_index)
        self.vocab = list(vocab) if vocab is not None else None
        self.resume = dict(resume or {})
        self.start_time = None
        self.best = None            # {'epoch': int, 'val_CER': float, ...}
        self._status = "starting"

    def _write(self, payload: dict):
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2,default=str)
        os.replace(tmp, self.path)  # atomic on POSIX

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self._status = "running"
        self._write({
            "status": self._status,
            "started_at": _now_iso(),
            "epoch": 0,
            "last_metrics": {},
            "best_so_far": None,
            "context": {
                "width": int(self.hyp.get("width", 0)),
                "height": int(self.hyp.get("height", 0)),
                "batch_size": int(self.hyp.get("batch_size", 0)),
                "epochs_planned": int(self.hyp.get("epochs", 0)),
                "learning_rate_init": float(self.hyp.get("learning_rate", 0.0)),
                "dropout": float(self.hyp.get("dropout", 0.0)),
                "arch": self.hyp.get("arch", ""),
                "rtl_policy": self.rtl_policy,
                "blank_index": self.blank_index,
                "vocab_size": (len(self.vocab) + 1) if self.vocab else None,
                "data_counts": self.data_counts,
                "resume": self.resume,
            },
            "paths": {
                "output_dir": str(self.out),
                "checkpoint_best": str(self.out / "checkpoints" / "best.keras"),
                "checkpoint_last": str(self.out / "checkpoints" / "last.keras"),
                "train_csv": str(self.out / "train.csv"),
                "final_model": str(self.out / "model.keras"),
                "export_dir": str(self.out / "export"),
                "vocab_json": str(self.out / "vocab.json"),
                "configs_json": str(self.out / "configs.json"),
            },
            "updated_at": _now_iso(),
        })

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = {
            "epoch": int(epoch + 1),
            "loss": float(logs.get("loss")) if logs.get("loss") is not None else None,
            "CER":  float(logs.get("CER"))  if logs.get("CER")  is not None else None,
            "WER":  float(logs.get("WER"))  if logs.get("WER")  is not None else None,
            "val_loss": float(logs.get("val_loss")) if logs.get("val_loss") is not None else None,
            "val_CER":  float(logs.get("val_CER"))  if logs.get("val_CER")  is not None else None,
            "val_WER":  float(logs.get("val_WER"))  if logs.get("val_WER")  is not None else None,
            "lr": _current_lr(self.model),
        }
        # Track best by lowest val_CER if present
        if current.get("val_CER") is not None:
            if (self.best is None) or (current["val_CER"] < self.best.get("val_CER", float("inf"))):
                self.best = dict(current)

        payload = {
            "status": self._status,
            "started_at": None,
            "epoch": current["epoch"],
            "last_metrics": current,
            "best_so_far": self.best,
            "updated_at": _now_iso(),
        }
        # Merge minimal immutable context so the file is self-descriptive
        try:
            prev = json.loads(Path(self.path).read_text(encoding="utf-8"))
            for k in ("context", "paths", "started_at"):
                if k in prev:
                    payload[k] = prev[k]
        except Exception:
            pass

        self._write(payload)
        
    def to_metrics_summary(self):
        return {
            "best": self.best,
            "last": self.best if self.best and self.best.get("epoch") == self.model.stop_training else None
        }


    # Optional: call after you save final artifacts, to stamp final paths/sizes
    def finalize(self, *, final_model_path=None, export_dir=None):
        try:
            st = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            st = {}
        st["status"] = "completed"
        st["completed_at"] = _now_iso()
        if self.start_time:
            st["duration_sec"] = round(max(0.0, time.time() - self.start_time), 3)
        # refresh any final paths you want to guarantee
        if final_model_path:
            st.setdefault("paths", {})
            st["paths"]["final_model"] = str(final_model_path)
        if export_dir:
            st.setdefault("paths", {})
            st["paths"]["export_dir"] = str(export_dir)
        st["updated_at"] = _now_iso()
        self._write(st)
