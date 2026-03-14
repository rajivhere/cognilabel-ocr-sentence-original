import tensorflow as tf
from cgl_data.outputs import OutputManager


class UploadBestOnImprove(tf.keras.callbacks.Callback):
    """
    When val_loss improves, upload both:
      - checkpoints/best.weights.h5
      - checkpoints/best.keras  (if exists)
    Train code does NOT know provider; calls outputs.py
    """
    def __init__(self, outputs: OutputManager, monitor="val_loss", mode="min"):
        self.outputs = outputs
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        v = logs.get(self.monitor)
        if v is None:
            return

        improved = (v < self.best) if self.mode == "min" else (v > self.best)
        if not improved:
            return

        self.best = v

        refs = self.outputs.refs()
        # upload weights
        self.outputs.upload_checkpoint(refs.checkpoints_dir / "best.weights.h5", "best.weights.h5")
        # upload full model if produced
        self.outputs.upload_checkpoint(refs.checkpoints_dir / "best.keras", "best.keras")


class UploadLastEveryEpoch(tf.keras.callbacks.Callback):
    """
    Every epoch upload:
      - checkpoints/last.weights.h5
      - checkpoints/last.keras (if exists)
    """
    def __init__(self, outputs: OutputManager):
        self.outputs = outputs

    def on_epoch_end(self, epoch, logs=None):
        refs = self.outputs.refs()
        self.outputs.upload_checkpoint(refs.checkpoints_dir / "last.weights.h5", "last.weights.h5")
        self.outputs.upload_checkpoint(refs.checkpoints_dir / "last.keras", "last.keras")
