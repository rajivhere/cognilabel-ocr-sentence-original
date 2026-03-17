import tensorflow as tf
import time 

def _fmt_num(v, digits=4):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _pick_metric(logs, *names):
    if not logs:
        return None
    for name in names:
        if name in logs and logs[name] is not None:
            return logs[name]
    return None


class EpochSummaryLogger(tf.keras.callbacks.Callback):
    """
    Prints exactly one clean line per epoch, suitable for CloudWatch.

    Example:
    [2026-03-16 20:12:44] (3m 27s) Epoch 4/50 - 423/423 - CER: 1.0302 - WER: 1.0000 - loss: 233.7086 - val_CER: 0.9921 - val_WER: 0.9810 - val_loss: 220.1134
    """

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        self.epoch_start_time = None
        self.steps_per_epoch = self.params.get("steps")
        self.total_epochs = self.params.get("epochs")
        print(
            f"[train] started - epochs={self.total_epochs} steps_per_epoch={self.steps_per_epoch}",
            flush=True
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_train_batch_end(self, batch, logs=None):
        step = batch + 1
        total = self.steps_per_epoch

        # emit step progress for client progress bar
        print(f"[STEP] {step}/{total}", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed_total = int(time.time() - self.train_start_time)
        elapsed_epoch = int(time.time() - self.epoch_start_time) if self.epoch_start_time else None

        mins = elapsed_total // 60
        secs = elapsed_total % 60

        cer = _pick_metric(logs, "CER", "cer")
        wer = _pick_metric(logs, "WER", "wer")
        loss = _pick_metric(logs, "loss")

        val_cer = _pick_metric(logs, "val_CER", "val_cer")
        val_wer = _pick_metric(logs, "val_WER", "val_wer")
        val_loss = _pick_metric(logs, "val_loss")

        steps_text = f"{self.steps_per_epoch}/{self.steps_per_epoch}" if self.steps_per_epoch else "?/?"
        epoch_time_text = f"{elapsed_epoch}s" if elapsed_epoch is not None else "n/a"

        line = (
            f"[{now_ts}] "
            f"({mins}m {secs}s) "
            f"Epoch {epoch + 1}/{self.total_epochs} "
            f"- {steps_text} "
            f"- epoch_time: {epoch_time_text} "
            f"- CER: {_fmt_num(cer)} "
            f"- WER: {_fmt_num(wer)} "
            f"- loss: {_fmt_num(loss)}"
        )

        if val_cer is not None:
            line += f" - val_CER: {_fmt_num(val_cer)}"
        if val_wer is not None:
            line += f" - val_WER: {_fmt_num(val_wer)}"
        if val_loss is not None:
            line += f" - val_loss: {_fmt_num(val_loss)}"

        print(line, flush=True)

    def on_train_end(self, logs=None):
        elapsed_total = int(time.time() - self.train_start_time)
        mins = elapsed_total // 60
        secs = elapsed_total % 60
        print(f"[train] finished - total_time={mins}m {secs}s", flush=True)