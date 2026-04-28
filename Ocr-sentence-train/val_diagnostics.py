import numpy as np
import tensorflow as tf
from collections import Counter, defaultdict


def bucket_by_edges(value, edges, labels=None):
    if value is None:
        return "unknown"

    if labels is None:
        labels = []
        prev = None
        for e in edges:
            if prev is None:
                labels.append(f"<= {e}")
            else:
                labels.append(f"{prev + 1}-{e}")
            prev = e
        labels.append(f"> {edges[-1]}")

    for i, e in enumerate(edges):
        if value <= e:
            return labels[i]
    return labels[-1]


def levenshtein_alignment(ref: str, hyp: str):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    i, j = n, m
    aligned = []
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((ref[i - 1], None, "del"))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            aligned.append((None, hyp[j - 1], "ins"))
            j -= 1
        else:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            aligned.append((ref[i - 1], hyp[j - 1], "eq" if cost == 0 else "sub"))
            i -= 1
            j -= 1

    aligned.reverse()
    return aligned


def cer_simple(ref: str, hyp: str):
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    aligned = levenshtein_alignment(ref, hyp)
    edits = sum(1 for _, _, op in aligned if op != "eq")
    return edits / max(1, len(ref))


def greedy_decode_logits(logits, vocab, blank_index):
    probs_all = tf.nn.softmax(logits, axis=-1).numpy()
    idxs = probs_all.argmax(axis=-1)

    texts = []
    for b in range(idxs.shape[0]):
        prev = -1
        tokens = []
        for k in idxs[b]:
            k = int(k)
            if k != blank_index and k != prev:
                tokens.append(k)
            prev = k
        texts.append("".join(vocab[k] for k in tokens if 0 <= k < len(vocab)))
    return texts


def _decode_label_indices(label_row, vocab, blank_index):
    chars = []
    for k in label_row:
        k = int(k)
        if k == blank_index:
            continue
        if 0 <= k < len(vocab):
            chars.append(vocab[k])
    return "".join(chars)


def _get_geom_from_provider_item(item):
    try:
        img_path = item[0]
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            return {
                "orig_width": None,
                "orig_height": None,
                "orig_aspect_ratio": None,
            }

        h, w = img.shape[:2]
        return {
            "orig_width": int(w),
            "orig_height": int(h),
            "orig_aspect_ratio": float(w / max(1, h)),
        }
    except Exception:
        return {
            "orig_width": None,
            "orig_height": None,
            "orig_aspect_ratio": None,
        }


def run_val_diagnostics(
    model,
    data_provider,
    dataset_items,
    vocab,
    blank_index,
    length_bucket_edges=None,
    width_bucket_edges=None,
    height_bucket_edges=None,
    aspect_bucket_edges=None,
):
    if length_bucket_edges is None:
        length_bucket_edges = [40, 80, 140]
    if width_bucket_edges is None:
        width_bucket_edges = [400, 800, 1200, 2000]
    if height_bucket_edges is None:
        height_bucket_edges = [40, 60, 80, 120]
    if aspect_bucket_edges is None:
        aspect_bucket_edges = [4, 8, 12, 20]

    rows = []
    dataset_items = list(dataset_items)
    consumed = 0

    total_batches = len(data_provider)
    print(f"[diag] provider batches={total_batches}")

    for batch_idx in range(total_batches):
        try:
            batch_x, batch_y = data_provider[batch_idx]
        except Exception as e:
            print(f"[diag] skip batch {batch_idx}: failed to load provider batch: {e}")
            continue

        if batch_x is None or len(batch_x) == 0:
            print(f"[diag] skip batch {batch_idx}: empty batch_x")
            continue

        current_batch_size = len(batch_x)

        try:
            logits = model(batch_x, training=False)
            if isinstance(logits, tf.Tensor):
                logits = logits.numpy()
        except Exception as e:
            print(
                f"[diag] batch predict failed: batch={batch_idx}, "
                f"x_shape={getattr(batch_x, 'shape', None)}, "
                f"y_shape={getattr(batch_y, 'shape', None)}, "
                f"err={e}"
            )
            consumed += current_batch_size
            continue

        preds = greedy_decode_logits(logits, vocab, blank_index)
        refs = [_decode_label_indices(row, vocab, blank_index) for row in batch_y]

        batch_items = dataset_items[consumed:consumed + current_batch_size]
        consumed += current_batch_size

        min_len = min(len(batch_items), len(refs), len(preds))
        batch_items = batch_items[:min_len]
        refs = refs[:min_len]
        preds = preds[:min_len]

        for item, ref, pred in zip(batch_items, refs, preds):
            geom = _get_geom_from_provider_item(item)

            rows.append({
                "img": item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else None,
                "ref": ref,
                "pred": pred,
                "cer": cer_simple(ref, pred),
                "label_length": len(ref),
                "space_count": ref.count(" "),
                "orig_width": geom["orig_width"],
                "orig_height": geom["orig_height"],
                "orig_aspect_ratio": geom["orig_aspect_ratio"],
                "label_length_bucket": bucket_by_edges(len(ref), length_bucket_edges),
                "orig_width_bucket": bucket_by_edges(geom["orig_width"], width_bucket_edges),
                "orig_height_bucket": bucket_by_edges(geom["orig_height"], height_bucket_edges),
                "orig_aspect_bucket": bucket_by_edges(geom["orig_aspect_ratio"], aspect_bucket_edges),
            })

    if not rows:
        return {
            "error": "Diagnostics produced no rows; all batches likely failed or were skipped",
            "mean_cer": None,
            "top_error_chars": [],
            "cer_by_label_length_bucket": {},
            "cer_by_orig_width_bucket": {},
            "cer_by_orig_height_bucket": {},
            "cer_by_orig_aspect_bucket": {},
            "sample_count": 0,
            "failed_or_skipped_samples": len(dataset_items),
        }

    error_char_counter = Counter()
    ref_char_counter = Counter()

    cer_by_length_bucket = defaultdict(list)
    cer_by_width_bucket = defaultdict(list)
    cer_by_height_bucket = defaultdict(list)
    cer_by_aspect_bucket = defaultdict(list)

    for row in rows:
        ref = row["ref"]
        pred = row["pred"]

        for ch in ref:
            ref_char_counter[ch] += 1

        aligned = levenshtein_alignment(ref, pred)
        for rch, _, op in aligned:
            if op != "eq" and rch is not None:
                error_char_counter[rch] += 1

        cer_by_length_bucket[row["label_length_bucket"]].append(row["cer"])
        cer_by_width_bucket[row["orig_width_bucket"]].append(row["cer"])
        cer_by_height_bucket[row["orig_height_bucket"]].append(row["cer"])
        cer_by_aspect_bucket[row["orig_aspect_bucket"]].append(row["cer"])

    return {
        "mean_cer": float(np.mean([r["cer"] for r in rows])),
        "top_error_chars": error_char_counter.most_common(100),
        "char_error_profile": [
            {
                "char": ch,
                "support": int(support),
                "errors": int(error_char_counter.get(ch, 0)),
                "error_rate": float(error_char_counter.get(ch, 0) / max(1, support)),
            }
            for ch, support in sorted(
                ref_char_counter.items(),
                key=lambda kv: (error_char_counter.get(kv[0], 0) / max(1, kv[1]), kv[1]),
                reverse=True
            )[:200]
        ],
        "cer_by_label_length_bucket": {
            k: float(np.mean(v)) for k, v in cer_by_length_bucket.items()
        },
        "cer_by_orig_width_bucket": {
            k: float(np.mean(v)) for k, v in cer_by_width_bucket.items()
        },
        "cer_by_orig_height_bucket": {
            k: float(np.mean(v)) for k, v in cer_by_height_bucket.items()
        },
        "cer_by_orig_aspect_bucket": {
            k: float(np.mean(v)) for k, v in cer_by_aspect_bucket.items()
        },
        "sample_count": len(rows),
        "failed_or_skipped_samples": max(0, len(dataset_items) - len(rows)),
    }