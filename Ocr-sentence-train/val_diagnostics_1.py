from collections import Counter, defaultdict
import numpy as np
import tensorflow as tf
import cv2


def bucket_by_edges(value, edges, labels=None):
    """
    edges: sorted list like [40, 80, 140]
    gives buckets:
      <=40, 41-80, 81-140, >140
    """
    if labels is None:
        labels = []
        prev = None
        for e in edges:
            if prev is None:
                labels.append(f"<= {e}")
            else:
                labels.append(f"{prev+1}-{e}")
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
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution / match
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
        for t, k in enumerate(idxs[b]):
            if k != blank_index and k != prev:
                tokens.append(k)
            prev = k
        txt = "".join(vocab[k] for k in tokens if 0 <= k < len(vocab))
        texts.append(txt)
    return texts


def prepare_image_for_model(img_path, width, height):
    """
    IMPORTANT:
    img_path should already point to the CURRENT validation image path used by training.
    If deterministic preprocessing cache is enabled, this is already the cached preprocessed PNG.
    So we do NOT apply deterministic preprocessing again here.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    h0, w0 = img.shape[:2]
    scale = min(width / w0, height / h0)
    nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    resized = cv2.resize(img, (nw, nh))

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    ox = (width - nw) // 2
    oy = (height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized

    # keep original geometry stats for later analysis
    geom = {
        "orig_width": int(w0),
        "orig_height": int(h0),
        "orig_aspect_ratio": float(w0 / max(1, h0)),
        "resized_width": int(nw),
        "resized_height": int(nh),
        "resize_scale": float(scale),
    }

    # model has Lambda(x / 255), so feed float32 in 0..255
    arr = canvas.astype("float32")
    return arr, geom


def run_val_diagnostics(
    model,
    items,
    vocab,
    blank_index,
    width,
    height,
    batch_size=32,
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

    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch_x = []
        refs = []
        paths = []
        geoms = []

        for img_path, txt in batch_items:
            arr, geom = prepare_image_for_model(img_path, width, height)
            batch_x.append(arr)
            refs.append(txt)
            paths.append(img_path)
            geoms.append(geom)

        if not batch_x:
            continue

        batch_x = np.stack(batch_x, axis=0)
        logits = model.predict(batch_x, verbose=0)
        preds = greedy_decode_logits(logits, vocab, blank_index)

        for img_path, ref, pred, geom in zip(paths, refs, preds, geoms):
            rows.append({
                "img": img_path,
                "ref": ref,
                "pred": pred,
                "cer": cer_simple(ref, pred),
                "label_length": len(ref),
                "space_count": ref.count(" "),
                "orig_width": geom["orig_width"],
                "orig_height": geom["orig_height"],
                "orig_aspect_ratio": geom["orig_aspect_ratio"],
                "resized_width": geom["resized_width"],
                "resized_height": geom["resized_height"],
                "resize_scale": geom["resize_scale"],
                "label_length_bucket": bucket_by_edges(len(ref), length_bucket_edges),
                "orig_width_bucket": bucket_by_edges(geom["orig_width"], width_bucket_edges),
                "orig_height_bucket": bucket_by_edges(geom["orig_height"], height_bucket_edges),
                "orig_aspect_bucket": bucket_by_edges(geom["orig_aspect_ratio"], aspect_bucket_edges),
            })

    error_char_counter = Counter()
    ref_char_counter = Counter()
    error_rate_by_ref_char = {}

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
        for rch, hch, op in aligned:
            if op != "eq" and rch is not None:
                error_char_counter[rch] += 1

        cer_by_length_bucket[row["label_length_bucket"]].append(row["cer"])
        cer_by_width_bucket[row["orig_width_bucket"]].append(row["cer"])
        cer_by_height_bucket[row["orig_height_bucket"]].append(row["cer"])
        cer_by_aspect_bucket[row["orig_aspect_bucket"]].append(row["cer"])

    for ch, support in ref_char_counter.items():
        errs = error_char_counter.get(ch, 0)
        error_rate_by_ref_char[ch] = {
            "support": int(support),
            "errors": int(errs),
            "error_rate": float(errs / max(1, support)),
        }

    worst_examples = sorted(rows, key=lambda x: x["cer"], reverse=True)[:100]

    report = {
        "samples": len(rows),
        "mean_cer": float(np.mean([r["cer"] for r in rows])) if rows else None,

        "top_error_chars": error_char_counter.most_common(100),

        "char_error_profile": [
            {"char": ch, **stats}
            for ch, stats in sorted(
                error_rate_by_ref_char.items(),
                key=lambda kv: (kv[1]["error_rate"], kv[1]["support"]),
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

        "worst_examples": worst_examples,
    }

    return report