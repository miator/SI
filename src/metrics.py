import numpy as np


def build_labels_scores(same_scores, diff_scores):
    same = np.asarray(same_scores, dtype=np.float64).reshape(-1)
    diff = np.asarray(diff_scores, dtype=np.float64).reshape(-1)
    scores = np.concatenate([same, diff], axis=0)
    labels = np.concatenate([np.ones_like(same, dtype=np.int64),
                            np.zeros_like(diff, dtype=np.int64)], axis=0)  # binar labels 1-positive, 0-negative
    return labels, scores


# Receiver operator characteristic
def roc_curve(labels, scores):
    """
    Higher score => more likely same-speaker.
    Returns fpr, tpr, thresholds (thresholds descending).
    Goal: produce arrays (fpr, tpr, thresholds).
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError("labels and scores must have same length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary 0/1")

    # highest similarity first
    order = np.argsort(-s, kind="mergesort")  # -s for descending sort
    y = y[order]  # sorted array of true labels
    s = s[order]  # sorted scores (cosine similarities)

    # count positives and negatives
    P = int(y.sum())
    N = int(len(y) - P)
    if P == 0 or N == 0:
        raise ValueError("need both positive and negative samples")

    # Simulate sweeping threshold
    # Array where each i represents the total number of tp/fp found up to that position
    tp = np.cumsum(y == 1)  # cumulative sum
    fp = np.cumsum(y == 0)  # summing y==0 true statements, not 0s)

    # ROC should only include a point when the threshold moves to a new unique score
    # otherwise you get repeated points that add nothing.
    distinct = np.where(np.diff(s))[0]
    # one ROC point per unique threshold
    idx = np.r_[distinct, len(s) - 1]  # concatenate 1D array of indices before score change & final run

    tpr = tp[idx] / P
    fpr = fp[idx] / N
    thr = s[idx]

    # starting point(0.0, 0.0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thr = np.r_[thr[0] + 1e-12, thr]

    return fpr, tpr, thr


def auc_trapz(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    o = np.argsort(fpr)  # default is quicksort, ascending
    return float(np.trapezoid(tpr[o], fpr[o]))


# point where FPR ≈ FNR, balance of security and usability
def eer_from_roc(fpr, tpr, thresholds):
    fnr = 1.0 - np.asarray(tpr, dtype=np.float64)
    fpr = np.asarray(fpr, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64)

    diff = fpr - fnr
    i = int(np.argmin(np.abs(diff)))  # index of the ROC points where |FPR-FNR| is minimal

    if i == 0 or i == len(diff) - 1 or diff[i] == 0.0:
        # 1.the first point, no i-1, 2.the last point, no i+1, 3.exact equality already
        eer = float(0.5 * (fpr[i] + fnr[i]))
        return eer, float(thr[i]), float(fpr[i]), float(fnr[i])

    x0, y0 = float(fpr[i - 1]), float(fnr[i - 1])
    x1, y1 = float(fpr[i]), float(fnr[i])
    d0 = x0 - y0
    d1 = x1 - y1
    t = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.0
    t = float(np.clip(t, 0.0, 1.0))

    # estimation
    fpr_e = x0 + t * (x1 - x0)
    fnr_e = y0 + t * (y1 - y0)
    eer = float(0.5 * (fpr_e + fnr_e))

    thr_e = float(thr[i - 1] + t * (thr[i] - thr[i - 1]))
    return eer, thr_e, float(fpr_e), float(fnr_e)


def compute_roc_auc_eer(same_scores, diff_scores):
    labels, scores = build_labels_scores(same_scores, diff_scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    auc = auc_trapz(fpr, tpr)
    eer, eer_thr, fpr_e, fnr_e = eer_from_roc(fpr, tpr, thr)
    return {
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "fpr_at_eer": fpr_e,
        "fnr_at_eer": fnr_e,
    }
