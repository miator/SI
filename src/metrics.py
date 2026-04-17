import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def build_labels_scores(same_scores, diff_scores):
    same = np.asarray(same_scores, dtype=np.float64).reshape(-1)
    diff = np.asarray(diff_scores, dtype=np.float64).reshape(-1)
    scores = np.concatenate([same, diff], axis=0)
    labels = np.concatenate(
        [
            np.ones_like(same, dtype=np.int64),
            np.zeros_like(diff, dtype=np.int64)],
        axis=0)
    return labels, scores


def eer_from_roc(fpr, tpr, thresholds):
    fnr = 1.0 - np.asarray(tpr, dtype=np.float64)
    fpr = np.asarray(fpr, dtype=np.float64)
    thr = np.asarray(thresholds, dtype=np.float64)

    diff = fpr - fnr
    i = int(np.argmin(np.abs(diff)))

    if i == 0 or i == len(diff) - 1 or diff[i] == 0.0:
        eer = float(0.5 * (fpr[i] + fnr[i]))
        return eer, float(thr[i]), float(fpr[i]), float(fnr[i])

    x0, y0 = float(fpr[i - 1]), float(fnr[i - 1])
    x1, y1 = float(fpr[i]), float(fnr[i])
    d0 = x0 - y0
    d1 = x1 - y1
    t = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.0
    t = float(np.clip(t, 0.0, 1.0))

    fpr_e = x0 + t * (x1 - x0)
    fnr_e = y0 + t * (y1 - y0)
    eer = float(0.5 * (fpr_e + fnr_e))

    thr_e = float(thr[i - 1] + t * (thr[i] - thr[i - 1]))
    return eer, thr_e, float(fpr_e), float(fnr_e)


def compute_roc_auc_eer(same_scores, diff_scores):
    labels, scores = build_labels_scores(same_scores, diff_scores)
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    auc = float(roc_auc_score(labels, scores))
    eer, eer_thr, fpr_e, fnr_e = eer_from_roc(fpr, tpr, thr)
    return {
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "fpr_at_eer": fpr_e,
        "fnr_at_eer": fnr_e}
