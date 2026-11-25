"""Data drift from KS and PSI, and label-based quality alerts."""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

from .data import CONFIG, IMG_H, IMG_W

SETTINGS = CONFIG["monitor"]


def image_stats(dataset):
    rows = []
    for index in range(len(dataset)):
        image, _ = dataset.load(index)
        grey = np.asarray(image.resize((IMG_W, IMG_H)).convert("L")).astype(np.float32)
        laplacian = (
            grey[:-2, 1:-1]
            + grey[2:, 1:-1]
            + grey[1:-1, :-2]
            + grey[1:-1, 2:]
            - 4 * grey[1:-1, 1:-1]
        )
        rows.append([grey.mean(), grey.std(), laplacian.var()])
    return np.array(rows)


def psi(reference, current):
    # Reference quantiles keep the bins fixed when a production batch changes.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, SETTINGS["psi_bins"] + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference) + SETTINGS["psi_epsilon"]
    cur = np.histogram(current, edges)[0] / len(current) + SETTINGS["psi_epsilon"]
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_alarm(flagged):
    return flagged >= SETTINGS["alarm_features"]


class DriftMonitor:
    def __init__(self, reference_features, reference_stats):
        self.names = ["brightness", "contrast", "sharpness"] + [
            f"feature_pc{i + 1}" for i in range(SETTINGS["pca_components"])
        ]
        self.pca = PCA(SETTINGS["pca_components"], random_state=SETTINGS["pca_seed"])
        self.pca.fit(reference_features)
        self.reference = np.hstack([reference_stats, self.pca.transform(reference_features)])

    def check(self, features, stats):
        current = np.hstack([stats, self.pca.transform(features)])
        rows = []
        for index, name in enumerate(self.names):
            p = float(ks_2samp(self.reference[:, index], current[:, index]).pvalue)
            value = psi(self.reference[:, index], current[:, index])
            rows.append(
                {
                    "feature": name,
                    "ks_p": p,
                    "psi": value,
                    "flag": p < SETTINGS["ks_alpha"] and value >= SETTINGS["psi_threshold"],
                }
            )
        flagged = sum(row["flag"] for row in rows)
        return {
            "features": rows,
            "flagged": flagged,
            "alarm": drift_alarm(flagged),
            "mean_psi": float(np.mean([row["psi"] for row in rows])),
            "max_psi": float(np.max([row["psi"] for row in rows])),
        }


def quality_alert(batch, reference):
    return bool(
        batch["ap"] < reference["ap"] - SETTINGS["ap_drop"]
        or batch["recall"] < reference["recall"] - SETTINGS["recall_drop"]
    )
