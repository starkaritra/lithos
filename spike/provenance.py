"""Provenance capture (spike-prereg.md §8). An unreproducible result is not
evidence — log versions, seeds, data hash, full parameters, git commit."""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def collect(cfg: dict, bundle, seed: int) -> dict:
    import numpy
    import sklearn
    import xgboost

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            "xgboost": xgboost.__version__,
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "seed": seed,
        "dataset": {
            "name": bundle.name,
            "n_features": bundle.n_features,
            "n_classes": bundle.n_classes,
            "objective": bundle.objective,
            "n_train_rows": int(bundle.X_train.shape[0]),
            "n_eval_rows": int(bundle.X_eval.shape[0]),
            "data_sha256": bundle.data_hash,
        },
        "config": cfg,
        "git_commit": _git_commit(),
        "modelled_not_measured": True,
    }
