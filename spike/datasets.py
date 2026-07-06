"""Dataset loading + XGBoost training with caching, hashing, provenance (§6, §8).

Canonical benchmark is HIGGS (binary, 28 dense feats); Covertype (7-class, 54
feats) is the robustness set that stresses the working-set guardrail; a small
synthetic dataset backs the offline smoke test. Data and trained models are
cached under spike/cache/ so a re-run is one command and deterministic.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


@dataclass
class DataBundle:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    n_features: int
    n_classes: int
    objective: str
    data_hash: str


def _sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _subsample(X, y, n, seed):
    if n and X.shape[0] > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return X[idx], y[idx]
    return X, y


def _make_synthetic(seed: int, n_features: int = 28, n_rows: int = 20000):
    """A small, fast, offline binary dataset shaped like HIGGS (smoke only)."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_rows, n_features=n_features, n_informative=16,
        n_redundant=6, n_classes=2, random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int32), 2, "binary:logistic"


def _load_raw(name: str, seed: int):
    """Return (X, y, n_classes, objective) for a named dataset."""
    if name == "synthetic":
        return _make_synthetic(seed)
    if name == "covertype":
        from sklearn.datasets import fetch_covtype

        ds = fetch_covtype()
        X = ds.data.astype(np.float32)
        y = (ds.target.astype(np.int32) - 1)  # 1..7 -> 0..6
        return X, y, 7, "multi:softprob"
    if name == "higgs":
        from sklearn.datasets import fetch_openml

        ds = fetch_openml("higgs", version=2, as_frame=False, parser="liac-arff")
        X = np.asarray(ds.data, dtype=np.float32)
        y = np.asarray(ds.target, dtype=np.float64)
        y = np.nan_to_num(y).astype(np.int32)
        # HIGGS from OpenML can carry a few NaN feature cells; the cost model
        # treats NaN as a "missing" branch, but training is cleaner imputed.
        X = np.nan_to_num(X, nan=0.0)
        return X, y, 2, "binary:logistic"
    raise ValueError(f"unknown dataset {name!r}")


def load_dataset(name: str, n_train_rows: int, n_eval_rows: int,
                 seed: int) -> DataBundle:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"data_{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y = d["X"], d["y"]
        n_classes, objective = int(d["n_classes"]), str(d["objective"])
    else:
        X, y, n_classes, objective = _load_raw(name, seed)
        np.savez_compressed(cache, X=X, y=y, n_classes=n_classes, objective=objective)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    n_eval = min(n_eval_rows, X.shape[0] // 4)
    X_eval = X[:n_eval]
    X_tr, y_tr = X[n_eval:], y[n_eval:]
    X_tr, y_tr = _subsample(X_tr, y_tr, n_train_rows, seed)

    return DataBundle(
        name=name, X_train=X_tr, y_train=y_tr, X_eval=X_eval,
        n_features=X.shape[1], n_classes=n_classes, objective=objective,
        data_hash=_sha256(X_eval, y[:n_eval]),
    )


def train_or_load(bundle: DataBundle, n_estimators: int, max_depth: int,
                  cfg: dict, seed: int):
    """Train (or load a cached) XGBoost booster for the given size."""
    import xgboost as xgb

    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"model_{bundle.name}_{n_estimators}_{max_depth}.ubj"
    path = os.path.join(CACHE_DIR, key)
    booster = xgb.Booster()
    if os.path.exists(path):
        booster.load_model(path)
        return booster

    params = {
        "max_depth": max_depth,
        "eta": cfg["xgb"]["canonical"]["learning_rate"],
        "tree_method": cfg["xgb"]["canonical"]["tree_method"],
        "objective": bundle.objective,
        "seed": seed,
    }
    if bundle.objective.startswith("multi"):
        params["num_class"] = bundle.n_classes

    dtrain = xgb.DMatrix(bundle.X_train, label=bundle.y_train)
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators)
    booster.save_model(path)
    return booster
