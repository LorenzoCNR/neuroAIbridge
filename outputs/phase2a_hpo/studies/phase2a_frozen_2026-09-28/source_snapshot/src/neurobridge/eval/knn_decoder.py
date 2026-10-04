# src/neurobridge/eval/knn_decoder.py

import numpy as np
from typing import Sequence, Optional, Tuple, Dict, Any
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score


def _as_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is (n, p)."""
    a = np.asarray(a)
    return a[:, None] if a.ndim == 1 else a


def r2_pos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² for position targets (multi-output if p > 1)."""
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    multi = 'uniform_average' if y_true.shape[1] > 1 else 'variance_weighted'
    return float(r2_score(y_true, y_pred, multioutput=multi))


def mederr_pos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median error: |x̂-x| for 1D, Euclidean norm for multi-D."""
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    if y_true.shape[1] == 1:
        err = np.abs(y_pred[:, 0] - y_true[:, 0])
    else:
        err = np.linalg.norm(y_pred - y_true, axis=1)
    return float(np.median(err))


def acc_dir(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy for direction labels; mean across columns if multi-output."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        return float(accuracy_score(y_true, y_pred))
    return float(np.mean([accuracy_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]))


def knn_decode_pos(
    emb_train: np.ndarray,
    y_pos_train: np.ndarray,
    emb_eval: np.ndarray,
    k: int,
    metric: str = 'cosine',
) -> np.ndarray:
    """KNN regression for continuous position (1D or multi-D)."""
    y_pos_train = _as_2d(y_pos_train)
    dec = KNeighborsRegressor(n_neighbors=int(k), metric=metric)
    dec.fit(emb_train, y_pos_train)
    y_pos_pred = dec.predict(emb_eval)
    return y_pos_pred


def knn_decode_dir(
    emb_train: np.ndarray,
    y_dir_train: np.ndarray,
    emb_eval: np.ndarray,
    k: int,
    metric: str = 'cosine',
) -> np.ndarray:
    """KNN classification for discrete direction (1D or multi-output)."""
    dec = KNeighborsClassifier(n_neighbors=int(k), metric=metric)
    dec.fit(emb_train, y_dir_train)
    y_dir_pred = dec.predict(emb_eval)
    return y_dir_pred


def select_k(
    emb_train: np.ndarray,
    emb_valid: np.ndarray,
    y_pos_train: Optional[np.ndarray] = None,
    y_pos_valid: Optional[np.ndarray] = None,
    y_dir_train: Optional[np.ndarray] = None,
    y_dir_valid: Optional[np.ndarray] = None,
    *,
    k_grid: Sequence[int] = (1, 4, 9, 16, 25),
    metric: str = 'cosine',
    objective: str = 'pos_r2',   # 'pos_r2' | 'pos_mederr' | 'dir_acc'
) -> Tuple[int, Dict[str, Any]]:
    """Select optimal k using validation set based on the chosen metric."""
    best_k, best_val = None, -np.inf
    best_stats: Dict[str, Any] = {}

    for k in k_grid:
        cur_stats: Dict[str, Any] = {"k": int(k)}

        if objective in ('pos_r2', 'pos_mederr'):
            if y_pos_train is None or y_pos_valid is None:
                raise ValueError("Objective 'pos_*' requires position labels.")
            y_pos_train_2d = _as_2d(y_pos_train)
            pos_pred_v = knn_decode_pos(emb_train, y_pos_train_2d, emb_valid, k, metric)
            cur_stats["val_pos_r2"] = r2_pos(y_pos_valid, pos_pred_v)
            cur_stats["val_pos_mederr"] = mederr_pos(y_pos_valid, pos_pred_v)
            score = cur_stats["val_pos_r2"] if objective == 'pos_r2' else -cur_stats["val_pos_mederr"]

        elif objective == 'dir_acc':
            if y_dir_train is None or y_dir_valid is None:
                raise ValueError("Objective 'dir_acc' requires direction labels.")
            dir_pred_v = knn_decode_dir(emb_train, y_dir_train, emb_valid, k, metric)
            cur_stats["val_dir_acc"] = acc_dir(y_dir_valid, dir_pred_v)
            score = cur_stats["val_dir_acc"]

        else:
            raise ValueError(f"Unknown objective: {objective}")

        if score > best_val:
            best_val = score
            best_k = int(k)
            best_stats = cur_stats

    return best_k, best_stats
