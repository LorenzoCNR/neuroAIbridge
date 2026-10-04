from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


def center_scale(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(X)
    if scale <= eps:
        return X
    return X / scale


def _as_2d_float(X, name):
    """Validate a representation/target matrix without changing its columns."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {values.shape}")
    if values.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _sampled_pair_distances(values, i, j, metric):
    """Compute distances for sampled pairs while retaining scipy metrics."""
    left = values[i]
    right = values[j]
    delta = left - right
    if metric == "euclidean":
        return np.sqrt(np.einsum("ij,ij->i", delta, delta))
    if metric == "sqeuclidean":
        return np.einsum("ij,ij->i", delta, delta)
    if metric == "cityblock":
        return np.abs(delta).sum(axis=1)
    if metric == "chebyshev":
        return np.abs(delta).max(axis=1)
    if metric == "cosine":
        numerator = np.einsum("ij,ij->i", left, right)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - numerator / denominator
        # scipy defines cosine distance for zero vectors as zero when both
        # vectors are zero and one otherwise; mirror that convention here.
        both_zero = (np.linalg.norm(left, axis=1) == 0) & (np.linalg.norm(right, axis=1) == 0)
        one_zero = (np.linalg.norm(left, axis=1) == 0) ^ (np.linalg.norm(right, axis=1) == 0)
        result[both_zero] = 0.0
        result[one_zero] = 1.0
        return result
    # Keep support for any additional scipy metric, while making the common
    # metrics above vectorized.  This branch is intentionally bounded by the
    # caller's max_pairs argument.
    return np.asarray([
        pdist(np.asarray([values[ii], values[jj]]), metric=metric)[0]
        for ii, jj in zip(i, j)
    ])


def procrustes_align(source, target):
    """
    Align source to target using orthogonal Procrustes.

    source and target must have the same shape.
    """
    source = center_scale(source)
    target = center_scale(target)

    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")

    # Solve min_R ||source @ R - target||_F with R.T @ R = I.
    #
    # This is the same SVD solution used by orthogonal Procrustes:
    # source.T @ target = U S V.T, then R = U V.T.
    # Keeping the small decomposition explicit avoids a SciPy LAPACK
    # deadlock observed on some Windows scientific-Python environments.
    cross_covariance = source.T @ target
    left_vectors, _, right_vectors_t = np.linalg.svd(
        cross_covariance,
        full_matrices=False,
    )
    R = left_vectors @ right_vectors_t
    aligned = source @ R
    return aligned, R


def procrustes_r2(embedding, latent):
    embedding = _as_2d_float(embedding, "embedding")
    latent = _as_2d_float(latent, "latent")

    if embedding.shape[0] != latent.shape[0]:
        raise ValueError("embedding and latent must have the same number of samples")

    if embedding.shape[1] != latent.shape[1]:
        n_components = latent.shape[1]
        if embedding.shape[1] < n_components:
            raise ValueError(
                "embedding must have at least as many dimensions as latent "
                "when dimensions differ"
            )
        embedding = PCA(n_components=n_components).fit_transform(embedding)

    aligned, _ = procrustes_align(embedding, latent)
    latent_scaled = center_scale(latent)
    residual_sum_squares = np.square(
        latent_scaled - aligned
    ).sum()
    total_sum_squares = np.square(latent_scaled).sum()
    if total_sum_squares <= 1e-12:
        return float("nan")
    return float(1.0 - residual_sum_squares / total_sum_squares)


def distance_geometry_correlation(
    embedding,
    latent,
    metric="euclidean",
    method="spearman",
    max_pairs=100_000,
    random_state=0,
):
    """Correlate pairwise geometries without materialising an N-by-N matrix.

    For small samples the exact upper triangle is used.  For large window
    collections, a deterministic uniform sample of pairs keeps RSA bounded
    in memory and makes the metric usable on full-sample neural recordings.
    """
    embedding = _as_2d_float(embedding, "embedding")
    latent = _as_2d_float(latent, "latent")

    if embedding.shape[0] != latent.shape[0]:
        raise ValueError("embedding and latent must have the same number of samples")
    n_samples = embedding.shape[0]
    n_pairs = n_samples * (n_samples - 1) // 2
    if max_pairs is None:
        max_pairs = n_pairs
    if int(max_pairs) < 1:
        raise ValueError("max_pairs must be positive or None")
    max_pairs = int(max_pairs)
    if n_pairs <= max_pairs:
        D_embedding = squareform(pdist(embedding, metric=metric))
        D_latent = squareform(pdist(latent, metric=metric))
        idx = np.triu_indices(n_samples, k=1)
        x = D_embedding[idx]
        y = D_latent[idx]
    else:
        rng = np.random.default_rng(random_state)
        i = rng.integers(0, n_samples, size=max_pairs)
        j = rng.integers(0, n_samples, size=max_pairs)
        valid = i != j
        i, j = i[valid], j[valid]
        # Preserve the requested scipy distance metric in the bounded-memory
        # path.  The previous implementation silently used Euclidean
        # distances for every metric when sampling pairs.
        x = _sampled_pair_distances(embedding, i, j, metric)
        y = _sampled_pair_distances(latent, i, j, metric)

    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")

    if method == "spearman":
        corr, _ = spearmanr(x, y)
    elif method == "pearson":
        corr, _ = pearsonr(x, y)
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

    return float(corr)


def linear_cka(X, Y, *, eps=1e-12):
    """Return linear centered kernel alignment between two sample matrices.

    CKA is complementary to Procrustes recovery: it is invariant to
    isotropic rescaling and orthogonal changes of coordinates, but it does not
    assert a one-to-one rigid alignment.  The implementation uses centered
    cross-covariance, so the two matrices may have different feature counts.
    """
    X = _as_2d_float(X, "X")
    Y = _as_2d_float(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = Xc.T @ Yc
    numerator = float(np.square(cross).sum())
    x_norm = float(np.square(Xc.T @ Xc).sum())
    y_norm = float(np.square(Yc.T @ Yc).sum())
    denominator = float(np.sqrt(x_norm * y_norm))
    if denominator <= eps:
        return float("nan")
    return float(numerator / denominator)


def evaluate_latent_recovery(embedding, latent, max_pairs=100_000):
    return {
        "procrustes_r2": procrustes_r2(embedding, latent),
        "rsa_spearman": distance_geometry_correlation(
            embedding, latent, method="spearman", max_pairs=max_pairs
        ),
        "rsa_pearson": distance_geometry_correlation(
            embedding, latent, method="pearson", max_pairs=max_pairs
        ),
        "linear_cka": linear_cka(embedding, latent),
    }


def lagged_alignment_scores(embedding_ref, embedding_other, lags):
    """
    Evaluate Procrustes R2 between two embeddings over candidate temporal lags.
    Positive lag compares ref[t] to other[t + lag].
    """
    scores = {}
    embedding_ref = np.asarray(embedding_ref)
    embedding_other = np.asarray(embedding_other)

    for lag in lags:
        lag = int(lag)
        if lag > 0:
            ref = embedding_ref[:-lag]
            other = embedding_other[lag:]
        elif lag < 0:
            ref = embedding_ref[-lag:]
            other = embedding_other[:lag]
        else:
            ref = embedding_ref
            other = embedding_other

        if len(ref) < 3:
            scores[lag] = np.nan
            continue

        scores[lag] = procrustes_r2(other, ref)

    best_lag = max(scores, key=lambda key: -np.inf if np.isnan(scores[key]) else scores[key])
    return best_lag, scores


def lagged_alignment_by_trial_time(
        embedding_ref,
        embedding_other,
        trial_id_ref,
        time_id_ref,
        trial_id_other,
        time_id_other,
        lags, *, common_support=False):
    """
    Evaluate Procrustes R2 over candidate lags using trial/time metadata.

    Positive lag compares ref(trial, time) with other(trial, time + lag).
    This is safer than shifting the flattened array because windows are grouped
    by trial and should not wrap across trial boundaries.
    """
    embedding_ref = np.asarray(embedding_ref)
    embedding_other = np.asarray(embedding_other)
    trial_id_ref = np.asarray(trial_id_ref).reshape(-1)
    time_id_ref = np.asarray(time_id_ref).reshape(-1)
    trial_id_other = np.asarray(trial_id_other).reshape(-1)
    time_id_other = np.asarray(time_id_other).reshape(-1)

    if len(embedding_ref) != len(trial_id_ref) or len(embedding_ref) != len(time_id_ref):
        raise ValueError("reference metadata must match reference embedding length")
    if len(embedding_other) != len(trial_id_other) or len(embedding_other) != len(time_id_other):
        raise ValueError("other metadata must match other embedding length")

    lags = list(lags)
    if not lags or any(not isinstance(lag, (int, np.integer)) for lag in lags):
        raise ValueError("lags must be a non-empty sequence of integers")
    for trial_ids, time_ids in ((trial_id_ref, time_id_ref), (trial_id_other, time_id_other)):
        if not np.all(np.isfinite(trial_ids)) or not np.all(np.isfinite(time_ids)):
            raise ValueError("trial/time metadata must be finite")
        if not np.all(trial_ids == np.floor(trial_ids)) or not np.all(time_ids == np.floor(time_ids)):
            raise ValueError("lag alignment requires integer trial/time metadata")
        if len(set(zip(trial_ids.tolist(), time_ids.tolist()))) != len(trial_ids):
            raise ValueError("Duplicate trial/time coordinates are ambiguous")

    other_lookup = {
        (int(trial), int(time)): embedding_other[idx]
        for idx, (trial, time) in enumerate(zip(trial_id_other, time_id_other))
    }

    scores = {}
    aligned_pairs = {}
    eligible = np.ones(len(embedding_ref), dtype=bool)
    if common_support:
        eligible = np.asarray([
            all((int(trial), int(time) + int(lag)) in other_lookup for lag in lags)
            for trial, time in zip(trial_id_ref, time_id_ref)
        ])
    for lag in lags:
        ref_points = []
        other_points = []
        for idx, (trial, time) in enumerate(zip(trial_id_ref, time_id_ref)):
            if not eligible[idx]:
                continue
            key = (int(trial), int(time) + int(lag))
            if key not in other_lookup:
                continue
            ref_points.append(embedding_ref[idx])
            other_points.append(other_lookup[key])

        if len(ref_points) < 3:
            scores[int(lag)] = np.nan
            aligned_pairs[int(lag)] = (np.empty((0, embedding_ref.shape[1])), np.empty((0, embedding_other.shape[1])))
            continue

        ref_points = np.asarray(ref_points)
        other_points = np.asarray(other_points)
        scores[int(lag)] = procrustes_r2(other_points, ref_points)
        aligned_pairs[int(lag)] = (ref_points, other_points)

    finite = {lag: score for lag, score in scores.items() if np.isfinite(score)}
    if not finite:
        raise ValueError("No candidate lag has sufficient non-degenerate aligned observations")
    best_lag = max(finite, key=finite.get)
    return best_lag, scores, aligned_pairs
