import unittest

import numpy as np

from neurobridge.experiments.real_monkey import _participation_ratio
from src.neurobridge.eval.representation import (
    distance_geometry_correlation,
    evaluate_latent_recovery,
    lagged_alignment_scores,
    linear_cka,
    procrustes_align,
    procrustes_r2,
)


class TestRepresentationEval(unittest.TestCase):
    @staticmethod
    def _latent(seed=0, n=80, d=3):
        return np.random.default_rng(seed).normal(size=(n, d))

    def test_identity_is_near_perfect_for_all_geometry_metrics(self):
        latent = self._latent()
        scores = evaluate_latent_recovery(latent, latent)
        self.assertAlmostEqual(scores["procrustes_r2"], 1.0, places=10)
        self.assertAlmostEqual(scores["rsa_spearman"], 1.0, places=10)
        self.assertAlmostEqual(scores["rsa_pearson"], 1.0, places=10)
        self.assertAlmostEqual(scores["linear_cka"], 1.0, places=10)

    def test_procrustes_handles_rotation(self):
        theta = np.pi / 4
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        X = np.random.default_rng(1).normal(size=(50, 2))
        Y = X @ R
        aligned, _ = procrustes_align(Y, X)
        self.assertEqual(aligned.shape, X.shape)
        self.assertGreater(procrustes_r2(Y, X), 0.95)

    def test_noise_and_distorted_geometry_lower_recovery(self):
        latent = self._latent(seed=2)
        noisy = latent + 0.75 * self._latent(seed=3)
        distorted = latent.copy()
        distorted[:, 0] = distorted[:, 0] ** 3
        identity = evaluate_latent_recovery(latent, latent)
        noisy_scores = evaluate_latent_recovery(noisy, latent)
        distorted_scores = evaluate_latent_recovery(distorted, latent)
        self.assertLess(noisy_scores["procrustes_r2"], identity["procrustes_r2"])
        self.assertLess(distorted_scores["rsa_spearman"], identity["rsa_spearman"])

    def test_trial_safe_shuffle_lowers_recovery(self):
        latent = self._latent(seed=4, n=60)
        trial_id = np.repeat(np.arange(6), 10)
        rng = np.random.default_rng(5)
        shuffled_trials = rng.permutation(np.unique(trial_id))
        shuffled = latent.copy()
        for old, new in zip(np.unique(trial_id), shuffled_trials):
            shuffled[trial_id == old] = latent[trial_id == new]
        self.assertLess(
            procrustes_r2(shuffled, latent),
            procrustes_r2(latent, latent) - 0.1,
        )
        self.assertLess(
            distance_geometry_correlation(shuffled, latent),
            distance_geometry_correlation(latent, latent),
        )

    def test_linear_cka_allows_different_feature_counts(self):
        latent = self._latent(seed=6, n=50, d=3)
        expanded = np.column_stack([latent, latent[:, :2] ** 2])
        score = linear_cka(expanded, latent)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0 + 1e-12)

    def test_linear_cka_invariances_and_constant_input(self):
        X = self._latent(seed=23)
        Q, _ = np.linalg.qr(self._latent(seed=24, n=3))
        self.assertAlmostEqual(linear_cka(X, 3.0 * X @ Q + 7.0), 1.0, places=10)
        self.assertTrue(np.isnan(linear_cka(X, np.ones_like(X))))

    def test_linear_cka_matches_centered_gram_definition(self):
        X = self._latent(seed=25, n=30)
        Y = self._latent(seed=26, n=30, d=5)
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        K, L = X @ X.T, Y @ Y.T
        expected = np.sum(K * L) / (np.linalg.norm(K) * np.linalg.norm(L))
        self.assertAlmostEqual(linear_cka(X, Y), expected, places=12)

    def test_sampled_rsa_preserves_requested_metric(self):
        latent = self._latent(seed=7, n=120, d=3)
        transformed = latent.copy()
        transformed[:, 0] *= 10.0
        euclidean = distance_geometry_correlation(
            transformed, latent, metric="euclidean", max_pairs=40, random_state=9
        )
        cityblock = distance_geometry_correlation(
            transformed, latent, metric="cityblock", max_pairs=40, random_state=9
        )
        self.assertTrue(np.isfinite(euclidean))
        self.assertTrue(np.isfinite(cityblock))

    def test_shape_mismatch_is_rejected(self):
        latent = self._latent(seed=8, n=20)
        with self.assertRaises(ValueError):
            procrustes_r2(latent[:-1], latent)

    def test_distance_geometry_correlation(self):
        X = self._latent(seed=10, n=30)
        corr = distance_geometry_correlation(X, X)
        self.assertGreater(corr, 0.99)

    def test_evaluate_latent_recovery_keys(self):
        X = self._latent(seed=11, n=30)
        scores = evaluate_latent_recovery(X, X)
        self.assertIn("procrustes_r2", scores)
        self.assertIn("rsa_spearman", scores)
        self.assertIn("rsa_pearson", scores)
        self.assertIn("linear_cka", scores)

    def test_lagged_alignment_scores(self):
        X = self._latent(seed=12, n=40)
        Y = np.vstack([X[:1], X[:-1]])
        best_lag, scores = lagged_alignment_scores(X, Y, lags=range(-2, 3))
        self.assertIn(best_lag, scores)

    def test_participation_ratio_uses_stable_precision_for_float32_embeddings(self):
        rng = np.random.default_rng(42)
        dominant = rng.normal(size=23_400).astype(np.float32)
        embedding = np.column_stack([
            dominant,
            dominant * np.float32(1e-4),
            rng.normal(scale=1e-7, size=len(dominant)).astype(np.float32),
        ])
        actual = _participation_ratio(embedding)
        expected = _participation_ratio(embedding.astype(np.float64))
        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
