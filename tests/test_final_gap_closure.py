"""Focused numerical tests for frozen-embedding gap closure."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import final_gap_closure as boot
import final_gap_temporal_null as temporal
import final_gap_c8 as c8
import final_gap_bootstrap_completion as completion
from neurobridge.eval.representation import procrustes_r2, linear_cka
from sklearn.metrics import r2_score


def test_procrustes_trial_sufficient_statistics_match_direct():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(80,3));y=rng.normal(size=(80,3))
    trial=np.repeat(np.arange(8),10)
    stats=boot.trial_stats(x,y,trial,np.arange(8))
    np.testing.assert_allclose(boot.procrustes_from_stats(stats,np.ones((1,8)))[0],
                               procrustes_r2(x,y),atol=1e-12)
    count=np.array([[2,0,1,0,1,2,1,1]])
    ix=np.concatenate([np.flatnonzero(trial==j) for j,n in enumerate(count[0]) for _ in range(n)])
    np.testing.assert_allclose(boot.procrustes_from_stats(stats,count)[0],
                               procrustes_r2(x[ix],y[ix]),atol=1e-12)


def test_temporal_null_identity_matches_direct():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(2,3,10,3));y=rng.normal(size=x.shape)
    identity=np.tile(np.arange(3),(2,1))[None]
    np.testing.assert_allclose(temporal.null_scores(x.reshape(-1,3),y.reshape(-1,3),identity)[0],
                               procrustes_r2(x.reshape(-1,3),y.reshape(-1,3)),atol=1e-12)


def test_c8_exact_condition_orbit_and_group_closure():
    theta=np.arange(8)*np.pi/4
    orbit=np.c_[np.cos(theta),np.sin(theta),np.zeros(8)]
    score,one_step,closure=c8.evaluate(orbit,orbit)
    np.testing.assert_allclose([score,one_step],[1,1],atol=1e-12)
    assert closure<1e-12


def test_continuous_r2_sufficient_statistics_match_direct():
    rng=np.random.default_rng(11)
    target=rng.normal(size=(80,2));prediction=rng.normal(size=(80,2))
    trial=np.repeat(np.arange(8),10)
    stats=completion.r2_stats(target,prediction,trial,np.arange(8))
    np.testing.assert_allclose(completion.r2_scores(stats,np.ones((1,8)))[0],
                               r2_score(target,prediction,multioutput="variance_weighted"),atol=1e-12)


def test_linear_cka_sufficient_statistics_match_direct():
    rng=np.random.default_rng(12)
    x=rng.normal(size=(80,3));y=rng.normal(size=(80,3));trial=np.repeat(np.arange(8),10)
    stats=completion.cka_stats(x,y,trial,np.arange(8))
    np.testing.assert_allclose(completion.cka_scores(stats,np.ones((1,8)))[0],linear_cka(x,y),atol=1e-12)
