import numpy as np
import pytest

from neurobridge.viz.manifold_plots import plot_direction_averaged_embedding


def test_direction_averaged_plot_uses_trial_time_and_validity_metadata(tmp_path):
    # Unequal and sparse trial supports intentionally cannot be represented by
    # one hard-coded points-per-trial reshape.
    embedding = np.array([
        [1.0, 0.0, 0.0],  # trial 0, condition 1, time 0
        [0.0, 1.0, 0.0],  # trial 0, condition 1, time 2
        [1.0, 0.0, 0.0],  # trial 1, condition 1, time 0
        [0.0, 0.0, 1.0],  # trial 1, condition 1, time 2 (invalid)
        [0.0, 0.0, 1.0],  # trial 0, condition 2, time 1
        [1.0, 1.0, 0.0],  # trial 0, condition 2, time 4
    ])
    labels = np.array([1, 1, 1, 1, 2, 2])
    trial_id = np.array([0, 0, 1, 1, 0, 0])
    time_id = np.array([0, 2, 0, 2, 1, 4])
    valid_mask = np.array([True, True, True, False, True, True])

    fig = plot_direction_averaged_embedding(
        embedding,
        labels,
        original_label_order=[1, 2],
        c_s="maroon",
        output_folder=str(tmp_path),
        name="metadata_trajectories.html",
        show=False,
        trial_id=trial_id,
        time_id=time_id,
        valid_mask=valid_mask,
    )

    trajectories = [trace for trace in fig.data if getattr(trace, "mode", None) == "lines"]
    assert len(trajectories) == 2
    assert all(len(trace.x) == 2 for trace in trajectories)
    np.testing.assert_allclose(trajectories[0].x, [1.0, 0.0])
    np.testing.assert_allclose(trajectories[0].y, [0.0, 1.0])
    assert (tmp_path / "metadata_trajectories.html").is_file()


def test_direction_averaged_plot_rejects_misaligned_metadata(tmp_path):
    with pytest.raises(ValueError, match="trial_id and time_id must match"):
        plot_direction_averaged_embedding(
            np.zeros((3, 3)),
            np.ones(3),
            original_label_order=[1],
            c_s="maroon",
            output_folder=str(tmp_path),
            name="bad.html",
            show=False,
            trial_id=np.arange(2),
            time_id=np.arange(3),
            valid_mask=np.ones(3, dtype=bool),
        )
