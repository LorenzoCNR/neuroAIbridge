import numpy as np
import pytest

from neurobridge.experiments.frozen_sensitivity import _evaluation_mask, _evaluation_scope


def test_branch_scope_and_masks_keep_full_sample_descriptive():
    masks = {
        "train": np.array([True, True, False, False, False, False]),
        "validation": np.array([False, False, True, True, False, False]),
        "test": np.array([False, False, False, False, True, True]),
    }
    valid = np.array([False, True, True, False, True, True])

    np.testing.assert_array_equal(
        _evaluation_mask(masks, valid, "held_out"),
        np.array([False, False, False, False, True, True]),
    )
    np.testing.assert_array_equal(
        _evaluation_mask(masks, valid, "full_sample"),
        valid,
    )
    assert _evaluation_scope("held_out") == "held_out_representation_generalization"
    assert _evaluation_scope("full_sample") == "descriptive_in_sample_representation"
    with pytest.raises(ValueError, match="branch"):
        _evaluation_scope("unknown")
