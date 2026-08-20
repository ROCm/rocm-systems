# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/pattern_matching.py."""

import pytest

H3 = "nn.Module.Net.forward/torch.nn.functional.relu/torch.relu"

# -- fnmatch_glob_matches ---------------------------------------------------


@pytest.mark.torch_ops
def test_glob_helper_matches_target():
    """``fnmatch_glob_matches`` performs case-sensitive fnmatch globbing."""
    from utils.pattern_matching import fnmatch_glob_matches

    assert fnmatch_glob_matches("*torch.relu", H3)
    assert fnmatch_glob_matches("*relu", H3)
    assert not fnmatch_glob_matches("sigmoid", H3)
