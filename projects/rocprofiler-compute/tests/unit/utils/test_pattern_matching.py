# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/pattern_matching.py."""

from utils.pattern_matching import fnmatch_glob_matches

H3 = "nn.Module.Net.forward/torch.nn.functional.relu/torch.relu"
H2 = "nn.Module.Net.forward/torch.nn.functional.conv2d"
H1 = "torch.relu"


# -- fnmatch_glob_matches ---------------------------------------------------


def test_glob_helper_matches_target():
    """fnmatch_glob_matches performs case-sensitive fnmatch globbing."""
    assert fnmatch_glob_matches("*torch.relu", H3)
    assert fnmatch_glob_matches("*relu", H3)
    assert not fnmatch_glob_matches("sigmoid", H3)


def test_all_keyword():
    """'all' maps to '**' and matches every hierarchy."""
    assert fnmatch_glob_matches("all", H3)
    assert fnmatch_glob_matches("all", H2)
    assert fnmatch_glob_matches("all", H1)
    assert not fnmatch_glob_matches("all", "")


def test_bare_pattern_requires_exact_match():
    """A pattern without wildcards matches only when equal to the full target."""
    assert fnmatch_glob_matches("torch.relu", H1)
    assert not fnmatch_glob_matches("torch.relu", H3)
    assert not fnmatch_glob_matches("torch.nn.functional.conv2d", H2)
    assert not fnmatch_glob_matches("relu", H3)
    assert not fnmatch_glob_matches("forward", H3)
    assert not fnmatch_glob_matches("sigmoid", H3)


def test_substring_wildcard_pattern():
    """'*' matches any run of characters, including '/'."""
    assert fnmatch_glob_matches("*torch.relu", H3)
    assert fnmatch_glob_matches("*torch.*", H3)
    assert fnmatch_glob_matches("*relu", H3)
    assert fnmatch_glob_matches("*conv*", H2)
    assert fnmatch_glob_matches("*relu*", H3)
    assert not fnmatch_glob_matches("conv*", H2)
    assert not fnmatch_glob_matches("sigm*", H3)


def test_hierarchy_glob():
    """Patterns with '/' match across multiple hierarchy components."""
    assert fnmatch_glob_matches("nn.Module.Net.forward/*/torch.relu", H3)
    assert fnmatch_glob_matches("*/torch.nn.functional.conv2d", H2)
    assert not fnmatch_glob_matches("nn.Module.Net.forward/torch.relu", H3)


def test_leading_slash_is_cosmetic():
    """A leading '/' in the pattern is stripped before matching."""
    assert fnmatch_glob_matches("/nn.Module.Net.forward/*/torch.relu", H3)
    assert fnmatch_glob_matches("/torch.relu", H1)


def test_trailing_slash_is_cosmetic():
    """A trailing '/' in the pattern is stripped before matching."""
    assert not fnmatch_glob_matches("nn.Module.Net.forward/", H3)
    assert fnmatch_glob_matches("torch.relu/", H1)


def test_regex_not_supported():
    """Regex syntax has no special meaning; treated as literal glob text."""
    assert not fnmatch_glob_matches("relu|conv2d", H3)
    assert not fnmatch_glob_matches("^torch\\.relu$", H3)
    assert not fnmatch_glob_matches("not:relu", H3)
    assert not fnmatch_glob_matches("2:functional", H3)


def test_empty_inputs():
    """Empty pattern or operator_name returns False."""
    assert not fnmatch_glob_matches("", H3)
    assert not fnmatch_glob_matches("relu", "")
    assert not fnmatch_glob_matches("", "")


def test_slash_only_markers():
    """Scope-marker-only tokens should not match any hierarchy."""
    assert not fnmatch_glob_matches("/", H3)
    assert not fnmatch_glob_matches("//", H3)


def test_double_star_explicit():
    """'**' matches any hierarchy depth."""
    assert fnmatch_glob_matches("**", H3)
    assert fnmatch_glob_matches("**", H2)
    assert fnmatch_glob_matches("**", H1)
    assert fnmatch_glob_matches("**", "a/b/c/d/e")
    assert not fnmatch_glob_matches("**", "")


def test_single_char_wildcard():
    """'?' matches exactly one character."""
    assert fnmatch_glob_matches("*torch.rel?", H3)
    assert fnmatch_glob_matches("*torch.?elu", H3)
    assert not fnmatch_glob_matches("torch.?", H3)
    assert not fnmatch_glob_matches("?", H1)
    assert fnmatch_glob_matches("*torch.nn.functional.conv?d", H2)


def test_long_hierarchy():
    """Patterns apply to deeply nested hierarchies."""
    deep = "/".join([f"level{i}" for i in range(20)])
    assert fnmatch_glob_matches("*level19", deep)
    assert fnmatch_glob_matches("*19", deep)
    assert fnmatch_glob_matches("*/level19", deep)
    assert fnmatch_glob_matches("all", deep)
    assert not fnmatch_glob_matches("level0", deep)
    assert fnmatch_glob_matches("*level0*", deep)


def test_long_component_names():
    """Patterns apply to components with long names."""
    long_name = "a" * 500
    hierarchy = f"root/{long_name}"
    assert fnmatch_glob_matches(f"*{long_name}", hierarchy)
    assert fnmatch_glob_matches("*a*", hierarchy)
    assert not fnmatch_glob_matches("a*", hierarchy)
    assert not fnmatch_glob_matches("b*", hierarchy)


def test_special_characters_in_names():
    """Dots and underscores are treated literally."""
    h = "nn.Module._internal/torch.nn.functional.conv2d"
    assert fnmatch_glob_matches("*torch.nn.functional.conv2d", h)
    assert fnmatch_glob_matches("*conv2d", h)
    assert fnmatch_glob_matches("nn.Module._internal/*", h)
    assert not fnmatch_glob_matches("nn_Module._internal/*", h)


def test_bracket_glob_pattern():
    """Character classes '[abc]' are supported."""
    assert fnmatch_glob_matches("*torch.rel[uv]", H3)
    assert not fnmatch_glob_matches("*torch.rel[ab]", H3)


def test_single_component_hierarchy():
    """A single-component target matches an equal bare pattern."""
    assert fnmatch_glob_matches("torch.relu", "torch.relu")
    assert fnmatch_glob_matches("*relu", "torch.relu")
    assert fnmatch_glob_matches("torch.*", "torch.relu")
    assert not fnmatch_glob_matches("*/torch.relu", "torch.relu")


def test_whitespace_only_pattern():
    """Whitespace-only patterns normalize to empty and return False."""
    assert not fnmatch_glob_matches("   ", H3)
    assert not fnmatch_glob_matches("\t", H3)


def test_star_pattern_matches_all():
    """'*' matches any non-empty target."""
    assert fnmatch_glob_matches("*", H3)
    assert fnmatch_glob_matches("*", H2)
    assert fnmatch_glob_matches("*", H1)
    assert fnmatch_glob_matches("*", "a/b/c/d/e")
    assert not fnmatch_glob_matches("*", "")


def test_star_normalize_equivalence():
    """'all' and '*' both match any non-empty target."""
    assert fnmatch_glob_matches("all", H3)
    assert fnmatch_glob_matches("*", H3)


def test_case_sensitivity():
    """Pattern matching is case-sensitive."""
    assert not fnmatch_glob_matches("Torch.Relu", H3)
    assert not fnmatch_glob_matches("TORCH.RELU", H3)
    assert not fnmatch_glob_matches("ALL", H3)
    assert fnmatch_glob_matches("all", H3)


def test_all_keyword_case_sensitive():
    """The 'all' alias is case-sensitive; other casings are literal."""
    assert fnmatch_glob_matches("all", H3)
    assert not fnmatch_glob_matches("ALL", H3)
    assert not fnmatch_glob_matches("All", H3)


def test_consecutive_slashes_in_target():
    """Consecutive slashes in the target are treated literally."""
    h = "a//b///torch.relu"
    assert fnmatch_glob_matches("*torch.relu", h)
    assert fnmatch_glob_matches("*relu", h)


def test_dots_in_patterns():
    """Dots are treated literally, not as regex wildcards."""
    assert fnmatch_glob_matches("*torch.relu", H3)
    assert not fnmatch_glob_matches("*torchXrelu", H3)
    h = "root/torchXrelu"
    assert not fnmatch_glob_matches("*torch.relu", h)
    assert fnmatch_glob_matches("*torchXrelu", h)


def test_pattern_with_spaces():
    """Spaces are treated literally."""
    h = "module/ spaced op /torch.relu"
    assert fnmatch_glob_matches("*torch.relu", h)
    assert not fnmatch_glob_matches(" spaced op ", h)
    assert fnmatch_glob_matches("* spaced op */*", h)


def test_colons_in_operator_names():
    """Colons are treated literally."""
    h = "nn.Module/aten::relu_"
    assert fnmatch_glob_matches("*aten::relu_", h)
    assert fnmatch_glob_matches("*relu_", h)
    assert fnmatch_glob_matches("*aten::*", h)
    assert not fnmatch_glob_matches("*relu", h)
    assert not fnmatch_glob_matches("*torch.relu", h)
