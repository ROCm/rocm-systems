# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Torch operator pattern matching (PurePosixPath glob)
# ---------------------------------------------------------------------------

H3 = "nn.Module.Net.forward/torch.nn.functional.relu/torch.relu"
H2 = "nn.Module.Net.forward/torch.nn.functional.conv2d"
H1 = "torch.relu"


@pytest.mark.torch_ops
def test_all_keyword():
    """'all' maps to '**' and matches every hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("all", H3)
    assert m("all", H2)
    assert m("all", H1)
    assert not m("all", "")


@pytest.mark.torch_ops
def test_bare_pattern_matches_last_component():
    """Bare token is matched via PurePosixPath.match() against the full hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", H3)
    assert m("torch.nn.functional.conv2d", H2)
    assert not m("relu", H3)
    assert not m("forward", H3)
    assert not m("sigmoid", H3)


@pytest.mark.torch_ops
def test_bare_wildcard_pattern():
    """Wildcard bare token matched via PurePosixPath.match()."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.*", H3)
    assert m("*relu", H3)
    assert m("*conv*", H2)
    assert not m("conv*", H2)
    assert not m("sigm*", H3)


@pytest.mark.torch_ops
def test_hierarchy_glob():
    """Patterns with '/' match across multiple hierarchy components."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("nn.Module.Net.forward/*/torch.relu", H3)
    assert m("*/torch.nn.functional.conv2d", H2)
    assert not m("nn.Module.Net.forward/torch.relu", H3)


@pytest.mark.torch_ops
def test_leading_slash_is_cosmetic():
    """Leading '/' is stripped during pattern normalization."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("/nn.Module.Net.forward/*/torch.relu", H3)
    assert m("/torch.relu", H3)


@pytest.mark.torch_ops
def test_trailing_slash_stripped_by_posixpath():
    """PurePosixPath strips trailing slashes, so they are cosmetic."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("nn.Module.Net.forward/", H3)
    assert m("torch.relu/", H3)


@pytest.mark.torch_ops
def test_regex_not_supported():
    """Regex syntax has no special meaning; treated as literal glob text."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("relu|conv2d", H3)
    assert not m("^torch\\.relu$", H3)
    assert not m("not:relu", H3)
    assert not m("2:functional", H3)


@pytest.mark.torch_ops
def test_empty_inputs():
    """Empty pattern or operator_name returns False."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("", H3)
    assert not m("relu", "")
    assert not m("", "")


@pytest.mark.torch_ops
def test_slash_only_markers():
    """Scope-marker-only tokens should not match any hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("/", H3)
    assert not m("//", H3)


# -- get_matched_torch_operators_for_display ---------------------------------


def get_matched_torch_operators_for_display(
    torch_operators: dict[str, pd.DataFrame],
    pattern_list: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    """Return (operator_name, filtered_df) for each operator matching any pattern.

    Test-only helper: iterates every unique Operator_Name across all torch trace
    DataFrames and checks each against the supplied glob patterns.
    """
    from utils.parser import torch_operator_pattern_matches

    if not torch_operators or not pattern_list:
        return []
    result: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for _, df in torch_operators.items():
        if df is None or df.empty or "Operator_Name" not in df.columns:
            continue
        for op_name in df["Operator_Name"].dropna().unique():
            op_str = str(op_name).strip()
            if op_str in seen:
                continue
            for pattern in pattern_list:
                if torch_operator_pattern_matches(pattern.strip(), op_str):
                    seen.add(op_str)
                    result.append((op_str, df.loc[df["Operator_Name"] == op_name]))
                    break
    return result


@pytest.mark.torch_ops
def test_display_match_hierarchy_glob():
    """Full hierarchy globs are honored by display helper."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H3, H2],
        "Kernel_Name": ["k1", "k2", "k3"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(torch_operators, ["*/torch.relu"])
    assert len(matched) == 1
    assert matched[0][0] == H3


@pytest.mark.torch_ops
def test_display_match_multi_patterns():
    """Multiple glob patterns match their respective operators."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H2],
        "Kernel_Name": ["k1", "k2"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(
        torch_operators, ["*relu", "*conv*"]
    )
    assert len(matched) == 2


@pytest.mark.torch_ops
def test_display_no_match():
    """No matches returns empty list."""
    df = pd.DataFrame({
        "Operator_Name": [H3],
        "Kernel_Name": ["k1"],
    })
    assert get_matched_torch_operators_for_display({"t": df}, ["sigmoid"]) == []


@pytest.mark.torch_ops
def test_display_empty_inputs():
    """Empty torch_operators or pattern_list returns []."""
    assert get_matched_torch_operators_for_display({}, ["relu"]) == []
    assert get_matched_torch_operators_for_display({"x": pd.DataFrame()}, []) == []


# -- parse_torch_operator_patterns ------------------------------------------


@pytest.mark.torch_ops
def test_parse_patterns_basic():
    """Single and multiple patterns are parsed correctly."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["relu"])
    assert parse_torch_operator_patterns(args) == ["relu"]

    args = Namespace(torch_operator=["relu", "conv2d"])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_comma_split():
    """Comma-separated patterns in a single arg are split."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["relu,conv2d"])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_whitespace():
    """Leading/trailing whitespace is stripped."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["  relu  ", " conv2d , linear "])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d", "linear"]


@pytest.mark.torch_ops
def test_parse_patterns_empty():
    """Flag given with no args defaults to '**'; absent flag returns empty."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    assert parse_torch_operator_patterns(Namespace(torch_operator=[])) == ["**"]
    assert parse_torch_operator_patterns(Namespace(torch_operator=None)) == []
    assert parse_torch_operator_patterns(Namespace()) == []


# -- PatternMatcherEngine ---------------------------------------------------


@pytest.mark.torch_ops
def test_engine_glob_hierarchy_mode():
    """Facade delegates matching to glob-hierarchy implementation."""
    from utils.pattern_matching import PatternMatcherEngine

    matcher = PatternMatcherEngine(mode="glob-hierarchy")
    assert matcher.matches("torch.relu", H3)
    assert matcher.matches("*relu", H3)
    assert not matcher.matches("sigmoid", H3)


@pytest.mark.torch_ops
def test_engine_invalid_mode():
    """Unsupported strategy names should raise ValueError."""
    from utils.pattern_matching import PatternMatcherEngine

    with pytest.raises(ValueError):
        PatternMatcherEngine(mode="regex")


# -- Additional coverage (xuchen #26) ----------------------------------------


@pytest.mark.torch_ops
def test_double_star_explicit():
    """'**' matches any hierarchy depth."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("**", H3)
    assert m("**", H2)
    assert m("**", H1)
    assert m("**", "a/b/c/d/e")
    assert not m("**", "")


@pytest.mark.torch_ops
def test_single_char_wildcard():
    """'?' matches exactly one character in a component."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.rel?", H3)
    assert m("torch.?elu", H3)
    assert not m("torch.?", H3)
    assert not m("?", H1)
    assert m("torch.nn.functional.conv?d", H2)


@pytest.mark.torch_ops
def test_long_hierarchy():
    """Deeply nested hierarchies match correctly."""
    from utils.parser import torch_operator_pattern_matches as m

    deep = "/".join([f"level{i}" for i in range(20)])
    assert m("level19", deep)
    assert m("*19", deep)
    assert m("*/level19", deep)
    assert m("all", deep)
    assert not m("level0", deep)


@pytest.mark.torch_ops
def test_long_component_names():
    """Components with very long names are handled correctly."""
    from utils.parser import torch_operator_pattern_matches as m

    long_name = "a" * 500
    hierarchy = f"root/{long_name}"
    assert m(f"{'a' * 500}", hierarchy)
    assert m("a*", hierarchy)
    assert not m("b*", hierarchy)


@pytest.mark.torch_ops
def test_special_characters_in_names():
    """Dots, underscores, and other non-glob chars are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "nn.Module._internal/torch.nn.functional.conv2d"
    assert m("torch.nn.functional.conv2d", h)
    assert m("*conv2d", h)
    assert m("nn.Module._internal/*", h)
    assert not m("nn_Module._internal/*", h)


@pytest.mark.torch_ops
def test_bracket_glob_pattern():
    """Character classes [abc] work in glob patterns."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.rel[uv]", H3)
    assert not m("torch.rel[ab]", H3)


@pytest.mark.torch_ops
def test_single_component_hierarchy():
    """Single-component hierarchy (no slashes) matches bare patterns."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", "torch.relu")
    assert m("*relu", "torch.relu")
    assert m("torch.*", "torch.relu")
    assert not m("*/torch.relu", "torch.relu")


@pytest.mark.torch_ops
def test_whitespace_only_pattern():
    """Whitespace-only patterns normalize to empty and return False."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("   ", H3)
    assert not m("\t", H3)


@pytest.mark.torch_ops
def test_star_pattern_matches_all():
    """Bare '*' is normalized to '**' and matches every hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("*", H3)
    assert m("*", H2)
    assert m("*", H1)
    assert m("*", "a/b/c/d/e")
    assert not m("*", "")


@pytest.mark.torch_ops
def test_star_normalize_equivalence():
    """'*' and 'all' produce the same normalization."""
    from utils.pattern_matching import PurePosixGlobHierarchyMatcher

    norm = PurePosixGlobHierarchyMatcher.normalize_pattern
    assert norm("*") == norm("all") == "**"


@pytest.mark.torch_ops
def test_case_sensitivity():
    """Pattern matching is case-sensitive."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("Torch.Relu", H3)
    assert not m("TORCH.RELU", H3)
    assert not m("ALL", H3)
    assert m("all", H3)


@pytest.mark.torch_ops
def test_all_keyword_case_sensitive():
    """Only lowercase 'all' is the special keyword; mixed case is a literal."""
    from utils.pattern_matching import PurePosixGlobHierarchyMatcher

    norm = PurePosixGlobHierarchyMatcher.normalize_pattern
    assert norm("all") == "**"
    assert norm("ALL") == "ALL"
    assert norm("All") == "All"


@pytest.mark.torch_ops
def test_consecutive_slashes_in_target():
    """Consecutive slashes in the target are collapsed by PurePosixPath."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "a//b///torch.relu"
    assert m("torch.relu", h)
    assert m("*relu", h)


@pytest.mark.torch_ops
def test_dots_in_patterns():
    """Dots are literal characters in glob patterns, not regex wildcards."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", H3)
    assert not m("torchXrelu", H3)
    h = "root/torchXrelu"
    assert not m("torch.relu", h)
    assert m("torchXrelu", h)


@pytest.mark.torch_ops
def test_pattern_with_spaces():
    """Spaces in patterns and targets are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "module/ spaced op /torch.relu"
    assert m("torch.relu", h)
    assert not m(" spaced op ", h)
    assert m("* spaced op */*", h)


@pytest.mark.torch_ops
def test_colons_in_operator_names():
    """Colons (e.g. aten::relu) are literal characters in glob matching."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "nn.Module/aten::relu_"
    assert m("aten::relu_", h)
    assert m("*relu_", h)
    assert m("aten::*", h)
    assert not m("*relu", h)
    assert not m("torch.relu", h)


@pytest.mark.torch_ops
def test_display_star_matches_all_operators():
    """'*' pattern matches all operators in display helper."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H2],
        "Kernel_Name": ["k1", "k2"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(torch_operators, ["*"])
    assert len(matched) == 2


@pytest.mark.torch_ops
def test_display_dedup_across_dataframes():
    """Same operator in multiple DataFrames is matched only once."""
    df1 = pd.DataFrame({"Operator_Name": [H3], "Kernel_Name": ["k1"]})
    df2 = pd.DataFrame({"Operator_Name": [H3], "Kernel_Name": ["k2"]})
    torch_operators = {"trace_0": df1, "trace_1": df2}

    matched = get_matched_torch_operators_for_display(torch_operators, ["all"])
    op_names = [name for name, _ in matched]
    assert op_names.count(H3) == 1


@pytest.mark.torch_ops
def test_parse_patterns_star():
    """'*' is passed through as-is by the pattern parser."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["*"])
    assert parse_torch_operator_patterns(args) == ["*"]

    args = Namespace(torch_operator=["*,torch.relu"])
    assert parse_torch_operator_patterns(args) == ["*", "torch.relu"]

