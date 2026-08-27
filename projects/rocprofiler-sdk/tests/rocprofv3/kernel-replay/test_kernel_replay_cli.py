#!/usr/bin/env python3
#
# Unit tests for the rocprofv3 kernel-replay CLI logic.
#
# These exercise rocprofv3.py's argument handling directly and need neither a GPU nor a built
# rocprofiler-sdk, so they run anywhere. They cover the input-file collapse that kernel replay
# performs: replay produces a single application run, so the counter groups from every job in an
# input file are merged and only one job can supply the remaining settings. Silently keeping the
# first job would discard the others' output configuration, kernel filters and ranges.

import argparse
import importlib.util
import os
import sys

from importlib.machinery import SourceFileLoader


def load_rocprofv3(script_path):
    """Import rocprofv3.py as a module. Its top level is guarded by __main__, so this is safe."""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"rocprofv3 script not found: {script_path}")
    # Installed launcher is named "rocprofv3" (no .py), so load via explicit loader.
    loader = SourceFileLoader("rocprofv3_under_test", script_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def default_script_path():
    """Locate rocprofv3.py relative to this file in the source tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "..", "source", "bin", "rocprofv3.py")
    )


_MODULE = None


def rocprofv3():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_rocprofv3(
            os.environ.get("ROCPROFV3_SCRIPT", default_script_path())
        )
    return _MODULE


def jobs(*dicts):
    return [rocprofv3().dotdict(d) for d in dicts]


def conflicts(*dicts, ignore=("pmc",)):
    return rocprofv3().conflicting_input_settings(jobs(*dicts), ignore=ignore)


def test_no_jobs_have_no_conflicts():
    assert rocprofv3().conflicting_input_settings([], ignore=("pmc",)) == []


def test_single_job_never_conflicts():
    assert conflicts({"pmc": ["SQ_WAVES"], "output_directory": "/a"}) == []


def test_jobs_differing_only_in_pmc_collapse_cleanly():
    # The normal kernel-replay shape: one job per counter group, everything else identical.
    # pmc is merged by the caller, so it must not be reported.
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"], "output_directory": "/a"},
            {"pmc": ["GRBM_COUNT"], "output_directory": "/a"},
        )
        == []
    )


def test_differing_setting_is_reported():
    # The reported bug: the second job's output_directory was dropped without a word.
    assert conflicts(
        {"pmc": ["SQ_WAVES"], "output_directory": "/a"},
        {"pmc": ["GRBM_COUNT"], "output_directory": "/b"},
    ) == ["output_directory"]


def test_setting_present_only_in_later_job_is_reported():
    # Keeping the first job would drop this entirely, which is just as silent a loss.
    assert conflicts(
        {"pmc": ["SQ_WAVES"]},
        {"pmc": ["GRBM_COUNT"], "kernel_include_regex": "foo.*"},
    ) == ["kernel_include_regex"]


def test_setting_present_only_in_first_job_is_kept():
    # The first job's settings survive the collapse, so nothing is lost and nothing is reported.
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"], "kernel_include_regex": "foo.*"},
            {"pmc": ["GRBM_COUNT"]},
        )
        == []
    )


def test_explicit_none_counts_as_unset():
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"], "output_directory": "/a"},
            {"pmc": ["GRBM_COUNT"], "output_directory": None},
        )
        == []
    )


def test_multiple_conflicts_are_sorted_and_deduplicated():
    assert conflicts(
        {"pmc": ["A"], "output_directory": "/a", "output_format": "csv"},
        {"pmc": ["B"], "output_directory": "/b", "output_format": "json"},
        {"pmc": ["C"], "output_directory": "/c", "kernel_iteration_range": "1-2"},
    ) == ["kernel_iteration_range", "output_directory", "output_format"]


def test_list_values_compare_by_value():
    assert (
        conflicts(
            {"pmc": ["A"], "runtime_trace": ["hip", "hsa"]},
            {"pmc": ["B"], "runtime_trace": ["hip", "hsa"]},
        )
        == []
    )
    assert conflicts(
        {"pmc": ["A"], "runtime_trace": ["hip"]},
        {"pmc": ["B"], "runtime_trace": ["hip", "hsa"]},
    ) == ["runtime_trace"]


def test_ignore_list_is_honored():
    # Without the caller's ignore, pmc itself is a conflict. This guards against the ignore
    # argument silently losing effect.
    assert conflicts({"pmc": ["A"]}, {"pmc": ["B"]}, ignore=()) == ["pmc"]


def test_sub_directory_added_by_the_parser_is_not_a_conflict():
    # parse_yaml / parse_json stamp the same sub_directory onto every job, so it must never
    # trip the check on its own.
    assert (
        conflicts(
            {"pmc": ["A"], "sub_directory": "pass_"},
            {"pmc": ["B"], "sub_directory": "pass_"},
        )
        == []
    )


def test_empty_later_job_is_skipped():
    assert conflicts({"pmc": ["A"], "output_directory": "/a"}, {}) == []


def service_conflicts(environ=None, **attrs):
    return rocprofv3().services_conflicting_with_kernel_replay(
        rocprofv3().dotdict(attrs), environ={} if environ is None else environ
    )


def test_counters_only_replay_has_no_service_conflict():
    # The supported combination. Nothing here may start reporting a conflict.
    assert service_conflicts(pmc=["SQ_WAVES"]) == []


def test_att_conflicts_with_replay():
    # ATT instruments every pass and reports them all under the dispatch id replay reuses.
    assert service_conflicts(advanced_thread_trace=True) == ["--att"]


def test_pc_sampling_flag_conflicts_with_replay():
    assert service_conflicts(pc_sampling_beta_enabled=True) == ["PC sampling"]


def test_pc_sampling_env_conflicts_with_replay():
    # The beta gate can be opened by environment instead of by flag; both reach the same service.
    assert service_conflicts(environ={"ROCPROFILER_PC_SAMPLING_BETA_ENABLED": "1"}) == [
        "PC sampling"
    ]


def test_spm_conflicts_with_replay():
    # The existing SPM check only looks at --pmc, so counter groups from an input file (the shape
    # kernel replay uses) would otherwise slip past it.
    assert service_conflicts(spm=["SQ_WAVES"]) == ["--spm"]


def test_every_conflicting_service_is_reported_together():
    # A user who asked for all of them should be told about all of them, not one per run.
    assert service_conflicts(
        advanced_thread_trace=True, pc_sampling_beta_enabled=True, spm=["SQ_WAVES"]
    ) == ["--att", "PC sampling", "--spm"]


def test_unset_service_options_do_not_conflict():
    # argparse leaves these as None/False rather than absent; none of them may look enabled.
    assert (
        service_conflicts(
            advanced_thread_trace=False,
            pc_sampling_beta_enabled=False,
            spm=None,
            pmc=["SQ_WAVES"],
        )
        == []
    )


def test_pc_sampling_env_set_to_zero_still_conflicts():
    """The gate is presence, not truthiness.

    rocprofv3 opens the PC sampling beta on the variable being set at all, so a user who exported
    it as 0 still gets the service. If this check tested truthiness instead, that user would be
    allowed into a replay run that silently collects N times the PC samples they asked for.
    """
    assert service_conflicts(environ={"ROCPROFILER_PC_SAMPLING_BETA_ENABLED": "0"}) == [
        "PC sampling"
    ]


def test_pc_sampling_env_set_to_empty_still_conflicts():
    assert service_conflicts(environ={"ROCPROFILER_PC_SAMPLING_BETA_ENABLED": ""}) == [
        "PC sampling"
    ]


def test_pc_sampling_flag_and_env_are_reported_once():
    # Both routes reach the same service; naming it twice would read like two separate problems.
    assert service_conflicts(
        pc_sampling_beta_enabled=True,
        environ={"ROCPROFILER_PC_SAMPLING_BETA_ENABLED": "1"},
    ) == ["PC sampling"]


def test_unrelated_environment_does_not_conflict():
    assert (
        service_conflicts(
            environ={"ROCPROFILER_SOMETHING_ELSE": "1", "PATH": "/usr/bin"},
            pmc=["SQ_WAVES"],
        )
        == []
    )


def test_empty_spm_list_does_not_conflict():
    # An empty list means the option was not given a value; only a real request should conflict.
    assert service_conflicts(spm=[]) == []


def test_conflicts_are_reported_in_a_stable_order():
    """The message joins these with " or ", so a varying order makes the same mistake produce
    different text on different runs and defeats matching in tests and docs."""
    for _ in range(5):
        assert service_conflicts(
            spm=["SQ_WAVES"], advanced_thread_trace=True, pc_sampling_beta_enabled=True
        ) == ["--att", "PC sampling", "--spm"]


def test_missing_attributes_are_treated_as_unset():
    """Not every caller builds a fully populated namespace.

    The helper reads options with getattr defaults, so an args object that never had these
    attributes must behave like one where they are off, rather than raising.
    """
    assert service_conflicts() == []


def test_att_alone_does_not_drag_in_other_services():
    assert service_conflicts(advanced_thread_trace=True, spm=None) == ["--att"]


def test_counter_collection_is_never_itself_a_conflict():
    """Counter collection is the one pass-aware service, so no shape of it may be rejected --
    including the list-of-lists form that replay itself builds."""
    assert service_conflicts(pmc=[["SQ_WAVES"], ["GRBM_COUNT"]]) == []
    assert service_conflicts(pmc_groups=[["SQ_WAVES"], ["GRBM_COUNT"]]) == []


# ---------------------------------------------------------------------------
# Input-file collapse: further edge cases
#
# Replay merges every job's counter groups into one application run, so the jobs must agree on
# everything except pmc. These cover the shapes real input files take.
# ---------------------------------------------------------------------------


def test_three_jobs_agreeing_collapse_cleanly():
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"], "output_format": "csv"},
            {"pmc": ["GRBM_COUNT"], "output_format": "csv"},
            {"pmc": ["SQ_INSTS"], "output_format": "csv"},
        )
        == []
    )


def test_conflict_between_the_first_and_last_job_is_found():
    # A disagreement must be found wherever it is, not only between adjacent jobs.
    assert "output_format" in conflicts(
        {"pmc": ["SQ_WAVES"], "output_format": "csv"},
        {"pmc": ["GRBM_COUNT"], "output_format": "csv"},
        {"pmc": ["SQ_INSTS"], "output_format": "json"},
    )


def test_kernel_filters_that_differ_are_reported():
    """Silently keeping the first job's filter would profile a different set of kernels than the
    input file asked for."""
    assert "kernel_include_regex" in conflicts(
        {"pmc": ["SQ_WAVES"], "kernel_include_regex": "gemm.*"},
        {"pmc": ["GRBM_COUNT"], "kernel_include_regex": "conv.*"},
    )


def test_matching_kernel_filters_are_not_reported():
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"], "kernel_include_regex": "gemm.*"},
            {"pmc": ["GRBM_COUNT"], "kernel_include_regex": "gemm.*"},
        )
        == []
    )


def test_lists_differing_only_in_order_are_a_conflict():
    """Order is meaningful in the settings this compares (ranges, filters), so two orderings are
    not interchangeable and picking one would change what the run does."""
    assert "kernel_iteration_range" in conflicts(
        {"pmc": ["SQ_WAVES"], "kernel_iteration_range": [1, 2]},
        {"pmc": ["GRBM_COUNT"], "kernel_iteration_range": [2, 1]},
    )


def test_nested_list_values_compare_by_value():
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"], "extra": [["a"], ["b"]]},
            {"pmc": ["GRBM_COUNT"], "extra": [["a"], ["b"]]},
        )
        == []
    )


def test_zero_and_false_are_not_treated_as_unset():
    """Only an explicit None means unset. A job that asked for 0 disagrees with one that asked
    for 1, and treating 0 as absent would let that difference through.

    The falsey value has to sit in the *later* job. conflicting_input_settings only skips a key
    on the later job's own has_set_attr check; the first job's value reaching None just makes
    the two differ, which still reports a conflict. Put the 0 in the first job instead and the
    assertion holds whether has_set_attr tests `is not None` or truthiness, guarding nothing.
    """
    assert "some_count" in conflicts(
        {"pmc": ["SQ_WAVES"], "some_count": 1},
        {"pmc": ["GRBM_COUNT"], "some_count": 0},
    )
    assert "some_flag" in conflicts(
        {"pmc": ["SQ_WAVES"], "some_flag": True},
        {"pmc": ["GRBM_COUNT"], "some_flag": False},
    )


def test_a_setting_equal_to_none_in_both_jobs_is_not_a_conflict():
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"], "output_format": None},
            {"pmc": ["GRBM_COUNT"], "output_format": None},
        )
        == []
    )


def test_ignoring_more_than_one_setting_is_honored():
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"], "output_format": "csv", "d": "one"},
            {"pmc": ["GRBM_COUNT"], "output_format": "json", "d": "two"},
            ignore=("pmc", "output_format", "d"),
        )
        == []
    )


def test_every_differing_setting_is_named_not_just_the_first():
    """A user fixing an input file should learn about all of the disagreements at once."""
    reported = conflicts(
        {"pmc": ["SQ_WAVES"], "output_format": "csv", "output_directory": "a"},
        {"pmc": ["GRBM_COUNT"], "output_format": "json", "output_directory": "b"},
    )
    assert "output_format" in reported
    assert "output_directory" in reported


def test_pmc_is_ignored_by_default_because_replay_merges_it():
    """pmc is the one setting jobs are supposed to differ in: replay merges the groups and derives
    its pass count from them. Reporting it would reject every valid multi-group input file.
    """
    assert (
        conflicts(
            {"pmc": ["SQ_WAVES"]},
            {"pmc": ["GRBM_COUNT"]},
        )
        == []
    )


def test_pmc_becomes_a_conflict_when_not_ignored():
    # Confirms the previous test passes because pmc is ignored, not because it compares equal.
    assert "pmc" in conflicts(
        {"pmc": ["SQ_WAVES"]},
        {"pmc": ["GRBM_COUNT"]},
        ignore=(),
    )


def test_many_jobs_are_handled():
    """Input files with one job per counter group are the normal way to reach replay, and a
    realistic counter list runs to dozens of groups."""
    agreeing = [{"pmc": [f"COUNTER_{i}"], "output_format": "csv"} for i in range(32)]
    assert conflicts(*agreeing) == []

    disagreeing = list(agreeing)
    disagreeing[-1] = {"pmc": ["COUNTER_31"], "output_format": "json"}
    assert "output_format" in conflicts(*disagreeing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        default=os.environ.get("ROCPROFV3_SCRIPT", default_script_path()),
        help="path to rocprofv3.py",
    )
    args = parser.parse_args()
    os.environ["ROCPROFV3_SCRIPT"] = args.script

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, func in tests:
        try:
            func()
        except (
            Exception
        ) as exc:  # noqa: BLE001 - report every failure, not just the first
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
