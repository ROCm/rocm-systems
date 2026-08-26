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


def load_rocprofv3(script_path):
    """Import rocprofv3.py as a module. Its top level is guarded by __main__, so this is safe."""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"rocprofv3 script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("rocprofv3_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
