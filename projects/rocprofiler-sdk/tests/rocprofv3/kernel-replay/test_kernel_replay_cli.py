#!/usr/bin/env python3
#
# Unit tests for the rocprofv3 kernel-replay CLI logic.
#
# These exercise rocprofv3.py's argument handling directly and need neither a GPU nor a built
# rocprofiler-sdk, so they run anywhere. They cover the two decisions replay makes on the command
# line: which services cannot be collected in the same run, and how counter groups and input-file
# jobs turn into application runs.

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import tempfile

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


@contextlib.contextmanager
def input_file(jobs):
    """Write `jobs` out as a rocprofv3 JSON input file, or yield None when there are none."""
    if jobs is None:
        yield None
        return
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as ofs:
        json.dump({"jobs": jobs}, ofs)
    try:
        yield path
    finally:
        os.remove(path)


def launched_runs(*argv, jobs=None):
    """Report the application runs rocprofv3 would start for the given command line.

    `run` is replaced for the duration, so nothing is executed and no environment is touched.
    Returns one settings object per run, in the order they would be launched.
    """
    module = rocprofv3()
    runs = []

    def record(app_args, args, **kwargs):
        runs.append(args)
        return 0

    original = module.run
    module.run = record
    try:
        with input_file(jobs) as path:
            argv = list(argv) + (["-i", path] if path else []) + ["--", "/bin/true"]
            module.main(argv)
    finally:
        module.run = original
    return runs


def test_each_input_file_job_is_a_run_of_its_own():
    """An input file's jobs are independent configurations, so replay must not fold them
    together and leave the later ones' settings behind."""
    runs = launched_runs(
        "--replay-mode", "kernel", "--kernel-replay-beta-enabled",
        jobs=[
            {"pmc": ["SQ_WAVES"], "output_directory": "/tmp/a"},
            {"pmc": ["GRBM_COUNT"], "output_directory": "/tmp/b"},
        ],
    )
    assert [itr.pmc for itr in runs] == [["SQ_WAVES"], ["GRBM_COUNT"]]
    assert [itr.output_directory for itr in runs] == ["/tmp/a", "/tmp/b"]


def test_replay_does_not_change_how_input_file_jobs_are_run():
    """The flag selects how a run collects its counter groups; it has no say over what the jobs
    in an input file are."""
    jobs = [
        {
            "pmc": ["SQ_WAVES"],
            "output_directory": "/tmp/a",
            "kernel_include_regex": "gemm.*",
        },
        {
            "pmc": ["GRBM_COUNT"],
            "output_directory": "/tmp/b",
            "kernel_include_regex": "conv.*",
        },
    ]
    keys = ("pmc", "output_directory", "kernel_include_regex")

    def shape(runs):
        return [tuple(getattr(itr, key) for key in keys) for itr in runs]

    assert shape(launched_runs("--replay-mode", "kernel", "--kernel-replay-beta-enabled", jobs=jobs)) == shape(
        launched_runs(jobs=jobs)
    )


def test_jobs_that_disagree_are_all_run():
    """Jobs are free to differ in whatever they like. None of these differences may cost a job
    its run or its settings."""
    runs = launched_runs(
        "--replay-mode", "kernel", "--kernel-replay-beta-enabled",
        jobs=[
            {"pmc": ["A"], "output_format": ["csv"], "kernel_iteration_range": "1-2"},
            {"pmc": ["B"], "output_format": ["json"], "kernel_iteration_range": "3-4"},
        ],
    )
    assert len(runs) == 2
    assert [itr.kernel_iteration_range for itr in runs] == [["1-2"], ["3-4"]]
    assert [itr.output_format for itr in runs] == [["csv"], ["json"]]


def test_many_jobs_all_run():
    """Input files with one job per counter group are the normal way to reach replay, and a
    realistic counter list runs to dozens of groups."""
    jobs = [{"pmc": [f"COUNTER_{idx}"]} for idx in range(32)]
    runs = launched_runs("--replay-mode", "kernel", "--kernel-replay-beta-enabled", jobs=jobs)
    assert [itr.pmc for itr in runs] == [job["pmc"] for job in jobs]


def test_command_line_groups_are_passes_of_one_run():
    """Replay's reason for existing: several groups collected without re-running the
    application."""
    runs = launched_runs(
        "--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT", "--replay-mode", "kernel", "--kernel-replay-beta-enabled"
    )
    assert len(runs) == 1
    assert runs[0].pmc == [["SQ_WAVES"], ["GRBM_COUNT"]]


def test_command_line_groups_are_separate_runs_without_replay():
    # Confirms the previous test passes because of the flag, not because of the shape of --pmc.
    runs = launched_runs("--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT")
    assert [itr.pmc for itr in runs] == [["SQ_WAVES"], ["GRBM_COUNT"]]


def test_one_command_line_group_is_one_run():
    runs = launched_runs(
        "--pmc", "SQ_WAVES", "GRBM_COUNT", "--replay-mode", "kernel", "--kernel-replay-beta-enabled"
    )
    assert len(runs) == 1
    assert runs[0].pmc == ["SQ_WAVES", "GRBM_COUNT"]


def test_command_line_groups_stay_one_run_alongside_input_file_jobs():
    runs = launched_runs(
        "--pmc",
        "A",
        "--pmc",
        "B",
        "--replay-mode", "kernel", "--kernel-replay-beta-enabled",
        jobs=[{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}],
    )
    assert [itr.pmc for itr in runs] == [[["A"], ["B"]], ["SQ_WAVES"], ["GRBM_COUNT"]]


def test_input_file_pmc_groups_are_counter_collection():
    """pmc_groups never arrives on the command line, so a check that only looks at cmd_args
    would reject this input file for having no counters."""
    runs = launched_runs(
        "--replay-mode", "kernel", "--kernel-replay-beta-enabled",
        jobs=[{"pmc_groups": [["SQ_WAVES"], ["GRBM_COUNT"]]}],
    )
    assert len(runs) == 1
    assert runs[0].pmc_groups == [["SQ_WAVES"], ["GRBM_COUNT"]]


def test_replay_without_beta_acknowledgement_is_rejected():
    try:
        launched_runs("--pmc", "SQ_WAVES", "--replay-mode", "kernel")
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError(
            "--replay-mode kernel without --kernel-replay-beta-enabled should have been rejected"
        )


def test_replay_without_counter_collection_is_rejected():
    try:
        launched_runs("--replay-mode", "kernel", "--kernel-replay-beta-enabled")
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError(
            "replay without counter collection should have been rejected"
        )


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
