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


def default_schema_path():
    """Locate the documented input-file schema relative to this file in the source tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(
            here, "..", "..", "..", "source", "docs", "rocprofv3_input_schema.json"
        )
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
def input_file(jobs, file_options=None, suffix=".json"):
    """Write `jobs` (plus any top-level `file_options`) out as a rocprofv3 JSON or YAML input file,
    or yield None when there are none."""
    if jobs is None:
        yield None
        return
    document = dict(file_options or {}, jobs=jobs)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as ofs:
        if suffix == ".json":
            json.dump(document, ofs)
        else:
            import yaml

            yaml.safe_dump(document, ofs)
    try:
        yield path
    finally:
        os.remove(path)


def launched_runs(*argv, jobs=None, file_options=None, suffix=".json"):
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
        with input_file(jobs, file_options, suffix) as path:
            argv = list(argv) + (["-i", path] if path else []) + ["--", "/bin/true"]
            module.main(argv)
    finally:
        module.run = original
    return runs


def test_each_input_file_job_is_a_run_of_its_own():
    """An input file's jobs are independent configurations, so replay must not fold them
    together and leave the later ones' settings behind."""
    runs = launched_runs(
        "--replay-mode",
        "kernel",
        "--kernel-replay-beta-enabled",
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

    assert shape(
        launched_runs(
            "--replay-mode", "kernel", "--kernel-replay-beta-enabled", jobs=jobs
        )
    ) == shape(launched_runs(jobs=jobs))


def test_jobs_that_disagree_are_all_run():
    """Jobs are free to differ in whatever they like. None of these differences may cost a job
    its run or its settings."""
    runs = launched_runs(
        "--replay-mode",
        "kernel",
        "--kernel-replay-beta-enabled",
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
    runs = launched_runs(
        "--replay-mode", "kernel", "--kernel-replay-beta-enabled", jobs=jobs
    )
    assert [itr.pmc for itr in runs] == [job["pmc"] for job in jobs]


def test_command_line_groups_are_passes_of_one_run():
    """Replay's reason for existing: several groups collected without re-running the
    application."""
    runs = launched_runs(
        "--pmc",
        "SQ_WAVES",
        "--pmc",
        "GRBM_COUNT",
        "--replay-mode",
        "kernel",
        "--kernel-replay-beta-enabled",
    )
    assert len(runs) == 1
    assert runs[0].pmc == [["SQ_WAVES"], ["GRBM_COUNT"]]


def test_command_line_groups_are_separate_runs_without_replay():
    # Confirms the previous test passes because of the flag, not because of the shape of --pmc.
    runs = launched_runs("--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT")
    assert [itr.pmc for itr in runs] == [["SQ_WAVES"], ["GRBM_COUNT"]]


def test_one_command_line_group_is_one_run():
    runs = launched_runs(
        "--pmc",
        "SQ_WAVES",
        "GRBM_COUNT",
        "--replay-mode",
        "kernel",
        "--kernel-replay-beta-enabled",
    )
    assert len(runs) == 1
    assert runs[0].pmc == ["SQ_WAVES", "GRBM_COUNT"]


def test_command_line_groups_stay_one_run_alongside_input_file_jobs():
    runs = launched_runs(
        "--pmc",
        "A",
        "--pmc",
        "B",
        "--replay-mode",
        "kernel",
        "--kernel-replay-beta-enabled",
        jobs=[{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}],
    )
    assert [itr.pmc for itr in runs] == [[["A"], ["B"]], ["SQ_WAVES"], ["GRBM_COUNT"]]


def test_input_file_pmc_groups_are_counter_collection():
    """pmc_groups never arrives on the command line, so a check that only looks at cmd_args
    would reject this input file for having no counters."""
    runs = launched_runs(
        "--replay-mode",
        "kernel",
        "--kernel-replay-beta-enabled",
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


def test_att_is_a_replay_pass_not_a_conflict():
    # Replay gives the dispatch thread trace a pass of its own ahead of the counter passes.
    assert service_conflicts(advanced_thread_trace=True, pmc=["SQ_WAVES"]) == []


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
    ) == ["PC sampling", "--spm"]


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
        ) == ["PC sampling", "--spm"]


def test_missing_attributes_are_treated_as_unset():
    """Not every caller builds a fully populated namespace.

    The helper reads options with getattr defaults, so an args object that never had these
    attributes must behave like one where they are off, rather than raising.
    """
    assert service_conflicts() == []


def test_att_alone_does_not_drag_in_other_services():
    assert service_conflicts(advanced_thread_trace=True, spm=None) == []


def att_conflicts(**attrs):
    return rocprofv3().att_options_conflicting_with_kernel_replay(
        rocprofv3().dotdict(attrs)
    )


def test_dispatch_att_has_no_replay_option_conflict():
    # The plain dispatch thread trace is what replay knows how to give a pass of its own.
    assert att_conflicts(advanced_thread_trace=True, att_target_cu="1") == []


def test_device_mode_att_options_conflict_with_replay():
    # Each of these traces the device rather than one dispatch, so it would capture every pass.
    for attr, name in (
        ("att_consecutive_kernels", "--att-consecutive-kernels"),
        ("att_no_intercept", "--att-no-intercept"),
        ("selected_regions", "--selected-regions"),
        ("collection_period", "--collection-period"),
    ):
        value = ["1:1:1"] if attr == "collection_period" else True
        assert att_conflicts(advanced_thread_trace=True, **{attr: value}) == [name], attr


def test_att_option_conflicts_need_att():
    """Without --att these options belong to other services (counters honor --selected-regions,
    for instance), and replay has no thread trace pass for them to escape."""
    assert (
        att_conflicts(
            advanced_thread_trace=False,
            selected_regions=True,
            collection_period=["1:1:1"],
            att_consecutive_kernels="4",
        )
        == []
    )


def test_att_option_conflicts_are_reported_together_in_a_stable_order():
    for _ in range(5):
        assert att_conflicts(
            collection_period=["1:1:1"],
            selected_regions=True,
            att_no_intercept=True,
            att_consecutive_kernels="4",
            advanced_thread_trace=True,
        ) == [
            "--att-consecutive-kernels",
            "--att-no-intercept",
            "--selected-regions",
            "--collection-period",
        ]


def test_unset_att_options_do_not_conflict():
    # argparse leaves these as None/False rather than absent; none of them may look enabled.
    assert (
        att_conflicts(
            advanced_thread_trace=True,
            att_consecutive_kernels=None,
            att_no_intercept=False,
            selected_regions=False,
            collection_period=None,
        )
        == []
    )


def test_counter_collection_is_never_itself_a_conflict():
    """Counter collection is the one pass-aware service, so no shape of it may be rejected --
    including the list-of-lists form that replay itself builds."""
    assert service_conflicts(pmc=[["SQ_WAVES"], ["GRBM_COUNT"]]) == []
    assert service_conflicts(pmc_groups=[["SQ_WAVES"], ["GRBM_COUNT"]]) == []


# job_replay_mode: kernel -- the jobs of an input file become the passes of one run.

KERNEL_JOBS = {"job_replay_mode": "kernel"}


def job_replay_runs(*argv, jobs, suffix=".json"):
    return launched_runs(
        "--kernel-replay-beta-enabled",
        *argv,
        jobs=jobs,
        file_options=KERNEL_JOBS,
        suffix=suffix,
    )


def rejected(*argv, jobs, file_options=KERNEL_JOBS, beta=True):
    """True when rocprofv3 refuses the command line and input file before launching anything."""
    argv = (("--kernel-replay-beta-enabled",) if beta else ()) + argv
    try:
        launched_runs(*argv, jobs=jobs, file_options=file_options)
    except SystemExit as exc:
        return exc.code != 0
    return False


def test_job_replay_runs_every_job_as_a_pass_of_one_run():
    runs = job_replay_runs(
        jobs=[
            {"advanced_thread_trace": True, "att_target_cu": 1},
            {"pmc": ["SQ_WAVES", "GRBM_COUNT"]},
            {"pmc": ["SQ_INSTS_VALU"]},
        ]
    )
    assert len(runs) == 1
    run = runs[0]
    assert run.replay_mode == "kernel"
    assert run.pmc == [["SQ_WAVES", "GRBM_COUNT"], ["SQ_INSTS_VALU"]]
    assert run.advanced_thread_trace is True
    assert run.att_target_cu == 1
    assert run.kernel_replay_att_after_groups == 0
    # One run writes one output directory, not a pass_N directory per job.
    assert run.sub_directory is None


def test_job_order_places_the_thread_trace_pass():
    att = {"advanced_thread_trace": True}
    a, b = {"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}
    for jobs, after in (([att, a, b], 0), ([a, att, b], 1), ([a, b, att], 2)):
        (run,) = job_replay_runs(jobs=jobs)
        assert run.kernel_replay_att_after_groups == after, jobs
        assert run.pmc == [["SQ_WAVES"], ["GRBM_COUNT"]], jobs


def test_counter_only_job_replay():
    (run,) = job_replay_runs(jobs=[{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}])
    assert run.pmc == [["SQ_WAVES"], ["GRBM_COUNT"]]
    assert not run.advanced_thread_trace
    assert run.kernel_replay_att_after_groups is None


def test_yaml_input_opts_in_the_same_way():
    try:
        import yaml  # noqa: F401
    except ImportError:
        return
    (run,) = job_replay_runs(
        jobs=[{"pmc": ["SQ_WAVES"]}, {"advanced_thread_trace": True}], suffix=".yaml"
    )
    # A lone counter group is flattened to its counter list, as it is on the command line.
    assert run.pmc == ["SQ_WAVES"]
    assert run.kernel_replay_att_after_groups == 1


def test_run_options_from_any_job_apply_to_the_run():
    (run,) = job_replay_runs(
        jobs=[
            {"pmc": ["SQ_WAVES"], "kernel_include_regex": "gemm"},
            {
                "pmc": ["GRBM_COUNT"],
                "output_format": ["json"],
                "kernel_include_regex": "gemm",
            },
        ]
    )
    assert run.kernel_include_regex == "gemm"
    assert run.output_format == ["json"]


def test_the_beta_acknowledgement_may_come_from_a_job():
    (run,) = launched_runs(
        jobs=[
            {"pmc": ["SQ_WAVES"], "kernel_replay_beta_enabled": True},
            {"pmc": ["GRBM_COUNT"]},
        ],
        file_options=KERNEL_JOBS,
    )
    assert run.kernel_replay_beta_enabled is True


def test_thread_trace_decoder_path_may_sit_on_any_job():
    # att_library_path only says where the decoder is, so it is not tied to the thread trace job.
    (run,) = job_replay_runs(
        jobs=[
            {"pmc": ["SQ_WAVES"], "att_library_path": ["/opt/x"]},
            {"advanced_thread_trace": True},
        ]
    )
    assert run.att_library_path == ["/opt/x"]


def test_job_replay_rejects_jobs_that_disagree_on_a_run_option():
    assert rejected(
        jobs=[
            {"pmc": ["SQ_WAVES"], "output_directory": "/tmp/a"},
            {"pmc": ["GRBM_COUNT"], "output_directory": "/tmp/b"},
        ]
    )


def test_job_replay_rejects_a_job_with_both_counters_and_thread_trace():
    assert rejected(jobs=[{"pmc": ["SQ_WAVES"], "advanced_thread_trace": True}])


def test_job_replay_rejects_a_job_that_collects_nothing():
    assert rejected(jobs=[{"pmc": ["SQ_WAVES"]}, {"kernel_trace": True}])


def test_job_replay_rejects_two_thread_trace_jobs():
    assert rejected(
        jobs=[
            {"advanced_thread_trace": True},
            {"pmc": ["SQ_WAVES"]},
            {"advanced_thread_trace": True},
        ]
    )


def test_job_replay_rejects_pmc_groups():
    assert rejected(jobs=[{"pmc_groups": [["SQ_WAVES"], ["GRBM_COUNT"]]}])


def test_job_replay_rejects_thread_trace_settings_on_a_counter_job():
    assert rejected(
        jobs=[{"pmc": ["SQ_WAVES"], "att_target_cu": 2}, {"advanced_thread_trace": True}]
    )


def test_job_replay_needs_a_counter_job():
    assert rejected(jobs=[{"advanced_thread_trace": True}])


def test_job_replay_rejects_pass_services_on_the_command_line():
    jobs = [{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}]
    assert rejected("--pmc", "SQ_INSTS_VALU", jobs=jobs)
    assert rejected("--att", jobs=jobs)


def test_job_replay_rejects_application_replay():
    jobs = [{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}]
    assert rejected("--replay-mode", "application", jobs=jobs)
    assert rejected(jobs=[{"pmc": ["SQ_WAVES"], "replay_mode": "application"}])


def test_job_replay_still_needs_the_beta_acknowledgement():
    assert rejected(jobs=[{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}], beta=False)


def test_job_replay_rejects_attach_mode_and_collection_periods():
    jobs = [{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}]
    assert rejected("--pid", "1234", jobs=jobs)
    assert rejected("--collection-period", "1:1:1", jobs=jobs)


def test_unknown_job_replay_mode_is_rejected():
    assert rejected(
        jobs=[{"pmc": ["SQ_WAVES"]}], file_options={"job_replay_mode": "dispatch"}
    )


def test_job_replay_mode_application_is_the_default():
    """Spelling out the default changes nothing: each job is still a run of its own."""
    jobs = [{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}]
    explicit = launched_runs(jobs=jobs, file_options={"job_replay_mode": "application"})
    assert [itr.pmc for itr in explicit] == [itr.pmc for itr in launched_runs(jobs=jobs)]
    assert [itr.pmc for itr in explicit] == [["SQ_WAVES"], ["GRBM_COUNT"]]


def test_schema_documents_the_job_replay_modes():
    """The input-file schema is documentation only -- rocprofv3 does not validate against it -- so
    nothing else would notice it drifting from what rocprofv3 accepts."""
    path = os.environ.get("ROCPROFV3_INPUT_SCHEMA", default_schema_path())
    with open(path, "r") as ifs:
        schema = json.load(ifs)
    mode = schema["properties"]["job_replay_mode"]
    assert mode["enum"] == list(rocprofv3().JOB_REPLAY_MODES)
    assert mode["default"] == "application"
    job_options = schema["properties"]["jobs"]["items"]["properties"]
    for option in ("pmc", "advanced_thread_trace", "kernel_replay_beta_enabled"):
        assert option in job_options, option


def test_files_without_job_replay_mode_keep_running_jobs_as_runs_under_replay():
    """The opt-in is the file's: --replay-mode kernel alone must not fold jobs into passes."""
    runs = launched_runs(
        "--replay-mode",
        "kernel",
        "--kernel-replay-beta-enabled",
        jobs=[{"pmc": ["SQ_WAVES"]}, {"pmc": ["GRBM_COUNT"]}],
    )
    assert [itr.pmc for itr in runs] == [["SQ_WAVES"], ["GRBM_COUNT"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        default=os.environ.get("ROCPROFV3_SCRIPT", default_script_path()),
        help="path to rocprofv3.py",
    )
    parser.add_argument(
        "--schema",
        default=os.environ.get("ROCPROFV3_INPUT_SCHEMA", default_schema_path()),
        help="path to rocprofv3_input_schema.json",
    )
    args = parser.parse_args()
    os.environ["ROCPROFV3_SCRIPT"] = args.script
    os.environ["ROCPROFV3_INPUT_SCHEMA"] = args.schema

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
