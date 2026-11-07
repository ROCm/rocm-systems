#!/usr/bin/env python3
#
# rocprofv3 --replay-mode accepts kernel and range. These check the range choice the same way the
# kernel replay CLI tests check kernel: nothing is executed.

import contextlib
import importlib.util
import json
import os
import sys
import tempfile

from importlib.machinery import SourceFileLoader


def load_rocprofv3(script_path):
    loader = SourceFileLoader("rocprofv3_range_under_test", script_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def default_script_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "..", "source", "bin", "rocprofv3.py")
    )


_MODULE = None


def rocprofv3():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_rocprofv3(os.environ.get("ROCPROFV3_SCRIPT", default_script_path()))
    return _MODULE


@contextlib.contextmanager
def input_file(jobs):
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


def rejected(*argv):
    try:
        launched_runs(*argv)
    except SystemExit as exc:
        return exc.code not in (0, None)
    return False


def test_range_groups_are_passes_of_one_run():
    runs = launched_runs(
        "--pmc",
        "SQ_WAVES",
        "--pmc",
        "GRBM_COUNT",
        "--replay-mode",
        "range",
        "--range-replay-beta-enabled",
    )
    assert len(runs) == 1
    assert runs[0].pmc == [["SQ_WAVES"], ["GRBM_COUNT"]]
    assert runs[0].replay_mode == "range"


def test_kernel_groups_are_still_passes_of_one_run():
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
    assert runs[0].replay_mode == "kernel"
    assert runs[0].pmc == [["SQ_WAVES"], ["GRBM_COUNT"]]


def test_application_replay_is_one_run_per_group():
    runs = launched_runs("--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT")
    assert [itr.pmc for itr in runs] == [["SQ_WAVES"], ["GRBM_COUNT"]]


def test_host_flag_is_carried_on_the_range_run():
    runs = launched_runs(
        "--pmc",
        "SQ_WAVES",
        "--pmc",
        "GRBM_COUNT",
        "--replay-mode",
        "range",
        "--range-replay-beta-enabled",
        "--range-replay-host",
    )
    assert len(runs) == 1
    assert runs[0].range_replay_host is True


def test_range_without_beta_acknowledgement_is_rejected():
    assert rejected("--pmc", "SQ_WAVES", "--replay-mode", "range")


def test_range_without_counters_is_rejected():
    assert rejected("--replay-mode", "range", "--range-replay-beta-enabled")


def test_host_flag_without_range_mode_is_rejected():
    assert rejected(
        "--pmc", "SQ_WAVES", "--replay-mode", "kernel",
        "--kernel-replay-beta-enabled", "--range-replay-host",
    )


def test_range_rejects_the_services_kernel_replay_rejects():
    # run() resolves the tool library before it checks replay conflicts, so give it a root that
    # contains the libraries. The conflict then exits before the application starts.
    with tempfile.TemporaryDirectory() as root:
        for rel in (
            "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so",
            "lib/librocprofiler-sdk.so",
            "lib/librocprofiler-sdk-roctx.so",
            "lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so",
            "lib/rocprofiler-sdk/librocprofv3-list-avail.so",
        ):
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").close()
        try:
            rocprofv3().main(
                [
                    "--rocm-root",
                    root,
                    "--pmc",
                    "SQ_WAVES",
                    "--replay-mode",
                    "range",
                    "--range-replay-beta-enabled",
                    "--att",
                    "--",
                    "/bin/true",
                ]
            )
        except SystemExit as exc:
            assert exc.code not in (0, None)
        else:
            raise AssertionError("--replay-mode range accepted --att")


def main():
    if "--script" in sys.argv:
        os.environ["ROCPROFV3_SCRIPT"] = sys.argv[sys.argv.index("--script") + 1]
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"PASS {test.__name__}")
    if failed:
        print(f"{failed} failed", file=sys.stderr)
        return 1
    print(f"{len(tests)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
