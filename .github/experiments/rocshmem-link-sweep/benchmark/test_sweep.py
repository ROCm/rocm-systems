"""Portable unit/orchestration tests; no ROCm compilation or timing is performed."""
from contextlib import contextmanager, nullcontext
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rocshmem_sweep", HERE / "sweep.py")
sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)


@contextmanager
def scratch():
    # tempfile's Windows mode-0700 ACL excludes the sandbox principal on this
    # host. Inherit the workspace ACL, and remove only this owned UUID path.
    directory = HERE / ("test-scratch-" + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == HERE and directory.name.startswith("test-scratch-")
        shutil.rmtree(directory)


def command(width=1, target="tools/rocshmem_info"):
    return ["clang++", "info.o", "librocshmem.a", "-o", target, "--hip-link",
            *(f"--offload-arch={arch}" for arch in sweep.ARCHITECTURES),
            *([] if width == 1 else [f"--offload-jobs={width}"])]


class Guards(unittest.TestCase):
    def test_all_widths_only_change_one_option(self):
        original = sweep.canonical_link(command(), 1, "tools/rocshmem_info")
        for width in sweep.WIDTHS:
            self.assertEqual(sweep.canonical_link(command(width), width, "tools/rocshmem_info"), original)

    def test_architecture_duplicates_or_drops_rejected(self):
        for argv in (command()[:-1], command() + ["--offload-arch=gfx900"]):
            with self.assertRaisesRegex(ValueError, "31 architecture"):
                sweep.canonical_link(argv, 1, "tools/rocshmem_info")

    def test_conflicting_width_rejected(self):
        for argv in (command(4), command(2) + ["--offload-jobs=4"]):
            with self.assertRaisesRegex(ValueError, "width-2"):
                sweep.canonical_link(argv, 2, "tools/rocshmem_info")

    def test_profiling_env_does_not_leak_value(self):
        for name in ("CC_PRINT_PROC_STAT", "CC_PRINT_OPTIONS", "CCC_OVERRIDE_OPTIONS", "MAKEFLAGS", "LD_PRELOAD"):
            with self.assertRaises(ValueError) as error:
                sweep.guard_environment({name: "DO_NOT_PRINT_SECRET"})
            self.assertIn(name, str(error.exception))
            self.assertNotIn("DO_NOT_PRINT_SECRET", str(error.exception))
        sweep.guard_environment({"CC_PRINT_PROC_STAT": "", "HIP_CLANG_NUM_PARALLEL_JOBS": "1"})

    def test_serializing_flags_and_response_files_rejected(self):
        for args in (["-v"], ["-fproc-stat-report=stats.csv"], ["-###"],
                     ["--driver-mode", "cl"], ["-save-temps"], ["@link.rsp"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                sweep.guard_argv(command() + args)

    def test_shell_side_effect_and_compile_work_rejected(self):
        parsed = sweep.parse_link_command(": && " + shlex.join(command()) + " && :")
        self.assertEqual(parsed, command())
        with self.assertRaises(ValueError):
            sweep.parse_link_command(shlex.join(command()) + " && unexpected-action")
        sweep.guard_dry_run("ninja: Entering directory build\n[1/1] Linking CXX executable tools/rocshmem_info\n")
        for text in ("ninja: no work to do", "[1/2] Building CXX object x\n[2/2] Linking CXX executable y"):
            with self.assertRaises(ValueError):
                sweep.guard_dry_run(text)


class MemoryAndSelection(unittest.TestCase):
    def test_proc_parser_handles_spaces_and_parentheses(self):
        fields = ["S", "10", "20"] + ["0"] * 19
        fields[19], fields[21] = "1234", "25"
        proc = sweep.parse_proc_stat(30, "30 (name with ) parens) " + " ".join(fields), 4096)
        self.assertEqual((proc.ppid, proc.pgrp, proc.started, proc.rss_kib), (10, 20, 1234, 100))

    def test_group_descendants_and_pid_reuse(self):
        records = {1: sweep.Proc(1, 0, 1, 10, 100, "ninja"),
                   2: sweep.Proc(2, 1, 8, 20, 200, "clang"),
                   3: sweep.Proc(3, 2, 8, 30, 300, "lld"),
                   4: sweep.Proc(4, 0, 9, 999, 1000, "unrelated")}
        members = sweep.group_members(records, 1, {4: 100})
        self.assertEqual({p.pid for p in members}, {1, 2, 3})
        self.assertEqual(sum(p.rss_kib for p in members), 600)

    def test_budget_honors_host_cgroup_and_user_limit(self):
        cg = {"memory.max": str(1000 * 1024), "memory.current": str(400 * 1024)}
        self.assertEqual(sweep.memory_budget_kib({"MemAvailable_kib": 900}, cg, 700, 100), 500)
        self.assertEqual(sweep.memory_budget_kib({"MemAvailable_kib": 50}, {}, 700, 100), 0)
        self.assertEqual(sweep.cgroup_remaining_kib(cg), 600)
        self.assertIsNone(sweep.cgroup_remaining_kib({"memory.max": "max"}))

    def test_monitor_stops_owned_group_at_rss_or_dynamic_cgroup_floor(self):
        healthy = {"memory.max": str(1000 * 1024), "memory.current": str(100 * 1024)}
        tight = {"memory.max": str(1000 * 1024), "memory.current": str(980 * 1024)}
        usage = SimpleNamespace(ru_utime=0.0, ru_stime=0.0)
        resource = SimpleNamespace(RUSAGE_CHILDREN=1, RUSAGE_SELF=2,
                                   getrusage=lambda _: usage)
        for limit, expected in [(5, "summed-rss-limit"), (1000, "cgroup-memory-floor")]:
            with self.subTest(limit=limit), scratch() as directory:
                process = Mock(pid=100, returncode=None)
                process.poll.return_value = None
                def stop(_):
                    process.returncode = -15
                with patch.dict(sys.modules, {"resource": resource}), \
                     patch.object(sweep.subprocess, "Popen", return_value=process) as popen, \
                     patch.object(sweep, "read_proc_snapshot", return_value=(
                         {100: sweep.Proc(100, 1, 100, 1, 10, "ninja")}, 0)), \
                     patch.object(sweep, "meminfo", return_value={"MemAvailable_kib": 10000}), \
                     patch.object(sweep, "cgroup_context", side_effect=[healthy, tight, tight]), \
                     patch.object(sweep, "stop_group", side_effect=stop) as terminate:
                    result = sweep.timed_link(["ninja"], {}, directory, 1, 0, limit, 50)
                self.assertEqual(result["stop_reason"], expected)
                self.assertEqual(result["memory"]["cgroup_memory_remaining_min_kib"], 20)
                self.assertEqual(result["memory"]["peak_summed_rss_kib"], 10)
                terminate.assert_called_once_with(process)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertEqual(popen.call_args.kwargs["stdin"], sweep.subprocess.DEVNULL)

    def test_prediction_and_complete_target_selection(self):
        rows = [dict(width=2, phase="sweep", target="info", status="passed", wall_seconds=10,
                     memory={"peak_summed_rss_kib": 200}),
                dict(width=2, phase="sweep", target="functional", status="passed", wall_seconds=30,
                     memory={"peak_summed_rss_kib": 400}),
                dict(width=4, phase="sweep", target="info", status="passed", wall_seconds=5,
                     memory={"peak_summed_rss_kib": 300})]
        self.assertEqual(sweep.predicted_rss_kib(rows, "info", 8), 1000)
        self.assertEqual(sweep.eligible_totals(rows, ("info", "functional")), {2: 40})
        self.assertEqual(sweep.repeat_order(8), [8, 2, 8, 2])
        self.assertEqual(sweep.repeat_order(2), [2, 2])


class Orchestration(unittest.TestCase):
    def test_existing_output_is_not_rewritten_on_failure(self):
        with scratch() as root:
            (root / "failure.json").write_text("prior-run-evidence")
            experiment = SimpleNamespace(output=root, owns_output=False,
                                         run=Mock(side_effect=FileExistsError("already exists")))
            with patch.object(sweep.platform, "system", return_value="Linux"), \
                 patch.object(sweep.signal, "signal"), \
                 patch.object(sweep, "Sweep", return_value=experiment):
                with self.assertRaises(FileExistsError):
                    sweep.main(["--source", str(root), "--build", str(root),
                                "--output", str(root), "--slot-lock", str(root / "lock")])
            self.assertEqual((root / "failure.json").read_text(), "prior-run-evidence")

    def test_full_both_target_sweep_and_winner_repeats(self):
        with scratch() as root:
            source, build = root / "source", root / "build"
            source.mkdir(); build.mkdir()
            header = build / "include/rocshmem/rocshmem_config_embedded.hpp"
            header.parent.mkdir(parents=True); header.write_text("same configuration")
            (build / "CMakeCache.txt").write_text("HIP_CLANG_NUM_PARALLEL_JOBS:STRING=1\n")
            (build / "compile_commands.json").write_text("[]")
            (build / "librocshmem.a").write_bytes(b"fixed archive")
            (build / "info.o").write_bytes(b"fixed object")
            args = SimpleNamespace(source=source, build=build, output=root / "results", slot_lock=root / "lock",
                targets="both", order=sweep.WIDTHS, repeat_best=True, sample_interval=1,
                timeout_seconds=0, max_summed_rss_gib=0, min_available_gib=0, expected_affinity_cpus=96)
            state = {"width": 1}

            def configure(argv, **kwargs):
                state["width"] = int(argv[-1].split("=")[-1])
                header.write_text("same configuration")

            def ninja(argv, **kwargs):
                if "commands" in argv:
                    return ": && " + shlex.join(command(state["width"], argv[-1])) + " && :"
                return "[1/1] Linking CXX executable " + argv[-1] + "\n"

            def timing(argv, env, case, *unused):
                target = argv[-1]
                output = build / target
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"same output for " + target.encode())
                seconds = {1: 80, 2: 45, 4: 30, 8: 20, 16: 23, 31: 28}[state["width"]]
                if "functional" in target:
                    seconds *= 2
                return dict(wall_seconds=seconds, returncode=0, stop_reason=None,
                            memory={"peak_summed_rss_kib": state["width"] * 100})

            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(sweep.subprocess, "run", side_effect=configure), \
                 patch.object(sweep.subprocess, "check_output", side_effect=ninja), \
                 patch.object(sweep, "timed_link", side_effect=timing), \
                 patch.object(sweep, "measurement_lock", side_effect=lambda _: nullcontext()), \
                 patch.object(sweep, "meminfo", return_value={"MemAvailable_kib": 10**8}), \
                 patch.object(sweep, "cgroup_context", return_value={}), \
                 patch.object(sweep.os, "sched_getaffinity", return_value=set(range(96)), create=True):
                experiment = sweep.Sweep(args)
                self.assertEqual(experiment.run(), 0)
            summary = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(summary["screening_winner"], 8)
            self.assertEqual(summary["repeat_order"], [8, 2, 8, 2])
            self.assertEqual(summary["repeat_width2_median_link_total_seconds"], 135)
            self.assertEqual(summary["repeat_winner_median_link_total_seconds"], 60)
            self.assertFalse(summary["product_default_changed"])
            self.assertEqual(len(experiment.rows), 20)
            self.assertTrue(all(row["raw_output_parity"] for row in experiment.rows))
            self.assertEqual((build / "librocshmem.a").read_bytes(), b"fixed archive")


if __name__ == "__main__":
    unittest.main()
