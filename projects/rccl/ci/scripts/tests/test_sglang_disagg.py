"""Tests for the sglang-disagg family in test_madengine.py.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_madengine as tm

WORKLOAD = "deepseek-r1-sglang-disagg"
CONFIG = tm.WORKLOAD_CONFIGS[WORKLOAD]
RUBY = tm.CLUSTER_CONFIGS["ruby"]
MAD_DOCKERFILE = """\
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/sgl-dev:v0.5.16-rocm720-mi35x-20260807
FROM $BASE_DOCKER
RUN echo payload
"""


def sweep_row(point: str, metric: str, value: float, status: str = "SUCCESS") -> dict:
    return {"model": f"{CONFIG['model_repo']}_2p2d_{point}", "metric": metric,
            "performance": value, "training_precision": "",
            "multi_results": {"launcher": "sglang-disagg"}, "status": status}


class TempWork(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name) / "work"
        (self.work / "scripts" / "sglang_disagg").mkdir(parents=True)
        (self.work / "MAD" / "docker").mkdir(parents=True)
        (self.work / "MAD" / tm.SGLANG_MAD_DOCKERFILE).write_text(MAD_DOCKERFILE)

    def manifest(self, phase: str) -> dict:
        run_dir = tm.prepare_phase_dir(self.work, phase)
        path = tm.generate_manifest(
            WORKLOAD, CONFIG, RUBY, f"image-{phase}", CONFIG["required_nodes"],
            self.work, run_dir=run_dir,
            dockerfile_name=(tm.SGLANG_BASELINE_DOCKERFILE if phase == tm.BASELINE
                             else tm.OVERLAY_DOCKERFILE),
        )
        return json.loads(path.read_text())


class SweepTest(unittest.TestCase):
    def test_every_gate_point_is_benchmarked(self):
        env = CONFIG["env_vars"]
        combos = set(env["BENCHMARK_COMBINATIONS"].split())
        levels = set(env["BENCHMARK_CONCURRENCY_LEVELS"].split())
        for point in CONFIG["precisions"]:
            isl, osl, con = re.fullmatch(r"isl(\d+)_osl(\d+)_con(\d+)", point).groups()
            self.assertIn(f"{isl}/{osl}", combos, point)
            self.assertIn(con, levels, point)


class GenerateManifestTest(TempWork):
    def test_launcher_and_layout(self):
        m = self.manifest(tm.CANDIDATE)
        dist = m["deployment_config"]["distributed"]
        self.assertEqual(dist["launcher"], "sglang-disagg")
        self.assertEqual(dist["sglang_disagg"], {"prefill_nodes": 2, "decode_nodes": 2})
        self.assertEqual(dist["nnodes"], 4)
        model = m["built_models"]["overlay"]
        self.assertEqual(model["scripts"], "scripts/sglang_disagg/run.sh")
        self.assertIn("--xp 2 --yd 2", model["args"])
        for env in (m["context"]["docker_env_vars"], m["deployment_config"]["env_vars"]):
            self.assertEqual(env["MOE_A2A_BACKEND"], "none")
            self.assertEqual(env["NCCL_DEBUG"], "INFO")
            self.assertEqual(env["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"], "0")
            self.assertEqual(env["OMP_NUM_THREADS"], "8")

    def test_docker_options(self):
        opts = self.manifest(tm.CANDIDATE)["built_models"]["overlay"][
            "additional_docker_run_options"]
        self.assertTrue(opts.startswith("--init "), opts)
        self.assertIn(f"-v {RUBY['models_dir']}:/models:ro", opts)
        self.assertNotIn("libibverbs", opts, "the image's own rdma-core must win")

    def test_each_phase_gets_its_own_shared_run_logs(self):
        mounts = {}
        for phase in (tm.BASELINE, tm.CANDIDATE):
            opts = self.manifest(phase)["built_models"]["overlay"][
                "additional_docker_run_options"]
            run_logs = self.work / "ab" / phase / "slurm_output" / "run_logs"
            self.assertIn(f"-v {run_logs}:/run_logs", opts)
            self.assertEqual(stat.S_IMODE(run_logs.stat().st_mode), 0o777)
            mounts[phase] = run_logs
        self.assertNotEqual(mounts[tm.BASELINE], mounts[tm.CANDIDATE])

    def test_megatron_keeps_host_ib_libs(self):
        config = tm.WORKLOAD_CONFIGS["llama-3.1-70b-training"]
        (self.work / "scripts" / "primus_megatron-lm").mkdir()
        path = tm.generate_manifest("llama-3.1-70b-training", config, RUBY, "img", 2,
                                    self.work, pull_only=True)
        opts = json.loads(path.read_text())["built_models"]["overlay"][
            "additional_docker_run_options"]
        self.assertIn("libibverbs", opts)
        self.assertNotIn("/run_logs", opts)

    def test_repro_bundle_points_run_logs_at_the_submit_dir(self):
        m = tm.repro_manifest(self.manifest(tm.CANDIDATE), self.work)
        opts = m["built_models"]["overlay"]["additional_docker_run_options"]
        self.assertIn("-v ${SLURM_SUBMIT_DIR}/slurm_output/run_logs:/run_logs", opts)
        self.assertNotIn(str(self.work), opts)


class RecipeTest(TempWork):
    def test_base_pinned_and_patchelf_added(self):
        text = tm.sglang_recipe(self.work / "MAD", "rocm/sgl-dev@sha256:abc")
        self.assertIn("\nFROM rocm/sgl-dev@sha256:abc\n", text)
        self.assertNotIn("FROM $BASE_DOCKER", text)
        self.assertTrue(text.rstrip().endswith(tm.SGLANG_SWAP_TOOLS.rstrip()))
        self.assertLess(text.index("RUN echo payload"), text.index("patchelf"))

    def test_unexpected_recipe_fails(self):
        (self.work / "MAD" / tm.SGLANG_MAD_DOCKERFILE).write_text("FROM ubuntu\n")
        with self.assertRaisesRegex(RuntimeError, "FROM \\$BASE_DOCKER"):
            tm.sglang_recipe(self.work / "MAD", "x")


class CollectResultsTest(unittest.TestCase):
    def test_only_gate_points_keyed_by_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            rows = [
                sweep_row("isl1024_osl1024_con8", CONFIG["metric_key"], 217.2),
                sweep_row("isl1024_osl1024_con64", CONFIG["metric_key"], 1638.6),
                sweep_row("isl1024_osl1024_con64", "mean_itl_ms", 37.2),
                sweep_row("isl8192_osl1024_con64", CONFIG["metric_key"], 1229.3, "FAILURE"),
            ]
            (run_dir / "perf_entry_super.json").write_text(json.dumps(rows))
            results = tm.collect_phase_results(
                run_dir, CONFIG["metric_key"], only=CONFIG["precisions"])
        got = {(r["precision"], r["metric_value"], r["status"]) for r in results}
        self.assertEqual(got, {("isl1024_osl1024_con64", 1638.6, "pass"),
                               ("isl8192_osl1024_con64", 1229.3, "fail")})

    def test_training_rows_unaffected(self):
        row = {"model": "primus_x_Llama-3.1-70B", "metric": "tok_per_s_per_gpu_avg",
               "performance": 1519.5, "training_precision": "",
               "multi_results": {"precision": "BF16"}, "status": "SUCCESS"}
        self.assertEqual(tm._row_precision(row), "BF16")
        self.assertEqual(tm._row_precision({"model": "primus_x_Llama-3.1-70B"}), "")


class BannerLogsTest(unittest.TestCase):
    def test_server_logs_preferred_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            out = run_dir / "slurm_output"
            logs = out / "run_logs" / "79405"
            logs.mkdir(parents=True)
            (out / "madengine-x_79405_node_0.out").write_text("no banner here\n")
            banner = "h:1:1 [0] NCCL INFO RCCL version : 2.32.3-develop:2b88119\n"
            for name in ("prefill_NODE0.log", "prefill_NODE1.log",
                         "decode_NODE2.log", "decode_NODE3.log"):
                (logs / name).write_text(banner)
            (logs / "proxy_NODE0.log").write_text("router\n")
            self.assertEqual([p.name for p in tm.rccl_banner_logs(run_dir)],
                             ["decode_NODE2.log", "decode_NODE3.log",
                              "prefill_NODE0.log", "prefill_NODE1.log"])
            ok, msg = tm.verify_rccl_replacement(
                run_dir, {"version": "2.32.3-develop:2b88119"})
            self.assertTrue(ok, msg)
            (logs / "decode_NODE3.log").write_text("died before init\n")
            ok, _ = tm.verify_rccl_replacement(
                run_dir, {"version": "2.32.3-develop:2b88119"})
            self.assertFalse(ok)


class AllocationScriptTest(TempWork):
    def run_allocation(self, workload: str, env: dict) -> tuple[list[str], str]:
        popen = mock.MagicMock()
        popen.return_value.wait.return_value = 0
        phases = [(tm.BASELINE, self.work / "ab" / tm.BASELINE)]
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(tm.subprocess, "Popen", popen), \
                mock.patch.object(tm.subprocess, "run"):
            tm.run_ab_in_one_allocation(
                phases, self.work, 4, workload, tm.WORKLOAD_CONFIGS[workload],
                RUBY, timeout_minutes=1)
        cmd = popen.call_args.args[0]
        return cmd, (self.work / "ab" / "run_phases.sh").read_text()

    def test_dns_preflight_for_sglang_only(self):
        _, script = self.run_allocation(WORKLOAD, {})
        self.assertIn("preflight_dns.sh", script)
        self.assertTrue((self.work / "ab" / "preflight_dns.sh").is_file())
        _, script = self.run_allocation("llama-3.1-405b-maxtext", {})
        self.assertNotIn("preflight_dns.sh", script)

    def test_exclude_from_environment(self):
        cmd, _ = self.run_allocation(WORKLOAD, {"MADENGINE_SLURM_EXCLUDE": "n[01-02],n07"})
        self.assertEqual(cmd[cmd.index("--exclude") + 1], "n[01-02],n07")
        cmd, _ = self.run_allocation(WORKLOAD, {"MADENGINE_SLURM_EXCLUDE": ""})
        self.assertNotIn("--exclude", cmd)
        with self.assertRaises(RuntimeError):
            self.run_allocation(WORKLOAD, {"MADENGINE_SLURM_EXCLUDE": "n01; rm -rf /"})


if __name__ == "__main__":
    unittest.main()
