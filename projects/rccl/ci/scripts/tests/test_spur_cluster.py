"""Tests for running test_madengine.py on Crusoe, a SPUR cluster.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import argparse
import json
import os
import pwd
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_madengine as tm

WORKLOAD = "llama-3.1-70b-training"
CONFIG = tm.WORKLOAD_CONFIGS[WORKLOAD]
RUBY = tm.CLUSTER_CONFIGS["ruby"]
CRUSOE = tm.CLUSTER_CONFIGS["crusoe"]


class TempWork(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name) / "work"
        (self.work / "scripts" / "primus_megatron-lm").mkdir(parents=True)

    def manifest(self, cluster: dict) -> dict:
        path = tm.generate_manifest(WORKLOAD, CONFIG, cluster, "img", 2, self.work)
        return json.loads(path.read_text())


class ManifestTest(TempWork):
    def test_docker_authz_refused_options_dropped(self):
        opts = self.manifest(CRUSOE)["built_models"]["overlay"][
            "additional_docker_run_options"]
        self.assertNotIn("--privileged", opts)
        self.assertNotIn("/run/udev", opts)
        for kept in ("--group-add render", "--device=/dev/infiniband",
                     "--cap-add IPC_LOCK", "--ulimit memlock=-1", "-v /sys:/sys:ro"):
            self.assertIn(kept, opts)
        self.assertNotIn("libibverbs", opts, "the image's own rdma-core must win")
        self.assertNotIn("  ", opts)

    def test_ruby_options_unchanged(self):
        opts = self.manifest(RUBY)["built_models"]["overlay"][
            "additional_docker_run_options"]
        self.assertTrue(opts.startswith(CONFIG["docker_run_options"]), opts)

    def test_denied_options_cover_every_workload(self):
        for name, config in tm.WORKLOAD_CONFIGS.items():
            rest = tm.drop_docker_run_options(config.get("docker_run_options", ""),
                                              CRUSOE["docker_run_options_denied"])
            self.assertNotIn("--privileged", rest, name)
            self.assertNotIn("/run/udev", rest, name)

    def test_rccl_env(self):
        m = self.manifest(CRUSOE)
        for env in (m["context"]["docker_env_vars"], m["deployment_config"]["env_vars"]):
            self.assertEqual(env["NCCL_IB_HCA"].split(","),
                             [f"ionic_{i}:1" for i in range(8)])
            self.assertEqual(env["NCCL_IB_GID_INDEX"], "1")
            self.assertEqual(env["NCCL_IB_DISABLE"], "0")
            self.assertEqual(env["RCCL_IB_QPS_PER_P2P"], "4")
            self.assertEqual(env["RCCL_USE_AMD_SMI_LIB"], "1")
            self.assertEqual(env["NCCL_TOPO_FILE"],
                             "/etc/crusoe/rccl_topo/mi355x-288gb-ib.xml")
            self.assertEqual(env["NCCL_IGNORE_CPU_AFFINITY"], "1")
            self.assertEqual(env["GLOO_SOCKET_IFNAME"], "ens3")
            for unset in ("NCCL_NET", "NCCL_NET_PLUGIN", "NCCL_IB_TC",
                          "NCCL_IB_FIFO_TC", "RCCL_FORCE_ENABLE_DMABUF"):
                self.assertNotIn(unset, env)

    def test_topology_file_mounted_read_only(self):
        opts = self.manifest(CRUSOE)["built_models"]["overlay"][
            "additional_docker_run_options"]
        topo_dir = os.path.dirname(CRUSOE["nccl_env"]["NCCL_TOPO_FILE"])
        self.assertIn(f" -v {topo_dir}:{topo_dir}:ro", opts)
        ruby_opts = self.manifest(RUBY)["built_models"]["overlay"][
            "additional_docker_run_options"]
        self.assertNotIn(topo_dir, ruby_opts)


class AllocationTest(TempWork):
    def run_allocation(self, cluster: dict, timeout: bool = False) -> mock.MagicMock:
        popen = mock.MagicMock()
        if timeout:
            popen.return_value.wait.side_effect = [
                subprocess.TimeoutExpired("salloc", 60), 0]
        else:
            popen.return_value.wait.return_value = 0
        self.subprocess_run = mock.MagicMock()
        phases = [(tm.BASELINE, self.work / "ab" / tm.BASELINE)]
        with mock.patch.dict(os.environ, {"MADENGINE_SLURM_EXCLUDE": ""}), \
                mock.patch.object(tm.subprocess, "Popen", popen), \
                mock.patch.object(tm.subprocess, "run", self.subprocess_run), \
                mock.patch.object(tm.os, "killpg"):
            tm.run_ab_in_one_allocation(
                phases, self.work, 2, WORKLOAD, CONFIG, cluster, timeout_minutes=1)
        return popen

    @staticmethod
    def option(cmd: list[str], name: str) -> str:
        return cmd[cmd.index(name) + 1]

    def test_spur_salloc_takes_the_script_on_stdin(self):
        popen = self.run_allocation(CRUSOE)
        cmd = popen.call_args.args[0]
        script = self.work / "ab" / "run_phases.sh"
        self.assertNotIn("--ntasks-per-node", cmd)
        self.assertNotIn("bash", cmd)
        self.assertEqual(self.option(cmd, "--gpus-per-node"), "8")
        self.assertEqual(self.option(cmd, "--account"), "amd-collectives")
        self.assertEqual(self.option(cmd, "--qos"), "amd-collectives-qos")
        self.assertEqual(self.option(cmd, "--partition"), "amd-spur-v2")
        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.PIPE)
        popen.return_value.stdin.write.assert_called_once_with(f"exec bash {script}\n")
        popen.return_value.stdin.close.assert_called_once()

    def test_slurm_salloc_unchanged(self):
        popen = self.run_allocation(RUBY)
        cmd = popen.call_args.args[0]
        self.assertEqual(self.option(cmd, "--ntasks-per-node"), "1")
        self.assertEqual(cmd[-2:], ["bash", str(self.work / "ab" / "run_phases.sh")])
        self.assertNotIn("--account", cmd)
        self.assertNotIn("--gpus-per-node", cmd)
        self.assertIsNone(popen.call_args.kwargs["stdin"])

    def test_repro_salloc_takes_the_clusters_options(self):
        cmd = self.run_allocation(CRUSOE).call_args.args[0]
        args = tm.repro_env(WORKLOAD, CONFIG, CRUSOE, 2, {}, {}, "", {})[
            "REPRO_SALLOC_ARGS"].split()
        self.assertEqual(args[::2], ["--qos", "--account", "--gpus-per-node"])
        for name, value in zip(args[::2], args[1::2]):
            self.assertEqual(self.option(cmd, name), value)
        self.assertNotIn("REPRO_SALLOC_ARGS",
                         tm.repro_env(WORKLOAD, CONFIG, RUBY, 2, {}, {}, "", {}))

    def test_release_cancels_by_user_and_name(self):
        self.run_allocation(CRUSOE, timeout=True)
        name = (self.work / "allocation.name").read_text().strip()
        user = pwd.getpwuid(os.getuid()).pw_name
        scancel = [c.args[0] for c in self.subprocess_run.call_args_list
                   if c.args[0][0] == "scancel"]
        self.assertEqual(scancel, [["scancel", "--user", user, "--name", name]])


class ReportTest(unittest.TestCase):
    def test_subject_names_the_cluster(self):
        subjects = set()
        for cluster in ("ruby", "crusoe"):
            args = argparse.Namespace(workload=WORKLOAD, cluster=cluster,
                                      notify_email="ci@example.com", teams_webhook="")
            with mock.patch.object(tm, "send_email_report") as send:
                tm.send_report(args, "report", 0)
            subject = send.call_args.kwargs["subject_prefix"]
            self.assertIn(cluster, subject)
            subjects.add(subject)
        self.assertEqual(len(subjects), 2)


if __name__ == "__main__":
    unittest.main()
