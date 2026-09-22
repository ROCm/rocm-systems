"""CPU-only regressions. Run: python3 -m unittest discover -s test -p test_accl_correctness.py

Requires a C++14 compiler (CXX, default c++), but neither ROCm nor gtest.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


class AcclCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.build = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.build.cleanup)
        cls.driver = Path(cls.build.name) / "driver"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++14", "-pthread", "-I" + str(root / "plugins/profiler/accl"),
            "-I" + str(root / "src/include/plugin"),
            str(root / "test/accl_correctness_driver.cc"), "-o", str(cls.driver),
        ], check=True)

    def run_case(self, mode, threshold=None, long_path=False):
        env = os.environ.copy()
        env.pop("ACCL_PROFILER_MIN_SIZE_BYTES", None)
        if threshold is not None:
            env["ACCL_PROFILER_MIN_SIZE_BYTES"] = threshold
        with tempfile.TemporaryDirectory() as output:
            directory = Path(output)
            if long_path:
                while len(str(directory)) < 1000:
                    directory /= "x" * max(1, min(180, 1000 - len(str(directory)) - 1))
                directory.mkdir(parents=True)
            run = subprocess.run([str(self.driver), str(directory), mode], env=env,
                                 text=True, capture_output=True, check=True, timeout=15)
            paths = [path for path in Path(output).rglob("*") if path.is_file()]
            if long_path:
                self.assertEqual(paths, [])  # Even a truncated non-JSONL filename is wrong.
            rows = [json.loads(line) for path in paths
                    for line in path.read_text().splitlines()]
        return run.stdout, rows

    def test_invalid_thresholds_fail_open_and_warn(self):
        for value in ("-1", "\n-1", "\r-1", "\v-1", "\f-1", "abc", "0x2000",
                      "18446744073709551616", "123garbage"):
            with self.subTest(value=value):
                out, rows = self.run_case("threshold", value)
                self.assertIn("accepted=1", out)
                self.assertIn("warnings=1", out)
                self.assertEqual(sum("coll_perf" in row for row in rows), 1)

    def test_threshold_boundary_and_whitespace(self):
        for value, accepted in (("4096", 1), ("4097", 0), (" \t\n4096\r\v\f", 1),
                                ("+4096", 1), (" \t", 1), ("", 1)):
            with self.subTest(value=value):
                out, _ = self.run_case("threshold", value)
                self.assertIn(f"accepted={accepted}", out)
                self.assertIn("warnings=0", out)

    def test_unset_threshold_is_not_inherited_by_next_communicator(self):
        out, _ = self.run_case("percomm", "8192")
        self.assertIn("accepted=0\n", out)
        self.assertIn("second_accepted=1", out)

    def test_state_intervals_and_directional_means(self):
        _, rows = self.run_case("timing")
        decomp = next(row["coll_perf"]["decomposition"] for row in rows if "coll_perf" in row)
        self.assertEqual({key: decomp[key] for key in (
            "proxy_gpu_wait_us", "proxy_peer_wait_us", "proxy_network_us", "proxy_flush_us",
            "proxy_gpu_recv_wait_us", "n_send_ops", "n_recv_ops")}, {
                "proxy_gpu_wait_us": 10, "proxy_peer_wait_us": 20, "proxy_network_us": 70,
                "proxy_flush_us": 50, "proxy_gpu_recv_wait_us": 60, "n_send_ops": 2, "n_recv_ops": 1})
        self.assertTrue(rows[-1]["summary"]["complete"])

    def test_absent_receive_direction_is_zero(self):
        _, rows = self.run_case("send-only")
        decomp = rows[0]["coll_perf"]["decomposition"]
        self.assertEqual(decomp["proxy_network_us"], 30)
        self.assertEqual(decomp["proxy_flush_us"], 0)
        self.assertEqual(decomp["proxy_gpu_recv_wait_us"], 0)

    def test_each_proxy_loss_is_reported(self):
        for mode, field in (("ops", "dropped_proxy_ops"), ("steps", "dropped_proxy_steps"),
                            ("overflow", "overflow_proxy_ops")):
            with self.subTest(mode=mode):
                _, rows = self.run_case(mode)
                summary = rows[-1]["summary"]
                self.assertEqual(summary[field], 1)
                self.assertFalse(summary["complete"])
                self.assertEqual(summary["pool_size"], 256)  # Keep existing consumers compatible.

    def test_failed_summary_write_warns(self):
        out, _ = self.run_case("io")
        self.assertIn("warnings=1", out)

    def test_truncated_output_path_is_not_opened(self):
        out, _ = self.run_case("threshold", long_path=True)
        self.assertIn("warnings=1", out)

    def test_earlier_write_error_clears_complete(self):
        out, rows = self.run_case("prior-io")
        self.assertFalse(rows[-1]["summary"]["complete"])
        self.assertNotIn("warnings=0", out)


if __name__ == "__main__":
    unittest.main()
