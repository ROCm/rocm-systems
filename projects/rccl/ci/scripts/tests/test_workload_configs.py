"""Tests that the workflows and WORKLOAD_CONFIGS in test_madengine.py agree.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_madengine as tm

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

WORKFLOWS = Path(__file__).resolve().parents[5] / ".github" / "workflows"
SCRIPT_MARGIN_MINUTES = 60  # what therock-rccl-test-madengine.yml keeps back


def load_workflow(name: str) -> dict:
    with open(WORKFLOWS / name) as f:
        return yaml.safe_load(f)


def madengine_legs() -> list[dict]:
    """Every (workload, nodes, job_timeout_minutes) the CI caller requests."""
    caller = load_workflow("therock-rccl-ci-linux.yml")
    callee = load_workflow("therock-rccl-test-madengine.yml")
    defaults = {k: v.get("default") for k, v in
                callee[True]["workflow_call"]["inputs"].items()}
    legs = []
    for job in caller["jobs"].values():
        if job.get("uses") != "./.github/workflows/therock-rccl-test-madengine.yml":
            continue
        entries = job.get("strategy", {}).get("matrix", {}).get("include", [{}])
        for entry in entries:
            leg = {}
            for key in ("workload", "nodes", "job_timeout_minutes"):
                value = str(job["with"].get(key, defaults[key]))
                if value.startswith("${{ matrix."):
                    value = str(entry[value[len("${{ matrix."):-len(" }}")]])
                elif value.startswith("${{"):
                    value = defaults[key]
                leg[key] = value
            legs.append(leg)
    return legs


@unittest.skipIf(yaml is None, "PyYAML not installed")
class WorkflowsMatchConfigs(unittest.TestCase):

    def test_dispatch_choices_are_the_configs(self):
        callee = load_workflow("therock-rccl-test-madengine.yml")
        options = callee[True]["workflow_dispatch"]["inputs"]["workload"]["options"]
        self.assertEqual(sorted(options), sorted(tm.WORKLOAD_CONFIGS))

    def test_scheduled_workloads_exist(self):
        legs = madengine_legs()
        self.assertGreater(len(legs), 1)
        for leg in legs:
            self.assertIn(leg["workload"], tm.WORKLOAD_CONFIGS)

    def test_scheduled_nodes_match_required_nodes(self):
        for leg in madengine_legs():
            required = tm.WORKLOAD_CONFIGS[leg["workload"]].get("required_nodes")
            if required is not None:
                self.assertEqual(int(leg["nodes"]), required, leg)

    def test_job_budget_covers_time_limit(self):
        for leg in madengine_legs():
            config = tm.WORKLOAD_CONFIGS[leg["workload"]]
            limit = tm.slurm_time_seconds(config["time_limit"]) // 60
            script = int(leg["job_timeout_minutes"]) - SCRIPT_MARGIN_MINUTES
            self.assertGreater(script, limit, leg)


class CheckMadKnobs(unittest.TestCase):

    def work_dir(self, script_text: str | None) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        if script_text is not None:
            d = tmp / "scripts" / "primus_megatron-lm"
            d.mkdir(parents=True)
            (d / "primus_megatron-lm_benchmark_report.sh").write_text(script_text)
        return tmp

    def test_pp4_needs_extra_args(self):
        config = tm.WORKLOAD_CONFIGS["llama-3.1-70b-pp4-training"]
        old = self.work_dir('train_iters="${PRIMUS_TRAIN_ITERS:-}"\n')
        with self.assertRaisesRegex(RuntimeError, "PRIMUS_EXTRA_ARGS"):
            tm.check_mad_knobs(old, config)
        tm.check_mad_knobs(self.work_dir('x="${PRIMUS_EXTRA_ARGS:-}"\n'), config)

    def test_missing_script_fails(self):
        config = tm.WORKLOAD_CONFIGS["llama-3.1-70b-pp4-training"]
        with self.assertRaises(RuntimeError):
            tm.check_mad_knobs(self.work_dir(None), config)

    def test_no_knobs_or_maxtext_is_not_checked(self):
        empty = self.work_dir(None)
        tm.check_mad_knobs(empty, tm.WORKLOAD_CONFIGS["llama-3.1-70b-training"])
        tm.check_mad_knobs(empty, tm.WORKLOAD_CONFIGS["llama-3.1-405b-maxtext"])


if __name__ == "__main__":
    unittest.main()
