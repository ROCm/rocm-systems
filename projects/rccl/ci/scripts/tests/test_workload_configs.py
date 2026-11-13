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

    def test_only_an_expansion_of_the_knob_counts(self):
        config = tm.WORKLOAD_CONFIGS["llama-3.1-70b-pp4-training"]
        for text in ("# PRIMUS_EXTRA_ARGS is read by run_primus()\n",
                     'x="${PRIMUS_EXTRA_ARGS_V2:-}"\n',
                     'x="${MY_PRIMUS_EXTRA_ARGS:-}"\n'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                tm.check_mad_knobs(self.work_dir(text), config)
        tm.check_mad_knobs(self.work_dir('read -a a <<< "$PRIMUS_EXTRA_ARGS"\n'), config)

    def test_comments_single_quotes_and_escapes_do_not_count(self):
        config = tm.WORKLOAD_CONFIGS["llama-3.1-70b-pp4-training"]
        for text in ('# x="${PRIMUS_EXTRA_ARGS:-}"\n',
                     'x=1  # "$PRIMUS_EXTRA_ARGS"\n',
                     "echo '${PRIMUS_EXTRA_ARGS:-}'\n",
                     'echo \\${PRIMUS_EXTRA_ARGS}\n',
                     'echo "\\$PRIMUS_EXTRA_ARGS"\n'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                tm.check_mad_knobs(self.work_dir(text), config)

    def test_expansions_around_other_quoting_count(self):
        config = tm.WORKLOAD_CONFIGS["llama-3.1-70b-pp4-training"]
        # The pinned script's own read (scripts/primus_megatron-lm, line 265).
        for text in ("IFS=$' \\t\\n' read -r -d '' -a extra_args "
                     '<<< "$PRIMUS_EXTRA_ARGS" || true\n',
                     'echo "don\'t"; x="${PRIMUS_EXTRA_ARGS:-}"\n',
                     "echo 'a'; n=${#x}; x=$PRIMUS_EXTRA_ARGS  # ok\n"):
            with self.subTest(text=text):
                tm.check_mad_knobs(self.work_dir(text), config)

    def test_missing_script_fails(self):
        config = tm.WORKLOAD_CONFIGS["llama-3.1-70b-pp4-training"]
        with self.assertRaises(RuntimeError):
            tm.check_mad_knobs(self.work_dir(None), config)

    def test_no_knobs_or_maxtext_is_not_checked(self):
        empty = self.work_dir(None)
        tm.check_mad_knobs(empty, tm.WORKLOAD_CONFIGS["llama-3.1-70b-training"])
        tm.check_mad_knobs(empty, tm.WORKLOAD_CONFIGS["llama-3.1-405b-maxtext"])


class SglangLauncherKnobs(unittest.TestCase):

    def test_barrier_timeout_uses_the_launcher_name(self):
        # sglang_disagg_mori_io_ep.sh reads BARRIER_TIMEOUT_SECONDS; any other
        # name leaves its 3600 s default in place.
        env = tm.WORKLOAD_CONFIGS["deepseek-r1-sglang-disagg"]["env_vars"]
        self.assertEqual(env.get("BARRIER_TIMEOUT_SECONDS"), "1800")
        self.assertNotIn("BARRIER_TIMEOUT_S", env)


class IgnoredMadKnobs(unittest.TestCase):

    # As MAD 36339bd8 prints it from warn_if_knobs_ignored().
    WARN = ("[WARN] PRIMUS_TRAIN_ITERS/PRIMUS_DISABLE_PROFILE/PRIMUS_EXTRA_ARGS are set, "
            "but Llama-3.3-70B launches primus-cli directly rather than through "
            "run_primus(), so they had no effect on this run.\n")

    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        self.run_dir = tmp
        self.logs = tmp / "slurm_output"
        self.logs.mkdir()

    def test_reports_the_logs_with_the_warning(self):
        (self.logs / "x_node_0.out").write_text("train\n" + self.WARN)
        (self.logs / "x_node_1.out").write_text("train\n")
        self.assertEqual(tm.ignored_mad_knobs(self.run_dir), ["x_node_0.out"])

    def test_honoured_knobs_are_not_reported(self):
        (self.logs / "x_node_0.out").write_text(
            "[INFO] Primus CLI knobs from environment: --pipeline_model_parallel_size 4\n")
        self.assertEqual(tm.ignored_mad_knobs(self.run_dir), [])

    def test_missing_slurm_output(self):
        self.assertEqual(tm.ignored_mad_knobs(self.run_dir / "nope"), [])


if __name__ == "__main__":
    unittest.main()
