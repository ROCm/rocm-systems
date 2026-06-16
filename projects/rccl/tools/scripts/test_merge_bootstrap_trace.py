#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("merge_bootstrap_trace.py")


def load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_bootstrap_trace", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MergeBootstrapTraceTest(unittest.TestCase):
    def test_collects_deploy_trace_text_events(self):
        mod = load_merge_module()
        with tempfile.TemporaryDirectory() as td:
            trace_dir = Path(td)
            (trace_dir / "rank0.log").write_text(
                "prefix DEPLOY_TRACE rank=0 pid=123 phase=deploy.exec "
                "t_ns=1000 dur_us=0 md=0 bytes=0 detail=constructor\n"
                "prefix DEPLOY_TRACE rank=0 pid=123 phase=mpi.init "
                "t_ns=2000 dur_us=50 md=0 bytes=0 detail=MPI_Init\n"
            )

            events = mod.collect([trace_dir])

        self.assertEqual(len(events), 2)
        self.assertEqual([e.name() for e in events], ["deploy.exec", "mpi.init"])
        self.assertEqual(events[0].rank, 0)
        self.assertEqual(events[1].dur_us, 50)


if __name__ == "__main__":
    unittest.main()
