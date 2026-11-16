# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Validates a `rocprofv3 --att --att-remap` run: the remapper placed every workgroup as recorded in
# every checked pass, the loop ended with the application's dispatch, and rocprofv3 decoded each
# traced pass on the SIMD the remapper chose for it.

import collections
import glob
import importlib.util
import json
import os
import pathlib
import sys

import pytest

SIMDS = {0, 1, 2, 3}


def traced(passes):
    return [p for p in passes["passes"] if p.get("trace")]


def pass_dir(output_dir, dispatch_id, pass_index):
    found = glob.glob(
        os.path.join(
            output_dir, f"ui_output_agent_*_dispatch_{dispatch_id}_pass_{pass_index}"
        )
    )
    assert len(found) == 1, f"pass {pass_index}: decoded output {found}"
    return found[0]


def test_remap_succeeded_and_placed_every_workgroup(report):
    assert report["status"] == "ok", report.get("reason")
    assert report["collection"]["launch_model"] == "table"
    assert report["placement"], "no placement checks"
    for check in report["placement"]:
        assert check["placed"] == check["total"] and check["every_tile_once"], check


def test_the_loop_ends_with_the_application_dispatch(passes):
    records = passes["passes"]
    assert [p["pass"] for p in records] == list(range(len(records)))
    stages = [p["stage"] for p in records]
    assert stages[0] == "record" and stages[-1] == "final"
    assert records[-1]["mode"] is None and "trace" not in records[-1]
    # Every check stage runs before any pass is traced.
    first_traced = next(i for i, p in enumerate(records) if p.get("trace"))
    assert {"record", "table", "mark_table", "off"} <= set(stages[:first_traced])
    assert all(p["stage"].startswith("att-") for p in traced(passes))


def test_each_sweep_traces_every_simd_once(passes):
    cells = collections.defaultdict(list)
    for p in traced(passes):
        if p["accepted"]:
            cells[(p["run"], p["sweep"])].append(p["trace"]["simd"])
    assert {run for run, _ in cells} == {"table", "natural", "original"}
    for key, simds in cells.items():
        assert sorted(simds) == sorted(SIMDS), f"{key}: SIMDs {simds}"


def test_every_traced_pass_was_decoded_on_its_planned_simd(output_dir, passes):
    for p in traced(passes):
        ui = pass_dir(output_dir, passes["dispatch_id"], p["pass"])
        with open(os.path.join(ui, "replay_target.json")) as ifs:
            target = json.load(ifs)
        assert target["simd_select"] == p["trace"]["simd"], (p, target)
        assert target["target_cu"] == p["trace"]["target_cu"], (p, target)
        assert target["shader_engine_mask"] == p["trace"]["shader_engine_mask"], (
            p,
            target,
        )
    untraced = [p["pass"] for p in passes["passes"] if not p.get("trace")]
    for pass_index in untraced:
        assert not glob.glob(
            os.path.join(output_dir, f"ui_output_agent_*_dispatch_*_pass_{pass_index}")
        ), f"untraced pass {pass_index} has thread trace output"


def test_layout_arranges_the_merge_input(dispatch_dir, output_dir, passes, layout_script):
    spec = importlib.util.spec_from_file_location("remapper_layout", layout_script)
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    runs = layout.arrange(pathlib.Path(dispatch_dir), pathlib.Path(output_dir))
    assert set(runs) == {"table", "natural", "original"}
    for run, traces in runs.items():
        cells = sorted(
            traces.glob("sweep-*/cu-*-simd-*-attempt-0/ui_output_agent_*_dispatch_1")
        )
        accepted = [p for p in traced(passes) if p["run"] == run and p["accepted"]]
        assert len(cells) == len(accepted), run


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
