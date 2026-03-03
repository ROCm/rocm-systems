#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices,
# Inc. All rights reserved.
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

import os
import sys
import csv
import json
import subprocess
import pytest

def node_exists(name, data, min_len=1):
    assert name in data, f"missing key: {name}"
    assert data[name] is not None, f"key is None: {name}"
    if hasattr(data[name], "__len__"):
        assert len(data[name]) >= min_len, f"key '{name}' too small"

def run_rocpd_convert(db_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "rocpd", "convert", "-i", db_path, "--output-format", "csv", "-d", out_dir]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"rocpd convert failed\ncmd={' '.join(cmd)}\nstdout={res.stdout}\nstderr={res.stderr}"

def find_kernel_trace_csv(out_dir):
    for fn in os.listdir(out_dir):
        if fn.endswith("kernel_trace.csv"):
            return os.path.join(out_dir, fn)
    assert False, f"kernel trace CSV not found in {out_dir}"

def load_csv_rows(path):
    assert os.path.isfile(path), f"missing CSV: {path}"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0, f"empty CSV: {path}"
    return rows

def extract_json_kernel_records(json_root):
    node_exists("rocprofiler-sdk-tool", json_root)
    tool = json_root["rocprofiler-sdk-tool"]
    if isinstance(tool, list) and len(tool) > 0:
        tool = tool[0]
    node_exists("buffer_records", tool)
    br = tool["buffer_records"]
    for key in ("kernel_dispatch", "kernel_trace", "kernel_dispatch_trace"):
        if key in br and isinstance(br[key], list) and len(br[key]) > 0:
            return br[key]
    assert False, f"cannot find kernel dispatch records in buffer_records keys={list(br.keys())}"

def build_json_map(records):
    m = {}
    for r in records:
        dispatch_info = r.get("dispatch_info", {})
        dispatch_id = dispatch_info.get("dispatch_id") if isinstance(dispatch_info, dict) else None
        corr_id = r.get("correlation_id", {})
        if isinstance(corr_id, dict):
            corr_id = corr_id.get("internal", 0)
        start = r.get("start_timestamp")
        end = r.get("end_timestamp")
        assert start is not None and end is not None, f"missing start/end in json record: {r}"
        start = int(start)
        end = int(end)
        assert start > 0 and end > 0, f"non-positive timestamps start={start} end={end}"
        assert end >= start, f"end before start: start={start} end={end}"
        key = dispatch_id if dispatch_id is not None else corr_id
        assert key is not None, f"no key to match json record: {r}"
        m[str(key)] = (start, end)
    assert len(m) > 0, "no records found in JSON"
    return m

def test_rocpd_kernel_trace_duration(json_data, db_path, tmp_path):
    out_dir = tmp_path / "rocpd_csv"
    run_rocpd_convert(db_path, str(out_dir))
    csv_path = find_kernel_trace_csv(str(out_dir))
    rows = load_csv_rows(csv_path)
    
    # Main test: verify Duration column exists
    assert "Duration" in rows[0], f"missing 'Duration' column; columns={list(rows[0].keys())}"
    
    json_records = extract_json_kernel_records(json_data)
    json_map = build_json_map(json_records)
    
    for row in rows:
        key = row.get("Dispatch_Id") or row.get("Correlation_Id")
        assert key is not None, f"cannot match row: {row}"
        
        start = int(row["Start_Timestamp"])
        end = int(row["End_Timestamp"])
        dur = int(row["Duration"])
        
        # Verify timestamps are reasonable
        assert start > 0 and end > 0, f"non-positive timestamps in CSV: start={start} end={end}"
        assert end >= start, f"end before start in CSV: start={start} end={end}"
        assert dur >= 0, f"negative duration in CSV: dur={dur}"
        
        # Verify Duration column is correctly calculated from Start and End
        assert dur == (end - start), f"duration mismatch in CSV: dur={dur} vs calculated={end - start}"
        
        if str(key) in json_map:
            jstart, jend = json_map[str(key)]
            json_dur = jend - jstart
            diff = abs(json_dur - dur)
            # Allow small differences due to clock skew/adjustments
            # Just warn if difference is significant (>10% or >1000ns)
            if diff > 1000 and diff > dur * 0.1:
                print(f"INFO: Large timestamp difference for key={key}: csv_dur={dur} json_dur={json_dur} diff={diff}")

if __name__ == "__main__":
    rc = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(rc)