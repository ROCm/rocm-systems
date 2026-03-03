#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
import subprocess
import pytest


def run_rocpd_convert(db_path, out_dir):
    """Convert rocpd database to CSV format."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "rocpd", "convert", "-i", db_path, "--output-format", "csv", "-d", out_dir]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"rocpd convert failed\ncmd={' '.join(cmd)}\nstdout={res.stdout}\nstderr={res.stderr}"


def find_kernel_trace_csv(out_dir):
    """Locate kernel_trace CSV file in output directory."""
    for fn in os.listdir(out_dir):
        if fn.endswith("kernel_trace.csv"):
            return os.path.join(out_dir, fn)
    assert False, f"kernel trace CSV not found in {out_dir}"


def load_csv_rows(path):
    """Load CSV file and return rows as list of dicts."""
    assert os.path.isfile(path), f"missing CSV: {path}"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0, f"empty CSV: {path}"
    return rows


def extract_json_kernel_records(json_root):
    """Extract kernel dispatch records from JSON output."""
    assert "rocprofiler-sdk-tool" in json_root, "missing rocprofiler-sdk-tool in JSON"
    tool = json_root["rocprofiler-sdk-tool"]
    if isinstance(tool, list) and len(tool) > 0:
        tool = tool[0]
    assert "buffer_records" in tool, "missing buffer_records in JSON"
    br = tool["buffer_records"]
    
    for key in ("kernel_dispatch", "kernel_trace", "kernel_dispatch_trace"):
        if key in br and isinstance(br[key], list) and len(br[key]) > 0:
            return br[key]
    assert False, f"no kernel dispatch records found in JSON buffer_records keys={list(br.keys())}"


def build_json_duration_map(records):
    """Build map of dispatch_id -> (start, end, duration) from JSON records."""
    m = {}
    for r in records:
        # Extract dispatch ID
        dispatch_info = r.get("dispatch_info", {})
        dispatch_id = dispatch_info.get("dispatch_id") if isinstance(dispatch_info, dict) else None
        
        # Fallback to correlation_id if no dispatch_id
        if dispatch_id is None:
            corr_id = r.get("correlation_id", {})
            if isinstance(corr_id, dict):
                dispatch_id = corr_id.get("internal", 0)
            else:
                dispatch_id = corr_id
        
        # Extract timestamps
        start = r.get("start_timestamp")
        end = r.get("end_timestamp")
        assert start is not None and end is not None, f"missing timestamps in JSON record: {r}"
        
        start = int(start)
        end = int(end)
        assert start > 0 and end > 0, f"invalid timestamps start={start} end={end}"
        assert end >= start, f"end before start: start={start} end={end}"
        
        duration = end - start
        m[str(dispatch_id)] = (start, end, duration)
    
    assert len(m) > 0, "no kernel records extracted from JSON"
    return m


def test_rocpd_kernel_trace_duration(json_data, db_path, tmp_path):
    """
    Test that rocpd CSV output contains Duration column and values match JSON.
    
    Test strategy:
    1. Generate JSON and rocpd output from SAME execution (using ROCPROF_OUTPUT_FORMAT env var)
    2. Use rocpd to convert database to CSV
    3. Compare CSV Duration with JSON-derived duration
    
    Since JSON and rocpd come from the same execution, timestamps should be IDENTICAL.
    We expect ZERO tolerance for differences.
    
    Validates:
    - Duration column exists in CSV
    - Duration values EXACTLY match between JSON and CSV (zero tolerance)
    - Duration correctly calculated as End - Start
    - Start and End timestamps also match exactly
    """
    # Convert rocpd DB to CSV
    out_dir = tmp_path / "rocpd_csv"
    run_rocpd_convert(db_path, str(out_dir))
    csv_path = find_kernel_trace_csv(str(out_dir))
    csv_rows = load_csv_rows(csv_path)
    
    # Verify Duration column exists
    assert "Duration" in csv_rows[0], f"missing 'Duration' column; columns={list(csv_rows[0].keys())}"
    
    # Extract JSON data
    json_records = extract_json_kernel_records(json_data)
    json_map = build_json_duration_map(json_records)
    
    # Track statistics
    matched_count = 0
    total_count = len(csv_rows)
    mismatches = []
    
    for csv_row in csv_rows:
        # Get CSV values
        csv_start = int(csv_row["Start_Timestamp"])
        csv_end = int(csv_row["End_Timestamp"])
        csv_dur = int(csv_row["Duration"])
        
        # Validate CSV internal consistency
        assert csv_start > 0 and csv_end > 0, f"invalid CSV timestamps: start={csv_start} end={csv_end}"
        assert csv_end >= csv_start, f"CSV end before start: {csv_end} < {csv_start}"
        assert csv_dur >= 0, f"negative CSV duration: {csv_dur}"
        assert csv_dur == (csv_end - csv_start), f"CSV duration mismatch: {csv_dur} != {csv_end - csv_start}"
        
        # Match with JSON and require EXACT match (zero tolerance)
        dispatch_id = csv_row.get("Dispatch_Id") or csv_row.get("Correlation_Id")
        if dispatch_id and str(dispatch_id) in json_map:
            matched_count += 1
            json_start, json_end, json_dur = json_map[str(dispatch_id)]
            
            # Check for exact match on all three values
            start_diff = csv_start - json_start
            end_diff = csv_end - json_end
            dur_diff = csv_dur - json_dur
            
            if start_diff != 0 or end_diff != 0 or dur_diff != 0:
                mismatches.append({
                    'dispatch_id': dispatch_id,
                    'csv_start': csv_start,
                    'json_start': json_start,
                    'start_diff': start_diff,
                    'csv_end': csv_end,
                    'json_end': json_end,
                    'end_diff': end_diff,
                    'csv_dur': csv_dur,
                    'json_dur': json_dur,
                    'dur_diff': dur_diff
                })
    
    # Report any mismatches
    if mismatches:
        error_lines = [
            "",
            "TIMESTAMP MISMATCHES DETECTED",
            f"{'Dispatch':<10} {'Start Diff':<12} {'End Diff':<12} {'Dur Diff':<12}",
            "=" * 50
        ]
        
        for m in mismatches[:10]:  # Show first 10
            error_lines.append(
                f"{m['dispatch_id']:<10} {m['start_diff']:<12} {m['end_diff']:<12} {m['dur_diff']:<12}"
            )
        
        if len(mismatches) > 10:
            error_lines.append(f"... and {len(mismatches) - 10} more mismatches")
        
        # Fail the test with detailed error
        first = mismatches[0]
        error_msg = "\n".join(error_lines) + "\n\n" + (
            f"Timestamp mismatch detected for dispatch {first['dispatch_id']}:\n"
            f"  CSV:  start={first['csv_start']}, end={first['csv_end']}, duration={first['csv_dur']}\n"
            f"  JSON: start={first['json_start']}, end={first['json_end']}, duration={first['json_dur']}\n"
            f"  Diff: start={first['start_diff']}, end={first['end_diff']}, duration={first['dur_diff']}\n"
            f"Total mismatches: {len(mismatches)}/{total_count}\n"
            f"NOTE: Since JSON and rocpd come from the same execution, timestamps should be identical."
        )
        assert False, error_msg
    
    # Ensure we matched all records
    assert matched_count > 0, f"No CSV rows matched with JSON records"
    assert matched_count == total_count, f"Only {matched_count}/{total_count} CSV rows matched JSON"


if __name__ == "__main__":
    rc = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(rc)
