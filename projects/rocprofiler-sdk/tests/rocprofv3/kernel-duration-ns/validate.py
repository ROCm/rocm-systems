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

import sys
import pytest


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


def _as_int(val, *, field="value"):
    assert val is not None, f"missing {field}"
    try:
        return int(val)
    except Exception as e:
        raise AssertionError(f"failed to parse int for {field}: {val!r} ({e})") from e


def _extract_dispatch_id_from_json_record(r):
    """
    Prefer dispatch_info.dispatch_id.
    Fallback to correlation_id.internal (or correlation_id if scalar).
    Return int.
    """
    dispatch_info = r.get("dispatch_info", {})
    dispatch_id = None
    if isinstance(dispatch_info, dict):
        dispatch_id = dispatch_info.get("dispatch_id", None)

    if dispatch_id is None:
        corr_id = r.get("correlation_id", {})
        if isinstance(corr_id, dict):
            dispatch_id = corr_id.get("internal", None)
        else:
            dispatch_id = corr_id

    return _as_int(dispatch_id, field="dispatch_id/correlation_id")


def build_json_duration_map(records):
    """
    Build map:
        key(int dispatch_id) -> (start, end, duration)
    """
    m = {}
    for r in records:
        did = _extract_dispatch_id_from_json_record(r)

        start = _as_int(r.get("start_timestamp"), field="start_timestamp")
        end = _as_int(r.get("end_timestamp"), field="end_timestamp")

        assert start > 0 and end > 0, f"invalid timestamps start={start} end={end}"
        assert end >= start, f"end before start: start={start} end={end}"

        m[did] = (start, end, end - start)

    assert len(m) > 0, "no kernel records extracted from JSON"
    return m


def load_kernel_rows_via_rocpd(db_path):
    """
    Use rocpd Python API to query the same underlying data used by rocpd/csv.py::write_kernel_csv().
    Returns list[dict].
    """
    try:
        import rocpd
    except Exception as e:
        raise AssertionError(
            f"failed to import rocpd python module. Ensure PYTHONPATH is set for rocprofiler-sdk build tree. ({e})"
        ) from e

    # RocpdImportData can take a list of inputs
    data = rocpd.connect([db_path])

    # Minimal columns required for strict consistency checks
    # NOTE: rocpd/csv.py::write_kernel_csv selects from "kernels"
    query = """
        SELECT
            dispatch_id AS Dispatch_Id,
            stack_id   AS Correlation_Id,
            start      AS Start_Timestamp,
            end        AS End_Timestamp,
            (end - start) AS Duration
        FROM "kernels"
        ORDER BY
            guid ASC, start ASC, end DESC
    """

    cur = rocpd.execute(data, query)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    assert len(rows) > 0, f"no rows returned from kernels table in db: {db_path}"
    return rows


def test_rocpd_kernel_trace_duration(json_data, db_path):
    """
    Test that rocpd DB content for kernel trace has Duration and it matches JSON derived durations.

    Strategy:
      - JSON and rocpd DB are generated from the SAME rocprofv3 execution (ROCPROF_OUTPUT_FORMAT=json,rocpd)
      - Read kernel records from JSON
      - Read kernel rows from rocpd DB using rocpd Python API (no CSV I/O)
      - Enforce:
          * DB Duration == End - Start
          * DB Start/End/Duration EXACTLY match JSON for each dispatch_id (zero tolerance)
          * All kernel rows in DB match to a JSON record
    """
    # Load DB rows via rocpd Python API
    db_rows = load_kernel_rows_via_rocpd(db_path)

    # Build JSON dispatch_id -> (start,end,dur)
    json_records = extract_json_kernel_records(json_data)
    json_map = build_json_duration_map(json_records)

    total_count = len(db_rows)
    matched_count = 0
    mismatches = []
    missing_in_json = []

    for row in db_rows:
        did = _as_int(row.get("Dispatch_Id"), field="Dispatch_Id")
        start = _as_int(row.get("Start_Timestamp"), field="Start_Timestamp")
        end = _as_int(row.get("End_Timestamp"), field="End_Timestamp")
        dur = _as_int(row.get("Duration"), field="Duration")

        # DB internal consistency
        assert start > 0 and end > 0, f"invalid DB timestamps: start={start} end={end} dispatch_id={did}"
        assert end >= start, f"DB end before start: start={start} end={end} dispatch_id={did}"
        assert dur >= 0, f"negative DB duration: duration={dur} dispatch_id={did}"
        assert dur == (end - start), (
            f"DB duration mismatch: duration={dur} != end-start={end - start} dispatch_id={did}"
        )

        if did not in json_map:
            missing_in_json.append(did)
            continue

        matched_count += 1
        j_start, j_end, j_dur = json_map[did]

        sd = start - j_start
        ed = end - j_end
        dd = dur - j_dur

        if sd != 0 or ed != 0 or dd != 0:
            mismatches.append(
                {
                    "dispatch_id": did,
                    "db_start": start,
                    "json_start": j_start,
                    "start_diff": sd,
                    "db_end": end,
                    "json_end": j_end,
                    "end_diff": ed,
                    "db_dur": dur,
                    "json_dur": j_dur,
                    "dur_diff": dd,
                }
            )

    # Hard failures with actionable context
    if missing_in_json:
        sample = missing_in_json[:10]
        raise AssertionError(
            "Some DB kernel rows had dispatch_id not present in JSON records. "
            "Since JSON and rocpd come from the same execution, dispatch IDs should align.\n"
            f"Missing count: {len(missing_in_json)}/{total_count}\n"
            f"Sample missing dispatch_ids: {sample}"
        )

    if mismatches:
        lines = [
            "",
            "TIMESTAMP/DURATION MISMATCHES DETECTED (zero tolerance)",
            f"{'Dispatch':<12} {'StartDiff':<12} {'EndDiff':<12} {'DurDiff':<12}",
            "=" * 56,
        ]
        for m in mismatches[:10]:
            lines.append(
                f"{m['dispatch_id']:<12} {m['start_diff']:<12} {m['end_diff']:<12} {m['dur_diff']:<12}"
            )
        if len(mismatches) > 10:
            lines.append(f"... and {len(mismatches) - 10} more mismatches")

        first = mismatches[0]
        detail = (
            f"\n\nExample mismatch for dispatch {first['dispatch_id']}:\n"
            f"  DB:   start={first['db_start']}, end={first['db_end']}, duration={first['db_dur']}\n"
            f"  JSON: start={first['json_start']}, end={first['json_end']}, duration={first['json_dur']}\n"
            f"  Diff: start={first['start_diff']}, end={first['end_diff']}, duration={first['dur_diff']}\n"
            f"Total mismatches: {len(mismatches)}/{total_count}\n"
            "NOTE: Since JSON and rocpd come from the same execution, these should be identical."
        )
        raise AssertionError("\n".join(lines) + detail)

    assert matched_count > 0, "No DB rows matched JSON records"
    assert matched_count == total_count, f"Only {matched_count}/{total_count} DB rows matched JSON"


if __name__ == "__main__":
    rc = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(rc)
