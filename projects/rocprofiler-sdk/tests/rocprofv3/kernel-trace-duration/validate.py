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
    buffer_records = tool["buffer_records"]

    for key in ("kernel_dispatch", "kernel_trace", "kernel_dispatch_trace"):
        if (
            key in buffer_records
            and isinstance(buffer_records[key], list)
            and len(buffer_records[key]) > 0
        ):
            return buffer_records[key]

    assert False, (
        "no kernel dispatch records found in JSON buffer_records keys="
        f"{list(buffer_records.keys())}"
    )


def _as_int(value, *, field="value"):
    assert value is not None, f"missing {field}"
    try:
        return int(value)
    except Exception as exc:
        raise AssertionError(
            f"failed to parse int for {field}: {value!r} ({exc})"
        ) from exc


def _get_first_present(mapping, *keys):
    """
    Look up keys in mapping in a case-insensitive manner.
    Builds a lowercased lookup table once, then checks each key (lowercased).
    Returns the first non-None value found, or None if nothing matches.
    """
    lowered = {k.lower(): v for k, v in mapping.items()}
    for key in keys:
        normalized = key.lower()
        if normalized in lowered and lowered[normalized] is not None:
            return lowered[normalized]
    return None


def _extract_dispatch_id_from_json_record(record):
    """
    Prefer dispatch_info.dispatch_id.
    Fallback to correlation_id.internal (or correlation_id if scalar).
    Return int.
    """
    dispatch_info = record.get("dispatch_info", {})
    dispatch_id = None

    if isinstance(dispatch_info, dict):
        dispatch_id = dispatch_info.get("dispatch_id", None)

    if dispatch_id is None:
        correlation_id = record.get("correlation_id", {})
        if isinstance(correlation_id, dict):
            dispatch_id = correlation_id.get("internal", None)
        else:
            dispatch_id = correlation_id

    return _as_int(dispatch_id, field="dispatch_id/correlation_id")


def build_json_duration_map(records):
    """
    Build map:
        key(int dispatch_id) -> (start, end, duration)
    """
    result = {}

    for record in records:
        dispatch_id = _extract_dispatch_id_from_json_record(record)

        start = _as_int(record.get("start_timestamp"), field="start_timestamp")
        end = _as_int(record.get("end_timestamp"), field="end_timestamp")

        assert start > 0 and end > 0, f"invalid timestamps start={start} end={end}"
        assert end >= start, f"end before start: start={start} end={end}"

        result[dispatch_id] = (start, end, end - start)

    assert len(result) > 0, "no kernel records extracted from JSON"
    return result


def load_kernel_rows_via_rocpd(db_path):
    """
    Use rocpd Python API and reuse rocpd.csv kernel query logic directly.
    Returns list[dict].
    """
    try:
        import rocpd
        from rocpd import csv as rocpd_csv
        from rocpd import output_config as rocpd_output_config
    except Exception as exc:
        raise AssertionError(
            "failed to import rocpd python modules. "
            f"Ensure PYTHONPATH is set for rocprofiler-sdk build tree. ({exc})"
        ) from exc

    data = rocpd.connect([db_path])

    config = rocpd_output_config.output_config()
    query = rocpd_csv.get_kernel_csv_query(config)

    cursor = rocpd.execute(data, query)
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    assert len(rows) > 0, f"no rows returned from kernel query in db: {db_path}"
    return rows


def _extract_dispatch_id_from_db_row(row):
    value = _get_first_present(row, "dispatch_id", "correlation_id")
    return _as_int(value, field="dispatch_id/correlation_id")


def _extract_start_from_db_row(row):
    value = _get_first_present(row, "start_timestamp")
    return _as_int(value, field="start_timestamp")


def _extract_end_from_db_row(row):
    value = _get_first_present(row, "end_timestamp")
    return _as_int(value, field="end_timestamp")


def _extract_duration_from_db_row(row):
    value = _get_first_present(row, "duration")
    return _as_int(value, field="duration")


def test_rocpd_kernel_trace_duration(json_data, db_path):
    """
    Validate that kernel trace Duration exists on the rocpd CSV path and matches JSON.

    Strategy:
      - JSON and rocpd DB are generated from the SAME rocprofv3 execution
        (ROCPROF_OUTPUT_FORMAT=json,rocpd)
      - Read kernel records from JSON
      - Read kernel rows from rocpd DB using rocpd Python API
      - Reuse rocpd.csv.get_kernel_csv_query(config) so test and CSV path stay aligned
      - Enforce:
          * Duration == End_Timestamp - Start_Timestamp
          * Start/End/Duration EXACTLY match JSON for each dispatch_id
          * All DB rows match JSON rows
    """
    db_rows = load_kernel_rows_via_rocpd(db_path)

    json_records = extract_json_kernel_records(json_data)
    json_map = build_json_duration_map(json_records)

    total_count = len(db_rows)
    matched_count = 0
    mismatches = []
    missing_in_json = []

    for row in db_rows:
        dispatch_id = _extract_dispatch_id_from_db_row(row)
        start = _extract_start_from_db_row(row)
        end = _extract_end_from_db_row(row)
        duration = _extract_duration_from_db_row(row)

        assert (
            start > 0 and end > 0
        ), f"invalid DB timestamps: start={start} end={end} dispatch_id={dispatch_id}"
        assert (
            end >= start
        ), f"DB end before start: start={start} end={end} dispatch_id={dispatch_id}"
        assert (
            duration >= 0
        ), f"negative DB duration: duration={duration} dispatch_id={dispatch_id}"
        assert duration == (end - start), (
            f"DB duration mismatch: duration={duration} != end-start={end - start} "
            f"dispatch_id={dispatch_id}"
        )

        if dispatch_id not in json_map:
            missing_in_json.append(dispatch_id)
            continue

        matched_count += 1
        json_start, json_end, json_duration = json_map[dispatch_id]

        start_diff = start - json_start
        end_diff = end - json_end
        duration_diff = duration - json_duration

        if start_diff != 0 or end_diff != 0 or duration_diff != 0:
            mismatches.append(
                {
                    "dispatch_id": dispatch_id,
                    "db_start": start,
                    "json_start": json_start,
                    "start_diff": start_diff,
                    "db_end": end,
                    "json_end": json_end,
                    "end_diff": end_diff,
                    "db_duration": duration,
                    "json_duration": json_duration,
                    "duration_diff": duration_diff,
                }
            )

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
            "TIMESTAMP/DURATION MISMATCHES DETECTED",
            f"{'Dispatch':<12} {'StartDiff':<12} {'EndDiff':<12} {'DurDiff':<12}",
            "=" * 56,
        ]

        for item in mismatches[:10]:
            lines.append(
                f"{item['dispatch_id']:<12} "
                f"{item['start_diff']:<12} "
                f"{item['end_diff']:<12} "
                f"{item['duration_diff']:<12}"
            )

        if len(mismatches) > 10:
            lines.append(f"... and {len(mismatches) - 10} more mismatches")

        first = mismatches[0]
        detail = (
            f"\n\nExample mismatch for dispatch {first['dispatch_id']}:\n"
            f"  DB:   start={first['db_start']}, end={first['db_end']}, "
            f"duration={first['db_duration']}\n"
            f"  JSON: start={first['json_start']}, end={first['json_end']}, "
            f"duration={first['json_duration']}\n"
            f"  Diff: start={first['start_diff']}, end={first['end_diff']}, "
            f"duration={first['duration_diff']}\n"
            f"Total mismatches: {len(mismatches)}/{total_count}\n"
            "NOTE: Since JSON and rocpd come from the same execution, these should be identical."
        )
        raise AssertionError("\n".join(lines) + detail)

    assert matched_count > 0, "No DB rows matched JSON records"
    assert (
        matched_count == total_count
    ), f"Only {matched_count}/{total_count} DB rows matched JSON"


if __name__ == "__main__":
    rc = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(rc)
