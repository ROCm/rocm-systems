"""
Validation for signal handler integration tests.

Both good and bad cases must produce VALID, COMPLETE JSON output.
The profiler intercepts signals to flush cleanly — truncated JSON is a bug.

Good case: exit_marker present in JSON (proves clean atexit finalization).
Bad case: valid JSON with >10 marker entries (proves profiler flushed before death).
"""

import json
import os
import glob


def find_json_files(output_dir):
    """Find all JSON result files in the output directory."""
    pattern = os.path.join(output_dir, "*.json")
    return sorted(glob.glob(pattern))


def load_json(path):
    """Load and validate a JSON file. Raises on truncated/corrupt JSON."""
    with open(path, "r") as f:
        return json.load(f)


def count_markers_in_json(data):
    """Count marker events in the rocprofv3 JSON structure."""
    count = 0
    try:
        for tool_entry in data.get("rocprofiler-sdk-tool", []):
            buffer_records = tool_entry.get("buffer_records", {})
            count += len(buffer_records.get("marker_api", []))
            callback_records = tool_entry.get("callback_records", {})
            count += len(callback_records.get("marker_api", []))
    except (AttributeError, TypeError):
        pass
    return count


def test_output_files_exist(output_dir, mode, process_type):
    """JSON output files must exist."""
    files = find_json_files(output_dir)
    assert len(files) > 0, f"No JSON output files found in {output_dir}"


def test_json_is_valid(output_dir, mode, process_type):
    """All JSON output files must be valid (not truncated)."""
    files = find_json_files(output_dir)
    assert len(files) > 0, f"No JSON files in {output_dir}"

    for path in files:
        try:
            load_json(path)
        except json.JSONDecodeError as e:
            assert False, (
                f"JSON file is truncated/corrupt: {path}\n"
                f"Error: {e}\n"
                f"This means the profiler did not flush cleanly before process death."
            )


def test_good_case_exit_marker(output_dir, mode, process_type):
    """Good case: mode-specific exit_marker must be present."""
    if mode != "good":
        return

    files = find_json_files(output_dir)
    assert len(files) > 0

    expected_marker = f"exit_marker parent {process_type}"

    all_content = ""
    for path in files:
        with open(path, "r") as f:
            all_content += f.read()

    assert expected_marker in all_content, (
        f"Expected marker '{expected_marker}' not found in output files in {output_dir}. "
        f"App did not exit cleanly (atexit finalization failed)."
    )


def test_bad_case_markers_flushed(output_dir, mode, process_type):
    """Bad case: JSON must contain >10 marker entries (proves profiler flushed)."""
    if mode != "bad":
        return

    files = find_json_files(output_dir)
    assert len(files) > 0

    total_markers = 0
    for path in files:
        data = load_json(path)
        total_markers += count_markers_in_json(data)

    assert total_markers > 10, (
        f"Expected >10 marker events in JSON output, got {total_markers}. "
        f"Profiler may not have flushed marker data before signal death. "
        f"Files: {files}"
    )
