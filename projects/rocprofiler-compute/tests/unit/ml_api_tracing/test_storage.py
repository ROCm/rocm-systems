# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API trace file retention."""

import gzip
from pathlib import Path

import pytest

import ml_api_tracing.storage as storage

COUNTER_BYTES = gzip.compress(b"Kernel_Name,Counter_Value\naddmm_kernel,7\n", mtime=0)
MARKER_PAYLOAD = (
    'Function,Thread_Id\n"nn.Module.Linear.forward:train.py:19'
    '|seqNr=3|tid=4|ftid=5|scope=0|args=(""caf\u00e9"",)|torch",4\n'
).encode("utf-8")
MARKER_BYTES = gzip.compress(MARKER_PAYLOAD, mtime=0)


@pytest.fixture
def storage_events(monkeypatch):
    events = []
    copyfile = storage.shutil.copyfile

    def record_copy(source, destination):
        events.append(("copy", source, destination))
        return copyfile(source, destination)

    monkeypatch.setattr(storage.shutil, "copyfile", record_copy)
    monkeypatch.setattr(
        storage, "console_log", lambda *args: events.append(("log", *args))
    )
    return events


def trace_paths(workload_dir):
    source_dir = workload_dir / "out" / "pmc_1"
    source_dir.mkdir(parents=True)
    return (
        workload_dir / "results_pmc_perf_2.csv.gz",
        source_dir / "pmc_perf_2_marker_api_trace.csv.gz",
        workload_dir / "ml_api_trace_pmc_perf_2_counter_collection.csv.gz",
        workload_dir / "ml_api_trace_pmc_perf_2_marker_api_trace.csv.gz",
    )


def test_storage_import_preserves_profile_dependency_boundaries(assert_profile_import):
    assert_profile_import("ml_api_tracing.storage", Path(storage.__file__).parents[1])


def test_save_ml_api_trace_inputs_preserves_bytes_names_and_copy_log_order(
    tmp_path, storage_events
):
    source_counter, source_marker, saved_counter, saved_marker = trace_paths(tmp_path)
    source_counter.write_bytes(COUNTER_BYTES)
    source_marker.write_bytes(MARKER_BYTES)

    storage.save_ml_api_trace_inputs(
        str(tmp_path), "pmc_perf_2", source_counter, source_marker
    )

    assert saved_counter.read_bytes() == COUNTER_BYTES
    assert saved_marker.read_bytes() == MARKER_BYTES
    assert gzip.decompress(saved_marker.read_bytes()) == MARKER_PAYLOAD
    assert source_counter.read_bytes() == COUNTER_BYTES
    assert source_marker.read_bytes() == MARKER_BYTES
    assert storage_events == [
        ("copy", source_counter, saved_counter),
        ("copy", source_marker, saved_marker),
        (
            "log",
            "ml api trace",
            "Copied counter collection and marker trace files "
            "to the workload directory.",
        ),
        ("log", "Counter Collection: ", str(saved_counter)),
        ("log", "Marker API Trace: ", str(saved_marker)),
    ]


def test_save_ml_api_trace_inputs_uses_explicit_marker_source(tmp_path, storage_events):
    source_counter, _, saved_counter, saved_marker = trace_paths(tmp_path)
    source_marker = tmp_path / "retained inputs" / "marker.csv.gz"
    source_marker.parent.mkdir()
    source_counter.write_bytes(COUNTER_BYTES)
    source_marker.write_bytes(MARKER_BYTES)

    storage.save_ml_api_trace_inputs(
        str(tmp_path), "pmc_perf_2", source_counter, source_marker
    )

    assert saved_counter.read_bytes() == COUNTER_BYTES
    assert saved_marker.read_bytes() == MARKER_BYTES
    assert gzip.decompress(saved_marker.read_bytes()) == MARKER_PAYLOAD
    assert source_marker.read_bytes() == MARKER_BYTES
    assert storage_events[:2] == [
        ("copy", source_counter, saved_counter),
        ("copy", source_marker, saved_marker),
    ]


@pytest.mark.parametrize("missing_input", ["counter", "marker"])
def test_save_ml_api_trace_inputs_preserves_missing_input_partial_state(
    tmp_path, storage_events, missing_input
):
    source_counter, source_marker, saved_counter, saved_marker = trace_paths(tmp_path)
    if missing_input != "counter":
        source_counter.write_bytes(COUNTER_BYTES)
    if missing_input != "marker":
        source_marker.write_bytes(MARKER_BYTES)

    with pytest.raises(FileNotFoundError) as failure:
        storage.save_ml_api_trace_inputs(
            str(tmp_path), "pmc_perf_2", source_counter, source_marker
        )

    missing_path = source_counter if missing_input == "counter" else source_marker
    assert failure.value.filename == str(missing_path)
    expected_events = [("copy", source_counter, saved_counter)]
    if missing_input == "marker":
        expected_events.append(("copy", source_marker, saved_marker))
        assert saved_counter.read_bytes() == COUNTER_BYTES
    else:
        assert not saved_counter.exists()
    assert not saved_marker.exists()
    assert storage_events == expected_events
