# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for pytest helper behavior.
"""

from __future__ import annotations
import pytest
import conftest as pytest_config
from conftest import RocprofsysTest
from rocprofsys import GPUInfo

pytestmark = [pytest.mark.pytest_impl]


@pytest.mark.class_name("gpu-info")
class TestGPUInfo(RocprofsysTest):
    def test_gfx1250_uses_gfx1250_counter_set(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1250"],
            device_count=1,
            categories={"instinct"},
        )

        assert (
            gpu_info.rocm_events_for_test
            == "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TX_VCA_VCA_BUSY"
        )
        assert gpu_info.counter_names == [
            "GRBM_COUNT",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "TX_VCA_VCA_BUSY",
        ]
        assert gpu_info.expected_counter_files == [
            "rocprof-device-[0-9]-GRBM_COUNT.txt",
            "rocprof-device-[0-9]-SQ_WAVES.txt",
            "rocprof-device-[0-9]-SQ_INSTS_VALU.txt",
            "rocprof-device-[0-9]-TX_VCA_VCA_BUSY.txt",
        ]

    def test_mi300_and_later_keep_ta_ta_busy(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx942"],
            device_count=1,
            categories={"instinct"},
        )

        assert (
            gpu_info.rocm_events_for_test
            == "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY"
        )
        assert gpu_info.counter_names == [
            "GRBM_COUNT",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "TA_TA_BUSY",
        ]

    def test_non_mi300_non_gfx1250_keep_single_counter(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1201"],
            device_count=1,
            categories={"radeon"},
        )

        assert gpu_info.rocm_events_for_test == "SQ_WAVES"
        assert gpu_info.counter_names == ["SQ_WAVES"]


@pytest.mark.class_name("perfetto-merge")
class TestPerfettoMergeRecovery(RocprofsysTest):
    def test_recover_merged_perfetto_trace_runs_merge_script(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        merged_trace = output_dir / "merged.proto"
        rank_trace = output_dir / "perfetto-trace-0.proto"
        rank_trace.write_text("rank trace")

        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        merge_script = script_dir / "rocprof-sys-merge-output.sh"
        merge_script.write_text(
            "#!/bin/sh\n" 'cat "$1"/perfetto-trace-0.proto > "$1"/merged.proto\n'
        )
        merge_script.chmod(0o755)

        monkeypatch.setenv("ROCPROFSYS_SCRIPT_PATH", str(script_dir))
        monkeypatch.setattr(pytest_config.time, "sleep", lambda _: None)

        result = pytest_config._recover_merged_perfetto_trace(merged_trace, tmp_path)

        assert result.ok
        assert merged_trace.read_text() == "rank trace"
        assert "Recovery succeeded" in result.message
        assert "returncode=0" in result.message

    def test_recover_merged_perfetto_trace_skips_non_merged_trace(self, tmp_path):
        result = pytest_config._recover_merged_perfetto_trace(
            tmp_path / "perfetto-trace-0.proto", tmp_path
        )

        assert not result.ok
        assert "not merged.proto" in result.message

    def test_recover_merged_perfetto_trace_requires_rank_traces(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(pytest_config.time, "sleep", lambda _: None)

        result = pytest_config._recover_merged_perfetto_trace(
            tmp_path / "merged.proto", tmp_path
        )

        assert not result.ok
        assert "no perfetto-trace-*.proto files found" in result.message

    def test_recover_merged_perfetto_trace_requires_executable_merge_script(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "perfetto-trace-0.proto").write_text("rank trace")

        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        merge_script = script_dir / "rocprof-sys-merge-output.sh"
        merge_script.write_text("#!/bin/sh\nexit 0\n")
        merge_script.chmod(0o644)

        monkeypatch.setenv("ROCPROFSYS_SCRIPT_PATH", str(script_dir))
        monkeypatch.setattr(pytest_config.time, "sleep", lambda _: None)

        result = pytest_config._recover_merged_perfetto_trace(
            output_dir / "merged.proto", tmp_path
        )

        assert not result.ok
        assert "rocprof-sys-merge-output.sh not found" in result.message

    def test_recover_merged_perfetto_trace_handles_merge_timeout(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "perfetto-trace-0.proto").write_text("rank trace")

        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        merge_script = script_dir / "rocprof-sys-merge-output.sh"
        merge_script.write_text("#!/bin/sh\nsleep 10\n")
        merge_script.chmod(0o755)

        monkeypatch.setenv("ROCPROFSYS_SCRIPT_PATH", str(script_dir))
        monkeypatch.setattr(pytest_config.time, "sleep", lambda _: None)
        monkeypatch.setattr(pytest_config, "_MERGE_SCRIPT_TIMEOUT_S", 0.01)

        result = pytest_config._recover_merged_perfetto_trace(
            output_dir / "merged.proto", tmp_path
        )

        assert not result.ok
        assert "timed out" in result.message

    def test_recover_merged_perfetto_trace_reports_merge_failure(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "perfetto-trace-0.proto").write_text("rank trace")

        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        merge_script = script_dir / "rocprof-sys-merge-output.sh"
        merge_script.write_text("#!/bin/sh\nexit 3\n")
        merge_script.chmod(0o755)

        monkeypatch.setenv("ROCPROFSYS_SCRIPT_PATH", str(script_dir))
        monkeypatch.setattr(pytest_config.time, "sleep", lambda _: None)

        result = pytest_config._recover_merged_perfetto_trace(
            output_dir / "merged.proto", tmp_path
        )

        assert not result.ok
        assert "Recovery failed" in result.message
        assert "returncode=3" in result.message

    def test_recover_merged_perfetto_trace_reports_missing_output_after_success(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "perfetto-trace-0.proto").write_text("rank trace")

        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        merge_script = script_dir / "rocprof-sys-merge-output.sh"
        merge_script.write_text("#!/bin/sh\nexit 0\n")
        merge_script.chmod(0o755)

        monkeypatch.setenv("ROCPROFSYS_SCRIPT_PATH", str(script_dir))
        monkeypatch.setattr(pytest_config.time, "sleep", lambda _: None)

        result = pytest_config._recover_merged_perfetto_trace(
            output_dir / "merged.proto", tmp_path
        )

        assert not result.ok
        assert "exited 0 but produced no merged.proto" in result.message
        assert "returncode=0" in result.message
