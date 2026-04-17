# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from unittest.mock import patch

import pytest

from utils.native_tool import NativeTool


class TestNativeTool:
    def test_when_prebuilt_library_exists__founds_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._create_prebuilt_library(tmp_path)
        fake_script_path = self._make_script_path(tmp_path)

        with patch("utils.native_tool.capture_subprocess_output") as mock_build:
            NativeTool().get_collector_library_path(
                fake_script_path,
                rocprofiler_sdk_tool_path=tmp_path / "lib" / "sdk-tool.so",
            )

        mock_build.assert_not_called()

    def test_triggers_build_when_prebuilt_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No pre-built .so found; hipcc build is invoked."""

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        with (
            patch("utils.native_tool.tempfile.mkdtemp", return_value=str(build_dir)),
            patch(
                "utils.native_tool.capture_subprocess_output",
                return_value=(True, "ok"),
            ) as mock_build,
            patch("utils.native_tool.console_error") as mock_error,
            patch("utils.native_tool.console_debug"),
        ):
            fake_script = self._make_script_path(tmp_path)
            NativeTool().get_collector_library_path(
                fake_script, rocprofiler_sdk_tool_path=tmp_path / "lib" / "sdk-tool.so"
            )

        mock_build.assert_called_once()
        mock_error.assert_not_called()

    def test_reports_error_when_build_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """hipcc build fails; console_error is called exactly once."""
        fake_script = self._make_script_path(tmp_path)

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        with (
            patch("utils.native_tool.tempfile.mkdtemp", return_value=str(build_dir)),
            patch(
                "utils.native_tool.capture_subprocess_output",
                return_value=(False, "hipcc: command not found"),
            ),
            patch("utils.native_tool.console_error") as mock_error,
            patch("utils.native_tool.console_debug"),
        ):
            NativeTool().get_collector_library_path(
                fake_script, rocprofiler_sdk_tool_path=tmp_path / "lib" / "sdk-tool.so"
            )

        mock_error.assert_called_once()

    def test_build_command_references_sdk_library_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Build command contains -L flag pointing two levels up from sdk tool."""
        fake_script = self._make_script_path(tmp_path)

        sdk_tool_path = tmp_path / "rocm" / "lib" / "librocprofiler-sdk-tool.so"
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        with (
            patch("utils.native_tool.tempfile.mkdtemp", return_value=str(build_dir)),
            patch(
                "utils.native_tool.capture_subprocess_output",
                return_value=(True, "ok"),
            ) as mock_build,
            patch("utils.native_tool.console_debug"),
        ):
            NativeTool().get_collector_library_path(
                fake_script, rocprofiler_sdk_tool_path=sdk_tool_path
            )

        issued_command: list[str] = mock_build.call_args[0][0]
        sdk_lib_dir = str(sdk_tool_path.parent.parent)
        assert any(sdk_lib_dir in token for token in issued_command), (
            f"-L {sdk_lib_dir} not found in build command: {issued_command}"
        )

    def test_uses_current_directory_as_base_when_script_path_is_shallow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Script at filesystem root level; base path falls back to Path()."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        with (
            patch("utils.native_tool.tempfile.mkdtemp", return_value=str(build_dir)),
            patch(
                "utils.native_tool.capture_subprocess_output",
                return_value=(False, "build failed"),
            ),
            patch("utils.native_tool.console_error"),
            patch("utils.native_tool.console_debug"),
        ):
            # Must not raise regardless of base path resolution
            NativeTool().get_collector_library_path(
                Path("/rocprof-compute"),
                rocprofiler_sdk_tool_path=tmp_path / "lib" / "sdk-tool.so",
            )

    def test_build_output_path_inside_temp_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The -o flag in the build command points inside the mkdtemp directory."""
        fake_script = self._make_script_path(tmp_path)

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        with (
            patch("utils.native_tool.tempfile.mkdtemp", return_value=str(build_dir)),
            patch(
                "utils.native_tool.capture_subprocess_output",
                return_value=(True, "ok"),
            ) as mock_build,
            patch("utils.native_tool.console_debug"),
        ):
            NativeTool().get_collector_library_path(
                fake_script, rocprofiler_sdk_tool_path=tmp_path / "lib" / "sdk-tool.so"
            )

        issued_command: list[str] = mock_build.call_args[0][0]
        output_index = issued_command.index("-o")
        output_path = issued_command[output_index + 1]
        assert output_path.startswith(str(build_dir))
        assert output_path.endswith("librocprofiler-compute-tool.so")

    def _make_script_path(self, tmp_path: Path) -> Path:
        """Return a fake script path whose parents[2] resolves to tmp_path."""
        script = tmp_path / "base" / "bin" / "rocprof-compute"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/bash\n")
        return script

    def _create_prebuilt_library(self, base_path: Path) -> Path:
        """Create a pre-built .so in the expected glob location under base_path."""
        lib_dir = base_path / "lib" / "rocprofiler-compute"
        lib_dir.mkdir(parents=True, exist_ok=True)
        so_file = lib_dir / "librocprofiler-compute-tool.so"
        so_file.write_bytes(b"\x7fELF")  # minimal ELF magic bytes
        return so_file
