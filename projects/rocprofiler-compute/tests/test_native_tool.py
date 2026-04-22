# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from unittest.mock import patch

import pytest

from utils.native_tool import NativeTool


class TestNativeTool:
    def test_when_run_from_opt__finds_prebuilt_native_collector(
        self,
        installed_lib_path: Path,
        installed_compute_path: Path,
        installed_sdk_tool_path: Path,
    ) -> None:
        lib_path = NativeTool().get_collector_library_path(
            installed_compute_path, installed_sdk_tool_path
        )
        assert lib_path == installed_lib_path

    @pytest.mark.skip()
    def test_when_run_from_opt_and_doesnt_find_prebuilt_native_collector__throws(
        self, installed_compute_path: Path, installed_sdk_tool_path
    ):
        with pytest.raises(RuntimeError):
            NativeTool().get_collector_library_path(
                installed_compute_path, installed_sdk_tool_path
            )

    def test_when_opt_lib_doesnt_exist_but_built_lib_exists__finds_it(
        self,
        installed_sdk_tool_path: Path,
        sources_compute_path: Path,
        sources_lib_path: Path,
    ):
        lib_path = NativeTool().get_collector_library_path(
            sources_compute_path, installed_sdk_tool_path
        )
        assert lib_path == sources_lib_path

    def test_when_incorrect_compute_path_provided__asserts(
        self, installed_sdk_tool_path: Path
    ):
        lib_path = None
        with pytest.raises(AssertionError):
            lib_path = NativeTool().get_collector_library_path(
                Path("incorrect_compute_path"), installed_sdk_tool_path
            )
        assert lib_path is None

    def test_when_incorrect_opt_path_provided__asserts(
        self, installed_compute_path: Path
    ):
        lib_path = None
        with pytest.raises(AssertionError):
            lib_path = NativeTool().get_collector_library_path(
                installed_compute_path, Path("incorrect_rocm_path")
            )
        assert lib_path is None

    def test_when_no_lib_exists__builds_it(
        self, sources_path, installed_sdk_tool_path: Path, sources_compute_path: Path
    ):
        def mock_build_collector(_src_path: Path) -> None:
            self.__create_file(sources_path, Path(NativeTool.lib_relative_path))

        with (
            patch.object(NativeTool, "_generate_cmake_project", return_value=True),
            patch.object(
                NativeTool, "_build_cmake_project", side_effect=mock_build_collector
            ),
        ):
            lib_path = NativeTool().get_collector_library_path(
                sources_compute_path, installed_sdk_tool_path
            )
        assert lib_path == sources_path / NativeTool.lib_relative_path

    @pytest.fixture
    def rocm_path(self, tmp_path: Path) -> Path:
        rocm_path = tmp_path / "opt" / "rocm"
        return rocm_path

    @pytest.fixture()
    def installed_sdk_tool_path(self, rocm_path: Path) -> Path:
        return self.__create_file(rocm_path, Path("lib/bin/librocprofiler-sdk-tool.so"))

    @pytest.fixture()
    def installed_compute_path(self, rocm_path: Path) -> Path:
        return self.__create_file(rocm_path, Path("bin/rocprof-compute"))

    @pytest.fixture(params=["lib", "lib32", "lib64"])
    def installed_lib_path(
        self, rocm_path: Path, request: pytest.FixtureRequest
    ) -> Path:
        return self.__create_file(
            rocm_path,
            Path(f"{request.param}/rocprofiler-compute/{NativeTool.lib_name}"),
        )

    @pytest.fixture
    def sources_path(self, tmp_path: Path) -> Path:
        sources_path = tmp_path / "src"
        return sources_path

    @pytest.fixture()
    def sources_compute_path(self, sources_path: Path) -> Path:
        return self.__create_file(sources_path, Path("rocprof-compute"))

    @pytest.fixture()
    def sources_lib_path(self, sources_path: Path) -> Path:
        return self.__create_file(sources_path, Path(NativeTool.lib_relative_path))

    def __create_file(self, rocm_path: Path, file_subpath: Path):
        file_path = rocm_path / file_subpath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("#!/bin/bash\n")
        return file_path
