# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from unittest.mock import patch

import pytest

from utils.native_tool_finder import NativeToolFinder


class TestNativeToolFinder:
    def test_when_incorrect_opt_path_provided__asserts(self, sources_path: Path):
        with pytest.raises(AssertionError):
            NativeToolFinder(sources_path, Path("incorrect_path"))

    def test_when_no_installed_collector_and_no_src_dir__throws(
        self,
        installed_sdk_tool_path: Path,
    ) -> None:
        with pytest.raises(RuntimeError):
            NativeToolFinder(
                Path("incorrect_src"), installed_sdk_tool_path
            ).get_collector_library_path()

    def test_when_run_from_opt__finds_prebuilt_native_collector(
        self,
        sources_path: Path,
        installed_lib_path: Path,
        installed_sdk_tool_path: Path,
    ) -> None:
        lib_path = NativeToolFinder(
            sources_path, installed_sdk_tool_path
        ).get_collector_library_path()
        assert lib_path == installed_lib_path

    def test_when_run_from_source_dir__builds_and_returns_collector(
        self, sources_path, installed_sdk_tool_path: Path
    ):
        def mock_build_collector(_: Path) -> None:
            self.__create_file(sources_path, Path(NativeToolFinder.lib_relative_path))

        with (
            patch.object(NativeToolFinder, "_generate_cmake", return_value=None),
            patch.object(
                NativeToolFinder,
                "_build_cmake",
                side_effect=mock_build_collector,
            ),
        ):
            lib_path = NativeToolFinder(
                sources_path, installed_sdk_tool_path
            ).get_collector_library_path()
        assert lib_path == sources_path / NativeToolFinder.lib_relative_path

    def test_when_run_from_source_dir_and_collector_not_found_after_build__throws(
        self, sources_path, installed_sdk_tool_path: Path
    ):
        with (
            patch.object(NativeToolFinder, "_generate_cmake", return_value=None),
            patch.object(NativeToolFinder, "_build_cmake", return_value=None),
        ):
            with pytest.raises(RuntimeError):
                NativeToolFinder(
                    sources_path, installed_sdk_tool_path
                ).get_collector_library_path()

    def test_when_run_from_source_dir_and_generation_fails__throws(
        self, installed_sdk_tool_path: Path, sources_path: Path
    ):
        lib_path = None
        with pytest.raises(RuntimeError):
            lib_path = NativeToolFinder(
                sources_path, installed_sdk_tool_path
            ).get_collector_library_path()
        assert lib_path == None

    @pytest.fixture
    def rocm_path(self, tmp_path: Path) -> Path:
        rocm_path = tmp_path / "opt" / "rocm"
        return rocm_path

    @pytest.fixture()
    def installed_sdk_tool_path(self, rocm_path: Path) -> Path:
        return self.__create_file(rocm_path, Path("lib/bin/librocprofiler-sdk-tool.so"))

    @pytest.fixture(params=["lib", "lib32", "lib64"])
    def installed_lib_path(
        self, rocm_path: Path, request: pytest.FixtureRequest
    ) -> Path:
        return self.__create_file(
            rocm_path,
            Path(f"{request.param}/rocprofiler-compute/{NativeToolFinder.lib_name}"),
        )

    @pytest.fixture
    def sources_path(self, tmp_path: Path) -> Path:
        sources_path = tmp_path / "src"
        sources_path.mkdir()
        return sources_path

    @pytest.fixture()
    def sources_lib_path(self, sources_path: Path) -> Path:
        return self.__create_file(
            sources_path, Path(NativeToolFinder.lib_relative_path)
        )

    def __create_file(self, rocm_path: Path, file_subpath: Path):
        file_path = rocm_path / file_subpath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("#!/bin/bash\n")
        return file_path
