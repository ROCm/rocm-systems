# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from unittest.mock import patch

import pytest

from utils.native_tool_finder import NativeToolFinder


class TestNativeToolFinder:
    def test_when_no_installed_collector_and_no_src_dir__throws(
        self,
    ) -> None:
        with pytest.raises(RuntimeError):
            NativeToolFinder(Path("incorrect_src")).get_collector_library_path()

    def test_when_run_from_install_dir__finds_prebuilt_native_collector(
        self, rocm_install_dir: tuple[Path, Path]
    ) -> None:
        root_path, installed_lib_path = rocm_install_dir
        lib_path = NativeToolFinder(root_path).get_collector_library_path()
        assert lib_path == installed_lib_path

    def test_when_run_from_source_dir__builds_and_returns_collector(self, sources_dir):
        root_path, built_lib_path = sources_dir

        def mock_build_collector(_: Path) -> None:
            self.__create_file(built_lib_path)

        with patch.object(NativeToolFinder, "_generate_cmake", return_value=None):
            with patch.object(
                NativeToolFinder,
                "_build_cmake",
                side_effect=mock_build_collector,
            ):
                lib_path = NativeToolFinder(root_path).get_collector_library_path()
        assert lib_path == built_lib_path

    def test_when_run_from_source_dir_and_collector_not_found_after_build__throws(
        self, sources_dir: tuple[Path, Path]
    ):
        root_path, built_lib_path = sources_dir
        built_lib_path.unlink()
        with patch.object(NativeToolFinder, "_generate_cmake", return_value=None):
            with patch.object(NativeToolFinder, "_build_cmake", return_value=None):
                with pytest.raises(RuntimeError):
                    NativeToolFinder(root_path).get_collector_library_path()

    def test_when_run_from_source_dir_and_generation_fails__throws(
        self, sources_dir: tuple[Path, Path]
    ):
        root_path, _ = sources_dir
        lib_path = None
        with pytest.raises(RuntimeError):
            lib_path = NativeToolFinder(root_path).get_collector_library_path()
        assert lib_path == None

    @pytest.fixture(params=["lib", "lib32", "lib64"])
    def rocm_install_dir(
        self, tmp_path: Path, request: pytest.FixtureRequest
    ) -> tuple[Path, Path]:
        rocm_path = tmp_path / "opt" / "rocm"
        compute_root_path = rocm_path / "libexec" / "rocprofiler-compute"
        compute_root_path.mkdir(parents=True, exist_ok=True)
        lib_path = (
            rocm_path
            / f"{request.param}/rocprofiler-compute/{NativeToolFinder.lib_name}"
        )
        self.__create_file(lib_path)
        return compute_root_path, lib_path

    @pytest.fixture
    def sources_dir(self, tmp_path: Path) -> tuple[Path, Path]:
        sources_path = tmp_path / "src"
        sources_path.mkdir(parents=True, exist_ok=True)
        lib_path = sources_path / Path(NativeToolFinder.lib_relative_path)
        self.__create_file(lib_path)
        return sources_path, lib_path

    def __create_file(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("#!/bin/bash\n")
        return file_path
