# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import pytest

from utils.native_tool import NativeTool


class TestNativeTool:
    def test_when_opt_lib_exists__founds_it(
        self,
        installed_lib_path: Path,
        installed_compute_path: Path,
        installed_rocprofv3_path: Path,
    ) -> None:
        lib_path = NativeTool().get_collector_library_path(
            installed_compute_path, installed_rocprofv3_path
        )
        assert lib_path == installed_lib_path

    def test_when_opt_lib_doesnt_exist_but_built_lib_exists__founds_it(
        self,
        installed_rocprofv3_path: Path,
        sources_compute_path: Path,
        sources_lib_path: Path,
    ):
        lib_path = NativeTool().get_collector_library_path(
            sources_compute_path, installed_rocprofv3_path
        )
        assert lib_path == sources_lib_path

    def test_when_incorrect_compute_path_provided__asserts(
        self, installed_rocprofv3_path: Path
    ):
        lib_path = None
        with pytest.raises(AssertionError):
            lib_path = NativeTool().get_collector_library_path(
                Path("incorrect_compute_path"), installed_rocprofv3_path
            )
        assert lib_path == None

    def test_when_incorrect_rocprofv3_path_provided__asserts(
        self, installed_compute_path: Path
    ):
        lib_path = None
        with pytest.raises(AssertionError):
            lib_path = NativeTool().get_collector_library_path(
                installed_compute_path, Path("incorrect_rocprofv3_path")
            )
        assert lib_path == None

    @pytest.fixture
    def rocm_path(self, tmp_path: Path) -> Path:
        rocm_path = tmp_path / "opt" / "rocm"
        return rocm_path

    @pytest.fixture()
    def installed_rocprofv3_path(self, rocm_path: Path) -> Path:
        return self.__create_file(rocm_path, Path("bin/rocprofv3.py"))

    @pytest.fixture()
    def installed_compute_path(self, rocm_path: Path) -> Path:
        return self.__create_file(rocm_path, Path("bin/rocprof-compute"))

    @pytest.fixture(params=["lib", "lib32", "lib64"])
    def installed_lib_path(
        self, rocm_path: Path, request: pytest.FixtureRequest
    ) -> Path:
        return self.__create_file(
            rocm_path,
            Path(f"{request.param}/rocprofiler-compute/librocprofiler-compute-tool.so"),
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
        return self.__create_file(
            sources_path, Path("lib/_build/bin/librocprofiler-compute-tool.so")
        )

    def __create_file(self, rocm_path: Path, file_subpath: Path):
        file_path = rocm_path / file_subpath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("#!/bin/bash\n")
        return file_path
