# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import pytest

from utils.native_tool import NativeTool


class TestNativeTool:
    def test_when_prebuilt_library_exists__founds_it(
        self,
        installed_lib_path: Path,
        installed_compute_path: Path,
        installed_rocprofv3_path: Path,
    ) -> None:
        lib_path = NativeTool().get_collector_library_path(
            installed_compute_path, installed_rocprofv3_path
        )
        assert lib_path == installed_lib_path

    def create_rocm_opt_file(self, rocm_path: Path, file_subpath: Path):
        file_path = rocm_path / file_subpath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("#!/bin/bash\n")
        return file_path

    @pytest.fixture
    def rocm_path(self, tmp_path: Path) -> Path:
        rocm_path = tmp_path / "opt" / "rocm"
        return rocm_path

    @pytest.fixture()
    def installed_rocprofv3_path(self, rocm_path: Path) -> Path:
        return self.create_rocm_opt_file(rocm_path, Path("bin/rocprofv3"))

    @pytest.fixture()
    def installed_compute_path(self, rocm_path: Path) -> Path:
        return self.create_rocm_opt_file(rocm_path, Path("bin/rocprof-compute"))

    @pytest.fixture()
    def installed_lib_path(self, rocm_path: Path) -> Path:
        return self.create_rocm_opt_file(
            rocm_path, Path("lib/rocprofiler-compute/librocprofiler-compute-tool.so")
        )
