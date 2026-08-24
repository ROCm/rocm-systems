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
            NativeToolFinder(Path("incorrect_src")).get_artifact_path()

    def test_when_run_from_install_dir__finds_prebuilt_native_collector(
        self, rocm_install_dir: tuple[Path, Path]
    ) -> None:
        root_path, installed_lib_path = rocm_install_dir
        lib_path = NativeToolFinder(root_path).get_artifact_path()
        assert lib_path == installed_lib_path

    def test_when_run_from_source_dir__builds_and_returns_collector(self, sources_dir):
        root_path, built_lib_path = sources_dir

        def mock_build_collector() -> None:
            self.__create_file(built_lib_path)

        with patch.object(NativeToolFinder, "_generate_cmake", return_value=None):
            with patch.object(
                NativeToolFinder,
                "_build_cmake",
                side_effect=mock_build_collector,
            ):
                lib_path = NativeToolFinder(root_path).get_artifact_path()
        assert lib_path == built_lib_path

    def test_when_run_from_source_dir_and_collector_not_found_after_build__throws(
        self, sources_dir: tuple[Path, Path]
    ):
        root_path, built_lib_path = sources_dir
        built_lib_path.unlink()
        with patch.object(NativeToolFinder, "_generate_cmake", return_value=None):
            with patch.object(NativeToolFinder, "_build_cmake", return_value=None):
                with pytest.raises(RuntimeError):
                    NativeToolFinder(root_path).get_artifact_path()

    def test_when_run_from_source_dir_and_generation_fails__throws(
        self, sources_dir: tuple[Path, Path]
    ):
        root_path, _ = sources_dir
        lib_path = None
        with pytest.raises(RuntimeError):
            lib_path = NativeToolFinder(root_path).get_artifact_path()
        assert lib_path == None

    def test_when_installed_search_is_disabled__builds_instead(
        self, rocm_install_dir: tuple[Path, Path]
    ) -> None:
        root_path, installed_lib_path = rocm_install_dir
        built_lib_path = root_path / NativeToolFinder.lib_relative_path
        self.__create_file(built_lib_path)

        def mock_build_collector() -> None:
            pass

        with patch.object(NativeToolFinder, "_generate_cmake", return_value=None):
            with patch.object(
                NativeToolFinder,
                "_build_cmake",
                side_effect=mock_build_collector,
            ):
                lib_path = NativeToolFinder(
                    root_path, search_installed=False
                ).get_artifact_path()
        assert lib_path == built_lib_path
        assert lib_path != installed_lib_path

    def test_when_a_named_artifact_is_requested__finds_it_in_the_install_dir(
        self, tmp_path: Path
    ) -> None:
        artifact_name = "torch_trace_collector-2.9.0.so"
        rocm_path = tmp_path / "opt" / "rocm"
        root_path = rocm_path / "libexec" / "rocprofiler-compute"
        root_path.mkdir(parents=True, exist_ok=True)
        artifact_path = rocm_path / "lib" / "rocprofiler-compute" / artifact_name
        self.__create_file(artifact_path)

        finder = NativeToolFinder(root_path, artifact_name=artifact_name)
        assert finder.get_artifact_path() == artifact_path

    def test_when_a_named_artifact_is_built__is_found_under_the_build_path(
        self, tmp_path: Path
    ) -> None:
        artifact_name = "torch_trace_collector-2.9.0.so"
        root_path = tmp_path / "src"
        build_path = tmp_path / "cache" / "_build"
        artifact_path = (
            build_path / NativeToolFinder.sources_bin_subdir_name / artifact_name
        )
        self.__create_file(artifact_path)

        with patch.object(NativeToolFinder, "_generate_cmake", return_value=None):
            with patch.object(NativeToolFinder, "_build_cmake", return_value=None):
                finder = NativeToolFinder(
                    root_path,
                    artifact_name=artifact_name,
                    build_path=build_path,
                )
                assert finder.get_artifact_path() == artifact_path

    def test_when_reuse_is_requested__an_existing_artifact_skips_cmake(
        self, tmp_path: Path
    ) -> None:
        artifact_name = "torch_trace_collector-2.9.0.so"
        root_path = tmp_path / "src"
        build_path = tmp_path / "cache" / "_build"
        artifact_path = (
            build_path / NativeToolFinder.sources_bin_subdir_name / artifact_name
        )
        self.__create_file(artifact_path)

        finder = NativeToolFinder(
            root_path,
            artifact_name=artifact_name,
            build_path=build_path,
            reuse_built_artifact=True,
        )
        with patch.object(NativeToolFinder, "_generate_cmake") as generate:
            with patch.object(NativeToolFinder, "_build_cmake") as build:
                assert finder.get_artifact_path() == artifact_path
        generate.assert_not_called()
        build.assert_not_called()

    def test_when_reuse_is_requested__a_missing_artifact_still_builds(
        self, tmp_path: Path
    ) -> None:
        artifact_name = "torch_trace_collector-2.9.0.so"
        root_path = tmp_path / "src"
        build_path = tmp_path / "cache" / "_build"
        artifact_path = (
            build_path / NativeToolFinder.sources_bin_subdir_name / artifact_name
        )

        finder = NativeToolFinder(
            root_path,
            artifact_name=artifact_name,
            build_path=build_path,
            reuse_built_artifact=True,
        )
        with patch.object(NativeToolFinder, "_generate_cmake", return_value=None):
            with patch.object(
                NativeToolFinder,
                "_build_cmake",
                side_effect=lambda: self.__create_file(artifact_path),
            ):
                assert finder.get_artifact_path() == artifact_path

    def test_cmake_commands_carry_the_configure_options_and_target(
        self, tmp_path: Path
    ) -> None:
        build_path = tmp_path / "_build"
        finder = NativeToolFinder(
            tmp_path / "src",
            artifact_name="torch_trace_collector-tag.so",
            build_target="torch_trace_collector-tag",
            build_path=build_path,
            configure_options=("-DTORCH_TRACE_PYTHON=/usr/bin/python3",),
            cmake_executable="/fake/cmake",
        )

        with patch(
            "utils.native_tool_finder.capture_subprocess_output",
            return_value=(True, ""),
        ) as execute:
            finder._generate_cmake()
            finder._build_cmake()

        generate_command, build_command = [
            call.args[0] for call in execute.call_args_list
        ]
        assert generate_command == [
            "/fake/cmake",
            "-S",
            str(tmp_path / "src" / NativeToolFinder.sources_dir_name),
            "-B",
            str(build_path),
            "-DTORCH_TRACE_PYTHON=/usr/bin/python3",
        ]
        assert build_command[:3] == ["/fake/cmake", "--build", str(build_path)]
        assert build_command[-2:] == ["--target", "torch_trace_collector-tag"]

    def test_the_collector_build_command_carries_no_target(
        self, tmp_path: Path
    ) -> None:
        finder = NativeToolFinder(tmp_path / "src")
        with patch(
            "utils.native_tool_finder.capture_subprocess_output",
            return_value=(True, ""),
        ) as execute:
            finder._build_cmake()
        assert "--target" not in execute.call_args.args[0]

    def test_when_a_command_fails__the_error_carries_the_output(
        self, tmp_path: Path
    ) -> None:
        finder = NativeToolFinder(tmp_path / "src")
        with patch(
            "utils.native_tool_finder.capture_subprocess_output",
            return_value=(False, "CMake Error: Could not find Torch\n"),
        ):
            with pytest.raises(RuntimeError, match="Could not find Torch"):
                finder._generate_cmake()

    def test_importing_the_finder_does_not_import_torch(self) -> None:
        """The profile path imports this module, which must stay free of torch."""
        import os
        import subprocess
        import sys

        import utils.native_tool_finder as finder_module

        src_root = Path(finder_module.__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\nimport utils.native_tool_finder\n"
                "assert 'torch' not in sys.modules, 'torch imported'\n",
            ],
            env={**os.environ, "PYTHONPATH": str(src_root)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

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
