#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free tests for rocprofv3 output-directory validation."""

import pytest


def forbid_application_launch(rocprofv3, monkeypatch):
    def unexpected_launch(*_args, **_kwargs):
        pytest.fail("application launched before output-directory validation")

    monkeypatch.setattr(rocprofv3.os, "execvpe", unexpected_launch)
    monkeypatch.setattr(rocprofv3.subprocess, "check_call", unexpected_launch)


def test_existing_file_output_directory_fails_before_launch(
    rocprofv3, tmp_path, capsys, monkeypatch
):
    output_path = tmp_path / "output"
    output_path.write_text("do not overwrite\n")
    forbid_application_launch(rocprofv3, monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.main(["--kernel-trace", "-d", str(output_path), "--", "/bin/true"])

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        f"[rocprofv3] Fatal error: Output path '{output_path}' already exists "
        "and is not a directory\n"
    )
    assert output_path.read_text() == "do not overwrite\n"


def test_non_directory_parent_fails_before_launch(
    rocprofv3, tmp_path, capsys, monkeypatch
):
    parent_path = tmp_path / "parent"
    parent_path.write_text("not a directory\n")
    output_path = parent_path / "output"
    forbid_application_launch(rocprofv3, monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.main(["--kernel-trace", "-d", str(output_path), "--", "/bin/true"])

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        f"[rocprofv3] Fatal error: Output path '{output_path}' has a "
        f"non-directory path component: '{parent_path}'\n"
    )


def test_output_directory_from_environment_fails_before_launch(
    rocprofv3, tmp_path, capsys, monkeypatch
):
    output_path = tmp_path / "output"
    output_path.write_text("not a directory\n")
    monkeypatch.setenv("ROCPROF_OUTPUT_PATH", str(output_path))
    forbid_application_launch(rocprofv3, monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.main(["--kernel-trace", "--", "/bin/true"])

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        f"[rocprofv3] Fatal error: Output path '{output_path}' already exists "
        "and is not a directory\n"
    )


def test_attach_output_directory_is_not_validated_in_launcher_context(
    rocprofv3, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").write_text("validity is determined in the target\n")
    args, app_args = rocprofv3.parse_arguments(
        ["--pid", "123", "--kernel-trace", "-d", "results"]
    )
    args = rocprofv3.get_args(args, rocprofv3.dotdict({}))

    class AttachLaunched(Exception):
        pass

    def attach_launched(*_args, **_kwargs):
        assert _kwargs["env"]["ROCPROF_OUTPUT_PATH"] == "results"
        raise AttachLaunched

    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, *_args: path)
    monkeypatch.setattr(rocprofv3.os, "execvpe", attach_launched)

    with pytest.raises(AttachLaunched):
        rocprofv3.run(app_args, args)


def test_relative_output_directory_is_anchored_to_launcher_cwd(
    rocprofv3, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    args, app_args = rocprofv3.parse_arguments(
        ["--kernel-trace", "-d", "results", "--", "/bin/true"]
    )
    args = rocprofv3.get_args(args, rocprofv3.dotdict({}))

    class ApplicationLaunched(Exception):
        pass

    def application_launched(*_args, **_kwargs):
        assert _kwargs["env"]["ROCPROF_OUTPUT_PATH"] == str(tmp_path / "results")
        raise ApplicationLaunched

    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, *_args: path)
    monkeypatch.setattr(rocprofv3.os, "execvpe", application_launched)

    with pytest.raises(ApplicationLaunched):
        rocprofv3.run(app_args, args)


def test_echo_skips_output_directory_validation(rocprofv3, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").write_text("echo mode does not create output\n")
    args, app_args = rocprofv3.parse_arguments(
        ["--kernel-trace", "--echo", "-d", "results", "--", "/bin/true"]
    )
    args = rocprofv3.get_args(args, rocprofv3.dotdict({}))
    forbid_application_launch(rocprofv3, monkeypatch)
    monkeypatch.setattr(rocprofv3, "resolve_library_path", lambda path, *_args: path)

    assert rocprofv3.run(app_args, args) is None
    assert "command: ['/bin/true']" in capsys.readouterr().out


@pytest.mark.parametrize("invalid_pass", [1, 2])
def test_multipass_output_directory_fails_before_launch(
    rocprofv3, tmp_path, capsys, monkeypatch, invalid_pass
):
    output_path = tmp_path / "output"
    output_path.mkdir()
    pass_output_path = output_path / f"pass_{invalid_pass}"
    pass_output_path.write_text("not a directory\n")
    forbid_application_launch(rocprofv3, monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.main(
            [
                "--pmc",
                "SQ_WAVES",
                "--pmc",
                "GRBM_COUNT",
                "-d",
                str(output_path),
                "--",
                "/bin/true",
            ]
        )

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        f"[rocprofv3] Fatal error: Output path '{pass_output_path}' already "
        "exists and is not a directory\n"
    )


def test_non_output_mpi_rank_skips_all_multipass_output_validation(
    rocprofv3, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output_path.mkdir()
    (output_path / "pass_2").write_text("not a directory in the launcher\n")
    monkeypatch.setenv("TEST_MPI_RANK", "1")
    monkeypatch.setenv("TEST_MPI_SIZE", "2")

    launches = []

    def application_launched(*args, **kwargs):
        launches.append((args, kwargs))
        return 0

    monkeypatch.setattr(rocprofv3.subprocess, "check_call", application_launched)

    result = rocprofv3.main(
        [
            "--pmc",
            "SQ_WAVES",
            "--pmc",
            "GRBM_COUNT",
            "--profile-mpi-ranks",
            "0",
            "--mpi-world-rank-variable",
            "TEST_MPI_RANK",
            "--mpi-world-size-variable",
            "TEST_MPI_SIZE",
            "-d",
            str(output_path),
            "--",
            "/bin/true",
        ]
    )

    assert result == 0
    assert len(launches) == 2


def test_non_output_mpi_rank_does_not_validate_output_directory(
    rocprofv3, tmp_path, monkeypatch
):
    output_path = tmp_path / "output"
    output_path.write_text("not a directory\n")
    args, app_args = rocprofv3.parse_arguments(
        [
            "--kernel-trace",
            "--profile-mpi-ranks",
            "0",
            "-d",
            str(output_path),
            "--",
            "/bin/true",
        ]
    )

    class ApplicationLaunched(Exception):
        pass

    def application_launched(*_args, **_kwargs):
        raise ApplicationLaunched

    monkeypatch.setattr(
        rocprofv3,
        "get_mpi_output_context",
        lambda _args: rocprofv3.dotdict(
            {
                "rank": 1,
                "size": 2,
                "rank_var": "RANK",
                "size_var": "SIZE",
                "should_produce_output": False,
            }
        ),
    )
    monkeypatch.setattr(rocprofv3.os, "execvpe", application_launched)

    with pytest.raises(ApplicationLaunched):
        rocprofv3.run(app_args, args)


def test_selected_mpi_rank_validates_output_directory(
    rocprofv3, tmp_path, capsys, monkeypatch
):
    output_path = tmp_path / "output"
    output_path.write_text("not a directory\n")
    monkeypatch.setenv("TEST_MPI_RANK", "0")
    monkeypatch.setenv("TEST_MPI_SIZE", "2")
    forbid_application_launch(rocprofv3, monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.main(
            [
                "--kernel-trace",
                "--profile-mpi-ranks",
                "0",
                "--mpi-world-rank-variable",
                "TEST_MPI_RANK",
                "--mpi-world-size-variable",
                "TEST_MPI_SIZE",
                "-d",
                str(output_path),
                "--",
                "/bin/true",
            ]
        )

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        f"[rocprofv3] Fatal error: Output path '{output_path}' already exists "
        "and is not a directory\n"
    )


def test_existing_output_directory_is_accepted(rocprofv3, tmp_path):
    rocprofv3.validate_output_directory(tmp_path)


def test_missing_output_directory_is_not_created_during_validation(rocprofv3, tmp_path):
    output_path = tmp_path / "new-output"

    rocprofv3.validate_output_directory(output_path)

    assert not output_path.exists()


@pytest.mark.parametrize("output_path", ["stdout", "STDOUT", "stderr", "STDERR"])
def test_output_stream_names_are_not_treated_as_paths(
    rocprofv3, tmp_path, monkeypatch, output_path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / output_path).write_text("ordinary file with a reserved stream name\n")

    rocprofv3.validate_output_directory(output_path)


def test_symlink_to_directory_is_accepted(rocprofv3, tmp_path):
    target_path = tmp_path / "target"
    target_path.mkdir()
    output_path = tmp_path / "output"
    output_path.symlink_to(target_path, target_is_directory=True)

    rocprofv3.validate_output_directory(output_path)


def test_symlink_to_file_is_rejected(rocprofv3, tmp_path, capsys):
    target_path = tmp_path / "target"
    target_path.write_text("not a directory\n")
    output_path = tmp_path / "output"
    output_path.symlink_to(target_path)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.validate_output_directory(output_path)

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        f"[rocprofv3] Fatal error: Output path '{output_path}' already exists "
        "and is not a directory\n"
    )


def test_dangling_symlink_is_rejected(rocprofv3, tmp_path, capsys):
    target_path = tmp_path / "missing"
    output_path = tmp_path / "output"
    output_path.symlink_to(target_path)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.validate_output_directory(output_path)

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        f"[rocprofv3] Fatal error: Output path '{output_path}' contains a "
        f"dangling symbolic link: '{output_path}'\n"
    )


def test_symlink_loop_reports_inspection_error(rocprofv3, tmp_path, capsys):
    output_path = tmp_path / "output"
    output_path.symlink_to(output_path)

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.validate_output_directory(output_path)

    assert exc_info.value.code == 1
    assert capsys.readouterr().err.startswith(
        f"[rocprofv3] Fatal error: Could not inspect output path '{output_path}':"
    )


@pytest.mark.parametrize("output_path", ["%pid%", "{pid}", "$ENV{USER}"])
def test_runtime_expanded_output_paths_are_left_to_cpp_validation(
    rocprofv3, tmp_path, monkeypatch, output_path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / output_path).write_text("unexpanded path\n")

    rocprofv3.validate_output_directory(output_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-x", __file__]))
