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


def test_multipass_output_directory_fails_before_launch(
    rocprofv3, tmp_path, capsys, monkeypatch
):
    output_path = tmp_path / "output"
    output_path.mkdir()
    pass_output_path = output_path / "pass_1"
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
        rocprofv3, "get_mpi_rank_and_size", lambda *_args: (1, 2, "RANK", "SIZE")
    )
    monkeypatch.setattr(rocprofv3, "should_rank_provide_output", lambda *_args: False)
    monkeypatch.setattr(rocprofv3.os, "execvpe", application_launched)

    with pytest.raises(ApplicationLaunched):
        rocprofv3.run(app_args, args)


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


def test_literal_prefix_of_runtime_expanded_path_is_validated(
    rocprofv3, tmp_path, capsys
):
    parent_path = tmp_path / "parent"
    parent_path.write_text("not a directory\n")
    output_path = parent_path / "%pid%"

    with pytest.raises(SystemExit) as exc_info:
        rocprofv3.validate_output_directory(output_path)

    assert exc_info.value.code == 1
    assert capsys.readouterr().err == (
        f"[rocprofv3] Fatal error: Output path '{output_path}' has a "
        f"non-directory path component: '{parent_path}'\n"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-x", __file__]))
