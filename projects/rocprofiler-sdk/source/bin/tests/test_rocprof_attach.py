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

"""GPU-free unit tests for the rocprof-attach launcher."""

import os
import sys

import pytest


class _FakeFunction:
    def __init__(self):
        self.calls = []
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        self.calls.append(args)
        return 0


class _FakeAttachLibrary:
    def __init__(self):
        self.rocattach_attach = _FakeFunction()
        self.rocattach_attach_tree = _FakeFunction()
        self.rocattach_detach = _FakeFunction()
        self.rocattach_detach_tree = _FakeFunction()


def test_default_attach_library_uses_runtime_soname(
    rocprof_attach, monkeypatch, tmp_path
):
    tool_library = tmp_path / "tool.so"
    tool_library.touch()

    loaded_paths = []
    fake_library = _FakeAttachLibrary()

    def load_library(path):
        loaded_paths.append(path)
        return fake_library

    monkeypatch.setattr(rocprof_attach.ctypes, "CDLL", load_library)
    monkeypatch.setattr(rocprof_attach.signal, "signal", lambda *_: None)
    monkeypatch.setattr(rocprof_attach.time, "sleep", lambda *_: None)
    monkeypatch.setenv("ROCPROF_ATTACH_TOOL_LIBRARY", "restore-after-test")

    result = rocprof_attach.main(
        [
            "--attach",
            "123",
            "--attach-tool-library",
            str(tool_library),
            "--attach-duration-msec",
            "0",
            "--attach-children=false",
        ]
    )

    expected_library = os.path.join(
        rocprof_attach.ROCM_DIR,
        "lib",
        f"librocprofiler-sdk-rocattach.so.{rocprof_attach.ROCPROFILER_SDK_SOVERSION}",
    )
    assert result == 0
    assert rocprof_attach.ROCPROFILER_SDK_SOVERSION.isdigit()
    assert loaded_paths == [expected_library]
    assert fake_library.rocattach_attach.calls == [(123,)]
    assert fake_library.rocattach_detach.calls == [(123,)]
    assert not fake_library.rocattach_attach_tree.calls
    assert not fake_library.rocattach_detach_tree.calls


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
