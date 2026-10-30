# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for layout_names.py, the kernel folder naming rules."""

import shutil
import subprocess

import pytest

from rocprof_compute_analyze import layout_names
from rocprof_compute_analyze.layout_names import (
    demangle_kernel_names,
    kernel_folder_name,
    kernel_function_name,
    kernel_index_width,
)


@pytest.mark.parametrize(
    ("kernel_name", "expected"),
    [
        ("vecCopy(double*, double*, double*, int, int) (.kd)", "vecCopy"),
        ("vecCopy(double*, double*, double*, int, int)", "vecCopy"),
        ("scale<float, 4>(float*, float)", "scale"),
        (
            "void at::native::vectorized_elementwise_kernel<4, char>(int)",
            "vectorized_elementwise_kernel",
        ),
        ("ns::inner::run(int)", "run"),
        ("foo(std::vector<int>, int)", "foo"),
        ("foo<std::pair<int, std::vector<float>>>(int) (.kd)", "foo"),
        ("(anonymous namespace)::helper(int) (.kd)", "helper"),
        ("__amd_rocclr_copyBufferAligned.kd", "__amd_rocclr_copyBufferAligned"),
        ("plainKernel", "plainKernel"),
        ("Foo::bar(int) const", "bar"),
        ("void Foo::bar<int>(int) const (.kd)", "bar"),
        ("Foo::bar() const", "bar"),
        ("Foo::bar(int)&&", "bar"),
        ("foo(int) [clone .isra.0]", "foo"),
        ("my_operator_kernel(int)", "my_operator_kernel"),
    ],
)
def test_function_name_drops_suffix_templates_parameters_and_namespaces(
    kernel_name, expected
):
    """The folder name is the bare function name, with the signature gone."""
    assert kernel_function_name(kernel_name) == expected


@pytest.mark.parametrize(
    ("kernel_name", "expected"),
    [
        ("a.b<int>(x) (.kd)", "a_b"),
        ("operator+(int)", "operator_"),
        ("operator<(int)", "operator_"),
        ("operator()(int)", "operator__"),
        ("Functor::operator()(int) const", "operator__"),
        ("operator new[](unsigned long)", "operator_new__"),
        ("my-kernel", "my_kernel"),
        ("unicodé_kernel", "unicod__kernel"),
    ],
)
def test_characters_outside_the_safe_set_become_underscores(kernel_name, expected):
    """Every character outside [A-Za-z0-9_] becomes an underscore."""
    assert kernel_function_name(kernel_name) == expected


def test_long_names_are_cut_to_64_characters():
    """A name longer than 64 characters keeps its first 64."""
    long_name = "k" * 100 + "(int)"

    assert kernel_function_name(long_name) == "k" * 64


def test_name_with_nothing_left_is_called_unnamed():
    """A name that is only a parameter list has no function name to show."""
    assert kernel_function_name("(int)") == "unnamed"


def test_mangled_names_stay_raw_when_c_filt_is_missing(monkeypatch):
    """Without c++filt the mangled name is used as it is."""

    def missing_demangler(*args, **kwargs):
        raise FileNotFoundError("c++filt")

    monkeypatch.setattr(layout_names.subprocess, "run", missing_demangler)

    assert demangle_kernel_names(["_Z6vecCopyPdS_S_iii"]) == {}
    assert kernel_function_name("_Z6vecCopyPdS_S_iii") == "_Z6vecCopyPdS_S_iii"


def test_a_name_c_filt_returns_unchanged_maps_to_itself(monkeypatch):
    """c++filt echoes a name it cannot demangle, so that name maps to itself."""

    def echoing_demangler(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=kwargs["input"], stderr=""
        )

    monkeypatch.setattr(layout_names.subprocess, "run", echoing_demangler)

    assert demangle_kernel_names(["_Zzzz"]) == {"_Zzzz": "_Zzzz"}


def test_only_mangled_names_reach_c_filt(monkeypatch):
    """Demangled names are never sent to c++filt, and it runs once."""
    calls = []

    def recording_demangler(command, **kwargs):
        calls.append(kwargs["input"])
        return subprocess.CompletedProcess(
            command, 0, stdout="demangled_one\ndemangled_two\n", stderr=""
        )

    monkeypatch.setattr(layout_names.subprocess, "run", recording_demangler)

    result = demangle_kernel_names(["plain", "_Z3fooi", "_Z3bari", "_Z3fooi"])

    assert calls == ["_Z3bari\n_Z3fooi\n"]
    assert result == {"_Z3bari": "demangled_one", "_Z3fooi": "demangled_two"}


def test_mangled_names_fall_back_when_c_filt_output_does_not_line_up(monkeypatch):
    """A c++filt run that returns a different number of lines is ignored."""

    def short_output(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="one\n", stderr="")

    monkeypatch.setattr(layout_names.subprocess, "run", short_output)

    assert demangle_kernel_names(["_Z3fooi", "_Z3bari"]) == {}


@pytest.mark.skipif(shutil.which("c++filt") is None, reason="c++filt is not installed")
def test_real_c_filt_demangles_a_mangled_kernel_name():
    """With c++filt installed, a mangled kernel name gets its demangled form."""
    demangled = demangle_kernel_names(["_Z3fooi"])

    assert demangled == {"_Z3fooi": "foo(int)"}
    assert kernel_function_name(demangled["_Z3fooi"]) == "foo"


@pytest.mark.parametrize(
    ("kernel_count", "expected_width"),
    [(0, 3), (1, 3), (1000, 3), (1001, 4), (10001, 5)],
)
def test_index_width_is_at_least_three_and_fits_the_last_index(
    kernel_count, expected_width
):
    """The index width covers kernel_count - 1, with three digits at least."""
    assert kernel_index_width(kernel_count) == expected_width


def test_folder_name_zero_pads_the_index_to_the_width():
    """The index is zero-padded to the width, then the function name follows."""
    assert kernel_folder_name(7, 3, "vecCopy") == "kernel-007-vecCopy"
    assert kernel_folder_name(1000, 4, "vecCopy") == "kernel-1000-vecCopy"
