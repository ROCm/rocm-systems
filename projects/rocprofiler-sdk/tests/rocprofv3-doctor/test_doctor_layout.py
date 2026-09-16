#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""GPU-free unit tests for ROCm install-layout handling in rocprofv3-doctor.

rocprofiler-sdk is not always under /opt/rocm. Each test here builds one real
install layout with ``FakeAccessor`` -- legacy system packages, TheRock native
packages, TheRock Python wheels, TheRock tarballs, distro packages -- and checks
that the root is found, classified, and compared correctly.
"""

import pytest

from conftest import FakeAccessor, make_healthy_accessor

SDK_LIBS = ("librocprofiler-sdk.so", "libhsa-runtime64.so")


@pytest.fixture
def layout(rocprofv3_package):
    from rocprofv3 import doctor_layout

    return doctor_layout


@pytest.fixture
def checks(rocprofv3_package):
    from rocprofv3 import (
        doctor_checks_install,
        doctor_checks_runtime,
        doctor_checks_tools,
        doctor_result,
    )

    return {
        "install": doctor_checks_install,
        "runtime": doctor_checks_runtime,
        "tools": doctor_checks_tools,
        "status": doctor_result,
    }


def _tree(root, libs=SDK_LIBS, suffix=""):
    """files + dirs for a ROCm tree at ``root``."""
    files = [root + "/lib/" + lib + suffix for lib in libs]
    files += [root + "/bin/rocprofv3", root + "/bin/rocprofv3-doctor"]
    dirs = [root, root + "/lib", root + "/bin"]
    return files, dirs


def therock_package_accessor(**overrides):
    """TheRock native packages: /opt/rocm/{bin,lib} are alternatives links
    into the versioned prefix /opt/rocm/core-10.0."""
    files, dirs = _tree("/opt/rocm/core-10.0")
    kwargs = dict(
        rocm_root="/opt/rocm/core-10.0",
        files=files,
        dirs=dirs + ["/opt/rocm", "/etc/alternatives"],
        links={
            "/opt/rocm/lib": "/etc/alternatives/rocm-lib",
            "/etc/alternatives/rocm-lib": "/opt/rocm/core-10.0/lib",
            "/opt/rocm/bin": "/etc/alternatives/rocm-bin",
            "/etc/alternatives/rocm-bin": "/opt/rocm/core-10.0/bin",
        },
    )
    kwargs.update(overrides)
    return FakeAccessor(**kwargs)


VENV = "/home/me/venv"
WHEEL_CORE = VENV + "/lib/python3.12/site-packages/_rocm_sdk_core"


def wheel_accessor(**overrides):
    """TheRock Python wheels: runtime wheels hold SONAME libraries only, and the
    venv bin/ holds compiled launchers rather than links into the wheel."""
    files, dirs = _tree(WHEEL_CORE, suffix=".1")
    files += [VENV + "/bin/rocprofv3", VENV + "/bin/python"]
    kwargs = dict(
        rocm_root=WHEEL_CORE,
        files=files,
        dirs=dirs + [VENV, VENV + "/bin"],
    )
    kwargs.update(overrides)
    return FakeAccessor(**kwargs)


TARBALL = "/home/me/therock/install"


def tarball_accessor(**overrides):
    files, dirs = _tree(TARBALL)
    kwargs = dict(rocm_root=TARBALL, files=files, dirs=dirs)
    kwargs.update(overrides)
    return FakeAccessor(**kwargs)


def legacy_accessor(**overrides):
    files, dirs = _tree("/opt/rocm-6.2.0")
    kwargs = dict(
        rocm_root="/opt/rocm",
        files=files,
        dirs=dirs,
        links={"/opt/rocm": "/opt/rocm-6.2.0"},
    )
    kwargs.update(overrides)
    return FakeAccessor(**kwargs)


# ----------------------------------------------------------------------
# root detection
# ----------------------------------------------------------------------
def test_doctor_layout_override_is_verbatim(layout):
    accessor = FakeAccessor(rocm_root=None)
    root, source = layout.detect_rocm_root(accessor, None, "/does/not/exist")
    assert root == "/does/not/exist"
    assert source == layout.SOURCE_OVERRIDE


def test_doctor_layout_detects_from_script_through_alternatives(layout):
    accessor = therock_package_accessor(rocm_root=None)
    root, source = layout.detect_rocm_root(accessor, "/opt/rocm/bin/rocprofv3-doctor")
    assert root == "/opt/rocm/core-10.0"
    assert source == layout.SOURCE_SCRIPT


def test_doctor_layout_detects_tarball_from_rocm_home(layout):
    accessor = tarball_accessor(rocm_root=None, env={"ROCM_HOME": TARBALL})
    root, source = layout.detect_rocm_root(accessor, "/elsewhere/bin/rocprofv3-doctor")
    assert root == TARBALL
    assert source == "ROCM_HOME"


def test_doctor_layout_detects_wheel_through_venv_launcher(layout):
    """rocprofv3 on PATH is a compiled launcher in the venv bin/, whose prefix
    holds no libraries; the wheel inside that venv is the real root."""
    accessor = wheel_accessor(rocm_root=None, env={"PATH": VENV + "/bin"})
    root, source = layout.detect_rocm_root(accessor, "/elsewhere/bin/rocprofv3-doctor")
    assert root == WHEEL_CORE
    assert source == layout.SOURCE_PATH


def test_doctor_layout_detects_wheel_via_python_import(layout):
    accessor = wheel_accessor(rocm_root=None, module_dirs={"_rocm_sdk_core": WHEEL_CORE})
    root, source = layout.detect_rocm_root(accessor, "/elsewhere/bin/rocprofv3-doctor")
    assert root == WHEEL_CORE
    assert "_rocm_sdk_core" in source


def test_doctor_layout_falls_back_to_newest_therock_prefix(layout):
    """No /opt/rocm compatibility links at all: pick the newest core-X.Y,
    comparing versions numerically."""
    files_old, dirs_old = _tree("/opt/rocm/core-9.9")
    files_new, dirs_new = _tree("/opt/rocm/core-10.0")
    accessor = FakeAccessor(
        rocm_root=None,
        files=files_old + files_new,
        dirs=dirs_old + dirs_new + ["/opt/rocm"],
    )
    root, source = layout.detect_rocm_root(accessor, "/elsewhere/bin/rocprofv3-doctor")
    assert root == "/opt/rocm/core-10.0"
    assert source == layout.SOURCE_DEFAULT


def test_doctor_layout_bin_rocprofv3_alone_is_not_a_root(layout):
    accessor = FakeAccessor(
        rocm_root=None, files=["/prefix/bin/rocprofv3"], dirs=["/prefix", "/prefix/bin"]
    )
    assert not layout.looks_like_rocm_root(accessor, "/prefix")


def test_doctor_layout_nothing_found_reports_script_prefix(layout):
    accessor = FakeAccessor(rocm_root=None)
    root, source = layout.detect_rocm_root(
        accessor, "/src/source/bin/rocprofv3-doctor.py"
    )
    assert root == "/src/source"
    assert source == layout.SOURCE_NOT_FOUND


# ----------------------------------------------------------------------
# classification and remediation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "factory,kind",
    [
        (legacy_accessor, "system-package"),
        (therock_package_accessor, "therock-package"),
        (wheel_accessor, "python-wheel"),
        (tarball_accessor, "standalone"),
    ],
)
def test_doctor_layout_install_kind(layout, factory, kind):
    assert layout.install_kind(factory()) == kind


def test_doctor_layout_install_kind_through_compat_links(layout):
    """/opt/rocm with TheRock packages is a plain directory of links; it must
    still classify by the tree those links point into."""
    accessor = therock_package_accessor(rocm_root="/opt/rocm")
    assert layout.install_kind(accessor) == layout.INSTALL_THEROCK_PACKAGE


def test_doctor_layout_install_kind_distro(layout):
    files, dirs = _tree("/usr")
    accessor = FakeAccessor(rocm_root="/usr", files=files, dirs=dirs)
    assert layout.install_kind(accessor) == layout.INSTALL_DISTRO_PACKAGE


def test_doctor_layout_reinstall_hints_match_layout(layout):
    assert "apt install --reinstall rocprofiler-sdk" in layout.reinstall_hint(
        legacy_accessor()
    )
    assert "amdrocm-core-sdk" in layout.reinstall_hint(therock_package_accessor())

    wheel_hint = layout.reinstall_hint(wheel_accessor())
    assert VENV + "/bin/python -m pip install" in wheel_hint
    assert "apt" not in wheel_hint

    tarball_hint = layout.reinstall_hint(tarball_accessor())
    assert TARBALL in tarball_hint
    assert "apt" not in tarball_hint and "pip" not in tarball_hint


def test_doctor_layout_reinstall_hint_names_component(layout):
    for factory in (legacy_accessor, therock_package_accessor, wheel_accessor):
        hint = layout.reinstall_hint(factory(), "aqlprofile")
        assert "aqlprofile" in hint, factory.__name__


def test_doctor_layout_rocm_root_failure_is_layout_neutral(checks):
    accessor = FakeAccessor(rocm_root="/nonexistent")
    result = checks["install"].check_rocm_root(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    for hint in ("/opt/rocm/core-X.Y", "rocm-sdk path --root", "extracted"):
        assert hint in result.remediation


def test_doctor_layout_rocm_root_pass_reports_kind_and_source(checks):
    accessor = therock_package_accessor(rocm_root_source="location of rocprofv3-doctor")
    result = checks["install"].check_rocm_root(accessor)
    assert result.status == checks["status"].STATUS_PASS
    assert result.data["install_kind"] == "therock-package"
    assert "TheRock system packages" in result.detail
    assert "rocprofv3-doctor" in result.detail


# ----------------------------------------------------------------------
# same-installation comparison (install.no-mixed-rocm)
# ----------------------------------------------------------------------
def test_doctor_layout_therock_compat_paths_are_not_a_conflict(checks):
    """ROCM_PATH=/opt/rocm and LD_LIBRARY_PATH=/opt/rocm/lib are the documented
    setup for TheRock packages, and name the same files as /opt/rocm/core-10.0."""
    accessor = therock_package_accessor(
        env={"ROCM_PATH": "/opt/rocm", "LD_LIBRARY_PATH": "/opt/rocm/lib"}
    )
    result = checks["install"].check_no_mixed_rocm(accessor)
    assert result.status == checks["status"].STATUS_PASS, result.detail


def test_doctor_layout_wheel_devel_tree_is_not_a_conflict(checks):
    """_rocm_sdk_devel (after `rocm-sdk init`) is a tree of links into
    _rocm_sdk_core; ROCM_HOME=$(rocm-sdk path --root) points there."""
    devel = VENV + "/lib/python3.12/site-packages/_rocm_sdk_devel"
    links = {
        devel + "/lib/" + lib + ".1": WHEEL_CORE + "/lib/" + lib + ".1"
        for lib in SDK_LIBS
    }
    accessor = wheel_accessor(
        env={"ROCM_HOME": devel},
        links=links,
        dirs=[WHEEL_CORE, WHEEL_CORE + "/lib", devel, devel + "/lib", VENV],
    )
    result = checks["install"].check_no_mixed_rocm(accessor)
    assert result.status == checks["status"].STATUS_PASS, result.detail


def test_doctor_layout_other_tarball_without_rocm_in_name_conflicts(checks):
    other_files, other_dirs = _tree("/home/me/nightly")
    files, dirs = _tree(TARBALL)
    accessor = FakeAccessor(
        rocm_root=TARBALL,
        files=files + other_files,
        dirs=dirs + other_dirs,
        env={"LD_LIBRARY_PATH": "/home/me/nightly/lib"},
    )
    result = checks["install"].check_no_mixed_rocm(accessor)
    assert result.status == checks["status"].STATUS_WARN
    assert "/home/me/nightly/lib" in result.detail


def test_doctor_layout_legacy_symlink_is_not_a_conflict(checks):
    accessor = legacy_accessor(env={"ROCM_PATH": "/opt/rocm-6.2.0"})
    result = checks["install"].check_no_mixed_rocm(accessor)
    assert result.status == checks["status"].STATUS_PASS, result.detail


def test_doctor_layout_known_installations_dedupes_views(layout):
    """/opt/rocm and /opt/rocm/core-10.0 are one installation, not two."""
    installs = layout.known_installations(therock_package_accessor())
    assert len(installs) == 1, installs


def test_doctor_layout_runtime_library_through_compat_root(checks):
    """Inspecting /opt/rocm under TheRock packages: a library under the
    versioned prefix belongs to the same installation."""
    accessor = therock_package_accessor(rocm_root="/opt/rocm")
    result = checks["runtime"].check_hsa_library(accessor)
    assert result.status == checks["status"].STATUS_PASS, result.detail


# ----------------------------------------------------------------------
# wheel specifics
# ----------------------------------------------------------------------
def test_doctor_layout_wheel_soname_only_library_passes(checks):
    result = checks["install"].check_sdk_library(wheel_accessor())
    assert result.status == checks["status"].STATUS_PASS, result.detail
    assert result.data["resolved_path"].endswith("librocprofiler-sdk.so.1")


def test_doctor_layout_tool_command_honors_shebang(checks):
    """Review P2: a Python tool runs through its own shebang, as the user would
    run it, so a broken interpreter or import path is not masked."""
    accessor = make_healthy_accessor(
        file_contents={"/fake/rocm/bin/rocprofv3-avail": "#!/usr/bin/env python3\n"}
    )
    command = checks["tools"].tool_command(
        accessor, "/fake/rocm/bin/rocprofv3-avail", ["info"]
    )
    assert command == ["/fake/rocm/bin/rocprofv3-avail", "info"]


def test_doctor_layout_tool_command_native_launcher(checks):
    """A TheRock venv launcher is a native executable and must run directly."""
    accessor = wheel_accessor()
    command = checks["tools"].tool_command(accessor, VENV + "/bin/rocprofv3", ["--help"])
    assert command == [VENV + "/bin/rocprofv3", "--help"]


def test_doctor_layout_library_from_other_installation_warns(checks):
    """Inspecting a tarball whose SDK library is missing: the loader would find
    /opt/rocm's copy, which must not count as this installation's."""
    files, dirs = _tree(TARBALL, libs=("libhsa-runtime64.so",))
    accessor = FakeAccessor(
        rocm_root=TARBALL,
        files=files + ["/opt/rocm/lib/librocprofiler-sdk.so"],
        dirs=dirs + ["/opt/rocm", "/opt/rocm/lib"],
        env={"LD_LIBRARY_PATH": "/opt/rocm/lib"},
    )
    result = checks["install"].check_sdk_library(accessor)
    assert result.status == checks["status"].STATUS_WARN, result.detail
    assert "/opt/rocm/lib/librocprofiler-sdk.so" in result.detail


# ----------------------------------------------------------------------
# soversion resolution (design review 2026-10-06)
# ----------------------------------------------------------------------
def test_doctor_layout_other_major_is_evidence_not_a_substitute(checks):
    """Review P2: with tool 1.4.0 and only librocprofiler-sdk.so.99 on disk the
    resolver used to pick .so.99. The launcher would not load it."""
    accessor = FakeAccessor(
        rocm_root="/fake/rocm",
        files=["/fake/rocm/lib/librocprofiler-sdk.so.99"],
        dirs=["/fake/rocm", "/fake/rocm/lib"],
        tool_version="1.4.0",
    )
    path, _ = checks["install"].resolve_library(accessor, "librocprofiler-sdk.so")
    assert path is None

    result = checks["install"].check_sdk_library(accessor)
    assert result.status == checks["status"].STATUS_FAIL
    assert "librocprofiler-sdk.so.99" in result.detail
    assert result.data["other_versions"] == ["/fake/rocm/lib/librocprofiler-sdk.so.99"]


def test_doctor_layout_matching_soname_still_resolves(checks):
    accessor = FakeAccessor(
        rocm_root="/fake/rocm",
        files=[
            "/fake/rocm/lib/librocprofiler-sdk.so.1",
            "/fake/rocm/lib/librocprofiler-sdk.so.99",
        ],
        dirs=["/fake/rocm", "/fake/rocm/lib"],
        tool_version="1.4.0",
    )
    path, _ = checks["install"].resolve_library(accessor, "librocprofiler-sdk.so")
    assert path == "/fake/rocm/lib/librocprofiler-sdk.so.1"


def test_doctor_layout_third_party_soname_is_scanned(checks):
    """HIP's soname (.so.7) has nothing to do with the SDK version."""
    accessor = FakeAccessor(
        rocm_root="/fake/rocm",
        files=["/fake/rocm/lib/libamdhip64.so.7"],
        dirs=["/fake/rocm", "/fake/rocm/lib"],
        tool_version="1.4.0",
    )
    path, _ = checks["install"].resolve_library(
        accessor, "libamdhip64.so", ("lib",), sdk_owned=False
    )
    assert path == "/fake/rocm/lib/libamdhip64.so.7"
