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
# THE SOFTWARE.

"""Installation checks: is the rocprofiler-sdk package actually on disk?"""

from __future__ import absolute_import

import re

from rocprofv3.doctor_layout import (
    install_kind,
    INSTALL_KIND_TITLES,
    known_installations,
    physical_root,
    reinstall_hint,
    same_installation,
)
from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_skip,
    make_warn,
    SEV_ERROR,
    SEV_WARNING,
)

# matches librocprofiler-sdk.so.0.4.0 / .so.0 / .so
_SOVERSION_RE = re.compile(r"\.so(?:\.(\d+(?:\.\d+)*))?$")


def _version_ladder(accessor):
    """Soversion suffixes to try, in order: "" (plain .so), MAJOR, then FULL.

    Mirrors resolve_library_path() in source/bin/rocprofv3.py, which appends
    the known soversion and then the known full version rather than guessing.
    """
    suffixes = [""]
    version = accessor.tool_version
    if version and not version.startswith("@"):
        parts = version.split(".")
        if parts[0]:
            suffixes.append("." + parts[0])
        if len(parts) >= 3:
            suffixes.append("." + ".".join(parts[0:3]))
    return suffixes


def _newest_versioned(accessor, candidate):
    """Highest ``candidate.<version>`` on disk, compared numerically.

    Used for libraries whose soversion is unrelated to the SDK version (HSA,
    HIP, ...) and for SDK libraries when the tool version is unknown. A plain
    lexicographic sort is wrong here: it ranks ".so.9" above ".so.10".
    """
    best = None
    best_key = None
    for path in accessor.glob(candidate + ".*"):
        suffix = path[len(candidate) + 1 :]
        try:
            key = tuple(int(part) for part in suffix.split("."))
        except ValueError:
            # not a pure numeric soversion (e.g. a .so.debug sidecar); rank last
            key = (-1,)
        if best_key is None or key > best_key:
            best = path
            best_key = key
    return best


def resolve_library(
    accessor,
    name,
    subdirs=("lib", "lib/rocprofiler-sdk", "lib64"),
    sdk_owned=True,
):
    """Find ``name`` under the ROCm root, tolerating soversion suffixes.

    Returns ``(path_or_None, searched_paths)``. Tries the plain ``libfoo.so``
    devel symlink first. For a library built by rocprofiler-sdk
    (``sdk_owned``), whose SOVERSION is the SDK's major version, it then tries
    ``libfoo.so.MAJOR`` and ``libfoo.so.FULL`` for the known tool version --
    and nothing else: a ``.so.99`` beside a 1.x tool is evidence of another
    release, not a substitute the launcher could use. Only when the tool
    version is unknown, or for third-party libraries whose sonames follow
    their own versioning, does it scan for the newest soversion on disk.
    """
    searched = []
    suffixes = _version_ladder(accessor) if sdk_owned else [""]
    exact_version_only = sdk_owned and len(suffixes) > 1

    for subdir in subdirs:
        directory = accessor.join(accessor.rocm_root, subdir)
        candidate = accessor.join(directory, name)

        for suffix in suffixes:
            versioned_candidate = candidate + suffix
            searched.append(versioned_candidate)
            if accessor.path_exists(versioned_candidate):
                return (versioned_candidate, searched)

        if exact_version_only:
            continue
        newest = _newest_versioned(accessor, candidate)
        if newest is not None:
            return (newest, searched)

    # fall back to the loader's own search path
    found, extra = accessor.find_library(name)
    searched.extend(extra)
    return (found, searched)


def other_soversions(accessor, name, subdirs):
    """Every ``name.<version>`` under the ROCm root, whatever the version."""
    found = []
    for subdir in subdirs:
        found.extend(
            accessor.glob(accessor.join(accessor.rocm_root, subdir, name) + ".*")
        )
    return found


def _library_check(accessor, name, subdirs, remediation, failure_factory):
    path, searched = resolve_library(accessor, name, subdirs)
    data = {"library": name, "searched": searched}
    if path is None:
        others = other_soversions(accessor, name, subdirs)
        if others:
            data["other_versions"] = others
            return failure_factory(
                "{} matching rocprofv3 {} not found under {}; only {} present, "
                "which the launcher will not load".format(
                    name,
                    accessor.tool_version,
                    accessor.rocm_root,
                    ", ".join(accessor.basename(other) for other in others),
                ),
                remediation,
                data,
            )
        return failure_factory(
            "{} not found under {}".format(name, accessor.rocm_root),
            remediation,
            data,
        )
    data["resolved_path"] = path
    data["realpath"] = accessor.realpath(path)
    # resolve_library falls back to LD_LIBRARY_PATH and the ldconfig cache, so
    # the copy found may belong to another installation entirely (a stale
    # /opt/rocm while inspecting a tarball or a venv, say)
    if not same_installation(accessor, accessor.dirname(path), accessor.rocm_root):
        return make_warn(
            "{} is missing from {}; the only copy found is {}, which belongs "
            "to a different ROCm installation".format(name, accessor.rocm_root, path),
            remediation,
            data,
        )
    return make_pass("found at {}".format(path), "", data)


LOCATE_ROOT_HINT = (
    "Point the tool at the ROCm installation that contains rocprofv3:\n"
    "  rocprofv3-doctor --rocm-root <prefix>\n"
    "Where that prefix is depends on how ROCm was installed:\n"
    "  ROCm system packages      /opt/rocm or /opt/rocm-X.Y.Z\n"
    "  TheRock system packages   /opt/rocm/core-X.Y\n"
    "  TheRock tarball           the directory it was extracted into\n"
    "  TheRock Python packages   rocm-sdk path --root   (in the venv)\n"
    "Setting ROCM_PATH (or ROCM_HOME) to the prefix also works."
)


@register(
    id="install.rocm-root",
    group="install",
    title="ROCm root directory found",
    severity=SEV_ERROR,
    order=10,
)
def check_rocm_root(accessor):
    root = accessor.rocm_root
    data = {
        "rocm_root": root,
        "realpath": accessor.realpath(root),
        "detected_via": accessor.rocm_root_source,
    }
    if not accessor.path_exists(root):
        return make_fail(
            "ROCm root {!r} does not exist".format(root),
            LOCATE_ROOT_HINT,
            data,
        )
    if not accessor.path_is_dir(root):
        return make_fail(
            "ROCm root {!r} is not a directory".format(root),
            LOCATE_ROOT_HINT,
            data,
        )

    kind = install_kind(accessor)
    data["install_kind"] = kind
    data["physical_root"] = physical_root(accessor, root)
    detail = "using ROCm root {} ({})".format(root, INSTALL_KIND_TITLES[kind])
    if accessor.rocm_root_source:
        detail += ", found via {}".format(accessor.rocm_root_source)
    return make_pass(detail, "", data)


@register(
    id="install.sdk-library",
    group="install",
    title="librocprofiler-sdk.so present",
    severity=SEV_ERROR,
    depends=["install.rocm-root"],
    order=11,
)
def check_sdk_library(accessor):
    return _library_check(
        accessor,
        "librocprofiler-sdk.so",
        ("lib", "lib64"),
        reinstall_hint(accessor),
        make_fail,
    )


@register(
    id="install.sdk-tool-library",
    group="install",
    title="librocprofiler-sdk-tool.so present",
    severity=SEV_ERROR,
    depends=["install.rocm-root"],
    order=12,
)
def check_sdk_tool_library(accessor):
    return _library_check(
        accessor,
        "librocprofiler-sdk-tool.so",
        ("lib/rocprofiler-sdk", "lib64/rocprofiler-sdk"),
        reinstall_hint(accessor),
        make_fail,
    )


@register(
    id="install.roctx-library",
    group="install",
    title="librocprofiler-sdk-roctx.so present",
    severity=SEV_ERROR,
    depends=["install.rocm-root"],
    order=13,
)
def check_roctx_library(accessor):
    return _library_check(
        accessor,
        "librocprofiler-sdk-roctx.so",
        ("lib", "lib64"),
        reinstall_hint(accessor),
        make_fail,
    )


@register(
    id="install.avail-library",
    group="install",
    title="librocprofv3-list-avail.so present",
    severity=SEV_ERROR,
    depends=["install.rocm-root"],
    order=14,
)
def check_avail_library(accessor):
    return _library_check(
        accessor,
        "librocprofv3-list-avail.so",
        ("lib/rocprofiler-sdk", "lib64/rocprofiler-sdk"),
        reinstall_hint(accessor),
        make_fail,
    )


@register(
    id="install.kokkosp-library",
    group="install",
    title="librocprofiler-sdk-tool-kokkosp.so present",
    severity=SEV_WARNING,
    depends=["install.rocm-root"],
    order=15,
)
def check_kokkosp_library(accessor):
    return _library_check(
        accessor,
        "librocprofiler-sdk-tool-kokkosp.so",
        ("lib/rocprofiler-sdk", "lib64/rocprofiler-sdk"),
        "Only needed for `rocprofv3 --kokkos-trace`; safe to ignore otherwise.",
        make_warn,
    )


@register(
    id="install.attach-library",
    group="install",
    title="librocprofiler-sdk-rocattach.so present",
    severity=SEV_WARNING,
    depends=["install.rocm-root"],
    order=16,
)
def check_attach_library(accessor):
    # The attach *injector*, which is a distinct artifact from the tool library
    # checked by install.sdk-tool-library. SYNC: ROCPROF_ATTACH_LIBRARY in
    # source/bin/rocprof-attach.py, built by
    # source/lib/rocprofiler-sdk-rocattach/CMakeLists.txt.
    return _library_check(
        accessor,
        "librocprofiler-sdk-rocattach.so",
        ("lib", "lib64"),
        "Only needed for `rocprof-attach` / `rocprofv3 --pid`.",
        make_warn,
    )


def _parse_soversion(path):
    """Extract the dotted soversion from a library path, or None."""
    match = _SOVERSION_RE.search(path)
    if not match:
        return None
    return match.group(1)


@register(
    id="install.version-consistency",
    group="install",
    title="Tool version matches installed library version",
    severity=SEV_ERROR,
    depends=["install.sdk-library"],
    order=17,
)
def check_version_consistency(accessor):
    tool_version = accessor.tool_version
    data = {"tool_version": tool_version}

    if not tool_version or tool_version.startswith("@"):
        # running from the source tree: configure_file has not substituted
        # @FULL_VERSION_STRING@ yet, so there is nothing to compare against
        return make_skip(
            "tool version unavailable (running from an unconfigured source tree)",
            "",
            data,
        )

    path, _ = resolve_library(accessor, "librocprofiler-sdk.so", ("lib", "lib64"))
    if path is None:
        # the matching soversion is absent; whatever versions *are* installed
        # are exactly the evidence this check exists to report
        others = other_soversions(accessor, "librocprofiler-sdk.so", ("lib", "lib64"))
        if not others:
            return make_skip("librocprofiler-sdk.so not resolvable", "", data)
        path = _newest_versioned(
            accessor, accessor.join(accessor.dirname(others[0]), "librocprofiler-sdk.so")
        )

    real = accessor.realpath(path)
    data["library_path"] = real
    lib_version = _parse_soversion(real)
    data["library_soversion"] = lib_version
    if not lib_version:
        return make_warn(
            "could not determine the soversion of {}".format(real),
            "",
            data,
        )

    tool_parts = tool_version.split(".")
    lib_parts = lib_version.split(".")

    # The soname major is the SDK's ABI version, which tracks but is not equal
    # to the release version; compare the full dotted versions when the library
    # carries one, and only the leading component otherwise.
    if len(lib_parts) >= 3 and len(tool_parts) >= 3:
        if lib_parts[:3] == tool_parts[:3]:
            return make_pass(
                "tool {} matches library {}".format(tool_version, lib_version), "", data
            )
        if lib_parts[0] != tool_parts[0]:
            return make_fail(
                "major version skew: tool is {} but {} is {}".format(
                    tool_version, real, lib_version
                ),
                "Use a rocprofv3 from the same ROCm installation as the library:\n"
                "  rocprofv3-doctor --rocm-root {}".format(
                    accessor.dirname(accessor.dirname(real))
                ),
                data,
            )
        return make_warn(
            "minor version skew: tool is {} but {} is {}".format(
                tool_version, real, lib_version
            ),
            "Usually harmless, but prefer matching versions if profiling misbehaves.",
            data,
        )

    if lib_parts[0] == tool_parts[0]:
        return make_pass(
            "tool {} is consistent with library soversion {}".format(
                tool_version, lib_version
            ),
            "",
            data,
        )
    return make_warn(
        "tool version {} and library soversion {} differ".format(
            tool_version, lib_version
        ),
        "Confirm rocprofv3 and librocprofiler-sdk.so come from the same install.",
        data,
    )


@register(
    id="install.no-mixed-rocm",
    group="install",
    title="No conflicting ROCm installations in the search path",
    severity=SEV_WARNING,
    depends=["install.rocm-root"],
    order=18,
)
def check_no_mixed_rocm(accessor):
    root = accessor.rocm_root
    root_real = accessor.realpath(root)
    data = {"rocm_root": root, "rocm_root_realpath": root_real}

    installs = known_installations(accessor)
    data["known_installations"] = installs

    conflicts = []

    # Compare installations, not path strings: /opt/rocm is a link (or, with
    # TheRock packages, a directory of links) into a versioned tree, and a
    # TheRock venv's _rocm_sdk_devel links back into _rocm_sdk_core. Any of
    # those reached from the environment is the same installation.
    ld_library_path = accessor.getenv("LD_LIBRARY_PATH", "") or ""
    data["LD_LIBRARY_PATH"] = ld_library_path
    for entry in ld_library_path.split(":"):
        if not entry:
            continue
        real = accessor.realpath(entry)
        # a ROCm directory need not have "rocm" in its name (a tarball under
        # ~/therock, say), so also recognise one by the libraries it holds
        is_rocm = "rocm" in real or physical_root(accessor, entry) is not None
        if is_rocm and not same_installation(accessor, entry, root):
            conflicts.append("LD_LIBRARY_PATH entry {} -> {}".format(entry, real))

    for var in ("ROCM_PATH", "ROCM_HOME", "ROCM_DIR"):
        value = accessor.getenv(var)
        if not value:
            continue
        data[var] = value
        if not same_installation(accessor, value, root):
            conflicts.append("{}={} -> {}".format(var, value, accessor.realpath(value)))

    if conflicts:
        return make_warn(
            "the environment points at a different ROCm installation than {}: "
            "{}".format(root_real, "; ".join(conflicts)),
            "unset LD_LIBRARY_PATH ROCM_PATH ROCM_HOME\n"
            "# or set them to match:\n"
            "export ROCM_PATH={}".format(root_real),
            data,
        )

    if len(installs) > 1:
        return make_pass(
            "{} ROCm installations found ({}), but nothing in the environment "
            "redirects away from {}".format(
                len(installs), ", ".join(installs), root_real
            ),
            "",
            data,
        )
    return make_pass("environment consistently points at {}".format(root_real), "", data)


@register(
    id="install.metrics-path",
    group="install",
    title="Counter metrics database accessible",
    severity=SEV_WARNING,
    depends=["install.rocm-root"],
    order=19,
)
def check_metrics_path(accessor):
    metrics_dir = accessor.join(accessor.rocm_root, "share/rocprofiler-sdk")
    data = {"metrics_path": metrics_dir}

    override = accessor.getenv("ROCPROFILER_METRICS_PATH")
    if override:
        data["ROCPROFILER_METRICS_PATH"] = override
        metrics_dir = override

    if not accessor.path_exists(metrics_dir):
        return make_warn(
            "metrics directory {} does not exist".format(metrics_dir),
            reinstall_hint(accessor),
            data,
        )

    # SYNC: the counter definitions shipped into share/rocprofiler-sdk are
    # basic_counters.xml / derived_counters.xml (see
    # source/lib/rocprofiler-sdk/counters/CMakeLists.txt). If the metrics
    # database ever moves to a different format, this glob must follow it.
    xml_files = accessor.glob(accessor.join(metrics_dir, "*.xml"))
    data["counter_files"] = xml_files
    if not xml_files:
        return make_warn(
            "no counter definition files found in {}".format(metrics_dir),
            reinstall_hint(accessor),
            data,
        )
    return make_pass(
        "{} counter definition file(s) in {}".format(len(xml_files), metrics_dir),
        "",
        data,
    )
