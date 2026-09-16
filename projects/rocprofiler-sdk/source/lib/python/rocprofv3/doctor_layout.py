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


"""Where ROCm lives, and how it got there.

``/opt/rocm`` is only one of several layouts rocprofiler-sdk ships in:

* legacy system packages:  ``/opt/rocm`` -> ``/opt/rocm-X.Y.Z``
* TheRock native packages: ``/opt/rocm/{bin,lib,...}`` are update-alternatives
  links into the versioned prefix ``/opt/rocm/core-X.Y``
* TheRock Python wheels:   ``<venv>/lib/pythonX.Y/site-packages/_rocm_sdk_core``
  (the venv ``bin/`` holds compiled launchers, not the real scripts, and the
  runtime wheels carry only SONAME libraries -- no plain ``libfoo.so``)
* TheRock tarballs, source builds and custom prefixes: any directory at all
* distribution packages:   ``/usr``

Everything that depends on the layout -- locating the root, deciding whether
two paths belong to the same installation, and telling the user how to
reinstall -- lives here so the checks stay layout-agnostic.
"""

from __future__ import absolute_import

import re

INSTALL_SYSTEM_PACKAGE = "system-package"
INSTALL_THEROCK_PACKAGE = "therock-package"
INSTALL_PYTHON_WHEEL = "python-wheel"
INSTALL_DISTRO_PACKAGE = "distro-package"
INSTALL_STANDALONE = "standalone"

INSTALL_KIND_TITLES = {
    INSTALL_SYSTEM_PACKAGE: "ROCm system packages",
    INSTALL_THEROCK_PACKAGE: "TheRock system packages",
    INSTALL_PYTHON_WHEEL: "TheRock Python packages",
    INSTALL_DISTRO_PACKAGE: "Linux distribution packages",
    INSTALL_STANDALONE: "standalone tree (tarball, source build, or custom prefix)",
}

# Environment variables that conventionally name a ROCm root. TheRock's own
# documentation uses ROCM_HOME; the legacy stack uses ROCM_PATH.
ROOT_ENV_VARS = ("ROCM_PATH", "ROCM_HOME", "ROCM_DIR")

# Python packages that hold a TheRock wheel install, in preference order:
# _rocm_sdk_core physically contains rocprofiler-sdk; _rocm_sdk_devel only
# exists after `rocm-sdk init` and its libraries are symlinks back into core.
WHEEL_PACKAGES = ("_rocm_sdk_core", "_rocm_sdk_devel")

# A directory is a ROCm tree only when one of these is under its lib/. A bare
# bin/rocprofv3 is not enough: a TheRock venv has a bin/rocprofv3 launcher but
# no ROCm libraries of its own.
MARKER_LIBRARIES = ("librocprofiler-sdk.so", "libhsa-runtime64.so", "libamdhip64.so")

DEFAULT_ROOT = "/opt/rocm"
THEROCK_PACKAGE_GLOB = "/opt/rocm/core*"

SOURCE_OVERRIDE = "--rocm-root"
SOURCE_SCRIPT = "location of rocprofv3-doctor"
SOURCE_PATH = "rocprofv3 on PATH"
SOURCE_DEFAULT = "default location"
SOURCE_NOT_FOUND = "location of rocprofv3-doctor; no ROCm installation found"

_WHEEL_RE = re.compile(r"/(site|dist)-packages/_rocm_sdk_[^/]+(/|$)")
_THEROCK_PACKAGE_RE = re.compile(r"^/opt/rocm/core(-[^/]+)?(/|$)")
_SYSTEM_PACKAGE_RE = re.compile(r"^/opt/rocm(-[^/]+)?(/|$)")
_DISTRO_PACKAGE_RE = re.compile(r"^/usr(/|$)")

# Package names in the legacy ROCm repositories, by component.
_SYSTEM_PACKAGE_NAMES = {
    "rocprofiler-sdk": "rocprofiler-sdk",
    "aqlprofile": "hsa-amd-aqlprofile",
    "hip-runtime": "rocm-hip-runtime",
    "rocprofiler-register": "rocprofiler-register",
}


def _version_key(path):
    """Numeric sort key for a versioned directory name such as core-10.1."""
    return tuple(int(part) for part in re.findall(r"\d+", path))


def find_marker_library(accessor, path, subdirs=("lib", "lib64", "")):
    """A ROCm marker library inside ``path`` (a root or a lib dir), or None."""
    for subdir in subdirs:
        directory = accessor.join(path, subdir) if subdir else path
        for name in MARKER_LIBRARIES:
            candidate = accessor.join(directory, name)
            if accessor.path_exists(candidate):
                return candidate
            versioned = accessor.glob(candidate + ".*")
            if versioned:
                return versioned[0]
    return None


def looks_like_rocm_root(accessor, path):
    if not path or not accessor.path_is_dir(path):
        return False
    return find_marker_library(accessor, path, ("lib", "lib64")) is not None


def physical_root(accessor, path):
    """The directory tree whose files actually back ``path``, or None.

    ``/opt/rocm`` under TheRock packages is a plain directory of symlinks into
    ``/opt/rocm/core-X.Y``, and ``_rocm_sdk_devel`` is a tree of symlinks into
    ``_rocm_sdk_core``; resolving ``path`` itself identifies neither. Resolving
    a library inside it does.
    """
    marker = find_marker_library(accessor, path)
    if marker is None:
        return None
    return accessor.dirname(accessor.dirname(accessor.realpath(marker)))


def same_installation(accessor, path, root):
    """True when ``path`` (a ROCm root or a directory in one) is ``root``'s install."""
    real = accessor.realpath(path)
    bases = [accessor.realpath(root)]
    root_physical = physical_root(accessor, root)
    if root_physical:
        bases.append(root_physical)
    for base in bases:
        if accessor.path_within(real, base):
            return True
    other = physical_root(accessor, path)
    return other is not None and other in bases


def wheel_roots_under(accessor, prefix):
    """TheRock wheel roots inside the Python environment at ``prefix``."""
    roots = []
    for package in WHEEL_PACKAGES:
        for site_dir in ("site-packages", "dist-packages"):
            pattern = accessor.join(prefix, "lib", "python3*", site_dir, package)
            roots.extend(accessor.glob(pattern))
    return roots


def known_installations(accessor):
    """Every ROCm installation discoverable without any hint from the user.

    Paths that are views of the same files (``/opt/rocm`` and the
    ``/opt/rocm/core-X.Y`` it links into) are reported once.
    """
    found = []
    seen = set()
    candidates = accessor.glob("/opt/rocm*") + sorted(
        accessor.glob(THEROCK_PACKAGE_GLOB), key=_version_key
    )
    module_dir = accessor.find_module_dir(WHEEL_PACKAGES[0])
    if module_dir:
        candidates.append(module_dir)
    for candidate in candidates:
        if not looks_like_rocm_root(accessor, candidate):
            continue
        physical = physical_root(accessor, candidate)
        if physical not in seen:
            seen.add(physical)
            found.append(candidate)
    return found


def detect_rocm_root(accessor, script_path, override=None):
    """Locate the ROCm root to inspect; return ``(root, how_it_was_found)``.

    An explicit ``--rocm-root`` is taken verbatim -- reporting on exactly the
    tree the user named, even a broken one, is the point of the flag.
    Otherwise the first of these that holds a ROCm tree wins:

    1. the prefix this script is installed under (the same rule rocprofv3 uses)
    2. ROCM_PATH / ROCM_HOME / ROCM_DIR
    3. the prefix of the rocprofv3 found on PATH
    4. the TheRock wheel importable by this interpreter
    5. /opt/rocm, then the newest /opt/rocm/core-X.Y

    Each prefix is also searched for TheRock wheels, so a venv prefix (whose
    bin/ holds only launchers) resolves to its ``_rocm_sdk_core``.
    """
    if override:
        return (accessor.abspath(override), SOURCE_OVERRIDE)

    candidates = []
    if script_path:
        script_dir = accessor.dirname(accessor.realpath(script_path))
        candidates.append((accessor.dirname(script_dir), SOURCE_SCRIPT))
    for var in ROOT_ENV_VARS:
        value = accessor.getenv(var)
        if value:
            candidates.append((accessor.abspath(value), var))
    tool, _ = accessor.which("rocprofv3")
    if tool:
        tool_dir = accessor.dirname(accessor.realpath(tool))
        candidates.append((accessor.dirname(tool_dir), SOURCE_PATH))
    for package in WHEEL_PACKAGES:
        module_dir = accessor.find_module_dir(package)
        if module_dir:
            candidates.append((module_dir, "Python package {}".format(package)))
    candidates.append((DEFAULT_ROOT, SOURCE_DEFAULT))
    for path in sorted(
        accessor.glob(THEROCK_PACKAGE_GLOB), key=_version_key, reverse=True
    ):
        candidates.append((path, SOURCE_DEFAULT))

    for path, source in candidates:
        for candidate in [path] + wheel_roots_under(accessor, path):
            if looks_like_rocm_root(accessor, candidate):
                return (candidate, source)

    if script_path:
        return (candidates[0][0], SOURCE_NOT_FOUND)
    return (DEFAULT_ROOT, SOURCE_NOT_FOUND)


def install_kind(accessor, root=None):
    """Classify how the ROCm tree at ``root`` (default: the inspected one) was installed."""
    root = accessor.rocm_root if root is None else root
    resolved = physical_root(accessor, root) or accessor.realpath(root)
    if _WHEEL_RE.search(resolved) or _WHEEL_RE.search(accessor.realpath(root)):
        return INSTALL_PYTHON_WHEEL
    if _THEROCK_PACKAGE_RE.match(resolved):
        return INSTALL_THEROCK_PACKAGE
    if _SYSTEM_PACKAGE_RE.match(resolved):
        return INSTALL_SYSTEM_PACKAGE
    if _DISTRO_PACKAGE_RE.match(resolved):
        return INSTALL_DISTRO_PACKAGE
    return INSTALL_STANDALONE


def _venv_python(accessor, root):
    """The interpreter of the venv holding a wheel root, else plain python3.

    ``<venv>/lib/pythonX.Y/site-packages/_rocm_sdk_core`` -> ``<venv>/bin/python``
    """
    venv = root
    for _ in range(4):
        venv = accessor.dirname(venv)
    candidate = accessor.join(venv, "bin", "python")
    if accessor.path_exists(candidate):
        return candidate
    return "python3"


def reinstall_hint(accessor, component="rocprofiler-sdk"):
    """How to restore ``component`` for the way this ROCm tree was installed."""
    root = accessor.rocm_root
    kind = install_kind(accessor)
    package = _SYSTEM_PACKAGE_NAMES.get(component, component)

    if kind == INSTALL_PYTHON_WHEEL:
        return (
            "{} ships in the ROCm Python packages. Reinstall them into this\n"
            "environment, using the same --index-url they came from:\n"
            '  {} -m pip install --force-reinstall "rocm[libraries]"\n'
            "then verify the layout:\n"
            "  rocm-sdk test".format(package, _venv_python(accessor, root))
        )
    if kind == INSTALL_THEROCK_PACKAGE:
        return (
            "{} ships in the ROCm Core SDK packages. Reinstall them:\n"
            "  sudo apt install --reinstall amdrocm-core-sdk\n"
            "or\n"
            "  sudo dnf reinstall amdrocm-core-sdk\n"
            "To find the amdrocm-* package that owns a file under {}:\n"
            "  dpkg -S <file>   # or: rpm -qf <file>".format(package, root)
        )
    if kind == INSTALL_SYSTEM_PACKAGE:
        return (
            "Reinstall the {} package:\n"
            "  sudo apt install --reinstall {}\n"
            "or\n"
            "  sudo dnf reinstall {}".format(package, package, package)
        )
    if kind == INSTALL_DISTRO_PACKAGE:
        return (
            "Reinstall the distribution package that provides {} with your\n"
            "package manager (apt, dnf, zypper, ...).".format(package)
        )
    return (
        "{} is not managed by a package manager (a TheRock tarball, a source\n"
        "build, or a custom prefix). Re-extract or rebuild {} in that tree,\n"
        "or point the tool at a complete installation:\n"
        "  rocprofv3-doctor --rocm-root /path/to/rocm".format(root, package)
    )
