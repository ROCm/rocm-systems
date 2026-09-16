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

"""Python environment checks: interpreter version and required/optional modules."""

from __future__ import absolute_import

from rocprofv3.doctor_layout import same_installation
from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_warn,
    SEV_ERROR,
    SEV_INFO,
    SEV_WARNING,
)

MIN_PYTHON = (3, 6)


def _module_check(accessor, module, purpose, remediation, failure_factory):
    ok, detail = accessor.can_import(module)
    data = {"module": module, "purpose": purpose}
    if not ok:
        data["error"] = detail
        return failure_factory(
            "{} is not importable ({})".format(module, detail), remediation, data
        )
    location = accessor.module_file(module)
    if location:
        data["location"] = location
    if detail:
        data["version"] = detail
        return make_pass("{} {} available".format(module, detail), "", data)
    return make_pass("{} available".format(module), "", data)


@register(
    id="python.version",
    group="python",
    title="Python interpreter is 3.6 or newer",
    severity=SEV_ERROR,
    order=70,
)
def check_python_version(accessor):
    version = accessor.python_version()
    data = {
        "version": ".".join(["{}".format(part) for part in version]),
        "executable": accessor.python_executable(),
    }
    if version >= MIN_PYTHON:
        return make_pass("Python {}".format(data["version"]), "", data)
    return make_fail(
        "Python {} is older than the required 3.6".format(data["version"]),
        "Install Python 3.6 or newer and re-run with it:\n"
        "  python3.9 $(which rocprofv3) ...",
        data,
    )


@register(
    id="python.sqlite3",
    group="python",
    title="sqlite3 module usable",
    severity=SEV_ERROR,
    order=71,
)
def check_sqlite3(accessor):
    ok, detail = accessor.can_import("sqlite3")
    data = {"module": "sqlite3", "purpose": "rocpd output database"}
    if not ok:
        data["error"] = detail
        return make_fail(
            "sqlite3 is not importable ({})".format(detail),
            "This Python was built without SQLite support. Install a complete "
            "Python:\n  sudo apt install python3 libsqlite3-0",
            data,
        )
    if not accessor.sqlite_usable():
        return make_fail(
            "sqlite3 imports but cannot open an in-memory database",
            "This Python was built against a broken or missing SQLite library. "
            "Install a complete Python:\n  sudo apt install python3 libsqlite3-0",
            data,
        )
    return make_pass("sqlite3 imports and can open a database", "", data)


@register(
    id="python.ctypes",
    group="python",
    title="ctypes module importable",
    severity=SEV_ERROR,
    order=72,
)
def check_ctypes(accessor):
    return _module_check(
        accessor,
        "ctypes",
        "used by rocprofv3-avail to bind the list-avail library",
        "This Python was built without libffi support. Install a complete "
        "Python:\n  sudo apt install python3 libffi-dev",
        make_fail,
    )


@register(
    id="python.importlib",
    group="python",
    title="importlib module importable",
    severity=SEV_ERROR,
    order=73,
)
def check_importlib(accessor):
    return _module_check(
        accessor,
        "importlib",
        "used to load rocpd output plugins",
        "Reinstall Python; importlib is part of the standard library.",
        make_fail,
    )


@register(
    id="python.pyyaml",
    group="python",
    title="pyyaml importable (YAML input files)",
    severity=SEV_WARNING,
    order=74,
)
def check_pyyaml(accessor):
    return _module_check(
        accessor,
        "yaml",
        "required for `rocprofv3 --input config.yaml`",
        "python3 -m pip install pyyaml\n"
        "Only needed for YAML input files; safe to ignore otherwise.",
        make_warn,
    )


@register(
    id="python.pandas",
    group="python",
    title="pandas importable (rocpd CSV conversion)",
    severity=SEV_INFO,
    order=75,
)
def check_pandas(accessor):
    return _module_check(
        accessor,
        "pandas",
        "used by some rocpd conversion paths",
        "python3 -m pip install pandas\nOptional; safe to ignore otherwise.",
        make_warn,
    )


@register(
    id="python.otf2",
    group="python",
    title="otf2 importable (OTF2 output format)",
    severity=SEV_INFO,
    order=76,
)
def check_otf2(accessor):
    return _module_check(
        accessor,
        "otf2",
        "required for `rocprofv3 --output-format otf2`",
        "python3 -m pip install otf2\n"
        "Only needed for OTF2 output; safe to ignore otherwise.",
        make_warn,
    )


@register(
    id="python.rocprofv3-package",
    group="python",
    title="rocprofv3 Python package importable",
    severity=SEV_ERROR,
    order=77,
)
def check_rocprofv3_package(accessor):
    ok, detail = accessor.can_import("rocprofv3")
    location = accessor.module_file("rocprofv3")
    data = {"module": "rocprofv3", "location": location}

    if not ok:
        data["error"] = detail
        site_packages = accessor.join(accessor.rocm_root, "lib/python3/site-packages")
        return make_fail(
            "the rocprofv3 Python package is not importable ({})".format(detail),
            "export PYTHONPATH={}:$PYTHONPATH".format(site_packages),
            data,
        )

    # This tool *is* the rocprofv3 package, so the import can only fail in
    # exotic situations -- but the location is worth reporting, because an
    # unexpected one means a stale copy is shadowing the installed package.
    if location:
        root_real = accessor.realpath(accessor.rocm_root)
        if not same_installation(accessor, location, accessor.rocm_root):
            return make_warn(
                "rocprofv3 package loaded from {}, which is outside {}".format(
                    location, root_real
                ),
                "A stale copy may be shadowing the installed package; check "
                "PYTHONPATH.",
                data,
            )
    return make_pass("rocprofv3 package importable from {}".format(location), "", data)
