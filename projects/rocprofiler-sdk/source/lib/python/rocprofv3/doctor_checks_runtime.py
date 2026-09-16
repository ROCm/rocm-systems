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

"""Runtime stack checks: HSA, HIP, AQLProfile, and the ATT trace decoder.

Note: the plan's ``runtime.kfd-ioctl-version`` check is intentionally absent.
There is no sysfs file exposing the KFD interface version (verified:
/sys/devices/virtual/kfd/kfd/ contains only dev, power, proc, subsystem,
topology, uevent); it is obtainable only via ioctl, which is out of scope for a
read-only diagnostic.
"""

from __future__ import absolute_import

from rocprofv3.doctor_checks_install import resolve_library
from rocprofv3.doctor_diagnose import diagnose
from rocprofv3.doctor_env import describe_returncode
from rocprofv3.doctor_layout import reinstall_hint, same_installation
from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_skip,
    make_warn,
    PROBE_PROCESS,
    SEV_ERROR,
    SEV_INFO,
    SEV_WARNING,
)


def _runtime_library(accessor, name, remediation, failure_factory, purpose):
    """Resolve a runtime library and describe where it came from.

    A library resolved only through LD_LIBRARY_PATH -- i.e. outside the ROCm
    tree this tool is inspecting -- is reported as ``warn``: it loads, but it
    may be a mismatched build.
    """
    # none of these are built by rocprofiler-sdk, so their sonames follow their
    # own versioning rather than the SDK's
    path, searched = resolve_library(accessor, name, ("lib", "lib64"), sdk_owned=False)
    data = {"library": name, "searched": searched, "purpose": purpose}

    if path is None:
        return failure_factory(
            "{} not found in LD_LIBRARY_PATH, {}/lib, or the ldconfig "
            "cache".format(name, accessor.rocm_root),
            remediation,
            data,
        )

    real = accessor.realpath(path)
    data["resolved_path"] = path
    data["realpath"] = real

    root_real = accessor.realpath(accessor.rocm_root)
    if not same_installation(accessor, accessor.dirname(path), accessor.rocm_root):
        return make_warn(
            "{} resolved to {}, which is outside {} -- this may be a mismatched "
            "build".format(name, real, root_real),
            "unset LD_LIBRARY_PATH\n"
            "# or make sure it points at {}/lib".format(root_real),
            data,
        )
    return make_pass("found at {}".format(path), "", data)


@register(
    id="runtime.hsa-library",
    group="runtime",
    title="libhsa-runtime64.so findable",
    severity=SEV_ERROR,
    depends=["install.sdk-library"],
    order=30,
)
def check_hsa_library(accessor):
    return _runtime_library(
        accessor,
        "libhsa-runtime64.so",
        reinstall_hint(accessor, "hip-runtime"),
        make_fail,
        "required by every rocprofv3 mode",
    )


@register(
    id="runtime.hip-library",
    group="runtime",
    title="libamdhip64.so findable",
    severity=SEV_WARNING,
    depends=["install.sdk-library"],
    order=31,
)
def check_hip_library(accessor):
    return _runtime_library(
        accessor,
        "libamdhip64.so",
        "Only needed for --hip-trace and HIP applications.\n"
        + reinstall_hint(accessor, "hip-runtime"),
        make_warn,
        "required for --hip-trace",
    )


@register(
    id="runtime.aqlprofile-library",
    group="runtime",
    title="libhsa-amd-aqlprofile64.so findable",
    severity=SEV_ERROR,
    depends=["install.sdk-library"],
    order=32,
)
def check_aqlprofile_library(accessor):
    # error severity: without AQLProfile, --pmc counter collection aborts at
    # runtime rather than degrading, which is a confusing first-contact failure
    return _runtime_library(
        accessor,
        "libhsa-amd-aqlprofile64.so",
        reinstall_hint(accessor, "aqlprofile"),
        make_fail,
        "required for --pmc counter collection",
    )


@register(
    id="runtime.att-decoder-library",
    group="runtime",
    title="librocprof-trace-decoder.so findable (ATT)",
    severity=SEV_INFO,
    depends=["install.sdk-library"],
    order=33,
)
def check_att_decoder_library(accessor):
    name = "librocprof-trace-decoder.so"
    searched = []

    override = accessor.getenv("ROCPROF_ATT_LIBRARY_PATH")
    if override:
        candidate = accessor.join(override, name)
        searched.append(candidate)
        if accessor.path_exists(candidate):
            return make_pass(
                "found at {} (via ROCPROF_ATT_LIBRARY_PATH)".format(candidate),
                "",
                {"library": name, "resolved_path": candidate, "searched": searched},
            )

    for subdir in ("lib/rocprofiler-sdk", "lib", "lib64"):
        candidate = accessor.join(accessor.rocm_root, subdir, name)
        searched.append(candidate)
        if accessor.path_exists(candidate):
            return make_pass(
                "found at {}".format(candidate),
                "",
                {"library": name, "resolved_path": candidate, "searched": searched},
            )

    return make_warn(
        "{} not found -- advanced thread trace (--att-*) is unavailable".format(name),
        "Install the rocprof-trace-decoder package, or point at an existing "
        "copy:\n  export ROCPROF_ATT_LIBRARY_PATH=/path/to/decoder/dir\n"
        "Only needed for thread trace; safe to ignore otherwise.",
        {"library": name, "searched": searched},
    )


@register(
    id="runtime.rocprofiler-register",
    group="runtime",
    title="librocprofiler-register.so findable",
    severity=SEV_WARNING,
    depends=["install.sdk-library"],
    order=34,
)
def check_rocprofiler_register(accessor):
    # the HSA/HIP runtimes use rocprofiler-register to advertise their intercept
    # tables; without it the SDK cannot attach to them
    return _runtime_library(
        accessor,
        "librocprofiler-register.so",
        reinstall_hint(accessor, "rocprofiler-register"),
        make_warn,
        "used by HSA/HIP to advertise intercept tables",
    )


# Loaded in a child process: a library whose constructor crashes must cost one
# failed check, not the doctor itself. RTLD_NOW resolves every symbol up
# front, so an ABI mismatch surfaces here rather than mid-profile.
_LOAD_SCRIPT = (
    "import ctypes, os, sys\n"
    "try:\n"
    "    ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW | os.RTLD_LOCAL)\n"
    "except OSError as exc:\n"
    "    sys.stdout.write(str(exc))\n"
    "    sys.exit(3)\n"
)

# Loading must stay side-effect free. The SDK's own constructor initializes
# only when ROCPROFILER_LIBRARY_CTOR is set, and the rocprofv3 tool library
# starts profiling only through its main() wrapper, which is active only when
# the library is LD_PRELOADed -- so keep both off in the child.
_LOAD_ENV = {
    "LD_PRELOAD": "",
    "ROCPROFILER_LIBRARY_CTOR": "0",
    "ROCPROFILER_LIBRARY_DTOR": "0",
}

# (library, subdirs, required, sdk_owned): a failure to load a required
# library fails the check; an optional one only warns. sdk_owned selects the
# soversion rule (see doctor_checks_install.resolve_library).
LOAD_TARGETS = (
    ("librocprofiler-sdk.so", ("lib", "lib64"), True, True),
    (
        "librocprofiler-sdk-tool.so",
        ("lib/rocprofiler-sdk", "lib64/rocprofiler-sdk"),
        True,
        True,
    ),
    ("libhsa-runtime64.so", ("lib", "lib64"), True, False),
    ("libhsa-amd-aqlprofile64.so", ("lib", "lib64"), True, False),
    ("libamdhip64.so", ("lib", "lib64"), False, False),
    ("librocprofiler-sdk-roctx.so", ("lib", "lib64"), False, True),
)


def load_environment(accessor):
    """The environment the probe child loads libraries under.

    Matches what rocprofv3 gives the profiled process, so a library that
    resolves its dependencies only through the launcher's search path is not
    reported as broken. SYNC: source/bin/rocprofv3.py run(), which appends
    "$ROCM_DIR/lib" to LD_LIBRARY_PATH (ROCM_DIR being the prefix rocprofv3
    is installed under -- the root inspected here). The one deliberate
    difference is LD_PRELOAD, cleared to keep the probe side-effect free.
    """
    env = dict(_LOAD_ENV)
    rocm_lib = accessor.join(accessor.rocm_root, "lib")
    current = accessor.getenv("LD_LIBRARY_PATH", "") or ""
    env["LD_LIBRARY_PATH"] = "{}:{}".format(current, rocm_lib) if current else rocm_lib
    return env


def missing_dependencies(accessor, path):
    """Every ``=> not found`` dependency of ``path`` according to ldd.

    The loader reports only the first missing dependency; ldd lists them all,
    which saves the user a fix-one-rerun loop. Best effort: [] without ldd.
    """
    returncode, stdout, _ = accessor.run(["ldd", path], timeout=10)
    if returncode != 0:
        return []
    missing = []
    for line in stdout.splitlines():
        if "=> not found" in line:
            missing.append(line.split("=>", 1)[0].strip())
    return missing


def load_library(accessor, path):
    """Load ``path`` in a child process; return a dict describing the outcome."""
    returncode, stdout, stderr = accessor.run(
        [accessor.python_executable(), "-I", "-c", _LOAD_SCRIPT, path],
        timeout=20,
        env=load_environment(accessor),
    )
    outcome = {"path": path, "returncode": returncode, "loaded": returncode == 0}
    if returncode == 0:
        return outcome

    error = (stdout.strip() or stderr.strip())[-600:]
    outcome["error"] = error
    outcome["outcome"] = describe_returncode(returncode)
    outcome["diagnoses"] = diagnose(accessor, error, returncode)
    if "cannot open shared object file" in error:
        outcome["missing_dependencies"] = missing_dependencies(accessor, path)
    return outcome


def _describe_load_failure(name, outcome):
    if outcome["diagnoses"]:
        reason = "; ".join(item["summary"] for item in outcome["diagnoses"])
    elif outcome["returncode"] == 3:
        reason = outcome["error"].splitlines()[0] if outcome["error"] else "unknown error"
    else:
        reason = "the loader process " + outcome["outcome"]
    missing = outcome.get("missing_dependencies") or []
    if len(missing) > 1:
        reason += " (all missing: {})".format(", ".join(missing))
    return "{}: {}".format(name, reason)


@register(
    id="runtime.libraries-load",
    group="runtime",
    title="ROCm libraries load and resolve all their symbols",
    severity=SEV_ERROR,
    depends=["install.sdk-library"],
    order=35,
    probe=PROBE_PROCESS,
)
def check_libraries_load(accessor):
    # A library being on disk is not enough: a missing dependency, a mixed-up
    # ROCm version, or an old libstdc++ all leave the file in place yet make
    # rocprofv3 die at startup. Actually loading each one is what catches them.
    results = {}
    required_failures = []
    optional_failures = []
    remediation = []

    for name, subdirs, required, sdk_owned in LOAD_TARGETS:
        path, _ = resolve_library(accessor, name, subdirs, sdk_owned=sdk_owned)
        if path is None:
            # absence is already reported by the install.* / runtime.* checks
            results[name] = {"path": None, "loaded": False, "skipped": "not found"}
            continue
        outcome = load_library(accessor, path)
        results[name] = outcome
        if outcome["loaded"]:
            continue

        (required_failures if required else optional_failures).append(
            _describe_load_failure(name, outcome)
        )
        for item in outcome["diagnoses"]:
            if item["remediation"] not in remediation:
                remediation.append(item["remediation"])

    data = {
        "libraries": results,
        "probe_environment": {
            "LD_LIBRARY_PATH": load_environment(accessor)["LD_LIBRARY_PATH"],
            "LD_PRELOAD": "",
        },
    }
    loaded = [name for name, outcome in results.items() if outcome.get("loaded")]
    if not required_failures and not optional_failures:
        if not loaded:
            return make_skip("none of the libraries could be located", "", data)
        return make_pass("{} libraries load cleanly".format(len(loaded)), "", data)

    if not remediation:
        remediation.append(reinstall_hint(accessor))
    remediation.append(
        "To see the loader's full search for one library:\n"
        "  LD_DEBUG=libs python3 -c \"import ctypes; ctypes.CDLL('<path>')\""
    )
    failures = required_failures + optional_failures
    factory = make_fail if required_failures else make_warn
    return factory(
        "failed to load -- " + "; ".join(failures), "\n\n".join(remediation), data
    )
