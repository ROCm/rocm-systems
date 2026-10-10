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


"""Turn the error output of a failed command into evidence-backed diagnoses.

Checks that run something -- loading a library, ``rocprofv3-avail``, a smoke
test -- hand its exit status and output to :func:`explain_failure`.

A regex match is evidence, not a conclusion. Each diagnosis therefore keeps
three things apart:

* ``observation`` -- what the output literally says (certain)
* ``cause``       -- the candidate explanation, if one is needed
* ``confidence``  -- ``high`` when facts gathered from this machine
                     corroborate the cause (or no inference is involved),
                     ``possible`` when the message is consistent with several
                     causes and nothing here narrows it down

Remediation is generated only for the mechanism actually implicated; when the
cause is uncertain, it names the next check that would tell the causes apart.

Every pattern matches text a real component prints; the comment on each names
the source, so a pattern can be re-checked when that message changes.
"""

from __future__ import absolute_import

import re

from rocprofv3.doctor_env import describe_returncode, signal_name
from rocprofv3.doctor_layout import physical_root, reinstall_hint, same_installation

HIGH = "high"
POSSIBLE = "possible"

KFD_DEVICE = "/dev/kfd"

# ROCm's own libraries, as opposed to system dependencies such as libdrm or
# libnuma: a missing one means a broken ROCm install rather than a missing
# distribution package.
_ROCM_LIBRARY_RE = re.compile(
    r"^lib(hsa-|hsakmt|amdhip|amd_comgr|hiprtc|rocprof|roctx|roctracer|"
    r"rocm|rocprofiler|rocprof-trace-decoder)"
)

# PyPI distribution names that differ from the import name.
_PIP_NAMES = {"yaml": "pyyaml"}

_VISIBILITY_VARS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
_TOOL_INJECTION_VARS = ("LD_PRELOAD", "HSA_TOOLS_LIB", "ROCP_TOOL_LIBRARIES")

_CRASH_SIGNALS = ("SIGSEGV", "SIGBUS", "SIGILL", "SIGABRT", "SIGFPE")

_MAX_EVIDENCE = 240


def _finding(observation, remediation, cause="", confidence=HIGH):
    return {
        "observation": observation,
        "cause": cause,
        "confidence": confidence,
        "remediation": remediation,
    }


def _foreign_rocm_entries(accessor):
    """LD_LIBRARY_PATH / LD_PRELOAD entries belonging to another ROCm install."""
    foreign = []
    for var in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        for entry in (accessor.getenv(var, "") or "").replace(" ", ":").split(":"):
            if not entry:
                continue
            directory = entry if var == "LD_LIBRARY_PATH" else accessor.dirname(entry)
            is_rocm = "rocm" in entry or physical_root(accessor, directory) is not None
            if is_rocm and not same_installation(accessor, directory, accessor.rocm_root):
                foreign.append("{}={}".format(var, entry))
    return foreign


# ----------------------------------------------------------------------
# assessments: (accessor, match, text) -> finding
# ----------------------------------------------------------------------
def _missing_dependency(accessor, match, text):
    library = match.group("library")
    observation = "a library it depends on, {}, cannot be found".format(library)
    if _ROCM_LIBRARY_RE.match(library):
        return _finding(
            observation,
            "{} is part of ROCm, so the installation is incomplete or its lib\n"
            "directory is not on the loader search path.\n{}".format(
                library, reinstall_hint(accessor)
            ),
        )
    return _finding(
        observation,
        "Install the system package that provides {0}:\n"
        "  apt-file search {0}      # Debian/Ubuntu\n"
        "  dnf provides '*/{0}'     # RHEL/Fedora/SLES".format(library),
    )


def _undefined_symbol(accessor, match, text):
    observation = "symbol {} is unresolved".format(match.group("symbol"))
    foreign = _foreign_rocm_entries(accessor)
    if foreign:
        return _finding(
            observation,
            "Remove the entries that point at another ROCm installation:\n  "
            + "\n  ".join(foreign),
            cause="libraries from another ROCm installation are loaded ({})".format(
                "; ".join(foreign)
            ),
        )
    return _finding(
        observation,
        "Find which file provides the symbol and where it was loaded from:\n"
        "  LD_DEBUG=libs,bindings <command> 2>&1 | grep -i 'symbol\\|error'\n"
        "Nothing in LD_LIBRARY_PATH or LD_PRELOAD points at another ROCm install,\n"
        "so check for libraries copied or partially upgraded inside {}.".format(
            accessor.rocm_root
        ),
        cause="libraries built from different ROCm versions are mixed",
        confidence=POSSIBLE,
    )


def _symbol_version(accessor, match, text):
    version = match.group("version")
    runtime = "libstdc++" if version.startswith(("GLIBCXX", "CXXABI")) else "glibc"
    conda = accessor.getenv("CONDA_PREFIX")
    if conda:
        remediation = (
            "A conda environment is active ({}); its bundled {} is the usual\n"
            "cause. Either update it:\n"
            "  conda install -c conda-forge 'libstdcxx-ng>=12'\n"
            "or keep {}/lib off LD_LIBRARY_PATH when profiling.".format(
                conda, runtime, conda
            )
        )
    else:
        remediation = (
            "Look for an old toolchain or bundled runtime on LD_LIBRARY_PATH:\n"
            "  echo $LD_LIBRARY_PATH\n"
            "If there is none, this OS is older than the ROCm build supports."
        )
    return _finding(
        "the {} being loaded lacks {}, which this ROCm build requires".format(
            runtime, version
        ),
        remediation,
    )


def _wrong_elf_class(accessor, match, text):
    return _finding(
        "a library built for the wrong architecture ({}) was found first".format(
            match.group("elf_class")
        ),
        "Remove 32-bit library directories (lib32, i386-linux-gnu, ...) from\n"
        "LD_LIBRARY_PATH.",
    )


def _bad_file(accessor, match, text):
    return _finding(
        "a library file is not a valid shared object",
        "The file is truncated or corrupt.\n" + reinstall_hint(accessor),
    )


def _out_of_resources(accessor, match, text):
    # ROCr returns this status both when /dev/kfd cannot be opened and for
    # genuine allocation failures (e.g. PM4 buffers in amd_aql_queue.cpp), so
    # look at /dev/kfd before naming a cause.
    observation = "the HSA runtime reported HSA_STATUS_ERROR_OUT_OF_RESOURCES"
    if not accessor.path_exists(KFD_DEVICE):
        return _finding(
            observation,
            "Load the amdgpu driver, or in a container pass the devices through:\n"
            "  docker run --device=/dev/kfd --device=/dev/dri --group-add video ...",
            cause="/dev/kfd does not exist, so no GPU can be opened",
        )
    if not (accessor.path_readable(KFD_DEVICE) and accessor.path_writable(KFD_DEVICE)):
        return _finding(
            observation,
            "Add the current user to the groups that own the GPU device nodes:\n"
            "  sudo usermod -a -G render,video $USER   # then log in again",
            cause="the current user cannot open /dev/kfd read-write",
        )
    return _finding(
        observation,
        "/dev/kfd is accessible, so this is not a permission problem. Check\n"
        "GPU memory use and other GPU processes:\n"
        "  amd-smi monitor        # or: rocm-smi --showmemuse --showpids\n"
        "and the locked-memory limit:\n"
        "  ulimit -l",
        cause="a genuine resource limit, such as exhausted GPU memory",
        confidence=POSSIBLE,
    )


def _no_device(accessor, match, text):
    observation = "HIP sees no GPU"
    masks = [
        "{}={}".format(var, accessor.getenv(var))
        for var in _VISIBILITY_VARS
        if accessor.getenv(var) is not None
    ]
    if masks:
        return _finding(
            observation,
            "Unset or correct the mask:\n  unset "
            + " ".join(mask.split("=", 1)[0] for mask in masks),
            cause="a device visibility mask is set ({})".format("; ".join(masks)),
        )
    return _finding(
        observation,
        "No visibility mask is set. Check that the driver enumerates a GPU:\n"
        "  rocm_agent_enumerator",
        cause="no GPU is usable by this process",
        confidence=POSSIBLE,
    )


def _kfd_permission(accessor, match, text):
    return _finding(
        "opening /dev/kfd was refused",
        "Add the current user to the groups that own the GPU device nodes:\n"
        "  sudo usermod -a -G render,video $USER\n"
        "then log out and back in (or run `newgrp render`).",
    )


def _device_busy(accessor, match, text):
    return _finding(
        "the GPU's profiler lock is held by another process; counter values may "
        "be inaccurate",
        "Stop the other profiling session first:\n"
        "  ps -eo pid,cmd | grep -E 'rocprof|omniperf|rocprofiler-compute'",
    )


def _device_lock_permission(accessor, match, text):
    # SDK: counters/ioctl.cpp, KFD_IOC_PROFILER_PMC lock failing with EPERM.
    # This is a capability check in KFD; perf_event_paranoid does not govern it.
    return _finding(
        "the GPU could not be locked for profiling without CAP_PERFMON. Dispatch "
        "counter values may be inaccurate if another process profiles the same "
        "GPU, and device-wide (system) counter collection is degraded",
        "Only needed for exact counter values under contention or for\n"
        "device-wide counter collection. Run with CAP_PERFMON, for example as\n"
        "root:\n"
        "  sudo -E rocprofv3 ...\n"
        "Setting file capabilities on rocprofv3 has no effect: it is a Python\n"
        "script, and Linux ignores file capabilities on scripts.",
    )


def _old_kfd(accessor, match, text):
    return _finding(
        "the amdgpu kernel driver does not support this rocprofiler-sdk feature",
        "Update the amdgpu driver to the version matching this ROCm release\n"
        "(for example `sudo amdgpu-install --usecase=dkms`), or use a newer\n"
        "distribution kernel.",
    )


def _context_conflict(accessor, match, text):
    observation = "another rocprofiler-sdk context already holds the requested service"
    injected = [
        "{}={}".format(var, accessor.getenv(var))
        for var in _TOOL_INJECTION_VARS
        if accessor.getenv(var)
    ]
    if injected:
        return _finding(
            observation,
            "Unset the variable that injects the other tool:\n  unset "
            + " ".join(item.split("=", 1)[0] for item in injected),
            cause="another tool is injected through the environment ({})".format(
                "; ".join(injected)
            ),
        )
    return _finding(
        observation,
        "Nothing in the environment injects another tool; check whether the\n"
        "application loads a profiler itself.",
        cause="another profiling tool is active in the process",
        confidence=POSSIBLE,
    )


def _missing_module(accessor, match, text):
    module = match.group("module").split(".")[0]
    observation = "Python module {} is missing".format(match.group("module"))
    if module == "rocprofv3":
        return _finding(
            observation,
            "The rocprofv3 Python package is not on the import path:\n"
            "  export PYTHONPATH={}/lib/python3/site-packages:$PYTHONPATH".format(
                accessor.rocm_root
            ),
        )
    return _finding(
        observation,
        "Install it:\n  python3 -m pip install {}".format(_PIP_NAMES.get(module, module)),
    )


# (id, pattern, assessment, related check ids)
#
# Every matching signature is reported, so a loader error and the Python
# traceback wrapping it both contribute.
SIGNATURES = (
    (
        # glibc dlerror(); also what ctypes.CDLL raises
        "loader.missing-dependency",
        re.compile(r"(?P<library>[\w.+-]+): cannot open shared object file"),
        _missing_dependency,
        ("install.no-mixed-rocm",),
    ),
    (
        # glibc dlerror() for an unresolved symbol under RTLD_NOW
        "loader.undefined-symbol",
        re.compile(r"undefined symbol: (?P<symbol>\S+)"),
        _undefined_symbol,
        ("install.no-mixed-rocm", "install.version-consistency"),
    ),
    (
        # glibc: "version `GLIBCXX_3.4.30' not found (required by ...)"
        "loader.symbol-version",
        re.compile(
            r"version [`'](?P<version>(?:GLIBCXX|CXXABI|GLIBC)_[\d.]+)' not found"
        ),
        _symbol_version,
        (),
    ),
    (
        "loader.wrong-elf-class",
        re.compile(r"wrong ELF class: (?P<elf_class>ELFCLASS\d+)"),
        _wrong_elf_class,
        (),
    ),
    (
        "loader.bad-file",
        re.compile(r"invalid ELF header|file too short"),
        _bad_file,
        (),
    ),
    (
        # ROCr hsa_status_string(), runtime/hsa-runtime/core/runtime/hsa.cpp
        "hsa.out-of-resources",
        re.compile(r"HSA_STATUS_ERROR_OUT_OF_RESOURCES"),
        _out_of_resources,
        ("driver.kfd-readable", "container.device-passthrough"),
    ),
    (
        # CLR hipGetErrorString(hipErrorNoDevice)
        "hip.no-device",
        re.compile(r"no ROCm-capable device is detected|hipErrorNoDevice"),
        _no_device,
        ("environ.gpu-visibility", "driver.gpu-topology"),
    ),
    (
        "kfd.permission-denied",
        re.compile(r"/dev/kfd[^\n]*Permission denied|Permission denied[^\n]*/dev/kfd"),
        _kfd_permission,
        ("driver.kfd-readable", "driver.render-group"),
    ),
    (
        # SDK counters/ioctl.cpp, profiler lock failing with EBUSY
        "sdk.device-busy",
        re.compile(r"has a profiler attached to it"),
        _device_busy,
        ("counters.no-profiler-lock",),
    ),
    (
        # SDK counters/ioctl.cpp (EPERM) and its status string in rocprofiler.cpp
        "sdk.device-lock-permission",
        re.compile(
            r"could not be locked for profiling due to lack of permissions|"
            r"ROCPROFILER_STATUS_ERROR_PERMISSION_DENIED|"
            r"Required permission \(CAP_PERFMON\) is not set"
        ),
        _device_lock_permission,
        ("counters.cap-perfmon",),
    ),
    (
        # SDK status string (rocprofiler.cpp) and ioctl.cpp EINVAL warning
        "sdk.incompatible-kernel",
        re.compile(
            r"ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL|"
            r"depends on a newer version of KFD|"
            r"Driver/Kernel version does not support locking device"
        ),
        _old_kfd,
        ("driver.amdgpu-module",),
    ),
    (
        "sdk.context-conflict",
        re.compile(
            r"ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT|"
            r"Context has a conflict with another context"
        ),
        _context_conflict,
        (
            "environ.ld-preload-conflict",
            "environ.hsa-tools-lib",
            "environ.rocp-tool-libraries",
        ),
    ),
    (
        # Python ImportError / ModuleNotFoundError
        "python.missing-module",
        re.compile(r"No module named '(?P<module>[\w.]+)'"),
        _missing_module,
        (),
    ),
)


def _evidence(text, match):
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    line = text[start : end if end != -1 else len(text)].strip()
    return line[:_MAX_EVIDENCE]


def _summary(finding):
    if not finding["cause"]:
        return finding["observation"]
    label = "likely cause" if finding["confidence"] == HIGH else "possible cause"
    return "{} ({}: {})".format(finding["observation"], label, finding["cause"])


def diagnose(accessor, text, returncode=None):
    """Match ``text`` against the known signatures; return a list of diagnoses.

    Each diagnosis is a dict with ``id``, ``observation``, ``cause``,
    ``confidence``, ``summary``, ``remediation``, ``related_checks`` and
    ``evidence`` (the line that matched). A crash with no recognisable message
    still yields a low-confidence diagnosis, since a crash is itself evidence.
    """
    text = text or ""
    found = []
    seen = set()
    for sig_id, pattern, assess, related in SIGNATURES:
        match = pattern.search(text)
        if match is None:
            continue
        key = (sig_id, match.group(0))
        if key in seen:
            continue
        seen.add(key)
        finding = assess(accessor, match, text)
        finding.update(
            {
                "id": sig_id,
                "summary": _summary(finding),
                "related_checks": list(related),
                "evidence": _evidence(text, match),
            }
        )
        found.append(finding)

    name = signal_name(returncode)
    if not found and name in _CRASH_SIGNALS:
        finding = _finding(
            "it crashed with {}".format(name),
            "Re-run it with SDK logging to see how far it gets:\n"
            "  ROCPROFILER_LOG_LEVEL=info <command>",
            cause="libraries from different installations are mixed",
            confidence=POSSIBLE,
        )
        finding.update(
            {
                "id": "process.crashed",
                "summary": _summary(finding),
                "related_checks": ["install.no-mixed-rocm", "runtime.libraries-load"],
                "evidence": "",
            }
        )
        found.append(finding)
    return found


def _last_line(text):
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()[:_MAX_EVIDENCE]
    return ""


def explain_failure(accessor, what, returncode, stdout, stderr, fallback):
    """Describe a failed command; return ``(detail, remediation, diagnoses)``.

    ``what`` names the command ("rocprofv3-avail info"). When nothing in the
    output is recognised, the detail keeps the last line of raw output as
    evidence, and ``fallback`` plus SDK logging is the next step suggested.
    """
    output = "{}\n{}".format(stderr or "", stdout or "")
    diagnoses = diagnose(accessor, output, returncode)
    detail = "{} {}".format(what, describe_returncode(returncode))
    if not diagnoses:
        last = _last_line(stderr) or _last_line(stdout)
        if last:
            detail += "; last output: {}".format(last)
        return (
            detail,
            fallback + "\nFor more detail, set ROCPROFILER_LOG_LEVEL=info.",
            diagnoses,
        )

    detail += ": " + "; ".join(item["summary"] for item in diagnoses)
    remediation = []
    for item in diagnoses:
        if item["remediation"] not in remediation:
            remediation.append(item["remediation"])
    related = []
    for item in diagnoses:
        for check_id in item["related_checks"]:
            if check_id not in related:
                related.append(check_id)
    if related:
        remediation.append("Related checks: {}".format(", ".join(related)))
    return (detail, "\n\n".join(remediation), diagnoses)
