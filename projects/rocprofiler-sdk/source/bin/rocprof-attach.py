#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import argparse
import ctypes
import os
import re
import signal
import stat
import sys
import time

ROCPROF_ATTACH_DIR = os.path.dirname(os.path.realpath(__file__))
ROCM_DIR = os.path.dirname(ROCPROF_ATTACH_DIR)
ROCPROFILER_SDK_SOVERSION = "@PROJECT_VERSION_MAJOR@"
ROCPROFILER_SDK_VERSION = "@PROJECT_VERSION@"
ROCPROF_ATTACH_LIBRARY = (
    f"{ROCM_DIR}/lib/librocprofiler-sdk-rocattach.so.{ROCPROFILER_SDK_SOVERSION}"
)

# procfs mount point; replaced by the unit tests
PROC_DIR = "/proc"

# name of the thread rocprofiler-sdk-attach starts in the target; rocattach refuses to attach
# to a process without it
ATTACH_THREAD_NAME = "rocp-bg-attach"

TOOL_SUBDIR = "rocprofiler-sdk"
TOOL_BASENAME = "librocprofiler-sdk-tool.so"

_SDK_LIBRARY_RE = re.compile(
    r"^(?P<name>librocprofiler-sdk(?:-attach|-tool)?)\.so(?P<suffix>(?:\.\d+)*)$"
)


class AttachError(RuntimeError):
    """Attachment cannot succeed; the target has not been modified."""


def _info(msg):
    print(f"rocprof-attach: {msg}")
    sys.stdout.flush()


def _warning(msg):
    sys.stdout.flush()
    print(f"rocprof-attach: WARNING: {msg}", file=sys.stderr)
    sys.stderr.flush()


def _version_key(suffix):
    return tuple(int(itr) for itr in suffix.split(".") if itr)


class MappedLibrary:
    """A rocprofiler-sdk library mapped into the target process."""

    def __init__(self, path, deleted, suffix):
        self.path = path
        self.deleted = deleted
        self.suffix = suffix

    @property
    def directory(self):
        return os.path.dirname(self.path)

    @property
    def version(self):
        return self.suffix.lstrip(".") if self.suffix.count(".") == 3 else None


class TargetSdk:
    """The rocprofiler-sdk libraries found in /proc/<pid>/maps."""

    def __init__(self, maps_readable=False):
        self.maps_readable = maps_readable
        self.attach_libs = []
        self.core_libs = []
        self.tool_libs = []

    @property
    def libdir(self):
        """Directory of the target's rocprofiler-sdk installation.

        The attach library is the one that rocattach talks to, so its directory wins.
        """
        for itr in (self.attach_libs, self.core_libs):
            if itr:
                return itr[0].directory
        return None

    @property
    def version(self):
        for itr in self.attach_libs + self.core_libs:
            if itr.version:
                return itr.version
        return None

    @property
    def deleted(self):
        return any(itr.deleted for itr in self.attach_libs + self.core_libs)


def read_target_maps(pid):
    """Returns the list of (path, deleted) for files mapped into the target, or None."""
    fname = os.path.join(PROC_DIR, f"{pid}", "maps")
    result = []
    seen = set()
    try:
        with open(fname, "rb") as ifs:
            for line in ifs:
                fields = line.rstrip(b"\n").split(maxsplit=5)
                if len(fields) < 6 or not fields[5].startswith(b"/"):
                    continue
                path = os.fsdecode(fields[5])
                deleted = path.endswith(" (deleted)")
                if deleted:
                    path = path[: -len(" (deleted)")]
                if (path, deleted) not in seen:
                    seen.add((path, deleted))
                    result.append((path, deleted))
    except (OSError, ValueError):
        return None
    return result


def find_target_sdk(pid):
    maps = read_target_maps(pid)
    sdk = TargetSdk(maps_readable=maps is not None)
    for path, deleted in maps or []:
        m = _SDK_LIBRARY_RE.match(os.path.basename(path))
        if not m:
            continue
        lib = MappedLibrary(path, deleted, m.group("suffix"))
        if m.group("name") == "librocprofiler-sdk-attach":
            sdk.attach_libs.append(lib)
        elif m.group("name") == "librocprofiler-sdk-tool":
            sdk.tool_libs.append(lib)
        else:
            sdk.core_libs.append(lib)

    if sdk.attach_libs and sdk.core_libs:
        attach_dirs = sorted(set(itr.directory for itr in sdk.attach_libs))
        core_dirs = sorted(set(itr.directory for itr in sdk.core_libs))
        if attach_dirs != core_dirs:
            _warning(
                f"PID {pid} maps rocprofiler-sdk libraries from more than one directory "
                f"(attach library: {', '.join(attach_dirs)}; core library: "
                f"{', '.join(core_dirs)}). Using {sdk.libdir}"
            )
    return sdk


def _target_root(pid):
    return os.path.join(PROC_DIR, f"{pid}", "root")


def _target_view(pid, path):
    return _target_root(pid) + path


def target_stat(pid, path):
    """stat() an absolute path from within the target's mount namespace.

    Returns ("present", stat_result), ("absent", None), or ("unknown", None) when the target's
    filesystem cannot be inspected (e.g. procfs permissions).
    """
    try:
        return ("present", os.stat(_target_view(pid, path)))
    except (FileNotFoundError, NotADirectoryError):
        try:
            os.stat(_target_root(pid))
        except OSError:
            return ("unknown", None)
        return ("absent", None)
    except OSError:
        return ("unknown", None)


def _is_regular_file(pid, path):
    state, st = target_stat(pid, path)
    if state == "present":
        return "present" if stat.S_ISREG(st.st_mode) else "absent"
    return state


def _same_file(pid, lhs, rhs):
    if not os.path.isabs(lhs) or not os.path.isabs(rhs):
        return None
    lstate, lst = target_stat(pid, lhs)
    rstate, rst = target_stat(pid, rhs)
    if lstate != "present" or rstate != "present":
        return None
    return (lst.st_dev, lst.st_ino) == (rst.st_dev, rst.st_ino)


def _contains_bytes(fname, needle, chunk_size=1 << 20):
    overlap = len(needle) - 1
    tail = b""
    with open(fname, "rb") as ifs:
        while True:
            chunk = ifs.read(chunk_size)
            if not chunk:
                return False
            if needle in tail + chunk:
                return True
            tail = chunk[-overlap:]


def has_attach_thread(pid):
    """Mirrors resolve_attach_tid in rocprofiler-sdk-rocattach."""
    task_dir = os.path.join(PROC_DIR, f"{pid}", "task")
    try:
        tasks = os.listdir(task_dir)
    except OSError:
        return False
    for tid in tasks:
        try:
            with open(os.path.join(task_dir, tid, "comm"), "rb") as ifs:
                name = ifs.readline().rstrip(b"\n")
        except OSError:
            continue
        if name == ATTACH_THREAD_NAME.encode():
            return True
    return False


def check_attachable(pid, sdk, attach_children):
    """Raises AttachError only when rocattach is certain to refuse the target."""
    if has_attach_thread(pid):
        return

    if attach_children:
        if sdk.maps_readable and not sdk.attach_libs:
            _info(
                f"PID {pid} does not have rocprofiler-sdk attachment enabled; its descendant "
                "processes will be attached where possible"
            )
        else:
            _warning(
                f"PID {pid} has no '{ATTACH_THREAD_NAME}' thread; it will be skipped unless the "
                "thread starts before attachment. Its descendant processes will be attached "
                "where possible"
            )
        return

    if not sdk.maps_readable:
        _warning(
            f"could not read the memory maps of PID {pid} and it has no '{ATTACH_THREAD_NAME}' "
            "thread. Attachment will likely be refused"
        )
        return

    if not sdk.attach_libs:
        _warning(
            f"PID {pid} has not loaded the rocprofiler-sdk attach library and has no "
            f"'{ATTACH_THREAD_NAME}' thread. Start the target with ROCP_TOOL_ATTACH=1 to enable "
            "attachment"
        )
        return

    # the attach library is mapped: if none of the mapped copies can ever start the
    # attach thread, rocattach will refuse the target. Say why before touching it.
    marker = ATTACH_THREAD_NAME.encode()
    for itr in sdk.attach_libs:
        if itr.deleted:
            break
        try:
            if _contains_bytes(_target_view(pid, itr.path), marker):
                break
        except OSError:
            break
    else:
        libdir = sdk.libdir
        prefix = os.path.dirname(libdir)
        version = f" (v{sdk.version})" if sdk.version else ""
        raise AttachError(
            f"PID {pid} runs rocprofiler-sdk{version} from {libdir}, which predates support for "
            f"attaching via the '{ATTACH_THREAD_NAME}' thread. To profile this process, use "
            f"the rocprofv3 from the target's own installation in {prefix}/bin"
        )

    _warning(
        f"PID {pid} has no '{ATTACH_THREAD_NAME}' thread; attachment will be refused unless it "
        "starts first"
    )


def default_tool_library():
    """This installation's tool library, as rocprofv3 resolves it by default."""
    path = f"{ROCM_DIR}/lib/{TOOL_SUBDIR}/{TOOL_BASENAME}"
    for itr in (path, f"{path}.{ROCPROFILER_SDK_SOVERSION}"):
        if os.path.exists(itr):
            return itr
    return f"{path}.{ROCPROFILER_SDK_SOVERSION}.{ROCPROFILER_SDK_VERSION}"


def _find_target_tool(pid, libdir, attach_suffix):
    """Returns (path, state, note) for the tool library of the installation in libdir.

    Mirrors how rocprofv3 resolves its own tool library: the unversioned name, then the SONAME.
    """
    tooldir = f"{libdir}/{TOOL_SUBDIR}"
    soversion = (attach_suffix or "").lstrip(".").split(".")[
        0
    ] or ROCPROFILER_SDK_SOVERSION
    candidates = [f"{tooldir}/{TOOL_BASENAME}", f"{tooldir}/{TOOL_BASENAME}.{soversion}"]
    for itr in candidates:
        state = _is_regular_file(pid, itr)
        if state == "present":
            return (itr, "present", None)
        if state == "unknown":
            return (candidates[0], "unknown", None)

    # the unversioned links are missing: look for a fully versioned file
    try:
        entries = os.listdir(_target_view(pid, tooldir))
    except OSError:
        entries = []
    found = []
    for itr in entries:
        m = _SDK_LIBRARY_RE.match(itr)
        if not m or m.group("name") != "librocprofiler-sdk-tool" or not m.group("suffix"):
            continue
        path = f"{tooldir}/{itr}"
        if os.path.islink(_target_view(pid, path)):
            continue
        if _is_regular_file(pid, path) == "present":
            found.append((m.group("suffix"), path))
    if not found:
        return (None, "absent", None)
    for suffix, path in found:
        if attach_suffix and suffix == attach_suffix:
            return (path, "present", "the installation's tool library links are missing")
    found.sort(key=lambda itr: _version_key(itr[0]))
    return (found[-1][1], "present", "the installation's tool library links are missing")


def select_tool_library(pid, sdk, fallback_tool_library=None):
    """Returns (tool library path, source), where source is "target" or "fallback".

    Prefers the tool library of the rocprofiler-sdk installation the target is running,
    otherwise the fallback: this installation's tool library, or the one rocprofv3 resolved.
    Raises AttachError if the fallback is not visible from the target process.
    """
    fallback = fallback_tool_library or default_tool_library()
    libdir = sdk.libdir
    version = f" v{sdk.version}" if sdk.version else ""

    if libdir is not None:
        attach_suffix = sdk.attach_libs[0].suffix if sdk.attach_libs else None
        path, state, note = _find_target_tool(pid, libdir, attach_suffix)
        if state == "unknown":
            _warning(
                f"could not verify {path} from the filesystem of PID {pid}; proceeding with it"
            )
            return (path, "target")
        if path is not None:
            if _same_file(pid, path, fallback):
                # same library: keep the exact string earlier rocprofv3 releases passed so
                # that reattaching to a process they attached to is accepted
                path = fallback
            if note:
                _warning(f"{note}; using {path}")
            else:
                _info(
                    f"using the tool library of the target's rocprofiler-sdk{version} "
                    f"installation: {path}"
                )
            if sdk.deleted:
                _warning(
                    f"rocprofiler-sdk libraries mapped by PID {pid} have been deleted or replaced "
                    f"on disk since it started; {path} may not match the running version"
                )
            return (path, "target")

        _warning(
            f"the tool library of the rocprofiler-sdk{version} installation in {libdir} "
            f"used by PID {pid} was not found; using {fallback} instead"
        )
    elif not sdk.maps_readable:
        _warning(
            f"could not read the memory maps of PID {pid} to locate its rocprofiler-sdk "
            f"installation; using {fallback}"
        )
    else:
        _info(f"PID {pid} has not loaded rocprofiler-sdk; using {fallback}")

    if os.path.isabs(fallback) and target_stat(pid, fallback)[0] == "absent":
        raise AttachError(
            f"no tool library was found for PID {pid}: {fallback} is not visible from the target "
            "process. Specify one with --attach-tool-library"
        )
    return (fallback, "fallback")


def check_user_tool_library(pid, attach_tool_library):
    """Checks each library in the colon-delimited list.

    Raises AttachError if the list is empty or if any absolute path is not a file visible from
    the target process.
    """
    libraries = [itr for itr in attach_tool_library.split(":") if itr]
    if not libraries:
        raise AttachError(
            f"no tool library was given in the tool library list '{attach_tool_library}'"
        )
    for itr in libraries:
        if not os.path.isabs(itr):
            _info(f"tool library '{itr}' will be resolved by the target's dynamic loader")
            continue
        state = _is_regular_file(pid, itr)
        if state == "absent":
            raise AttachError(
                f"tool library '{itr}' is not a file visible from the target process PID {pid}"
            )
        elif state == "unknown":
            _warning(
                f"could not verify tool library '{itr}' from the filesystem of PID {pid}"
            )


def is_same_install(pid, sdk):
    """True if the target runs this installation's rocprofiler-sdk attach library."""
    own = f"{ROCM_DIR}/lib/librocprofiler-sdk-attach.so.{ROCPROFILER_SDK_SOVERSION}"
    if not sdk.attach_libs or sdk.deleted:
        return False
    return all(_same_file(pid, itr.path, own) for itr in sdk.attach_libs)


def warn_option_support(pid, sdk, source):
    if sdk.libdir is None or is_same_install(pid, sdk):
        return
    version = f" v{sdk.version}" if sdk.version else ""
    if source == "target":
        _warning(
            f"PID {pid} runs a different rocprofiler-sdk{version} installation than this one "
            f"(v{ROCPROFILER_SDK_VERSION}); its tool library ignores options it does not know. "
            "Check that the requested options exist in that version. Aggregate options such as "
            "--sys-trace, --runtime-trace, --hip-trace, --hsa-trace and --kfd-trace only enable "
            "what that version supports"
        )
    else:
        _warning(
            f"PID {pid} runs a different rocprofiler-sdk{version} installation than the tool "
            "library being attached. Tracing options that rocprofiler-sdk does not support are "
            "skipped with warnings in the target's log, and thread trace is disabled if it is "
            "older than v1.5.0. Select only options it supports, and prefer individual tracing "
            "options over aggregate options such as --sys-trace or --runtime-trace"
        )


def warn_loaded_tool(pid, sdk, tool_library):
    """Reattaching with a different tool library path string is refused by the target."""
    for itr in sdk.tool_libs:
        if _same_file(pid, itr.path, tool_library) is False:
            _warning(
                f"PID {pid} already has {itr.path} loaded from an earlier attachment. If that "
                "attachment used a different tool library path, this one will be refused; "
                "re-run with --attach-tool-library set to the earlier path"
            )
            return


def parse_arguments(args=None):

    def format_help(formatter, w=120, h=40):
        """Return a wider HelpFormatter, if possible."""
        try:
            kwargs = {"width": w, "max_help_position": h}
            formatter(None, **kwargs)
            return lambda prog: formatter(prog, **kwargs)
        except TypeError:
            return formatter

    usage_examples = """

%(prog)s, e.g.

    $ rocprof-attach -p <pid> [-t <tool library> -a <attach tool library> -d <msec duration>]
    $ rocprof-attach -p 12345 -d 5000
    $ rocprof-attach -p 12345 -t path/to/your-tool-library.so -d 5000

"""
    parser = argparse.ArgumentParser(
        description="rocprofiler-sdk attachment profiler. By default, the target is profiled "
        "with the rocprofiler-sdk tool library of the installation it is running",
        usage="%(prog)s [options] ",
        epilog=usage_examples,
        formatter_class=format_help(argparse.RawTextHelpFormatter),
    )

    parser.add_argument(
        "-p",
        "--pid",
        "--attach",
        help="""Attachment target's process identifier (PID).
  Can also be specified in environment variable ROCPROF_ATTACH_PID. This option overrides the environment variable if both are set.""",
        type=int,
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_PID", None),
    )

    parser.add_argument(
        "--attach-children",
        help="""Attach to the target process and all of its descendant processes (default: true).
  Can also be specified in environment variable ROCPROF_ATTACH_CHILDREN. This option overrides the environment variable if both are set.""",
        type=lambda v: v.lower() not in ("0", "false", "no", "off"),
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_CHILDREN", "1")
        not in ("0", "false", "no", "off"),
        metavar="BOOL",
    )

    parser.add_argument(
        "-t",
        "--attach-tool-library",
        help="""Colon delimited list of tool libraries to use during attachment. Paths are used as given and are resolved in the target process.
  Attachment fails if any library in the list cannot be found or loaded.
  When unset, the rocprofiler-sdk tool library of the installation the target process is running is used. If it cannot be located,
  the tool library of this installation is used with a warning. When attaching to process descendants, all of them use the selection made for the target PID.
  Can also be specified in environment variable ROCPROF_ATTACH_TOOL_LIBRARY. This option overrides the environment variable if both are set.""",
        type=str,
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_TOOL_LIBRARY", None),
    )

    parser.add_argument(
        "-d",
        "--attach-duration-msec",
        help="""Sets the amount of time in milliseconds the profiler will be attached before detaching. When unset, the profiler will wait until Enter is pressed or SIGINT (Ctrl+C) to detach.
  Can also be specified in environment variable ROCPROF_ATTACH_DURATION. This option overrides the environment variable if both are set.""",
        type=int,
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_DURATION", None),
    )

    advanced_options = parser.add_argument_group("Advanced options")

    advanced_options.add_argument(
        "--attach-library",
        help=f"""Library used to attach and detach from the target process. Default will work for nearly all configurations.
  Defaults to the rocprofiler-sdk-rocattach SONAME from this ROCm install, i.e. <ROCmdirectory>/lib/librocprofiler-sdk-rocattach.so.{ROCPROFILER_SDK_SOVERSION}
  Can also be specified in environment variable ROCPROF_ATTACH_LIBRARY. This option overrides the environment variable if both are set.""",
        type=str,
        required=False,
        default=os.environ.get("ROCPROF_ATTACH_LIBRARY", ROCPROF_ATTACH_LIBRARY),
    )

    # used by rocprofv3: the tool library it resolved for this installation
    advanced_options.add_argument(
        "--fallback-tool-library",
        help=argparse.SUPPRESS,
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args(args)


def attach(
    pid,
    attach_tool_library,
    attach_duration_msec,
    attach_library=ROCPROF_ATTACH_LIBRARY,
    attach_children=True,
    fallback_tool_library=None,
):

    if pid is None:
        raise RuntimeError("rocprof-attach called with no PID specified")

    user_tool_libraries = os.environ.pop("ROCP_TOOL_LIBRARIES", None)
    if user_tool_libraries is not None:
        _warning(
            f"ignoring ROCP_TOOL_LIBRARIES={user_tool_libraries} in attach mode. Use "
            "--attach-tool-library or ROCPROF_ATTACH_TOOL_LIBRARY to attach a custom tool library"
        )

    sdk = find_target_sdk(pid)
    check_attachable(pid, sdk, attach_children)

    if attach_tool_library:
        check_user_tool_library(pid, attach_tool_library)
    else:
        attach_tool_library, source = select_tool_library(pid, sdk, fallback_tool_library)
        warn_option_support(pid, sdk, source)
        warn_loaded_tool(pid, sdk, attach_tool_library)

    # Program option overrides environment variable. This is consumed by rocprofiler-sdk on the target program side.
    os.environ["ROCPROF_ATTACH_TOOL_LIBRARY"] = attach_tool_library

    print(f"Attaching to PID {pid} using library {attach_library}")

    # Load the shared library into ctypes and attach
    try:
        c_lib = ctypes.CDLL(attach_library)
        c_lib.rocattach_attach.restype = ctypes.c_int
        c_lib.rocattach_attach.argtypes = [ctypes.c_int]
        c_lib.rocattach_attach_tree.restype = ctypes.c_int
        c_lib.rocattach_attach_tree.argtypes = [ctypes.c_int]
        c_lib.rocattach_detach.restype = ctypes.c_int
        c_lib.rocattach_detach.argtypes = [ctypes.c_int]
        c_lib.rocattach_detach_tree.restype = ctypes.c_int
        c_lib.rocattach_detach_tree.argtypes = [ctypes.c_int]
        if attach_children:
            attach_status = c_lib.rocattach_attach_tree(pid)
        else:
            attach_status = c_lib.rocattach_attach(pid)
    except Exception as e:
        raise RuntimeError(f"Exception during library load and attachment: {e}")

    if attach_status != 0:
        raise RuntimeError(
            f"Calling attach in {attach_library} returned non-zero status {attach_status}"
        )

    print(f"Attaching to PID {pid} using library {attach_library} :: success")

    def detach():
        print("Detaching. Please wait, this can take up to 1-2 minutes")
        sys.stdout.flush()
        try:
            if attach_children:
                detach_status = c_lib.rocattach_detach_tree(int(pid))
            else:
                detach_status = c_lib.rocattach_detach(int(pid))
        except Exception as e:
            print(f"Exception during detachment: {e}")

        if detach_status != 0:
            print(
                f"Calling detach in {attach_library} returned non-zero status {detach_status}"
            )
        else:
            print(f"Detaching from PID {pid} using library {attach_library} :: success")

    def signal_handler(sig, frame):
        print("\nCaught signal SIGINT")
        detach()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if attach_duration_msec is None:
        sys.stdout.write("Press Enter to detach...")
        sys.stdout.flush()  # Force the prompt to appear immediately
        input()  # Now wait for input
    else:
        print(f"Attaching for {attach_duration_msec} msec...\n")
        sys.stdout.flush()
        time.sleep(int(attach_duration_msec) / 1000)

    detach()


def main(cmd_args=None):
    args = parse_arguments(cmd_args)

    try:
        attach(
            pid=args.pid,
            attach_tool_library=args.attach_tool_library,
            attach_duration_msec=args.attach_duration_msec,
            attach_library=args.attach_library,
            attach_children=args.attach_children,
            fallback_tool_library=args.fallback_tool_library,
        )
    except AttachError as e:
        sys.stdout.flush()
        print(f"rocprof-attach: ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    ec = main(sys.argv[1:])
    sys.exit(ec)
