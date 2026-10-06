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

"""``SystemAccessor``: the single seam through which checks touch the system.

Check modules must never import ``os``, ``subprocess``, or call ``open()``
directly. Everything they need is a method here. That restriction is what makes
every check unit-testable on a machine with no GPU, no ROCm install, and no
particular kernel: a test subclasses ``SystemAccessor`` (or uses the
``FakeAccessor`` in the test suite) and overrides only the handful of methods
the check under test actually calls.
"""

from __future__ import absolute_import

import glob as _glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

# Every subprocess invocation gets a timeout. A hung child -- a real risk when
# the KFD driver is wedged -- must degrade into one failed check, not a hung
# diagnostic tool.
DEFAULT_TIMEOUT = 10

# run() return codes for outcomes that are not a process exit. Negative codes
# from 1 to 64 are already taken by "killed by signal N", so these sit well
# outside that range rather than overloading -1 (SIGHUP) and -2 (SIGINT).
RUN_TIMEOUT = -1001
RUN_SPAWN_FAILED = -1002

# Captured output kept per stream. The tail is kept: errors come last.
MAX_OUTPUT_CHARS = 256 * 1024

# Imports of optional packages run in a child: a native extension that aborts
# or hangs while importing must cost one check, not the whole report.
IMPORT_TIMEOUT = 30
_IMPORT_MARKER = "@@rocprofv3-doctor-import@@"
_IMPORT_SCRIPT = (
    "import json, sys\n"
    "sys.path[:] = json.loads(sys.argv[3])\n"
    "try:\n"
    "    module = __import__(sys.argv[1])\n"
    "    version = getattr(module, '__version__', None) or getattr(module, 'version', None)\n"
    "    info = {'ok': True, 'version': '' if version is None else str(version),\n"
    "            'file': getattr(module, '__file__', None)}\n"
    "except BaseException as exc:\n"
    "    info = {'ok': False, 'error': '{}: {}'.format(type(exc).__name__, exc)}\n"
    "sys.stdout.write('\\n' + sys.argv[2] + json.dumps(info))\n"
)


def signal_name(returncode):
    """``SIGSEGV`` for a returncode of -11, or None for any other outcome."""
    if (
        returncode is None
        or returncode >= 0
        or returncode in (RUN_TIMEOUT, RUN_SPAWN_FAILED)
    ):
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return "signal {}".format(-returncode)


def describe_returncode(returncode):
    """How a command ended, phrased to follow its name."""
    if returncode == RUN_TIMEOUT:
        return "timed out"
    if returncode == RUN_SPAWN_FAILED:
        return "could not be started"
    name = signal_name(returncode)
    if name:
        return "crashed ({})".format(name)
    return "exited {}".format(returncode)


# Well-known capability bit positions in the /proc/self/status CapEff mask.
CAP_SYS_ADMIN_BIT = 21
CAP_PERFMON_BIT = 38  # Linux 5.8+


class SystemAccessor(object):
    """Real-system implementation of the check-facing system interface."""

    def __init__(self, rocm_root, tool_version=None, rocm_root_source=None):
        self.rocm_root = rocm_root
        # how rocm_root was located (see doctor_layout.detect_rocm_root)
        self.rocm_root_source = rocm_root_source
        self.tool_version = tool_version
        self._ldconfig_cache = None
        self._import_cache = {}

    # ------------------------------------------------------------------
    # filesystem
    # ------------------------------------------------------------------
    def path_exists(self, path):
        try:
            return os.path.exists(path)
        except OSError:
            return False

    def path_is_dir(self, path):
        try:
            return os.path.isdir(path)
        except OSError:
            return False

    def path_readable(self, path):
        try:
            return os.access(path, os.R_OK)
        except OSError:
            return False

    def path_writable(self, path):
        try:
            return os.access(path, os.W_OK)
        except OSError:
            return False

    def path_executable(self, path):
        try:
            return os.access(path, os.X_OK)
        except OSError:
            return False

    def path_is_link(self, path):
        try:
            return os.path.islink(path)
        except OSError:
            return False

    def readlink(self, path):
        try:
            return os.readlink(path)
        except OSError:
            return ""

    def realpath(self, path):
        try:
            return os.path.realpath(path)
        except OSError:
            return path

    def path_within(self, path, root):
        """True when ``path`` is ``root`` or lies beneath it.

        Both sides are resolved first, and the comparison is anchored on a
        separator so that "/opt/rocm-6.2.0-alt/lib" is NOT judged to be inside
        "/opt/rocm-6.2" -- a plain startswith() would call that a match and let
        a genuinely mismatched ROCm tree pass as consistent.
        """
        resolved_path = self.realpath(path)
        resolved_root = self.realpath(root)
        if resolved_path == resolved_root:
            return True
        return resolved_path.startswith(resolved_root.rstrip(os.sep) + os.sep)

    def abspath(self, path):
        return os.path.abspath(path)

    def dirname(self, path):
        return os.path.dirname(path)

    def basename(self, path):
        return os.path.basename(path)

    def join(self, *parts):
        return os.path.join(*parts)

    def glob(self, pattern):
        try:
            return sorted(_glob.glob(pattern))
        except OSError:
            return []

    def read_file(self, path):
        """Read a text file. Returns None when unreadable for any reason.

        Returning None rather than raising keeps the sysfs/procfs-reading checks
        free of try/except noise; "could not read" is a normal, expected
        outcome for most of them.
        """
        try:
            with open(path, "r") as handle:
                return handle.read()
        except (IOError, OSError, UnicodeDecodeError):
            return None

    def touch_probe(self, path):
        """Create and remove a temp file in ``path``; True when it succeeds.

        mkstemp creates the file exclusively (O_CREAT | O_EXCL) under an
        unpredictable name, so the probe can never truncate an existing file
        or write through a symlink planted at a known name.
        """
        try:
            handle, probe = tempfile.mkstemp(prefix=".rocprofv3-doctor-probe-", dir=path)
        except (IOError, OSError):
            return False
        try:
            os.close(handle)
        finally:
            try:
                os.remove(probe)
            except OSError:
                pass
        return True

    def make_temp_dir(self, prefix="rocprofv3-doctor-"):
        """A new private directory owned by this run, or None."""
        try:
            return tempfile.mkdtemp(prefix=prefix)
        except (IOError, OSError):
            return None

    def remove_tree(self, path):
        shutil.rmtree(path, ignore_errors=True)

    def free_bytes(self, path):
        """Free bytes available to this user on the filesystem holding ``path``."""
        try:
            stat = os.statvfs(path)
        except OSError:
            return None
        return stat.f_bavail * stat.f_frsize

    # ------------------------------------------------------------------
    # environment
    # ------------------------------------------------------------------
    def getenv(self, key, default=None):
        return os.environ.get(key, default)

    def environ_items(self):
        return sorted(os.environ.items())

    # ------------------------------------------------------------------
    # subprocess
    # ------------------------------------------------------------------
    def run(self, cmd, timeout=DEFAULT_TIMEOUT, env=None):
        """Run ``cmd``; return ``(returncode, stdout, stderr)``.

        Never raises. A timeout yields ``RUN_TIMEOUT`` and a command that
        cannot be started ``RUN_SPAWN_FAILED``, each with an explanatory
        stderr; a negative code otherwise means "killed by that signal".

        The child leads its own process group, so a timeout (or Ctrl+C)
        kills everything it started -- rocprofv3 runs the application as a
        grandchild -- not just the immediate child. Each stream keeps its last
        MAX_OUTPUT_CHARS characters.
        """
        run_env = None
        if env is not None:
            run_env = dict(os.environ)
            run_env.update(env)
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=run_env,
                universal_newlines=True,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            return (RUN_SPAWN_FAILED, "", "{}".format(exc))

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            return (RUN_TIMEOUT, "", "timed out after {} seconds".format(timeout))
        except KeyboardInterrupt:
            self._kill_group(proc)
            raise
        except Exception as exc:  # noqa: BLE001
            self._kill_group(proc)
            return (RUN_SPAWN_FAILED, "", "{}".format(exc))

        return (
            proc.returncode,
            (stdout or "")[-MAX_OUTPUT_CHARS:],
            (stderr or "")[-MAX_OUTPUT_CHARS:],
        )

    @staticmethod
    def _kill_group(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001 -- best-effort reap
            pass

    # ------------------------------------------------------------------
    # user / group
    # ------------------------------------------------------------------
    def getpid(self):
        return os.getpid()

    def getuid(self):
        return os.getuid()

    def getgid(self):
        return os.getgid()

    def getgroups(self):
        try:
            return sorted(os.getgroups())
        except OSError:
            return []

    def get_group_gid(self, group_name):
        """gid of ``group_name``, or None when the group does not exist."""
        import grp

        try:
            return grp.getgrnam(group_name).gr_gid
        except KeyError:
            return None
        except Exception:  # noqa: BLE001 -- NSS backends can raise anything
            return None

    def get_group_members(self, group_name):
        """Member usernames of ``group_name``, or None when it does not exist."""
        import grp

        try:
            return list(grp.getgrnam(group_name).gr_mem)
        except KeyError:
            return None
        except Exception:  # noqa: BLE001
            return None

    def get_username(self):
        try:
            import pwd

            return pwd.getpwuid(os.getuid()).pw_name
        except Exception:  # noqa: BLE001
            return self.getenv("USER") or self.getenv("LOGNAME") or ""

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    def get_cap_eff(self):
        """Effective capability mask from /proc/self/status, or None."""
        content = self.read_file("/proc/self/status")
        if not content:
            return None
        for line in content.splitlines():
            if line.startswith("CapEff:"):
                try:
                    return int(line.split(":", 1)[1].strip(), 16)
                except ValueError:
                    return None
        return None

    # ------------------------------------------------------------------
    # python introspection
    # ------------------------------------------------------------------
    def python_version(self):
        return tuple(sys.version_info[:3])

    def python_executable(self):
        return sys.executable

    def _probe_import(self, module_name):
        """Import ``module_name`` in a child interpreter; cached per module.

        The child runs isolated (-I: no current directory, PYTHONPATH, or
        user site on its path) and then adopts exactly this process's
        sys.path, so it resolves modules as an in-process import would --
        including the rocprofv3 package this tool bootstrapped -- while a
        crash or hang stays contained in the child. A plain ``python -c``
        child would put the current directory first, letting a stray
        ``rocprofv3.py`` or ``pandas.py`` there shadow, and run instead of,
        the real package.
        """
        if module_name in self._import_cache:
            return self._import_cache[module_name]

        search_path = json.dumps([entry for entry in sys.path if entry])
        returncode, stdout, stderr = self.run(
            [
                sys.executable,
                "-I",
                "-c",
                _IMPORT_SCRIPT,
                module_name,
                _IMPORT_MARKER,
                search_path,
            ],
            timeout=IMPORT_TIMEOUT,
        )
        info = None
        if _IMPORT_MARKER in stdout:
            try:
                info = json.loads(stdout.rsplit(_IMPORT_MARKER, 1)[1])
            except ValueError:
                info = None
        if info is None:
            tail = stderr.strip().splitlines()[-1:] or [""]
            info = {
                "ok": False,
                "error": "importing {} {} {}".format(
                    module_name, describe_returncode(returncode), tail[0]
                ).strip(),
            }
        self._import_cache[module_name] = info
        return info

    def can_import(self, module_name):
        """Try to import ``module_name``; return ``(ok, detail)``.

        ``detail`` is the module's ``__version__``/``version`` when available on
        success, or the error (including a crash or timeout) on failure.
        """
        info = self._probe_import(module_name)
        if info["ok"]:
            return (True, info.get("version") or "")
        return (False, info.get("error") or "")

    def module_file(self, module_name):
        """Filesystem location of an importable module, or None."""
        info = self._probe_import(module_name)
        return info.get("file") if info["ok"] else None

    def find_module_dir(self, module_name):
        """Directory of top-level package ``module_name``, or None.

        Uses find_spec rather than an import so that locating a package (for
        example a TheRock ``_rocm_sdk_core`` wheel) never executes its code.
        """
        try:
            import importlib.util

            spec = importlib.util.find_spec(module_name)
        except Exception:  # noqa: BLE001 -- broken finders raise anything
            return None
        if spec is None or not spec.submodule_search_locations:
            return None
        return list(spec.submodule_search_locations)[0]

    def sqlite_usable(self):
        """True when sqlite3 imports *and* can open an in-memory database.

        Minimal container images ship a Python built without SQLite support;
        the import succeeds there but connecting does not.
        """
        try:
            import sqlite3

            conn = sqlite3.connect(":memory:")
            conn.close()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # library resolution
    # ------------------------------------------------------------------
    def ldconfig_entries(self):
        """Map of soname -> path from the ldconfig cache (empty when absent)."""
        if self._ldconfig_cache is not None:
            return self._ldconfig_cache

        cache = {}
        returncode, stdout, _ = self.run(["ldconfig", "-p"], timeout=DEFAULT_TIMEOUT)
        if returncode == 0:
            for line in stdout.splitlines():
                if "=>" not in line:
                    continue
                left, right = line.split("=>", 1)
                soname = left.strip().split(" ", 1)[0].strip()
                path = right.strip()
                if soname and soname not in cache:
                    cache[soname] = path
        self._ldconfig_cache = cache
        return cache

    def which(self, name):
        """Locate executable ``name`` on PATH; return ``(path_or_None, searched)``.

        Distro packages commonly symlink the ROCm tools into /usr/bin, and a
        non-default install prefix may not have a bin/ under the ROCm root at
        all, so a tool that is absent from ``rocm_root/bin`` may still be
        perfectly usable.
        """
        searched = []
        path_env = self.getenv("PATH", "") or ""
        for directory in path_env.split(os.pathsep):
            if not directory:
                continue
            candidate = os.path.join(directory, name)
            searched.append(candidate)
            if self.path_exists(candidate) and self.path_executable(candidate):
                return (candidate, searched)
        return (None, searched)

    def library_search_dirs(self):
        """Directories searched for libraries, in precedence order."""
        dirs = []
        ld_library_path = self.getenv("LD_LIBRARY_PATH", "") or ""
        for entry in ld_library_path.split(":"):
            if entry and entry not in dirs:
                dirs.append(entry)
        for suffix in ("lib", "lib64", "lib/rocprofiler-sdk"):
            candidate = os.path.join(self.rocm_root, suffix)
            if candidate not in dirs:
                dirs.append(candidate)
        return dirs

    def find_library(self, name):
        """Locate shared library ``name``; return ``(path_or_None, searched)``.

        Search order is LD_LIBRARY_PATH, then the ROCm tree, then the ldconfig
        cache -- matching how the dynamic loader would resolve it for a
        rocprofv3 run.
        """
        searched = []
        for directory in self.library_search_dirs():
            searched.append(directory)
            candidate = os.path.join(directory, name)
            if self.path_exists(candidate):
                return (candidate, searched)

        searched.append("ldconfig cache")
        cached = self.ldconfig_entries().get(name)
        if cached and self.path_exists(cached):
            return (cached, searched)
        return (None, searched)
