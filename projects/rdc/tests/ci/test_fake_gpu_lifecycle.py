"""rdcd and embedded RDC start and stop against the fake amd-smi.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import os
import pwd
import shutil
import socket
import stat
import subprocess
import sys

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (tools is a fixture)
    FACTOR,
    PREFIX,
    Rdcd,
    _first_file,
    _free_port,
    _missing,
    _rdcd_lock_problem,
    tools,
)

ROOT_LOCK = "/var/run/rdcd.lock"  # kDaemonLockFileRoot in server/src/rdc_server_main.cc
USER_LOCK = "/tmp/rdcd.lock"  # kDaemonLockFile

# Sets a 400 W power limit on GPU 0 (the fake accepts 300 to 750 W) through
# embedded RDC, or through rdcd at argv[2], then stops. Prints "ok" if every
# call succeeded.
_CONFIG_SET = r"""
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
server = sys.argv[2] if len(sys.argv) > 2 else None
class Setting(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("target_value", ctypes.c_uint64)]
lib.rdc_config_set.argtypes = [ctypes.c_void_p, ctypes.c_uint32, Setting]
def call(name, *args):
    st = getattr(lib, name)(*args)
    if st != 0:
        sys.exit(f"{name} returned {st}")
handle, group = ctypes.c_void_p(), ctypes.c_uint32()
call("rdc_init", ctypes.c_uint64(0))
if server:
    call("rdc_connect", server.encode(), ctypes.byref(handle), None, None, None)
else:
    call("rdc_start_embedded", 0, ctypes.byref(handle))
call("rdc_group_gpu_create", handle, 1, b"config_group", ctypes.byref(group))  # RDC_GROUP_EMPTY
call("rdc_group_gpu_add", handle, group, 0)
call("rdc_config_set", handle, group, Setting(2, 400))  # RDC_CFG_POWER_LIMIT, watts
call("rdc_disconnect" if server else "rdc_stop_embedded", handle)
call("rdc_shutdown")
print("ok")
"""


def _config_set(env, server=None):
    bootstrap = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if bootstrap is None:
        _missing("librdc_bootstrap.so")
    try:
        return subprocess.run(
            [sys.executable, "-c", _CONFIG_SET, bootstrap] + ([server] if server else []),
            env=env,
            capture_output=True,
            text=True,
            timeout=60 * FACTOR,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("config child hung (killed)")


def _run(cmd, env, what):
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30 * FACTOR)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"rdcd kept running ({what})\n{e.stdout}{e.stderr}")
    return r.returncode, r.stdout + r.stderr


def test_rdcd_exits_with_an_error_when_its_port_is_in_use(tools, tmp_path):
    d = Rdcd(tools, tmp_path)
    # rdcd checks both lock files; it closed fd -1 for a missing one and said so.
    a_lock_file_is_missing = not (os.path.exists(ROOT_LOCK) and os.path.exists(USER_LOCK))
    # gRPC binds with SO_REUSEPORT, which works only if every socket on the port
    # sets it; this one does not, so rdcd cannot bind.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("0.0.0.0", 0))
        busy.listen()
        port = busy.getsockname()[1]
        code, out = _run([d.rdcd_path, "-u", "-p", str(port)], d.env, "port in use")
    assert code == 1, f"rdcd exited with {code}\n{out[-3000:]}"
    assert f"Failed to listen on 0.0.0.0:{port}" in out, out[-3000:]
    assert "Server listening" not in out, out[-3000:]
    if a_lock_file_is_missing:
        assert "Bad file descriptor" not in out, out[-3000:]


def test_embedded_rdc_stops_with_a_config_set(tools):
    fake = tools[2]
    r = _config_set(dict(os.environ, LD_PRELOAD=fake, RDC_FAKE_GPUS="2"))
    assert r.returncode == 0, f"embedded child exited {r.returncode}\n{r.stdout}{r.stderr[-3000:]}"
    assert "terminate called" not in r.stderr, r.stderr[-3000:]
    assert r.stdout.strip().endswith("ok"), r.stdout


def test_rdcd_stops_with_a_config_set(tools, tmp_path):
    d = Rdcd(tools, tmp_path).start()
    try:
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "LD_PRELOAD" and not k.startswith("RDC_FAKE_")
        }
        r = _config_set(env, f"127.0.0.1:{d.port}")
        assert r.returncode == 0, (
            f"config child exited {r.returncode}\n{r.stdout}{r.stderr[-3000:]}"
        )
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code} after SIGTERM\n{d.output()}"
    assert "RDC server successfully shut down." in d.output(), d.output()


@pytest.mark.parametrize(
    "mode, message",
    [(0o644, f"{USER_LOCK} owned by user: root"), (0o600, f"Failed to open lock file {USER_LOCK}")],
    ids=["readable", "unreadable"],
)
def test_rdcd_reports_a_lock_file_of_another_user(tools, tmp_path, mode, message):
    # rdcd run as nobody checks root's lock, then opens its own: USER_LOCK, here root's.
    if os.geteuid() != 0:
        pytest.skip("needs root, to run rdcd as another user against root's lock file")
    setpriv = shutil.which("setpriv")
    try:
        nobody = pwd.getpwnam("nobody")
    except KeyError:
        nobody = None
    if setpriv is None or nobody is None:
        pytest.skip("needs setpriv and the user nobody")
    if os.path.exists(ROOT_LOCK) and not os.stat(ROOT_LOCK).st_mode & stat.S_IROTH:
        pytest.skip(f"{ROOT_LOCK} is not readable by other users")
    for lock in (ROOT_LOCK, USER_LOCK):
        if _rdcd_lock_problem(lock, takes_it=False):
            pytest.skip(f"another rdcd holds {lock}")
    try:
        with open(USER_LOCK, "x"):
            pass
    except FileExistsError:
        pytest.skip(f"{USER_LOCK} exists; this test creates its own")
    try:
        os.chmod(USER_LOCK, mode)
        d = Rdcd(tools, tmp_path)
        cmd = [setpriv, f"--reuid={nobody.pw_uid}", f"--regid={nobody.pw_gid}", "--clear-groups"]
        code, out = _run(cmd + [d.rdcd_path, "-u", "-p", str(_free_port())], d.env, "lock check")
    finally:
        os.remove(USER_LOCK)
    assert code == 1, f"rdcd exited with {code}\n{out[-3000:]}"
    assert message in out, out[-3000:]
