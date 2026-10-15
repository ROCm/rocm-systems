"""rdcd and embedded RDC start and stop against the fake amd-smi.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import contextlib
import fcntl
import os
import pwd
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (tools is a fixture)
    FACTOR,
    PREFIX,
    Rdcd,
    _first_file,
    _free_port,
    _listening,
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
    # rdcd closed fd -1 for a missing lock file it checked, and said so. Only a non-root rdcd
    # checks a lock file it does not take (the root one).
    a_lock_file_is_missing = os.geteuid() != 0 and not os.path.exists(ROOT_LOCK)
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


@pytest.mark.parametrize("mode", [0o644, 0o600], ids=["readable", "unreadable"])
def test_rdcd_reports_a_lock_file_of_another_user(tools, tmp_path, mode):
    # rdcd run as nobody checks root's lock, then opens its own: USER_LOCK, here root's.
    message = f"{USER_LOCK} owned by user: root"
    if os.geteuid() != 0:
        pytest.skip("needs root, to run rdcd as another user against root's lock file")
    setpriv = shutil.which("setpriv")
    try:
        nobody = pwd.getpwnam("nobody")
    except KeyError:
        nobody = None
    if setpriv is None or nobody is None:
        pytest.skip("needs setpriv and the user nobody")
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


# rdcd's messages for a lock file it refuses to use, and for another rdcd that holds a lock.
REFUSED = "must be a regular file owned by"
ALREADY_RUNNING = "is locked. Is rdcd already running?"


def _need_root():
    if os.geteuid() != 0:
        pytest.skip("needs root: a root rdcd takes /var/run/rdcd.lock")


def _nobody():
    """setpriv arguments that run a command as the user nobody (the test runs as root)."""
    _need_root()
    setpriv = shutil.which("setpriv")
    try:
        nobody = pwd.getpwnam("nobody")
    except KeyError:
        nobody = None
    if setpriv is None or nobody is None:
        pytest.skip("needs setpriv and the user nobody")
    return nobody, [
        setpriv,
        f"--reuid={nobody.pw_uid}",
        f"--regid={nobody.pw_gid}",
        "--clear-groups",
    ]


@contextlib.contextmanager
def _set_aside(path):
    """Moves an unlocked lock file aside for the test, and puts it back afterwards."""
    if _rdcd_lock_problem(path, takes_it=False):
        pytest.skip(f"another rdcd holds {path}")
    saved = path + ".set-aside-by-test"
    if os.path.lexists(saved):
        pytest.skip(f"{saved} is left from an earlier run; put it back or remove it")
    had = os.path.lexists(path)
    if had:
        os.rename(path, saved)
    try:
        yield
    finally:
        if os.path.lexists(path):
            os.remove(path)
        if had:
            os.rename(saved, path)


def _start(cmd, env, port, **popen_args):
    """Starts rdcd; returns it once it listens on `port`, or fails the test if it exits."""
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **popen_args
    )
    deadline = time.monotonic() + 10 * FACTOR
    while not _listening(port):
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            out = proc.communicate()[0]
            pytest.fail(f"rdcd did not start (exit {proc.returncode})\n{out[-3000:]}")
        time.sleep(0.1)
    return proc


def _stop(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        out = proc.communicate(timeout=15 * FACTOR)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        out = proc.communicate()[0]
        pytest.fail(f"rdcd did not stop on SIGTERM\n{out[-3000:]}")
    assert proc.returncode == 0, f"rdcd exited with {proc.returncode}\n{out[-3000:]}"


def _env_for_nobody(d, log_dir):
    # The fake's call log goes where nobody can write.
    return dict(d.env, RDC_FAKE_LOG=os.path.join(log_dir, "fake_calls.log"))


@contextlib.contextmanager
def _writable_dir():
    path = tempfile.mkdtemp(prefix="rdcd-nobody-")
    os.chmod(path, 0o1777)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_rdcd_refuses_a_symbolic_link_as_its_lock_file(tools, tmp_path):
    _need_root()
    target = tmp_path / "target"
    with _set_aside(ROOT_LOCK):
        os.symlink(target, ROOT_LOCK)
        d = Rdcd(tools, tmp_path)
        code, out = _run([d.rdcd_path, "-u", "-p", str(_free_port())], d.env, "symbolic link")
    assert code == 1, f"rdcd exited with {code}\n{out[-3000:]}"
    assert f"{ROOT_LOCK} {REFUSED} root" in out, out[-3000:]
    assert not target.exists(), "rdcd created the file the link points to"


def test_rdcd_refuses_a_lock_file_of_another_user(tools, tmp_path):
    nobody, _ = _nobody()
    with _set_aside(ROOT_LOCK):
        with open(ROOT_LOCK, "x"):
            pass
        os.chown(ROOT_LOCK, nobody.pw_uid, nobody.pw_gid)
        os.chmod(ROOT_LOCK, 0o666)
        d = Rdcd(tools, tmp_path)
        code, out = _run([d.rdcd_path, "-u", "-p", str(_free_port())], d.env, "foreign lock file")
    assert code == 1, f"rdcd exited with {code}\n{out[-3000:]}"
    assert f"{ROOT_LOCK} {REFUSED} root" in out, out[-3000:]


def test_rdcd_refuses_a_lock_file_with_another_name(tools, tmp_path):
    # rdcd writes into its lock file; through a hard link it would write into another file.
    _need_root()
    other = ROOT_LOCK + ".other-name-by-test"
    if os.path.lexists(other):
        pytest.skip(f"{other} exists")
    with _set_aside(ROOT_LOCK):
        try:
            with open(other, "w") as f:
                f.write("keep\n")
            os.link(other, ROOT_LOCK)
            d = Rdcd(tools, tmp_path)
            code, out = _run([d.rdcd_path, "-u", "-p", str(_free_port())], d.env, "hard link")
            with open(other) as f:
                content = f.read()
        finally:
            if os.path.lexists(other):
                os.remove(other)
    assert code == 1, f"rdcd exited with {code}\n{out[-3000:]}"
    assert f"{ROOT_LOCK} {REFUSED} root" in out, out[-3000:]
    assert content == "keep\n", content


def test_rdcd_writes_only_its_pid_into_the_lock_file(tools, tmp_path):
    _need_root()
    with _set_aside(ROOT_LOCK):
        with open(ROOT_LOCK, "x") as f:
            f.write("9999999999")  # a longer pid of an earlier rdcd
        os.chmod(ROOT_LOCK, 0o644)
        d = Rdcd(tools, tmp_path).start()
        try:
            with open(ROOT_LOCK) as f:
                content = f.read()
        finally:
            code = d.stop()
    assert code == 0, d.output()
    assert content == str(d.proc.pid), content


def test_a_locked_file_in_tmp_does_not_stop_a_root_rdcd(tools, tmp_path):
    # Any local user can create /tmp/rdcd.lock and lock it.
    nobody, _ = _nobody()
    with _set_aside(USER_LOCK):
        fd = os.open(USER_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.fchown(fd, nobody.pw_uid, nobody.pw_gid)
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            d = Rdcd(tools, tmp_path).start()
            assert d.stop() == 0, d.output()
        finally:
            os.close(fd)


def test_a_second_rdcd_exits_with_1(tools, tmp_path):
    # On another port, so that only the lock can stop it.
    d = Rdcd(tools, tmp_path).start()
    try:
        code, out = _run([d.rdcd_path, "-u", "-p", str(_free_port())], d.env, "second rdcd")
    finally:
        assert d.stop() == 0, d.output()
    assert code == 1, f"rdcd exited with {code}\n{out[-3000:]}"
    assert ALREADY_RUNNING in out, out[-3000:]


@pytest.mark.parametrize("mode", [0o644, 0o600], ids=["readable", "unreadable"])
def test_an_unlocked_root_lock_file_does_not_stop_a_non_root_rdcd(tools, tmp_path, mode):
    _, as_nobody = _nobody()
    with _set_aside(ROOT_LOCK), _set_aside(USER_LOCK), _writable_dir() as log_dir:
        with open(ROOT_LOCK, "x"):
            pass
        os.chmod(ROOT_LOCK, mode)
        d = Rdcd(tools, tmp_path)
        port = _free_port()
        _stop(
            _start(
                as_nobody + [d.rdcd_path, "-u", "-p", str(port)], _env_for_nobody(d, log_dir), port
            )
        )


@pytest.mark.parametrize("mode", [0o644, 0o600], ids=["readable", "unreadable"])
def test_a_non_root_rdcd_next_to_a_root_rdcd(tools, tmp_path, mode):
    # A non-root rdcd can tell that the root rdcd runs only if it can read its lock file.
    _, as_nobody = _nobody()
    with _set_aside(ROOT_LOCK), _set_aside(USER_LOCK), _writable_dir() as log_dir:
        root = Rdcd(tools, tmp_path).start()  # creates ROOT_LOCK; _set_aside puts the old one back
        try:
            os.chmod(ROOT_LOCK, mode)
            port = _free_port()
            cmd = as_nobody + [root.rdcd_path, "-u", "-p", str(port)]
            env = _env_for_nobody(root, log_dir)
            if mode & stat.S_IROTH:
                code, out = _run(cmd, env, "non-root rdcd next to the root rdcd")
                assert code == 1, f"rdcd exited with {code}\n{out[-3000:]}"
                assert f"{ROOT_LOCK} {ALREADY_RUNNING}" in out, out[-3000:]
            else:
                _stop(_start(cmd, env, port))
        finally:
            assert root.stop() == 0, root.output()


@pytest.mark.parametrize("bad", [["--no-such-option"], ["-a"]], ids=["unknown", "no-argument"])
def test_an_unknown_option_stops_rdcd(tools, tmp_path, bad):
    # Both after a valid port, so that an rdcd that ignored them would start and keep running.
    d = Rdcd(tools, tmp_path)
    cmd = [d.rdcd_path, "-u", "-p", str(_free_port())] + bad
    code, out = _run(cmd, d.env, "bad option")
    assert code == 1, f"rdcd exited with {code}\n{out[-3000:]}"
    assert "Optional rdcd Arguments" in out, out[-3000:]


def test_rdcd_closes_inherited_descriptors(tools, tmp_path):
    # Started with stdin closed and one more descriptor open, as a service manager may.
    d = Rdcd(tools, tmp_path)
    port = _free_port()
    with open(tmp_path / "inherited", "w") as inherited:
        proc = _start(
            [d.rdcd_path, "-u", "-p", str(port)],
            d.env,
            port,
            stdin=subprocess.DEVNULL,
            pass_fds=(inherited.fileno(),),
            preexec_fn=lambda: os.close(0),
        )
    fds = {}
    try:
        for fd in os.listdir(f"/proc/{proc.pid}/fd"):
            try:
                fds[int(fd)] = os.readlink(f"/proc/{proc.pid}/fd/{fd}")
            except OSError:
                pass  # closed meanwhile (gRPC's connections come and go)
    finally:
        _stop(proc)
    assert fds.get(0) == "/dev/null", fds
    assert str(tmp_path / "inherited") not in fds.values(), fds


@pytest.mark.stress
def test_rdcd_stops_on_sigterm_while_it_logs(tools, tmp_path):
    # rdcd's signal handler runs on whichever thread the signal hits, also one that is writing a
    # debug log line; rdcd must still stop.
    d = Rdcd(tools, tmp_path)
    for cycle in range(40):
        port = _free_port()
        proc = _start([d.rdcd_path, "-u", "-d", "-p", str(port)], d.env, port)
        host = f"127.0.0.1:{port}"
        dmon = subprocess.Popen(
            [
                d.rdci_path,
                "dmon",
                "-u",
                "--host",
                host,
                "-i",
                "0",
                "-e",
                "100,201,300",
                "-d",
                "100",
            ],
            env=d.rdci_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3 + (cycle % 5) * 0.1)
        try:
            _stop(proc)
        finally:
            dmon.kill()
            dmon.wait()
