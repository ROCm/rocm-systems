"""Active TUI session authority helpers."""

from __future__ import annotations

import hmac
import os
import secrets
import shutil
import socket
import stat
import struct
import sysconfig
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import pwd
except ImportError:  # pragma: no cover - non-Unix import-safety guard
    pwd = None  # type: ignore[assignment]


INTERACTIVE_TUI_ENV = "PERFXPERT_TUI_INTERACTIVE"
TUI_SESSION_TOKEN_ENV = "PERFXPERT_TUI_SESSION_TOKEN"
TUI_SESSION_SOCKET_ENV = "PERFXPERT_TUI_SESSION_SOCKET"

# Legacy env from the first implementation; always scrub it so stale shells do
# not carry confusing authority-looking state into noninteractive commands.
TUI_SESSION_TOKEN_FILE_ENV = "PERFXPERT_TUI_SESSION_TOKEN_FILE"

_TOKEN_TTL_SECONDS = 12 * 60 * 60
_SERVER_TIMEOUT_SECONDS = 0.25
_CLIENT_TIMEOUT_SECONDS = 1.0
_AUTH_OK = b"ok"
_AUTH_NO = b"no"


@dataclass(frozen=True)
class _ServerRef:
    listener: socket.socket
    stop: threading.Event
    thread: threading.Thread
    session_dir: Path


_SERVER_REGISTRY: dict[str, _ServerRef] = {}


def clear_tui_session_env(env: dict[str, str]) -> None:
    """Remove inherited active-TUI authority from an environment mapping."""

    env.pop(INTERACTIVE_TUI_ENV, None)
    env.pop(TUI_SESSION_TOKEN_ENV, None)
    env.pop(TUI_SESSION_SOCKET_ENV, None)
    env.pop(TUI_SESSION_TOKEN_FILE_ENV, None)


def bind_tui_session_env(env: dict[str, str]) -> Path:
    """Bind a launcher-owned active-session socket into ``env``.

    ``perfxpert workflow import`` is intentionally an internal helper for a
    live ``perfxpert-code`` TUI. A plain shell can set environment variables,
    so the marker is backed by a private Unix socket whose server validates
    that the caller is a descendant of the launcher process that created it.
    The helper also validates that the socket peer is an ancestor process,
    preventing a shell-created sibling socket from satisfying the gate.
    """

    if not _platform_supports_active_tui_session():
        raise RuntimeError("active TUI workflow authorization requires Linux peer-credential sockets")

    token = secrets.token_urlsafe(32)
    session_dir = Path(tempfile.mkdtemp(prefix="perfxpert-tui-"))
    socket_path = session_dir / "auth.sock"
    listener: socket.socket | None = None
    try:
        os.chmod(session_dir, 0o700)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        listener.listen(16)

        stop = threading.Event()
        owner_pid = os.getpid()
        owner_uid = os.getuid()
        created_at = int(time.time())
        thread = threading.Thread(
            target=_auth_server_main,
            args=(listener, stop, owner_pid, owner_uid, token, created_at),
            name="perfxpert-tui-auth",
            daemon=True,
        )
        thread.start()
        _SERVER_REGISTRY[str(socket_path)] = _ServerRef(
            listener=listener,
            stop=stop,
            thread=thread,
            session_dir=session_dir,
        )
        listener = None
    except Exception:
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        shutil.rmtree(session_dir, ignore_errors=True)
        raise

    env[INTERACTIVE_TUI_ENV] = "1"
    env[TUI_SESSION_TOKEN_ENV] = token
    env[TUI_SESSION_SOCKET_ENV] = str(socket_path)
    env.pop(TUI_SESSION_TOKEN_FILE_ENV, None)
    return socket_path


def cleanup_tui_session_env(env: dict[str, str]) -> None:
    """Best-effort shutdown of the active-session socket server."""

    raw_path = env.get(TUI_SESSION_SOCKET_ENV)
    if raw_path:
        ref = _SERVER_REGISTRY.pop(raw_path, None)
        if ref is not None:
            ref.stop.set()
            try:
                ref.listener.close()
            except OSError:
                pass
            ref.thread.join(timeout=1.0)
            shutil.rmtree(ref.session_dir, ignore_errors=True)
        else:
            path = Path(raw_path)
            try:
                path.unlink()
            except OSError:
                pass
            try:
                path.parent.rmdir()
            except OSError:
                pass
    clear_tui_session_env(env)


def has_active_tui_session(env: dict[str, str] | None = None) -> bool:
    """Return True when ``env`` carries a valid active TUI socket token."""

    if not _platform_supports_active_tui_session():
        return False
    env = os.environ if env is None else env
    if env.get(INTERACTIVE_TUI_ENV) != "1":
        return False
    token = env.get(TUI_SESSION_TOKEN_ENV)
    raw_path = env.get(TUI_SESSION_SOCKET_ENV)
    if not token or not raw_path:
        return False
    path = Path(raw_path)
    try:
        st = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISSOCK(st.st_mode):
        return False
    if st.st_mode & 0o077:
        return False

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(_CLIENT_TIMEOUT_SECONDS)
            client.connect(str(path))
            credentials = _peer_credentials(client)
            if not _peer_is_current_ancestor(credentials):
                return False
            client.sendall(token.encode("utf-8"))
            response = client.recv(16)
    except OSError:
        return False
    return response == _AUTH_OK


def _auth_server_main(
    listener: socket.socket,
    stop: threading.Event,
    owner_pid: int,
    owner_uid: int,
    token: str,
    created_at: int,
) -> None:
    listener.settimeout(_SERVER_TIMEOUT_SECONDS)
    token_bytes = token.encode("utf-8")
    deadline = created_at + _TOKEN_TTL_SECONDS
    while not stop.is_set():
        now = int(time.time())
        if now > deadline or not _process_alive(owner_pid):
            break
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        with conn:
            conn.settimeout(_CLIENT_TIMEOUT_SECONDS)
            ok = _peer_is_launcher_descendant(conn, owner_pid, owner_uid)
            try:
                candidate = conn.recv(512).strip()
            except OSError:
                candidate = b""
            if ok and hmac.compare_digest(candidate, token_bytes):
                _send_auth_response(conn, _AUTH_OK)
            else:
                _send_auth_response(conn, _AUTH_NO)
    try:
        listener.close()
    except OSError:
        pass


def _send_auth_response(conn: socket.socket, response: bytes) -> None:
    try:
        conn.sendall(response)
    except OSError:
        pass


def _peer_is_launcher_descendant(conn: socket.socket, owner_pid: int, owner_uid: int) -> bool:
    credentials = _peer_credentials(conn)
    if credentials is None:
        return False
    client_pid, client_uid, _client_gid = credentials
    if client_uid != owner_uid:
        return False
    if client_pid <= 1 or client_pid == owner_pid:
        return False
    return _pid_has_ancestor(client_pid, owner_pid)


def _peer_is_current_ancestor(credentials: tuple[int, int, int] | None) -> bool:
    if credentials is None:
        return False
    peer_pid, peer_uid, _peer_gid = credentials
    if peer_uid != os.getuid():
        return False
    if peer_pid <= 1 or peer_pid == os.getpid():
        return False
    return _pid_in_current_ancestry(peer_pid) and _process_is_perfxpert_code_launcher(peer_pid)


def _process_is_perfxpert_code_launcher(pid: int) -> bool:
    script = _launcher_script_arg(_read_proc_args(pid))
    if script is None:
        return False
    try:
        resolved = Path(script).resolve(strict=True)
    except OSError:
        return False
    if not any(_same_file(resolved, trusted) for trusted in _trusted_launcher_paths()):
        return False
    return _launcher_script_has_expected_entrypoint(resolved)


def _launcher_script_arg(args: list[str]) -> str | None:
    if not args:
        return None
    first = Path(args[0]).name.lower()
    if first.startswith("python") and len(args) > 1:
        return args[1]
    return args[0]


def _trusted_launcher_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        candidates.append(Path(scripts_dir) / "perfxpert-code")
    home = _account_home()
    if home is not None:
        candidates.append(home / ".local" / "bin" / "perfxpert-code")
    candidates.extend(
        [
            Path("/usr/local/bin/perfxpert-code"),
            Path("/usr/bin/perfxpert-code"),
        ]
    )

    resolved: list[Path] = []
    for candidate in candidates:
        try:
            path = candidate.resolve(strict=True)
        except OSError:
            continue
        if path not in resolved:
            resolved.append(path)
    return tuple(resolved)


def _account_home() -> Path | None:
    if pwd is None:
        try:
            return Path.home()
        except RuntimeError:
            return None
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        return None


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


def _launcher_script_has_expected_entrypoint(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "perfxpert.cli.opencode_launcher" in text and "main" in text


def _peer_credentials(conn: socket.socket) -> tuple[int, int, int] | None:
    if not hasattr(socket, "SO_PEERCRED"):
        return None
    try:
        raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    except OSError:
        return None
    try:
        return struct.unpack("3i", raw)
    except struct.error:
        return None


def _platform_supports_active_tui_session() -> bool:
    return hasattr(socket, "AF_UNIX") and hasattr(socket, "SO_PEERCRED") and Path("/proc").is_dir()


def _pid_in_current_ancestry(target_pid: int) -> bool:
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid == target_pid:
            return True
        seen.add(pid)
        next_pid = _parent_pid(pid)
        if next_pid is None or next_pid == pid:
            break
        pid = next_pid
    return False


def _pid_has_ancestor(pid: int, ancestor_pid: int) -> bool:
    seen: set[int] = set()
    current = _parent_pid(pid)
    while current and current > 1 and current not in seen:
        if current == ancestor_pid:
            return True
        seen.add(current)
        current = _parent_pid(current)
    return False


def _process_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _parent_pid(pid: int) -> int | None:
    stat_text = _read_proc_text(pid, "stat")
    if not stat_text:
        return None
    try:
        after_comm = stat_text.rsplit(")", 1)[1].strip()
        fields = after_comm.split()
        return int(fields[1])
    except (IndexError, ValueError):
        return None


def _read_proc_text(pid: int, name: str) -> str:
    try:
        return Path("/proc").joinpath(str(pid), name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_proc_args(pid: int) -> list[str]:
    raw = _read_proc_text(pid, "cmdline")
    return [arg for arg in raw.split("\x00") if arg]
