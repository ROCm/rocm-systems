"""Wait for the container's entrypoint to print '=== Ready ==='.

Implementation notes:
  * ``docker logs --follow`` is read by a background thread so the main
    thread can enforce a real wall-clock deadline even when the
    container goes silent (a blocking ``readline()`` would otherwise
    hang indefinitely on quiet containers).
  * The container state is also probed periodically; if the container
    has exited before printing Ready, we surface that instead of
    waiting for the full timeout.
"""

import subprocess
import threading
import time
from typing import Optional

from ..utils import log, run_capture


def inspect_container_state(cfg):
    # type: (object) -> Optional[str]
    """Return the container's current state (e.g. ``running``) or None."""
    r = run_capture([
        "docker", "inspect", "-f", "{{.State.Status}}",
        cfg.container_name,
    ])
    if not r.ok:
        return None
    return r.stdout_text.strip() or None


def wait_for_entrypoint(cfg, timeout=600):
    # type: (object, int) -> None
    """Follow container logs until '=== Ready ===' prints or we time out."""
    log("")
    log("  Waiting for entrypoint to finish (timeout {}s) ...".format(
        timeout,
    ))

    proc = subprocess.Popen(
        ["docker", "logs", "--follow", cfg.container_name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    ready_evt = threading.Event()
    done_evt = threading.Event()

    def _reader():
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\n\r")
                log("    {}".format(line))
                if "=== Ready ===" in line:
                    ready_evt.set()
                    return
        finally:
            done_evt.set()

    t = threading.Thread(target=_reader)
    t.daemon = True
    t.start()

    start = time.time()
    last_state_check = 0.0
    timed_out = False
    container_exited = False
    try:
        while True:
            if ready_evt.wait(timeout=1.0):
                break
            if done_evt.is_set():
                # docker logs ended (container removed or daemon hung up)
                break
            elapsed = time.time() - start
            if elapsed > timeout:
                timed_out = True
                log("  WARNING: entrypoint did not become ready "
                    "within {}s".format(timeout))
                break
            # Probe container state every 10s so a crashed/exited
            # container is detected without waiting for the full timeout.
            if elapsed - last_state_check > 10.0:
                last_state_check = elapsed
                state = inspect_container_state(cfg)
                if state and state not in ("running", "created", "restarting"):
                    container_exited = True
                    log("  WARNING: container is '{}' before printing "
                        "'=== Ready ===' (entrypoint likely failed)"
                        .format(state))
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        # Reader thread should exit quickly once the pipe is closed.
        t.join(timeout=2)

    if ready_evt.is_set():
        elapsed = int(time.time() - start)
        log("  Entrypoint ready ({}s)".format(elapsed))
    elif container_exited:
        log("  Inspect with: docker logs {}".format(cfg.container_name))
    elif timed_out:
        log("  Inspect with: docker logs {}".format(cfg.container_name))
    log("")
