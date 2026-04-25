"""Multi-node orchestration: host setup, launch-all, stop-all.

The host-setup phase creates shared directories and installs SSH keys.
``launch_all`` / ``stop_all`` fan out to every node in the hostfile by
spawning all SSH commands as concurrent subprocesses (Popen).  Reader
threads stream output line-by-line from each node in real time.

Package layout:

  * :mod:`.host_setup` — :func:`setup_host`
  * :mod:`.distribute` — :func:`distribute_files`,
    :func:`push_pubkey_to_remotes`, :func:`is_path_shared`
  * :mod:`.forward`    — :func:`build_forward_args`
  * :mod:`.launch_all` — :func:`launch_all`
  * :mod:`.stop_all`   — :func:`stop_all`
  * :mod:`.failures`   — :func:`report_launch_failures`

Only :func:`setup_host`, :func:`launch_all`, and :func:`stop_all` are part
of the public API consumed by :mod:`mnctl.__main__`.  The other modules
are split out for readability and testability and may be imported by
sibling code (e.g. preflight) when needed.
"""

from .host_setup import setup_host
from .launch_all import launch_all
from .stop_all import stop_all


__all__ = ["setup_host", "launch_all", "stop_all"]
