"""Pure inspection predicates used by both ``validate.py`` and ``preflight.py``.

Each function reads the filesystem (and ``cfg``) and returns a structured
result.  No printing, no ``sys.exit``: the callers decide how to report.
This keeps the "is X true?" logic in one place while letting validate (collect
errors) and preflight (live OK/FAIL output) format the answers differently.
"""

import os
from typing import List, Optional, Tuple

from .config import Config
from .utils import parse_hostfile


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------
def resolve_dockerfile(cfg):
    # type: (Config) -> str
    """Return the absolute path of the Dockerfile that would be used."""
    if os.path.isabs(cfg.dockerfile):
        return cfg.dockerfile
    return os.path.join(cfg.script_dir, cfg.dockerfile)


def list_available_dockerfiles(cfg):
    # type: (Config) -> List[str]
    """Sorted list of ``Dockerfile.*`` siblings in ``cfg.script_dir``."""
    try:
        return sorted(
            f for f in os.listdir(cfg.script_dir)
            if f.startswith("Dockerfile.")
        )
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Hostfile
# ---------------------------------------------------------------------------
def hostfile_status(cfg):
    # type: (Config) -> Tuple[bool, List[str]]
    """Return ``(exists, hosts)``.

    ``hosts`` is the parsed hostlist (possibly empty if the file exists but
    contains no entries).  When the file is missing, ``hosts`` is ``[]`` and
    the first element is ``False``.
    """
    if not os.path.isfile(cfg.hostfile):
        return False, []
    return True, parse_hostfile(cfg.hostfile)


# ---------------------------------------------------------------------------
# SSH keys
# ---------------------------------------------------------------------------
# Status codes returned by :func:`ssh_key_status`.
SSH_KEY_OK = "ok"               # explicit key, both files present
SSH_KEY_MISSING = "missing"     # explicit key, at least one file missing
SSH_KEY_KEYGEN = "keygen"       # will auto-generate (--ssh)
SSH_KEY_NONE = "none"           # no SSH config requested


def ssh_key_status(cfg):
    # type: (Config) -> Tuple[str, Optional[str], Optional[str]]
    """Return ``(status, priv_path, pub_path)``.

    ``status`` is one of the ``SSH_KEY_*`` module constants.  Path values are
    ``None`` for the ``KEYGEN`` and ``NONE`` cases.
    """
    if cfg.ssh.key:
        priv = cfg.ssh.priv_key
        pub = cfg.ssh.pub_key
        if priv and os.path.isfile(priv) and pub and os.path.isfile(pub):
            return SSH_KEY_OK, priv, pub
        return SSH_KEY_MISSING, priv, pub
    if cfg.ssh.keygen:
        return SSH_KEY_KEYGEN, None, None
    return SSH_KEY_NONE, None, None


# ---------------------------------------------------------------------------
# Post-setup directory
# ---------------------------------------------------------------------------
POST_SETUP_MISSING = "missing"  # cfg.post_setup_dir is not a directory
POST_SETUP_EMPTY = "empty"      # exists but no setup.sh / env.sh
POST_SETUP_OK = "ok"            # exists and has at least one of the scripts


def post_setup_status(cfg):
    # type: (Config) -> Tuple[str, List[str]]
    """Return ``(status, files)``.

    ``files`` is the subset of ``["setup.sh", "env.sh"]`` that exists in
    ``cfg.post_setup_dir``.  Only meaningful when ``cfg.post_setup_dir`` is
    truthy; callers are expected to gate on that themselves.
    """
    d = cfg.post_setup_dir
    if not os.path.isdir(d):
        return POST_SETUP_MISSING, []
    files = []
    if os.path.isfile(os.path.join(d, "setup.sh")):
        files.append("setup.sh")
    if os.path.isfile(os.path.join(d, "env.sh")):
        files.append("env.sh")
    if not files:
        return POST_SETUP_EMPTY, []
    return POST_SETUP_OK, files
