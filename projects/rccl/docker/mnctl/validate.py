"""Configuration validation.

Collects all errors and warnings, then reports them in a single pass.
This replaces the monolithic validate_options() bash function.
"""

import os
import sys
from typing import List

from .checks import (
    SSH_KEY_MISSING, POST_SETUP_MISSING, POST_SETUP_EMPTY,
    hostfile_status, ssh_key_status, post_setup_statuses,
    resolve_dockerfile, list_available_dockerfiles,
)
from .config import Action, Config
from .utils import warn, error


def validate(cfg):
    # type: (Config) -> None
    """Validate configuration; exits with code 1 on any error."""
    errors = []    # type: List[str]
    warnings = []  # type: List[str]
    action = cfg.action

    # --- Hostfile required for multi-node actions ---
    if action in (Action.LAUNCH_ALL, Action.STOP_ALL, Action.VERIFY):
        exists, _hosts = hostfile_status(cfg)
        if not exists:
            errors.append(
                "Hostfile not found: {hf}\n"
                "  Required by: --{act}\n"
                "  Create it:   echo 'hostname slots=8' > {hf}\n"
                "  Or specify:  --hostfile /path/to/hostfile\n"
                "  Or run inside a SLURM allocation "
                "(auto-detected from SLURM_NODELIST)".format(
                    hf=cfg.hostfile, act=action.value
                )
            )

    # --- --rebuild without a relevant action ---
    if cfg.force_rebuild and action in (Action.VERIFY, Action.STOP_ALL):
        warnings.append(
            "--rebuild has no effect with --{}; "
            "it only applies to build/run/launch actions".format(action.value)
        )

    # --- SSH key existence ---
    status, priv, pub = ssh_key_status(cfg)
    if status == SSH_KEY_MISSING:
        # SSH_KEY_MISSING means at least one of the pair is absent;
        # report each one individually so the user knows what to fix.
        if priv and not os.path.isfile(priv):
            errors.append("SSH private key not found: {}".format(priv))
        if pub and not os.path.isfile(pub):
            errors.append("SSH public key not found: {}".format(pub))

    # --- Dockerfile existence for build actions ---
    if action in (Action.BUILD, Action.RUN, Action.LAUNCH_ALL, Action.SETUP_DEPS):
        df_path = resolve_dockerfile(cfg)
        if not os.path.isfile(df_path):
            available = list_available_dockerfiles(cfg)
            errors.append(
                "Dockerfile not found: {df}\n"
                "  Available in {sd}/:\n"
                "    {avail}".format(
                    df=df_path,
                    sd=cfg.script_dir,
                    avail="\n    ".join(available) if available else "(none)",
                )
            )

    # --- Post-setup directories (each user-supplied dir validated) ---
    for path, ps_status, _files in post_setup_statuses(cfg):
        if ps_status == POST_SETUP_MISSING:
            errors.append("Post-setup directory not found: {}".format(path))
        elif ps_status == POST_SETUP_EMPTY:
            errors.append(
                "Post-setup dir must contain setup.sh and/or env.sh: {}".format(
                    path
                )
            )

    # --- Report ---
    for w in warnings:
        warn(w)

    if errors:
        print("", file=sys.stderr)
        for e in errors:
            error(e)
        print("", file=sys.stderr)
        print(
            "Run 'python3 -m mnctl --help' for usage information.",
            file=sys.stderr,
        )
        sys.exit(1)
