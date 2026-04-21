"""Configuration validation.

Collects all errors and warnings, then reports them in a single pass.
This replaces the monolithic validate_options() bash function.
"""

import os
import sys
from typing import List

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
        if not os.path.isfile(cfg.hostfile):
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
    if cfg.ssh.key:
        priv = cfg.ssh.priv_key
        pub = cfg.ssh.pub_key
        if priv and not os.path.isfile(priv):
            errors.append("SSH private key not found: {}".format(priv))
        if pub and not os.path.isfile(pub):
            errors.append("SSH public key not found: {}".format(pub))

    # --- Dockerfile existence for build actions ---
    if action in (Action.BUILD, Action.RUN, Action.LAUNCH_ALL, Action.SETUP_DEPS):
        if os.path.isabs(cfg.dockerfile):
            df_path = cfg.dockerfile
        else:
            df_path = os.path.join(cfg.script_dir, cfg.dockerfile)
        if not os.path.isfile(df_path):
            available = [
                f for f in os.listdir(cfg.script_dir)
                if f.startswith("Dockerfile.")
            ]
            errors.append(
                "Dockerfile not found: {df}\n"
                "  Available in {sd}/:\n"
                "    {avail}".format(
                    df=df_path,
                    sd=cfg.script_dir,
                    avail="\n    ".join(sorted(available)) if available
                    else "(none)",
                )
            )

    # --- Post-setup directory ---
    if cfg.post_setup_dir:
        if not os.path.isdir(cfg.post_setup_dir):
            errors.append(
                "Post-setup directory not found: {}".format(cfg.post_setup_dir)
            )
        else:
            has_setup = os.path.isfile(
                os.path.join(cfg.post_setup_dir, "setup.sh")
            )
            has_env = os.path.isfile(
                os.path.join(cfg.post_setup_dir, "env.sh")
            )
            if not has_setup and not has_env:
                errors.append(
                    "Post-setup dir must contain setup.sh and/or env.sh: {}".format(
                        cfg.post_setup_dir
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
