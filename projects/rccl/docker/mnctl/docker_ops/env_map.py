"""Host-to-container environment mapping and post-setup directory resolution.

Pure data: no subprocess calls, no I/O beyond ``os.path.isdir`` for the
post-setup discovery.  Both functions are referenced by
:mod:`mnctl.docker_ops.launch` (via ``-e`` and ``-v`` argument assembly)
and indirectly by :mod:`mnctl.orchestrate.distribute` (which iterates the
raw ``cfg.post_setup_dirs`` for rsync).  The naming convention and full
host->container mapping reference are documented in
:func:`container_env_pairs`.
"""

import os
from typing import List


def container_env_pairs(cfg, render_gid):
    # type: (object, str) -> List[tuple]
    """Build the (NAME, value) pairs injected as ``-e`` flags on ``docker run``.

    This is the **single source of truth** for how mnctl's host-side
    settings reach the container's entrypoint and post-setup scripts.
    The naming convention is intentional:

      * On the **host**, all mnctl-controlled settings are namespaced
        ``MNCTL_*`` (e.g. ``MNCTL_GPUS``, ``MNCTL_NIC_TYPE``) so they
        cannot collide with the user's existing environment.
      * Inside the **container**, the prefix is dropped because these
        variables are consumed by user-maintained shell scripts
        (entrypoint, post-setup) where short, conventional names are
        more ergonomic. The container side is "private" to mnctl, so
        collision is not a concern.

    Mapping reference (host -> container)::

        MNCTL_GPUS         -> GPUS
        MNCTL_VERBOSE      -> VERBOSE             (only set when truthy)
        MNCTL_NIC_TYPE     -> NIC_TYPE
        MNCTL_GPU_TARGETS  -> GPU_TARGETS         (only set when non-empty)
        (derived)          -> HOST_UID, HOST_GID, RENDER_GID
        (derived)          -> FORCE_POST_SETUP    (when --rebuild/--replace)
        (derived)          -> POST_SETUP_DIRS     (colon-separated list of
                                                   in-container post-setup
                                                   mountpoints; see
                                                   :func:`resolve_post_setup_dirs`)

    To add a new pair: append it here AND document it in the epilog of
    ``__main__.py`` and (for end-user-visible settings) in
    ``post-setup/`` script docs.
    """
    pairs = [
        ("GPUS", cfg.gpus),
        ("HOST_UID", str(os.getuid())),
        ("HOST_GID", str(os.getgid())),
        ("RENDER_GID", render_gid),
        ("NIC_TYPE", cfg.nic_type),
    ]
    if cfg.verbose:
        pairs.append(("VERBOSE", "1"))
    if cfg.gpu_targets:
        pairs.append(("GPU_TARGETS", cfg.gpu_targets))
    if cfg.force_rebuild or cfg.force_replace:
        pairs.append(("FORCE_POST_SETUP", "1"))
    resolved = resolve_post_setup_dirs(cfg)
    if resolved:
        in_container = ":".join(
            "/opt/post-setup.{}".format(i) for i in range(len(resolved))
        )
        pairs.append(("POST_SETUP_DIRS", in_container))
    return pairs


def resolve_post_setup_dirs(cfg):
    # type: (object) -> List[str]
    """Resolve the final ordered list of host-side post-setup directories.

    Order: NIC-type built-in dir (if applicable) FIRST, then user dirs in
    CLI declaration order.  Later dirs run after earlier ones, so user
    dirs can override NIC defaults via env.sh exports or follow-up
    setup.sh actions.

    The NIC-type built-in is auto-prepended when:
      * ``cfg.no_builtin_nic_setup`` is False (the default), AND
      * ``<script_dir>/post-setup/<nic_type>/`` exists on disk, AND
      * that path is not already explicitly listed by the user.

    A non-existent NIC dir is silently skipped (not an error): the
    user may have a custom ``--nic-type`` that has no built-in recipe.

    The result is memoized on ``cfg._resolved_post_setup_dirs`` so the
    expensive path checks (and stable ordering) are computed once per
    invocation.  Call ``cfg.invalidate_resolved_post_setup_dirs()`` if
    any input attribute changes after the first call.
    """
    cached = getattr(cfg, "_resolved_post_setup_dirs", None)
    if cached is not None:
        return cached

    user_dirs = list(cfg.post_setup_dirs or [])
    if cfg.no_builtin_nic_setup or not cfg.nic_type:
        result = user_dirs
    else:
        builtin = os.path.join(cfg.script_dir, "post-setup", cfg.nic_type)
        if not os.path.isdir(builtin):
            result = user_dirs
        elif any(
            os.path.abspath(d) == os.path.abspath(builtin) for d in user_dirs
        ):
            # De-dup: user explicitly listed the same dir; do not double-mount.
            result = user_dirs
        else:
            result = [builtin] + user_dirs

    cfg._resolved_post_setup_dirs = result
    return result
