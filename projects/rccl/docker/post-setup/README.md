# Post-Setup Hooks

Run custom setup scripts inside containers after the core multi-node
infrastructure (SSH, user, shared deps) is ready. Use this for anything
that needs to happen once per container launch: NIC driver installation,
Python package installs, custom library builds, benchmark tooling, etc.

## Convention

Each post-setup config is a directory containing one or both of:

| File | Required | Purpose |
|---|---|---|
| `env.sh` | Optional | Environment variables (sourced in every shell) |
| `setup.sh` | Optional | Installation script (runs once at container startup) |

At least one of `env.sh` or `setup.sh` must be present.

## Usage

```bash
# Use a built-in post-setup config
./setup_multinode.sh --launch-all --post-setup ./post-setup/ainic

# Use a custom config directory
./setup_multinode.sh --launch-all --post-setup /path/to/my-setup

# AINIC with driver source mounted
./setup_multinode.sh --launch-all \
    --post-setup ./post-setup/ainic \
    --volume /path/to/drivers-linux:/opt/nic-drivers:ro
```

## Writing a Post-Setup Config

### env.sh

Sourced into `/etc/profile.d/post-setup-env.sh` so all shells pick up the
variables. Use only `export` statements:

```bash
#!/bin/bash
export NCCL_IB_GID_INDEX=1
export MY_CUSTOM_VAR=value
```

### setup.sh

Runs once at container startup (as root). Must:
- Start with `#!/bin/bash` (or `#!/usr/bin/env bash`) shebang
- Be idempotent (safe to re-run)
- Exit 0 on success

The script runs from a writable copy in `/tmp` (the post-setup dir is
mounted read-only). Results are cached in `/opt/builds/` — if the script
content hasn't changed, subsequent container starts skip re-execution.

```bash
#!/bin/bash
apt-get update && apt-get install -y some-package
cp -r /opt/my-drivers /tmp/build && cd /tmp/build && make install
```

## Security

- Post-setup directory is mounted **read-only** in the container
- `setup.sh` is validated for a bash shebang before execution
- SHA256 hash of each script is logged for auditability
- The `--post-setup` flag must be explicitly passed (nothing runs by default)
- Scripts execute inside the container (already privileged); no host-side code execution

## Built-in Examples

| Directory | Use Case | Notes |
|---|---|---|
| `ainic/` | AMD AINIC NIC driver | Requires driver source mounted at `/opt/nic-drivers` |
| `mellanox/` | Mellanox ConnectX tuning | Host RDMA libs are auto bind-mounted; env template only |
