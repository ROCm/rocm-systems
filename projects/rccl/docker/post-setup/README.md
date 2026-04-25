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

`--post-setup` is repeatable. Multiple dirs run in CLI order; later
`env.sh` exports override earlier ones, and each `setup.sh` is cached
independently by content hash.

```bash
# Single user-provided config
python3 -m mnctl --launch-all --post-setup /path/to/my-setup

# Stack several configs (dirs run in this order):
python3 -m mnctl --launch-all \
    --post-setup ./post-setup/rccl-tests \
    --post-setup ./my-experiment

# AINIC with driver source mounted (the ainic dir is auto-prepended;
# see "NIC-type integration" below)
python3 -m mnctl --launch-all --nic-type ainic \
    --volume /path/to/drivers-linux:/opt/nic-drivers:ro
```

In the container, mnctl bind-mounts each host dir at
`/opt/post-setup.<N>` (read-only) and sets `POST_SETUP_DIRS` to the
colon-separated list of in-container paths. The entrypoint iterates that
list in order.

## NIC-type integration

`--nic-type X` (default `mellanox`) does two things:

1. Adjusts host-side bind-mounts (e.g. RDMA libs for `mellanox`).
2. **Auto-prepends** the built-in dir `post-setup/<X>/` to the
   post-setup list, *if it exists*. So:

   ```bash
   --nic-type ainic --post-setup ./my-app
   ```

   is equivalent to:

   ```bash
   --post-setup ./post-setup/ainic --post-setup ./my-app
   ```

   The built-in dir always runs first so user dirs can override its
   environment via their own `env.sh`.

3. If `post-setup/<X>/` does not exist (custom NIC names), nothing is
   auto-prepended -- not an error.

To disable the auto-prepend and ship your own NIC config:

```bash
python3 -m mnctl --launch-all --nic-type ainic \
    --no-builtin-nic-setup \
    --post-setup /my/custom/ainic-config
```

This unifies the model across base images: NIC driver installation is
just another post-setup hook, not something baked into the Dockerfile.

## Writing a Post-Setup Config

### env.sh

Concatenated (in CLI order) into `/etc/profile.d/post-setup-env.sh` so
all shells pick up the variables. Use only `export` statements:

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
mounted read-only). Results are cached in `/opt/builds/` keyed by the
SHA256 of `setup.sh` -- if the script content hasn't changed,
subsequent container starts skip re-execution. Pass `--rebuild` (or
`--replace`) to clear the marker and force a re-run.

```bash
#!/bin/bash
apt-get update && apt-get install -y some-package
cp -r /opt/my-drivers /tmp/build && cd /tmp/build && make install
```

## Security

- Each post-setup directory is mounted **read-only** in the container
- `setup.sh` is validated for a bash shebang before execution
- SHA256 hash of each script is logged for auditability
- Nothing user-provided runs unless `--post-setup` (or `--nic-type`
  matching a built-in dir) is in effect
- Scripts execute inside the container (already privileged); no
  host-side code execution

## Built-in Examples

| Directory | Use Case | Notes |
|---|---|---|
| `ainic/` | AMD AINIC NIC driver | Auto-loaded with `--nic-type ainic`. Needs driver source mounted via `--volume /path/to/drivers-linux:/opt/nic-drivers:ro` |
| `mellanox/` | Mellanox ConnectX tuning | Auto-loaded with `--nic-type mellanox` (the default). Host RDMA libs are auto bind-mounted; env template only |
