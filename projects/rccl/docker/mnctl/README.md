# `mnctl` — Get started

A Python tool that builds, deploys, and tears down ROCm Docker
containers across one or many hosts in a single command.

> **TL;DR**
> ```bash
> python3 -m mnctl --run                    # one node
> python3 -m mnctl --launch-all --ssh       # all nodes in your hostfile
> python3 -m mnctl --stop-all               # tear down everywhere
> ```

---

## Prerequisites

* **Python 3.6+** (standard library only — no `pip install` needed)
* **Docker** with the user in the `docker` group
* **AMD GPU + ROCm** kernel modules (`/dev/kfd`, `/dev/dri/render*`)
* For multi-node:
  * SSH access to all nodes (port 22 by default)
  * A hostfile listing every node (auto-detected from SLURM if you're
    in a SLURM allocation)

---

## Single-node quick start

```bash
cd projects/rccl/docker

# Build the image and launch a container
python3 -m mnctl --run

# Open a shell inside it
docker exec -it rccl-mn bash
```

That's it. The first run builds the image (a few minutes); subsequent
runs are seconds.

---

## Multi-node quick start

```bash
cd projects/rccl/docker

# 1. Tell mnctl which nodes to use (one-time, OR skip if SLURM)
cat > ~/.mnctl_hostfile <<'EOF'
node-a slots=8
node-b slots=8
EOF

# 2. Build + launch on every node, auto-generating shared SSH keys
python3 -m mnctl --launch-all --ssh

# 3. Verify SSH between containers (optional but recommended)
python3 -m mnctl --verify

# 4. Run an MPI workload
docker exec -it rccl-mn bash
mpirun -np 16 \
  --hostfile ~/.mnctl_hostfile --map-by slot \
  --mca plm_rsh_agent "ssh -p 2224 -o StrictHostKeyChecking=no -q" \
  --allow-run-as-root \
  /opt/rccl-tests/build/all_reduce_perf -b 1 -e 16G -f 2 -g 1
```

If you'd rather use your own SSH key pair instead of generating one:
```bash
python3 -m mnctl --launch-all --ssh ~/.ssh/id_rsa
```

---

## Common commands

| Goal | Command |
|---|---|
| Build the image only | `python3 -m mnctl` |
| Build + launch on this node | `python3 -m mnctl --run` |
| Build + launch on all nodes | `python3 -m mnctl --launch-all --ssh` |
| Force a clean rebuild | `python3 -m mnctl --launch-all --rebuild` |
| Replace running containers (no rebuild) | `python3 -m mnctl --launch-all --replace` |
| Verify SSH across containers | `python3 -m mnctl --verify` |
| Stop and remove everywhere | `python3 -m mnctl --stop-all` |
| Pre-flight check without launching | `python3 -m mnctl --launch-all --dry-run` |
| Verbose debug output | `python3 -m mnctl --launch-all --verbose` |
| Use a different ROCm image | `python3 -m mnctl --launch-all rocm/dev-ubuntu-24.04:latest` |
| Use a different Dockerfile | `python3 -m mnctl --launch-all --dockerfile Dockerfile.Multinode.ALinux3` |
| Mount extra host paths | `python3 -m mnctl --run --volume /data:/data` |
| Add custom container setup | `python3 -m mnctl --run --post-setup ./my-setup` |

Run `python3 -m mnctl --help` for the full option list.

---

## Where things go (defaults, all under `~/`)

| Path | Purpose |
|---|---|
| `~/.mnctl_hostfile` | List of nodes (`hostname slots=N`) |
| `~/.docker-shared/` | Shared UCX + OpenMPI install (mounted at `/opt/shared`) |
| `~/.docker-builds/` | Post-setup completion markers (mounted at `/opt/builds`) |
| `~/.docker-ssh-keys/` | Container-side SSH key pair (mounted read-only) |

Override any of these with the matching `MNCTL_*` env var or CLI flag.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Container launch failed (exit 125)` | Run with `--verbose` and check the `docker run` line for missing devices / permission issues |
| `[FAIL] node-X` during `--launch-all` | Check `ssh -p 22 user@node-X hostname` works manually; if not, fix host SSH first |
| `mnctl --verify` says container can't SSH to itself | Re-run with `--ssh` (regenerates keys) or `--rebuild` |
| Hangs at "Waiting for entrypoint to finish…" | Re-run with `--verbose` and inspect `docker logs rccl-mn`; mnctl times out after 600s |
| Want to wipe everything and start over | `python3 -m mnctl --stop-all && rm -rf ~/.docker-{shared,builds,ssh-keys}` |
| Anything else | Re-run with `--verbose`; the output prints exact `docker`/`ssh` commands you can replay manually |

---

## More

* `python3 -m mnctl --help` — every flag and env var
* [`../post-setup/README.md`](../post-setup/README.md) — custom
  container-side setup hooks (NIC drivers, framework installs, etc.)
