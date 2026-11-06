# nccl_m2n on AMD Instinct (gfx90a, gfx942, gfx950): build and checks

Scripts for MI350X/MI355X nodes with Broadcom Thor2 RoCE (bnxt_re), GIN proxy transport
(`NCCL_GIN_TYPE=2`), inside a Docker container. They build RCCL and `nccl_m2n` from this tree, run the
basic API tests on 1 and 2 nodes, `reshard_bench --validate` for every copy mode, a model bench, and the
`nccl.m2n` Python checks (`projects/rccl/bindings/nccl_m2n`, on the RCCL nccl4py port in `projects/rccl/bindings/nccl4py`).

Cluster-specific defaults are for Ruby and are only examples: the network interface `fenic0`
(`NCCL_SOCKET_IFNAME`), the container sshd port `2222` (`SSH_PORT`), the key `~/.ssh/id_m2n` (`SSH_KEY`) and
the work directory `~/m2n-ruby` (`WORK`). Override them through these variables on other clusters.

## 0. Once

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_m2n         # mpirun key for the containers ($HOME is shared)
mkdir -p ~/m2n-ruby && cd ~/m2n-ruby
git clone --filter=blob:none --sparse --branch <branch> https://github.com/ROCm/rocm-systems.git rocm-systems
git -C rocm-systems sparse-checkout set projects/rccl
git clone --depth 1 -b v1.14.0 https://github.com/google/googletest.git googletest
export SD=~/m2n-ruby/rocm-systems/projects/rccl/contrib/nccl_m2n/scripts
```

## 1. Nodes, image, container

```bash
salloc -N 2 -p <partition> --exclusive -t 4:00:00      # N1 = first node, N2 = second
scontrol show hostnames $SLURM_JOB_NODELIST | sed 's/$/.<domain> slots=8/' > ~/m2n-ruby/hostfile
export NAME=m2n-$USER
for n in $N1 $N2; do
  ssh $n "sudo systemctl start docker; sudo docker image inspect m2n-ruby:primus-rdma61 >/dev/null 2>&1 ||
          (cd $SD && sudo docker build --network host -t m2n-ruby:primus-rdma61 .)"
  ssh $n "NAME=$NAME bash $SD/start-m2n-container.sh"   # must print nofile soft=1048576
done
ssh -t $N1 "sudo docker exec -it -u $(id -u) -e HOME=$HOME -e SD=$SD $NAME bash -l"   # steps below run here
```

The image is Primus (TheRock HIP, OpenMPI in `/opt/openmpi`) plus rdma-core v61 for the bnxt_re
driver, sshd on port 2222 for `mpirun`, and the Python setup for section 4: no NVIDIA
`cuda-python`/`cuda-core`/`cuda-bindings`/`cuda-pathfinder` (a real `cuda.core` shadows the port's HIP
shim), `hip-python`, `mpi4py`, `setuptools < 82` (torch), Cython, and `/opt/hipdev/libamdhip64.so`.

## 2. Build

```bash
bash $SD/build.sh            # rccl -> ~/m2n-ruby/rccl-install, nccl_m2n + tests + benches -> ~/m2n-ruby/mk,
                             # nccl4py + nccl.m2n wheels -> ~/m2n-ruby/whl;  or: build.sh rccl|m2n|bench|py
```

`M2N_GPU_ARCHS` (default `gfx90a;gfx942;gfx950`) sets the archs of RCCL and `libnccl_m2n.so`. The staging kernels
need wave64, so gfx1xxx targets fail to compile. `Makefile.rocm` writes the base archs (`gfx950:xnack-` ->
`gfx950`) to `gpu_archs.txt` next to the library and rebuilds the device code when the list changes;
`nccl.m2n` raises at load if a visible GPU has another arch. Only gfx950 is validated on hardware.

The `py` step stages `libnccl_m2n.so` (RUNPATH `$ORIGIN`) and `gpu_archs.txt` into the package and builds
the wheel with `NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS=1`. It then runs `auditwheel repair` with the ROCm runtime
excluded (`librccl.so.1`, `libamdhip64.so.7`, ...; the same command as `repair-wheel-command` in
`bindings/nccl_m2n/pyproject.toml`) into `whl/repaired/`, which gives the manylinux tag (`REPAIR=0` skips it).
Release wheels for all Python versions: on a host with Docker, after the `py` step,
`cd ~/m2n-ruby/py-build && cibuildwheel --platform linux nccl_m2n` (staged copy with the library and the
nccl4py port next to it; configuration in `[tool.cibuildwheel]`, manylinux_2_28 image). On Ruby the default
Docker bridge has no DNS, so add `CIBW_CONTAINER_ENGINE="docker; create_args: --network=host"`, and with
`sudo docker` use a local copy and `--output-dir` (root cannot write the NFS home).

Install the wheels as root in every container that runs Python ranks (from the login node; `--no-deps`):

```bash
for n in $N1 $N2; do
  ssh $n "sudo docker exec $NAME bash -c \"/opt/venv/bin/pip install --no-deps --force-reinstall $HOME/m2n-ruby/whl/nccl4py-*.whl $HOME/m2n-ruby/whl/repaired/nccl_extensions-*.whl\""
done
```

## 3. C checks

```bash
bash $SD/run_matrix.sh                  # basic API: local, 1 node, 2 nodes (ring = PACK, direct = DIRECT copy)
bash $SD/dl_check.sh                    # reshard_bench --validate: PACK/DIRECT/PIPE/default x 4 shapes x 3 sizes, 4 -> 4
bash $SD/dl_check.sh 2node              # the same on 2 nodes, 8 -> 8 (hostfile ~/m2n-ruby/hostfile or HF=)
MPI_BIN=~/m2n-ruby/mk/bin/reshard_model_bench bash $SD/run_mpi.sh model 8 1node \
  --model-config $SD/../benchmarks/configs/model_configs/qwen3-32b.model.json \
  --system-config $SD/configs/amd-qwen3-32b-tp2pp2-to-tp1dp4-8gpus-mi355.json \
  --gpus-per-node 8 --iterations 10 --warmup 2 --algorithm direct --copy-algorithm pack --validate --validate-iterations 1
```

For the model bench on 2 nodes use `16 2node` and `configs/amd-qwen3-32b-tp4pp2-to-tp2dp4-16gpus-mi355.json`.
Logs: `~/m2n-ruby/logs/` (`dl/summary.txt`, `dl2/summary.txt` for `dl_check.sh`). Every script exits nonzero if any run failed or hung.

## 4. Python (`nccl.m2n`)

```bash
bash $SD/py_check.sh                                   # 1 node: python reshard_bench + DTensor (rccl, pg)
HF=~/m2n-ruby/hostfile bash $SD/py_check.sh 16         # 2 nodes
```

`py_env.sh` sets the runtime environment instead of `env.sh` (torch's `_rocm_sdk_core` HIP first,
`/opt/hipdev`, `PYTHONPATH` on `bindings/nccl_m2n/rocm` for `sitecustomize.py`, which keeps one `librccl` per
process, `LD_PRELOAD` of our RCCL for the Python ranks). This is a workaround, not the intended user setup:
torch from TheRock loads its own RCCL 2.28.9, which lacks five APIs `libnccl_m2n` calls, so our RCCL has to be
loaded first (details in `bindings/nccl_m2n/README.md`). `sitecustomize.py` is test-only (it writes the
private `rocm_sdk._ALL_CDLLS`). `/opt/hipdev/libamdhip64.so` is a symlink for hip-python, which dlopens the
unversioned name that `_rocm_sdk_core` lacks. `libnccl_m2n.so` comes from the wheel; `NCCL_M2N_LIBRARY`
overrides it. If the wrong RCCL is loaded, the first M2N call fails with `unsuitable RCCL` and the loaded paths.
The DTensor check takes the communicator from `nccl.core.Communicator` (`--comm rccl`) or from
`ProcessGroupNCCL._comm_ptr()` (`--comm pg`).

## Expected

Measured on 2 MI355X nodes (passed / failed / skipped, GiB/s per source rank). The failures below are
expected on this base; update the table once the staging pool fix, the DDA fix
([#15022](https://github.com/ROCm/rocm-systems/pull/15022)) and the PIPE commits land:

| Check | Result |
|---|---|
| `run_matrix.sh` | local_ring 231/0/14, mpi2_ring/direct 206/0/15, mpi2_all 410/0/24. On the current base `local_direct` fails 75 single-process `window_case*_2d_placement` cases, and `mpi1_*` fail `case0209 split_reverse_mesh` (HIP `invalid device pointer`) |
| `dl_check.sh` | 1 and 2 nodes: 45/48 PASS; PIPE 1xS -> 2x(S/2) fails at all 3 sizes (`VALIDATION FAILED`, LSA-sourced PIPE path not in this base), so it exits 1. The validation dims are odd (unaligned rows), so bandwidth is lower than with power-of-two shapes: ~1 GB PACK same 52 / cross 86 (1 node), 39 / 33 (2 nodes) |
| model bench | `VALIDATION PASSED`; aggregate ~630 GB/s (1 node), ~330 GB/s (2 nodes) |
| `py_check.sh` | bench PASS, cross-dim 8000x8000x16 PACK 172 (1 node) / 39.9 (2 nodes); DTensor `RESULT PASS failures=0` for `--comm rccl` and `--comm pg` |

## Troubleshooting

- `Too many open files` / `connection refused` in DevComm/GIN setup on 2 nodes: restart the container with
  `start-m2n-container.sh` (raised nofile).
- Rare SIGSEGV in `ld.so` right after `mpirun` starts (before any NCCL call): rerun.
- A killed run keeps the GPUs: `sudo docker exec $NAME pkill -x basic_api_test_` (or `reshard_bench`).
