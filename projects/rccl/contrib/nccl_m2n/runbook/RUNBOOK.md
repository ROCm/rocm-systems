# nccl_m2n on gfx950 (MI350X/MI355X): build, basic API matrix, copy-mode check and benchmarks on ruby

What this runs: the `nccl_m2n` (NVIDIA M2N reshard v0.2) basic API test matrix on top of RCCL,
single node and two nodes, GIN proxy transport (`NCCL_GIN_TYPE=2`) over the bnxt_re RoCE NICs
(sections 0-6). Sections 7-11 add the copy-mode / deadlock check, the `reshard_bench` matrix on
1/2/4 nodes, `reshard_batch_bench_user_window`, `reshard_model_bench` and the reference/SOL numbers. Section 12
covers the native CI ROCm 10.1 stack, > 4 nodes and PIPE in the basic matrix; section 13 the optional PyTorch
(`nccl.m2n`) check.

Code: `ROCm/rocm-systems`, branch `users/nikitaxgusev/m2n-enabling`;
RCCL and `nccl_m2n` are both built from it, nothing else from the monorepo is needed. The scripts live next
to this file (`projects/rccl/contrib/nccl_m2n/runbook`); below `$RB` is that directory and `$SRC` the clone.

## Expected result

Every run must finish with `failed 0` and `exit 0`. Skips are declared by the matrix itself
(shape not applicable to the world size, or the case needs its own `--filter`), they are not failures.

| Run | Layout | Copy path | Passed | Skipped | Time |
|---|---|---|---|---|---|
| `local_ring` | 1 node, 1 process, 8 GPUs | PACK | 231 | 14 | ~2 min |
| `local_direct` | 1 node, 1 process, 8 GPUs | DIRECT | 231 | 14 | ~2 min |
| `mpi1_ring` | 1 node, 8 MPI ranks | PACK | 208 | 13 | ~1.5 min |
| `mpi1_direct` | 1 node, 8 MPI ranks | DIRECT | 208 | 13 | ~1.5 min |
| `mpi1_all` | 1 node, 8 MPI ranks | PACK + DIRECT | 412 | 22 | ~2 min |
| `mpi2_ring` | 2 nodes, 16 MPI ranks | PACK | 206 | 15 | ~1.5 min |
| `mpi2_direct` | 2 nodes, 16 MPI ranks | DIRECT | 206 | 15 | ~1.5 min |
| `mpi2_all` | 2 nodes, 16 MPI ranks | PACK + DIRECT | 410 | 24 | ~2.5 min |

Reproduced 1:1 on two different pairs of nodes: all 8 runs gave exactly these counts, `exit 0`,
~13 min in total.

`--algorithm ring` uses the PACK copy path and `--algorithm direct` the DIRECT copy path unless
`--copy-algorithm` is given. PIPE (`--copy-algorithm pipe`) is not part of this runbook, see the end.

## 0. One-time prerequisites

- An account on ruby (`ruby-slurmlogin02.rckg.g03.cpe.ice.amd.com`) and `sudo docker` on the compute nodes.
- A passwordless SSH key that `mpirun` uses to reach the container on the other node.
  `$HOME` is on NFS, so one key works on every node:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_m2n
```

- Sources: a sparse clone with only `projects/rccl` (~190 MB, ~30 s), plus GoogleTest. `$HOME` is on NFS,
  so this is done once:

```bash
mkdir -p ~/m2n-ruby && cd ~/m2n-ruby
git clone --filter=blob:none --sparse --branch users/nikitaxgusev/m2n-enabling \
  https://github.com/ROCm/rocm-systems.git rocm-systems
git -C rocm-systems sparse-checkout set projects/rccl
git clone --depth 1 -b v1.14.0 https://github.com/google/googletest.git googletest
export SRC=~/m2n-ruby/rocm-systems RB=~/m2n-ruby/rocm-systems/projects/rccl/contrib/nccl_m2n/runbook
```

## 1. Allocate two nodes

```bash
salloc -N 2 -p meta64 --exclusive -t 4:00:00 -J m2n
scontrol show hostnames $SLURM_JOB_NODELIST          # e.g. cv350-rck-g03-c13-18, cv350-rck-g03-e17-18
```

Below, `N1` is the first node (you launch everything from it) and `N2` the second one.

## 2. Docker image (per node)

The image is Primus (TheRock HIP, OpenMPI in `/opt/openmpi`) plus rdma-core v61, which the
ruby bnxt_re kernel driver needs, plus sshd on port 2222 for `mpirun`, plus the Python setup of
section 13 (no NVIDIA cuda-python, hip-python, setuptools < 82). Check first whether the node already
has it (an image built before section 13 existed lacks the Python layer; rebuild it for section 13):

```bash
ssh $N1 'sudo docker images | grep m2n-ruby'
```

If not, build it on that node (the base image is ~76 GB, the first pull takes a while; `--network host`
is required, the default bridge network on the ruby nodes has no DNS/NAT and `apt-get` fails):

```bash
ssh $N1 "cd $RB && sudo docker build --network host -t m2n-ruby:primus-rdma61 ."
```

## 3. Start the container on both nodes

```bash
export NAME=${NAME:-m2n-multi}   # container name; set your own if m2n-multi is taken on the node
for n in $N1 $N2; do ssh $n "NAME=$NAME bash $RB/start-m2n-container.sh"; done
```

Each node must print `nofile soft=1048576 hard=1048576`. The default limit of 1024 open files is
not enough: two-node DIRECT then fails with `Too many open files` / `connection refused`.

Check that `mpirun` will be able to reach the second node:

```bash
ssh $N1 "ssh -p 2222 -i ~/.ssh/id_m2n -o StrictHostKeyChecking=no $N2 hostname"
```

## 4. Hostfile

Hostfile for the two-node runs:

```bash
scontrol show hostnames $SLURM_JOB_NODELIST | sed 's/$/.rck.dcgpu slots=8/' > ~/m2n-ruby/hostfile
```

## 5. Build (inside the container on N1)

```bash
ssh $N1
sudo docker exec -it -u $(id -u) -e HOME=$HOME ${NAME:-m2n-multi} bash -l
export RB=~/m2n-ruby/rocm-systems/projects/rccl/contrib/nccl_m2n/runbook
bash $RB/build.sh                      # RCCL (gfx950) -> ~/m2n-ruby/rccl-install, nccl_m2n -> ~/m2n-ruby/mk,
                                       # benches, rccl-tests
```

`build.sh rccl|m2n|bench|extras` rebuilds one part. Build logs go to `~/m2n-ruby/logs/build_*.log`.

The public base image ships an older `librccl.so` and `nccl.h` under `$ROCM_PATH`. `Makefile.rocm` links the
RCCL from `NCCL_HOME` by full path and puts its headers before `$ROCM_PATH/include`, so the old copy is not
picked up at build time; at run time `env.sh` puts `rccl-install/lib` first in `LD_LIBRARY_PATH`.

## 6. Run (same shell inside the container on N1)

```bash
bash $RB/run_matrix.sh                 # local, 1node and 2node
bash $RB/run_matrix.sh local 1node     # single node only
```

All eight runs take about 13 minutes. At the end it prints a `run / passed / failed / skipped / exit`
table; compare it with the table above. Full logs: `~/m2n-ruby/logs/<run>.log`.

Single runs, e.g. one family of shapes:

```bash
bash $RB/run_local.sh my_local --algorithm direct --filter full_sharding --gtest_filter='Matrix*'
bash $RB/run_mpi.sh   my_mpi2  16 2node --algorithm ring --filter tensor_size_sensitivity --gtest_filter='Matrix*'
```

## 7. Copy-mode / deadlock check (1 node, ~25 min)

`reshard_bench --validate`, 8 ranks (4 -> 4), each copy mode x 4 shapes (same-dim 0->0, cross-dim
0->1 and 1->0, mesh change 1x4 -> 2x2) x 3 sizes (2 MiB, 64 MiB, 1 GiB, int8). Every case has a timeout
(180 s, 300 s for 1 GiB); a case that does not finish is recorded as `HANG` with a `rocgdb` backtrace of
two ranks (`<case>.log.bt`), and the remaining sizes of that shape are skipped.

```bash
bash $RB/build.sh bench                                                    # once
bash $RB/dl_check.sh                                             # -> ~/m2n-ruby/logs/dl/summary.txt
MODES="pack direct" bash $RB/dl_check.sh                         # subset
```

Modes: `pack|direct|pipe` = `--algorithm direct --copy-algorithm <mode>`; `default` = no flags;
`auto` = `--algorithm auto` with `NCCL_RESHARD_LOG_LEVEL=INFO`. In nccl_m2n v0.2 `NCCL_RESHARD_ALGORITHM`
is parsed but has no effect: the copy path (`NCCL_RESHARD_COPY_ALGORITHM`, default PACK) decides the data
movement, so RING, DIRECT and AUTO give the same numbers for the same copy path.

Expected: **all PASS, no HANG**, except the known PIPE 1x4 -> 2x2 FAIL below. GiB/s per rank, 2 MiB / 64 MiB / 1 GiB:

| Mode | same 0->0 | cross 0->1 | cross 1->0 | 1x4 -> 2x2 (source ranks) |
|---|---|---|---|---|
| PACK | 3.7 / 39 / 51 | 3.6 / 66 / 176 | 3.7 / 68 / 176 | 2.2 / 35 / 52 |
| DIRECT | 0.8 / 3.3 / 3.8 | 0.7 / 7.1 / 9.6 | 0.6 / 7.5 / 9.5 | 0.5 / 7.7 / 6.0 |
| PIPE | 2.6 / 27 / 44 | 1.7 / 18 / 19 | 1.7 / 18 / 19 | **FAIL** (known; 0.6 / 2.1 / 40 with the LSA-tile path) |

Copy DIRECT moves every byte with GIN put, also inside a node (no xGMI/LSA path), and runs one CTA per
peer, so it is far below PACK. PIPE is slow for strided destination rows shorter than 64 KiB (cross-dim
up to 1 GiB, 1x4 -> 2x2 below 1 GiB here): those are copied one row per thread.

Known FAIL on the current base: PIPE with a destination replicated inside a node (1x4 -> 2x2, LSA-sourced
tiles) ends with `VALIDATION FAILED` and `exit 1` at every size, because this base does not carry the AMD
LSA-tile path without TMA (`e0f6f109c26`). `dl_check.sh` and the aggregators report it as FAIL.
`dl_check.sh` now validates on odd dims (124x124x129 / 508x508x257 / 1020x1020x1021) so that source shards do
not alias under the `% 256` pattern; correctness is the same as above, but rows are unaligned and bandwidth is
lower than in the table (measured on power-of-two dims), e.g. ~1 GB PACK cross-dim ~84 GiB/s.

## 8. reshard_bench matrix (1, 2, 4 nodes + asymmetric)

`bench_matrix.sh` runs one scale: shapes x sizes (64 KiB ... 4 GiB, x4 steps, int8 tensors
`[n, n, 16]`) x configs `D_PACK, D_DIRECT, D_PIPE, R_PACK` (`--algorithm direct|ring --copy-algorithm ...`).
Each point is launched `ROUNDS=4` times; the config order alternates per round (ABBA) to spread drift,
a validate-only launch on non-aliasing dims (round `v`, not in the tables; per axis the lcm of its shard counts
times an odd number, depth 17, so every shard is distinct modulo 256 up to 16 shards on dim1) runs first
(the bench validates `(global index) % 256`, which aliases on power-of-two dims), then round 0 with `--validate`,
the others perf-only. Per-launch timeout 240 s (400 s for
1 GiB, 600 s for 4/16 GiB); a `HANG` or `FAIL` stops that config at that point. One CSV line per launch
goes to `~/m2n-ruby/logs/bench/<label>/results.csv`, logs next to it.

`phases.sh` chains phases inside the container, `m2n_job.sbatch` is the host-side wrapper (builds the
image on nodes that lack it, starts `$NAME` on every node, runs the phases on the first node, removes the
containers). Pass `PHASES` through the environment: `sbatch --export=PHASES=a,b` splits at commas.

```bash
cd $RB; S=same00,cross01,cross10,mesh2; C=D_PACK,D_DIRECT,D_PIPE,R_PACK; export NAME=m2n-$USER
PHASES="m:n1:8:4:4:$S:$C:0 auto:n1:8:4:4:0 a2a:n1:8:0" \
  sbatch -J m2n-n1 -N 1 --export=ALL m2n_job.sbatch                                   # ~4 h
PHASES="m:n2:16:8:8:$S:$C:1 m:n2asym:12:8:4:same00,cross01:$C:1 auto:n2:16:8:8:1 a2a:n2:16:1" \
  sbatch -J m2n-n2 -N 2 --export=ALL m2n_job.sbatch                                   # ~3 h
PHASES="m:n4:32:16:16:$S:$C:1 m:n4asym:24:16:8:same00,cross01:$C:1 m:n4x16g:32:16:16:cross01:D_PACK:1:16G auto:n4:32:16:16:1 a2a:n4:32:1" \
  sbatch -J m2n-n4 -N 4 --export=ALL m2n_job.sbatch                                   # ~4 h
```

Phase syntax: `m:<label>:<np>:<src ranks>:<dst ranks>:<shapes>:<configs>:<hostfile 0|1>[:<sizes>]`,
`auto:<label>:<np>:<src>:<dst>:<hf>`, `a2a:<label>:<np>:<hf>`. Ranks are placed 8 per node in order, so
`src` ranks come first: 8 -> 4 on 2 nodes puts the trainer on node 1 and the generator on half of node 2.
To rerun a few points (for example after interference on the node) use a `<label>rerun` label with the
same shapes/configs and a size list; `aggregate.py` replaces those points of `<label>`.

Aggregate on any host with Python 3 (copy `~/m2n-ruby/logs/bench` first):

```bash
python3 $RB/aggregate.py ~/m2n-ruby/logs/bench > matrix.md          # per scale x shape tables
python3 $RB/aggregate_batch.py ~/m2n-ruby/logs/bench/batch > batch.md
```

Cells are the median over launches of the per-source-rank bandwidth (`--- Sources only` block of the
bench output; the bench's "GB/s" is GiB/s), `±N%` = (max - min) / median when >= 5%, plus `%SOL` and the
max per-iteration latency for D_PACK.

**SOL model** (per source rank, GiB/s): intra-node xGMI 76.8 GB/s per link per direction (MI350X/MI355X:
7 links x 153.6 GB/s bidirectional) times the number of distinct destination peers (1 for same-dim, `dst`
for cross-dim, 2 for 1xS -> 2x(D/2)); inter-node one 400G Thor2 NIC per GPU = 50 GB/s, times
`min(1, dst/src)` for asymmetric splits (the destination NICs are the bottleneck).

**alltoall reference**: `rccl-tests/alltoall_perf -b 64K -e 4G -f 4` on the same ranks (`a2a` phase).
Compare per link / per NIC: on 1 node alltoall moves `size/8` per peer link, on 2 (4) nodes `1/2` (`3/4`)
of the per-rank volume crosses the NIC.

Expected: all
~2,350 launches PASS/OK except one intermittent SIGSEGV in `ld.so` during MPI start (see
Troubleshooting). Large-message D_PACK (4 GiB), GiB/s per source rank and % SOL:

| Scale | same 0->0 | cross 0->1 | cross 1->0 | 1xS -> 2x(D/2) | alltoall_perf algbw 4 GiB |
|---|---|---|---|---|---|
| 1 node 4->4 | 54.5 (76%) | 189 (66%) | 189 (66%) | 53.1 (37%) | 383 GB/s (47.8 GB/s per link) |
| 2 nodes 8->8 | 42.7 (92%) | 42.6 (92%) | 42.6 (92%) | 23.1 (50%) | 96.7 GB/s (48.3 GB/s NIC) |
| 2 nodes 8->4 | 21.8 (94%) | 21.8 (94%) | - | - | - |
| 4 nodes 16->16 | 40.8 (88%) | 39.7 (85%) | 39.8 (85%) | 11.0 (24%) | 64.4 GB/s (48.3 GB/s NIC) |
| 4 nodes 16->8 | 21.4 (92%) | 21.3 (92%) | - | - | - |

16 GiB cross-dim on 4 nodes: 42.9 GiB/s per rank (92% SOL), PASS. R_PACK equals D_PACK within noise.
DIRECT copy reaches 4-17 GiB/s. For 1xS -> 2x(D/2) (destination replicated x2) PACK sends every replica
over the network and is bound by the destination NICs (23.3 GiB/s per source rank for S = D): 23.1 on 2 nodes,
11.0 on 4 nodes. PIPE fans the replicas out inside the destination node and reaches 43.1 / 19.0 GiB/s,
but has reproducible cliffs (1 node cross 1 GiB: 18.8 vs 27.5 at 256 MiB; 2 nodes mesh change 1 GiB:
2.1 vs 3.5 at 256 MiB and 43 at 4 GiB; PIPE mesh change was measured with the LSA-tile path and fails
validation on the current base, see section 7). All numbers are the `Sources only` block (per source rank);
for the replicated shape the bench's `Overall` average is larger because destination ranks hold twice the bytes. Below 16 MiB everything is latency bound:
~0.13 ms per call on 1 node, ~0.3-0.4 ms on 2 nodes, ~0.45-1.0 ms on 4 nodes.

## 9. reshard_batch_bench_user_window (sequential vs concurrent, `--num-comms`)

`batch_phases.sh`: 4 tensors (`[1024,1024]`, `[4096,4096]`, `[16384,16384]` int8), same-dim and
cross-dim patterns, copy PACK/DIRECT/PIPE x `--num-comms 1 2 4`, 3 launches (first with `--validate`),
after a validate-only launch on non-aliasing tensors (`1008`/`4080`/`16368`, round `v`).

```bash
PHASES_SCRIPT=batch_phases.sh PHASES="b1:8:4:4:0" sbatch -J m2n-b1 -N 1 -t 2:00:00 --export=ALL m2n_job.sbatch
PHASES_SCRIPT=batch_phases.sh PHASES="b2:16:8:8:1" sbatch -J m2n-b2 -N 2 -t 2:00:00 --export=ALL m2n_job.sbatch
```

Expected: every launch PASS/OK (54 launches: 27 per scale). 1 node, 256 MiB tensors, PACK seq/con: same-dim 51/55 GiB/s,
cross-dim 148/192 GiB/s at `--num-comms 2`. On 2 nodes concurrent issue helps small/medium tensors (up to 1.8x at
`--num-comms 2`) and is ~1.0-1.1x at 256 MiB, where the NIC is already saturated; `--num-comms 4` is worse
than 2 for PACK/PIPE.

## 10. reshard_model_bench (Qwen3-32B weight sync)

`model_phases.sh` runs `reshard_model_bench --model-config qwen3-32b.model.json` with `--algorithm auto`
and `--algorithm direct --copy-algorithm pack`, 4 launches each (first with `--validate`). The shipped
system configs are written for GB200/H100 (4 GPUs per node); `amd-configs/` adds 8-GPU-per-node layouts.

```bash
SC=$SRC/projects/rccl/contrib/nccl_m2n/benchmarks/configs/system_configs; A=$RB/amd-configs
PHASES_SCRIPT=model_phases.sh MODEL_RUNS="n1_tp4:8:0:$SC/qwen3-32b-dp1-tp4-to-dp1-tp4.json n1_amd:8:0:$A/amd-qwen3-32b-tp2pp2-to-tp1dp4-8gpus-mi355.json n1_tp2:4:0:$SC/qwen3-32b-dp1-tp2-to-dp1-tp2.json" \
  sbatch -J m2n-model1 -N 1 -t 2:00:00 --export=ALL m2n_job.sbatch
PHASES_SCRIPT=model_phases.sh MODEL_RUNS="n2_amd:16:1:$A/amd-qwen3-32b-tp4pp2-to-tp2dp4-16gpus-mi355.json" \
  sbatch -J m2n-model2 -N 2 -t 2:00:00 --export=ALL m2n_job.sbatch
PHASES_SCRIPT=model_phases.sh MODEL_RUNS="n4_gb200cfg:32:1:$SC/qwen3-32b-tp2pp4dp2-to-tp1dp16-32gpus-gb200.json" \
  sbatch -J m2n-model4 -N 4 -t 2:00:00 --export=ALL m2n_job.sbatch
```

Expected (all validations PASS, AUTO == DIRECT+PACK within 1%): max latency of one full weight sync

| Config | Ranks | Latency | Aggregate |
|---|---|---|---|
| TP4 -> TP4 (shipped) | 8 | 18.6 ms | 408 GB/s |
| TP2 -> TP2 (shipped) | 4 | 35.9 ms | 212 GB/s |
| TP2 PP2 -> TP1 DP4 (amd, 1 node) | 8 | 37.4 ms | 630 GB/s |
| TP4 PP2 -> TP2 DP4 (amd, node 1 -> node 2) | 16 | 70.4 ms | 335 GB/s |
| TP2 PP4 DP2 -> TP1 DP16 (shipped GB200 32-GPU, 4 MI355 nodes) | 32 | 310 ms | 379 GB/s |

The DeepSeek-V3 (128/256 GPUs), Qwen-235B (128) and the H100 64-GPU configs need 8-32 MI355 nodes and
were not run.

## 11. Where to look

- `~/m2n-ruby/logs/dl/summary.txt` (section 7), `~/m2n-ruby/logs/bench/<label>/results.csv` + logs
  (section 8), `logs/bench/batch/`, `logs/bench/model/`, `logs/bench/<label>/alltoall.log`,
  `logs/bench/job-*.out` (one per sbatch job).

## 12. Beyond the container: native CI ROCm 10.1, > 4 nodes, PIPE basic matrix

- Native stack: `native/env-ci-rocm-10.1.sh` (CI ROCm 10.1.0 + CI OpenMPI 5.0.9, plain ssh with the default
  cluster key on port 22, Unix Makefiles instead of Ninja). Use it as `env.sh` in a copy of this directory;
  `build.sh` and the launchers take `MPI_HOME`, `CMAKE_GENERATOR`, `SSH_KEY` and `SSH_PORT` from it.
  Submit with `sbatch --propagate=NONE` and raise `ulimit -n` to the hard limit (the env file does it):
  Slurm hands the login node's soft `nofile` (1024) to the job, which hangs `mpi1_direct` at
  `case0064 uneven_ratio` (ranks diverge inside `ncclDevCommCreate`) and aborts 8+ node runs
  (`Too many open files`, then HIP `out of memory`).
- Expected on the native stack: basic matrix identical to the table above; best launches equal to the
  container, but 12-50% of `reshard_bench` launches show a ~21 ms per-iteration floor (0% in the container).
- More than 16 shards per side: cross-dim and mesh change fail with `invalid argument`
  (`validateReshardPlanLimits ... sources=32 > MAX_SOURCES=16`). Use the replicated shapes `rcross01`
  (`S/16 x 16 -> D/16 x 16`) and `rmesh` (`S/16 x 16 -> D/8 x 8`) of `bench_matrix.sh`. PIPE `rmesh` hangs
  on 8 and 16 nodes.
- Model bench at 64/128 ranks: shipped `dsv3-pp-64gpus-gb200`, `dsv3-128gpus-gb200`,
  `qwen235b-TP2CP2PP4EP16-128gpus-gb200` fit 8-GPU nodes as is; model manifests `dsv3-toy.model.json` and
  `amd-configs/generate_qwen3_235b.py` (4 of 94 layers). Pass the manifest as the 5th field of `MODEL_RUNS`.
- PIPE in the basic matrix: `pipe_chunks.sh local|1node|2node` (one decade of cases per process, PIPE has
  32 persistent control slots per communicator). The 4 single-node `uneven_ratio 240x240` cases that hang
  (0066, 0078, 0081, 0093, `KNOWN_HANG`) run in their own processes, and a decade that still times out is
  rerun one case per process, so a hang only loses its own case. Expected: 2 nodes 206/0/16; 1 node those 4
  cases hang (`exit 124`), everything else passes (local 228/0/14).
- Proxy CPU: PACK ~600% CPU per rank (main busy-wait + `ncclGinProgress` and `ncclRmaProxyProgressThread`
  per communicator, two communicators with split-comm), PIPE ~300%.

## 13. Optional: PyTorch (`nccl.m2n`)

`nccl.m2n` is in this tree: `projects/rccl/bindings/nccl_m2n` (the `nccl_m2n` part of upstream
`nccl-extensions` `python/` at its `UPSTREAM_VERSION`, built on the RCCL nccl4py port in
`projects/rccl/bindings/nccl4py`). Build, install and checks are in `../scripts/README.md` (sections 2 and 4);
they use the same `~/m2n-ruby` layout and image as this runbook. The communicator comes from
`nccl.core.Communicator` (`--comm rccl`) or `ProcessGroupNCCL._comm_ptr()` (`--comm pg`).

```bash
SD=$SRC/projects/rccl/contrib/nccl_m2n/scripts
bash $SD/build.sh py                      # nccl4py + nccl.m2n wheels -> ~/m2n-ruby/whl (auditwheel-tagged: whl/repaired/)
# install the wheels as root in every container (../scripts/README.md section 2), then in the container:
bash $SD/py_check.sh                      # 1 node: python reshard_bench --validate + DTensor (--comm rccl, pg)
HF=~/m2n-ruby/hostfile bash $SD/py_check.sh 16   # 2 nodes
```

Release wheels (manylinux_2_28, cp310-cp314) come from `../scripts/manylinux_wheel.sh` on a host with Docker and
cibuildwheel: `M2N_RCCL_HOME=~/m2n-ruby/rccl-install bash $SD/manylinux_wheel.sh [outdir]` (default
`~/m2n-ruby/wheelhouse`). It builds `libnccl_m2n.so` and the bindings inside the manylinux image; the Docker/root
notes for Ruby are in `../scripts/README.md`.

## Troubleshooting

- **`Too many open files`, `connection refused` in DevComm/GIN setup on two nodes:** the container
  was started without the raised fd limit. Restart it with `start-m2n-container.sh`.
- **A run hangs and the next one cannot get the GPUs:** a test process from a killed run is still
  alive. On each node: `sudo docker exec ${NAME:-m2n-multi} pkill -x basic_api_test_` (the process name is
  truncated to 15 characters; do not use `pkill -f` with a pattern that matches your own shell).
- **Only rank 0 prints test output in MPI runs:** that is the harness, it redirects stdout of the
  other ranks to `/dev/null`. For per-rank NCCL logs set `NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/tmp/nccl.%h.%p`.
- **Every run has a timeout** (`TMO`, default 3600 s); a hang ends with `=== EXIT 124`.
- **`docker build` fails in the `apt-get` step without output:** the default bridge network on the ruby
  nodes has no DNS/NAT; use `--network host` (the scripts do).
- **`undefined symbol: ncclPutSignal` (tests) / `NCCL M2N requires NCCL 2.30.5 or newer` (model bench):**
  the base image's old librccl/nccl.h wins over `rccl-install`; `Makefile.rocm` on this branch avoids it (section 5).
- **`ncclDevCommCreate ... invalid usage` in every case on one node:** seen once, outside the container
  and before `rccl: agree on DDA IPC init across ranks and clear its HIP errors`, with `NCCL_CUMEM_ENABLE=1`
  (the same error as with `NCCL_CUMEM_ENABLE=0`); the same binaries passed on other nodes, and the node's
  RDMA/amdgpu setup matched a working node. Rerun with `NCCL_DEBUG=WARN`; exclude the node if it persists.
- **Rare SIGSEGV in `ld.so` from `libopen-pal` right after `mpirun` starts** (1 launch in ~2,900, before
  any NCCL call): a dlopen of an MPI component from the NFS home. Rerun the point. Fresh libraries on NFS
  can also SIGBUS on first load (seen on pit2); reading them once on every node before the run avoids it.
- **Disk on the compute nodes:** the image is ~76 GB per node in the local Docker store and some nodes
  are above 85% of 3.4 TB; check `df -h /` and remove your own images (`sudo docker rmi m2n-ruby:primus-rdma61
  rocm/primus:v26.4`) when done. Never prune other users' images.
- **`--algorithm` / `NCCL_RESHARD_ALGORITHM` change nothing:** expected in v0.2 (section 7). The benches
  accept `--algorithm auto|ring|direct` and otherwise keep `NCCL_RESHARD_ALGORITHM` from the environment.
- **Another job on the same node** (builds, other containers) skews small-message timings; rerun the
  affected points with a `<label>rerun` phase (section 8).

## PIPE in the basic matrix (not covered by run_matrix.sh)

`--copy-algorithm pipe` passes on two nodes, but on a single node 4 `uneven_ratio` 240x240 cases
(sr->rs 1:3 and rs->sr 3:1, both `sd` variants) hang: the PIPE channel planner puts two RDMA peers
of one rank on the same channel and that rank bails out of a collective setup. PIPE also keeps 32
persistent control slots per communicator, so the matrix has to be split into chunks of at most 32
shapes, one process each (`pipe_chunks.sh`, section 12). Leave PIPE out unless you are working on that planner.
