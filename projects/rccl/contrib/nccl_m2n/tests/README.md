# `tests/` — basic_api correctness suite

Two gtest binaries exercise `ncclReshardWithWindow` across a parameter
matrix mirroring an external pytest reference suite. Both share the test
descriptor table and per-case logic in `basic_api_test_core.h`; only the
bootstrap differs.

| Binary | Bootstrap | Use when |
|---|---|---|
| `build/bin/basic_api_test_mpi`   | MPI (`mpirun`)              | Multi-host MPI runs, large rank counts. |
| `build/bin/basic_api_test_local` | Single-process pthreads via `ncclCommInitAll` | Dev workstation, no MPI install required. |

Both use only the public API in `src/nccl_m2n.h`.

## Building

`pipe_commb_resources_test` is a host-only test of stable PIPE parent namespace
IDs, exhaustion without wrapping, control-slot ranges, resource sizing and
overflow. It does not require CUDA, NCCL or GTest at runtime. Build directly:

```sh
c++ -std=c++17 -Wall -Wextra -Isrc tests/pipe_commb_resources_test.cc -o /tmp/pipe_commb_resources_test
/tmp/pipe_commb_resources_test
```

From a configured CUDA/NCCL build, `make -C tests pipe_commb_resources_test`
builds and runs it. CMake includes it when tests are enabled, even without GTest.
`NCCL_RESHARD_GEN_COMM_PARENT_CAPACITY` reserves PIPE's persistent namespaces
before first use; it is independent of `NCCL_RESHARD_GEN_COMM_CONCURRENCY`.
The latter selects PIPE lanes and optionally caps PACK's effective physical
slots per bucket. An unset value leaves PACK's configured slot counts unchanged.

```sh
make tests
```

`make tests` builds the public basic_api binaries when a googletest source
tree is available. `tests/gtest.mk` looks for one in this order:

1. `$(GTEST_DIR)` — caller-supplied (highest priority).
2. `third_party/googletest/googletest/` — the repository-root googletest
   submodule, shared with the other extensions in this repo. Fetch it with
   `git submodule update --init third_party/googletest`.
3. `tests/googletest/` — vendored alongside the tests.

If none resolves, `make tests` prints a friendly skip message and
returns success — the library and benchmarks build either way. To build
the tests against any googletest checkout
(<https://github.com/google/googletest>):

```sh
make tests GTEST_DIR=/path/to/googletest
```

## Test groups (mirror of pytest matrix)

| Group | Minimum world | Tensor shape | dtype | Notes |
|---|---|---|---|---|
| `full_replication`         | `world ≥ 4`, even           | (200, 200) | fp32 / bf16 / uint8 | 1D mesh per side, both replicate |
| `full_sharding`            | `world ≥ 4`, even           | (200, 200) | fp32 / bf16 / uint8 | 1D mesh per side, sharding_dims ∈ {(0,0),(0,1),(1,0),(1,1)} |
| `2d_placement`             | `world ≥ 4`, even           | (200, 200) | bf16 | n_shards ∈ {(2,4),(4,2),(2,2)}, placement ∈ {sr/sr, rs/rs, sr/rs, rs/sr} |
| `uneven_ratio`             | `world ≥ 4`, `world % 4 == 0` | (240, 240) | bf16 | ratio ∈ {(3,1),(1,3)}, n_shards (2,2) |
| `tensor_size_sensitivity`  | `world ≥ 4`, even           | 576², 3072², 3072×6144 | bf16 | n_shards (4,4) |
| `nd_tensors`               | `world ≥ 4`, even           | (64,128,128), (128,64,64) | bf16 | 3D only; the `(64,128,128), sd=(0,1)` historical case lives in `cross_dim_regression` |
| `staging_slot_pressure`    | `world = 4`                 | (64,128,128) | bf16 | 3D Shard(0) → Shard(2) default-API reproducer for CI when run with a 4 MiB staging channel |
| `stream_churn`             | `world ≥ 4`, even           | (64,64) | fp32 | Reuses the case with fresh destination streams in the MPI driver |
| `stream_ordering`          | `world ≥ 4`, even           | (64,64) | fp32 | Asynchronous producer, reshard, and consumer ordering |
| `graph_capture`            | `world ≥ 4`, even           | (64,64) | fp32 | MPI-only first-use and prewarmed variants verify capture rejection |
| `1d_full_sharding`            | `world ≥ 4`, even           | (8192,)            | fp32 / bf16 / uint8 | 1D mesh, sd=(0,0) (only one tensor axis) |
| `1d_2d_placement`             | `world ≥ 4`, even           | (8192,)            | bf16 | n_shards ∈ {(2,4),(4,2),(2,2)}, placement ∈ {sr/sr, rs/rs, sr/rs, rs/sr} |
| `1d_uneven_ratio`             | `world ≥ 4`, `world % 4 == 0` | (16384,)         | bf16 | ratio ∈ {(3,1),(1,3)}, n_shards (2,2) |
| `1d_tensor_size_sensitivity`  | `world ≥ 4`, even           | (16384,), (1048576,), (4194304,) | bf16 | n_shards (4,4) |
| `cross_dim_regression`        | `world ≥ 8`, even           | (200, 200), (64,128,128) | bf16 | Targeted historical cross-dimension data-layout regressions |
| `split_tiny_contribution`  | `world ≥ 4`, even           | (4,) | uint8 | Every CTA signals when a contribution is smaller than the CTA count |
| `split_reverse_mesh`       | `world ≥ 4`, even           | (4096,) | uint8 | Source communicator ranks precede destination ranks when the destination mesh starts at rank 0 |

`basic_api_test_local` omits `graph_capture`. Its pthread ranks share one
process, while CUDA graph-capture coverage runs through `basic_api_test_mpi`.
The MPI driver runs `stream_churn`, `stream_ordering`, and `graph_capture`
with both caller streams and internal stream-pool mode. Its
`group_mixed_context` filter also checks ordering across overlapping
communicator contexts.

Each case carries additional **runtime feasibility checks** (n_shards
must divide `src_total` / `dst_total`, the chosen tensor dim must divide
by the shard count). Cases that don't fit at the current world size are
reported as `SKIP` and don't fail the run. `world_min` is set to the
smallest world where *some* case in the group can possibly run; the
pytest source uses higher floors (8 / 32) but those are arbitrary —
the C kernel has no inherent 8-rank requirement.

## Sample invocations

### Single-host, no MPI

```sh
# List the case matrix without running anything.
./build/bin/basic_api_test_local --list

# Default: ranks = cudaGetDeviceCount(); skip cases that need more.
./build/bin/basic_api_test_local

# Smaller rank count (e.g., a 2-GPU box).
./build/bin/basic_api_test_local -N 2

# Run only one group.
./build/bin/basic_api_test_local --filter full_replication

# Sweep CTA counts in separate processes, including counts that do not divide
# evenly by the GIN context count.
bash tests/run_cta_count_regression.sh -N 4

# Select the DIRECT reshard algorithm.
./build/bin/basic_api_test_local --algorithm direct
```

### Multi-host, MPI

```sh
# Wrapper script (defaults to N=8):
tests/run_basic_api_tests.sh
tests/run_basic_api_tests.sh -N 32 --filter nd_tensors

# Or directly:
mpirun -np 8 ./build/bin/basic_api_test_mpi --filter 2d_placement

# Focused PACK host-RMA lease regression: two warmed communicators
# share one destination GPU and one physical staging slot. Require INFO logs
# to contain "pack-lsa-hput" so a kernel fallback cannot certify it.
NCCL_NUM_RMA_CTX=4 \
NCCL_RESHARD_PACK_BUFFSIZES=8192:1 \
NCCL_RESHARD_SPLIT_COMM=0 \
NCCL_RESHARD_LOG_LEVEL=INFO \
mpirun -np 3 ./build/bin/basic_api_test_mpi \
  --filter pack_reduced_bucket \
  --gtest_filter='PackMpiTest.*' \
  --algorithm ring --api default --copy-algorithm pack \
  --lb-mode uniform
```

## CLI flags (both binaries)

| Flag | Effect |
|---|---|
| `--list`                              | Print `idx min_world name` per case and exit (skips NCCL init). Honors `--filter` / `--max-world` / `--min-world`. |
| `--filter <substr>`                   | Only run cases whose name contains `<substr>`. |
| `--gtest_filter=<pattern>`            | Native gtest filtering over sanitized parameter names, after the custom matrix filters above. |
| `--max-world <N>`                     | Skip cases whose minimum required world size is `> N`. |
| `--min-world <N>`                     | Skip cases whose minimum required world size is `< N`. |
| `--algorithm ring\|direct`            | Legacy test-scenario label (default ring). If no copy algorithm is supplied, `ring` selects PACK and `direct` selects DIRECT. `basic_api_test_mpi` also accepts `all`, registering one parameter per label. |
| `--copy-algorithm direct\|pack` | Copy transport exercised through every selected API surface. |
| `--lb-mode uniform\|node`             | Load-balance mode (default uniform). |
| `--verbose`                           | Library verbose + per-rank diagnostics. |
| `-N <ranks>` *(local only)*           | Number of ranks/threads (clamped to device count). |

### Min-world introspection (CI / scheduler integration)

Each test case carries dynamic preconditions (n_shards must divide src/dst
totals, global tensor dim must divide the resulting shard count). The
helper `computeMinWorldForCase` walks those preconditions to find the
smallest world size at which a case actually runs, and `--list` exposes it
so a CI script can request just enough ranks per batch:

```
$ ./build/bin/basic_api_test_local --list | head
# total_cases=226
# columns: idx min_world name
[   0]    4 full_replication[gd=200x200,esz=4]
[   5]    4 full_sharding[gd=200x200,sd=0/0,esz=4]
[  25]    8 2d_placement[gd=200x200,m=2x4_sr/sr,sd=0/0,esz=2]
[  73]    8 uneven_ratio[gd=240x240,m=2x2_sr/sr,sd=0/0,esz=2,ratio=3:1]
[ 105]    8 tensor_size_sensitivity[gd=576x576,m=4x4_sr/sr,sd=0/0,esz=2]
```

Combine `--max-world` and `--min-world` to bin a CI run into rank tiers
and provision exactly the allocation each tier needs:

```bash
# Tier 1: smoke at 4 ranks
mpirun -np 4  basic_api_test_mpi --max-world 4

# Tier 2: cases that need 5..8 ranks, run at world=8
mpirun -np 8  basic_api_test_mpi --min-world 5 --max-world 8

# Tier 3: cases that need >8 ranks, run at world=32
mpirun -np 32 basic_api_test_mpi --min-world 9 --max-world 32
```

## Exit code

Returns `1` if any case reports `FAIL`, `0` otherwise. `SKIP` does not
fail the run.

## What the validator checks

Every byte in the global tensor takes value `(global_idx % 256)`.
Source ranks initialize their slice with that pattern; destination
ranks read the post-reshard contents and confirm each byte matches the
expected value at its global index. The validator is exact (no
`atol` tolerance), works for any `element_size`, and is correct for
both same-dim and cross-dim sharding because it depends only on each
byte's global position — not on where the data was sourced from.

## Case naming

`<group>[gd=DxD,m=AxB_<sr|rs>/<sr|rs>,sd=S/D,esz=N,ratio=R:R]`

- `gd` — global tensor dims
- `m` — mesh axis-0 sizes (src x dst)
- placement strings are `sr` (Shard outer, Replicate inner) or `rs` (Replicate outer, Shard inner)
- `sd` — sharding dims (src/dst)
- `esz` — element size in bytes (1=uint8, 2=bf16, 4=fp32)
- `ratio` only present for uneven src/dst splits

Filter substrings target any portion of the name, so
`--filter "esz=2"` runs every bf16 case across all groups, and
`--filter "sr/rs"` runs every "sr→rs" placement combo.

---

# Smoke test

A practical pre-merge smoke is to run the broad C-level functional
matrix plus the RING/DIRECT x same-dim/cross-dim acceptance combos with
`--validate`, then check the PASS/FAIL and bandwidth summary. Exits are
non-zero on any failure so the run can be used as a single CI gate.

## What it runs

| § | Check | Expected output |
|---|---|---|
| §1 | `basic_api_test_mpi` — full C-level matrix | Cases run/skip, `0` exit if no FAIL |
| §2a | `reshard_bench --validate` RING cross-dim (`sd=0 dd=1`) | `*** VALIDATION PASSED ***`, bandwidth |
| §2b | `reshard_bench --validate` RING same-dim (`sd=0 dd=0`) | same |
| §2c | `reshard_bench --validate` DIRECT cross-dim (`sd=0 dd=1`) | same |
| §2d | `reshard_bench --validate` DIRECT same-dim (`sd=0 dd=0`) | same |
| §3 | Summary table | one row per check with PASS/FAIL + bandwidth |

Logs for each check land under a per-job directory printed at the end of
the run.

Cluster-specific launch wrappers and perf-regression workflows are not
shipped with this contrib; copy `run_basic_api_tests.sh` and adapt its
environment for your system.

## Multi-dimensional placement coverage

With `NCCL_M2N_BUILD_TESTS=ON`, build `multi_dim_geometry_test` and
`multi_dim_test_mpi` (neither requires GTest). Make provides the same target
names under `nccl_m2n/tests`. `make -C nccl_m2n tests`, or the default
`make` inside `nccl_m2n/tests`, builds both placement tests even without
GTest; GTest-dependent binaries are included when GTest is available.

The local CI runner requires `MULTI_DIM_GEOMETRY_PASS pairs=109872`.
The MPI runner requires each sweep to report `MULTI_DIM_GPU_PASS cases=356`
with the expected API, grouping, nonblocking mode, and mesh shapes. Its
four-rank profile runs two sweeps on `2 x 1` meshes. The eight-rank profile
runs ten sweeps on `2 x 2` meshes: DIRECT/PACK with RING/DIRECT, default and
grouped-window APIs, plus PACK nonblocking default/window calls. Blocking
default sweeps use UNIFORM balancing; window and nonblocking sweeps use
NODE_AWARE.

`multi_dim_geometry_test` checks every placement pair for tensor ranks 1–3
on non-square and unit-axis meshes. Its independent global-box oracle checks
rank mapping, overlap volume, local bounds, element correspondence, complete
coverage, and rejection of invalid dimensions, global shapes, and overflow.
It requires no GPU operations.

`multi_dim_test_mpi` checks all 353 source/destination placement pairs across
1D, 2D, and 3D tensors, plus three larger transfers. Each destination and
source is checked against a global-coordinate data oracle and guard bytes.
Use 8 ranks (4 GPUs per side) by default:

```bash
NCCL_RESHARD_COPY_ALGORITHM=PACK mpirun -np 8 build/bin/multi_dim_test_mpi
NCCL_RESHARD_COPY_ALGORITHM=PACK mpirun -np 8 build/bin/multi_dim_test_mpi --window --grouped
```

Repeat with `DIRECT`, and `NCCL_RESHARD_ALGORITHM=RING` or `DIRECT`.
`--nonblocking` tests a nonblocking caller communicator. Optional
`--src-ranks`, `--src-rows`, and `--dst-rows` select asymmetric or non-square
meshes; their shard products must divide the test shapes (24 and 48/96/192).
Incompatible products fail the whole launch before CUDA/NCCL setup; cases
are not silently skipped.
For example, 12 ranks with `--src-ranks 6 --src-rows 2 --dst-rows 3` exercises
`2 x 3` to `3 x 2` meshes.
