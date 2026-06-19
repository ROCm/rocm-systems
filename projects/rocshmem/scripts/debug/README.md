# rocSHMEM Deadlock Analysis Tools

Scripts for identifying and diagnosing deadlocks in rocSHMEM applications using
AMD's ROCm debugger (`rocgdb`).

## Files

| File | Purpose |
|---|---|
| `rocgdb_deadlock_analysis.py` | rocgdb Python script: coalesces GPU wavefront backtraces, identifies rocSHMEM API entry points, and provides deadlock hints |
| `attach_deadlock_analysis.sh` | Shell wrapper: finds all running instances of an executable on the local node, attaches to each, saves per-process output, prints a compact summary |
| `mpiexec_deadlock_analysis.sh` | Multi-node wrapper: deploys `attach_deadlock_analysis.sh` to every node via mpiexec, then runs cross-rank coalescing |
| `cross_rank_deadlock_analysis.py` | Cross-rank coalescer: reads per-rank output files, groups identical backtrace patterns across ranks, highlights patterns uncommon among ranks |
| `deadlock_test.cc` | Intentional-deadlock test program used to validate the scripts |

---

## Prerequisites

- `rocgdb` in `PATH` (provided by ROCm; override path with `ROCGDB` env var)
- `timeout` command (standard on Linux)
- Python support in rocgdb (built-in for ROCm 6+)

For meaningful backtraces the application and rocSHMEM library should be built
with debug symbols.  A `RelWithDebInfo` build (CMake `-DCMAKE_BUILD_TYPE=RelWithDebInfo`)
gives full inlined call chains while keeping optimized performance.  A `Release`
build will still show the outermost non-inlined API frame but not the internal
wait loops.

---

## Quick Start

```bash
# 1. Start your (deadlocked) application
mpirun -np 4 ./my_rocshmem_app &

# 2. Attach to all instances, analyze, save output
./scripts/debug/attach_deadlock_analysis.sh my_rocshmem_app

# Output is written to ./rocshmem_deadlock_analysis_<timestamp>/
# A compact summary is printed to stdout.
```

---

## `attach_deadlock_analysis.sh`

### Usage

```
attach_deadlock_analysis.sh <executable_name> [--directory <dir>] [--cull[=p1,p2,...]]
```

| Argument | Description |
|---|---|
| `executable_name` | Exact process name to search for |
| `--directory <dir>` | Directory for per-process output files (default: `./rocshmem_deadlock_analysis_<timestamp>/`) |
| `--cull` | Cull groups stuck in GPU barriers / gridsync using the built-in default pattern list |
| `--cull=p1,p2,...` | Cull groups whose backtrace contains any of the comma-separated substrings |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ROCGDB` | `rocgdb` | Path to the rocgdb binary |
| `ROCSHMEM_GDB_TIMEOUT` | `120` | Per-process timeout in seconds |
| `ROCSHMEM_DEADLOCK_COLOR` | `always` | Color mode for output files (see [Color Support](#color-support)) |

### Output Files

Each analyzed process produces a file named:

```
<output_dir>/pe<PE_RANK>_pid_<PID>.txt
```

The MPI PE rank is read from the process's `OMPI_COMM_WORLD_RANK` environment
variable.  If the variable is not set (non-MPI process or already exited), the
rank is shown as `unknown`.

### Stdout Summary

After all processes are analyzed, a compact summary is printed:

```
=== Attach and Analyze Summary ===
Processes analyzed: 2 / 2

  PE 0     PID 12345   groups=?    (no GPU threads found)
  PE 1     PID 12346   groups=1    Waiting for remote PE memory update...

Full output in: ./rocshmem_deadlock_analysis_20260416_143022/
```

- **groups** — number of distinct backtrace groups found in that process
- **dominant** — the most frequent `[HINT]` line across all groups, or a
  status string if no deadlock was detected

### Error Handling

| Situation | Behavior |
|---|---|
| Process exits before attach | Logged as SKIP, counted as failure |
| `rocgdb` exceeds timeout | Logged as TIMEOUT; output file kept for inspection |
| `rocgdb` exits non-zero but produced analysis | Still counted as success |
| `rocgdb` not found in `PATH` | Fatal error with hint to set `ROCGDB` |

---

## `rocgdb_deadlock_analysis.py`

The Python script can be used in three modes.

### Mode 1: Batch attach (used by `attach_deadlock_analysis.sh`)

Attach `rocgdb` to a running process and analyze immediately:

```bash
ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1 \
    rocgdb -batch -p <pid> -x scripts/debug/rocgdb_deadlock_analysis.py
```

The `ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1` variable tells the script to run the
analysis as soon as the process is stopped (which happens automatically on
attach).

### Mode 2: Interactive session

Attach `rocgdb` manually, then source and run the script:

```
rocgdb -p <pid>
(rocgdb) source scripts/debug/rocgdb_deadlock_analysis.py
(rocgdb) rocshmem-deadlock-analyze
```

Write the report to a file:

```
(rocgdb) rocshmem-deadlock-analyze /tmp/analysis.txt
```

Force colored output in the file:

```
(rocgdb) rocshmem-deadlock-analyze --color /tmp/analysis.txt
```

### Mode 3: Launch mode

Run the application under rocgdb from the start; analysis triggers on first
stop (e.g. `pkill --signal SIGINT my_rocshmem_app`), or when the program crashes:
```bash
mpirun -n 2 -x ROCSHMEM_DEADLOCK_AUTO_ANALYZE=1 rocgdb -batch \
    -x scripts/debug/rocgdb_deadlock_analysis.py \
    -ex run \
    --args ./my_rocshmem_app arg1 arg2

### GDB Command Reference

After sourcing the script, the `rocshmem-deadlock-analyze` command is available:

```
(rocgdb) rocshmem-deadlock-analyze [--color|--no-color] [--cull[=pat,...]] [--check-lanes] [output_file]
```

| Option | Description |
|---|---|
| `--color` | Force ANSI color codes on, even when writing to a file |
| `--no-color` | Disable color codes, even on a TTY |
| `--cull` | Cull groups stuck in GPU barriers / gridsync using the built-in default pattern list |
| `--cull=p1,p2,...` | Cull groups whose backtrace contains any of the comma-separated substrings |
| `--check-lanes` | Enable lane-level parameter mismatch detection for `_wg` and `_wave` collective calls (slow; see below) |
| `output_file` | Write report to this file instead of stdout |

#### Cull option

When `--cull` is given without a pattern list, the following substrings are used to remove groups
that are simply waiting at a non-rocSHMEM GPU synchronization primitive (and are therefore not
rocSHMEM deadlocks):

| Pattern | Covers |
|---|---|
| `__syncthreads` | HIP workgroup barrier and reduction variants (`_count`, `_and`, `_or`) |
| `__builtin_amdgcn_s_barrier` | Direct AMDGCN scalar barrier intrinsic |
| `__work_group_barrier` | Workgroup barrier with memory-fence flags |
| `__named_sync` | Named barrier intrinsic |
| `__ockl_grid_sync` | OCKL grid-level synchronization |
| `__ockl_gws_barrier` | OCKL global-workspace barrier |
| `__ockl_multi_grid_sync` | OCKL multi-grid synchronization |
| `cooperative_groups` | HIP cooperative groups (`grid_group`, `thread_block`, `multi_grid_group`) |

Culled groups are not shown in the report; the header prints a count of how many
groups and wavefronts were removed:

```
Unique backtrace groups: 1
  (+ 3 group(s) / 192 wavefront(s) culled — GPU barrier / gridsync)
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ROCSHMEM_DEADLOCK_AUTO_ANALYZE` | `0` | Set to `1` to trigger analysis automatically on process stop |
| `ROCSHMEM_DEADLOCK_COLOR` | `always` | Color mode (see [Color Support](#color-support)) |

---

## Output Format

```
=== rocSHMEM Deadlock Analysis ===
Process(es): 12346
Total GPU wavefronts analyzed: 64
Unique backtrace groups: 2

--- Group 1 (63 wavefront(s)) ---
  WFs: (0,0,0)/0,(1,0,0)/0,(2,0,0)/0,...
  Backtrace:
    #0  rocshmem::uncached_load<long volatile> (...) at src/assembly.hpp:125
    #1  rocshmem::Context::test<long> (...) at src/context_tmpl_device.hpp:432
    #2  rocshmem::Context::wait_until<long> (...) at src/context_tmpl_device.hpp:224
    #3  rocshmem::IPCContext::internal_direct_barrier (...) at ...
    ...
    #7  rocshmem::Context::barrier_all_wg (...) at src/context_device.cpp:151
    #8  my_kernel () at my_app.cpp:42
  [HINT] Waiting for remote PE memory update (barrier/sync or user
         rocshmem_wait_until). Check if the remote PE is alive and making
         progress.

--- Group 2 (1 wavefront(s)) ---
  WGs: (0,0,0)  WFs: 0
  Backtrace:
    #0  rocshmem::acquire_lock (...) at src/gda/mlx5/queue_pair_mlx5.cpp:53
    #1  rocshmem::QueuePair::mlx5_post_wqe_rma (...) at ...
    #2  rocshmem::GDAContext::put (...) at ...
    #3  rocshmem::rocshmem_ctx_put (...) at ...    <<< rocSHMEM API entry
    #4  my_kernel () at my_app.cpp:38
  [rocSHMEM] Stuck in: rocshmem_ctx_put
  [HINT] Waiting for SQ spinlock held by another wavefront. The lock holder
         may itself be deadlocked in mlx5_poll_cq_until.

=== Summary ===
  1 wavefront(s) inside rocSHMEM
  63 wavefront(s) outside rocSHMEM
```

### Reading the Output

**Groups** — Wavefronts with identical backtraces (after stripping runtime-variable
parts like addresses and argument values) are coalesced into a single group.
Each group shows which workgroups (WGs) and wavefronts (WFs) share that call
stack.  Template parameters and source file locations are kept in the key, so
`wait_until<int>` and `wait_until<long>` at different call sites produce
separate groups.

**`<<< rocSHMEM API entry`** — Marks the outermost rocSHMEM public API function
in the backtrace: the call the application made that led to the deadlock.

**`[rocSHMEM] Stuck in:`** — Names the matched API function for easy grepping.

**`[HINT]`** — A short diagnosis of the likely deadlock cause based on the
innermost recognized frame.

The innermost deadlock frame (the actual spinning loop) is printed in bold cyan.

**`=== Collective API Usage Issues ===`** — Appears after the per-group section
when incorrect `_wg` or `_wave` API usage is detected from the backtraces.
Three kinds of violations are reported:

| Issue | Description |
|---|---|
| `[BAD] … non-collective _wg call` | Wavefronts in the same workgroup are at different `_wg` API calls, or some are not in any `_wg` call. All wavefronts in a WG must call the same `_wg` primitive together. |
| `[BAD] … parameter mismatch in rocshmem_*_wg` | All wavefronts in a WG are at the same `_wg` call but with different argument values (e.g., different `pe` or `nelems`). |

`_wave` primitives are collective only within a single wavefront (all 64 lanes
must call with matching parameters). Different wavefronts — even in the same WG —
may independently call any `_wave` function with different parameters; that is valid.
Lane-level divergence within a wavefront is not visible from wavefront-level
backtraces and requires `--check-lanes` to detect.

#### Lane-level parameter mismatch (`--check-lanes`)

When `--check-lanes` is added to the `rocshmem-deadlock-analyze` command, the
script iterates over the active lanes of every wavefront that is inside a `_wg`
or `_wave` collective call.  It switches rocgdb to each active lane, collects
the per-lane backtrace, and extracts the argument values for the collective
API frame.

If any two active lanes in the same wavefront disagree on a parameter, a
`LANE_PARAM_MISMATCH` issue is reported:

```
=== Collective API Usage Issues ===
[BAD] WG (0,0,0) WF 0 — lane parameter mismatch in rocshmem_putmem_wave (_wave contract)
  Differing parameter(s): pe
    Lane 0 [0,0,0]: pe=0
    Lane 1 [1,0,0]: pe=1
    Lane 2 [2,0,0]: pe=0
    ...
```

This check covers both `_wave` (lanes diverging within one wavefront) and `_wg`
(lanes in a wavefront that should present unified parameters to the WG-wide
collective).

> **Performance note:** `--check-lanes` issues up to 64 additional `bt` calls per
> wavefront in a collective call.  On a busy kernel with many wavefronts this can
> be slow (several minutes).  Use it only when wavefront-level analysis has
> already identified a collective call of interest.

### Recognized Deadlock Patterns and Hints

#### mlx5 (Mellanox/ConnectX) backend

| Innermost frame | Hint |
|---|---|
| `mlx5_poll_cq_until` | Waiting for NIC completion (CQ polling). Check if NIC is responsive and if the remote PE is also stuck. |
| `acquire_lock` | Waiting for SQ spinlock held by another wavefront. The lock holder may itself be deadlocked in `mlx5_poll_cq_until`. |
| `mlx5_quiet` | Quiet operation waiting for all outstanding RMA ops to complete. Check NIC health. |

#### bnxt (Broadcom) backend

| Innermost frame | Hint |
|---|---|
| `bnxt_poll_cq_until` | Waiting for NIC completion (CQ polling). Check if the bnxt NIC is responsive and if the remote PE is also stuck. |
| `bnxt_post_wqe_rma` | Waiting for bnxt SQ spinlock held by another wavefront. The lock holder is likely itself deadlocked in `bnxt_poll_cq_until`. |
| `bnxt_quiet` | Quiet operation waiting for all outstanding RMA ops to complete (bnxt). Check NIC health. |

#### ionic (AMD/Pensando) backend

| Innermost frame | Hint |
|---|---|
| `ionic_quiet_internal_ccqe_single` | Waiting for NIC completion in CCQE mode (single-thread path). Check if the ionic NIC is responsive and if the remote PE is also stuck. |
| `ionic_quiet_internal_ccqe` | Waiting for NIC completion in CCQE mode. Check if the ionic NIC is responsive and if the remote PE is also stuck. |
| `ionic_quiet_internal` | Waiting for NIC completion (CQ polling). Check if the ionic NIC is responsive and if the remote PE is also stuck. |
| `spin_lock_acquire_unique` | Waiting for ionic SQ doorbell spinlock (exclusive) held by another wavefront. The lock holder may itself be stuck in `ionic_quiet_internal`. |
| `spin_lock_acquire_shared` | Waiting for ionic CQ spinlock (shared) held by another wavefront. The lock holder may itself be stuck in `ionic_quiet_internal`. |
| `ionic_quiet` | Quiet operation waiting for all outstanding RMA ops to complete (ionic). Check NIC health. |

#### Shared / IPC / user-level waits (all backends)

| Innermost frame | Hint |
|---|---|
| `wait_until_any` | Waiting for any element of a multi-element condition. Check if the remote PE is alive. |
| `wait_until_all` | Waiting for all elements of a multi-element condition. Check if the remote PE is alive. |
| `wait_until_some` | Waiting for some elements of a multi-element condition. Check if the remote PE is alive. |
| `wait_until` | Waiting for remote PE memory update (barrier/sync or user wait). Check if the remote PE is alive. |

---

## Color Support

Color is controlled by the `ROCSHMEM_DEADLOCK_COLOR` environment variable:

| Value | Behavior |
|---|---|
| `always` (default) | Always enable ANSI color codes |
| `auto` | Enable color when the output is a TTY; disable for files |
| `1`, `yes`, `true` | Always enable ANSI color codes |
| `0`, `no`, `false`, `never` | Always disable color codes |

Color can also be forced per-invocation with the `--color` / `--no-color`
flags of the interactive `rocshmem-deadlock-analyze` command.

The shell script's stdout summary is always plain text; ANSI codes are
stripped from the per-process output files before extracting the dominant hint.

### Color Scheme

| Element | Color |
|---|---|
| Section headers (`=== ... ===`) | Bold blue |
| Group headers (`--- Group N ---`) | Bold bright blue |
| Innermost deadlock frame | Bold cyan |
| `<<< rocSHMEM API entry` annotation | Bold green |
| `[rocSHMEM] Stuck in:` line | Bold bright red |
| `[HINT]` line | Bold cyan |
| Stuck-wavefront count in summary | Bold bright red |
| Not-stuck count in summary | Bold green |
| "No GPU threads found" notice | Red |

---

## Building with Debug Symbols

Full call-chain visibility requires debug symbols in the rocSHMEM library and
the application.  Use `RelWithDebInfo` for the best balance of performance and
debuggability:

```bash
# Configure the rocSHMEM build tree
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo <source_dir>
cmake --build . --target rocshmem --parallel 8

# Compile the application with matching flags
hipcc -O2 -g -fgpu-rdc -x hip my_app.cc \
      --offload-arch=<target> \
      -I<rocshmem_include> -I<mpi_include> \
      -c -o my_app.o

hipcc -fgpu-rdc --hip-link my_app.o \
      --offload-arch=<target> \
      <rocshmem_lib>/librocshmem.a <mpi_lib>/libmpi.so \
      -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 \
      -o my_app
```

Without debug symbols only the outermost non-inlined frame is visible.  The
`<<< rocSHMEM API entry` annotation still works (those functions are marked
`__attribute__((noinline))`), but the `[HINT]` requires the inner frames
(`wait_until`, `mlx5_poll_cq_until`, etc.) to be present.

---

## Known Limitations

- **IPC backend**: `rocshmem_barrier_all_wg` and related functions are fully
  inlined in the IPC backend even at `RelWithDebInfo`.  The `[rocSHMEM] Stuck in:`
  line will not appear, but the `[HINT]` fires from the inner `wait_until` frame.

- **Release builds**: Without debug symbols, internal frames (`wait_until`,
  `mlx5_poll_cq_until`, etc.) are invisible and no `[HINT]` is generated.
  The API entry frame is still detected when it is non-inlined.

- **Lane-level analysis performance**: `--check-lanes` issues up to 64 `bt` calls
  per wavefront in a collective call.  On large kernels this can take several minutes.

- **Single GPU per process**: The script handles one GPU device per MPI rank.
  Multi-GPU-per-rank configurations are untested.

- **GPU thread format**: Tested with rocgdb from ROCm 7.2.  Earlier ROCm versions
  use a different thread name format (`AMDGPU Thread X.Y (GPU, WG (...), WF (N))`);
  the script includes a regex for this legacy format but it is untested.
