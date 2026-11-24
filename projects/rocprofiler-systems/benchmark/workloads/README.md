# Synthetic test programs

Small C++ programs for developing and validating the benchmark tool. They are **not** the real use case: `bench` measures any command. Each program stresses one property of the profiler, runs for roughly one second at its defaults and is deterministic.

| Program | Stresses | `--size` | `--iters` |
| --- | --- | --- | --- |
| `noop` | Fixed cost of process start and shutdown (control) | ignored | ignored |
| `cpu_compute` | Pure CPU: matrix multiplication | matrix dimension N | number of N×N multiplications |
| `call_heavy` | Many short function calls (cost per call) | call depth per chain | number of chains (total calls = size × iters) |
| `syscall_heavy` | Many short system calls (`pread` on `/dev/zero`) | ignored | number of syscalls |
| `mem_alloc` | Allocate and touch a buffer, random read-modify-write access | buffer size in MB | number of accesses |
| `io_files` | Writing and reading files in a temp dir (removed on exit) | file size in KB | number of files |
| `threads` | Threads with and without locking (every 4th iteration takes a shared mutex) | thread count | iterations per thread |
| `gpu_kernels` | HIP: many short kernels (tens of µs each) on one stream; built only if HIP is found | float elements per launch | number of launches |
| `ai_hpc_cpu` | Realistic AI/HPC compute on CPU: worker threads push activations through a tanh MLP layer (N×N GEMM, 2·N³ flops) reached through `depth` nested calls. Stresses stack unwinding (depth), thread handling (threads) and compute load (size) independently | matrix dimension N | layers per thread |
| `ai_hpc_gpu` | Same workload on the GPU: one stream per thread, tiled HIP GEMM + tanh, stream sync at the leaf of the deep call chain; built only if HIP is found | matrix dimension N | layers per thread |
| `gpu_long_kernel` | HIP: few long kernels (tens of ms each, compute-bound FMA chain); same metrics as `gpu_kernels`; built only if HIP is found | float elements per launch | number of launches |

`gpu_kernels` and `gpu_long_kernel` also print `metric <name> <value>` lines (before the checksum line), measured with HIP events and the host clock:

- `gpu_interval_ms`: event pair around the whole launch loop.
- `gpu_kernel_us`: average kernel time (event pair around each of the first min(iters, 2048) kernels).
- `gpu_gap_us`: average gap between a kernel's end and the next kernel's start (same kernels).
- `gpu_launch_us`: average host duration of the launch call, over all launches.

A warm-up launch runs before `BENCH_HOOK_WORK_BEGIN`. Use `--size 67108864 --iters 40` for few large kernels.

`ai_hpc_cpu` and `ai_hpc_gpu` take two more options, `--depth` (nested calls above each layer, default 64; real stack frames, verified with gdb) and `--threads` (worker threads, default 4). Total work = threads × iters × 2·N³ flops. Each thread uses its own seed (`seed + thread index`) and the checksum is combined in thread order, so it does not depend on scheduling. Keep depth below ~100000 (default 8 MB thread stack).

## Contract

- Same CLI: `--size`, `--iters`, `--seed`; the same arguments do identical work.
- Prints `checksum: <hex>` as the last stdout line; a different `--seed` gives a different checksum.
- Exit code 0 on success, 2 on a bad argument.
- Calls `bench_hook(BENCH_HOOK_WORK_BEGIN)` right before the work and `bench_hook(BENCH_HOOK_WORK_END)` right after (`include/bench_hooks.h`).
- No network access and no input that changes between runs.

## Build and test

```bash
cmake -S workloads -B workloads/build
cmake --build workloads/build -j
ctest --test-dir workloads/build
```

## Run with the benchmark tool

```bash
W=$PWD/workloads/build
.venv/bin/python -m bench --mode light --cmd A="$W/cpu_compute" \
    --cmd R="rocprof-sys-run -o /tmp/out -- $W/cpu_compute"
```

The GPU programs add 1 to every element per launch, so the checksum depends on every launch (a lost launch changes it). `gpu_kernels` and `gpu_long_kernel` share one implementation (`src/gpu_workload.hpp`); they differ only in the defaults and in the length of the FMA chain per element.
