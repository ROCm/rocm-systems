# rocprofiler-systems Examples

This directory contains example applications demonstrating various profiling scenarios with rocprofiler-systems. Each example targets a specific workload type - GPU compute, CPU threading, distributed computing, or library integration - and shows how to capture performance data using `rocprof-sys-run`.

## Examples by Category

### GPU Compute

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [transpose](transpose/) | Tiled matrix transpose on GPU with multi-threaded stream execution | HIP |
| [scratch-memory](scratch-memory/) | GPU scratch memory allocation stress test across primary and overflow slots | HIP, HSA |
| [sdma_test](sdma_test/) | SDMA engine bandwidth benchmark for H2D, D2D, and D2H transfers | HIP |
| [transferBench](transferBench/) | All-to-all transfer benchmark across CPU, GPU, SDMA, and NIC executors | HIP, HSA |

### Profiler API

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [user-api](user-api/) | User API for named regions, annotations, and selective thread tracing | rocprofiler-systems user library |
| [roctx](roctx/) | ROCTx range/marker API with thread naming, pause/resume, and device labeling | rocprofiler-sdk-roctx, HIP |
| [causal](causal/) | Causal profiling with slow/fast parallel workloads and progress point tracking | rocprofiler-systems user library |
| [rewrite-caller](rewrite-caller/) | Minimal call chain for binary rewrite instrumentation testing | None |
| [trace-time-window](trace-time-window/) | Mixed CPU-bound and sleep workload for time-windowed trace analysis | None |

### CPU Threading

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [thread-limit](thread-limit/) | Thread scaling stress test with batched Fibonacci workers | pthreads |
| [parallel-overhead](parallel-overhead/) | Mutex vs. atomic synchronization overhead comparison | pthreads |
| [code-coverage](code-coverage/) | Dual code-path execution for coverage analysis testing | pthreads |
| [fork](fork/) | Multi-process forking from worker threads with child process profiling | pthreads |

### Distributed Computing

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [mpi](mpi/) | MPI collective and point-to-point operations with communicator patterns | MPI |
| [rccl](rccl/) | RCCL collective communication performance tests across GPUs | HIP, RCCL |
| [shmem](shmem/) | OpenSHMEM hello world and ping-pong latency benchmark | OpenSHMEM (oshcc) |

### OpenMP

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [openmp](openmp/) | NAS Parallel Benchmarks (CG, LU) with OpenMP threading | OpenMP |

### GPU Libraries

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [jpegdecode](jpegdecode/) | Batch JPEG decoding performance benchmark using rocJPEG | HIP, rocJPEG |
| [videodecode](videodecode/) | Batch video decoding benchmark using ROCDecode with VCN hardware | HIP, ROCDecode, FFmpeg |

### HPC

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [lulesh](lulesh/) | LULESH shock hydrodynamics mini-app with Kokkos parallelism | Kokkos, optional MPI |
| [hpc](hpc/) | Six HPC training examples covering Jacobi solvers, matrix exponentials, and stream overlap | HIP, rocBLAS, Fortran (varies) |

### Python

| Example | Description | Dependencies |
|---------|-------------|--------------|
| [python](python/) | Python profiling with decorators, user regions, and selective tracing | Python 3, optional NumPy |

## Building All Examples

- The examples are built as part of the `rocprofiler-systems` CMake project.
- There is an option to build them also as a **standalone** applications or as a part of **examples suite**
- The following commands will focus on a building a whole **examples suite**:

- From `examples` directory run:

```bash
cmake -B <build_dir> \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_INSTALL_PREFIX=./install \
    .

cmake --build <build_dir> --parallel
```

- Or from the repository root:

```bash
cmake -B <build_dir> \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    projects/rocprofiler-systems/examples

cmake --build <build_dir> --parallel
```

- Individual examples can be built by specifying the target:

```bash
cmake --build <build_dir> --target <example_name>
```

GPU examples require ROCm (`hipcc` or `amdclang++`) and detect available architectures automatically. To specify architectures manually:

```bash
cmake -B <build_dir> -DROCPROFSYS_GFX_TARGETS="gfx90a;gfx942" ...
```

## Profiling Modes

rocprofiler-systems supports several instrumentation modes:

| Mode | Command | Description |
|------|---------|-------------|
| System-level | `rocprof-sys-run -- ./app` | Lightweight tracing via `LD_PRELOAD`, no binary modification |
| Binary rewrite | `rocprof-sys-instrument -o app.inst -- ./app` then `rocprof-sys-run -- ./app.inst` | Statically rewrite the binary for repeated profiling |
| Runtime instrument | `rocprof-sys-instrument -- ./app` | Dynamically instrument at launch without modifying the binary |
| Sampling | `rocprof-sys-sample -- ./app` | Statistical sampling of call stacks at configurable frequency |
| Causal | `rocprof-sys-causal -- ./app` | Causal profiling to identify optimization opportunities |

### Common Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCPROFSYS_TRACE` | Enable Perfetto trace output | `true` |
| `ROCPROFSYS_PROFILE` | Enable call-stack profile output | `true` |
| `ROCPROFSYS_USE_ROCPD` | Generate `rocpd` database output | `false` |
| `ROCPROFSYS_USE_SAMPLING` | Enable statistical sampling | `false` |
| `ROCPROFSYS_SAMPLING_FREQ` | Sampling frequency (interrupts/sec) | `50` |
| `ROCPROFSYS_USE_PROCESS_SAMPLING` | Enable process-level resource sampling | `true` |
| `ROCPROFSYS_OUTPUT_PATH` | Base directory for output files | `rocprofsys-output` |
| `ROCPROFSYS_TIME_OUTPUT` | Timestamp output subdirectories | `true` |
| `ROCPROFSYS_ROCM_DOMAINS` | ROCm API domains to trace | all |
| `ROCPROFSYS_USE_MPIP` | Enable MPI profiling interposition | `false` |
