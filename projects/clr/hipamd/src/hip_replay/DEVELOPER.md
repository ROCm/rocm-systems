# HIP Record & Replay (HRR) — Developer Guide

HRR captures HIP GPU workloads (kernel launches, memory operations, buffer
contents) and replays them deterministically. Use cases: bug reproduction,
performance regression testing, kernel-level benchmarking, and portable
cross-machine workload packaging.

## Branch

```
ROCm/rocm-systems  users/powderluv/hip-replay
```

## Directory Layout

```
hipamd/src/
  hip_hrr.h / hip_hrr.cpp          In-tree recording hooks (CLR runtime)
  hip_replay/
    hrr_reader.h / hrr_reader.cpp  Trace archive reader
    hrr_replay.cpp                 hrr-replay tool
    hrr_bench.cpp                  hrr-bench tool
    hrr_info.cpp                   hrr-info tool
    CMakeLists.txt
    out_of_tree/                   Standalone recording for pre-built ROCm
      hrr_trace_writer.c/h         Trace writer (no HIP dependency)
      hrr_code_object.c/h          ELF/msgpack kernel arg parser
      hrr_interposer_linux.c       LD_PRELOAD interposer
      hrr_interposer_cxx.cpp       C++ extension (hipExtModuleLaunchKernel)
      hrr_proxy_win.c              Windows proxy DLL
      record_model.py              MIGraphX ONNX two-phase record script
      test_migraphx_replay.py      CTest integration test
      bench_models.sh              Benchmark script
      CMakeLists.txt
```

---

## Path A: In-Tree Recording (CLR rebuild required)

Recording hooks built into the HIP runtime. Triggered by `HIP_RECORD=1`.
Gives full access to CLR internals: kernel arg metadata, code objects,
buffer sizes. Use this when you can rebuild TheRock.

### Build

```bash
git clone https://github.com/ROCm/TheRock
cd TheRock
git submodule update --init rocm-systems
cd rocm-systems
git fetch origin users/powderluv/hip-replay
git checkout users/powderluv/hip-replay

cd ../..
cmake -B therock-build -S . -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942   # adjust for your GPU
ninja -C therock-build clr+dist
```

The tools (`hrr-replay`, `hrr-bench`, `hrr-info`) are built as part of CLR:

```bash
ninja -C therock-build clr+dist
# tools land in therock-build/dist/...
```

### Record

```bash
HIP_RECORD=1 HIP_RECORD_OUTPUT=./capture.hrr ./my_hip_app
```

### Environment Variables (In-Tree)

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_RECORD=1` | off | Enable recording |
| `HIP_RECORD_OUTPUT=<path>` | `./capture.hrr` | Output directory |
| `HIP_RECORD_MODE=inputs\|full\|timeline` | `inputs` | Recording depth |
| `HIP_RECORD_KERNEL_FILTER=<glob>` | all | Record only matching kernels |
| `HIP_RECORD_MAX_BLOB_MB=<N>` | unlimited | Skip buffers above threshold |
| `HIP_RECORD_COMPRESS=1` | off | zstd-compress blobs |

**Modes:**
- `timeline` — API call sequence only, no buffer data (minimal overhead)
- `inputs` — snapshot input buffers before each kernel launch (default)
- `full` — snapshot inputs + outputs (syncs after every kernel, slowest)

---

## Path B: Out-of-Tree Recording (pre-built ROCm)

`libhrr_record.so` is an LD_PRELOAD interposer that works with any ROCm
installation without rebuilding the runtime.

### Build

```bash
cd rocm-systems/projects/clr/hipamd/src/hip_replay/out_of_tree
cmake -B build -S .
cmake --build build -j$(nproc)
# Produces: build/libhrr_record.so
```

If HIP is available and the parent `hrr_reader.cpp` etc. are present, the
tools (`hrr-bench`, `hrr-replay`, `hrr-info`) are also built. If deploying
the `out_of_tree/` directory standalone (without the parent sources), only
`libhrr_record.so` is built; pass `HRR_BENCH_PATH` at cmake time to point
the CTest integration at a pre-built `hrr-bench`.

### Record

```bash
HRR_RECORD=1 HRR_OUTPUT=./capture.hrr \
  LD_PRELOAD=/path/to/libhrr_record.so \
  ./my_hip_app
```

### Environment Variables (Out-of-Tree)

| Variable | Default | Description |
|----------|---------|-------------|
| `HRR_RECORD=1` | off | Enable recording |
| `HRR_OUTPUT=<path>` | `./capture.hrr` | Output directory |
| `HRR_MODE=inputs\|full\|timeline` | `inputs` | Recording depth |
| `HRR_KERNEL_FILTER=<glob>` | all | Record only matching kernels |
| `HRR_MAX_BLOB_MB=<N>` | unlimited | Skip buffers above threshold |
| `HRR_VERBOSE=1` | off | Print diagnostic messages to stderr |

---

## Trace Archive Format

```
capture.hrr/
  manifest.json        Device info, ROCm version, capture config
  events.bin           Binary event stream (32-byte fixed headers)
  blobs/               Content-addressed buffer store (XXH3-128)
    ab/ab1234...blob
  code_objects/        Captured .hsaco ELFs
    <xxh3_hash>.hsaco
```

---

## Tools

### hrr-info — Inspect a trace

```bash
hrr-info capture.hrr
# Events: 47   Kernels: 12   Blobs: 8   Code objects: 2
```

### hrr-replay — Replay a trace

```bash
hrr-replay capture.hrr
hrr-replay capture.hrr --verify   # compare outputs vs recorded snapshots
```

### hrr-bench — Benchmark and reproduce

All subcommands share `--global-work-size` (see Legacy Traces below).

#### list — Show all kernels in a capture

```bash
hrr-bench list capture.hrr
#  ID  Kernel                    Grid        Block      Calls
#   1  _ZN4gemm...               [256,1,1]   [256,1,1]    47
#   2  _ZN8softmax...            [128,1,1]   [128,1,1]    47
```

#### diagnose — Detect common replay issues

```bash
hrr-bench diagnose capture.hrr
```

Checks for untracked pointers, code object mismatches, handle reuse.

#### kernel — Benchmark a single kernel

```bash
hrr-bench kernel capture.hrr --id 1 --iterations 1000 --warmup 50
# Min: 1.801ms  Median: 1.843ms  P95: 1.892ms  P99: 1.923ms  Max: 2.104ms
```

#### app — Replay full trace with timing

```bash
hrr-bench app capture.hrr --iterations 5 --warmup 2
# Run 1: 4.823ms   Run 2: 4.819ms   ...
# Mean: 4.820ms  Median: 0.506ms
```

#### repro — Reproduce crashes with diagnostics

```bash
hrr-bench repro capture.hrr
hrr-bench repro capture.hrr --check-nan   # also scan outputs for NaN/Inf
```

#### export — Export kernel as standalone repro

```bash
hrr-bench export capture.hrr --id 1 --output gemm_repro/
# Creates: gemm_repro/repro.hip, CMakeLists.txt, kernel.hsaco, input_*.bin

hrr-bench export capture.hrr --id 1 --output gemm_repro/ --safe
# Same, but buffer contents randomized (safe for external bug reports)
```

Build the exported repro:
```bash
cmake -B gemm_repro/build -S gemm_repro && cmake --build gemm_repro/build
./gemm_repro/build/repro
```

#### stress — Hunt for intermittent failures

```bash
hrr-bench stress capture.hrr --id 1 --iterations 10000
```

---

## MIGraphX / ONNX Integration Test

Records and replays three light ONNX models via MIGraphX. Verifies no GPU
fault occurs and median kernel time is in a sane range.

### Run manually

```bash
cd out_of_tree
python3 test_migraphx_replay.py \
  --hrr-bench  /path/to/hrr-bench \
  --libhrr     /path/to/libhrr_record.so \
  --record-script record_model.py \
  --iterations 20 --warmup 5
```

Expected output:
```
HRR MIGraphX replay test — 3 models
  PASS: light_resnet50    median=0.506ms  kernels=51
  PASS: light_squeezenet  median=0.162ms  kernels=28
  PASS: light_densenet121 median=1.275ms  kernels=183
PASSED: all 3 models
```

The test skips cleanly (exit 0) when migraphx, onnx, or the tools are not
available. It only fails on GPU faults or unexpected hrr-bench errors.

### Run via CTest

```bash
# In-tree build (tools built alongside libhrr_record.so):
cmake -B build -S out_of_tree
cmake --build build -j$(nproc)
ctest --test-dir build -L migraphx -V

# Standalone deploy (tools built separately):
cmake -B build -S out_of_tree \
  -DHRR_BENCH_PATH=/path/to/hrr-bench
cmake --build build -j$(nproc)
ctest --test-dir build -L migraphx -V
```

### MIGraphX model benchmark (bench_models.sh)

Compares MIGraphX native end-to-end latency vs HRR pure kernel replay:

```bash
cd out_of_tree
HRR_DIR=~/hrr_test HRR_ITERS=100 MGX_ITERS=50 ./bench_models.sh
# Model                MGX       HRR_min   HRR_med   HRR_p95   HRR_mean  kernels
# resnet50         1.842ms    0.483ms    0.506ms    0.521ms    0.507ms      51
# squeezenet       0.724ms    0.155ms    0.162ms    0.169ms    0.163ms      28
# densenet121      5.913ms    1.244ms    1.275ms    1.290ms    1.276ms     183
```

MGX includes Python/framework overhead. HRR measures pure GPU kernel time.

---

## Two-Phase MIGraphX Recording

MIGraphX compiles ONNX models to GPU code on first run. To keep compilation
kernels out of the replay trace, compile and record in separate phases:

```bash
# Phase 1: compile to .mxr (no HRR)
python3 record_model.py --compile model.onnx

# Phase 2: record one clean inference from the compiled model
HRR_RECORD=1 HRR_OUTPUT=model.hrr \
  LD_PRELOAD=libhrr_record.so \
  python3 record_model.py --record model.onnx
```

---

## Legacy Traces (pre-fix interposer)

Traces recorded with the original out-of-tree interposer stored
`hipExtModuleLaunchKernel`'s `globalWorkSize` as the grid dimension instead
of `numBlocks`. Replaying these with `hrr-bench` requires:

```bash
hrr-bench app old_capture.hrr --global-work-size --iterations 10
```

New traces (recorded with the fixed interposer on this branch) do not need
this flag.
