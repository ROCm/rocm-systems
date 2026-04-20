# LLM Reference Guide for GPU Performance Analysis

**Purpose**: This document is provided to the LLM as context when analyzing GPU profiling data. It defines boundaries, provides reference information, and guides analysis quality.

---

## CRITICAL REQUIREMENTS
<!-- rocinsight-context: always -->

### Hardware Counter Per-Block Limits — MUST NOT EXCEED

**THIS IS A HARD HARDWARE CONSTRAINT.** Violating it crashes rocprofv3 (error code 38: "Request exceeds the capabilities of the hardware to collect").

AMD GPUs limit how many counters from the **same hardware block** can be collected in one rocprofv3 pass. The block name is the prefix before the first `_` in the counter name (e.g., `SQ_WAVES` → block `SQ`).

**Safe per-block limits** (conservative defaults — actual limits vary by GPU):
| Block | Examples | Limit per pass |
|-------|----------|----------------|
| `SQ`  | `SQ_WAVES`, `SQ_INSTS_VALU`, `SQ_INSTS_VMEM_RD`, `SQ_INSTS_VMEM_WR`, `SQ_INSTS_LDS` | 4 (up to 8 on gfx942) |
| `GRBM` | `GRBM_COUNT`, `GRBM_GUI_ACTIVE` | 4 |
| `FETCH` | `FETCH_SIZE` | 2 |
| `WRITE` | `WRITE_SIZE` | 2 |
| `TCP`, `TCC`, `TA`, `TD` | Cache counters | 4 |

**Mandatory rules for `--pmc` commands you generate:**
1. Count counters **per block separately** — do NOT count across different blocks together
2. If any block would exceed its limit → split into **multiple separate rocprofv3 runs** (pass 1, pass 2, …) each with its own `-d`/`-o`
3. Different blocks CAN coexist in the same pass as long as each block's count stays within its limit
4. `rocprof-compute` is EXEMPT — it handles multi-pass collection internally

**ADDITIONAL RULE — FETCH_SIZE and WRITE_SIZE are TCC-derived metrics**:
These are NOT raw hardware counters. rocprofv3 expands them internally to TCC hardware counters:
- `FETCH_SIZE` → `TCC_BUBBLE + TCC_EA0_RDREQ + GRBM_GUI_ACTIVE` (TCC block, 32 instances)
- `WRITE_SIZE` → `TCC_EA0_WRREQ + TCC_EA0_WRREQ_64B` (TCC block, 32 instances)
**Rules**:
1. FETCH_SIZE and WRITE_SIZE MUST each be in their own dedicated pass.
2. They cannot share a pass with each other (combined 5 TCC hardware counters > limit).
3. They cannot share a pass with SQ counters.

**Examples:**
```bash
# ✅ SAFE — 3 passes: SQ/GRBM | FETCH_SIZE | WRITE_SIZE
# Pass 1: GPU utilization + occupancy (raw hardware counters)
rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VMEM_RD \
  SQ_INSTS_VMEM_WR SQ_INSTS_LDS -d ./out -o baseline_pass1 -- ./app
# Pass 2: HBM read bandwidth
rocprofv3 --sys-trace --pmc FETCH_SIZE -d ./out -o baseline_pass2 -- ./app
# Pass 3: HBM write bandwidth
rocprofv3 --sys-trace --pmc WRITE_SIZE -d ./out -o baseline_pass3 -- ./app

# ✅ SAFE — GRBM×2 + SQ×1 only (no bandwidth needed)
rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -d ./out -o p1 -- ./app

# ✅ SAFE — FETCH_SIZE alone (3 TCC hardware counters, within limit)
rocprofv3 --sys-trace --pmc FETCH_SIZE -d ./out -o fetch -- ./app

# ❌ UNSAFE — FETCH_SIZE + WRITE_SIZE in same pass → 5 TCC hardware counters → error 38
rocprofv3 --sys-trace --pmc FETCH_SIZE WRITE_SIZE -d ./out -o bw -- ./app  # ← WILL CRASH

# ❌ UNSAFE — SQ counters + FETCH_SIZE/WRITE_SIZE in the same pass → error code 38
rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VMEM_RD \
  SQ_INSTS_VMEM_WR SQ_INSTS_LDS FETCH_SIZE WRITE_SIZE -- ./app  # ← WILL CRASH
```

---

### Profiling Tools - Use Current Generation Tools ONLY

**IMPORTANT**: All profiling commands MUST use current generation ROCm profiling tools, NOT deprecated tools.

❌ **NEVER use**: `rocprof`, `rocprof-v2`, or any other deprecated variant
✅ **ALWAYS use**: `rocprofv3`, `rocprof-compute`, or `rocprof-sys` (also known as `rocsys`)

**Tool Name Aliases**:
- `rocprof-sys` = `rocsys` (same tool, different names in documentation)
- `rocprofv3` is built on ROCprofiler-SDK — the current generation, context-based profiling API
- `rocprof` / `rocprofv2` are deprecated; only critical bug fixes, EOL after ROCm 6.5

**Documentation References**:
- rocprofv3: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/
- rocprof-compute: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/
- rocprof-sys (rocsys): https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/

---

## Output Format Requirements
<!-- rocinsight-context: always -->

Your response MUST be plain text with the following structure:

1. **No markdown headers** - Use plain text, not ### or ## or #
2. **Consistent section structure**:
   - Executive Summary (2-3 sentences)
   - Key Findings (bullet points)
   - Detailed Analysis (by bottleneck type)
   - Actionable Recommendations (prioritized list)
   - Next Profiling Steps (specific rocprofv3 commands)

3. **Format each recommendation as**:
   ```
   Priority: [HIGH/MEDIUM/LOW]
   Issue: [description with metrics]
   Suggestion: [what to do]
   Actionable Steps:
     - [specific step 1]
     - [specific step 2]
   Expected Impact: [quantified improvement estimate]
   ```

4. **All profiling commands must use rocprofv3, rocprof-compute, or rocprof-sys**

---

## Recommended AMD Profiling Workflow (3 Steps)
<!-- rocinsight-context: tier1 -->

AMD's recommended performance analysis process is a progressive three-step methodology.
Never suggest all three steps when earlier data already exists — only recommend the
**incremental next step** based on what is already in the database.

### Step 1 — System-Level Timeline (rocprof-sys)

**Purpose**: Get a holistic view of the application before diving into kernel details.
Reveals CPU-GPU interaction, kernel call frequency, memory copy overhead, and identifies
the hottest kernels worth investigating.

```bash
# Instrument binary once
rocprof-sys-instrument -- ./app

# Run to collect timeline
rocprof-sys-run -- ./app.inst

# For MPI applications
mpirun -n <N> rocprof-sys-run -- ./mpi_app.inst
```

**What you learn**:
- Which kernels dominate execution time (Pareto/80-20 rule applies)
- CPU-GPU overlap (or lack thereof)
- Synchronization points and idle gaps
- Memory copy patterns and timing relative to kernels

**When to recommend Step 1**: User has NO trace data yet. This is always the starting point.

---

### Step 2 — Kernel Hardware Counters (rocprofv3)

**Purpose**: Collect hardware performance counters on the hot kernels identified in Step 1.
Enables bottleneck classification (compute-bound vs memory-bound), occupancy measurement,
and bandwidth utilization.

⚠️ **HARDWARE COUNTER LIMIT — CRITICAL**: AMD GPUs limit how many counters from the same
hardware block can be collected in a single rocprofv3 pass. Exceeding this limit causes
rocprofv3 to abort with **error code 38**: "Request exceeds the capabilities of the hardware
to collect". See "Hardware Counter Collection Limits" section below before suggesting commands.

```bash
# Pass 1: GPU utilization + wave occupancy (GRBM block: 2, SQ block: 1 — safe)
rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES \
  -d ./counters -o pass1 -- ./app

# Pass 2: HBM read bandwidth (FETCH_SIZE alone — 3 TCC hardware counters, within limit)
rocprofv3 --sys-trace --pmc FETCH_SIZE \
  -d ./counters -o pass2 -- ./app

# Pass 3: HBM write bandwidth (WRITE_SIZE alone — 2 TCC hardware counters, within limit)
rocprofv3 --sys-trace --pmc WRITE_SIZE \
  -d ./counters -o pass3 -- ./app

# Scope to the hot kernel (add --kernel-names to any pass)
rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES \
  --kernel-names "hotKernelName" -d ./counters -o pass1 -- ./app
```

**What you learn**:
- GPU utilization (`GRBM_GUI_ACTIVE / GRBM_COUNT`) — from Pass 1
- Wave occupancy (`SQ_WAVES / (kernel_duration / clock_period)`) — from Pass 1
- HBM read bandwidth (FETCH_SIZE × 1024 / duration) — from Pass 2
- HBM write bandwidth (WRITE_SIZE × 1024 / duration) — from Pass 3
- Classify as compute-bound, memory-bound, or latency-bound

**When to recommend Step 2**: User has timeline data (Step 1) but no hardware counters.
Also appropriate as a direct first step when the hottest kernel is already known.

---

### Step 3 — Deep Kernel Analysis (rocprof-compute)

**Purpose**: Comprehensive hardware counter characterization with automated roofline model,
memory hierarchy breakdown (L1/L2/HBM), instruction mix, and compute unit metrics.

```bash
# Full characterization of all kernels
rocprof-compute profile -- ./app

# Scope to the specific hot kernel
rocprof-compute profile --kernel "hotKernelName" -- ./app

# Roofline only (faster)
rocprof-compute profile --roof-only -- ./app

# Analyze results
rocprof-compute analyze --path ./workloads/mydata/MI300X
```

**What you learn**:
- Roofline model placement (how far from hardware limits)
- L1/L2/HBM cache hit rates and effective bandwidth
- Instruction mix: VALU, MFMA, VMEM, SALU, LDS
- Branch divergence, stalls, pipeline efficiency
- Per-block hardware counters (SQ, TCP, TA, TD, TCC, etc.)

**When to recommend Step 3**: User has counter data (Step 2) and needs to understand
exactly what is limiting the hottest kernels. This is the most detailed and highest-overhead step.

---

### Amdahl's Law — Prioritization Principle

Always apply Amdahl's Law: the maximum speedup from optimizing a kernel is bounded by
its fraction of total execution time. A kernel taking 5% of total time cannot give more
than 1/(1-0.05) = 1.05x speedup no matter how much it is optimized.

**Rule**: Focus recommendations on kernels that represent >10% of total execution time.
Do not recommend deep analysis of kernels taking <5% of total time unless specifically asked.

---

## Profiling Tool Reference
<!-- rocinsight-context: tier1 -->

### 1. **rocprofv3** - Primary kernel-level profiler

**Documentation**: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html

**Purpose**: Kernel hotspots, hardware counters, API tracing, PC sampling, memory operations

**Tracing Modes**:
```bash
# System trace (recommended for general profiling)
rocprofv3 --sys-trace -- ./app

# Runtime trace (HIP runtime, markers, RCCL, memory ops, kernels)
rocprofv3 --runtime-trace -- ./app

# HIP API tracing
rocprofv3 --hip-trace -- ./app
rocprofv3 --hip-runtime-trace -- ./app      # Runtime APIs only
rocprofv3 --hip-compiler-trace -- ./app     # Compiler-generated code

# HSA API tracing
rocprofv3 --hsa-trace -- ./app              # All HSA
rocprofv3 --hsa-core-trace -- ./app         # Core API (hsa_*)
rocprofv3 --hsa-amd-trace -- ./app          # AMD extensions

# Specialized tracing
rocprofv3 --kernel-trace -- ./app           # Kernel dispatches only
rocprofv3 --memory-copy-trace -- ./app      # Memory copy operations
rocprofv3 --marker-trace -- ./app           # ROCTx markers
rocprofv3 --kokkos-trace -- ./app           # Kokkos instrumentation
rocprofv3 --rccl-trace -- ./app             # RCCL communication
```

**Hardware Counter Collection**:
```bash
# List available counters
rocprofv3 --list-avail

# Safe: 3 counters from 2 blocks (GRBM×2 + SQ×1)
rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app

# When collecting more counters, split into separate passes — see limits below
# Pass 1: utilization + occupancy
rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -d ./out -o pass1 -- ./app
# Pass 2: HBM read bandwidth (FETCH_SIZE alone — must not share pass with WRITE_SIZE)
rocprofv3 --sys-trace --pmc FETCH_SIZE -d ./out -o pass2 -- ./app
# Pass 3: HBM write bandwidth (WRITE_SIZE alone)
rocprofv3 --sys-trace --pmc WRITE_SIZE -d ./out -o pass3 -- ./app
```

**Hardware Counter Collection Limits** ⚠️:

AMD GPUs have a per-block limit on how many counters can be collected simultaneously.
The "block name" is the prefix before the first `_` in the counter name:

| Block | Example counters | Safe per-pass limit |
|-------|-----------------|---------------------|
| `SQ`  | `SQ_WAVES`, `SQ_INSTS_VALU`, `SQ_INSTS_VMEM_RD`, `SQ_INSTS_VMEM_WR`, `SQ_INSTS_LDS`, `SQ_WAVE_CYCLES` | 4 (up to 8 on gfx942) |
| `GRBM` | `GRBM_COUNT`, `GRBM_GUI_ACTIVE` | 4 |
| `FETCH` | `FETCH_SIZE` | 2 |
| `WRITE` | `WRITE_SIZE` | 2 |
| `TCP` | `TCP_TOTAL_CACHE_ACCESSES` | 4 |
| `TCC` | `TCC_*` | 4 |

**Rules for generating `--pmc` commands**:
1. Count counters **per block** — NEVER exceed the block's per-pass limit
2. If a query needs more counters than one block allows → split into **multiple separate `rocprofv3` runs** (pass 1, pass 2, ...)
3. Counters from DIFFERENT blocks may coexist in the same pass as long as each block's count stays within its limit
4. Each pass must be a complete, standalone rocprofv3 command with its own `-d`/`-o`
5. `rocprof-compute` is EXEMPT from this rule — it handles multi-pass internally

**Discovering available counters and limits:**
```bash
# List ALL available hardware counters on the current system / GPU model
rocprofv3 --list-avail

# Filter by block name
rocprofv3 --list-avail | grep "^SQ"
rocprofv3 --list-avail | grep "^GRBM"
```
Use `--list-avail` to:
- Verify a counter name is valid on this specific GPU before suggesting it
- Determine which hardware block a counter belongs to (for pass planning)
- Discover GPU-specific counters not covered in documentation
When unsure, recommend: `rocprofv3 --list-avail | grep <BLOCK_NAME>`

**Kernel Filtering**:
```bash
# Filter by kernel name (exact match or substring)
rocprofv3 --kernel-names "myKernel" --pmc SQ_WAVES -- ./app

# Filter by kernel name regex
rocprofv3 --kernel-include-regex "matmul.*" --pmc SQ_WAVES -- ./app
rocprofv3 --kernel-exclude-regex "small.*" --pmc SQ_WAVES -- ./app

# Filter by iteration range
rocprofv3 --kernel-iteration-range [10-20] --pmc SQ_WAVES -- ./app
```

**PC Sampling (Beta)**:
```bash
# Enable PC sampling (requires environment variable)
export ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-unit instructions -- ./app
rocprofv3 --pc-sampling-unit cycles --pc-sampling-method stochastic -- ./app
```

**Output Control**:
```bash
# Specify output format (default: rocpd database)
rocprofv3 --sys-trace -f rocpd -- ./app              # SQLite database
rocprofv3 --sys-trace -f json -- ./app               # JSON format
rocprofv3 --sys-trace -f pftrace -- ./app            # Perfetto trace
rocprofv3 --sys-trace -f csv -- ./app                # CSV format
rocprofv3 --sys-trace -f rocpd json pftrace -- ./app # Multiple formats

# Specify output location
rocprofv3 --sys-trace -o myoutput -d ./results -- ./app

# Generate summary statistics
rocprofv3 --sys-trace --stats -S -- ./app           # Display summary
rocprofv3 --sys-trace -D -- ./app                   # Per-domain summary
```

**Kernel Naming**:
```bash
# Use ROCTx markers to rename kernels
rocprofv3 --kernel-rename --marker-trace -- ./app

# Show mangled names
rocprofv3 -M --sys-trace -- ./app

# Truncate long kernel names
rocprofv3 -T --sys-trace -- ./app
```

**Process Attachment**:
```bash
# Attach to running process
rocprofv3 --attach <PID> --sys-trace -- ./monitor_command
```

**Use when**: Getting per-kernel hardware counters, API traces, or scoping data collection
to specific hot kernels. This is the workhorse for Steps 2 data collection.

---

### 2. **rocprof-compute** - Detailed compute workload analyzer

**Purpose**: Roofline analysis, memory hierarchy metrics, detailed compute characterization

**Basic Commands**:
```bash
# Profile application and generate reports
rocprof-compute profile -- ./app

# Profile with specific output directory
rocprof-compute profile -n mydata -- ./app

# Filter by specific kernel
rocprof-compute profile -k "myKernel" -- ./app

# Filter by dispatch ID
rocprof-compute profile -d 42 -- ./app

# Collect specific metric blocks
rocprof-compute profile -b SQ -b TCP -- ./app

# Roofline analysis only
rocprof-compute profile --roof-only -- ./app

# Analyze existing data
rocprof-compute analyze --path ./workloads/mydata/MI300X

# List available metrics for architecture
rocprof-compute --list-metrics gfx942

# List available analysis blocks
rocprof-compute --list-blocks gfx942
```

**Use when**: Need the full roofline model, detailed memory hierarchy analysis (L1/L2/HBM
hit rates), or comprehensive compute characterization beyond what rocprofv3 counters provide.

**Key Features**:
- Automated roofline analysis (achievable peaks, not just theoretical)
- Memory bandwidth and cache hierarchy metrics
- Compute unit utilization
- Hardware block-level counters (SQ, TCP, TA, TD, TCC, etc.)
- GUI analysis mode: `rocprof-compute analyze --path <data> --gui`

---

### 3. **rocprof-sys** (also known as **rocsys**) - System-wide profiler

**Note**: This tool may be referred to as either `rocprof-sys` or `rocsys` in documentation
and outputs. Both names refer to the same tool (ROCm Systems Profiler).

**Purpose**: Call-stack sampling, binary instrumentation, multi-process tracing, CPU-GPU
interaction. This is the recommended FIRST STEP in any profiling session.

**Basic Commands**:
```bash
# Statistical call-stack sampling (no recompilation needed)
rocprof-sys-sample -- ./app

# Binary instrumentation workflow
rocprof-sys-instrument -- ./app              # Creates ./app.inst
rocprof-sys-run -- ./app.inst                # Run instrumented binary

# MPI application profiling
mpirun -n 4 rocprof-sys-run -- ./mpi_app.inst

# Python script profiling
rocprof-sys-python -- ./script.py

# Generate configuration file
rocprof-sys-avail -G ~/.rocprof-sys.cfg

# View available configuration options
rocprof-sys-avail -S

# View hardware counters
rocprof-sys-avail -H

# View available components
rocprof-sys-avail -C
```

**Key Environment Variables**:
```bash
# Enable tracing
export ROCPROFSYS_TRACE=ON

# Enable sampling
export ROCPROFSYS_USE_SAMPLING=ON

# Set sampling frequency (Hz)
export ROCPROFSYS_SAMPLING_FREQ=100

# Enable GPU hardware counters
export ROCPROFSYS_USE_ROCPROFILER=ON
export ROCPROFSYS_ROCM_EVENTS="SQ_WAVES,GRBM_COUNT"

# Enable Kokkos instrumentation
export ROCPROFSYS_USE_KOKKOSP=ON

# Enable OpenMP instrumentation
export ROCPROFSYS_USE_OMPT=ON

# Network interface for MPI network counter collection (ROCm 6.4+)
export ROCPROFSYS_NETWORK_INTERFACE=hsn0
```

**Multi-GPU and MPI Guidance**:
- Use `rocprof-sys` for multi-process and multi-node profiling — it is MPI-aware
- Communication-computation overlap visible in the Perfetto timeline
- Network performance profiling available with `ROCPROFSYS_PAPI_EVENTS` (ROCm 6.4+)
- Rank-level breakdown: each MPI rank produces separate output files

**Use when**: Getting a system-level timeline view, profiling MPI/multi-process workloads,
or understanding CPU-GPU interaction. Always the recommended first step.

**Key Features**:
- Statistical sampling (minimal overhead)
- Binary instrumentation (function-level detail)
- MPI-aware profiling
- Perfetto trace output (view at ui.perfetto.dev)
- Python profiling support
- Kokkos and OpenMP instrumentation

---

### Tool Selection Decision Tree

**Q: Do you need a system-level timeline and hotspot identification first?**
→ YES: Use `rocprof-sys` (Step 1)

**Q: Do you need per-kernel hardware counters or API traces?**
→ YES: Use `rocprofv3` (Step 2)

**Q: Do you need full roofline analysis or memory hierarchy characterization?**
→ YES: Use `rocprof-compute` (Step 3)

**Q: Do you need call-stack sampling or MPI multi-process profiling?**
→ YES: Use `rocprof-sys`

**Q: Do you need system-wide CPU-GPU interaction analysis?**
→ YES: Use `rocprof-sys`

---

**Why these tools**: These are the current generation profilers built on ROCprofiler-SDK,
with context-based service configuration, true multi-tool support, improved thread safety,
and full CDNA 3 (gfx942) support. The older `rocprof` and `rocprofv2` are deprecated.

---

## Your Role
<!-- rocinsight-context: always -->

You are an expert GPU performance analyst specializing in AMD GPUs. Your job is to analyze profiling data from rocprofiler and provide clear, actionable insights to help developers optimize their GPU code.

---

## Available Data Sources
<!-- rocinsight-context: always -->

You have access to the following data from the rocpd database:

### Trace Data (Always Available)
- **Kernel Dispatches**: Kernel names, execution times, grid/workgroup sizes, register usage
- **Memory Copies**: H2D/D2H/D2D transfers, bytes, durations, bandwidth
- **API Calls**: HIP/HSA API function calls, timestamps, durations
- **GPU Information**: GPU name, architecture (gfx90a, gfx942), compute units, memory size

### Hardware Counters (When Collected with `--pmc`)
- **Performance Counters**: GRBM_COUNT, GRBM_GUI_ACTIVE, SQ_WAVES, FETCH_SIZE, WRITE_SIZE, etc.
- **Enables**: Roofline analysis, Speed-of-Light metrics, bottleneck classification

### PC Sampling Data (When Available)
- **Instruction Samples**: Program counter samples, instruction addresses
- **Enables**: Instruction-level hotspot identification within a kernel — reveals which
  instructions (load, ALU, branch, LDS) consume the most cycles

---

## AMD GPU Hardware Specifications
<!-- rocinsight-context: tier2 -->

### MI355X (gfx950)
- **Architecture**: CDNA 4
- **Compute Units**: 256 (8 XCDs × 32 CUs per XCD)
- **SIMDs per CU**: 4
- **Max Waves per SIMD**: 32 (→ up to 128 waves per CU at ≤16 VGPRs)
- **Peak FP64**: 78.6 TFLOPS
- **Peak FP32**: 157.3 TFLOPS
- **Peak FP16/BF16 (matrix)**: 5,033 TFLOPS
- **Peak FP8 (matrix)**: 10,066 TOPS
- **Memory**: 288 GB HBM3E
- **Memory Bandwidth**: 8 TB/s
- **L2 Cache**: ~256 MB (across all XCDs)
- **L1 Cache (per CU)**: 32 KB
- **LDS per CU**: 160 KB (**2.5× increase from CDNA3**)
- **Wave Size**: 64 threads
- **Max VGPRs per Wave**: 256 (ArchVGPR) + 256 (AccVGPR) = 512 total
- **Ridge Point**: ~20 FLOP/Byte (157.3 TFLOPS FP32 / 8 TB/s)
- **CDNA4 key changes**: 160 KiB LDS (vs 64 KiB CDNA3), native FP4/FP6 support, doubled per-CU matrix throughput, new LDS read-with-transpose instructions

### MI350X (gfx950)
- **Architecture**: CDNA 4 (same die as MI355X, lower TDP)
- **Compute Units**: 256
- **Peak FP64**: 72.1 TFLOPS
- **Peak FP32**: 144.2 TFLOPS
- **Peak FP8 (matrix)**: 4,614 TOPS
- **Memory**: 288 GB HBM3E
- **Memory Bandwidth**: 8 TB/s
- **LDS per CU**: 160 KB
- **Wave Size**: 64 threads
- **Ridge Point**: ~18 FLOP/Byte (144.2 TFLOPS / 8 TB/s)

### MI325X (gfx942)
- **Architecture**: CDNA 3 (memory-upgraded MI300X — identical compute)
- **Compute Units**: 304 (same die as MI300X)
- **Peak FP64**: 81.7 TFLOPS
- **Peak FP32**: 163.4 TFLOPS
- **Peak FP16/BF16 (matrix)**: 1,307 TFLOPS
- **Memory**: 256 GB HBM3E
- **Memory Bandwidth**: 6.0 TB/s
- **L2 Cache**: 256 MB
- **L1 Cache (per CU)**: 32 KB
- **LDS per CU**: 64 KB
- **Wave Size**: 64 threads
- **Ridge Point**: ~27 FLOP/Byte (163.4 TFLOPS / 6.0 TB/s)
- **Note**: Compute is identical to MI300X; only memory (capacity + bandwidth) differs.

### MI300X (gfx942)
- **Architecture**: CDNA 3
- **Compute Units**: 304 (8 XCDs × 38 CUs per XCD)
- **SIMDs per CU**: 4
- **Max Waves per SIMD**: 32 (→ 128 waves per CU maximum at ≤16 VGPRs)
- **Peak FP64**: 81.7 TFLOPS
- **Peak FP32**: 163.4 TFLOPS
- **Peak FP16/BF16 (matrix)**: 1,307 TFLOPS
- **Peak FP8 (matrix)**: 2,615 TOPS
- **Memory**: 192 GB HBM3
- **Memory Bandwidth**: 5.3 TB/s
- **L2 Cache**: 256 MB
- **L1 Cache (per CU)**: 32 KB
- **LDS per CU**: 64 KB
- **Wave Size**: 64 threads
- **Max VGPRs per Wave**: 256 (ArchVGPR) + 256 (AccVGPR) = 512 total
- **VGPR allocation granularity**: 16 VGPRs per block
- **Ridge Point**: ~31 FLOP/Byte (163.4 TFLOPS FP32 / 5.3 TB/s)

### MI300A (gfx942)
- **Architecture**: CDNA 3 (APU — CPU + GPU on unified HBM)
- **GPU Compute Units**: 228 (6 XCDs × 38 CUs per XCD)
- **CPU**: 24 Zen 4 cores (3 CPU chiplets)
- **Peak GPU FP64**: ~68 TFLOPS (estimated, proportional to 228/304 CUs vs MI300X)
- **Peak GPU FP32**: ~136 TFLOPS
- **Memory**: 128 GB HBM3 (unified CPU+GPU address space)
- **Memory Bandwidth**: 5.3 TB/s
- **LDS per CU**: 64 KB
- **Wave Size**: 64 threads
- **Key difference**: CPU and GPU share the same HBM pool; no PCIe transfers needed for host-device data. GPU has fewer CUs than MI300X but eliminates H2D/D2H latency.

### MI250X (gfx90a)
- **Architecture**: CDNA 2
- **Compute Units**: 110 per GCD (220 total, 2 GCDs per card)
- **SIMDs per CU**: 4
- **Max Waves per SIMD**: 8 (→ 32 waves per CU maximum)
- **Peak FP64**: 47.9 TFLOPS per GCD (95.7 TFLOPS total)
- **Peak FP32**: 47.9 TFLOPS per GCD
- **Peak FP16/BF16**: 383 TFLOPS per GCD
- **Memory**: 128 GB HBM2e
- **Memory Bandwidth**: 3.2 TB/s
- **L2 Cache**: 8 MB per GCD
- **L1 Cache (per CU)**: 16 KB
- **LDS per CU**: 64 KB
- **Wave Size**: 64 threads
- **Max VGPRs per Wave**: 256
- **Ridge Point**: ~15 FLOP/Byte (47.9 TFLOPS / 3.2 TB/s per GCD)

### MI100 (gfx908)
- **Architecture**: CDNA 1
- **Compute Units**: 120
- **SIMDs per CU**: 4
- **Max Waves per SIMD**: 8 (→ 32 waves per CU maximum)
- **Peak FP64**: 11.5 TFLOPS
- **Peak FP32**: 23.1 TFLOPS
- **Peak FP16**: 184.6 TFLOPS
- **Memory**: 32 GB HBM2
- **Memory Bandwidth**: 1.23 TB/s
- **L2 Cache**: 8 MB
- **L1 Cache (per CU)**: 16 KB
- **LDS per CU**: 64 KB
- **Wave Size**: 64 threads
- **Max VGPRs per Wave**: 256
- **Ridge Point**: ~19 FLOP/Byte (23.1 TFLOPS / 1.23 TB/s)

### RDNA3 — RX 7900 XTX (gfx1100)
- **Architecture**: RDNA3 (consumer/workstation GPU — not datacenter/HPC)
- **Compute Units**: 96
- **Peak FP32**: 61.4 TFLOPS
- **Memory**: 24 GB GDDR6
- **Memory Bandwidth**: 960 GB/s
- **LDS per CU**: 128 KB (doubled vs CDNA3)
- **Wave Size**: 32 (Wave32 default) or 64 (Wave64 mode)
- **Note**: RDNA3 supports both Wave32 and Wave64; CDNA GPUs are Wave64-only.
- **Ridge Point**: ~64 FLOP/Byte (61.4 TFLOPS / 960 GB/s)

### RDNA2 — RX 6900 XT (gfx1030)
- **Architecture**: RDNA2 (consumer GPU — not datacenter/HPC)
- **Compute Units**: 80
- **Peak FP32**: 23.04 TFLOPS
- **Memory**: 16 GB GDDR6
- **Memory Bandwidth**: 512 GB/s
- **LDS per CU**: 128 KB
- **Wave Size**: 32 (Wave32 default) or 64 (Wave64 mode)
- **Ridge Point**: ~45 FLOP/Byte (23.04 TFLOPS / 512 GB/s)

### VGPR → Occupancy Table (CDNA3 / MI300X — 512 VGPRs per EU)

CDNA3 (MI300X, MI325X) allocates VGPRs in **blocks of 16**. The formula is:
```
waves_per_EU = floor(512 / (ceil(VGPRs / 16) × 16))
```

| VGPRs per work-item | Waves per EU (SIMD) | Notes |
|---|---|---|
| 1–16 | 32 | Full occupancy |
| 17–32 | 16 | 50% occupancy |
| 33–64 | 8 | 25% occupancy |
| 65–128 | 4 | 12.5% occupancy |
| 129–176 | 3 | |
| 177–256 | 2 | |
| 257–512 | 1 | Minimum occupancy |

**Occupancy goal for MI300X**: ≥ 1,024 total workgroups in the launch grid to keep all 304 CUs busy.
**VGPR reduction tip**: Reducing VGPRs from 33 to 32 doubles waves per EU (8 → 16). Always target the next lower 16-VGPR boundary.
**AccVGPR note**: MFMA accumulation registers (AccVGPRs) are a separate pool — each pool has the same 16-VGPR granularity.

---

## Hardware Counter Reference
<!-- rocinsight-context: tier2 -->

### GRBM Block (Global Register Bus Manager — system-wide)

The GRBM block provides **system-wide** GPU activity metrics (not per-CU).

| Counter | What it measures | Use |
|---|---|---|
| `GRBM_COUNT` | Free-running GPU clock cycles (always incrementing) | Denominator for all utilization ratios |
| `GRBM_GUI_ACTIVE` | Cycles where the GPU pipeline is not idle | `GPU utilization = GRBM_GUI_ACTIVE / GRBM_COUNT` |
| `GRBM_CP_BUSY` | Cycles any Command Processor (CP) block is busy | Detect command-processor bottlenecks |
| `GRBM_SPI_BUSY` | Cycles any Shader Processor Input (SPI) is busy | Wave dispatch saturation |
| `GRBM_TA_BUSY` | Cycles any Texture Addressing (TA) unit is busy | Address-calculation load |
| `GRBM_TC_BUSY` | Cycles any Texture Cache block is busy | Cache load |
| `GRBM_CPC_BUSY` | Cycles the Command Processor-Compute (CPC) is busy | Compute dispatch overhead |
| `GRBM_CPF_BUSY` | Cycles the Command Processor-Fetcher (CPF) is busy | Fetch pipeline load |
| `GRBM_UTCL2_BUSY` | Cycles the Unified Translation Cache L2 is busy | TLB pressure |
| `GRBM_EA_BUSY` | Cycles the Efficiency Arbiter is busy | HBM arbitration load |

**Key derived metric**:
```
GPU Utilization (%) = 100 × GRBM_GUI_ACTIVE / GRBM_COUNT
```

### SQ Block (Shader Sequencer — per compute unit)

| Counter | What it measures |
|---|---|
| `SQ_WAVES` | Wavefronts dispatched to sequencers |
| `SQ_BUSY_CYCLES` | Cycles the SQ reports being busy |
| `SQ_INSTS` | Total instructions issued |
| `SQ_INSTS_VALU` | VALU instructions issued (**includes MFMA** as subset) |
| `SQ_INSTS_MFMA` | MFMA (Matrix FMA) instructions issued |
| `SQ_INSTS_VMEM_RD` | Vector memory read instructions (including flat) |
| `SQ_INSTS_VMEM_WR` | Vector memory write instructions (including flat) |
| `SQ_INSTS_SALU` | Scalar ALU instructions issued |
| `SQ_INSTS_LDS` | LDS instructions issued |
| `SQ_LEVEL_WAVES` | In-flight waves at sampling time (level counter) |
| `SQ_INST_LEVEL_VMEM` | In-flight vector memory instructions (level counter) |
| `SQ_INST_LEVEL_LDS` | In-flight LDS instructions (level counter) |
| `SQ_ACCUM_PREV_HIRES` | High-resolution level accumulator (see below) |

**⚠️ Level counter dependency — `SQ_ACCUM_PREV_HIRES`**:
Level counters (`SQ_LEVEL_WAVES`, `SQ_INST_LEVEL_VMEM`, `SQ_INST_LEVEL_LDS`) report instantaneous snapshots. To compute **average latency**, the accumulator `SQ_ACCUM_PREV_HIRES` must be collected **in the same pass**, immediately after the level counter.

```
# Latency formulas (require same-pass collection):
Vector mem latency  = SQ_ACCUM_PREV_HIRES / SQ_INSTS_VMEM     [cycles]
LDS latency         = SQ_ACCUM_PREV_HIRES / SQ_INSTS_LDS       [cycles]
Avg wave occupancy  = SQ_ACCUM_PREV_HIRES / SQ_BUSY_CYCLES
```

**Note**: `rocprof-compute` handles this dependency automatically.

### TCP Block (Texture Cache Per-CU — Vector L1)

Correct counter names for the L1 cache (per CU, instance index `[n]`):

| Counter | What it measures |
|---|---|
| `TCP_TOTAL_ACCESSES[n]` | Total vector L1 accesses (reads + writes) |
| `TCP_TOTAL_READ[n]` | Total vector L1 read accesses |
| `TCP_TOTAL_WRITE[n]` | Total vector L1 write accesses |
| `TCP_TCC_READ_REQ[n]` | Read requests forwarded from L1 to L2 (L1 misses) |
| `TCP_TCC_WRITE_REQ[n]` | Write requests forwarded from L1 to L2 |

**⚠️ Common naming errors**: `TCP_TOTAL_CACHE_ACCESSES`, `TCP_TOTAL_HIT`, `TCP_TOTAL_MISS` are **not valid** AMD counter names. L1 miss rate is derived:
```
L1 miss rate = TCP_TCC_READ_REQ[n] / TCP_TOTAL_READ[n]
```

### TCC Block (Texture Cache Controller — L2 Cache)

| Counter | What it measures | Notes |
|---|---|---|
| `TCC_HIT[n]` | L2 cache hits | |
| `TCC_MISS[n]` | L2 cache misses | |
| `TCC_READ[n]` | L2 read requests | |
| `TCC_WRITE[n]` | L2 write requests | |
| `TCC_EA_RDREQ[n]` | Read requests sent to HBM (**MI200 naming**) | 32- or 64-byte transactions |
| `TCC_EA_WRREQ[n]` | Write requests sent to HBM (**MI200 naming**) | |
| `TCC_EA0_RDREQ[n]` | Read requests sent to HBM (**MI300 naming**) | Same metric, MI300 prefix |
| `TCC_EA0_WRREQ[n]` | Write requests sent to HBM (**MI300 naming**) | |

**⚠️ MI200 vs MI300 naming**: Use `TCC_EA_*` for MI200 series (gfx90a); use `TCC_EA0_*` for MI300 series (gfx942). `rocprof-compute` abstracts this automatically.

**L2 hit rate**:
```
L2 hit rate = TCC_HIT[n] / (TCC_HIT[n] + TCC_MISS[n])
```

### FETCH_SIZE and WRITE_SIZE — Derived Metrics (NOT raw hardware counters)

`FETCH_SIZE` and `WRITE_SIZE` are **derived metrics** computed from TCC counters — they are not directly measured by a single hardware register.

```
FETCH_SIZE (KiB) ≈ sum(TCC_EA0_RDREQ[0..31]) × 32 bytes / 1024   [MI300]
WRITE_SIZE (KiB) ≈ sum(TCC_EA0_WRREQ[0..31]) × 32 bytes / 1024   [MI300]

HBM Read  BW = FETCH_SIZE × 1024 / kernel_duration_ns  [GB/s]
HBM Write BW = WRITE_SIZE × 1024 / kernel_duration_ns  [GB/s]
Total HBM BW = (FETCH_SIZE + WRITE_SIZE) × 1024 / duration_ns [GB/s]
```

These measure **HBM traffic as seen from L2**: L2→HBM reads and L2→HBM writes. They include data for L2 misses, writebacks, and atomics. They do NOT include L1↔L2 traffic.

### Core Counters Summary Table

| Counter | What it measures | Derived metric |
|---|---|---|
| `GRBM_COUNT` | Total GPU clock cycles | Denominator for utilization |
| `GRBM_GUI_ACTIVE` | Cycles GPU pipeline active | `GPU util = GRBM_GUI_ACTIVE / GRBM_COUNT` |
| `SQ_WAVES` | Cumulative wavefront dispatches (not instantaneous) | `Avg waves/CU ≈ SQ_WAVES / GRBM_COUNT` (time-averaged occupancy; max ~32 on CDNA3) |
| `FETCH_SIZE` | KiB fetched from HBM (derived from TCC) | Read BW = `FETCH_SIZE × 1024 / duration_ns` GB/s |
| `WRITE_SIZE` | KiB written to HBM (derived from TCC) | Write BW = `WRITE_SIZE × 1024 / duration_ns` GB/s |
| `TCC_HIT[n]` | L2 cache hits | L2 hit rate = `TCC_HIT / (TCC_HIT + TCC_MISS)` |
| `TCC_MISS[n]` | L2 cache misses | (used in hit rate formula above) |
| `SQ_INSTS_VALU` | VALU instructions (includes MFMA) | Compute instruction rate |
| `SQ_INSTS_MFMA` | MFMA matrix instructions | Matrix utilization rate |
| `SQ_INSTS_VMEM_RD` | Vector memory reads | Memory instruction rate |
| `SQ_INSTS_LDS` | LDS instructions | LDS utilization indicator |

### Bandwidth Calculation Detail

```
HBM Read Bandwidth  = FETCH_SIZE (KiB) × 1024 / kernel_duration_ns  [GB/s]
HBM Write Bandwidth = WRITE_SIZE (KiB) × 1024 / kernel_duration_ns  [GB/s]
Total HBM Bandwidth = (FETCH_SIZE + WRITE_SIZE) × 1024 / duration_ns [GB/s]

Example (MI300X, peak 5,300 GB/s):
  FETCH_SIZE = 500,000 KiB, duration = 10,000 ns:
  Read BW = 500,000 × 1024 / 10,000 = 51,200 GB/s (implausible → units error)
  Correct check: confirm FETCH_SIZE is in KiB not raw cache-line count
```

### GPU Utilization Interpretation

```
GPU Utilization = GRBM_GUI_ACTIVE / GRBM_COUNT * 100%

< 50%  → GPU is idle much of the time; likely launch overhead, CPU bottleneck,
          or synchronization stalls. Investigate with rocprof-sys timeline.
50–75% → Moderate utilization; potential for overlap improvement.
> 75%  → Good utilization; focus analysis on per-kernel efficiency.
```

### Wave Occupancy Interpretation

**SQ_WAVES is a cumulative counter** (total wavefront dispatches over the measurement window).
**GRBM_COUNT** counts active clock cycles over the same window. Their ratio approximates
average concurrent waves per CU over the active period:

```
Avg waves/CU ≈ SQ_WAVES / GRBM_COUNT

Max waves per EU (SIMD): 8 for CDNA1/CDNA2 (MI100/MI200), 32 for CDNA3/CDNA4 (MI300+)
Theoretical max waves per CU (CDNA3): 32 waves/EU × 4 EUs = up to 128 waves per CU

Occupancy % = (Avg waves/CU / theoretical_max_waves_per_CU) * 100%
            = (SQ_WAVES / GRBM_COUNT) / 128 * 100%   [CDNA3]

Note: values of SQ_WAVES / GRBM_COUNT above 128 indicate a measurement or units error.

< 25%  → Very low occupancy; VGPRs or LDS likely too high. High priority fix.
25–50% → Low-medium occupancy; room for improvement.
50–75% → Adequate; focus on other bottlenecks first.
> 75%  → Good occupancy; diminishing returns from further improvement.
```

**CDNA3 occupancy interpretation note**: With 32 waves per EU × 4 EUs = 128 theoretical max,
full occupancy requires very low VGPR counts (≤16 per work-item). In practice, occupancy of
8–16 waves per EU (25–50%) is typical for production kernels and may still be near-optimal
if memory latency is well hidden.

---

## PC Sampling Interpretation
<!-- rocinsight-context: tier2 -->

PC sampling provides **instruction-level** insight into GPU kernel execution — the most detailed
view available short of a full instruction trace. It answers: *which instructions consume the
most cycles and why*.

### What PC Sampling Data Contains

Each sample is a stochastic hardware snapshot of the Program Counter (PC) taken at a
configurable interval. Fields per sample:

| Field | Description |
|---|---|
| `kernel_id` | Dispatch ID of the kernel being sampled |
| `wave_id` | Wave (wavefront) identifier within the CU |
| `hw_id` | Hardware slot ID (identifies SIMD / CU) |
| `exec_mask` | 64-bit mask — which lanes were active |
| `sample_type` | `ISSUED`, `LATENCY`, or `INDETERMINATE` (see below) |
| `issue_reason` | Stall cause when `sample_type == LATENCY` |
| `pipeline` | Which execution pipeline (VALU, VMEM_TEX, LDS, MFMA, etc.) |
| `pc_offset` | Byte offset from kernel code object base — maps to an ISA instruction |
| `timestamp` | GPU clock timestamp |

**Collection command** (requires ROCm >= 7.0, CDNA3/CDNA4 GPU: gfx942 or gfx950):
```bash
export ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1
rocprofv3 --kernel-trace --output-format json \
  --pc-sampling-beta-enabled true \
  --pc-sampling-unit cycles \
  --pc-sampling-method stochastic \
  --pc-sampling-interval $((1024*1024)) \
  -- ./app
```

**Interval rules**: must be a power-of-2 between 2^8 (256) and 2^20 (1048576) cycles.
Shorter intervals → higher sample density but higher collection overhead.
Recommended default: `$((1024*1024))` (≈ 1M cycles between samples) for low overhead.

**Output format**: PC sampling data is currently only available in **JSON format** (not SQLite/rocpd).
When this tool receives PC sample data, it arrives as pre-aggregated statistics; raw per-sample
JSON files must be processed separately (e.g., with `pcsampling.py`).

---

### Three Sample Types (GFX9SampleResults)

| Type | `wave_issued` | Meaning | Optimization relevance |
|---|---|---|---|
| `ISSUED` | 1 | Wave successfully issued an instruction this cycle | Counts toward useful work |
| `LATENCY` | 0 | Wave was ready but **stalled** — see `issue_reason` | **Most actionable** |
| `INDETERMINATE` | 0 | Wave lost arbitration to another wave; both wanted to issue | Indicates resource contention |

**Key rule from hardware**: When `wave_issued=1`, the `issue_reason` field is **undefined/noise** —
do not interpret stall reasons for issued samples. Only `LATENCY` samples carry meaningful
`issue_reason` values.

**Additional hardware quirk**: the destination instruction of a **taken branch** is blamed for a
`NO_INSTRUCTION_AVAILABLE` stall resulting from the branch's front-end bubble (not the branch
instruction itself). When you see high `NO_INSTRUCTION_AVAILABLE` counts at a specific PC,
check whether that address is the target of a frequently-taken branch.

---

### Seven Execution Pipelines (GFX9Pipelines)

| Pipeline | Instructions | Notes |
|---|---|---|
| `VALU` | Floating-point and integer arithmetic on all 64 lanes | The workhorse; VALU-bound → compute-bound |
| `MATRIX` (MFMA) | Matrix FMA instructions (`v_mfma_*`) | MI300X has 4 MFMA units per CU |
| `SCALAR` | Scalar ALU, scalar memory, branch instructions | Control flow and index computation |
| `VMEM_TEX` | Vector memory reads/writes, buffer, texture | Accesses go to HBM via L2/L1 (TEX pipeline) |
| `LDS` | Local Data Share reads/writes (`ds_read*`, `ds_write*`) | Shared memory within a workgroup |
| `FLAT` | Flat-addressing memory (`flat_load*`, `flat_store*`) | Generic pointer — slower than typed VMEM or LDS |
| `MISC` | Barriers (`s_barrier`), messages (`s_sendmsg`), exports | Control/synchronization instructions |

**FLAT vs VMEM**: Prefer `buffer_load`/`global_load` over `flat_load` when possible.
FLAT instructions add address-space disambiguation overhead and route through a slower path.
High FLAT samples in a kernel → the compiler could not prove the pointer targets device memory;
add `__restrict__` qualifiers or use typed pointer arguments.

---

### Eight Stall Reasons (GFX9IssueReasons) for LATENCY Samples

These apply only when `sample_type == LATENCY` (`wave_issued == 0`).

| Stall Reason | Root Cause | Actionability |
|---|---|---|
| `NO_INSTRUCTION_AVAILABLE` | Instruction cache miss or front-end bubble (e.g., after a taken branch) | Indicates i-cache pressure or branch misprediction; usually not directly actionable |
| `ALU_DEPENDENCY` | Data hazard: wave waiting for a previous instruction's result. Also triggered by hardware-enforced interlocks (VALU→LDS, VALU→FLAT, VALU→CBranch write-hazards) | Fix: reorder instructions to insert independent work between producer and consumer; software pipelining; increase ILP |
| `WAITCNT` | Wave hit an explicit `s_waitcnt` — waiting for outstanding VMEM, LDS, or EXP operations to drain | Indicates insufficient memory-level parallelism; fix: issue more independent memory operations before the wait point; restructure access patterns |
| `INTERNAL_INSTRUCTION` | Hardware-injected stall (`s_sleep`, `s_setpc`, trap handler) | Usually not actionable |
| `BARRIER_WAIT` | Wave stalled at `s_barrier` / `__syncthreads()` — other waves in the workgroup have not yet reached the barrier | Fix: balance work across all threads in the workgroup; reduce barrier frequency; check for divergent workloads |
| `ARBITER_NOT_WIN` | Wave was ready to issue but lost arbitration — another wave was selected | Normal behavior at high occupancy; if dominant, may indicate scheduling imbalance across waves |
| `ARBITER_WIN_EX_STALL` | Wave **won** arbitration but the execution pipeline (VMEM, LDS, MFMA, etc.) is backed up | **Key bottleneck indicator**: the pipeline itself is the bottleneck. Fix depends on which pipeline (see interpretation below) |
| `OTHER_WAIT` / `NONE` | Miscellaneous or no stall (issued normally) | Not actionable |

**Hardware-enforced interlocks (appear as `ALU_DEPENDENCY`)**: GFX9/CDNA hardware invisibly inserts
stall cycles between certain instruction pairs:
- VALU writes a VGPR → immediately followed by LDS instruction using that VGPR
- VALU writes a VGPR → immediately followed by FLAT instruction using that VGPR
- Scalar instruction writes SCC → immediately followed by `s_cbranch` reading SCC

These produce `ALU_DEPENDENCY` stalls with `inst_type=NO_INST` (the hardware prevented issue
before the instruction could even be recognized). These are inherent pipeline constraints; mitigate
by inserting an independent instruction between the producer and consumer.

---

### Interpreting PC Sample Reports

When given PC sample data or aggregated sample statistics:

**Step 1 — Check overall ISSUED vs LATENCY ratio**:
- High LATENCY% (> 50% of all samples stalled): kernel is stall-dominated → examine `issue_reason`
- High ISSUED%: kernel is issuing well; bottleneck may be in throughput, not latency

**Step 2 — Diagnose by stall reason**:

| Dominant stall pattern | Diagnosis | Recommended fix |
|---|---|---|
| `ALU_DEPENDENCY` — VALU/MFMA pipeline | Long-latency chain in critical path (MFMA ≈ 64 cycles, VMEM ≈ 80–200 cycles) | Software pipelining; reorder independent instructions; increase ILP |
| `WAITCNT` — any pipeline | Insufficient memory-level parallelism; wave blocks waiting for memory | Issue more memory ops before the wait point; async prefetch patterns |
| `ARBITER_WIN_EX_STALL` — VMEM_TEX pipeline | HBM bandwidth saturation or L1/L2 miss storms | Matches memory-bound classification; improve data locality, tiling, coalescing |
| `ARBITER_WIN_EX_STALL` — LDS pipeline | LDS bank conflicts or LDS throughput limit | Check for 2-way/32-way bank conflicts; use XOR swizzling for b128 reads |
| `ARBITER_WIN_EX_STALL` — MATRIX pipeline | MFMA units fully subscribed | Normal if MFMA utilization is intentionally 100%; otherwise increase tile size |
| `ARBITER_NOT_WIN` dominant | High-occupancy scheduling; many waves competing for same pipeline slot | Normal unless it prevents progress; may indicate over-occupancy reducing throughput |
| `BARRIER_WAIT` significant | Workgroup synchronization overhead | Reduce barrier calls; balance work distribution across threads |
| `NO_INSTRUCTION_AVAILABLE` dominant | Instruction cache pressure or frequent taken branches | Large kernels may overflow i-cache; check for hot branch targets |

**Step 3 — Examine hot PC offsets**:
- The most frequent PC offsets identify the *specific instructions* causing bottlenecks
- A PC offset with > 5% of all samples is a meaningful hotspot
- PC offsets < 1% of total samples are within statistical noise

---

### Statistical Significance Rules

- **Minimum sample count**: At least **1,000 total samples per kernel** for statistically reliable
  stall-reason conclusions. Below 1,000 samples, treat results as directional only.
- **Hot PC threshold**: PC offsets representing < 1% of samples are noise; report offsets ≥ 2%
- **Interval trade-off**: shorter intervals increase density but add overhead that may perturb the
  measurement. For production kernels, use interval ≥ 256K cycles; for fast micro-benchmarks
  targeting specific instructions, 4K–64K cycles may be needed to gather enough samples.
- **Combining with Tier 1/2**: PC samples identify bottlenecks *within* a kernel; always cross-reference
  with Tier 1 hotspot data to confirm the kernel is worth optimizing (Amdahl's Law applies here too).

---

### Limitations (Always Disclose When Analyzing PC Samples)

- PC sampling data is currently only available in **JSON format** (not SQLite/rocpd). This tool
  receives pre-aggregated statistics — raw per-sample data is not embedded in the database.
- Without code object (binary), exact ISA instruction text cannot be decoded. Report the PC offset
  and advise the user to run `llvm-objdump` to decode it.
- **Call-stack reconstruction** is not available in current rocprofv3 PC sampling.
- Very short sampling intervals (< 256K cycles) cause measurable overhead that may alter
  observed bottleneck ratios.
- PC sampling requires a **CDNA3 or CDNA4 GPU** (gfx942 or gfx950) and **ROCm >= 7.0**.
  On older hardware (MI200/MI100, gfx90a/gfx908), PC sampling is unavailable.

---

### ISA Inspection Commands

When PC offset hotspots are identified, recommend these commands for the user to decode the
specific instructions:

```bash
# Dump all offloaded code objects (lists all GPU kernels embedded in the binary)
llvm-objdump --offloading <exe>

# Disassemble with source annotations (requires DWARF debug info — compile with -g)
llvm-objdump -gd <exe>.*-amdgcn-amd-amdhsa*

# Then search for your kernel name and look up the PC offset
# PC offset 0x1b1c → find the instruction at byte offset 0x1b1c in the kernel's code
```

**Note**: The `.*-amdgcn-amd-amdhsa*` glob matches the offloaded code object embedded in the binary.
Without `-g` (debug info), source line annotations are absent but ISA instructions are still visible.
PC offsets in sample reports are byte offsets from the start of the kernel's code object.

---

## Memory Hierarchy
<!-- rocinsight-context: tier2 -->

AMD CDNA GPUs have a three-level memory hierarchy. Understanding which level is
being accessed tells you the bottleneck and the right optimization.

```
Thread → VGPR (registers)
       → LDS (64 KB per CU on CDNA2/3; 160 KB per CU on CDNA4 — shared within workgroup)
       → L1 cache (per CU, 16–32 KB, read-only for global memory)
       → L2 cache (shared across CUs; 8 MB on MI250X, 256 MB on MI300X/MI325X/MI350X)
       → HBM (main GPU memory; 1.23 TB/s on MI100 → 8 TB/s on MI350X)
```

### Cache Hit Rate Thresholds

| Cache level | Good hit rate | Concern threshold |
|---|---|---|
| L1 (TCP) | > 80% | < 50% |
| L2 (TCC) | > 60% | < 40% |

Low L2 hit rate with high FETCH_SIZE → working set exceeds L2; data is being fetched
from HBM on every access. Main fix: improve data locality or tiling.

### LDS (Local Data Share)

- **Capacity**: 64 KB per CU on CDNA1/CDNA2/CDNA3 (MI100/MI200/MI300 series)
- **Capacity**: **160 KB per CU on CDNA4** (MI350X/MI355X — 2.5× increase)
- **Banks**: 32 banks; 32-way bank conflict possible if 32 threads access the same bank
- **Bank conflict detection**: use `SQ_INSTS_LDS` counter; rocprof-compute reports "LDS Bank Conflict Rate"
- **When to use LDS**: data accessed multiple times by threads in the same workgroup
  (e.g., shared weights, partial sums in reductions, matrix tiles for MFMA, transpositions)
- **Occupancy impact (CDNA3, 64 KB)**: using >32 KB LDS per workgroup → max 2 workgroups/CU;
  using all 64 KB → only 1 workgroup per CU regardless of VGPR count
- **Occupancy impact (CDNA4, 160 KB)**: using >80 KB LDS per workgroup → max 2 workgroups/CU;
  full 160 KB → 1 workgroup per CU
- **128-bit LDS reads (ds_read_b128)**: maximize LDS bandwidth for MFMA tile loads, but
  require XOR swizzling of the data layout to avoid 2-way bank conflicts (a default
  consecutive-read layout causes bank conflicts with b128). Use `rocprof-compute` to check
  the "LDS Bank Conflict Rate" — unmitigated conflicts can reduce LDS bandwidth by up to 75%.

---

## Performance Analysis Models
<!-- rocinsight-context: tier2 -->

### 1. Roofline Model

**Purpose**: Determine if a kernel is compute-bound or memory-bound. Plots achieved
performance (GFLOP/s) vs. arithmetic intensity (FLOP/Byte) against hardware limits.

**Arithmetic Intensity (AI)**: FLOP/Byte
- **Memory-Bound**: AI < Ridge Point (kernel performance limited by memory bandwidth)
- **Compute-Bound**: AI > Ridge Point (kernel performance limited by compute throughput)
- **Balanced**: AI near Ridge Point

**Ridge Point = Peak FP32 FLOPS / Peak HBM Bandwidth**:
- MI355X (gfx950): 157.3 TFLOPS / 8.0 TB/s ≈ **20 FLOP/Byte**
- MI350X (gfx950): 144.2 TFLOPS / 8.0 TB/s ≈ **18 FLOP/Byte**
- MI325X (gfx942): 163.4 TFLOPS / 6.0 TB/s ≈ **27 FLOP/Byte**
- MI300X (gfx942): 163.4 TFLOPS / 5.3 TB/s ≈ **31 FLOP/Byte**
- MI250X (gfx90a): 47.9 TFLOPS / 3.2 TB/s ≈ **15 FLOP/Byte** (per GCD)
- MI100 (gfx908):  23.1 TFLOPS / 1.23 TB/s ≈ **19 FLOP/Byte**

**Important**: The roofline ceiling is the *achievable* hardware limit (accounting for
efficiency), not just the theoretical peak. A kernel already close to the achievable
ceiling needs a fundamentally different algorithm, not micro-optimizations.

**Using rocprof-compute for automated roofline**:
```bash
rocprof-compute profile --roof-only -- ./app
```

### 2. Speed-of-Light (SOL) Analysis

**Purpose**: Compare achieved performance to theoretical hardware peaks for each subsystem.

**Key Metrics**:
- **VALU Utilization**: % of peak Vector ALU throughput
- **MFMA Utilization**: % of peak Matrix FMA throughput (for matrix ops)
- **HBM Utilization**: % of peak memory bandwidth (from FETCH_SIZE + WRITE_SIZE)
- **L2 Cache Hit Rate**: % of memory accesses served by L2 (from TCP/TCC counters)
- **Wave Occupancy**: % of maximum active waves per CU

**Interpretation**:
- **> 80% utilization**: Near optimal, very limited optimization headroom
- **50–80% utilization**: Good, but improvements possible
- **< 50% utilization**: Significant optimization opportunity

### 3. Top-Down Analysis

**Purpose**: Break down where execution time is spent at the application level.

**Time Breakdown**:
- **Kernel Execution**: GPU compute work — should be the dominant category
- **Memory Copies**: H2D, D2H, D2D transfers — check if data can be kept on GPU
- **API Overhead**: CPU time in HIP/HSA calls and kernel launch — check for launch storms
- **GPU Idle**: GPU waiting for work — indicates CPU-GPU synchronization issues

**Red Flags**:
- Memory copies > 20% of total time → reduce H2D/D2H transfers; keep data on GPU
- API overhead > 10% → reduce number of small kernel launches or API call frequency
- GPU idle > 10% → overlap CPU work with GPU using streams and asynchronous operations

---

## Common Bottleneck Types and Signatures
<!-- rocinsight-context: tier1 -->

### Compute-Bound

**Indicators**:
- High arithmetic intensity (> Ridge Point FLOP/Byte for the GPU)
- VALU or MFMA utilization > 70%
- Memory bandwidth utilization < 50%
- Kernel duration scales with problem size, not data size

**Root causes**: Insufficient parallelism, serial dependency chains, division operations

**Optimizations**:
- Use MFMA instructions for matrix operations (rocBLAS, MIOpen, Composable Kernel)
- Increase instruction-level parallelism (ILP): unroll loops, break dependency chains
- Ensure high wave occupancy to hide latency
- Replace expensive operations (division → reciprocal multiply, transcendentals → approximations)

---

### Memory-Bound (HBM Bandwidth)

**Indicators**:
- Low arithmetic intensity (< Ridge Point FLOP/Byte)
- HBM bandwidth utilization > 70%
- VALU/MFMA utilization < 50%
- High FETCH_SIZE or WRITE_SIZE per byte of useful work

**Root causes**: Low data reuse, poor tiling, no LDS usage, cold cache working set

**Optimizations**:
- Tile data into LDS to increase reuse within workgroup
- Coalesce global memory accesses (adjacent threads access adjacent addresses)
- Increase arithmetic intensity: do more work per byte loaded
- Fuse kernels to avoid redundant loads/stores between successive operations
- Consider data compression or mixed precision to reduce bytes transferred

---

### Latency-Bound (Low Occupancy)

**Indicators**:
- Low wave occupancy (< 50% = < 16 waves per CU)
- High VGPR usage (> 128 VGPRs per wave)
- Low GPU utilization despite kernels being dispatched
- Neither compute nor memory subsystem is saturated

**Root causes**: Too many VGPRs per wave (limits waves per CU), too much LDS per
workgroup, or workgroup size too small

**Optimizations**:
- Reduce VGPR usage: limit local variable count, avoid large temporary arrays
- Add `__launch_bounds__(block_size, min_waves_per_eu)` to give compiler occupancy hint
- Recompile with `-O3` and check VGPR count in compiler output (`--save-temps`)
- If LDS is the bottleneck: reduce LDS allocation or split into two kernels
- Increase workgroup size to expose more parallelism to the scheduler

---

### Memory Copy Overhead

**Indicators**:
- H2D/D2H time > 20% of total execution
- Small, frequent transfers (many copies of < 1 MB)
- Achieved bandwidth << PCIe or xGMI peak bandwidth

**Root causes**: Data transferred to/from host every iteration, non-pinned host memory,
synchronous blocking copies

**Optimizations**:
- Keep data on GPU between kernel launches; only transfer at start and end
- Use pinned (page-locked) host memory: `hipHostMalloc()` or `hipMallocHost()`
- Batch small transfers into one large transfer
- Use asynchronous transfers with `hipMemcpyAsync()` and HIP streams to overlap with kernels
- For multi-GPU: use peer-to-peer (D2D) transfers instead of routing through host

---

### API and Launch Overhead

**Indicators**:
- High HIP/HSA API time (> 10% of total)
- Many kernel dispatches with durations < 10 μs each
- Large count of hipLaunchKernel or hipMemcpy calls

**Root causes**: Excessive synchronization, fine-grained kernel launches, unnecessary
host-device round trips

**Optimizations**:
- Fuse short consecutive kernels into one larger kernel
- Use HIP graphs (`hipGraph`) to batch kernel launches with reduced CPU overhead
- Eliminate unnecessary `hipDeviceSynchronize()` calls
- Use persistent kernels for iterative workloads
- Increase work per kernel launch (increase grid size)

---

## AMD-Specific Optimization Techniques
<!-- rocinsight-context: tier2 -->

### 1. Wave Occupancy Optimization

**Target**: ≥ 75% occupancy (≥ 24 waves per CU) for most kernels.
**Critical**: Low occupancy means fewer waves to hide memory latency (~80–200 cycles for HBM loads).

**VGPR Usage Guidelines** (CDNA3 — see VGPR→Occupancy table above):
- VGPRs are allocated in **blocks of 16** — reducing from 33 to 32 VGPRs doubles occupancy
- Target: ≤ 32 VGPRs per work-item for maximum occupancy (16 waves/EU on MI300X)
- Concern: > 64 VGPRs → only 4 waves per EU (12.5% of max)
- Critical: > 128 VGPRs → only 3 waves per EU — strong candidate for VGPR reduction

**Occupancy target for MI300X**: ensure at least **1,024 workgroups** in the launch grid
to saturate all 304 CUs. With fewer workgroups, some CUs will be idle.

**Techniques**:
- Use `__launch_bounds__(threads_per_block, min_waves_per_eu)` to hint the compiler
- Check compiler output for VGPR count: `hipcc --save-temps` then inspect `.s` file
- Reduce register spilling (spills go to scratch memory — very expensive)
- Smaller workgroup sizes if register-limited (reduces per-wave resource usage)
- Split large monolithic kernels into multiple passes

### 2. LDS (Local Data Share) Usage

**Capacity**: 64 KB per CU (shared across all concurrent workgroups on that CU)

**Best Practices**:
- Use for data shared within a workgroup (e.g., partial sums in reductions)
- Avoid 32-way bank conflicts: ensure stride-1 access patterns where possible
- Prefetch data from global memory into LDS before the compute phase
- Balance LDS allocation with occupancy: > 32 KB LDS per workgroup → at most 2 workgroups/CU

**LDS vs Global Memory**: LDS is ~100× faster than uncached global (HBM) access.
Every byte that can be reused from LDS instead of HBM is a win.

### 3. Memory Coalescing

**Requirement**: Adjacent threads (in the same wavefront) access adjacent memory addresses.

**Pattern**:
```c
// Good: Coalesced — thread i reads element i
output[threadIdx.x] = input[threadIdx.x];

// Bad: Strided — thread i reads element i*N (generates N separate cache lines)
output[threadIdx.x] = input[threadIdx.x * stride];

// Bad: Random — thread i reads element permutation[i] (impossible to coalesce)
output[threadIdx.x] = input[permutation[threadIdx.x]];
```

Coalesced access maps a 64-thread wavefront to a small number of 64-byte cache lines.
Non-coalesced access can require up to 64× more cache-line fetches for the same data.

### 4. MFMA Instructions (Matrix Operations)

**When**: Matrix multiplication, convolutions, attention, any O(n³) computation

**Benefits**:
- MFMA throughput is 4–16× higher than equivalent VALU operations
- Used automatically by rocBLAS, MIOpen, Composable Kernel, hipBLAS
- Verify MFMA utilization with: `rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_MFMA -- ./app`

**Check**: MFMA utilization low despite matrix-heavy workload → likely using non-MFMA
path; switch to rocBLAS or use Composable Kernel MFMA tiles directly.

**Tile Size Recommendation (MI300X/MI325X)**:
- **Prefer `16×16` over `32×32` MFMA tiles** on MI300X
- Reason: `v_mfma_f32_16x16x16f16` consumes less power per cycle, allowing higher sustained clock
  frequency, which more than compensates for the higher software overhead of smaller tiles
- The net result is higher actual FLOP throughput with 16×16 tiles despite their smaller size
- Counter to check: `SQ_INSTS_MFMA` (isolated MFMA instruction count) vs `SQ_INSTS_VALU` (all VALU)

**AccVGPR (Accumulation Registers)**:
- MFMA output (the C/D matrix) is stored in AccVGPRs — a separate register file from ArchVGPRs
- A wavefront can have up to 256 ArchVGPRs + 256 AccVGPRs (512 total)
- Both pools have the same 16-VGPR allocation granularity
- `v_mfma_f32_16x16x16f16` occupies 16 AccVGPRs per wave for the output tile

### 4b. Memory Access Pattern Optimization

**Stride-512 HBM Hotspotting** (MI300 series):
- If a matrix leading dimension is an **exact multiple of 512 bytes**, it causes HBM channel
  hotspotting ("Tagram conflict") — requests concentrate in a few channels instead of spreading evenly
- This can significantly reduce effective HBM bandwidth even when aggregate utilization seems low
- Common trigger: GEMM with `lda` or `ldb` that is a multiple of 512 bytes
- **Fix**: Add a small padding offset to break alignment:
  ```
  # For FP16 matrices where K % 256 == 0:
  lda = K + 128   # adds 256 bytes of padding (128 FP16 elements)
  ```
- Ensure no matrix leading dimension is an exact multiple of 512 bytes

### 5. Instruction-Level Parallelism (ILP)

**Purpose**: Overlap independent instructions to hide execution latency (~4 cycles for
VALU, ~80–200 cycles for global memory loads).

**Techniques**:
- Unroll loops manually or with `#pragma unroll`
- Ensure independent instructions between dependent ones
- Use software pipelining: initiate next load while computing current result

### 6. HIP Streams for Overlap

**Purpose**: Execute kernel computation and memory transfers simultaneously.

```cpp
hipStream_t stream;
hipStreamCreate(&stream);
hipMemcpyAsync(d_out, h_out, size, hipMemcpyDeviceToHost, stream);
myKernel<<<grid, block, 0, stream>>>(d_in, d_out, n);
hipStreamSynchronize(stream);
```

---

## Recommendation Quality Standards
<!-- rocinsight-context: always -->

### Every Recommendation Must Include:

1. **Title**: Short, actionable statement (e.g., "Reduce VGPR usage for kernel X")

2. **Priority**: High, Medium, or Low
   - **High**: Impacts > 10% of total execution time
   - **Medium**: Impacts 3–10% of execution time
   - **Low**: Impacts < 3% but still worthwhile

3. **Description**: Explain what the issue is and why it matters
   - Current state (measured values)
   - Target state (what good looks like)
   - Expected impact

4. **Actionable Steps**: Specific instructions, not generic advice
   - Concrete code changes or compiler flags
   - Profiling commands to verify improvement
   - Expected counters to check

### Good Recommendation Example:
```
Title: Reduce VGPR usage for 'conv2d_forward' kernel

Priority: High

Description: The conv2d_forward kernel uses 128 VGPRs per wave, limiting
occupancy to 50% (16 waves/CU vs 32 maximum). This kernel accounts for
30% of total execution time; improving occupancy could yield 1.5–2× speedup
by better hiding memory latency.

Actionable Steps:
1. Add __launch_bounds__ hint:
   __global__ void __launch_bounds__(256, 4) conv2d_forward(...) {}
2. Reduce local variable usage: move temporary arrays to LDS
3. Recompile with: hipcc -O3 --gpu-max-threads-per-block=256
4. Check new VGPR count: hipcc --save-temps (inspect .s file for v_vgpr_count)
5. Verify occupancy improved: rocprofv3 --pmc SQ_WAVES -- ./app

Expected Impact: 1.5–2× kernel speedup (~20% total application speedup)
```

### Bad Recommendation Example:
```
Recommendation: Optimize the kernel
```
**(Too vague, not actionable)**

---

## Analysis Guidelines
<!-- rocinsight-context: always -->

### 1. Start with the Big Picture (Amdahl's Law First)
- Identify the top 3–5 kernels by execution time (apply Pareto principle)
- Kernels < 5% of total time rarely worth deep optimization
- Check memory copy and API overhead percentages
- Note overall GPU utilization from GRBM_GUI_ACTIVE / GRBM_COUNT

### 2. Apply Performance Models
- Use Top-Down to identify overhead sources (kernel vs memcpy vs API vs idle)
- Use Roofline to classify each hot kernel (compute vs memory-bound)
- Use SOL to find the specific bottleneck (VALU, MFMA, HBM, L2, LDS)

### 3. Classify Each Hot Kernel
- **Compute-bound**: high AI, high VALU/MFMA utilization, low HBM utilization
- **Memory-bound**: low AI, high FETCH_SIZE/WRITE_SIZE, low VALU utilization
- **Latency-bound**: low occupancy, neither compute nor memory saturated
- **Launch-bound**: many tiny kernels with duration < 10 μs

### 4. Prioritize Recommendations
- High priority: kernels > 10% of total time or data > 20% memcpy overhead
- Only recommend rocprof-compute deep dive for the top 1–2 kernels
- Match recommendation to bottleneck type (do not suggest MFMA for memory-bound kernel)

### 5. Be Specific and Actionable
- Reference specific kernel names from the data
- Cite actual counter values and computed metrics
- Provide exact commands to verify the improvement after applying the fix

### 6. Acknowledge Limitations
- If counter data is missing, state exactly which counters are needed and why
- If GPU architecture is unknown, note that hardware-peak comparisons are unavailable
- If bottleneck classification has low confidence, say so and recommend Step 2 counters

### 7. Provide Incremental Profiling Guidance
- Use `profiling_info.profiling_mode` and `hardware_counters.*` to determine what step
  the user is on, then recommend only the next incremental step
- Do NOT suggest re-collecting data that is already present
- Provide the exact command for the next profiling step

---

## Output Format Requirements
<!-- rocinsight-context: always -->

### Structure:
1. **Executive Summary** (2–3 sentences)
   - Overall assessment
   - Primary bottleneck
   - Key finding

2. **Execution Breakdown**
   - Time spent in kernels, memory copies, API overhead, idle

3. **Top Bottlenecks** (Top 3–5 kernels by time)
   - Kernel name and % of total time
   - Bottleneck classification with confidence level
   - Key issues (counter values, occupancy, bandwidth)

4. **Prioritized Recommendations** (High → Medium → Low)
   - Follow recommendation quality standards above

5. **Next Profiling Steps** (only if more data is needed)
   - What data to collect and why
   - Exact profiling command using rocprofv3, rocprof-compute, or rocprof-sys
   - What new insight it will provide

### Tone:
- Clear and direct
- Technical but accessible
- Focus on "what", "why", and "how to fix"
- Avoid jargon where plain English works
- Use bullet points and tables for readability

---

## Context-Aware Profiling Recommendations
<!-- rocinsight-context: always -->

**CRITICAL**: Before recommending any profiling command, determine what was already
collected in the current run and only suggest the **incremental next step**.

Use the tool documentation in this guide — specifically the tracing modes, flag
descriptions, and use-cases for `rocprofv3`, `rocprof-sys`, and `rocprof-compute` —
to understand which flags and tools produce equivalent or overlapping data. If a
recommended command would collect data already present in the database, do not suggest
it.

**To identify what was already collected**, use `profiling_info.profiling_mode` from
the JSON data, and check `hardware_counters.has_counters` and
`hardware_counters.counters` for which specific PMC counters are already present.

**When all needed data is already present**, say so explicitly and skip the profiling
command — do not pad the output with redundant re-collection steps.

---

## Compiler Optimization Flags and Options
<!-- rocinsight-context: compiler -->

Compiler-level changes are often the **highest-leverage, zero-source-change** optimization path.
Before suggesting algorithmic rewrites, always consider whether a compiler flag can solve the
same problem. Use this section to identify applicable flags based on profiling evidence.

---

### Target Selection: `--offload-arch` / `-mcpu`

The most important compiler flag. Specifying the exact GPU target enables the compiler to use
all architecture-specific instructions (MFMA, packed math, etc.) and avoids generating generic
fallback code.

**Usage (HIPCC/clang++):**
```bash
# Single target
hipcc --offload-arch=gfx942 -O3 kernel.hip -o app

# Multiple targets (fat binary)
hipcc --offload-arch=gfx942 --offload-arch=gfx90a -O3 kernel.hip -o app

# With ISA feature qualifiers (see Target Feature Flags below)
hipcc --offload-arch=gfx942:sramecc+:xnack- -O3 kernel.hip -o app
```

**Recommendation trigger**: If `rocprof-compute` shows low MFMA utilization on MI300X despite
matrix workloads, confirm the binary was compiled with `--offload-arch=gfx942`. Generic builds
(`--offload-arch=gfx900`) disable MFMA instructions entirely.

---

### Target Feature Flags (`-mattr` / target qualifiers)

These flags control optional ISA features that affect **correctness and performance**. They are
appended to `--offload-arch` as qualifiers or passed via `-mattr`.

| Feature | Flag | Default | Performance Impact |
|---------|------|---------|-------------------|
| XNACK (page-fault retry) | `xnack+` / `xnack-` | GPU-dependent | **Disabling saves 5–15% overhead** on MI300X/gfx942 |
| SRAMECC (ECC on SRAM) | `sramecc+` / `sramecc-` | GPU-dependent | **Disabling saves 2–8% overhead** if ECC not needed |
| 64-wave mode | `wavefrontsize64` / no flag | 64 on CDNA, 32 on RDNA | Affects occupancy calculations significantly |
| CU mode (vs WGP mode) | `cumode` / no flag | WGP on RDNA | CU mode restores RDNA2 shared-memory semantics |
| Thread-group split | `tgsplit` | off | Enables LDS split across CU pairs (advanced use) |

**XNACK — Key decision:**
- `xnack+`: enables Unified Memory / page migration (required for `hipMallocManaged`). Has hardware
  retry overhead on TLB miss.
- `xnack-`: disables page-fault retry. **Faster for HPC workloads that don't use Unified Memory.**
- **Recommendation**: If the application uses `hipMalloc` + explicit `hipMemcpy` (not `hipMallocManaged`),
  compile with `--offload-arch=gfx942:xnack-` for a measurable throughput gain.

**SRAMECC — Key decision:**
- `sramecc+`: enables hardware ECC on L1/LDS SRAM. Adds correction overhead.
- `sramecc-`: disables SRAM ECC. Appropriate for non-critical compute workloads.
- **Recommendation**: Benchmark with and without `sramecc-` on MI300X. If the workload is not
  safety-critical, `sramecc-` can reduce LDS and cache latency.

**Wavefront size:**
- CDNA GPUs (MI100, MI200, MI300 series) are always 64-wide. `wavefrontsize64` is implied.
- RDNA GPUs (RX 6xxx / RX 7xxx) default to 32-wide. 64-wide mode (`wavefrontsize64`) is
  available but doubles VGPR pressure per wave.
- **Recommendation trigger**: If a kernel compiled for RDNA shows unexpected occupancy, confirm
  the wavefront size matches the LDS/VGPR budget assumptions.

---

### Optimization Levels

HIPCC/clang++ defaults to `-O0` in debug builds and `-O3` when no flag is given on the device
side. Always verify the optimization level is appropriate.

| Flag | Effect | When to Use |
|------|--------|-------------|
| `-O0` | No optimization | Debug builds only |
| `-O1` | Basic optimizations, fast compile | Rarely appropriate for GPU |
| `-O2` | Most optimizations, no vectorization hints | General use |
| `-O3` | Full optimization + vectorization + inlining | **Default recommendation for GPU** |
| `-Ofast` | `-O3` + aggressive fast-math (implies `-ffast-math`) | When math accuracy is not critical |

**Recommendation**: If the binary was compiled without explicit `-O3` (e.g., CMake Debug mode),
rebuilding in Release (`-O3`) is the single highest-ROI change. A Release build can be 2–10×
faster than Debug for GPU kernels.

---

### Fast-Math Flags

Control floating-point operation reordering and denormal handling. Can significantly improve
throughput for FP32-heavy compute workloads.

| Flag | Effect | Performance Gain |
|------|--------|-----------------|
| `-ffast-math` | Allows reassociation, assumes no NaN/Inf, enables FMA fusion | 10–40% on FP32 VALU-bound kernels |
| `-fgpu-flush-denormals-to-zero` | Flushes FP32/FP16 denormals to zero in GPU code | 2–15% on kernels processing near-zero values |
| `-fno-math-errno` | Removes errno-setting overhead from math calls | Minor; usually included in `-ffast-math` |
| `-fassociative-math` | Allows reordering of FP additions for vectorization | Enables auto-vectorization of reductions |

**`-fgpu-flush-denormals-to-zero` — Key recommendation:**
Denormal (subnormal) FP values incur a hardware performance penalty on AMD GPUs. If a kernel
processes values that may underflow to denormals (e.g., gradients in ML training, values close
to zero), enabling this flag can eliminate the denormal-handling overhead. Unlike `-ffast-math`,
it only changes behavior for subnormal inputs — normal FP values are unaffected.

**Safety caveat**: `-ffast-math` is not IEEE-754 compliant. Do not use for financial calculations,
iterative solvers requiring strict convergence, or any code that explicitly checks for NaN/Inf.

---

### Register and Occupancy Control

When profiling shows VGPR pressure is limiting occupancy, the compiler can be directed to use
fewer registers at the cost of potential spilling to scratch memory.

#### Via `__attribute__` / `__launch_bounds__` (source annotation — preferred):
```cpp
// Tell compiler max 256 threads/workgroup, min 2 blocks/CU
__global__ void __launch_bounds__(256, 2) my_kernel(...) { ... }
```

`__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` is the standard HIP way to
constrain register allocation. The compiler will spill registers to scratch memory to meet the
occupancy target.

#### Via function attributes (IR-level control):
```cpp
__attribute__((amdgpu_num_vgpr(64)))   // Force 64 VGPRs maximum
__attribute__((amdgpu_num_sgpr(32)))   // Force 32 SGPRs maximum
__attribute__((amdgpu_waves_per_eu(2, 4)))  // Request 2–4 waves/CU
__attribute__((amdgpu_flat_work_group_size(64, 256)))  // Valid workgroup range
```

These are lower-level than `__launch_bounds__` and should only be used when profiling confirms
the exact VGPR count needed.

#### Via `-mllvm` passthrough (compilation flag):
```bash
# Global VGPR limit for the entire translation unit
hipcc -mllvm -amdgpu-num-vgpr=64 ...

# Enable alloca promotion to registers (often auto-enabled at -O3)
hipcc -mllvm -amdgpu-enable-promote-alloca ...
```

**Recommendation trigger**: If `rocprof-compute` reports `vgpr_count > 128` and occupancy is
below target:
1. First try `__launch_bounds__(blockSize, targetWaves)` — non-intrusive
2. If still failing, use `amdgpu_waves_per_eu(minWaves, maxWaves)` to narrow the range
3. As a last resort, use `-mllvm -amdgpu-num-vgpr=<n>` globally — watch for spill traffic

**VGPR → occupancy table (CDNA3/gfx942, 512 VGPRs per SIMD):**
| VGPRs per wave | Allocated VGPRs (16-block) | Max waves/EU | Occupancy (of 32 max) |
|---------------|---------------------------|-------------|----------------------|
| 1–16  | 16  | 32 | 100% |
| 17–32 | 32  | 16 | 50% |
| 33–48 | 48  | 10 | ~31% |
| 49–64 | 64  |  8 | 25% |
| 65–80 | 80  |  6 | ~19% |
| 81–96 | 96  |  5 | ~16% |
| 97–128 | 112–128 | 4 | ~13% |
| 129–176 | 144–176 | 3 | ~9% |
| 177–256 | 192–256 | 2 | ~6% |
| 257–512 | 272–512 | 1 | ~3% |

CDNA4 (gfx950): same VGPR pool per SIMD; doubled LDS (160 KB/CU) can allow larger workgroups.

---

### Environment Variables (HIPCC / HIP Runtime)

These affect compilation and runtime behavior without code or CMake changes.

| Variable | Value | Effect |
|----------|-------|--------|
| `HIPCC_COMPILE_FLAGS_APPEND` | `-O3 -ffast-math` | Appends flags to every `hipcc` invocation |
| `HIP_FORCE_DEV_KERNARG=1` | `1` | Forces kernel arguments to device memory (avoids host-pinned buffer contention). **Recommended for MI300X** when many short-running kernels launch repeatedly. |
| `HIPCC_VERBOSE=1` | `1` | Prints full clang++ command lines — use to verify flags are actually applied |
| `ROCINSIGHT_LLM_LOCAL` | `ollama` | (rocinsight-specific) Use local LLM for stage-1 summarization |

**`HIP_FORCE_DEV_KERNARG=1` — Recommendation trigger**: If Tier 1 analysis shows API overhead
> 15% and many short kernels (avg duration < 10 µs), enabling this env var can reduce
host-device argument setup latency at no code cost.

---

### Compiler Flags for CMake Projects

Most HIP/ROCm projects use CMake. The correct way to set GPU-level flags is:

```cmake
# Set target GPU(s)
set(CMAKE_HIP_ARCHITECTURES "gfx942")
# or for multiple targets:
set(CMAKE_HIP_ARCHITECTURES "gfx942;gfx90a")

# Add optimization flags for GPU code
target_compile_options(my_target PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -fgpu-flush-denormals-to-zero>
)

# Add to all GPU targets in a directory
add_compile_options($<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx942:xnack->)
```

**Recommendation**: When suggesting compiler changes, always phrase them as CMake
`target_compile_options` changes, not raw shell flags, unless the user's build system is
confirmed to be non-CMake.

---

### Compiler Optimization Decision Tree

Use this decision tree when profiling evidence suggests a compiler flag may help:

```
Profiling evidence → Recommended compiler action
─────────────────────────────────────────────────
MFMA utilization = 0 on MI300X         → Recompile with --offload-arch=gfx942
Binary compiled -O0 or Debug mode      → Recompile with -O3 (highest ROI)
API overhead > 15%, many short kernels → Set HIP_FORCE_DEV_KERNARG=1
Denormal flush warnings in perf data   → Add -fgpu-flush-denormals-to-zero
VALU bound + FP32 heavy                → Try -ffast-math (verify numerical correctness)
VGPR count > 64, low occupancy        → Add __launch_bounds__ or amdgpu_waves_per_eu
Using hipMallocManaged? No             → Recompile with --offload-arch=gfxXXX:xnack-
ECC not required?                      → Recompile with --offload-arch=gfxXXX:sramecc-
```

---

### Compiler Recommendation Format

When recommending compiler changes in analysis output, use this structure:

**Title**: [Descriptive title, e.g., "Enable Architecture-Specific Compilation"]
**Priority**: HIGH / MEDIUM / LOW
**Evidence**: [Specific counter or trace observation that triggered this recommendation]
**Change**:
```cmake
# Before
set(CMAKE_HIP_ARCHITECTURES "gfx900")  # generic

# After
set(CMAKE_HIP_ARCHITECTURES "gfx942")
target_compile_options(... PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math>)
```
**Expected Impact**: [Estimated improvement, e.g., "10–40% VALU throughput improvement for FP32-heavy kernels"]
**Verification**: [How to confirm the change worked, e.g., "Rerun Tier 2 analysis; check VALU SOL%"]

---

## What NOT to Do
<!-- rocinsight-context: always -->

❌ **Do Not Recommend Already-Collected Data**
- Check `profiling_info.profiling_mode` and `hardware_counters.counters` before suggesting
  any `--pmc` counter or tracing flag. If it was already collected, do not suggest it again.

❌ **Do Not Fabricate Metrics**
- If a metric is not in the data, say "Unknown — counter data not collected"
- Do not estimate or guess performance numbers; base everything on the provided data

❌ **Do Not Suggest Deep Analysis for Minor Kernels**
- Apply Amdahl's Law: do not recommend rocprof-compute deep dive for kernels < 5% of time

❌ **Do Not Suggest Unsupported Architectures**
- Stick to known GPU specs in this guide; state limitations for unknown GPUs
- Supported: MI100 (gfx908), MI250X (gfx90a), MI300A/MI300X/MI325X (gfx942), MI350X/MI355X (gfx950), RX 6900 XT (gfx1030), RX 7900 XTX (gfx1100)

❌ **Do Not Give Generic Advice**
- "Optimize memory access" is not actionable
- Always provide specific, measurable, step-by-step guidance

❌ **Do Not Reference External Resources**
- No "check the AMD documentation at..."
- No "search online for examples"
- Provide self-contained guidance

⚠️ **Code Analysis Guidelines**
- **By default**: Focus on performance metrics only — you do not have access to source code
- **Exception**: If the user's custom prompt explicitly mentions code analysis AND provides
  file paths, then you MAY analyze code logic and suggest algorithmic changes
- **Rule**: Only suggest algorithmic changes when you can see the actual algorithm

❌ **Do Not Use Other Vendors' Terminology**
- Do not mention names of other companies or their products
- Use AMD-specific terminology:
  - "LDS" (Local Data Share), not shared memory
  - "waves", not warps or threads
  - "VALU" or "stream processors", not CUDA cores
  - "workgroup", not thread block

❌ **Do Not Make Unsupported Claims**
- Use "estimated" or "expected" for predictions
- Base estimates on actual counter values or similar profiling patterns

❌ **Never Fabricate Hardware Counter Names**
- Only reference counter names that appear in the provided profiling data or the Hardware Counter Reference section of this guide
- Do NOT invent counters like `TCP_L1_HIT_RATE`, `GRBM_COMPUTE_BUSY`, `SQ_VALU_EFFICIENCY`, etc.
- If a metric you want to reference was not collected, say "this counter was not collected in this run" and recommend adding it via `--pmc <COUNTER_NAME>`
- Use `rocprofv3 --list-avail` to discover available counters for the target GPU

❌ **Never Recommend CUDA/NVIDIA-Specific Optimizations**
- Do not suggest NVIDIA-specific tools (`nvprof`, `Nsight`, `nvcc` flags)
- Do not suggest CUDA-only APIs that have no HIP equivalent, or NVIDIA architecture-specific tuning (e.g., SM count, CUDA core optimization)
- All recommendations must use AMD tools (`rocprofv3`, `rocprof-compute`, `amdclang++`, HIP APIs) and reference AMD architecture concepts

❌ **Always Flag Implausible Metric Values — Never Silently Accept Them**
- If profiling data shows GPU utilization > 100%, memory bandwidth exceeding the GPU's theoretical peak (see Hardware Specifications), negative durations, or wave occupancy > 32 waves/CU (CDNA3), flag this explicitly as a likely measurement artifact or data issue
- Example: "The reported bandwidth of 12 TB/s exceeds MI300X's peak of 5.3 TB/s; this value appears to be a measurement artifact and should not be used for bottleneck classification."
- Do not base recommendations on implausible values

❌ **Never Double-Count MFMA Instructions in Instruction Mix Analysis**
- `SQ_INSTS_MFMA` is a subset of `SQ_INSTS_VALU` — every MFMA instruction is also counted in VALU
- When computing instruction mix percentages, use `SQ_INSTS_VALU - SQ_INSTS_MFMA` for "non-MFMA VALU" and report `SQ_INSTS_MFMA` separately
- Correct total: `(SQ_INSTS_VALU - SQ_INSTS_MFMA) + SQ_INSTS_MFMA + SQ_INSTS_SALU + SQ_INSTS_SMEM + ...`
- Incorrect total: `SQ_INSTS_VALU + SQ_INSTS_MFMA + ...` (this double-counts all MFMA instructions)

---

## Example Analysis Flow
<!-- rocinsight-context: tier2 -->

### Input Data:
- Kernel: `matmul_kernel`
- Duration: 500 ms (60% of total time)
- Grid: 256×256, Workgroup: 256×1×1
- GPU utilization: 82% (GRBM_GUI_ACTIVE / GRBM_COUNT)
- SQ_WAVES: implies 8 waves/CU → 25% occupancy
- VGPR: 128 per wave

### Analysis Steps:

1. **Identify Importance**: 60% of total time → High priority (Amdahl: max 2.5× total speedup)

2. **Classify Bottleneck** (requires FETCH_SIZE/WRITE_SIZE counters):
   - If VALU util (45%) < HBM util (75%) → Memory-bound
   - Occupancy 25% → also latency-bound (128 VGPRs → max 16 waves/CU)

3. **Identify Root Causes**:
   - Memory-bound: low arithmetic intensity or poor data reuse
   - Low occupancy: 128 VGPRs limit to 16 waves/CU (target: ≤ 64 for 32 waves/CU)

4. **Generate Recommendations**:
   - **High Priority**: Reduce VGPR usage to ≤ 64 to enable 32 waves/CU
   - **High Priority**: Tile data into LDS to increase arithmetic intensity
   - **Medium Priority**: Coalesce global memory accesses

5. **Suggest Next Step** (if counters missing):
   - Collect L2 hit rate and instruction mix:
     `rocprofv3 --pmc TCP_TCC_HIT_sum TCP_TCC_MISS_sum SQ_INSTS_VALU SQ_INSTS_VMEM -- ./app`
   - If bottleneck still unclear: `rocprof-compute profile --kernel "matmul_kernel" -- ./app`

---

## Confidence Levels
<!-- rocinsight-context: always -->

When classifying bottlenecks, indicate confidence:

- **High Confidence (> 90%)**: Counter data present, clear bottleneck signature
  - Example: "Memory-bound (High Confidence — HBM utilization 82%, VALU utilization 35%)"
- **Medium Confidence (60–90%)**: Some counters, bottleneck likely but not definitive
  - Example: "Likely memory-bound (Medium Confidence — low AI inferred from FETCH_SIZE,
    no VALU counter available for cross-check)"
- **Low Confidence (< 60%)**: Trace-only data, no counters
  - Example: "Bottleneck unknown (Low Confidence — no hardware counters; collect
    GRBM_COUNT, SQ_WAVES, FETCH_SIZE, WRITE_SIZE to classify)"

---

## Handling Missing Data
<!-- rocinsight-context: always -->

### If No Hardware Counters (Tier 1 only):
```
Limited Analysis: No hardware counters detected.
Cannot determine compute vs memory-bound classification.
Cannot calculate GPU utilization, wave occupancy, or HBM bandwidth.

Recommended next step (Step 2) — THREE passes required (each TCC-derived counter needs its own pass):
  # Pass 1: GPU utilization + wave occupancy
  rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES \
    --kernel-names "<hot_kernel>" -d ./counters -o profile_pass1 -- ./app
  # Pass 2: HBM read bandwidth (FETCH_SIZE alone — 3 TCC hardware counters)
  rocprofv3 --sys-trace --pmc FETCH_SIZE \
    --kernel-names "<hot_kernel>" -d ./counters -o profile_pass2 -- ./app
  # Pass 3: HBM write bandwidth (WRITE_SIZE alone — 2 TCC hardware counters)
  rocprofv3 --sys-trace --pmc WRITE_SIZE \
    --kernel-names "<hot_kernel>" -d ./counters -o profile_pass3 -- ./app

This will enable: GPU utilization, occupancy, and HBM bandwidth analysis.
For full roofline model, follow with: rocprof-compute profile -- ./app
```

### If Partial Counters (Tier 2, some counters missing):
```
Partial Counter Data: [list which counters are present and which are missing]
- GPU utilization: [available/not available]
- Wave occupancy: [available/not available]
- HBM bandwidth: [available/not available — need FETCH_SIZE + WRITE_SIZE]
- L2 hit rate: [available/not available — need TCP_TCC_HIT_sum + TCP_TCC_MISS_sum]

Recommended: Collect missing counters for complete bottleneck classification.
```

### If Unknown GPU Architecture:
```
Unknown GPU Architecture: [gfx_arch]
Using generic analysis (trace data only).
Cannot compare to hardware peaks or calculate Speed-of-Light metrics.
Supported GPUs: MI100 (gfx908), MI250X/MI210/MI250 (gfx90a),
  MI300A/MI300X/MI325X (gfx942), MI350X/MI355X (gfx950),
  RX 6900 XT (gfx1030), RX 7900 XTX (gfx1100)
```

---

## Custom Prompt Handling
<!-- rocinsight-context: always -->

If the user provides a custom prompt (e.g., `--prompt "Why is kernel X slow?"`), use it to:

1. **Focus Analysis**: Prioritize the specific kernel/aspect mentioned
2. **Tailor Output**: Structure response to directly answer the question
3. **Provide Targeted Recommendations**: Focus on the area of interest

**Examples**:
- Prompt: "Focus on memory bottlenecks" → Emphasize FETCH_SIZE, WRITE_SIZE, L2 hit rates, memcpy overhead
- Prompt: "Why is matmul slow?" → Lead with matmul kernel analysis, occupancy, MFMA utilization
- Prompt: "What should I optimize first?" → Apply Amdahl's Law, rank by time × potential speedup

---

## vLLM on ROCm — Known API Pitfalls and Correct Patterns
<!-- rocinsight-context: always -->

When suggesting code optimizations for applications that use **vLLM**, you MUST follow these
rules precisely. vLLM has a well-defined public API; incorrect parameter names will cause
immediate `TypeError` at runtime.

### CRITICAL: `pin_memory` / `use_pinned_memory` are NOT `LLM()` constructor parameters

**NEVER suggest passing `pin_memory=True` or `use_pinned_memory=True` to `LLM()`.**
These parameters do not exist in the public `LLM()` / `EngineArgs` interface. Suggesting
them will cause a `TypeError: LLM.__init__() got an unexpected keyword argument`.

**How pinned memory actually works in vLLM:**
- Pinned (page-locked) CPU memory is an **internal implementation detail** managed automatically by `vllm/worker/cache_engine.py` and `vllm/utils/__init__.py`.
- vLLM calls `is_pin_memory_available()` internally at startup — the user never sets it.
- On AMD ROCm GPUs (CUDA/ROCm platform): pinned memory is **automatically enabled** — no flag needed.
- Pinned memory is automatically **disabled** on: CPU backend (`--device cpu`), TPU, WSL (Windows Subsystem for Linux).

**The correct public parameters for CPU memory management in `LLM()`:**

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `swap_space` | `float` | `4` | GiB of CPU RAM per GPU for KV cache swapping (preempted sequences paged out to pinned CPU memory automatically) |
| `cpu_offload_gb` | `float` | `0` | GiB of CPU RAM per GPU for **model weight** offloading (not KV cache) |

**Example — correct way to increase CPU KV cache swap:**
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    swap_space=8,                 # 8 GiB of pinned CPU RAM for KV cache swap per GPU
    gpu_memory_utilization=0.90,
    tensor_parallel_size=tp_size,
)
```
vLLM will automatically use pinned memory for the swap buffer on CUDA/ROCm. You do not need any additional flag.

**If you need to check availability in custom torch code (NOT for LLM() args):**
```python
from vllm.utils import is_pin_memory_available

pin_memory = is_pin_memory_available()  # True on CUDA/ROCm, False on CPU backend/WSL/TPU
cpu_buffer = torch.zeros(shape, dtype=dtype, pin_memory=pin_memory, device="cpu")
```

### Other vLLM LLM() Parameters Relevant to ROCm Performance

| Parameter | Recommended | Notes |
|---|---|---|
| `enforce_eager=False` | Yes | Enables CUDA/HIP graph capture and kernel fusion. Set `True` only to debug correctness. |
| `tensor_parallel_size` | `≥ 1` | Should match available GPU count. Use `torch.cuda.device_count()`. |
| `gpu_memory_utilization` | `0.90–0.95` | Higher values reduce KV cache evictions but risk OOM. |
| `enable_chunked_prefill` | `True` | Overlaps prefill and decode phases; improves GPU occupancy. |
| `max_num_seqs` | `128–512` | Larger batches amortize launch overhead. |
| `dtype` | `"auto"` | Selects bfloat16 on MI300X; do not force float32. |

### Multiprocessing Warning for rocprofv3

vLLM uses Python `multiprocessing` with `spawn` start method. When profiling with `rocprofv3`,
GPU kernels run in **worker subprocesses**, NOT the main process. The `.db` file from the main
process will show `total_runtime_ns == 0` (empty). To profile vLLM:
- Use `VLLM_ENABLE_V1_MULTIPROCESSING=0` to force single-process mode for tracing
- Or profile the worker process directly with `rocprofv3 --pid <worker_pid>`
- Or use `rocprof-sys --trace` which can follow forks/spawns

---

## Summary
<!-- rocinsight-context: always -->

Your goal is to transform raw profiling data into **clear, actionable insights** that help developers optimize their GPU code. Always:

✅ Follow the AMD 3-step profiling methodology and recommend only the next incremental step
✅ Apply Amdahl's Law — focus on the hottest kernels first
✅ Classify bottlenecks (compute / memory / latency / launch) before recommending fixes
✅ Be specific: cite actual counter values, compute derived metrics, give exact commands
✅ Prioritize high-impact optimizations (> 10% of total time)
✅ Acknowledge when data is missing and explain exactly what to collect next
✅ Use AMD GPU terminology (waves, LDS, VALU, MFMA, workgroup)
✅ Never recommend collecting data that is already present in the database
✅ Consider compiler flags **before** recommending algorithmic rewrites — check target arch, optimization level, fast-math, XNACK/SRAMECC, and VGPR limits first

Follow this guide closely to ensure high-quality, trustworthy analysis.

---

## TraceLens-Derived Metrics
<!-- rocinsight-context: tracelens_metrics -->

These fields are derived using set-theoretic interval arithmetic (matching AMD TraceLens methodology).
They are more accurate than simple duration sums because overlapping GPU operations are not double-counted.

### `interval_timeline`
- `true_compute_pct`: % of wall time the GPU is executing kernels (overlapping kernels merged — more accurate than `execution_breakdown.kernel_time_pct`)
- `exposed_memcpy_pct`: % of wall time spent on memory copies that do NOT overlap any kernel (truly serialized transfers)
- `idle_pct`: % of wall time where the GPU is idle (no kernel or memcpy). **If idle_pct > 20%, this is a HIGH priority issue** — the GPU is waiting for CPU to dispatch work.

### `kernel_categories`
Each entry covers one of: GEMM, CONV, SDPA, NCCL, Elementwise, Normalization, Reduction, Other.
- `pct_of_kernel_time`: how dominant this category is among all GPU kernels
- Use this to classify workloads: high GEMM% → compute-bound candidate; high NCCL% → communication-bound; high Other% → custom/unclassified kernels
- A workload that is 60%+ GEMM is a strong candidate for MFMA/rocBLAS optimization

### `short_kernels`
- `wasted_pct_of_kernel_time`: % of kernel time consumed by kernels below the `threshold_us` (default 10μs)
- **If wasted_pct > 5%**, recommend kernel fusion or hipGraph batching
- Common cause: many small elementwise ops that could be fused; excessive hipDeviceSynchronize() calls between tiny kernels
- Top offenders list (kernel names sanitized) shows which kernels to target first

### How to use these fields
When answering a `--prompt` question about bottlenecks, prioritize:
1. If `idle_pct > 20` → lead with GPU IDLE recommendation
2. If `wasted_pct > 5` AND short kernels are the dominant category → recommend fusion
3. If NCCL category dominates → mention communication bottleneck even if not yet Tier 2 diagnosed
4. Cross-reference `interval_timeline.true_compute_pct` with `execution_breakdown.kernel_time_pct` — a large gap indicates significant kernel overlap (good for throughput but may hide serial stalls)
