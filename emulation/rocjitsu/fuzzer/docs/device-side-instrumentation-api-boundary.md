# Device-Side Instrumentation Mechanics and ROCm/HIP Boundary

This note summarizes cuFuzz's current CUDA/NVBit instrumentation path and the
API boundary that would make a ROCm/HIP backend feasible. The goal is to keep
the fuzzer and coverage semantics independent from the binary surgery backend.

## Current CUDA/NVBit Information Flow

1. AFL starts the target with `AFL_PRELOAD=.../cufuzz_cov.so`.
2. `nvbit_at_init` maps AFL's shared `trace_bits` bitmap via `__AFL_SHM_ID`.
3. `nvbit_tool_init` allocates device-visible coverage state:
   - `exec_cov_bb`: 32-bit device edge counters.
   - `exec_cov_bb_quantized`: byte-sized staging map.
   - `prev_cov_bb`: per-thread previous-basic-block state.
4. On each CUDA kernel launch, `nvbit_at_cuda_event` gets the launched
   `CUfunction`, instruments it once if needed, and enables the instrumented
   version for that launch.
5. Instrumentation walks NVBit's static CFG for the kernel and related device
   functions. For each basic block, cuFuzz inserts one call before the first
   instruction:

   ```c++
   record_coverage_edge_count(bb_id, exec_cov_bb, prev_cov_bb)
   ```

6. On the device, the injected hook computes an AFL-style edge id:

   ```text
   device_edge = 32768 + ((prev_cov_bb[tid] ^ bb_id) % 32768)
   ```

   It updates `prev_cov_bb[tid] = bb_id >> 1`, then one active lane per warp
   atomically increments `exec_cov_bb[device_edge]`.
7. At the run boundary, cuFuzz quantizes the 32-bit device counters into AFL
   byte buckets, merges them with host-side `trace_bits`, and writes the merged
   map back to AFL's shared bitmap.

The lower half of the 64K map is used for host AFL coverage. The upper half is
reserved for device edges. The fuzzer only sees the final merged bitmap.

## Coverage Map Shape

The AFL-facing map is a byte counter array, not a bitset:

```text
trace_bits[65536] : uint8_t
  [0..32767]      host coverage
  [32768..65535]  device coverage
```

Device hooks update raw counters plus per-work-item edge state:

```text
device_counters[65536] : uint32_t
previous_bb[work_item] : previous basic-block id
```

At the run boundary, the host buckets `device_counters` into AFL-style bytes and
merges them into the upper half of `trace_bits`.

## Hooks Inserted

cuFuzz inserts exactly one hook at each basic-block entry, not at every
instruction. The hook's logical contract is:

```text
record_edge(static_basic_block_id, coverage_counter_map, previous_edge_state)
```

The CUDA implementation realizes that contract as a device function call
inserted by NVBit into SASS-level code. The higher-level coverage logic should
not depend on whether the backend emits a call, inlines an instruction sequence,
or patches control flow through a trampoline.

## Divergence and Mask-Sensitive Coverage

The baseline edge contract is active-mask gated: a hook executes only for lanes
that are active at that basic-block entry, and only those lanes update their
`previous_bb[work_item]` state. In divergent control flow, lanes that are masked
off for one path keep their previous-BB value until their own path executes.

The cuFuzz-compatible counter policy still collapses some SIMT detail: it uses
one selected active lane to increment the edge counter for a warp/wave hook. That
is enough for a first DBI slice, but it means a reconverged block reached by
lanes with different previous BBs may contribute only one counted edge.

The logical extension point is the device edge hook's coverage key and counter
policy. A ROCm backend can preserve the baseline `previous_bb ^ bb_id` key, or
extend it with a compact representation of `EXEC`, active-lane popcount, or
distinct previous-BB groups when divergence-sensitive feedback becomes useful.
The DBI PoC keeps this as a planning/tested semantic before emitting the full
AMDGCN sequence.

## Manual HIP PoC Stage

Before building dynamic AMDGCN instrumentation, validate the runtime contract
with explicit HIP source-level hooks. The isolated PoC in
[../manual-coverage-poc/](../manual-coverage-poc/) does this by splitting the
system into:

- An AFL-instrumented host driver that maps `__AFL_SHM_ID`, loads a HIP code
  object, launches the kernel, copies device counters back, quantizes them, and
  merges them into `trace_bits`.
- A HIP kernel with manual `record_edge(...)` calls at basic-block-like points.
  These calls stand in for future AMDGCN-inserted hooks.

This confirmed the useful boundary: the host coverage runtime can be developed
and tested without any binary instrumentation. The future AMDGCN backend only
needs to reproduce the device-side `record_edge` side effect.

## AFL++ vs libFuzzer

A ROCm/HIP port should keep AFL++ as the outer fuzzing engine. The important
constraint is not that HIP can be compiled by LLVM/Clang; it is that many
targets may be assembly kernels or precompiled code objects where compiler
inserted SanitizerCoverage is unavailable. For those targets, the framework
must collect feedback through dynamic binary instrumentation and then report it
to the fuzzer after each input.

AFL++ fits that shape better than libFuzzer:

- AFL++ already has a process-oriented execution model, corpus management, crash
  handling, persistent mode, and a shared `trace_bits` bitmap that an external
  coverage runtime can update.
- The coverage runtime can merge DBI-collected AMDGCN counters into the same
  bitmap as host coverage before AFL++ decides whether an input is interesting.
- Separate processes make it easier to isolate GPU runtime state and to run
  incompatible sanitizers or checking tools without forcing them into the same
  address space as the coverage collector.

libFuzzer is still useful for narrow, source-available unit-style harnesses, and
`LLVMFuzzerTestOneInput` harnesses can be reused. It is a weaker default for the
full framework because its main advantages come from in-process execution and
compiler-inserted coverage, while the ROCm assembly-kernel path needs
out-of-process DBI feedback.

## Recommended API Boundary

Split the system into four layers:

```text
AFL / fuzz policy
  Owns testcase scheduling and reads trace_bits.

Coverage runtime
  Owns AFL shared-memory mapping, device coverage buffers, reset, sync,
  quantization, and host/device bitmap merge.

Kernel/module runner
  Owns loading HIP code objects or observing native launches, passing coverage
  state to kernels, and defining input/run boundaries.

Device instrumentation backend
  Owns code discovery, CFG/basic-block recovery, binary patching, hook emission,
  and launch/module integration for CUDA SASS or AMDGCN.
```

The backend should expose semantic operations, not NVBit-shaped operations:

```c++
struct InstrumentationSite {
  FunctionId function;
  BasicBlockId basic_block;
  InstructionAddress entry;
  uint32_t stable_bb_id;
};

struct DeviceInstrumentationBackend {
  void on_module_loaded(ModuleHandle module);
  void on_kernel_launch_begin(KernelHandle kernel, LaunchInfo launch);
  void on_kernel_launch_end(KernelHandle kernel, LaunchInfo launch);

  std::vector<InstrumentationSite> enumerate_basic_block_entries(
      FunctionHandle function);

  void apply_edge_hook(InstrumentationSite site, DeviceCoverageState state);
  void make_instrumented_code_active(KernelHandle kernel);
};
```

For ROCm/HIP, `apply_edge_hook` may be implemented by AMDGCN assembly patching
rather than a call insertion API. The core runtime should only require that the
patched code performs the `record_edge` semantics.

The manual HIP PoC uses a trivial backend: the hooks are already present in the
source. That lets the coverage runtime and runner harden before the real backend
has to solve CFG recovery, relocation, metadata, and patching.

## ROCm/HIP Backend Responsibilities

A ROCm/HIP backend needs equivalents for these NVBit conveniences:

- Runtime interception: observe HIP or HSA module/code-object loading and kernel
  launch begin/end.
- Code discovery: map a launchable kernel symbol to its AMDGCN code object and
  any device functions that should be instrumented with it.
- CFG recovery: identify basic-block entry instructions and assign stable ids.
- Hook emission: insert either a device call, trampoline, or inline AMDGCN
  sequence that updates the coverage map.
- State plumbing: make coverage-map and previous-BB pointers available to the
  patched code, either through kernel arguments, device globals, constant memory,
  or another ABI-stable mechanism.
- Synchronization: know when all instrumented GPU work for an input is complete
  before the host quantizes and merges coverage.

The PoC currently chooses explicit kernel arguments for state plumbing because
that is the simplest testable contract. A dynamic instrumentation backend may
choose a different mechanism, but it should preserve the same logical inputs:
coverage counters and per-work-item previous-BB state.

## CUDA vs ROCm/HIP Constraints to Keep Out of the Core

- NVBit provides CFGs, insertion APIs, and per-launch enable/disable behavior.
  A ROCm backend may need to own code-object rewriting, relocation handling,
  symbol injection, and code-cache invalidation itself.
- CUDA warps are 32 lanes. AMD wavefront width can be target- or mode-dependent,
  so the backend should not hardcode warp-level assumptions in shared code.
- Calling a device helper is convenient with NVBit. On AMDGCN, an inline update
  sequence may be safer if function-call ABI, register pressure, or metadata
  updates are painful.
- AMDGCN patching must preserve code-object metadata such as SGPR/VGPR usage,
  scratch requirements, LDS usage, kernarg layout, and relocation information.
- Launch-boundary detection should be backend-specific. Persistent-mode
  start/end should be represented as logical iteration events, not as a CUDA
  notification-kernel name baked into shared code.

## Clean Porting Target

The shared cuFuzz logic should ask the backend for only this guarantee:

```text
For every executed instrumented device basic block, update a device coverage
counter at a stable edge-derived index. At an input boundary, all updates are
visible to the host so they can be merged into AFL trace_bits.
```

Everything below that line is backend-specific binary surgery. Everything above
that line is fuzzer policy and coverage-map management.
