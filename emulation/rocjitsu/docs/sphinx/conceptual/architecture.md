---
myst:
    html_meta:
        "description": "Architecture and component layers of rocJITsu, covering the simulation engine, configuration, code, ISA, VM, and KMD layers plus the three execution strategies."
        "keywords": "rocJITsu, ROCm, architecture, simulation, DBT, DBI, GPU emulation, PDES, layers"
---
# Architecture and component layers

rocJITsu is organized into layered components that together form a
full-system GPU emulation toolkit. Each layer has a well-defined
responsibility and maps to a distinct source directory. The three
execution strategies --- simulation, dynamic binary translation (DBT),
and dynamic binary instrumentation (DBI) --- compose different subsets
of these layers to serve different use cases.

## Layer diagram

The following diagram shows the major layers from bottom (simulation
infrastructure) to top (unmodified user binaries).

```{mermaid}
flowchart
   A["HIP / ROCR / RCCL"] --> B["KMD emulation"]
   B --> C["VM / hardware model"]
   C --> D["ISA layer"]
   D --> E["Code layer"]
   E --> F["Configuration"]
   F --> G["simdojo PDES engine"]
```
Each layer in the stack depends on the layers below it, with the
configuration and simulation engine providing the foundation.

## Component layers

### simdojo PDES engine

**Source directory:** `lib/simdojo/`

The simdojo library provides the Parallel Discrete Event Simulation
(PDES) framework on which the GPU virtual machine is built. It supplies
the component model, topology builder, event queue, and multi-threaded
execution engine. The engine runs a barrier-based epoch loop with Lower
Bound on Time Stamp (LBTS) synchronization: each epoch drains
cross-partition events, processes events up to the global LBTS,
publishes a local next timestamp, and arrives at a `std::barrier` whose
completion function computes the new LBTS.

The simulation engine also includes a wall-clock pacing controller that
maps simulation ticks to real time, supporting real-time or scaled-speed
operation.

### Configuration

**Source directory:** `lib/rocjitsu/src/rocjitsu/config/`

GPU topology --- XCDs, shader engines, compute units, caches, and memory
--- is described declaratively in JSON files validated against
FlatBuffers schemas. Range expansion (for example, `xcd[0:8]`) and
pattern-based link wiring allow compact multi-GPU configurations. The
configuration layer builds the full component hierarchy from a single
JSON file. For details on the JSON format, see
[JSON topology configuration](json-configuration.md).

### Code layer

**Source directory:** `lib/rocjitsu/src/rocjitsu/code/`

This layer handles loading, decoding, and transforming GPU code objects:

-   **Executable loading** --- Loads x86 HIP fat binaries, extracts
    Clang offload bundles, and parses AMD GPU HSA device ELFs.
-   **Basic block analysis** --- Constructs control-flow basic blocks
    from decoded instruction streams.
-   **DBT** --- The dynamic binary translator performs cross-ISA code
    object recompilation using auto-generated legalization tables and
    encoding translators (under `code/dbt/`).
-   **DBI** --- The code object patcher instruments existing binaries
    with probe functions for profiling, tracing, and analysis (under
    `code/patch/`).
-   **Register analysis** --- Backward liveness analysis and def-use
    chains over kernel CFGs provide free-register search used by both
    DBT and DBI (under `code/analysis/`).

### ISA layer

**Source directory:** `lib/rocjitsu/src/rocjitsu/isa/`

Instruction decoding and execution for AMD GPU architectures (CDNA1
through CDNA4, RDNA1 through RDNA4, gfx1250) plus RISC-V. Most files in
this layer are auto-generated from AMD Machine-Readable ISA XML
specifications by the amdisa Python codegen pipeline
(`lib/python/amdisa/`). The layer contains per-architecture
subdirectories with decoders, encoding structs, and execution bodies,
along with hand-written files for address calculation, matrix math, and
ISA-specific traits.

### VM and hardware model

**Source directory:** `lib/rocjitsu/src/rocjitsu/vm/amdgpu/`

Models the GPU hardware pipeline as simdojo components:

-   **Command processor** --- Monitors doorbells, fetches AQL packets,
    parses kernel descriptors, and dispatches workgroups to compute
    units. Handles SDMA packets.
-   **Compute unit** --- Executes wavefronts, managing SGPR/VGPR
    register files, LDS, and scratch memory.
-   **GPU memory** --- VRAM model with per-process VMID page tables and
    support for both passthrough mode and daemon-mode shared `memfd`
    mappings.
-   **Cache hierarchy** --- L1 vector cache, L1 scalar cache, L2 cache,
    and memory-side cache with MTYPE awareness.
-   **Execution plugins** --- Pluggable hooks for runtime analysis. See
    [Execution plugin system](execution-plugins.md).

For a deeper discussion of the hardware model, see
[GPU virtual machine design](gpu-vm-design.md).

### KMD emulation

**Source directory:** `lib/rocjitsu/src/rocjitsu/kmd/linux/`

On Linux, `librocjitsu_kmd.so` intercepts `open`, `ioctl`, `mmap`,
`fopen`, and `close` via `LD_PRELOAD` to route `/dev/kfd` operations to
the simulated GPU. In local mode, a `SimulatedDriver` handles KFD ioctls
in-process. In daemon mode, a `RemoteDriver` RPC stub forwards
operations over a Unix socket to the daemon process, with GPU memory
shared using `memfd` file descriptors passed via `SCM_RIGHTS`.

### Shared utilities

**Source directory:** `lib/util/`

A header-heavy C++ utility library providing bit manipulation,
floating-point format conversions, SIMD helpers, arena allocators,
spinlocks, intrusive lists, logging, and exception types. Both rocJITsu
and simdojo depend on it.

## Execution strategies

rocJITsu supports three execution strategies. Each uses a different
combination of the layers described above.

### Simulation

Full ISA emulation on a CPU-hosted GPU model via `LD_PRELOAD`
interposition. No physical GPU or kernel module is required. This
strategy uses the complete layer stack: the simdojo engine drives the VM
hardware model, which dispatches wavefronts to compute units that decode
and execute instructions through the ISA layer. The KMD emulation layer
intercepts host runtime calls so that unmodified HIP, ROCR, and RCCL
binaries run transparently against the simulated GPU.

The VM is created using `rj_vm_create` with mode `RJ_VM_MODE_LOCAL`
(single-process) or `RJ_VM_MODE_DAEMON` (multi-process with shared
`memfd` memory). Simulation can be advanced tick by tick with
`rj_vm_step` or run to completion with `rj_vm_run`.

### Dynamic binary translation

Cross-ISA code object recompilation that lets applications compiled for
one GPU architecture run on another. DBT operates primarily in the code
layer: the binary translator classifies every source instruction using
auto-generated legalization tables (IDENTITY, SUBSTITUTE, LOWER, or
EXPAND), performs encoding translation and semantic translation (such as
waitcnt splitting and MFMA-to-WMMA conversion), translates kernel
descriptor ABIs, and inserts code caves for expanded instruction
sequences. The translated code object is then loaded either by the
simulated VM or by real hardware through an HSA tools hook.

Translation is exposed through the `rj_code_translate` function in the
public C API, which takes a source code object and a
`rj_code_dbt_options_t` specifying the guest and host architectures.

### Dynamic binary instrumentation

Runtime instrumentation of GPU kernels for profiling, tracing, and
analysis. DBI operates in the code layer's patching subsystem,
intercepting HSA code-object loads and patching native GPU code in
place. It uses the register liveness analysis from the analysis layer to
find safe scratch registers and the code cave mechanism to insert
instrumentation sequences. DBI does not require the full simulation
stack --- it can run on real hardware through the HSA tools interface.

## Project layout

``` text
lib/
  simdojo/              Simulation engine (PDES framework)
  rocjitsu/
    include/rocjitsu/   Public C API
    src/rocjitsu/
      vm/amdgpu/        GPU hardware model (CP, CU, caches, memory)
      isa/arch/amdgpu/  ISA execution (10 targets, mostly autogenerated)
      kmd/linux/        KFD driver emulation + LD_PRELOAD interposer
      code/             Code object loader, basic block analysis
      code/dbt/         Dynamic binary translator
      code/patch/       Code object patcher, spill manager
      code/analysis/    Register liveness and def-use analysis
      config/           JSON/FlatBuffers configuration
  util/                 Shared utilities
  python/amdisa/        ISA codegen pipeline
tools/rocjitsu/         CLI (local, daemon, attach modes)
configs/                GPU topology JSON files
schemas/                FlatBuffers schemas
```
