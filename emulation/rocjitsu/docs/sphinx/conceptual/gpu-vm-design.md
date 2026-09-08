---
myst:
    html_meta:
        "description": "GPU virtual machine design in rocJITsu, covering SoC component hierarchy, command processor dispatch, wavefront execution, memory hierarchy, MTYPE coherency, and supported ISA targets."
        "keywords": "rocJITsu, GPU, virtual machine, SoC, XCD, shader engine, compute unit, command processor, wavefront, memory hierarchy, cache, MTYPE, coherency, ROCm"
---
# GPU virtual machine design

rocJITsu models a complete GPU system-on-chip (SoC) as a hierarchy of
simulation components. This page describes how those components are
organized, how work flows from an AQL dispatch packet through the
command processor to wavefront execution on compute units, and how the
simulated memory hierarchy maintains coherency.

For information on how the SoC topology is declared in JSON, see
[Configure a simulated GPU topology](../how-to/configure-topology.md). For details
on how the KMD emulation layer routes host-side ioctls to the simulated
driver, see [Architecture and component layers](architecture.md).

## SoC component hierarchy

The virtual machine root owns a single SoC. The SoC contains one or more
accelerator complex dies (XCDs), an I/O die (IOD), and GPU memory. Each
XCD contains a command processor (CP), shader engines (SEs), an L2
cache, and a memory-side cache. Each shader engine contains an array of
compute units (CUs).

```{mermaid}
flowchart LR
  SoC --> XCD
  SoC --> IOD
  SoC --> GpuMemory
  XCD --> CP[CommandProcessor]
  XCD --> SE[ShaderEngine]
  XCD --> L2[L2 Cache]
  SE --> CU[ComputeUnit]
```
The JSON configuration defines this hierarchy declaratively. Range
expansion syntax (for example, `xcd[0:8]`) creates multiple instances,
and pattern-based link wiring connects component ports. The
`rj_vm_gpu_info_t` structure returned by `rj_vm_gpu_info` reports
topology dimensions such as `num_xcc`, `num_shader_engines`,
`num_cu_per_sh`, and `simd_per_cu` back to the host runtime.

## Command processor and AQL dispatch

The command processor monitors doorbells, fetches AQL packets from
hardware queue ring buffers, and parses kernel descriptors. When a
compute-dispatch packet arrives, the CP reads the kernel descriptor
embedded in the code object, determines the grid and workgroup
dimensions, and dispatches workgroups to available CUs across the shader
engines within its XCD.

The CP also handles SDMA packets for host-to-device and device-to-device
copy, fill, GCR (global cache request), and HDP flush operations. A
completion tracker inside each XCD monitors per-dispatch workgroup
retirement and fires completion signals in submission order. When a
hardware queue becomes idle, the tracker writes the queue-inactive
signal.

In the public API, `rj_vm_run` drives the simulation to completion,
while `rj_vm_step` advances the simulation by one tick. The VM creation
mode (`RJ_VM_MODE_LOCAL` or `RJ_VM_MODE_DAEMON`) controls whether the CP
operates in-process or as part of a daemon serving remote clients.

## Wavefront execution lifecycle

When the command processor assigns a workgroup to a CU, the CU allocates
register file resources (SGPRs and VGPRs), LDS, and scratch memory, then
launches one or more wavefronts. Each wavefront executes instructions
sequentially from the decoded instruction stream. The CU supports both
functional mode (instruction-accurate, no timing) and cycle-accurate
mode, as selected by the `exec_mode` field in the JSON configuration.

During execution, the CU resolves operands from SGPR and VGPR register
files, inline constants, and literal values. Wavefronts retire when they
encounter a program terminator instruction (such as `s_endpgm`). On
architectures that support AccVGPRs (CDNA2 through CDNA4), the CU also
manages the accumulator register file.

Execution plugins receive callbacks at key points in the wavefront
lifecycle --- dispatch, memory operations, register reads, barriers, and
`s_waitcnt` events. See
[Execution plugin system](execution-plugins.md) for
details on the plugin interface.

## Memory hierarchy

The simulated memory hierarchy mirrors the structure of AMD GPU
hardware:

-   **L1 scalar cache** --- Per-CU cache serving scalar memory (SMEM)
    loads. Line size and associativity are configurable through the JSON
    topology and reported through `rj_vm_gpu_info_t` fields
    `l1_size_kb`, `l1_line_size`, and `l1_assoc`.
-   **L1 vector cache** --- Per-CU cache serving vector memory (flat,
    global, buffer, and scratch) loads and stores.
-   **L2 cache** --- Shared across shader engines within an XCD. Size
    and geometry are reported through `l2_size_kb`, `l2_line_size`, and
    `l2_assoc`. Atomic operations execute at the L2 level.
-   **Memory-side cache** --- Sits between the L2 and GPU memory (VRAM),
    providing an additional caching layer for off-chip accesses.
-   **GpuMemory** --- Models VRAM with per-process VMID page tables.
    Supports passthrough mode (GPU VA equals host VA) and daemon mode
    (shared memfd mappings). The `local_mem_size` field in
    `rj_vm_gpu_info_t` reports the configured VRAM capacity.

## MTYPE-aware coherency

Memory accesses carry coherency attributes that control caching behavior
at each level of the hierarchy. The simulator models three primary
memory types:

-   **UC (Uncacheable)** --- Bypasses caches entirely. Every load goes
    to memory; every store is written through immediately.
-   **CC (Coherently Cacheable)** --- Cacheable with coherency
    maintained across all CUs and XCDs. Used for data that is shared and
    requires visibility guarantees.
-   **RW (Read-Write)** --- Cacheable without cross-CU coherency.
    Suitable for data private to a single wavefront or workgroup.

The coherency encoding varies by architecture generation. CDNA1 and
CDNA2 use a GLC-based model. CDNA3 and CDNA4 use SC0, SC1, and NT fields
for vector memory and GLC for scalar memory. RDNA1 and RDNA2 use GLC,
DLC, and SLC. RDNA3 and RDNA3.5 use SC0, SC1, and TH. RDNA4 uses a
two-bit SCOPE field plus a TH hint. The simulator interprets these
encoding-level fields and applies the appropriate caching policy at each
cache level.

## Atomic operations at L2

Atomic instructions (compare-and-swap, add, min, max, and, or, xor, and
others) execute at the L2 cache. The L2 performs the read-modify-write
cycle atomically, so all CUs within the same XCD observe a consistent
result. The set of supported atomic operations includes both integer and
floating-point variants, matching the instruction set of the configured
architecture.

## Supported ISA targets

The virtual machine supports simulation across a range of AMD GPU
architectures and RISC-V:

| Architecture | GFX target | ISA family |
|--------------|------------|------------|
| CDNA1 | gfx908 | GFX9 |
| CDNA2 | gfx90a | GFX9 |
| CDNA3 | gfx94x | GFX9 |
| CDNA4 | gfx950 | GFX9 |
| RDNA1 | gfx1010 | GFX10 |
| RDNA2 | gfx1030 | GFX10 |
| RDNA3 | gfx1100 | GFX11 |
| RDNA3.5 | gfx1150 | GFX11 |
| RDNA4 | gfx1200 | GFX12 |
| gfx1250 | gfx1250 | GFX12 |
| RISC-V | RV32IMAFDC | RV |

The architecture is selected through the `vm.arch` field in the JSON
configuration and determines which ISA decoder and execution bodies the
CUs use. For a complete list of simulation, DBT, and DBI support status
per architecture, see
[Supported GPU architectures](../reference/supported-architectures.md).
