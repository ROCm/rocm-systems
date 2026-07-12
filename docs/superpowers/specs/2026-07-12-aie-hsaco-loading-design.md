# AIE hsaco Loading via HSA APIs — Design

**Date:** 2026-07-12
**Status:** Approved for implementation planning
**Component:** rocr-runtime (loader, AIE agent, XDNA driver, rocrtst AIE suite)

## Summary

Enable loading AIE (XDNA NPU) kernels through the standard HSA executable/loader
APIs — `hsa_executable_create_alt` → `hsa_code_object_reader_create_from_memory`
→ `hsa_executable_load_agent_code_object` → `hsa_executable_freeze` →
`hsa_executable_get_symbol_by_name` → `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT` —
instead of the current scheme where the test manually loads split `design.pdi`
and `insts.bin` files and passes their raw addresses in the dispatch packet.

The kernel payload (instruction sequence, and an optional PDI) is packaged into a
single architecture-named section (`aie2` / `aie2p`) of an hsaco. A Python tool
injects this section into an hsaco. At load time the runtime parses the section,
copies the blobs to device memory, and builds an internal kernel descriptor owned
by the AIE agent. The dispatch packet carries an **opaque `kernel_object`
handle**; the runtime resolves the PDI/insts from the agent at submit time. This
mirrors how GPU AQL packets carry a `kernel_object` handle rather than raw code
pointers.

## Goals

- Load AIE kernels via standard HSA loader APIs, matching the GPU code-object flow
  as closely as the hardware allows.
- Package instruction + optional PDI payloads into an hsaco section named after the
  architecture (`aie2`, `aie2p`).
- Keep the internal kernel-descriptor layout **hidden from users**: the packet
  carries only an opaque handle.
- Support multiple kernels per code object and multiple independent code objects
  per executable.
- Future-proof the format and loader so deferred capabilities (relocations,
  cross-object linking, combined GPU+AIE hsaco, on-device descriptors) are
  additive, not rewrites.

## Non-Goals (Deferred)

These are explicitly out of scope for the MVP. Each is designed to be additive.

| Deferred capability | Enabled later by |
|---|---|
| On-device kernel descriptor (read by hardware, like `amd_kernel_code_t`) | Descriptor `version` field + new section `version_minor` |
| Relocations (patching device blobs at load) | AIE relocation section (producer-side) + `ApplyAieRelocations` pass |
| Cross-object linking (kernel in object B references symbol in object A) | Requires relocations first, then load-time symbol resolution |
| Combined GPU+AIE hsaco (GPU segments + AIE section in one object) | Detection already section-based; only the load *branch* changes to do both |
| Avoiding per-packet hwctx reconfigure | Future loading mechanism; MVP accepts the reconfigure-on-new-PDI overhead |

## Background: How GPU hsaco Loading Works

The AIE design mirrors this flow.

- **Container:** hsaco is a standard ELF, `e_machine == EM_AMDGPU`, arch encoded in
  `EF_AMDGPU_MACH` flags. Code lives in loadable **segments** (PT_LOAD).
- **Load:** `hsa_executable_load_agent_code_object` parses the ELF, allocates device
  memory per segment (`SegmentAlloc`), copies data (`SegmentCopy`), applies
  relocations (`ApplyRelocations`), and freezes (`SegmentFreeze`).
- **Descriptor:** the compiler emits an `amd_kernel_code_t` inside the loaded code
  segment. Each kernel symbol (`foo.kd`) resolves to the device VA of its
  descriptor.
- **`kernel_object`:** `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT` returns that
  device VA (`hsa.h:3051`). Returns 0 while the executable is unfrozen.
- **Dispatch:** the AQL packet carries the `kernel_object` VA. The command
  processor reads the descriptor on-device and runs the kernel.
- **Multi-object:** an executable holds N loaded code objects
  (`loaded_code_objects`); each `load_agent_code_object` call appends symbols to
  shared `program_symbols_` / `agent_symbols_` maps.

## GPU vs. AIE: Differences

| Aspect | GPU (hsaco) | AIE (this design) |
|---|---|---|
| Container | ELF, `e_machine == EM_AMDGPU`, arch in `EF_AMDGPU_MACH` | ELF (hsaco), arch selected by **section name** (`aie2`/`aie2p`) |
| Where code lives | Loadable segments (PT_LOAD) | One arch-named **section**: header + kernel table + blob pool |
| Payload per kernel | One code stream | Two blobs: **insts** (required) + **PDI** (optional, `pdi_size==0` ⇒ none) |
| Blob sharing | N/A | Kernels share a blob via identical `(offset,size)`; loader copies each unique blob once |
| Kernel descriptor | `amd_kernel_code_t`, compiler-emitted, in code segment | Runtime-synthesized, agent-owned, **not** in the ELF |
| Device placement | `SegmentAlloc`/`Copy`/`Freeze` + relocations | Same alloc/copy/freeze primitives; **no relocations** (MVP) |
| `KERNEL_OBJECT` handle | Device VA of `amd_kernel_code_t` | Host pointer to agent-owned `AieKernelDescriptor` |
| Symbol table | Compiler-emitted `foo.kd` symbols | Kernel table in section header; loader creates one `AieKernelSymbol` each |
| Packet → kernel binding | Packet carries VA; CP reads descriptor on-device | Packet carries host pointer; **driver** dereferences it at submit |
| Who reads descriptor | Hardware command processor | XDNA driver (in-process), host-side at submit (MVP) |
| Producer | Compiler/linker emit hsaco | Post-hoc Python tool injects arch section |

Two deliberate divergences: the AIE descriptor is **runtime-synthesized** (the
section is packaged after compilation), and descriptor resolution is **host-side**
(MVP reuses the existing reconfigure-on-new-PDI driver path).

## Design

### 1. hsaco Architecture Section Format

A single section named after the target architecture (`aie2` or `aie2p`) holds a
versioned header, a kernel table, and an offset-addressed blob pool. All offsets
are **section-relative**.

```c
// Magic 'A','I','E','K' little-endian.
#define AIE_SECTION_MAGIC 0x4B454941u

struct aie_section_header {
  uint32_t magic;             // AIE_SECTION_MAGIC
  uint16_t version_major;     // reject on mismatch
  uint16_t version_minor;     // additive-only changes
  uint32_t header_size;       // bytes; offset from section base to kernel table
  uint32_t kernel_count;
  uint32_t kernel_entry_size; // stride between kernel entries (forward-compat)
  uint32_t string_table_offset;
  uint32_t string_table_size;
  uint32_t blob_pool_offset;  // section-relative; blobs live in [this, section_end)
  uint32_t reserved[4];       // must be 0
  // aie_kernel_entry[kernel_count] follows at header_size
};

struct aie_kernel_entry {
  uint32_t name_offset;       // relative to string_table_offset; NUL-terminated
  uint32_t insts_offset;      // REQUIRED: section-relative
  uint32_t insts_size;        // REQUIRED: > 0
  uint32_t pdi_offset;        // OPTIONAL: 0 if no PDI
  uint32_t pdi_size;          // OPTIONAL: 0 if no PDI
  uint32_t kernarg_size;
  uint32_t num_cols;
  uint32_t reserved[4];       // must be 0; headroom for per-kernel metadata
};
```

**Forward-compatibility rules:**
- Readers reject a section whose `version_major` differs from what they support.
- Readers use `header_size` to locate the kernel table and `kernel_entry_size` to
  stride entries, so new trailing fields in either struct are safe to add under a
  `version_minor` bump.
- `reserved` fields must be written as 0 and ignored by current readers.

#### Blob-Pool Usage Contract (documented for producers and consumers)

- All `*_offset` values are **section-relative** (byte offset from the start of the
  `aie2`/`aie2p` section).
- A kernel's instruction blob is `section_base + insts_offset`, length
  `insts_size`. `insts_size` **must** be `> 0`.
- A PDI is **absent** if and only if `pdi_offset == 0 && pdi_size == 0`. When
  present, it is `section_base + pdi_offset`, length `pdi_size`.
- Two kernels **share** a blob by carrying identical `offset` and `size` values.
  The loader copies each unique `(offset, size)` pair to device memory exactly
  once and points both descriptors at the same device buffer.
- Blobs need not be contiguous or ordered within the pool; the pool is addressed
  purely by `(offset, size)`.
- All blob bytes must lie within `[blob_pool_offset, section_end)`. A blob that
  overruns the section is a parse error.

### 2. Python Packaging Tool

A standalone tool (in the AIE suite dir) that injects/updates the arch
section in an hsaco.

- **Inputs:** target hsaco path; architecture (`aie2` | `aie2p`); one or more
  kernel specs, each `{name, insts_path (required), pdi_path (optional),
  kernarg_size, num_cols}`. `kernarg_size` and `num_cols` are **explicit inputs**
  supplied by the CMake pipeline — `aiecc` does not emit them today, so the caller
  provides them (as the current test hardcodes them).
- **Behavior:** builds the `aie_section_header` + `aie_kernel_entry[]` + string
  table + blob pool per the format above, deduplicating identical blobs into a
  single pool entry, and writes it as a section named after the architecture.
  Replaces an existing same-named section if present.
- **Implementation:** ELF manipulation via `pyelftools` (read) plus section
  injection. Emits structs with explicit little-endian packing (`struct` module)
  matching the C definitions.
- **Usage:** invoked by the AIE test CMake pipeline after `aiecc` produces
  `insts.bin` and `design.pdi`, to produce a loadable hsaco.

Another standalone tool (in the AIE suite dir) that reads an hsaco and prints the
contents of the `aie2`/`aie2p` section in human-readable form. It should validate that the section is well-formed and report any parse errors.

### 3. Loader

Extend the existing AIE path (`core/runtime/amd_aie_code.cpp`,
`loader/executable.cpp`). The current path parses a `.ctrltext`/`.ctrldata`
NPU-ELF and has no PDI handling; it is replaced by the arch-section format.

- **Detection (`IsAieCodeObject`):** key off **presence of an `aie2`/`aie2p`
  section**, independent of `e_machine`. (Current code keys off `EM_NONE`; this
  change is what makes a future combined GPU+AIE hsaco detectable without
  reworking detection.) The check runs on **every** code-object load, including
  GPU ones, so it must be cheap (a section-name scan) and must not false-positive
  on GPU hsacos.
- **Arch-vs-agent validation:** the section name must match the loading agent's
  architecture. On mismatch, return `HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS`.
- **Parse:** validate `magic` and `version_major`; read the kernel table and
  string table.
- **Device placement:** for each unique blob `(offset, size)` referenced by any
  kernel, `SegmentAlloc` device memory, `SegmentCopy` the bytes, `SegmentFreeze`.
  Reuse one device buffer for shared blobs.
- **Descriptor:** build one internal `AieKernelDescriptor` per kernel:
  `{version, insts_dev_addr, insts_size, pdi_dev_addr (0 if none), pdi_size,
  kernarg_size, num_cols}`. Owned by the AIE agent (see §4). The descriptor struct
  carries a `version` field so a future on-device format is a version bump.
- **Symbols:** create one `AieKernelSymbol` per kernel. `KERNEL_OBJECT` returns the
  **opaque handle** to the descriptor (not the raw insts address as today).
  Returns 0 while unfrozen, matching GPU. Surface `kernarg_size`, alignment,
  group/private sizes, name, type, agent via symbol info identically to GPU.
  **Parity scope:** GPU parity applies to *symbol-info queries only*. The AIE
  kernarg ABI is unchanged and deliberately **not** made GPU-identical — the
  kernarg buffer remains the packed `[addresses…, sizes…]` layout the current AIE
  driver/test use, not the GPU `amd_kernel_code_t` convention.
- **Multi-object:** append the loaded object to `objects`; insert kernels into the
  shared `agent_symbols_` map. On a **duplicate `(name, agent)`**, return
  `HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED` (the current code blindly
  overwrites — this is fixed). Symbol lookup by name spans all loaded objects.

### 4. AIE Kernel Descriptor and Ownership

The `kernel_object` handle is a **host pointer to an internal
`AieKernelDescriptor`**. The XDNA driver runs in-process, so it dereferences the
pointer directly at submit time — there is no separate registry/lookup table.

```c
struct AieKernelDescriptor {
  uint32_t version;        // reserved for a future on-device format
  uint32_t reserved0;
  uint64_t insts_dev_addr; // device address of the instruction blob (an XDNA BO)
  uint64_t insts_size;
  uint64_t pdi_dev_addr;   // device address of the PDI blob (an XDNA BO), or 0
  uint64_t pdi_size;       // 0 if no PDI
  uint32_t kernarg_size;
  uint32_t num_cols;
};
```

- **Ownership/lifetime:** the AIE agent (`amd_aie_agent`) owns every
  `AieKernelDescriptor` and the underlying device buffers. Descriptors outlive any
  in-flight dispatch that references them.
- **`KERNEL_OBJECT`:** returns this host pointer (as `uint64_t`). Returns 0 while
  the executable is unfrozen, matching GPU.
- **Freeing:** descriptors and device buffers are freed at **executable destroy**,
  together with the loaded objects held in `objects`. Per-object unload is not a
  goal of the MVP (the current loader structure frees at executable scope).
- **PDI-less kernels:** `pdi_dev_addr == 0` ⇒ the driver skips PDI configuration.

See §6 for the exact load↔submit address contract.

### 5. Packet Change (`hsa_ext_amd_aie.h`)

Required now, to avoid leaking the internal descriptor structure to users.

- The packet carries a single opaque **`kernel_object`** handle (a `uint64_t`,
  matching the GPU `hsa_kernel_dispatch_packet_t::kernel_object`) — the host
  pointer to the `AieKernelDescriptor` (§4).
- **ABI:** the handle is stored in the existing `insts_addr_low` /
  `insts_addr_high` words (as one 64-bit value). The former `insts_size` and
  `pdi_addr` fields become **reserved, must be 0**. The packet struct size is
  unchanged. Users no longer set any address/size directly — the descriptor holds
  them.
- **Driver change:** `XdnaDriver::SubmitCmdChain` currently reads `pkt->pdi_addr`,
  `pkt->insts_addr_{low,high}`, and `pkt->insts_size` directly. It is changed to
  (1) reconstruct the descriptor pointer from `insts_addr_low/high`, (2) read
  `pdi_dev_addr` / `insts_dev_addr` / `insts_size` from the descriptor, and (3)
  resolve those device addresses via `FindBOHandle` exactly as today. When
  `pdi_dev_addr == 0`, PDI configuration and caching are skipped for that packet.
- `kernarg_address` / `num_kernargs` semantics are unchanged (see kernarg note in
  §3).
- The test and any packet-construction helper set the handle from
  `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT`.

### 6. Load↔Submit Address Contract

The load path and the submit path must agree on how device addresses are produced
and resolved.

- **Loader allocation:** the loader copies each unique blob to device memory using
  the **same allocation mechanism the current tests use** for PDI/insts — the XDNA
  dev-heap path that produces an XDNA BO registered in the driver's
  `vmem_addr_mappings`. This guarantees the device addresses stored in the
  descriptor are resolvable by `FindBOHandle` at submit.
- **Handle flow:** loader builds `AieKernelDescriptor` → `KERNEL_OBJECT` returns
  its host pointer → user places it in the packet (`insts_addr_low/high`) →
  `SubmitCmdChain` dereferences it → reads device addresses → `FindBOHandle`
  resolves them to BO handles → existing reconfigure-on-new-PDI path runs.
- **Arch reporting:** arch-vs-agent validation (§3) uses the architecture(s)
  reported by the AIE agent/driver. If the current agent surface is insufficient to
  map `aie2`/`aie2p` ↔ the agent's ISA/architecture name, add a query function on
  the agent or driver to report supported architectures. The exact mapping
  (section name ↔ agent architecture string) is defined as part of implementation
  and asserted in tests.

## Testing

Test-first: add a failing test that drives the full HSA-API load path, then
implement toward green.

- **`SingleDispatchHsaco`** (extends `SingleDispatch`): package the vector-scalar
  kernel into an hsaco via the Python tool; load it with
  `hsa_executable_create_alt` → `hsa_code_object_reader_create_from_memory` →
  `hsa_executable_load_agent_code_object` → `hsa_executable_freeze`; get the
  symbol and `KERNEL_OBJECT`; dispatch a packet carrying the handle; verify
  `output[i] == input[i] + 1`.
- **Symbol-query parity:** assert `KERNEL_OBJECT` is 0 before freeze and non-zero
  after; assert kernarg size and name resolve.
- **Multi-object:** load two independent AIE hsacos into one executable, dispatch a
  kernel from each, verify both outputs.
- **Duplicate-name rejection:** loading two objects that define the same kernel
  name returns `HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED`.
- **Optional PDI:** a kernel packaged with no PDI (`pdi_size == 0`) loads, and its
  descriptor reports `pdi_dev_addr == 0`. (Runtime dispatch behavior for a
  PDI-less kernel depends on kernel content; the test asserts the load/descriptor
  path, not necessarily a successful hardware run.)

## Affected Code

- `runtime/hsa-runtime/inc/hsa_ext_amd_aie.h` — packet struct: opaque handle.
- `runtime/hsa-runtime/core/inc/amd_aie_code.hpp`,
  `core/runtime/amd_aie_code.cpp` — arch-section parser (replaces
  `.ctrltext`/`.ctrldata` format); PDI support; blob dedup.
- `runtime/hsa-runtime/loader/executable.cpp`,
  `loader/executable.hpp` — section-based detection; arch validation; descriptor
  build; opaque `KERNEL_OBJECT`; duplicate-name rejection; multi-object append.
- `runtime/hsa-runtime/core/runtime/amd_aie_agent.cpp`,
  `core/inc/amd_aie_agent.h` — descriptor ownership/lifetime; supported-arch query.
- XDNA driver submit path (`core/driver/xdna/amd_xdna_driver.cpp`) — resolve
  handle → PDI/insts at submit; skip PDI config when absent.
- `rocrtst/suites/aie/` — Python packaging tool, CMake pipeline step, new tests.

## Open Risks

- **Section injection into hsaco:** confirm the chosen ELF-editing approach
  produces an object the loader's ELF reader (`amd_elf_image`) accepts. Validate
  early with a round-trip parse test.
- **Loader allocation ↔ `FindBOHandle`:** the loader must allocate PDI/insts via
  the XDNA dev-heap BO path (§6) so submit-time `FindBOHandle` resolves them.
  Verify the loader's device-memory allocation for AIE produces a registered BO,
  not an allocation invisible to the driver.
- **Descriptor lifetime:** the host pointer in the packet must remain valid for the
  duration of any in-flight dispatch; freeing at executable destroy must not race
  with pending submissions.
- **Existing `.ctrltext` path consumers:** verify nothing else depends on the
  current NPU-ELF format before replacing it.
