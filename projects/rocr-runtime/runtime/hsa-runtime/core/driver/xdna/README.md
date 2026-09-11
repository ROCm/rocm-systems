# XDNA backend: full-ELF dispatch

This directory implements ROCr's backend for the XDNA kernel driver, which drives the AIE/NPU
found on Ryzen AI parts. It supports two dispatch shapes. This document covers why the second
one exists.

## The problem

Until recently the backend had exactly one way to dispatch: a **PDI plus a separate instruction
sequence**, submitted as `ERT_START_CU`. The PDI configures the AIE array; the instruction
sequence is a TXN stream the hardware runs against it. `SubmitCmdChain` resolves the packet's
`pdi_addr` to a BO, looks it up in a 32-entry cache, and selects it at dispatch with a bit in the
command's CU mask.

That design has two hard limits, both structural rather than incidental:

**The PDI ceiling is 32.** The driver's `MAX_NUM_CUS` is 32 and the wire format is a fixed
`config_cu_req.cfgs[32]`, enforced in `aie2_config_cu`, `aie2_register_pdis` and
`aie2_legacy_config_cu`. Extra CU-mask words do not help: `amdxdna_cmd_get_cu_idx()` computes
`ffs(cu_mask[i]) - 1` and breaks on the first word with a set bit, never adding `i * 32`, so bits
in the second word alias onto CUs 0-31. An application with more than 32 distinct designs cannot
run them on one queue, and raising the limit needs a coordinated driver and firmware change.

**Every new design costs a hardware-context teardown.** `aie2_ctx_cu_config()` rejects a second
`CONFIG_CU` outright ("Not support re-config CU"), so `ctx->cus` is write-once. When a dispatch
arrives with a PDI the cache has not seen, the only way to admit it is to destroy the hardware
context and build a new one with the larger CU set.

## Motivation

Both limits bite the same workload: an application that runs many different AIE designs.

- Past 32 designs it does not degrade, it stops working.
- Below 32, the first dispatch of each new design pays a full context destroy plus create. That
  cost is invisible in a benchmark that loads one design and dispatches it in a loop, and very
  visible in anything that cycles through designs.
- The ceiling cannot be lifted from userspace. It is the size of a struct in the firmware
  message ABI.

Full ELF removes both limits at once rather than raising either, because the firmware path it
uses never derives a CU index at all: no index means no CU mask, no `CONFIG_CU`, no 32-entry
table, and no teardown when a new design appears.

## The technical solution

Full ELF dispatches with `ERT_START_NPU_PREEMPT_ELF` (opcode 22). The PDI is not a separate
buffer the firmware is told about out of band; it travels **inside the ELF as a section**, and its
address is written into the control code as a relocation before the dispatch runs. The control
code itself is the TXN stream plus a leading `load_pdi` operation, so the design configures the
array as part of running.

For the `vector_scalar_add` design in `rocrtst/suites/aie`, the ELF produced by
`aiecc --get-full-elf` looks like this:

```
Sections   .pdi.1        2992 B   the PDI, as an ELF section
           .ctrltext.0    316 B   TXN control code (instruction sequence + one load_pdi)
           .group.0                COMDAT group, signature symbol `sequence`
.rela.dyn  0x18 type 8  -> .pdi.1  address_64  : PDI device address
           0x30 type 5  -> "0"     shim_dma_48 : argument 0 device address
           0xa8 type 5  -> "1"     shim_dma_48 : argument 1 device address
```

A kernel name such as `main:sequence` resolves as signature symbol -> COMDAT group ->
`.ctrltext.0`.

### Division of responsibility

There are **no register-map arguments** on this path: every buffer address is patched into the
control code before submission. That splits cleanly in one place:

- **The application** parses the ELF, allocates the control code and PDI from the agent's device
  pool, and patches its own argument addresses in. Argument addresses are ordinary host virtual
  addresses, so it needs nothing from the runtime to do this.
- **The runtime** writes the PDI's *device* address into the control code and builds the command.
  This is the one address the application cannot know, and it is why the runtime is involved at
  all. Patching a host VA there makes the dispatch time out.

Consequently ROCr gained no ELF handling. The reader lives in the test suite
(`rocrtst/suites/aie/aie_full_elf.h`) and is shared with the benchmarks by include path.

### Packet ABI

There is **no new packet opcode**. `hsa_amd_aie_kernel_dispatch_packet_t::pdi_patch_offset` (the
former `reserved4`) selects the shape:

| field | `pdi_patch_offset == 0` (PDI + insts) | `pdi_patch_offset != 0` (full ELF) |
|---|---|---|
| `insts_addr_*`, `insts_size` | instruction sequence | control code |
| `pdi_addr` | PDI to configure the array with | PDI to relocate into the control code |
| `kernarg_address` | argument addresses, patched by hardware | residency and cache maintenance only |

This is safe because a full-ELF control code begins with a transaction header, so its PDI patch
site is never at offset 0 (it is `0x18` in the design above), while a PDI + instruction sequence
dispatch has no such site.

A **queue is homogeneous**: its first packet fixes the mode for the queue's lifetime. The two
shapes need hardware contexts whose CU configuration cannot be reconciled after the fact, and a
context's CU configuration is write-once. The mode is committed only after every packet in a
batch has been accepted, so a rejected submission cannot pin an otherwise-unused queue.

### Constraints worth knowing

- **The control code must be 16 KiB aligned in device memory.** The PDI has no alignment
  requirement. This is not documented anywhere and is not visible in the driver; it was
  established by sweeping the control code across offsets and observing which dispatches
  complete. A misaligned control code is *accepted* and then never signals, so the runtime
  rejects it up front rather than leaving the dispatch to hang.
- **`shim_dma_48` is additive on the buffer descriptor already in place.** Re-patching must start
  from a pristine copy of the control code, or addresses accumulate.
- **Arguments live in the control code, not the command.** Two dispatches with different
  arguments need two control-code buffers, including two dispatches in the same batch.
- **Full ELF is aie2p only.** Phoenix (`0x1502`, aie2) has neither the firmware command nor the
  preemption support it builds on. Every `0x17f0` revision is aie2p, so the gate is by device id.
  The firmware gate (`mgmt_prot_minor >= 15`) is not exposed through any ioctl, so too-old
  firmware on a supported part can only surface as an error at submit.
- **A design with no preemption sections leaves the save and restore buffers null**, which aie2p
  firmware accepts.
- **The firmware aborts a command chain whose slot carries an odd argument count below 15.** The
  driver takes that count from the command's declared payload length, so a command whose real
  payload is an odd number of dwords has to declare one dword of padding. This affects the PDI +
  instruction sequence shape, whose payload is odd; full ELF has an argument count of 2 and is
  unaffected. Verified on npu4: with no padding the chain comes back in state `abort`.

## What has been implemented

### Runtime

- Full-ELF command construction (`ERT_START_NPU_PREEMPT_ELF`) alongside the unchanged
  `ERT_START_CU` path, selected per packet by `pdi_patch_offset`.
- PDI device-address relocation into the control code, with bounds and alignment validation of
  the patch site against the control-code allocation.
- Rejection of a control code that is host-only, misaligned, or smaller than the declared
  `insts_size`, and of a null or host-only PDI.
- Queue-mode homogeneity, committed only once a whole batch is accepted.
- aie2p gating by PCI device id.
- Lazy `CONFIG_CU`: the ioctl is skipped entirely when the PDI cache is empty. This is what makes
  a CU-less full-ELF context possible, and it also removes a wasted ioctl from the existing path.
- Command chains bounded to the driver's 4 KiB chain buffer and split when longer, instead of
  letting the driver reject one oversized chain. The per-slot cost is
  `52 + 4 * arg_cnt`, giving `floor(4096 / (52 + 4 * arg_cnt))` commands: 68 for full ELF
  (`arg_cnt = 2`) and 44 for PDI + instruction sequence at two kernel arguments (`arg_cnt = 10`).
- Reduced the padding declared on a PDI + instruction sequence command from three dwords to the
  one the parity rule actually requires, and allocated it. The driver copies the whole declared
  payload into the chain slot, so the extra dwords were both uninitialised memory and wasted slot
  space; this raises the chain capacity from 40 commands to 44.
- PDI cache rollback when a batch fails partway, so cached entries can never claim compute units
  the hardware context was not given.
- Chain-failure diagnostics: on failure the device's `submit_index` and `error_index` are
  reported along with the chain and failing sub-command states.
- Fixed `CreateKernelModeQueue` passing a hardcoded tile count. The driver derives columns as
  `num_tiles / core.row_count`, so on a 4-row array asking for one tile asks for zero columns and
  `CREATE_HWCTX` fails with `EINVAL`.

### Tests (`rocrtst/suites/aie`)

- `aie_full_elf.h`, a header-only ELF32 reader: sections, symbols, COMDAT groups and `.rela.dyn`,
  plus `address_64` and `shim_dma_48` patching. It refuses anything it cannot represent rather
  than silently dropping it.
- Ten tests that run today: parsing and its rejections, single and batched dispatch, a full queue,
  re-patching one control code with different arguments, both buffers at a non-zero offset into
  their allocation, and dispatching more than 32 distinct PDIs — which the PDI path cannot do.
- Seven tests compiled out behind `AIE_TEST_ASYNC_ERRORS`, covering packet-level rejections and
  both queue-mixing directions. They need a rejected dispatch to be observable; today the runtime
  signals it by throwing from a void HSA entry point, which aborts the process. Each was confirmed
  to exercise a real rejection.

### Benchmarks (`rocrtst/suites/aie-performance`)

- `VectorScalarAddHSAELF`, `VectorScalarAddHSAELFAllocKernargs` and `VectorScalarAddHSAELFParse`,
  mirroring the existing PDI + instruction sequence benchmarks.

## Measured on Strix (npu4, aie2p), batch sizes 1 and 32

| path | 1 | 32 |
|---|---|---|
| HSA full ELF | 90 µs | 1263 µs |
| HSA PDI + instruction sequence | 90 µs | 1538 µs |
| XRT full ELF, kernel | 197 µs | 4128 µs |
| XRT full ELF, runlist | 68 µs | 4040 µs |

ELF parsing costs about 0.39 µs and happens once per kernel, not per dispatch. Running the same
`aie.elf` through both stacks is the cross-check that ROCr's host-side patching and command
construction are correct.

## Known gaps

- The 68-command full-ELF chain ceiling is arithmetic, never exercised: the AIE queue is pinned at
  64 packets, so a full-ELF chain cannot reach it. Only the PDI path's 44-command split is covered
  by a test.
- The aie2 rejection path has no test that can run on aie2p hardware.
- Too-old firmware on a supported part is not distinguished from other submit failures.

## Building and running

See `rocrtst/suites/aie/CMakeLists.txt` for the artifact pipeline. The full-ELF kernel additionally
needs `xclbinutil` and an `aiebu-asm` supporting the `aie2_config` target; without them the ELF is
not built and the full-ELF tests skip themselves. `aiebu-asm` exits 0 on an unrecognised target and
`aiecc` does not check that the ELF appeared, so the build probes for the capability and guards the
rule rather than trusting the exit status.

Note that four `Memory.VMemSetAccess*` tests hang and two `VMemUnmapRemap*` tests fail on current
hardware, independently of this work, so a bare `ctest` does not terminate. Excluding those six,
the suite passes 59/59 in about five seconds.
