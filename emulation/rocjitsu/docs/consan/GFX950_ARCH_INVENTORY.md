# ConSan gfx950 Architecture Inventory

This is the closed A1 inventory for the gfx950 port. Locations name the owning
function or helper rather than treating line numbers as stable identifiers.
`CDNA4` means a separate encoding or ABI implementation is required; `target`
means policy is shared but must dispatch on a target capability; `RDNA4-only`
means the path must continue to reject gfx950.

## Reproducible Environment Baseline

- Host C/C++: Clang 21.1.8 from `$HOME/LLVM-21.1.8-Linux-X64`.
- Device compiler: TheRock AMD Clang 23.0.0git, commit
  `5c9bfa94a37c59923dee3c55942566db7904b659` plus the reported patch.
- Runtime: TheRock HSA runtime 1.21, extension version 1.24.
- Kernel driver: ROCk/amdgpu 6.14.14 on Linux 5.15.0-70-generic.
- Agent: `gfx950`, AMD Instinct MI355X, target
  `amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-`, wavefront size 64.
- PCI ID: `1002:75A3`.

The uninstrumented HIP suites, HSA hook load, native IREE ROCm tests, and the
standalone CDNA4 scratch tests use the same TheRock distribution under
`$WORKSPACE_ROOT/TheRock/build/dist/rocm`.

## Source Inventory And Disposition

| Area and source owner | Existing assumption | Disposition | DAG owner |
| --- | --- | --- | --- |
| `consan.cpp`: `target_name`, `arch_name`, `arch_for_target` | target/architecture mapping | shared; gfx950 already maps to CDNA4 | A2 |
| `consan.cpp`: descriptor VGPR helpers | wave size and VGPR allocation granularity | target; CDNA4 is wave64 with groups of 8 | R1A |
| `consan.cpp`: `record_flat_site` | RDNA4 `VflatMachineInst` raw fields | CDNA4 FLAT layout or typed exclusion | FL1A/FL1B/FL1X |
| `consan.cpp`: `record_atomic_site` and raw fillers | RDNA4 DS/FLAT/GLOBAL/BUFFER layouts and TH/SCOPE | DS sites now retain CDNA4 raw fields and typed non-access dispositions; atomic emission remains per admitted family | D1C done; AT1A |
| `consan.cpp`: trap, VCC branch, DS wait, compare, barrier helpers | gfx12 SOPP/VOP encodings and counters | CDNA4 scalar/vector/wait builders plus distinct VM/LGKM-zero pre-barrier wait and `s_barrier` | P1A/P1B/W1/B1A done |
| `consan.cpp`: DS store-to-load conversion | RDNA4 VDS opcode/layout | CDNA4 DS form-specific conversion | D1A/D1B/SC1 |
| `consan.cpp`: LDS check/trap backend | blanket RDNA4 gate | target capability after CDNA4 DS primitives | SC1 |
| `consan.cpp`: report-buffer mismatch action | RDNA4 FLAT stores | CDNA4 publication backend | P1C |
| `consan.cpp`: flat proof backends | blanket RDNA4 gate and gfx12 raw patch shapes | keep gated until FL1A selects FL1B; otherwise typed FL1X | FL1A/FL1B/FL1X |
| `consan_moi.cpp`: descriptor workgroup sources | RDNA4 runtime TTMP payload; other targets use enabled system SGPRs | CDNA4 entry snapshot into persistent VGPRs before compiler reuse; never assume TTMP payload | I1A |
| `consan_moi.cpp`: descriptor wave/VGPR helpers | wave32 bit and gfx12 wave64 granularity 4 | target; CDNA4 forces wave64 and granularity 8 | R1A |
| `consan_moi.cpp`: SGPR allocation helpers | groups of 8 and limit 106 | CDNA4 ordinary user SGPR geometry and special-state split | R1A/R1B |
| `consan_moi.cpp`: owner derivation | RDNA4 lane count plus TTMP/HW_ID inputs | CDNA4 packed workitem ID plus AQL workgroup dimensions; reject HW_ID and TTMP as standard sources | I1B/I1C |
| `consan_moi.cpp`: inline-shadow backend | blanket RDNA4 gate; gfx12 FLAT atomics/waits | target capability after CDNA4 atomic publication | IS1A-IS1C |
| `consan_moi.cpp`: sampled backend | wave owner and VCC/SCC save convention | Static publication, runtime selection, bounded immediate checking, and the generation-qualified host oracle agree with record/replay on gfx950 | SA1A-SA1C done |
| `consan_moi.cpp`: access-record backend | blanket RDNA4 gate and gfx12 global stores | target capability after CDNA4 publication/identity | RR1A/RR1B |
| `consan_moi.cpp`: owner/private-epoch prologues | RDNA4 state and direct gfx12 scratch accesses | target capability; use SpillManager CDNA4 backend | I1B/S5 |
| `consan_moi.cpp`: barrier record/epoch backends | blanket RDNA4 gate | CDNA4 barrier decode, pre-wait, then engine policy | B1A/B1B |
| `consan_moi.cpp`: atomic record/handoff backends | blanket RDNA4 gate | CDNA4 atomic form and ordering capability | AT1A/AT1B |
| `instruction_builder.h`: generic SOPP/SOP1/SOP2/VOP2 selectors | opcode differences | target-parametric; retain generated opcode dispatch | P1A/P1B |
| `instruction_builder.h`: readfirstlane, MBCNT, compares, literal moves | hard-coded gfx12 words | CDNA4 encoders validated against LLVM | P1B/I1B |
| `instruction_builder.h`: FLAT load/store/atomics | gfx12 12-byte layouts and SCOPE/TH | CDNA4 report/atomic builders | P1C/IS1A |
| `instruction_builder.h`: scalar EXEC/VCC/SCC helpers | gfx12 opcodes and VCC alias assumptions | CDNA4 scalar-state backend | R1B/P1A |
| `instruction_builder.h`: scratch and scratch waits | gfx12 VSCRATCH signed-24 and split load/store counters | CDNA4 FLAT_SCRATCH signed-13 plus `VM_CNT` | S1/S2 |
| `spill_manager.cpp`: `build_vgpr_spill_sequence` | RDNA4-only emission | target-dispatched RDNA4/CDNA4 backend | S5 |
| HSA hook: target and dispatch association | no direct RDNA4 encoding gate | shared; private-size transaction remains target-independent | S3 |
| HIP/CTest live tests | ConSan binaries and recipes hard-coded to gfx1201 | target-aware sources and registration | A3A-A3C/T1A/T1B |

No active RDNA4 gate is justified solely by the existence of a generic CDNA4
decoder. Each gate above remains until its named native contract and test owner
are complete.

## CDNA4 Scratch Contract

LLVM 23 for `-mcpu=gfx950` and rocJITsu's CDNA4 decoder agree on:

```text
scratch_store_dword off, v7, off offset:4  DC704004 007F0700
scratch_load_dword  v7, off, off offset:4  DC504004 077F0000
s_waitcnt vmcnt(0)                         BF8C0F70
```

The FLAT_SCRATCH offset is a signed 13-bit byte field. ConSan uses only
non-negative dword-aligned slots, so the last accepted slot begins at 4092 and
the first rejected private extent is 4096 bytes. Both accesses are predicated
by EXEC. Empty EXEC performs no lane access; a partial EXEC saves and restores
only active lanes. A `VM_CNT=0` wait follows the stores before any victim VGPR
is clobbered and follows the loads before guest use.

CPU tests round-trip the words through the CDNA4 decoder and preserve the
existing gfx1201 bytes. `ConSanSpillHipTest.Gfx950VgprScratchRoundTrip` and
`Gfx950PartialExecVgprScratchRoundTrip` execute those exact instructions on the
MI355X. Disassembly of the bundled gfx950 object contains the words above, and
both hardware tests pass.

## CDNA4 Stable Identity Contract

gfx950 does not provide a usable ordinary-code wave-in-workgroup value through
the tempting sources. TTMP registers are trap-handler/runtime state and read as
zero in the focused compute probes. The ISA manual also explicitly warns that
`HW_ID` reports physical placement which may change during a wave's lifetime.
Neither source participates in standard ConSan correctness.

ConSan instead performs a reversible AMDHSA entry transaction. It enables the
standard AQL dispatch-pointer user SGPR pair, snapshots the descriptor-selected
workgroup X/Y/Z system SGPRs into three persistent VGPRs before guest code can
reuse them, and reads packed 16-bit workgroup X/Y dimensions from the dispatch
packet. From AMDHSA's packed workitem VGPR0 it computes
`x + size_x * (y + size_y * z)`, divides by the wave64 width, initializes the
persistent owner/epoch pair, and then shifts every original ABI SGPR back to
its pre-instrumentation number before branching to the original entry.

LLVM-matched SMEM and VOP3 fixtures cover the dispatch load and multiply-add
encodings. A synthetic descriptor test covers dispatch-pointer insertion,
system-SGPR snapshots, original-ABI restoration, and persistent record inputs.
The guarded `StableThreeDimensionalIdentityRecordReplay` hardware test launches
eight 3D workgroups with two waves each and requires all eight distinct
workgroup keys and the exact owner set `{0,1}` in every group.
