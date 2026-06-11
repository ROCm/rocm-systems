# DBI Design Document

## Overview

The Dynamic Binary Instrumentation (DBI) system patches AMDGPU HSA code objects in-place, before they are loaded into device memory, to inject calls into user-supplied instrumentation functions. Patched code objects can (will) then be loaded by either the simulated KMD (`SimulatedDriver`) or by real ROCR via the HSA tools layer (`HSA_TOOLS_LIB=librocjitsu_hooks.so`). DBI itself is (will be) target-agnostic.

This document describes the DBI subsystem as currently implemented. Most of the planned DBI machinery (trampoline builder, probe registry, instrumentation pass, public C API) does not exist yet. The pieces in tree today are the foundational analyses and resource managers that future instrumentation passes will consume.

---

## Architecture

TODO. The shipped DBI surface is a set of foundational building blocks (liveness, register set, spill manager) rather than an end-to-end pipeline. A meaningful block diagram will land once the trampoline builder and instrumentation pass exist and there is a real flow to draw.

---

## Register Liveness Analysis [shared with DBT]

**Files:** `analysis/liveness.h`, `analysis/liveness.cpp`, `analysis/def_use_chain.h`, `analysis/def_use_chain.cpp`
**Used by:** DBT semantic translator (today); DBI passes (planned)

Kernel-scoped backward register liveness over the CFG embedded in `BasicBlock`. Callers construct a `LivenessAnalysis` from one `KernelBlockScope` (the blocks reachable from one kernel descriptor entry). Successor/predecessor edges that leave the scope are ignored, so one decoded code object containing N kernels yields N independent analyses.

### What it tracks

Ordinary SGPRs, VGPRs, and AccVGPRs via `RegisterSet`. `InstDefUse` records explicit operand defs and uses plus instruction-level implicit hooks; the only implicit hook today is the FLAT `saddr` SGPR-pair use that does not appear as an explicit operand.

### What it does NOT track

- EXEC, VCC, SCC, M0, FLAT_SCRATCH, TTMP — special architectural state. The `RegClass` enum names these for future use, but they are not in the dataflow set.
- Cross-kernel CFG. Edges that leave the kernel scope are silently dropped.
- Memory dependencies. Liveness is purely register-based.

---

## SpillManager

**Files:** `code/patch/spill_manager.h`, `code/patch/spill_manager.cpp`

Per-kernel scratch-layout planner for DBI spill/fill slots. Future instrumentation passes will use it to reserve byte offsets within per-lane scratch where saved SGPRs / VGPRs / AccVGPRs go before a probe runs and from which they are restored after.

### Responsibilities

- Reserve a "DBI spill zone" appended above the kernel's existing `private_segment_fixed_size`, aligned to 16 bytes.
- Hand out stable per-register byte offsets within that zone. Registers cannot get more than one offset.
- Enforce a hard per-lane scratch cap. Allocations that would push the bumped total past the cap fail; on failure the manager state is unchanged.
- Compute the bumped `private_segment_fixed_size` that the kernel descriptor patcher will write back.

### What it is not

- Not a memory allocator. SpillManager only computes layout.
- Not the code generator. SpillManager hands out offsets; emitting the actual `scratch_store` / `scratch_load` (or equivalents) will be the trampoline builder's job.
- Not an MFMA-clobber tracker. AccVGPR clobbering by long-latency MFMA is deferred

### Public API

```cpp
class SpillManager final {
public:
  static constexpr uint32_t kSlotBytes = 4;         // one 32-bit lane per slot
  static constexpr uint32_t kDbiZoneAlignment = 16; // zone start alignment

  SpillManager(uint32_t original_private_bytes, uint32_t per_lane_scratch_limit);

  std::optional<uint32_t> allocate_slot(RegisterRef reg);
  std::optional<uint32_t> allocate_slots(RegisterRef reg, unsigned width);
  bool                    reserve(const RegisterSet &set);
  uint32_t                total_private_bytes() const;
  std::optional<uint32_t> offset_for(RegisterRef reg) const;
};
```

`allocate_slots` is the common multi-lane case (SGPR pair, 64-bit VGPR pair). `reserve` performs an upfront capacity check across the whole set so a partial allocation can never become visible. `offset_for` is the lookup used by code generators when emitting the matching `scratch_load` after a probe.

---

## Supporting Types

### RegisterRef / RegisterSet [shared with DBT]

**Files:** `isa/register_set.h`, `isa/register_set.cpp`
**Used by:** DBT semantic translator, DBI SpillManager and liveness

ISA-independent register-file model. `RegisterRef` is `(RegClass, uint16_t index, uint8_t width)` measured in 32-bit lanes. `RegisterSet` is three disjoint bitsets (SGPR / VGPR / ACC_VGPR) sized to the union of CDNA and RDNA hardware bounds (`REGISTER_SET_MAX_*`). For scratch selection across both families, `REGISTER_SET_ALLOCATABLE_SGPRS` gives the conservative `min(CDNA, RDNA)` bound.

`RegisterSet` exposes `expand` / `erase` / `contains` / `none` / `size` / `intersects`, the standard set operators (`|=`, `&=`, `-=`), and a `for_each` visitor that yields tracked single-lane `RegisterRef`s in (SGPR, VGPR, AccVGPR) ascending-index order.
