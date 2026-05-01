# Design: Remove CL Types from hipamd

**Branch:** `users/cpaquot/rocclr-phase1-type-replacement`
**Date:** 2026-05-01
**Scope:** `hipamd/src/hip_conversions.hpp`, `hip_event.hpp`, `hip_memory.cpp`, `hip_texture.cpp`, `hip_device_runtime.cpp`

## Alternatives Considered

**Keep `getCL*` function names, only change return types:** Simpler diff, but leaves
`getCLChannelType` returning `amd::ChannelDataType` — a misleading name that would require a
follow-up rename pass. Rejected as incomplete.

**Replace CL types with a minimal local stub (no cl_common.hpp, no amd_types.hpp):** Define
`typedef int cl_channel_type` locally to avoid changing return types. Removes the header
dependency but leaves CL names in the codebase. Rejected as cosmetic.

**Include hip_gl.cpp in scope:** GL interop works by creating a real OpenCL context via
`cl_context_properties`. That usage is the actual interop mechanism, not an internal type leak.
Deferred to a later phase.

## Architecture

Two commits. Commit 1 refactors the conversion hub (`hip_conversions.hpp`, `hip_event.hpp`)
so callers can be updated against a stable new API. Commit 2 updates all callers.

### Commit 1: `hip_conversions.hpp` + `hip_event.hpp`

**Conversion function rename and return type change:**

| Old name | New name | Return type |
|---|---|---|
| `getCLChannelType` | `getAMDChannelDataType` | `cl_channel_type` → `amd::ChannelDataType` |
| `getCLChannelOrder` | `getAMDChannelOrder` | `cl_channel_order` → `amd::ChannelOrder` |
| `getCLMemObjectType` (3 overloads) | `getAMDMemObjectType` | `cl_mem_object_type` → `amd::MemObjectType` |
| `getCLAddressingMode` | dropped | callers use `hipTextureAddressMode` directly |
| `getCLFilterMode` | dropped | callers use `hipTextureFilterMode` directly |
| `getCL2hipArrayFormat` | `getHipArrayFormat` | param `cl_channel_type` → `amd::ChannelDataType` |

Function bodies swap CL constants for numerically identical `amd::` enum values
(`CL_R` → `amd::ChannelOrder::R`, `CL_FLOAT` → `amd::ChannelDataType::Float`, etc.).
No logic changes.

**`cl_mem` cast on line 100** — skip the `as_amd()` intermediary:
```cpp
// Before
const cl_mem dstMemObj = reinterpret_cast<const cl_mem>(arr->data);
const amd::Image* dstImage = as_amd(dstMemObj)->asImage();
// After
const amd::Image* dstImage = reinterpret_cast<amd::Memory*>(arr->data)->asImage();
```

**Header cleanup:** Drop `#include "cl_common.hpp"`. Add `#include "rocclr/include/amd_types.hpp"`.

**`hip_event.hpp`:** The `cl_event` parameter in the callback typedef becomes `void*`, consistent
with the `vdi_agent_amd.h` change in the rocclr phase.

Files changed:
- `hipamd/src/hip_conversions.hpp`
- `hipamd/src/hip_event.hpp`

### Commit 2: Caller updates

**`hip_texture.cpp`:** Replace `getCL*` call sites with `getAMD*`. Results flow directly into
`amd::Image` constructors that already accept `amd::ImageFormat` and `amd::MemObjectType` — no
further casting needed. Replace `cl_addressing_mode` and `cl_filter_mode` locals with
`hipTextureAddressMode` / `hipTextureFilterMode` used directly at the `amd::Sampler` API, or
cast inline to the `amd::` equivalents. Drop `#include "cl_common.hpp"`.

**`hip_memory.cpp`:** Replace all `cl_mem` casts with `amd::Memory*` casts (same pattern as
`hip_conversions.hpp`). Replace `getCL*` call sites with `getAMD*`. Drop `#include "cl_common.hpp"`.

**`hip_device_runtime.cpp`:** Replace `cl_int` and `cl_uint` loop/counter variables with
`int32_t` and `uint32_t`. Drop any `cl_common.hpp` include if present.

Files changed:
- `hipamd/src/hip_texture.cpp`
- `hipamd/src/hip_memory.cpp`
- `hipamd/src/hip_device_runtime.cpp`

## Commit Plan

| # | Commit | Files |
|---|--------|-------|
| 1 | `hipamd: refactor hip_conversions.hpp to use amd:: types` | `hip_conversions.hpp`, `hip_event.hpp` |
| 2 | `hipamd: remove cl_common.hpp from memory, texture, and device runtime` | `hip_memory.cpp`, `hip_texture.cpp`, `hip_device_runtime.cpp` |

## Testing

Build HIP (`make -j$(nproc) amdhip64` in `projects/clr/build`) after each commit. No functional
change is intended — all `amd::` enum values are numerically identical to their CL counterparts.
