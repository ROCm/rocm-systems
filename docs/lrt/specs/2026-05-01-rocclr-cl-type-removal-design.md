# Design: Remove CL Types from rocclr

**Branch:** `users/cpaquot/rocclr-phase1-type-replacement`
**Date:** 2026-05-01
**Scope:** rocclr internal, vdi_agent_amd.h, opencl/cltrace, hipamd/kernel.cpp

## Alternatives Considered

**Leave agent.cpp alone, only fix PAL:** Simpler, no API churn. Rejected because `vdi_agent_amd.h`
pulling in `<CL/cl.h>` defeats the goal of making rocclr CL-free.

**Use `reinterpret_cast` in agent.cpp without changing `vdi_agent_amd.h`:** Removes the visible CL
type at the call site but doesn't remove the CL header dependency. Rejected as cosmetic.

**Exclude D3D interop files:** Accepted. `interop_d3d9/10/11` straddle the OpenCL extension API
boundary and are deferred to a later phase.

## Architecture

Two independent clusters of changes, delivered as separate commits.

### Cluster 1: vdi_agent_amd.h + agent.cpp

The `post*` functions in `rocclr/platform/agent.cpp` already accept `void*`. The only remaining
CL types appear in the function bodies as `static_cast<cl_foo>(ptr)` before invoking callbacks.
Those casts exist because the callback typedefs in `vdi_agent_amd.h` declare `cl_context`,
`cl_sampler`, `cl_event`, etc. in their signatures.

**Change:** Replace all `cl_*` object parameters in `vdi_agent_amd.h` callback typedefs with
`void*`. Remove the `static_cast<cl_*>` casts in `agent.cpp` — the `void*` passes through
directly.

**Header cleanup:** After removing all `cl_*` from callback typedefs, drop `#include <CL/cl.h>`
and `#include "amdocl/cl_icd_amd.h"` from `vdi_agent_amd.h`. The `_vdi_agent` struct contains
two methods that reference `cl_platform_id` and `cl_icd_dispatch_table` — these require a
targeted `#include "amdocl/cl_icd_amd.h"` to remain (or a forward declaration if feasible).

**Ripple — opencl/cltrace:** `cltrace.cpp` implements `vdiAgent_OnLoad` and registers callbacks
whose signatures must match the updated typedefs. Each callback implementation changes its
object parameter from `cl_foo` to `void*` and casts internally. This is correct: cltrace is
the OpenCL layer and the cast belongs there.

Files changed:
- `rocclr/include/vdi_agent_amd.h` — typedef parameters `cl_*` → `void*`; drop `<CL/cl.h>`
- `rocclr/platform/agent.cpp` — remove `static_cast<cl_*>` at callback invocation sites
- `opencl/tools/cltrace/cltrace.cpp` — callback implementations accept `void*`, cast internally

### Cluster 2: PAL image format + rocdevice + kernel.cpp

All target types already have `amd::` equivalents in `rocclr/include/amd_types.hpp`.

**`cl_image_format` → `amd::ImageFormat`**

`KernelBlitManager::createView(const Memory&, const cl_image_format)` in `palblit` changes to
`amd::ImageFormat`. All callers construct `cl_image_format{cast, cast}` from an existing
`amd::ImageFormat` — they pass the struct directly after the change. PAL memory and resource
constructors (`palmemory.cpp`, `palresource.cpp`) taking `cl_image_format` are updated in the
same commit.

**`cl_mem_object_type` → `amd::MemObjectType`**

Appears alongside `cl_image_format` in the same PAL constructor signatures. Replaced in the
same commit.

**`sizeof(cl_mem)` → `sizeof(void*)`**

`palblit.cpp` uses `sizeof(cl_mem)` as a kernel argument size. `cl_mem` is a pointer typedef,
so `sizeof(cl_mem) == sizeof(void*)`. Replaced with `sizeof(void*)`.

**`int cl_error` in `rocdevice.cpp`**

Local variable using bare CL macro integer values. Replaced with `amd::Status cl_error` using
the corresponding `amd::Status` enum values, which exist with matching numeric values.

**`cl_sampler*` / `cl_mem*` in `kernel.cpp`**

Kernel argument parsing casts a raw `const void*` value pointer to `const cl_mem*` or
`const cl_sampler*` to read a handle-typed pointer stored by the amdocl caller. Replaced with
`reinterpret_cast<const amd::Memory* const*>` and `reinterpret_cast<const amd::Sampler* const*>`
respectively. The cast remains correct because amdocl stores `amd::Memory*` and `amd::Sampler*`
in those argument slots.

Files changed:
- `rocclr/device/pal/palblit.hpp` / `palblit.cpp` — `cl_image_format` → `amd::ImageFormat`; `sizeof(cl_mem)` → `sizeof(void*)`
- `rocclr/device/pal/palmemory.hpp` / `palmemory.cpp` — constructor params
- `rocclr/device/pal/palresource.hpp` / `palresource.cpp` — constructor params
- `rocclr/device/pal/paldevice.hpp` / `paldevice.cpp` — `cl_image_format` at call sites
- `rocclr/device/pal/paldevicegl.cpp` — `cl_image_format` at call sites
- `rocclr/device/rocm/rocdevice.cpp` — `int cl_error` → `amd::Status cl_error`
- `rocclr/platform/kernel.cpp` — kernel arg pointer casts

## Commit Plan

| # | Commit | Files |
|---|--------|-------|
| 1 | `vdi_agent_amd.h: replace cl_* callback params with void*` | `vdi_agent_amd.h`, `agent.cpp`, `cltrace.cpp` |
| 2 | `rocclr/pal: replace cl_image_format and cl_mem_object_type with amd:: types` | PAL files + `rocdevice.cpp` + `kernel.cpp` |

## Testing

Build HIP (`make -j$(nproc)` in `projects/clr/build`) after each commit. No functional change
is intended — these are mechanical type substitutions with identical numeric representations.
