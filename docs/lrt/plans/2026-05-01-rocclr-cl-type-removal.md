# rocclr CL Type Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use lrt-rocm:subagent-driven-development (recommended) or lrt-rocm:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all remaining raw `cl_*` types from rocclr internals by replacing VDI agent callback typedefs with `void*` and migrating PAL image format types to their `amd::` equivalents.

**Architecture:** Two independent commit clusters: (1) `vdi_agent_amd.h` callback params `cl_*` → `void*` plus `agent.cpp` cast removal; (2) PAL image format cluster (`cl_image_format` → `amd::ImageFormat`, `cl_mem_object_type` → `amd::MemObjectType`) plus `rocdevice.cpp` and `kernel.cpp` cleanups.

**Tech Stack:** C++17, ROCm CLR monorepo, PAL GPU backend

**Worktree:** `/home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement`
**Build dir:** `projects/clr/build` (relative to worktree root), uses `make -j$(nproc)`

---

## Files Modified

| File | Change |
|------|--------|
| `rocclr/include/vdi_agent_amd.h` | Drop `<CL/cl.h>`; all callback params `cl_*` → `void*`; `cl_command_type` → `uint32_t` |
| `rocclr/platform/agent.cpp` | Remove `static_cast<cl_*>` at 19 callback invocation sites |
| `rocclr/device/pal/palresource.hpp` | `format_` field `cl_image_format` → `amd::ImageFormat`; `topology_` field `cl_mem_object_type` → `amd::MemObjectType`; constructor param types |
| `rocclr/device/pal/palresource.cpp` | Constructor param types; `GetHSAImageFormatType`/`GetHSAImageOrderType` param types and index expressions; `desc_.format_` field access; default init |
| `rocclr/device/pal/palmemory.hpp` | Two `Memory` constructor params; two `Image` constructor params |
| `rocclr/device/pal/palmemory.cpp` | Two constructor definition param types |
| `rocclr/device/pal/palblit.hpp` | `createView` param `cl_image_format` → `amd::ImageFormat` |
| `rocclr/device/pal/palblit.cpp` | `createView` definition param; 5 call sites; `sizeof(cl_mem)` → `sizeof(void*)` |
| `rocclr/device/pal/paldevice.hpp` | `resGLAssociate` param `cl_image_format&` → `amd::ImageFormat&` |
| `rocclr/device/pal/paldevice.cpp` | 2 `pal::Image` constructor call sites |
| `rocclr/device/pal/paldevicegl.cpp` | `resGLAssociate` definition param |
| `rocclr/device/rocm/rocdevice.cpp` | `int cl_error` → `amd::Status cl_error`; CL macro values → `amd::Status` enumerators |
| `rocclr/platform/kernel.cpp` | `cl_mem*` / `cl_sampler*` casts → `amd::Memory**` / `amd::Sampler**` |

---

## Task 1: vdi_agent_amd.h — Replace cl_* callback params with void*

**Files:**
- Modify: `rocclr/include/vdi_agent_amd.h`

All paths below are relative to the worktree root.

- [ ] **Step 1: Replace the two includes at the top**

In `rocclr/include/vdi_agent_amd.h`, replace:
```c
#include <CL/cl.h>
#include "amdocl/cl_icd_amd.h"
#include <stdint.h>
```
With:
```c
#include "amdocl/cl_icd_amd.h"
#include <stdint.h>
```
(`amdocl/cl_icd_amd.h` is retained because `_vdi_agent::GetPlatform` returns `cl_platform_id*` and `GetICDDispatchTable`/`SetICDDispatchTable` use `cl_icd_dispatch_table*`.)

- [ ] **Step 2: Replace all object-typed callback params with void***

Replace every callback typedef in the file. The complete new set (replace the entire typedef block, lines 24–85):

```c
/* Context Callbacks */

typedef void(CL_CALLBACK* acContextCreate_fn)(vdi_agent* /* agent */, void* /* context */);

typedef void(CL_CALLBACK* acContextFree_fn)(vdi_agent* /* agent */, void* /* context */);

/* Command Queue Callbacks */

typedef void(CL_CALLBACK* acCommandQueueCreate_fn)(vdi_agent* /* agent */,
                                                   void* /* queue */);

typedef void(CL_CALLBACK* acCommandQueueFree_fn)(vdi_agent* /* agent */,
                                                 void* /* queue */);

/* Event Callbacks */

typedef void(CL_CALLBACK* acEventCreate_fn)(vdi_agent* /* agent */, void* /* event */,
                                            uint32_t /* type */);

typedef void(CL_CALLBACK* acEventFree_fn)(vdi_agent* /* agent */, void* /* event */);

typedef void(CL_CALLBACK* acEventStatusChanged_fn)(vdi_agent* /* agent */, void* /* event */,
                                                   int32_t /* execution_status */,
                                                   int64_t /* epoch_time_stamp */);

/* Memory Object Callbacks */

typedef void(CL_CALLBACK* acMemObjectCreate_fn)(vdi_agent* /* agent */, void* /* memobj */);

typedef void(CL_CALLBACK* acMemObjectFree_fn)(vdi_agent* /* agent */, void* /* memobj */);

typedef void(CL_CALLBACK* acMemObjectAcquired_fn)(vdi_agent* /* agent */, void* /* memobj */,
                                                  void* /* device */,
                                                  int64_t /* elapsed_time */);

/* Sampler Callbacks */

typedef void(CL_CALLBACK* acSamplerCreate_fn)(vdi_agent* /* agent */, void* /* sampler */);

typedef void(CL_CALLBACK* acSamplerFree_fn)(vdi_agent* /* agent */, void* /* sampler */);

/* Program Callbacks */

typedef void(CL_CALLBACK* acProgramCreate_fn)(vdi_agent* /* agent */, void* /* program */);

typedef void(CL_CALLBACK* acProgramFree_fn)(vdi_agent* /* agent */, void* /* program */);

typedef void(CL_CALLBACK* acProgramBuild_fn)(vdi_agent* /* agent */, void* /* program */);

/* Kernel Callbacks */

typedef void(CL_CALLBACK* acKernelCreate_fn)(vdi_agent* /* agent */, void* /* kernel */);

typedef void(CL_CALLBACK* acKernelFree_fn)(vdi_agent* /* agent */, void* /* kernel */);

typedef void(CL_CALLBACK* acKernelSetArg_fn)(vdi_agent* /* agent */, void* /* kernel */,
                                             int32_t /* arg_index */, size_t /* size */,
                                             const void* /* value_ptr */);
```

---

## Task 2: agent.cpp — Remove static_cast<cl_*> at callback invocation sites

**Files:**
- Modify: `rocclr/platform/agent.cpp`

- [ ] **Step 1: Remove casts in all post* functions**

The 19 call sites in `agent.cpp` (lines ~296–455) cast `void*` to CL types before passing to callbacks. After Task 1's typedef change, the callbacks accept `void*` directly. Replace the entire block of `post*` function bodies:

```cpp
void Agent::postContextCreate(void* context) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acContextCreate_fn callback = agent->callbacks_.ContextCreate;
    if (callback != NULL && agent->canGenerateContextEvents()) {
      callback(agent, context);
    }
  }
}

void Agent::postContextFree(void* context) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acContextFree_fn callback = agent->callbacks_.ContextFree;
    if (callback != NULL && agent->canGenerateContextEvents()) {
      callback(agent, context);
    }
  }
}

void Agent::postCommandQueueCreate(void* queue) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acCommandQueueCreate_fn callback = agent->callbacks_.CommandQueueCreate;
    if (callback != NULL && agent->canGenerateCommandQueueEvents()) {
      callback(agent, queue);
    }
  }
}

void Agent::postCommandQueueFree(void* queue) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acCommandQueueFree_fn callback = agent->callbacks_.CommandQueueFree;
    if (callback != NULL && agent->canGenerateCommandQueueEvents()) {
      callback(agent, queue);
    }
  }
}

void Agent::postEventCreate(void* event, amd::CommandType type) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acEventCreate_fn callback = agent->callbacks_.EventCreate;
    if (callback != NULL && agent->canGenerateEventEvents()) {
      callback(agent, event, static_cast<uint32_t>(type));
    }
  }
}

void Agent::postEventFree(void* event) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acEventFree_fn callback = agent->callbacks_.EventFree;
    if (callback != NULL && agent->canGenerateEventEvents()) {
      callback(agent, event);
    }
  }
}

void Agent::postEventStatusChanged(void* event, int32_t status, int64_t ts) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acEventStatusChanged_fn callback = agent->callbacks_.EventStatusChanged;
    if (callback != NULL && agent->canGenerateEventEvents()) {
      callback(agent, event, status, ts);
    }
  }
}

void Agent::postMemObjectCreate(void* memobj) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acMemObjectCreate_fn callback = agent->callbacks_.MemObjectCreate;
    if (callback != NULL && agent->canGenerateMemObjectEvents()) {
      callback(agent, memobj);
    }
  }
}

void Agent::postMemObjectFree(void* memobj) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acMemObjectFree_fn callback = agent->callbacks_.MemObjectFree;
    if (callback != NULL && agent->canGenerateMemObjectEvents()) {
      callback(agent, memobj);
    }
  }
}

void Agent::postMemObjectAcquired(void* memobj, void* device, int64_t elapsed) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acMemObjectAcquired_fn callback = agent->callbacks_.MemObjectAcquired;
    if (callback != NULL && agent->canGenerateMemObjectEvents()) {
      callback(agent, memobj, device, elapsed);
    }
  }
}

void Agent::postSamplerCreate(void* sampler) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acSamplerCreate_fn callback = agent->callbacks_.SamplerCreate;
    if (callback != NULL && agent->canGenerateSamplerEvents()) {
      callback(agent, sampler);
    }
  }
}

void Agent::postSamplerFree(void* sampler) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acSamplerFree_fn callback = agent->callbacks_.SamplerFree;
    if (callback != NULL && agent->canGenerateSamplerEvents()) {
      callback(agent, sampler);
    }
  }
}

void Agent::postProgramCreate(void* program) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acProgramCreate_fn callback = agent->callbacks_.ProgramCreate;
    if (callback != NULL && agent->canGenerateProgramEvents()) {
      callback(agent, program);
    }
  }
}

void Agent::postProgramFree(void* program) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acProgramFree_fn callback = agent->callbacks_.ProgramFree;
    if (callback != NULL && agent->canGenerateProgramEvents()) {
      callback(agent, program);
    }
  }
}

void Agent::postProgramBuild(void* program) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acProgramBuild_fn callback = agent->callbacks_.ProgramBuild;
    if (callback != NULL && agent->canGenerateProgramEvents()) {
      callback(agent, program);
    }
  }
}

void Agent::postKernelCreate(void* kernel) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acKernelCreate_fn callback = agent->callbacks_.KernelCreate;
    if (callback != NULL && agent->canGenerateKernelEvents()) {
      callback(agent, kernel);
    }
  }
}

void Agent::postKernelFree(void* kernel) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acKernelFree_fn callback = agent->callbacks_.KernelFree;
    if (callback != NULL && agent->canGenerateKernelEvents()) {
      callback(agent, kernel);
    }
  }
}

void Agent::postKernelSetArg(void* kernel, int32_t index, size_t size, const void* value_ptr) {
  for (Agent* agent = list_; agent != NULL; agent = agent->next_) {
    acKernelSetArg_fn callback = agent->callbacks_.KernelSetArg;
    if (callback != NULL && agent->canGenerateKernelEvents()) {
      callback(agent, kernel, index, size, value_ptr);
    }
  }
}
```

- [ ] **Step 2: Build and verify**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement/projects/clr/build
make -j$(nproc) 2>&1 | tail -20
```
Expected: `[100%] Built target amdhip64` with no errors.

- [ ] **Step 3: Commit**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement
git add projects/clr/rocclr/include/vdi_agent_amd.h projects/clr/rocclr/platform/agent.cpp
git commit -m "rocclr: replace cl_* VDI agent callback params with void*

vdi_agent_amd.h no longer includes <CL/cl.h>. All object handle
params in callback typedefs are now void*; cl_command_type becomes
uint32_t. agent.cpp removes the static_cast<cl_*> at each invocation
site, passing void* through directly."
```

---

## Task 3: palresource.hpp/cpp — Replace cl_image_format and cl_mem_object_type

**Files:**
- Modify: `rocclr/device/pal/palresource.hpp`
- Modify: `rocclr/device/pal/palresource.cpp`

- [ ] **Step 1: Update palresource.hpp — struct fields and constructor**

In `palresource.hpp`, the `Descriptor` struct (around line 194) has:
```cpp
cl_image_format format_;       //!< CL image format
cl_mem_object_type topology_;  //!< CL mem object type
```
Replace with:
```cpp
amd::ImageFormat format_;      //!< image format
amd::MemObjectType topology_;  //!< mem object type
```

The Image Resource constructor declaration (around line 225):
```cpp
Resource(const Device& gpuDev,
         size_t width,
         size_t height,
         size_t depth,
         cl_image_format format,        //!< resource format
         cl_mem_object_type imageType,  //!< CL image type
         uint mipLevels = 1
);
```
Replace with:
```cpp
Resource(const Device& gpuDev,
         size_t width,
         size_t height,
         size_t depth,
         amd::ImageFormat format,       //!< resource format
         amd::MemObjectType imageType,  //!< image type
         uint mipLevels = 1
);
```

- [ ] **Step 2: Update palresource.cpp — constructor definition**

The constructor definition (around line 318):
```cpp
Resource::Resource(const Device& gpuDev, size_t width, size_t height, size_t depth,
                   cl_image_format format, cl_mem_object_type imageType, uint mipLevels)
```
Replace with:
```cpp
Resource::Resource(const Device& gpuDev, size_t width, size_t height, size_t depth,
                   amd::ImageFormat format, amd::MemObjectType imageType, uint mipLevels)
```

- [ ] **Step 3: Update palresource.cpp — default format_ initialization**

Around line 298–299, the 1D buffer constructor initializes `desc_.format_`:
```cpp
desc_.format_.image_channel_order = CL_R;
desc_.format_.image_channel_data_type = CL_FLOAT;
```
Replace with:
```cpp
desc_.format_ = {amd::ChannelOrder::R, amd::ChannelDataType::Float};
```

- [ ] **Step 4: Update palresource.cpp — GetHSAImageFormatType**

Around line 396, replace:
```cpp
static uint32_t GetHSAImageFormatType(const cl_image_format& format) {
```
with:
```cpp
static uint32_t GetHSAImageFormatType(const amd::ImageFormat& format) {
```
And replace the index expression at the end of the function:
```cpp
uint idx = format.image_channel_data_type - CL_SNORM_INT8;
assert((idx <= (CL_UNORM_INT24 - CL_SNORM_INT8)) && "Out of range format channel!");
```
with:
```cpp
uint idx = static_cast<uint32_t>(format.channelDataType) - static_cast<uint32_t>(amd::ChannelDataType::SNormInt8);
assert((idx <= (static_cast<uint32_t>(amd::ChannelDataType::UNormInt24) - static_cast<uint32_t>(amd::ChannelDataType::SNormInt8))) && "Out of range format channel!");
```

- [ ] **Step 5: Update palresource.cpp — GetHSAImageOrderType**

Around line 420, replace:
```cpp
static uint32_t GetHSAImageOrderType(const cl_image_format& format) {
```
with:
```cpp
static uint32_t GetHSAImageOrderType(const amd::ImageFormat& format) {
```
And replace the index expression:
```cpp
uint idx = format.image_channel_order - CL_R;
assert((idx <= (CL_ABGR - CL_R)) && "Out of range format order!");
```
with:
```cpp
uint idx = static_cast<uint32_t>(format.channelOrder) - static_cast<uint32_t>(amd::ChannelOrder::R);
assert((idx <= (static_cast<uint32_t>(amd::ChannelOrder::ABGR) - static_cast<uint32_t>(amd::ChannelOrder::R))) && "Out of range format order!");
```

- [ ] **Step 6: Update palresource.cpp — format_ field accesses**

There are multiple sites that read `desc().format_.image_channel_order` and `desc().format_.image_channel_data_type` with static_cast wrappers (lines ~510, 748, 794, 1218). Each looks like:
```cpp
static_cast<amd::ChannelOrder>(desc().format_.image_channel_order),
static_cast<amd::ChannelDataType>(desc().format_.image_channel_data_type)
```
Replace each with:
```cpp
desc().format_.channelOrder,
desc().format_.channelDataType
```
(The cast is no longer needed since the field is already `amd::ChannelOrder`/`amd::ChannelDataType`.)

- [ ] **Step 7: Update palresource.cpp — depth/stencil format comparison**

Around line 1038–1039:
```cpp
if ((desc().format_.image_channel_order == CL_DEPTH_STENCIL) &&
    (desc().format_.image_channel_data_type == CL_UNORM_INT24)) {
```
Replace with:
```cpp
if ((desc().format_.channelOrder == amd::ChannelOrder::DepthStencil) &&
    (desc().format_.channelDataType == amd::ChannelDataType::UNormInt24)) {
```

---

## Task 4: palmemory.hpp/cpp — Replace constructor params

**Files:**
- Modify: `rocclr/device/pal/palmemory.hpp`
- Modify: `rocclr/device/pal/palmemory.cpp`

- [ ] **Step 1: Update palmemory.hpp — Memory constructors**

Two `Memory` constructor declarations (around lines 50–51 and 61–62):
```cpp
cl_image_format format,        //!< Memory format
cl_mem_object_type imageType,  //!< CL image type
```
Replace both with:
```cpp
amd::ImageFormat format,       //!< Memory format
amd::MemObjectType imageType,  //!< image type
```

Two `Image` constructor declarations (around lines 218–219 and 231–232) — same replacement.

- [ ] **Step 2: Update palmemory.cpp — constructor definitions**

Two constructor definitions (around lines 44 and 55) have the same `cl_image_format format, cl_mem_object_type imageType` params. Replace both with:
```cpp
amd::ImageFormat format, amd::MemObjectType imageType
```

---

## Task 5: palblit.hpp/cpp — createView and sizeof

**Files:**
- Modify: `rocclr/device/pal/palblit.hpp`
- Modify: `rocclr/device/pal/palblit.cpp`

- [ ] **Step 1: Update palblit.hpp — createView declaration**

Around line 517–518:
```cpp
Memory* createView(const Memory& parent,         //!< Parent memory object
                   const cl_image_format format  //!< The new format for a view
```
Replace with:
```cpp
Memory* createView(const Memory& parent,          //!< Parent memory object
                   const amd::ImageFormat format  //!< The new format for a view
```

- [ ] **Step 2: Update palblit.cpp — createView definition**

Around line 2669:
```cpp
Memory* KernelBlitManager::createView(const Memory& parent, const cl_image_format format) const {
```
Replace with:
```cpp
Memory* KernelBlitManager::createView(const Memory& parent, const amd::ImageFormat format) const {
```
The body passes `format` and `parent.desc().topology_` to the `Image` constructor — those are now typed correctly and need no other changes.

- [ ] **Step 3: Update palblit.cpp — createView call sites**

Five call sites construct `cl_image_format{cast, cast}` from an `amd::ImageFormat`. Replace each one. All follow the same pattern:

Line 1068:
```cpp
dstView = createView(gpuMem(dstMemory), cl_image_format{static_cast<uint32_t>(newFormat.channelOrder), static_cast<uint32_t>(newFormat.channelDataType)});
```
→
```cpp
dstView = createView(gpuMem(dstMemory), newFormat);
```

Line 1386:
```cpp
srcView = createView(gpuMem(srcMemory), cl_image_format{static_cast<uint32_t>(newFormat.channelOrder), static_cast<uint32_t>(newFormat.channelDataType)});
```
→
```cpp
srcView = createView(gpuMem(srcMemory), newFormat);
```

Line 1561:
```cpp
srcView = createView(gpuMem(srcMemory), cl_image_format{static_cast<uint32_t>(srcFormat.channelOrder), static_cast<uint32_t>(srcFormat.channelDataType)});
```
→
```cpp
srcView = createView(gpuMem(srcMemory), srcFormat);
```

Line 1569:
```cpp
dstView = createView(gpuMem(dstMemory), cl_image_format{static_cast<uint32_t>(dstFormat.channelOrder), static_cast<uint32_t>(dstFormat.channelDataType)});
```
→
```cpp
dstView = createView(gpuMem(dstMemory), dstFormat);
```

Line 2345:
```cpp
memView = createView(gpuMem(memory), cl_image_format{static_cast<uint32_t>(newFormat.channelOrder), static_cast<uint32_t>(newFormat.channelDataType)});
```
→
```cpp
memView = createView(gpuMem(memory), newFormat);
```

- [ ] **Step 4: Update palblit.cpp — sizeof(cl_mem) → sizeof(void*)**

Around line 398 in `kernel.cpp` (this is actually in `palblit.cpp` — check the grep output):
```cpp
lastSize = sizeof(cl_mem);
```
Replace with:
```cpp
lastSize = sizeof(void*);
```

---

## Task 6: paldevice.hpp/cpp + paldevicegl.cpp — call sites

**Files:**
- Modify: `rocclr/device/pal/paldevice.hpp`
- Modify: `rocclr/device/pal/paldevice.cpp`
- Modify: `rocclr/device/pal/paldevicegl.cpp`

- [ ] **Step 1: Update paldevice.hpp — resGLAssociate declaration**

Around line 609:
```cpp
void** mbResHandle, size_t* offset, cl_image_format& newClFormat
```
Replace with:
```cpp
void** mbResHandle, size_t* offset, amd::ImageFormat& newClFormat
```

- [ ] **Step 2: Update paldevicegl.cpp — resGLAssociate definition**

Around line 829:
```cpp
void** mbResHandle, size_t* offset, cl_image_format& newClFormat
```
Replace with:
```cpp
void** mbResHandle, size_t* offset, amd::ImageFormat& newClFormat
```

- [ ] **Step 3: Update paldevice.cpp — pal::Image constructor call sites**

Two sites construct `cl_image_format{...}` and `static_cast<cl_mem_object_type>(...)`. Both follow this pattern.

Line ~1694:
```cpp
gpuImage = new pal::Image(*this, owner, image.getWidth(), image.getHeight(), image.getDepth(),
                          cl_image_format{static_cast<uint32_t>(image.getImageFormat().channelOrder),
                                          static_cast<uint32_t>(image.getImageFormat().channelDataType)},
                          static_cast<cl_mem_object_type>(image.getType()), image.getMipLevels());
```
Replace with:
```cpp
gpuImage = new pal::Image(*this, owner, image.getWidth(), image.getHeight(), image.getDepth(),
                          image.getImageFormat(), image.getType(), image.getMipLevels());
```

Line ~1853 (inside `Device::createView`):
```cpp
pal::Memory* gpuImage =
    new pal::Image(*this, owner, image.getWidth(), image.getHeight(), image.getDepth(),
                   cl_image_format{static_cast<uint32_t>(image.getImageFormat().channelOrder),
                                   static_cast<uint32_t>(image.getImageFormat().channelDataType)},
                   static_cast<cl_mem_object_type>(image.getType()), image.getMipLevels());
```
Replace with:
```cpp
pal::Memory* gpuImage =
    new pal::Image(*this, owner, image.getWidth(), image.getHeight(), image.getDepth(),
                   image.getImageFormat(), image.getType(), image.getMipLevels());
```

---

## Task 7: rocdevice.cpp + kernel.cpp — remaining cleanups

**Files:**
- Modify: `rocclr/device/rocm/rocdevice.cpp`
- Modify: `rocclr/platform/kernel.cpp`

- [ ] **Step 1: Update rocdevice.cpp — cl_error variable**

The `ConvertHSAErrorIntoCLError` function (around line 3888) uses `int cl_error` with raw integer literals. Replace the entire function:

```cpp
amd::Status ConvertHSAErrorIntoCLError(hsa_status_t hsa_status) {
  amd::Status cl_error = amd::Status::Success;
  switch (hsa_status) {
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:
      cl_error = amd::Status::OutOfResources;
      break;
    case HSA_STATUS_ERROR_EXCEPTION:
      cl_error = amd::Status::InvalidOperation;
      break;
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
      cl_error = amd::Status::InvalidArgValue;
      break;
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:
      cl_error = amd::Status::MemObjectAllocationFailure;
      break;
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
      cl_error = amd::Status::InvalidProgram;
      break;
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
      cl_error = amd::Status::InvalidOperation;
      break;
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:
      cl_error = amd::Status::InvalidArgValue;
      break;
    case HSA_STATUS_ERROR_INVALID_ISA:
      cl_error = amd::Status::InvalidKernel;
      break;
    case (hsa_status_t)HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION:
      cl_error = amd::Status::BuildProgramFailure;
      break;
    case (hsa_status_t)HSA_STATUS_ERROR_MEMORY_FAULT:
      cl_error = amd::Status::InvalidMemObject;
      break;
    case (hsa_status_t)HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION:
      cl_error = amd::Status::InvalidMemObject;
      break;
    case HSA_STATUS_ERROR:
    default:
      cl_error = amd::Status::DeviceNotAvailable;
      break;
  }
  return cl_error;
}
```

`ConvertHSAErrorIntoCLError` has no header declaration — it is defined and called only within `rocdevice.cpp`. The single call site (around line 3976) assigns its return value into `amd::Device::gpu_error_`, which is `static int` in `device/device.hpp`. Change that assignment to cast the `amd::Status` result back to `int`:
```cpp
amd::Device::gpu_error_ = static_cast<int>(ConvertHSAErrorIntoCLError(status));
```

- [ ] **Step 2: Update kernel.cpp — cl_mem* and cl_sampler* casts**

Around line 161–172, the kernel argument parsing block:
```cpp
} else if ((value == NULL) || (static_cast<const cl_mem*>(value) == NULL)) {
  desc.info_.rawPointer_ = false;
  memoryObjects_[desc.info_.arrayIndex_] = nullptr;
} else {
  desc.info_.rawPointer_ = false;
  memoryObjects_[desc.info_.arrayIndex_] =
      reinterpret_cast<amd::Memory*>(*static_cast<const cl_mem*>(value));
}
} else if (desc.type_ == amd::KernelArgValueType::Sampler) {
  samplerObjects_[desc.info_.arrayIndex_] =
      reinterpret_cast<amd::Sampler*>(*static_cast<const cl_sampler*>(value));
```
Replace with:
```cpp
} else if ((value == NULL) || (static_cast<const amd::Memory* const*>(value) == NULL)) {
  desc.info_.rawPointer_ = false;
  memoryObjects_[desc.info_.arrayIndex_] = nullptr;
} else {
  desc.info_.rawPointer_ = false;
  memoryObjects_[desc.info_.arrayIndex_] =
      *reinterpret_cast<amd::Memory* const*>(value);
}
} else if (desc.type_ == amd::KernelArgValueType::Sampler) {
  samplerObjects_[desc.info_.arrayIndex_] =
      *reinterpret_cast<amd::Sampler* const*>(value);
```

- [ ] **Step 3: Update kernel.cpp — sizeof(cl_mem)**

Around line 398:
```cpp
lastSize = sizeof(cl_mem);
```
Replace with:
```cpp
lastSize = sizeof(void*);
```

- [ ] **Step 4: Build and verify**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement/projects/clr/build
make -j$(nproc) 2>&1 | tail -20
```
Expected: `[100%] Built target amdhip64` with no errors.

- [ ] **Step 5: Commit**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement
git add \
  projects/clr/rocclr/device/pal/palresource.hpp \
  projects/clr/rocclr/device/pal/palresource.cpp \
  projects/clr/rocclr/device/pal/palmemory.hpp \
  projects/clr/rocclr/device/pal/palmemory.cpp \
  projects/clr/rocclr/device/pal/palblit.hpp \
  projects/clr/rocclr/device/pal/palblit.cpp \
  projects/clr/rocclr/device/pal/paldevice.hpp \
  projects/clr/rocclr/device/pal/paldevice.cpp \
  projects/clr/rocclr/device/pal/paldevicegl.cpp \
  projects/clr/rocclr/device/rocm/rocdevice.cpp \
  projects/clr/rocclr/platform/kernel.cpp
git commit -m "rocclr/pal: replace cl_image_format and cl_mem_object_type with amd:: types

- cl_image_format -> amd::ImageFormat in palresource, palmemory,
  palblit (createView + 5 call sites), paldevice, paldevicegl
- cl_mem_object_type -> amd::MemObjectType in same files
- sizeof(cl_mem) -> sizeof(void*) in palblit and kernel
- int cl_error -> amd::Status in rocdevice ConvertHSAErrorIntoCLError
- cl_mem*/cl_sampler* casts in kernel.cpp -> amd::Memory**/amd::Sampler**"
```
