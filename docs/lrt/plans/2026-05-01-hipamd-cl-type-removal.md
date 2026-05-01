# hipamd CL Type Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use lrt-rocm:subagent-driven-development (recommended) or lrt-rocm:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all OpenCL types, enums, and includes from `clr/hipamd` (excluding `hip_gl.cpp`) by replacing them with `amd::` equivalents from `amd_types.hpp`.

**Architecture:** Two commits: Commit 1 refactors the conversion hub (`hip_conversions.hpp`, `hip_event.hpp`) and extends `amd_types.hpp` with a missing `FilterMode::None` value; Commit 2 updates all callers (`hip_memory.cpp`, `hip_texture.cpp`, `hip_device_runtime.cpp`).

**Tech Stack:** C++17, ROCm CLR (`rocclr/include/amd_types.hpp`), HIP runtime types (`hip/driver_types.h`, `hip/texture_types.h`).

---

## File Map

| File | Change |
|------|--------|
| `rocclr/include/amd_types.hpp` | Add `FilterMode::None = 0x1142` |
| `hipamd/src/hip_conversions.hpp` | Rename `getCL*` → `getAMD*`; replace CL return types with `amd::` types; drop `cl_common.hpp` |
| `hipamd/src/hip_event.hpp` | Fix `ihipStreamCallback` declaration to match existing definition |
| `hipamd/src/hip_texture.cpp` | Update callers; drop `cl_common.hpp` |
| `hipamd/src/hip_memory.cpp` | Update callers; update `ihipImageCreate`; drop `cl_common.hpp` |
| `hipamd/src/hip_device_runtime.cpp` | `cl_int`/`cl_uint` → `int32_t`/`uint32_t` |

**Build command (after each commit):**
```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement/projects/clr/build
make -j$(nproc) amdhip64 2>&1 | tail -30
```
Expected: zero errors.

---

## Task 1: Extend amd_types.hpp and refactor hip_conversions.hpp + hip_event.hpp

**Files:**
- Modify: `projects/clr/rocclr/include/amd_types.hpp` (around line 542)
- Modify: `projects/clr/hipamd/src/hip_conversions.hpp` (lines 1–179)
- Modify: `projects/clr/hipamd/src/hip_event.hpp` (line 55)

No tests exist for this layer; verification is a successful build.

- [ ] **Step 1: Add `FilterMode::None` to `amd_types.hpp`**

In `amd_types.hpp`, find the `FilterMode` enum (around line 540). Add `None` after `Linear`:

```cpp
enum class FilterMode : uint32_t {
  Nearest = 0x1140, // CL_FILTER_NEAREST
  Linear  = 0x1141, // CL_FILTER_LINEAR
  None    = 0x1142, // No filtering (HIP extension, used for mipmap base case)
};
```

- [ ] **Step 2: Replace the include and add amd_types in `hip_conversions.hpp`**

Replace line 11:
```cpp
// Before:
#include "cl_common.hpp"

// After:
#include "amd_types.hpp"
```

- [ ] **Step 3: Rename `getCLChannelType` → `getAMDChannelDataType`**

Replace lines 14–58 (the full `getCLChannelType` function) with:

```cpp
inline amd::ChannelDataType getAMDChannelDataType(const hipArray_Format hipFormat,
                                                  const hipTextureReadMode hipReadMode) {
  if (hipReadMode == hipReadModeElementType) {
    switch (hipFormat) {
      case HIP_AD_FORMAT_UNSIGNED_INT8:
        return amd::ChannelDataType::UnsignedInt8;
      case HIP_AD_FORMAT_SIGNED_INT8:
        return amd::ChannelDataType::SignedInt8;
      case HIP_AD_FORMAT_UNSIGNED_INT16:
        return amd::ChannelDataType::UnsignedInt16;
      case HIP_AD_FORMAT_SIGNED_INT16:
        return amd::ChannelDataType::SignedInt16;
      case HIP_AD_FORMAT_UNSIGNED_INT32:
        return amd::ChannelDataType::UnsignedInt32;
      case HIP_AD_FORMAT_SIGNED_INT32:
        return amd::ChannelDataType::SignedInt32;
      case HIP_AD_FORMAT_HALF:
        return amd::ChannelDataType::HalfFloat;
      case HIP_AD_FORMAT_FLOAT:
        return amd::ChannelDataType::Float;
    }
  } else if (hipReadMode == hipReadModeNormalizedFloat) {
    switch (hipFormat) {
      case HIP_AD_FORMAT_UNSIGNED_INT8:
        return amd::ChannelDataType::UNormInt8;
      case HIP_AD_FORMAT_SIGNED_INT8:
        return amd::ChannelDataType::SNormInt8;
      case HIP_AD_FORMAT_UNSIGNED_INT16:
        return amd::ChannelDataType::UNormInt16;
      case HIP_AD_FORMAT_SIGNED_INT16:
        return amd::ChannelDataType::SNormInt16;
      case HIP_AD_FORMAT_UNSIGNED_INT32:
        return amd::ChannelDataType::UnsignedInt32;
      case HIP_AD_FORMAT_SIGNED_INT32:
        return amd::ChannelDataType::SignedInt32;
      case HIP_AD_FORMAT_HALF:
        return amd::ChannelDataType::HalfFloat;
      case HIP_AD_FORMAT_FLOAT:
        return amd::ChannelDataType::Float;
    }
  }

  // error scenario
  return {};
}
```

- [ ] **Step 4: Rename `getCLChannelOrder` → `getAMDChannelOrder`**

Replace lines 60–74 with:

```cpp
inline amd::ChannelOrder getAMDChannelOrder(const unsigned int hipNumChannels, const int sRGB) {
  switch (hipNumChannels) {
    case 1:
      return amd::ChannelOrder::R;
    case 2:
      return amd::ChannelOrder::RG;
    case 4:
      return (sRGB == 1) ? amd::ChannelOrder::sRGBA : amd::ChannelOrder::RGBA;
    default:
      break;
  }

  // error scenario
  return {};
}
```

- [ ] **Step 5: Rename `getCLMemObjectType` (width/height/depth/flags) → `getAMDMemObjectType`**

Replace lines 76–97 with:

```cpp
inline amd::MemObjectType getAMDMemObjectType(const unsigned int hipWidth,
                                              const unsigned int hipHeight,
                                              const unsigned int hipDepth,
                                              const unsigned int flags) {
  if ((flags & hipArrayLayered) == hipArrayLayered) {
    if ((hipWidth != 0) && (hipHeight == 0) && (hipDepth != 0)) {
      return amd::MemObjectType::Image1DArray;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth != 0)) {
      return amd::MemObjectType::Image2DArray;
    }
  } else {
    if ((hipWidth != 0) && (hipHeight == 0) && (hipDepth == 0)) {
      return amd::MemObjectType::Image1D;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth == 0)) {
      return amd::MemObjectType::Image2D;
    } else if ((hipWidth != 0) && (hipHeight != 0) && (hipDepth != 0)) {
      return amd::MemObjectType::Image3D;
    }
  }
  // error scenario. ShouldNotReachHere()
  return amd::MemObjectType::Buffer;
}
```

- [ ] **Step 6: Rename `getCLMemObjectType(hipArray*)` → `getAMDMemObjectType(hipArray*)`**

Replace lines 99–104 with:

```cpp
inline amd::MemObjectType getAMDMemObjectType(const hipArray* arr) {
  const amd::Image* dstImage = reinterpret_cast<amd::Memory*>(arr->data)->asImage();
  return dstImage ? dstImage->getType() : amd::MemObjectType::Buffer;
}
```

- [ ] **Step 7: Update `isLayered1D`**

Replace lines 106–108 with:

```cpp
inline bool isLayered1D(const hipArray* arr) {
  return amd::MemObjectType::Image1DArray == getAMDMemObjectType(arr);
}
```

- [ ] **Step 8: Rename `getCLAddressingMode` → `getAMDAddressingMode`**

Replace lines 110–124 with:

```cpp
inline amd::AddressingMode getAMDAddressingMode(const hipTextureAddressMode hipAddressMode) {
  switch (hipAddressMode) {
    case hipAddressModeWrap:
      return amd::AddressingMode::Repeat;
    case hipAddressModeClamp:
      return amd::AddressingMode::ClampToEdge;
    case hipAddressModeMirror:
      return amd::AddressingMode::MirroredRepeat;
    case hipAddressModeBorder:
      return amd::AddressingMode::Clamp;
  }

  // error scenario
  return {};
}
```

- [ ] **Step 9: Rename `getCLFilterMode` → `getAMDFilterMode`**

Replace lines 126–136 with:

```cpp
inline amd::FilterMode getAMDFilterMode(const hipTextureFilterMode hipFilterMode) {
  switch (hipFilterMode) {
    case hipFilterModePoint:
      return amd::FilterMode::Nearest;
    case hipFilterModeLinear:
      return amd::FilterMode::Linear;
  }

  // error scenario
  return {};
}
```

- [ ] **Step 10: Rename `getCLMemObjectType(hipResourceType)` → `getAMDMemObjectType(hipResourceType)`**

Replace lines 138–150 with:

```cpp
inline amd::MemObjectType getAMDMemObjectType(const hipResourceType hipResType) {
  switch (hipResType) {
    case hipResourceTypeLinear:
      return amd::MemObjectType::Image1DBuffer;
    case hipResourceTypePitch2D:
      return amd::MemObjectType::Image2D;
    default:
      break;
  }

  // error scenario
  return {};
}
```

- [ ] **Step 11: Rename `getCL2hipArrayFormat` → `getHipArrayFormat`**

Replace lines 152–179 with:

```cpp
inline hipArray_Format getHipArrayFormat(const amd::ChannelDataType type) {
  switch (type) {
    case amd::ChannelDataType::SNormInt8:
    case amd::ChannelDataType::SignedInt8:
      return HIP_AD_FORMAT_SIGNED_INT8;

    case amd::ChannelDataType::UnsignedInt16:
      return HIP_AD_FORMAT_UNSIGNED_INT16;

    case amd::ChannelDataType::SignedInt16:
      return HIP_AD_FORMAT_SIGNED_INT16;

    case amd::ChannelDataType::SignedInt32:
      return HIP_AD_FORMAT_SIGNED_INT32;

    case amd::ChannelDataType::UnsignedInt32:
      return HIP_AD_FORMAT_UNSIGNED_INT32;

    case amd::ChannelDataType::Float:
      return HIP_AD_FORMAT_FLOAT;

    case amd::ChannelDataType::UnsignedInt8:
    case amd::ChannelDataType::UNormInt8:
    case amd::ChannelDataType::UNormInt101010:
    default:
      return HIP_AD_FORMAT_UNSIGNED_INT8;
  }
}
```

- [ ] **Step 12: Fix `ihipStreamCallback` declaration in `hip_event.hpp`**

In `hip_event.hpp` line 55, replace:
```cpp
void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data);
```
with:
```cpp
void ihipStreamCallback(void* event_handle, int32_t command_exec_status, void* user_data);
```

(This matches the definition already at `hip_stream.cpp:181`.)

- [ ] **Step 13: Build and verify**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement/projects/clr/build
make -j$(nproc) amdhip64 2>&1 | tail -30
```
Expected: zero errors.

- [ ] **Step 14: Commit**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement
git add projects/clr/rocclr/include/amd_types.hpp \
        projects/clr/hipamd/src/hip_conversions.hpp \
        projects/clr/hipamd/src/hip_event.hpp
git commit -m "hipamd: refactor hip_conversions.hpp to use amd:: types

Replace all getCL* conversion functions with getAMD* equivalents returning
amd:: types (amd::ChannelDataType, amd::ChannelOrder, amd::MemObjectType,
amd::AddressingMode, amd::FilterMode). Rename getCL2hipArrayFormat to
getHipArrayFormat with amd::ChannelDataType parameter.

Add amd::FilterMode::None (0x1142) to amd_types.hpp for the HIP mipmap
filter-mode-none sentinel value.

Fix ihipStreamCallback declaration in hip_event.hpp to match the existing
definition in hip_stream.cpp (was cl_event/cl_int, already void*/int32_t
in the definition).

Drop #include \"cl_common.hpp\" from hip_conversions.hpp."
```

---

## Task 2: Update hip_texture.cpp

**Files:**
- Modify: `projects/clr/hipamd/src/hip_texture.cpp` (lines 45–46, 167–202, 230–360)

- [ ] **Step 1: Update `ihipImageCreate` forward declaration (line 45)**

Replace lines 45–46:
```cpp
// Before:
amd::Image* ihipImageCreate(const cl_channel_order channelOrder, const cl_channel_type channelType,
                            const cl_mem_object_type imageType, const size_t imageWidth,
// After:
amd::Image* ihipImageCreate(const amd::ImageFormat fmt, const amd::MemObjectType imageType,
                            const size_t imageWidth,
```

The rest of the parameter list (imageHeight through status) is unchanged.

- [ ] **Step 2: Replace addressing mode array (lines 167–180)**

Replace lines 167–180:
```cpp
// Before:
  cl_addressing_mode addressMode[3] = {CL_ADDRESS_NONE, CL_ADDRESS_NONE, CL_ADDRESS_NONE};
  for (int i = 0; i < 3; i++) {
    if ((pTexDesc->normalizedCoords == 0) && ((pTexDesc->addressMode[i] == hipAddressModeWrap) ||
                                              (pTexDesc->addressMode[i] == hipAddressModeMirror))) {
      addressMode[i] = hip::getCLAddressingMode(hipAddressModeClamp);
    }
    else if (pResDesc->resType != hipResourceTypeLinear) {
      addressMode[i] = hip::getCLAddressingMode(pTexDesc->addressMode[i]);
    }
  }

// After:
  uint32_t addressMode[3] = {
      static_cast<uint32_t>(amd::AddressingMode::NoAddressing),
      static_cast<uint32_t>(amd::AddressingMode::NoAddressing),
      static_cast<uint32_t>(amd::AddressingMode::NoAddressing)};
  for (int i = 0; i < 3; i++) {
    if ((pTexDesc->normalizedCoords == 0) && ((pTexDesc->addressMode[i] == hipAddressModeWrap) ||
                                              (pTexDesc->addressMode[i] == hipAddressModeMirror))) {
      addressMode[i] = static_cast<uint32_t>(hip::getAMDAddressingMode(hipAddressModeClamp));
    }
    else if (pResDesc->resType != hipResourceTypeLinear) {
      addressMode[i] = static_cast<uint32_t>(hip::getAMDAddressingMode(pTexDesc->addressMode[i]));
    }
  }
```

- [ ] **Step 3: Replace filter mode variables (lines 182–196)**

Replace lines 182–196:
```cpp
// Before:
#ifndef CL_FILTER_NONE
#define CL_FILTER_NONE 0x1142
#endif
  cl_filter_mode filterMode = CL_FILTER_NONE;
  cl_filter_mode mipFilterMode = CL_FILTER_NONE;
#undef CL_FILTER_NONE
  if (pResDesc->resType != hipResourceTypeLinear) {
    filterMode = hip::getCLFilterMode(pTexDesc->filterMode);
  }

  if (pResDesc->resType == hipResourceTypeMipmappedArray) {
    mipFilterMode = hip::getCLFilterMode(pTexDesc->mipmapFilterMode);
  }

// After:
  uint32_t filterMode = static_cast<uint32_t>(amd::FilterMode::None);
  uint32_t mipFilterMode = static_cast<uint32_t>(amd::FilterMode::None);
  if (pResDesc->resType != hipResourceTypeLinear) {
    filterMode = static_cast<uint32_t>(hip::getAMDFilterMode(pTexDesc->filterMode));
  }

  if (pResDesc->resType == hipResourceTypeMipmappedArray) {
    mipFilterMode = static_cast<uint32_t>(hip::getAMDFilterMode(pTexDesc->mipmapFilterMode));
  }
```

- [ ] **Step 4: Update array resource format section (lines 234–244)**

Replace:
```cpp
// Before:
        const cl_channel_order channelOrder =
            (pResViewDesc != nullptr)
                ? hip::getCLChannelOrder(hip::getNumChannels(pResViewDesc->format), pTexDesc->sRGB)
                : hip::getCLChannelOrder(pResDesc->res.array.array->NumChannels, pTexDesc->sRGB);
        const cl_channel_type channelType =
            (pResViewDesc != nullptr)
                ? hip::getCLChannelType(hip::getArrayFormat(pResViewDesc->format), readMode)
                : hip::getCLChannelType(pResDesc->res.array.array->Format, readMode);
        const amd::Image::Format imageFormat(amd::ImageFormat{
            static_cast<amd::ChannelOrder>(channelOrder),
            static_cast<amd::ChannelDataType>(channelType)});

// After:
        const amd::ChannelOrder channelOrder =
            (pResViewDesc != nullptr)
                ? hip::getAMDChannelOrder(hip::getNumChannels(pResViewDesc->format), pTexDesc->sRGB)
                : hip::getAMDChannelOrder(pResDesc->res.array.array->NumChannels, pTexDesc->sRGB);
        const amd::ChannelDataType channelType =
            (pResViewDesc != nullptr)
                ? hip::getAMDChannelDataType(hip::getArrayFormat(pResViewDesc->format), readMode)
                : hip::getAMDChannelDataType(pResDesc->res.array.array->Format, readMode);
        const amd::Image::Format imageFormat(amd::ImageFormat{channelOrder, channelType});
```

Also find the `cl_mem_object_type imageType` and `ihipImageCreate` call in the same array block (search for `getCLMemObjectType` / `ihipImageCreate` near line 244). Replace:
```cpp
// Before:
        const cl_mem_object_type imageType = hip::getCLMemObjectType(...);
        image = ihipImageCreate(channelOrder, channelType, imageType, ...);

// After:
        const amd::MemObjectType imageType = hip::getAMDMemObjectType(...);
        image = ihipImageCreate(amd::ImageFormat{channelOrder, channelType}, imageType, ...);
```

- [ ] **Step 5: Update mipmapped array resource format section (lines 278–285)**

Replace:
```cpp
// Before:
        const cl_channel_order channelOrder =
            (pResViewDesc != nullptr)
                ? hip::getCLChannelOrder(hip::getNumChannels(pResViewDesc->format), pTexDesc->sRGB)
                : hip::getCLChannelOrder(pResDesc->res.mipmap.mipmap->num_channels, pTexDesc->sRGB);
        const cl_channel_type channelType =
            (pResViewDesc != nullptr)
                ? hip::getCLChannelType(hip::getArrayFormat(pResViewDesc->format), readMode)
                : hip::getCLChannelType(pResDesc->res.mipmap.mipmap->format, readMode);
        const amd::Image::Format imageFormat(amd::ImageFormat{
            static_cast<amd::ChannelOrder>(channelOrder),
            static_cast<amd::ChannelDataType>(channelType)});

// After:
        const amd::ChannelOrder channelOrder =
            (pResViewDesc != nullptr)
                ? hip::getAMDChannelOrder(hip::getNumChannels(pResViewDesc->format), pTexDesc->sRGB)
                : hip::getAMDChannelOrder(pResDesc->res.mipmap.mipmap->num_channels, pTexDesc->sRGB);
        const amd::ChannelDataType channelType =
            (pResViewDesc != nullptr)
                ? hip::getAMDChannelDataType(hip::getArrayFormat(pResViewDesc->format), readMode)
                : hip::getAMDChannelDataType(pResDesc->res.mipmap.mipmap->format, readMode);
        const amd::Image::Format imageFormat(amd::ImageFormat{channelOrder, channelType});
```

Also update the nearby `cl_mem_object_type imageType` and `ihipImageCreate` call:
```cpp
// Before:
        const cl_mem_object_type imageType = hip::getCLMemObjectType(...);
        image = ihipImageCreate(channelOrder, channelType, imageType, ...);

// After:
        const amd::MemObjectType imageType = hip::getAMDMemObjectType(...);
        image = ihipImageCreate(amd::ImageFormat{channelOrder, channelType}, imageType, ...);
```

- [ ] **Step 6: Update linear resource format section (lines 304–316)**

Same substitution pattern. Around line 311:
```cpp
// Before:
      const cl_channel_order channelOrder =
          hip::getCLChannelOrder(hip::getNumChannels(pResDesc->res.linear.desc), pTexDesc->sRGB);
      const cl_channel_type channelType =
          hip::getCLChannelType(hip::getArrayFormat(pResDesc->res.linear.desc), pTexDesc->readMode);
      const amd::Image::Format imageFormat(amd::ImageFormat{
          static_cast<amd::ChannelOrder>(channelOrder),
          static_cast<amd::ChannelDataType>(channelType)});
      const cl_mem_object_type imageType = hip::getCLMemObjectType(pResDesc->resType);

// After:
      const amd::ChannelOrder channelOrder =
          hip::getAMDChannelOrder(hip::getNumChannels(pResDesc->res.linear.desc), pTexDesc->sRGB);
      const amd::ChannelDataType channelType =
          hip::getAMDChannelDataType(hip::getArrayFormat(pResDesc->res.linear.desc), pTexDesc->readMode);
      const amd::Image::Format imageFormat(amd::ImageFormat{channelOrder, channelType});
      const amd::MemObjectType imageType = hip::getAMDMemObjectType(pResDesc->resType);
```

Update the `ihipImageCreate` call below to pass `amd::ImageFormat{channelOrder, channelType}`.

- [ ] **Step 7: Update pitch2D resource format section (lines 336–356)**

Same substitution pattern as Step 6 for the pitch2D block.

- [ ] **Step 8: Drop `cl_common.hpp` include**

Remove the line `#include "cl_common.hpp"` from `hip_texture.cpp`. Add `#include "amd_types.hpp"` if not already transitively included (it is via `hip_conversions.hpp`).

- [ ] **Step 9: Build and verify**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement/projects/clr/build
make -j$(nproc) amdhip64 2>&1 | tail -30
```
Expected: zero errors.

---

## Task 3: Update hip_memory.cpp

**Files:**
- Modify: `projects/clr/hipamd/src/hip_memory.cpp` (lines 991–1003, 1135–1155, 1284–1303, 2202–2210, 3882–3892, 4500–4631)

- [ ] **Step 1: Update `ihipImageCreate` definition (lines 1135–1155)**

Replace lines 1135–1137:
```cpp
// Before:
amd::Image* ihipImageCreate(const cl_channel_order channelOrder, const cl_channel_type channelType,
                            const cl_mem_object_type imageType, const size_t imageWidth,

// After:
amd::Image* ihipImageCreate(const amd::ImageFormat fmt, const amd::MemObjectType imageType,
                            const size_t imageWidth,
```

Inside the body, replace lines 1143–1148:
```cpp
// Before:
  const amd::Image::Format imageFormat(amd::ImageFormat{
      static_cast<amd::ChannelOrder>(channelOrder),
      static_cast<amd::ChannelDataType>(channelType)});
  if (!imageFormat.isValid()) {
    LogPrintfError("Invalid Image format for channel Order:%u Type:%u", channelOrder, channelType);

// After:
  const amd::Image::Format imageFormat(fmt);
  if (!imageFormat.isValid()) {
    LogPrintfError("Invalid Image format for channel Order:%u Type:%u",
                   static_cast<uint32_t>(fmt.channelOrder),
                   static_cast<uint32_t>(fmt.channelDataType));
```

Also update line 1152 (the `isSupported` call):
```cpp
// Before:
  if (!imageFormat.isSupported(context, static_cast<amd::MemObjectType>(imageType))) {

// After:
  if (!imageFormat.isSupported(context, imageType)) {
```

- [ ] **Step 2: Replace first `cl_mem` pattern (lines 991–1001)**

Replace:
```cpp
// Before:
  cl_mem memObj = reinterpret_cast<cl_mem>(array->data);
  if (is_valid(memObj) == false) {
    return hipErrorInvalidValue;
  }

  auto image = as_amd(memObj);

// After:
  amd::Memory* memObj = reinterpret_cast<amd::Memory*>(array->data);
  if (memObj == nullptr) {
    return hipErrorInvalidValue;
  }

  auto image = memObj;
```

- [ ] **Step 3: Update `hipAllocateArray` block (lines 1284–1303)**

Replace:
```cpp
// Before:
  const cl_channel_order channelOrder = hip::getCLChannelOrder(pAllocateArray->NumChannels, 0);
  const cl_channel_type channelType =
      hip::getCLChannelType(pAllocateArray->Format, hipReadModeElementType);
  const cl_mem_object_type imageType = hip::getCLMemObjectType(
      pAllocateArray->Width, pAllocateArray->Height, pAllocateArray->Depth, pAllocateArray->Flags);
  hipError_t status = hipSuccess;
  amd::Image* image = ihipImageCreate(channelOrder, channelType, imageType, pAllocateArray->Width,

// After:
  const amd::ChannelOrder channelOrder = hip::getAMDChannelOrder(pAllocateArray->NumChannels, 0);
  const amd::ChannelDataType channelType =
      hip::getAMDChannelDataType(pAllocateArray->Format, hipReadModeElementType);
  const amd::MemObjectType imageType = hip::getAMDMemObjectType(
      pAllocateArray->Width, pAllocateArray->Height, pAllocateArray->Depth, pAllocateArray->Flags);
  hipError_t status = hipSuccess;
  amd::Image* image = ihipImageCreate(amd::ImageFormat{channelOrder, channelType}, imageType,
                                      pAllocateArray->Width,
```

Also replace lines 1300–1302:
```cpp
// Before:
  cl_mem memObj = as_cl<amd::Memory>(image);
  *array = new hipArray{reinterpret_cast<void*>(memObj)};

// After:
  *array = new hipArray{static_cast<void*>(image)};
```

- [ ] **Step 4: Replace second `cl_mem` pattern (lines 2202–2210)**

Replace:
```cpp
// Before:
  cl_mem memObj = reinterpret_cast<cl_mem>(array->data);
  if (!is_valid(memObj)) {
    return hipErrorInvalidValue;
  }

  image = as_amd(memObj)->asImage();

// After:
  amd::Memory* memObj = reinterpret_cast<amd::Memory*>(array->data);
  if (memObj == nullptr) {
    return hipErrorInvalidValue;
  }

  image = memObj->asImage();
```

- [ ] **Step 5: Replace third `cl_mem` pattern (lines 3882–3892)**

Replace:
```cpp
// Before:
        cl_mem dstMemObj = reinterpret_cast<cl_mem>((static_cast<hipArray*>(ptr))->data);
        if (!is_valid(dstMemObj)) {
          *reinterpret_cast<uint32_t*>(data) = 0;
          return hipErrorInvalidValue;
        }
        amd::Image* dstImage = as_amd(dstMemObj)->asImage();

// After:
        amd::Memory* dstMemObj = reinterpret_cast<amd::Memory*>((static_cast<hipArray*>(ptr))->data);
        if (dstMemObj == nullptr) {
          *reinterpret_cast<uint32_t*>(data) = 0;
          return hipErrorInvalidValue;
        }
        amd::Image* dstImage = dstMemObj->asImage();
```

- [ ] **Step 6: Update mipmapped array allocation block (lines 4500–4521)**

Replace:
```cpp
// Before:
  const cl_channel_order channel_order =
      hip::getCLChannelOrder(mipmapped_array_desc_ptr->NumChannels, 0);
  const cl_channel_type channel_type =
      hip::getCLChannelType(mipmapped_array_desc_ptr->Format, hipReadModeElementType);
  const cl_mem_object_type image_type =
      hip::getCLMemObjectType(mipmapped_array_desc_ptr->Width, mipmapped_array_desc_ptr->Height,
                              mipmapped_array_desc_ptr->Depth, mipmapped_array_desc_ptr->Flags);

// After:
  const amd::ChannelOrder channel_order =
      hip::getAMDChannelOrder(mipmapped_array_desc_ptr->NumChannels, 0);
  const amd::ChannelDataType channel_type =
      hip::getAMDChannelDataType(mipmapped_array_desc_ptr->Format, hipReadModeElementType);
  const amd::MemObjectType image_type =
      hip::getAMDMemObjectType(mipmapped_array_desc_ptr->Width, mipmapped_array_desc_ptr->Height,
                               mipmapped_array_desc_ptr->Depth, mipmapped_array_desc_ptr->Flags);
```

Update the `ihipImageCreate` call nearby:
```cpp
// Before:
      ihipImageCreate(channel_order, channel_type, image_type, mipmapped_array_desc_ptr->Width,

// After:
      ihipImageCreate(amd::ImageFormat{channel_order, channel_type}, image_type,
                      mipmapped_array_desc_ptr->Width,
```

Replace lines 4519–4521:
```cpp
// Before:
  cl_mem cl_mem_obj = as_cl<amd::Memory>(image);
  *mipmapped_array_pptr = new hipMipmappedArray();
  (*mipmapped_array_pptr)->data = reinterpret_cast<void*>(cl_mem_obj);

// After:
  *mipmapped_array_pptr = new hipMipmappedArray();
  (*mipmapped_array_pptr)->data = static_cast<void*>(image);
```

Also find the assignment `(*mipmapped_array_pptr)->type = image_type` (a few lines below, around line 4525). Since `hipMipmappedArray::type` is `unsigned int` and `image_type` is now `amd::MemObjectType`, add a cast:
```cpp
// Before:
  (*mipmapped_array_pptr)->type = image_type;

// After:
  (*mipmapped_array_pptr)->type = static_cast<unsigned int>(image_type);
```

- [ ] **Step 7: Replace fourth `cl_mem` pattern (lines 4571–4576)**

Replace:
```cpp
// Before:
  cl_mem cl_mem_obj = reinterpret_cast<cl_mem>(mipmapped_array_ptr->data);
  if (is_valid(cl_mem_obj) == false) {
    return hipErrorInvalidValue;
  }

  amd::Image* image = as_amd(cl_mem_obj)->asImage();

// After:
  amd::Memory* cl_mem_obj = reinterpret_cast<amd::Memory*>(mipmapped_array_ptr->data);
  if (cl_mem_obj == nullptr) {
    return hipErrorInvalidValue;
  }

  amd::Image* image = cl_mem_obj->asImage();
```

- [ ] **Step 8: Update `getAMDMemObjectType` call at line 4598**

Replace:
```cpp
// Before:
  const cl_mem_object_type image_type =
      hip::getCLMemObjectType((*level_array_pptr)->width, (*level_array_pptr)->height,
                              (*level_array_pptr)->depth, mipmapped_array_ptr->flags);
  (*level_array_pptr)->type = image_type;

// After:
  const amd::MemObjectType image_type =
      hip::getAMDMemObjectType((*level_array_pptr)->width, (*level_array_pptr)->height,
                               (*level_array_pptr)->depth, mipmapped_array_ptr->flags);
  (*level_array_pptr)->type = static_cast<uint32_t>(image_type);
```

- [ ] **Step 9: Replace fifth `cl_mem` pattern (lines 4625–4630)**

Replace:
```cpp
// Before:
  cl_mem cl_mem_obj = reinterpret_cast<cl_mem>(mipmap->data);
  if (is_valid(cl_mem_obj) == false) {
    return hipErrorInvalidValue;
  }

  amd::Image* image = as_amd(cl_mem_obj)->asImage();

// After:
  amd::Memory* cl_mem_obj = reinterpret_cast<amd::Memory*>(mipmap->data);
  if (cl_mem_obj == nullptr) {
    return hipErrorInvalidValue;
  }

  amd::Image* image = cl_mem_obj->asImage();
```

- [ ] **Step 10: Drop `cl_common.hpp` include**

Remove `#include "cl_common.hpp"` from `hip_memory.cpp`.

- [ ] **Step 11: Build and verify**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement/projects/clr/build
make -j$(nproc) amdhip64 2>&1 | tail -30
```
Expected: zero errors.

---

## Task 4: Update hip_device_runtime.cpp

**Files:**
- Modify: `projects/clr/hipamd/src/hip_device_runtime.cpp` (lines 26, 30, 32–33, 481)

- [ ] **Step 1: Replace `cl_uint`/`cl_int` variables**

In `ihipChooseDevice` (around lines 26–33), replace:
```cpp
// Before:
  cl_uint maxMatchedCount = 0;
  ...
  for (cl_int i = 0; i < count; ++i) {
    ...
    cl_uint validPropCount = 0;
    cl_uint matchedCount = 0;

// After:
  uint32_t maxMatchedCount = 0;
  ...
  for (int32_t i = 0; i < count; ++i) {
    ...
    uint32_t validPropCount = 0;
    uint32_t matchedCount = 0;
```

At line 481, replace:
```cpp
// Before:
    for (cl_int i = 0; i < count; i++) {

// After:
    for (int32_t i = 0; i < count; i++) {
```

- [ ] **Step 2: Check for `cl_common.hpp` include and remove if present**

```bash
grep -n "cl_common\|cl_int\|cl_uint" \
  /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement/projects/clr/hipamd/src/hip_device_runtime.cpp | head -20
```

If `cl_common.hpp` is included, remove it. If `cl_int`/`cl_uint` are pulled in from another header (e.g., `stdint.h` replacements for `int32_t`/`uint32_t`), confirm `<cstdint>` is included.

- [ ] **Step 3: Build and verify**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement/projects/clr/build
make -j$(nproc) amdhip64 2>&1 | tail -30
```
Expected: zero errors.

- [ ] **Step 4: Commit Tasks 2–4 together**

```bash
cd /home/cpaquot/src/rocm-systems/projects/clr/.worktrees/rocclr-phase1-type-replacement
git add projects/clr/hipamd/src/hip_texture.cpp \
        projects/clr/hipamd/src/hip_memory.cpp \
        projects/clr/hipamd/src/hip_device_runtime.cpp
git commit -m "hipamd: remove cl_common.hpp from memory, texture, and device runtime

Update all callers to use getAMD* functions from hip_conversions.hpp:
- cl_channel_order/type → amd::ChannelOrder/ChannelDataType
- cl_mem_object_type → amd::MemObjectType
- cl_addressing_mode → uint32_t (cast from amd::AddressingMode)
- cl_filter_mode → uint32_t (cast from amd::FilterMode)
- cl_int/cl_uint loop vars → int32_t/uint32_t (hip_device_runtime)

Change ihipImageCreate signature to (amd::ImageFormat, amd::MemObjectType, ...)
eliminating the internal static_cast from cl_* to amd:: types.

Replace cl_mem casts (reinterpret_cast<cl_mem> + as_amd/as_cl) with direct
amd::Memory* reinterpret_cast; replace is_valid() with nullptr checks.

Drop #include \"cl_common.hpp\" from all three files."
```
