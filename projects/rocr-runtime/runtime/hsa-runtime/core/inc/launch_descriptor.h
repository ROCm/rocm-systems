// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#ifndef HSA_RUNTIME_CORE_INC_LAUNCH_DESCRIPTOR_H_
#define HSA_RUNTIME_CORE_INC_LAUNCH_DESCRIPTOR_H_

#include <cstring>

#include "core/inc/agent.h"
#include "core/inc/exceptions.h"
#include "core/inc/isa.h"
#include "inc/amd_launch_descriptor.h"
#include "inc/hsa_ven_amd_launch_descriptor.h"

namespace rocr {
namespace core {

/// @brief Map an agent's ISA to a launch descriptor version.
inline uint8_t GetDescriptorVersion(const Agent* agent) {
  if (agent->supported_isas().empty())
    return AMD_LAUNCH_DESCRIPTOR_VERSION_NONE;

  const Isa* isa = agent->supported_isas()[0];

  if (isa->GetMajorVersion() == 12 && isa->GetMinorVersion() == 5)
    return AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250;

  return AMD_LAUNCH_DESCRIPTOR_VERSION_NONE;
}

/// @brief Return the hardware data size in bytes for a descriptor version.
inline uint16_t GetDescriptorDataSize(uint8_t version) {
  switch (version) {
    case AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250: return sizeof(amd_launch_descriptor_data_t);
    default: return 0;
  }
}

/// @brief Validate a launch descriptor handle via the version byte.
/// Returns a pointer to the raw descriptor data, or throws on invalid handle.
inline void* ConvertDescriptorHandle(amd_launch_descriptor_t handle) {
  if (handle.handle == 0)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                             "Null launch descriptor handle.");
  uint8_t* data = reinterpret_cast<uint8_t*>(handle.handle);
  uint8_t version = data[0];
  if (version == AMD_LAUNCH_DESCRIPTOR_VERSION_NONE || GetDescriptorDataSize(version) == 0)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                             "Invalid launch descriptor handle.");
  return data;
}

/// @brief Convert a raw data pointer to a public handle.
inline amd_launch_descriptor_t ToDescriptorHandle(void* data) {
  amd_launch_descriptor_t handle;
  handle.handle = reinterpret_cast<uint64_t>(data);
  return handle;
}

/// @brief Allocate and initialize a launch descriptor for the given version.
inline void* CreateDescriptor(uint8_t version) {
  uint16_t size = GetDescriptorDataSize(version);
  if (size == 0) return nullptr;
  void* data = ::operator new(size, std::nothrow);
  if (data == nullptr) return nullptr;
  std::memset(data, 0, size);
  static_cast<uint8_t*>(data)[0] = version;
  return data;
}

/// @brief Invalidate and free a launch descriptor.
inline void DestroyDescriptor(void* data) {
  static_cast<uint8_t*>(data)[0] = AMD_LAUNCH_DESCRIPTOR_VERSION_NONE;
  ::operator delete(data);
}

/// @brief Return the max prefetch regions for a descriptor version.
inline uint32_t GetMaxPrefetchRegions(uint8_t version) {
  switch (version) {
    case AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250: return 2;
    default: return 0;
  }
}

/// @brief Get a typed pointer to the descriptor data, given a raw data pointer.
inline amd_launch_descriptor_data_t* GetDescriptorData(void* data) {
  return static_cast<amd_launch_descriptor_data_t*>(data);
}

/// @brief Set a V1 (gfx1250) field.
inline void SetDescriptorFieldV1(amd_launch_descriptor_data_t* d, uint32_t field, uint64_t value) {
  switch (field) {
    case HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_PRIORITY:
      d->priority = static_cast<uint8_t>(value); break;
    case HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_PM_HINT:
      d->pm_hint = static_cast<uint8_t>(value); break;
    case HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_CU_START:
      if (value >= 16) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "CU start must be < 16");
      d->cu_start = static_cast<uint32_t>(value);
      break;
    case HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_CU_COUNT:
      if (value >= 16) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "CU count must be < 16");
      d->cu_count = static_cast<uint32_t>(value);
      break;
    case HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_SE_EN:
      if (value >= 4) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "SE enable must be < 4");
      d->se_en = static_cast<uint32_t>(value);
      break;
    case HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_DISPATCH_GRANULARITY_LIMITER:
      d->dispatch_granularity_limiter = static_cast<uint16_t>(value); break;
    case HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_TG_CHUNK_SIZE:
      d->tg_chunk_size = static_cast<uint16_t>(value); break;
    default:
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                               "Field not valid for this descriptor version");
  }
}

/// @brief Set a field in the descriptor, dispatching by version.
inline void SetDescriptorField(void* data, uint32_t field, uint64_t value) {
  if (field == HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_VERSION)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                             "Version is read-only; set by create()");

  uint8_t version = static_cast<uint8_t*>(data)[0];
  switch (version) {
    case AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250:
      SetDescriptorFieldV1(GetDescriptorData(data), field, value);
      break;
    default:
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                               "Unknown descriptor version");
  }
}

/// @brief Set a prefetch region, validated against the descriptor's version.
inline void SetDescriptorPrefetch(void* data, uint32_t index,
                                  const amd_data_prefetch_t* prefetch) {
  uint8_t version = static_cast<uint8_t*>(data)[0];
  auto* d = GetDescriptorData(data);
  if (index >= GetMaxPrefetchRegions(version)) {
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                             "Prefetch index out of range for this version");
  }
  if (prefetch) {
    d->prefetch[index] = *prefetch;
  } else {
    std::memset(&d->prefetch[index], 0, sizeof(amd_data_prefetch_t));
  }
}

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_LAUNCH_DESCRIPTOR_H_
