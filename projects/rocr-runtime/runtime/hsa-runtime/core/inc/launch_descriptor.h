// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#ifndef HSA_RUNTIME_CORE_INC_LAUNCH_DESCRIPTOR_H_
#define HSA_RUNTIME_CORE_INC_LAUNCH_DESCRIPTOR_H_

#include <cstring>

#include "core/inc/agent.h"
#include "core/inc/checked.h"
#include "core/inc/exceptions.h"
#include "core/inc/isa.h"
#include "inc/amd_launch_descriptor.h"

namespace rocr {
namespace core {

/// @brief Map an agent's ISA to a launch descriptor version.
inline uint8_t GetDescriptorVersion(const Agent* agent) {
  if (agent->supported_isas().empty())
    return AMD_LAUNCH_DESCRIPTOR_VERSION_NONE;

  const Isa* isa = agent->supported_isas()[0];
  int32_t major = isa->GetMajorVersion();
  int32_t minor = isa->GetMinorVersion();

  if (major == 12 && minor == 5)
    return AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250;

  return AMD_LAUNCH_DESCRIPTOR_VERSION_NONE;
}

/// @brief Return the hardware data size in bytes for a descriptor version.
inline uint16_t GetDescriptorDataSize(uint8_t version) {
  switch (version) {
    case AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250: return 52;
    default: return 0;
  }
}

/// @brief Runtime wrapper for launch descriptor hardware data.
///
/// Follows the hsa_signal_t / SharedSignal convention:
///   - amd_launch_descriptor_data_t is the FIRST member so that
///     &wrapper == &wrapper.data. The handle stores this address,
///     allowing FW to read the 52-byte structure directly.
///   - Check<> magic number for validity checking on Convert().
///   - Static Convert() methods for handle <-> object conversion.
struct LaunchDescriptor {
  // --- FIRST MEMBER: hardware data (FW reads from handle address) ---
  amd_launch_descriptor_data_t data;

  // --- Runtime metadata (after data, invisible to FW) ---
  Check<0xA4D3E1F07B926C58> id;

  explicit LaunchDescriptor(uint8_t version) {
    std::memset(&data, 0, sizeof(data));
    data.version = version;
  }

  bool IsValid() const { return id.IsValid(); }

  /// @brief Convert public handle to internal object pointer.
  static __forceinline LaunchDescriptor* Convert(amd_launch_descriptor_t handle) {
    if (handle.handle == 0) return nullptr;
    LaunchDescriptor* desc = reinterpret_cast<LaunchDescriptor*>(handle.handle);
    if (!desc->IsValid())
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                               "Invalid launch descriptor handle.");
    return desc;
  }

  /// @brief Convert internal object pointer to public handle.
  static __forceinline amd_launch_descriptor_t Convert(LaunchDescriptor* desc) {
    amd_launch_descriptor_t handle;
    handle.handle = reinterpret_cast<uint64_t>(desc);
    return handle;
  }
};

/// @brief Return the max prefetch regions for a descriptor version.
inline uint32_t GetMaxPrefetchRegions(uint8_t version) {
  switch (version) {
    case AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250: return 2;
    default: return 0;
  }
}

/// @brief Set a V1 (gfx1250) field.
inline void SetDescriptorFieldV1(LaunchDescriptor* desc, uint32_t field, uint64_t value) {
  auto& d = desc->data;
  switch (field) {
    case 1: d.priority = static_cast<uint8_t>(value); break;
    case 2: d.pm_hint = static_cast<uint8_t>(value); break;
    case 3:
      if (value >= 16) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "CU start must be < 16");
      d.cu_start = static_cast<uint32_t>(value);
      break;
    case 4:
      if (value >= 16) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "CU count must be < 16");
      d.cu_count = static_cast<uint32_t>(value);
      break;
    case 5:
      if (value >= 4) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "SE enable must be < 4");
      d.se_en = static_cast<uint32_t>(value);
      break;
    case 6: d.dispatch_granularity_limiter = static_cast<uint16_t>(value); break;
    case 7: d.tg_chunk_size = static_cast<uint16_t>(value); break;
    default:
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                               "Field not valid for this descriptor version");
  }
}

/// @brief Set a field in the descriptor, dispatching by version.
inline void SetDescriptorField(LaunchDescriptor* desc, uint32_t field, uint64_t value) {
  if (field == 0)
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                             "Version is read-only; set by create()");

  switch (desc->data.version) {
    case AMD_LAUNCH_DESCRIPTOR_VERSION_GFX1250:
      SetDescriptorFieldV1(desc, field, value);
      break;
    default:
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                               "Unknown descriptor version");
  }
}

/// @brief Set a prefetch region, validated against the descriptor's version.
inline void SetDescriptorPrefetch(LaunchDescriptor* desc, uint32_t index,
                                  const amd_data_prefetch_t* prefetch) {
  if (index >= GetMaxPrefetchRegions(desc->data.version)) {
    throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                             "Prefetch index out of range for this version");
  }
  if (prefetch) {
    desc->data.prefetch[index] = *prefetch;
  } else {
    std::memset(&desc->data.prefetch[index], 0, sizeof(amd_data_prefetch_t));
  }
}

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_LAUNCH_DESCRIPTOR_H_
