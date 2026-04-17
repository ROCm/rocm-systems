// hipamd/src/hip_type_map.hpp
// Bidirectional translation between amd::* and hip* types.
// HIP error codes (hipError_t) differ from CL error codes; translation is
// value-mapped, not cast-based.
//
// Note: Most type translations (MemFlags, MemObjectType, ChannelOrder,
// ChannelDataType, ImageFormat) are context-dependent and are done at
// individual HIP API entry points rather than here.
#pragma once
#include "amd_types.hpp"
#include "hip/hip_runtime_api.h"

namespace amd::hip {

// ── Status / error code translation ─────────────────────────────────────────
// hipError_t values differ from CL error codes; requires a switch.

// Note: amd::Status::MemObjectAllocationFailure maps to hipErrorOutOfMemory.
inline hipError_t to_hip(amd::Status s) {
  switch (s) {
    case amd::Status::Success:                     return hipSuccess;
    case amd::Status::MemObjectAllocationFailure:  return hipErrorOutOfMemory;
    case amd::Status::InvalidValue:                return hipErrorInvalidValue;
    case amd::Status::InvalidDevice:               return hipErrorInvalidDevice;
    case amd::Status::InvalidContext:              return hipErrorInvalidContext;
    case amd::Status::InvalidMemObject:            return hipErrorInvalidHandle;
    case amd::Status::OutOfResources:              return hipErrorLaunchOutOfResources;
    case amd::Status::BuildProgramFailure:         return hipErrorNoBinaryForGpu;
    default:                                       return hipErrorUnknown;
  }
}

inline amd::Status from_hip(hipError_t e) {
  switch (e) {
    case hipSuccess:                  return amd::Status::Success;
    case hipErrorOutOfMemory:         return amd::Status::MemObjectAllocationFailure;
    case hipErrorInvalidValue:        return amd::Status::InvalidValue;
    case hipErrorInvalidDevice:       return amd::Status::InvalidDevice;
    case hipErrorInvalidContext:      return amd::Status::InvalidContext;
    case hipErrorInvalidHandle:       return amd::Status::InvalidMemObject;
    default:                          return amd::Status::InvalidValue;
  }
}

// ── Not translated here ──────────────────────────────────────────────────────
// DeviceType:    HIP has no cl_device_type equivalent bitfield.
// MemFlags:      HIP uses hipMemAllocFlags / hipMemoryType — translated per API call.
// MemObjectType: HIP uses hipArray / hipMemoryType — translated per API call.
// ChannelOrder / ChannelDataType: HIP uses hipChannelFormatDesc — translated per API call.

}  // namespace amd::hip
