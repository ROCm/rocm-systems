// opencl/amdocl/cl_type_map.hpp
// Bidirectional translation between amd::* and cl_* types.
// static_asserts here are the compile-time tests for Phase 1 value correctness.
#pragma once
#include "amd_types.hpp"
#include "CL/opencl.h"
#include <type_traits>

namespace amd::cl {

// ── Value verification (compile-time tests) ──────────────────────────────────

static_assert(static_cast<uint64_t>(amd::DeviceType::Default)     == CL_DEVICE_TYPE_DEFAULT);
static_assert(static_cast<uint64_t>(amd::DeviceType::CPU)         == CL_DEVICE_TYPE_CPU);
static_assert(static_cast<uint64_t>(amd::DeviceType::GPU)         == CL_DEVICE_TYPE_GPU);
static_assert(static_cast<uint64_t>(amd::DeviceType::Accelerator) == CL_DEVICE_TYPE_ACCELERATOR);
static_assert(static_cast<uint64_t>(amd::DeviceType::Custom)      == CL_DEVICE_TYPE_CUSTOM);
static_assert(static_cast<uint64_t>(amd::DeviceType::All)         == CL_DEVICE_TYPE_ALL);

static_assert(static_cast<uint64_t>(amd::FpConfig::Denorm)          == CL_FP_DENORM);
static_assert(static_cast<uint64_t>(amd::FpConfig::InfNan)          == CL_FP_INF_NAN);
static_assert(static_cast<uint64_t>(amd::FpConfig::RoundToNearest)  == CL_FP_ROUND_TO_NEAREST);
static_assert(static_cast<uint64_t>(amd::FpConfig::RoundToZero)     == CL_FP_ROUND_TO_ZERO);
static_assert(static_cast<uint64_t>(amd::FpConfig::RoundToInf)      == CL_FP_ROUND_TO_INF);
static_assert(static_cast<uint64_t>(amd::FpConfig::Fma)             == CL_FP_FMA);
static_assert(static_cast<uint64_t>(amd::FpConfig::SoftFloat)       == CL_FP_SOFT_FLOAT);
static_assert(static_cast<uint64_t>(amd::FpConfig::CorrectlyRounded) == CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT);

static_assert(static_cast<uint32_t>(amd::MemCacheType::None)           == CL_NONE);
static_assert(static_cast<uint32_t>(amd::MemCacheType::ReadOnlyCache)  == CL_READ_ONLY_CACHE);
static_assert(static_cast<uint32_t>(amd::MemCacheType::ReadWriteCache) == CL_READ_WRITE_CACHE);

static_assert(static_cast<uint32_t>(amd::LocalMemType::None)   == CL_NONE);
static_assert(static_cast<uint32_t>(amd::LocalMemType::Local)  == CL_LOCAL);
static_assert(static_cast<uint32_t>(amd::LocalMemType::Global) == CL_GLOBAL);

static_assert(static_cast<uint64_t>(amd::ExecCapabilities::Kernel)       == CL_EXEC_KERNEL);
static_assert(static_cast<uint64_t>(amd::ExecCapabilities::NativeKernel) == CL_EXEC_NATIVE_KERNEL);

static_assert(static_cast<uint64_t>(amd::SvmCapabilities::CoarseGrainBuffer) == CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
static_assert(static_cast<uint64_t>(amd::SvmCapabilities::FineGrainBuffer)   == CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
static_assert(static_cast<uint64_t>(amd::SvmCapabilities::FineGrainSystem)   == CL_DEVICE_SVM_FINE_GRAIN_SYSTEM);
static_assert(static_cast<uint64_t>(amd::SvmCapabilities::Atomics)           == CL_DEVICE_SVM_ATOMICS);

static_assert(static_cast<uint64_t>(amd::QueueProperties::OutOfOrderExec)  == CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
static_assert(static_cast<uint64_t>(amd::QueueProperties::Profiling)       == CL_QUEUE_PROFILING_ENABLE);
static_assert(static_cast<uint64_t>(amd::QueueProperties::OnDevice)        == CL_QUEUE_ON_DEVICE);
static_assert(static_cast<uint64_t>(amd::QueueProperties::OnDeviceDefault) == CL_QUEUE_ON_DEVICE_DEFAULT);

static_assert(static_cast<uint64_t>(amd::MemFlags::ReadWrite)       == CL_MEM_READ_WRITE);
static_assert(static_cast<uint64_t>(amd::MemFlags::WriteOnly)       == CL_MEM_WRITE_ONLY);
static_assert(static_cast<uint64_t>(amd::MemFlags::ReadOnly)        == CL_MEM_READ_ONLY);
static_assert(static_cast<uint64_t>(amd::MemFlags::UseHostPtr)      == CL_MEM_USE_HOST_PTR);
static_assert(static_cast<uint64_t>(amd::MemFlags::AllocHostPtr)    == CL_MEM_ALLOC_HOST_PTR);
static_assert(static_cast<uint64_t>(amd::MemFlags::CopyHostPtr)     == CL_MEM_COPY_HOST_PTR);
static_assert(static_cast<uint64_t>(amd::MemFlags::HostWriteOnly)   == CL_MEM_HOST_WRITE_ONLY);
static_assert(static_cast<uint64_t>(amd::MemFlags::HostReadOnly)    == CL_MEM_HOST_READ_ONLY);
static_assert(static_cast<uint64_t>(amd::MemFlags::HostNoAccess)    == CL_MEM_HOST_NO_ACCESS);
static_assert(static_cast<uint64_t>(amd::MemFlags::SvmFineGrain)   == CL_MEM_SVM_FINE_GRAIN_BUFFER);
static_assert(static_cast<uint64_t>(amd::MemFlags::SvmAtomics)     == CL_MEM_SVM_ATOMICS);
static_assert(static_cast<uint64_t>(amd::MemFlags::KernelReadWrite) == CL_MEM_KERNEL_READ_AND_WRITE);

static_assert(static_cast<uint32_t>(amd::MemObjectType::Buffer)        == CL_MEM_OBJECT_BUFFER);
static_assert(static_cast<uint32_t>(amd::MemObjectType::Image2D)       == CL_MEM_OBJECT_IMAGE2D);
static_assert(static_cast<uint32_t>(amd::MemObjectType::Image3D)       == CL_MEM_OBJECT_IMAGE3D);
static_assert(static_cast<uint32_t>(amd::MemObjectType::Image2DArray)  == CL_MEM_OBJECT_IMAGE2D_ARRAY);
static_assert(static_cast<uint32_t>(amd::MemObjectType::Image1D)       == CL_MEM_OBJECT_IMAGE1D);
static_assert(static_cast<uint32_t>(amd::MemObjectType::Image1DArray)  == CL_MEM_OBJECT_IMAGE1D_ARRAY);
static_assert(static_cast<uint32_t>(amd::MemObjectType::Image1DBuffer) == CL_MEM_OBJECT_IMAGE1D_BUFFER);

static_assert(static_cast<uint32_t>(amd::ChannelOrder::RGBA)  == CL_RGBA);
static_assert(static_cast<uint32_t>(amd::ChannelOrder::BGRA)  == CL_BGRA);
static_assert(static_cast<uint32_t>(amd::ChannelOrder::R)     == CL_R);
static_assert(static_cast<uint32_t>(amd::ChannelOrder::Depth) == CL_DEPTH);
static_assert(static_cast<uint32_t>(amd::ChannelOrder::sRGB)  == CL_sRGB);
static_assert(static_cast<uint32_t>(amd::ChannelOrder::ABGR)  == CL_ABGR);

static_assert(static_cast<uint32_t>(amd::ChannelDataType::UNormInt8)      == CL_UNORM_INT8);
static_assert(static_cast<uint32_t>(amd::ChannelDataType::UNormInt16)     == CL_UNORM_INT16);
static_assert(static_cast<uint32_t>(amd::ChannelDataType::Float)          == CL_FLOAT);
static_assert(static_cast<uint32_t>(amd::ChannelDataType::HalfFloat)      == CL_HALF_FLOAT);
static_assert(static_cast<uint32_t>(amd::ChannelDataType::UNormInt101010) == CL_UNORM_INT_101010);
static_assert(static_cast<uint32_t>(amd::ChannelDataType::SignedInt32)    == CL_SIGNED_INT32);
static_assert(static_cast<uint32_t>(amd::ChannelDataType::UnsignedInt8)   == CL_UNSIGNED_INT8);

static_assert(static_cast<int32_t>(amd::Status::Success)              == CL_SUCCESS);
static_assert(static_cast<int32_t>(amd::Status::DeviceNotFound)       == CL_DEVICE_NOT_FOUND);
static_assert(static_cast<int32_t>(amd::Status::OutOfResources)       == CL_OUT_OF_RESOURCES);
static_assert(static_cast<int32_t>(amd::Status::OutOfHostMemory)      == CL_OUT_OF_HOST_MEMORY);
static_assert(static_cast<int32_t>(amd::Status::InvalidValue)         == CL_INVALID_VALUE);
static_assert(static_cast<int32_t>(amd::Status::InvalidDevice)        == CL_INVALID_DEVICE);
static_assert(static_cast<int32_t>(amd::Status::InvalidContext)       == CL_INVALID_CONTEXT);
static_assert(static_cast<int32_t>(amd::Status::InvalidMemObject)     == CL_INVALID_MEM_OBJECT);
static_assert(static_cast<int32_t>(amd::Status::BuildProgramFailure)  == CL_BUILD_PROGRAM_FAILURE);

static_assert(static_cast<uint32_t>(amd::CommandType::NdRangeKernel)  == CL_COMMAND_NDRANGE_KERNEL);
static_assert(static_cast<uint32_t>(amd::CommandType::ReadBuffer)     == CL_COMMAND_READ_BUFFER);
static_assert(static_cast<uint32_t>(amd::CommandType::User)           == CL_COMMAND_USER);
static_assert(static_cast<uint32_t>(amd::CommandType::Barrier)        == CL_COMMAND_BARRIER);
static_assert(static_cast<uint32_t>(amd::CommandType::FillBuffer)     == CL_COMMAND_FILL_BUFFER);
static_assert(static_cast<uint32_t>(amd::CommandType::SvmFree)        == CL_COMMAND_SVM_FREE);
static_assert(static_cast<uint32_t>(amd::CommandType::SvmUnmap)       == CL_COMMAND_SVM_UNMAP);

static_assert(static_cast<int32_t>(amd::ExecutionStatus::Complete)  == CL_COMPLETE);
static_assert(static_cast<int32_t>(amd::ExecutionStatus::Running)   == CL_RUNNING);
static_assert(static_cast<int32_t>(amd::ExecutionStatus::Submitted) == CL_SUBMITTED);
static_assert(static_cast<int32_t>(amd::ExecutionStatus::Queued)    == CL_QUEUED);

static_assert(static_cast<uint64_t>(amd::MapFlags::Read)            == CL_MAP_READ);
static_assert(static_cast<uint64_t>(amd::MapFlags::Write)           == CL_MAP_WRITE);
static_assert(static_cast<uint64_t>(amd::MapFlags::WriteInvalidate) == CL_MAP_WRITE_INVALIDATE_REGION);

static_assert(static_cast<uint64_t>(amd::MemMigrationFlags::Host)             == CL_MIGRATE_MEM_OBJECT_HOST);
static_assert(static_cast<uint64_t>(amd::MemMigrationFlags::ContentUndefined) == CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);

// ── Translation functions ─────────────────────────────────────────────────────
//
// Note on from_cl() naming: the OpenCL C headers define cl_device_type,
// cl_device_fp_config, cl_device_exec_capabilities, cl_device_svm_capabilities,
// cl_command_queue_properties, cl_mem_flags, and cl_svm_mem_flags all as plain
// typedef aliases for cl_bitfield (uint64_t), and cl_device_mem_cache_type,
// cl_device_local_mem_type, cl_mem_object_type, and cl_command_type all as
// aliases for cl_uint.  Because these are not distinct types, the compiler
// cannot resolve overloads of from_cl() that differ only by parameter type.
// The from_cl functions therefore carry a unique suffix naming the amd:: type
// they produce (e.g. from_cl_DeviceType).  The to_cl() direction is fine
// because overloading on the amd:: enum class argument types is unambiguous.

inline cl_device_type to_cl(amd::DeviceType t) {
  return static_cast<cl_device_type>(t);
}
inline amd::DeviceType from_cl_DeviceType(cl_device_type t) {
  return static_cast<amd::DeviceType>(t);
}

inline cl_device_fp_config to_cl(amd::FpConfig c) {
  return static_cast<cl_device_fp_config>(c);
}
inline amd::FpConfig from_cl_FpConfig(cl_device_fp_config c) {
  return static_cast<amd::FpConfig>(c);
}

inline cl_device_mem_cache_type to_cl(amd::MemCacheType t) {
  return static_cast<cl_device_mem_cache_type>(t);
}
inline amd::MemCacheType from_cl_MemCacheType(cl_device_mem_cache_type t) {
  return static_cast<amd::MemCacheType>(t);
}

inline cl_device_local_mem_type to_cl(amd::LocalMemType t) {
  return static_cast<cl_device_local_mem_type>(t);
}
inline amd::LocalMemType from_cl_LocalMemType(cl_device_local_mem_type t) {
  return static_cast<amd::LocalMemType>(t);
}

inline cl_device_exec_capabilities to_cl(amd::ExecCapabilities c) {
  return static_cast<cl_device_exec_capabilities>(c);
}
inline amd::ExecCapabilities from_cl_ExecCapabilities(cl_device_exec_capabilities c) {
  return static_cast<amd::ExecCapabilities>(c);
}

inline cl_device_svm_capabilities to_cl(amd::SvmCapabilities c) {
  return static_cast<cl_device_svm_capabilities>(c);
}
inline amd::SvmCapabilities from_cl_SvmCapabilities(cl_device_svm_capabilities c) {
  return static_cast<amd::SvmCapabilities>(c);
}

inline cl_command_queue_properties to_cl(amd::QueueProperties p) {
  return static_cast<cl_command_queue_properties>(p);
}
inline amd::QueueProperties from_cl_QueueProperties(cl_command_queue_properties p) {
  return static_cast<amd::QueueProperties>(p);
}

inline cl_mem_flags to_cl(amd::MemFlags f) {
  return static_cast<cl_mem_flags>(f);
}
inline amd::MemFlags from_cl_MemFlags(cl_mem_flags f) {
  return static_cast<amd::MemFlags>(f);
}
inline cl_svm_mem_flags to_cl_svm(amd::MemFlags f) {
  return static_cast<cl_svm_mem_flags>(f);
}
inline amd::MemFlags from_cl_svm_MemFlags(cl_svm_mem_flags f) {
  return static_cast<amd::MemFlags>(f);
}

inline cl_mem_object_type to_cl(amd::MemObjectType t) {
  return static_cast<cl_mem_object_type>(t);
}
inline amd::MemObjectType from_cl_MemObjectType(cl_mem_object_type t) {
  return static_cast<amd::MemObjectType>(t);
}

inline cl_image_format to_cl(const amd::ImageFormat& f) {
  return {static_cast<cl_channel_order>(f.channelOrder),
          static_cast<cl_channel_type>(f.channelDataType)};
}
inline amd::ImageFormat from_cl_ImageFormat(const cl_image_format& f) {
  return {static_cast<amd::ChannelOrder>(f.image_channel_order),
          static_cast<amd::ChannelDataType>(f.image_channel_data_type)};
}

inline cl_int to_cl(amd::Status s) {
  return static_cast<cl_int>(s);
}
inline amd::Status from_cl_Status(cl_int s) {
  return static_cast<amd::Status>(s);
}

inline cl_command_type to_cl(amd::CommandType t) {
  return static_cast<cl_command_type>(t);
}
inline amd::CommandType from_cl_CommandType(cl_command_type t) {
  return static_cast<amd::CommandType>(t);
}

inline cl_int to_cl(amd::ExecutionStatus s) {
  return static_cast<cl_int>(s);
}
inline amd::ExecutionStatus from_cl_ExecutionStatus(cl_int s) {
  return static_cast<amd::ExecutionStatus>(s);
}

inline cl_map_flags to_cl(amd::MapFlags f) {
  return static_cast<cl_map_flags>(f);
}
inline amd::MapFlags from_cl_MapFlags(cl_map_flags f) {
  return static_cast<amd::MapFlags>(f);
}

inline cl_mem_migration_flags to_cl(amd::MemMigrationFlags f) {
  return static_cast<cl_mem_migration_flags>(f);
}
inline amd::MemMigrationFlags from_cl_MemMigrationFlags(cl_mem_migration_flags f) {
  return static_cast<amd::MemMigrationFlags>(f);
}

// Verify KernelArgAddressQualifier values match CL_KERNEL_ARG_ADDRESS_* constants.
static_assert(static_cast<uint32_t>(amd::KernelArgAddressQualifier::Global)   == CL_KERNEL_ARG_ADDRESS_GLOBAL);
static_assert(static_cast<uint32_t>(amd::KernelArgAddressQualifier::Local)    == CL_KERNEL_ARG_ADDRESS_LOCAL);
static_assert(static_cast<uint32_t>(amd::KernelArgAddressQualifier::Constant) == CL_KERNEL_ARG_ADDRESS_CONSTANT);
static_assert(static_cast<uint32_t>(amd::KernelArgAddressQualifier::Private)  == CL_KERNEL_ARG_ADDRESS_PRIVATE);

// Verify KernelArgAccessQualifier values match CL_KERNEL_ARG_ACCESS_* constants.
static_assert(static_cast<uint32_t>(amd::KernelArgAccessQualifier::ReadOnly)  == CL_KERNEL_ARG_ACCESS_READ_ONLY);
static_assert(static_cast<uint32_t>(amd::KernelArgAccessQualifier::WriteOnly) == CL_KERNEL_ARG_ACCESS_WRITE_ONLY);
static_assert(static_cast<uint32_t>(amd::KernelArgAccessQualifier::ReadWrite) == CL_KERNEL_ARG_ACCESS_READ_WRITE);
static_assert(static_cast<uint32_t>(amd::KernelArgAccessQualifier::None)      == CL_KERNEL_ARG_ACCESS_NONE);

// Verify KernelArgTypeQualifier values match CL_KERNEL_ARG_TYPE_* constants.
static_assert(static_cast<uint64_t>(amd::KernelArgTypeQualifier::None)     == CL_KERNEL_ARG_TYPE_NONE);
static_assert(static_cast<uint64_t>(amd::KernelArgTypeQualifier::Const)    == CL_KERNEL_ARG_TYPE_CONST);
static_assert(static_cast<uint64_t>(amd::KernelArgTypeQualifier::Restrict) == CL_KERNEL_ARG_TYPE_RESTRICT);
static_assert(static_cast<uint64_t>(amd::KernelArgTypeQualifier::Volatile) == CL_KERNEL_ARG_TYPE_VOLATILE);
static_assert(static_cast<uint64_t>(amd::KernelArgTypeQualifier::Pipe)     == CL_KERNEL_ARG_TYPE_PIPE);

// Verify KernelArgValueType values match clk_value_type_t from amdocl/cl_kernel.h.
// clk_value_type_t is a plain C enum starting at T_VOID=0.
#include "amdocl/cl_kernel.h"
static_assert(static_cast<int32_t>(amd::KernelArgValueType::Void)    == T_VOID);
static_assert(static_cast<int32_t>(amd::KernelArgValueType::Pointer) == T_POINTER);
static_assert(static_cast<int32_t>(amd::KernelArgValueType::Struct)  == T_STRUCT);
static_assert(static_cast<int32_t>(amd::KernelArgValueType::Queue)   == T_QUEUE);
static_assert(static_cast<int32_t>(amd::KernelArgValueType::Pad)     == T_PAD);

// Verify DeviceClockMode values match cl_DeviceClockMode_AMD from amdocl/cl_profile_amd.h.
#include "amdocl/cl_profile_amd.h"
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::Default)       == CL_DEVICE_CLOCK_MODE_DEFAULT_AMD);
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::Query)         == CL_DEVICE_CLOCK_MODE_QUERY_AMD);
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::Profiling)     == CL_DEVICE_CLOCK_MODE_PROFILING_AMD);
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::MinimumMemory) == CL_DEVICE_CLOCK_MODE_MINIMUMMEMORY_AMD);
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::MinimumEngine) == CL_DEVICE_CLOCK_MODE_MINIMUMENGINE_AMD);
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::Peak)          == CL_DEVICE_CLOCK_MODE_PEAK_AMD);
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::QueryProfiling) == CL_DEVICE_CLOCK_MODE_QUERYPROFILING_AMD);
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::QueryPeak)     == CL_DEVICE_CLOCK_MODE_QUERYPEAK_AMD);
static_assert(static_cast<uint32_t>(amd::DeviceClockMode::Count)         == CL_DEVICE_CLOCK_MODE_COUNT_AMD);

// Verify SetDeviceClockModeOutput struct layout matches cl_set_device_clock_mode_output_amd.
static_assert(sizeof(amd::SetDeviceClockModeOutput) == sizeof(cl_set_device_clock_mode_output_amd));

// Verify FilterMode values match CL_FILTER_* constants.
static_assert(static_cast<uint32_t>(amd::FilterMode::Nearest) == CL_FILTER_NEAREST);
static_assert(static_cast<uint32_t>(amd::FilterMode::Linear)  == CL_FILTER_LINEAR);

// Verify AddressingMode values match CL_ADDRESS_* constants.
static_assert(static_cast<uint32_t>(amd::AddressingMode::None)           == CL_ADDRESS_NONE);
static_assert(static_cast<uint32_t>(amd::AddressingMode::ClampToEdge)    == CL_ADDRESS_CLAMP_TO_EDGE);
static_assert(static_cast<uint32_t>(amd::AddressingMode::Clamp)          == CL_ADDRESS_CLAMP);
static_assert(static_cast<uint32_t>(amd::AddressingMode::Repeat)         == CL_ADDRESS_REPEAT);
static_assert(static_cast<uint32_t>(amd::AddressingMode::MirroredRepeat) == CL_ADDRESS_MIRRORED_REPEAT);

}  // namespace amd::cl
