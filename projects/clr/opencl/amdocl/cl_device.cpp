/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "cl_common.hpp"
#include "vdi_common.hpp"
#include "device/device.hpp"
#include "platform/runtime.hpp"
#include "rocclr/utils/versions.hpp"
#include "os/os.hpp"
#include "cl_semaphore_amd.h"

#include "CL/cl_ext.h"

#include <algorithm>  // for std::find
#include <cstdlib>    // for alloca
#include <cstring>
#include <vector>

namespace {

//! clGetInfo for a variable-length array. Unlike amd::clGetInfo this accepts an empty
//! list - several cl_khr_external_* queries legitimately report zero handle types - and
//! reports it as a zero param_value_size_ret rather than CL_INVALID_VALUE.
template <typename T>
cl_int clGetInfoVector(const std::vector<T>& values, size_t param_value_size, void* param_value,
                       size_t* param_value_size_ret) {
  const size_t valueSize = values.size() * sizeof(T);
  *not_null(param_value_size_ret) = valueSize;
  if (param_value != nullptr) {
    if (param_value_size < valueSize) {
      return CL_INVALID_VALUE;
    }
    if (valueSize != 0) {
      ::memcpy(param_value, values.data(), valueSize);
    }
  }
  return CL_SUCCESS;
}

}  // namespace

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_PlatformInfo
 *  @{
 */

/*! \brief Get the list of available platforms.
 *
 * \param num_entries is the number of cl_platform_id entries that can be added
 * to platforms. If \a platforms is not NULL, the \a num_entries must be greater
 * than zero.
 *
 * \param platforms returns a list of OpenCL platforms found. The cl_platform_id
 * values returned in \a platforms can be used to identify a specific OpenCL
 * platform. If \a platforms argument is NULL, this argument is ignored. The
 * number of OpenCL platforms returned is the mininum of the value specified by
 * \a num_entries or the number of OpenCL platforms available.
 *
 * \param num_platforms returns the number of OpenCL platforms available. If
 * \a num_platforms is NULL, this argument is ignored.
 *
 * \return CL_INVALID_VALUE if num_entries is equal to zero and platforms is not
 * NULL or if both num_platforms and platforms are NULL, and returns CL_SUCCESS
 * if the function is executed successfully.
 *
 * \version 1.0r33
 */

RUNTIME_ENTRY(cl_int, clGetPlatformIDs,
              (cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms)) {
  if (!amd::Runtime::initialized()) {
    amd::Runtime::init();
  }

  if (((num_entries > 0 || num_platforms == NULL) && platforms == NULL) ||
      (num_entries == 0 && platforms != NULL)) {
    return CL_INVALID_VALUE;
  }
  if (num_platforms != NULL && platforms == NULL) {
    *num_platforms = 1;
    return CL_SUCCESS;
  }

  assert(platforms != NULL && "check the code above");
  *platforms = AMD_PLATFORM;

  *not_null(num_platforms) = 1;
  return CL_SUCCESS;
}
RUNTIME_EXIT

//! Union of one of the per-device interop handle-type lists across the platform.
//
// These used to be fixed tables here, which was only correct while ROCr was the sole
// backend. Which handle types can actually be imported depends on the backend that
// initialized: ROCr routes through libhsakmt, which rejects KMT semaphore handles,
// while PAL opens them via OpenExternalSharedQueueSemaphore and accepts both. So the
// platform answer has to be assembled from the devices rather than decided at compile
// time. Order is preserved and duplicates dropped; the lists are a handful of entries,
// so the linear search costs nothing.
template <typename T, typename Selector>
static std::vector<T> unionOfDeviceHandleTypes(Selector select) {
  std::vector<T> types;
  for (const auto* device : amd::Device::getDevices(CL_DEVICE_TYPE_ALL, false)) {
    for (const T type : *select(device->info())) {
      if (std::find(types.begin(), types.end(), type) == types.end()) {
        types.push_back(type);
      }
    }
  }
  return types;
}

/*! \brief Get specific information about the OpenCL platform.
 *
 *  \param param_name is an enum that identifies the platform information being
 *  queried.
 *
 *  \param param_value is a pointer to memory location where appropriate values
 *  for a given \a param_name will be returned. If \a param_value is NULL,
 *  it is ignored.
 *
 *  \param param_value_size specifies the size in bytes of memory pointed to by
 *  \a param_value. This size in bytes must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data being
 *  queried by param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *    - CL_INVALID_VALUE if \a param_name is not one of the supported
 *      values or if size in bytes specified by \a param_value_size is < size of
 *      return type and \a param_value is not a NULL value.
 *    - CL_SUCCESS if the function is executed successfully.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetPlatformInfo,
              (cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  if (platform != NULL && platform != AMD_PLATFORM) {
    return CL_INVALID_PLATFORM;
  }

  const char* value = NULL;
  switch (param_name) {
    case CL_PLATFORM_PROFILE:
      value = "FULL_PROFILE";
      break;
    case CL_PLATFORM_VERSION:
      value = "OpenCL " XSTR(OPENCL_MAJOR) "." XSTR(OPENCL_MINOR) " " AMD_PLATFORM_INFO;
      break;
    case CL_PLATFORM_NAME:
      value = AMD_PLATFORM_NAME;
      break;
    case CL_PLATFORM_VENDOR:
      value = "Advanced Micro Devices, Inc.";
      break;
    case CL_PLATFORM_EXTENSIONS:
      value =
          "cl_khr_icd "
#ifdef _WIN32
          "cl_khr_d3d10_sharing "
          "cl_khr_d3d11_sharing "
          "cl_khr_dx9_media_sharing "
          "cl_khr_external_memory_win32 "
          "cl_khr_external_semaphore_win32 "
          // Kept under the same guard as the backends: rocsettings.cpp and palsettings.cpp only
          // enableExtension() these inside _WIN32, so advertising them at platform level
          // elsewhere would promise capabilities no device on the platform reports.
          "cl_khr_semaphore "
          "cl_khr_external_memory "
          "cl_khr_external_semaphore "
#endif  //_WIN32
          "cl_khr_device_uuid "
          "cl_amd_event_callback "
          ;
      break;
    case CL_PLATFORM_ICD_SUFFIX_KHR:
      value = "AMD";
      break;
    case CL_PLATFORM_HOST_TIMER_RESOLUTION: {
      cl_ulong resolution = (cl_ulong)amd::Os::timerResolutionNanos();
      return amd::clGetInfo(resolution, param_value_size, param_value, param_value_size_ret);
    }

    // cl_khr_semaphore / cl_khr_external_* platform-level capability lists. All devices
    // on this platform share one backend, so these mirror the per-device lists.
    case CL_PLATFORM_SEMAPHORE_TYPES_KHR: {
      // An array, like the handle-type lists below, and empty where cl_khr_semaphore is not
      // enabled - reporting BINARY there would contradict the extension string.
      std::vector<cl_semaphore_type_khr> types;
#ifdef _WIN32
      types.push_back(CL_SEMAPHORE_TYPE_BINARY_KHR);
#endif  //_WIN32
      return clGetInfoVector(types, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR: {
      const auto types = unionOfDeviceHandleTypes<cl_external_memory_handle_type_khr>(
          [](const device::Info& i) { return &i.externalMemoryImportHandleTypes_; });
      return clGetInfoVector(types, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR: {
      const auto types = unionOfDeviceHandleTypes<cl_external_semaphore_handle_type_khr>(
          [](const device::Info& i) { return &i.semaphoreImportHandleTypes_; });
      return clGetInfoVector(types, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR: {
      const auto types = unionOfDeviceHandleTypes<cl_external_semaphore_handle_type_khr>(
          [](const device::Info& i) { return &i.semaphoreExportHandleTypes_; });
      return clGetInfoVector(types, param_value_size, param_value, param_value_size_ret);
    }

    default:
      break;
  }
  if (value != NULL) {
    return amd::clGetInfo(value, param_value_size, param_value, param_value_size_ret);
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_Devices
 *  @{
 */

/*! \brief Get the list of available devices.
 *
 *  \param device_type is a bitfield that identifies the type of OpenCL device.
 *  The \a device_type can be used to query specific OpenCL devices or all
 *  OpenCL devices available.
 *
 *  \param num_entries is the number of cl_device_id entries that can be added
 *  to devices. If devices is not NULL, the \a num_entries must be greater than
 *  zero.
 *
 *  \param devices returns a list of OpenCL devices found. The cl_device_id
 *  values returned in devices can be used to identify a specific OpenCL device.
 *  If \a devices argument is NULL, this argument is ignored. The number of
 *  OpenCL devices returned is the mininum of value specified by \a num_entries
 *  or the number of OpenCL devices whose type matches device_type.
 *
 *  \param num_devices returns the number of OpenCL devices available that match
 *  device_type. If \a num_devices is NULL, this argument is ignored.
 *
 *  \return One of the following values:
 *    - CL_INVALID_DEVICE_TYPE if \a device_type is not a valid value.
 *    - CL_INVALID_VALUE if \a num_entries is equal to zero and devices is
 *      not NULL or if both \a num_devices and \a devices are NULL.
 *    - CL_DEVICE_ NOT_FOUND if no OpenCL devices that matched \a device_type
 *      were found.
 *    - CL_SUCCESS if the function is executed successfully.
 *
 *  The application can query specific capabilities of the OpenCL device(s)
 *  returned by clGetDeviceIDs. This can be used by the application to
 *  determine which device(s) to use.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetDeviceIDs,
              (cl_platform_id platform, cl_device_type device_type, cl_uint num_entries,
               cl_device_id* devices, cl_uint* num_devices)) {
  if (platform != NULL && platform != AMD_PLATFORM) {
    return CL_INVALID_PLATFORM;
  }

  if (((num_entries > 0 || num_devices == NULL) && devices == NULL) ||
      (num_entries == 0 && devices != NULL)) {
    return CL_INVALID_VALUE;
  }
  if (device_type == 0) {
    return CL_INVALID_DEVICE_TYPE;
  }
  // Get all available devices
  if (!amd::Device::getDeviceIDs(device_type, num_entries, devices, num_devices, false)) {
    return CL_DEVICE_NOT_FOUND;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \fn clGetDeviceInfo
 *
 *  \brief Get specific information about an OpenCL device.
 *
 *  \param device is a device returned by clGetDeviceIDs.
 *
 *  \param param_name is an enum that identifies the device information being
 *  queried.
 *
 *  \param param_value is a pointer to memory location where appropriate values
 *  for a given \a param_name will be returned. If \a param_value is NULL,
 *  it is ignored.
 *
 *  \param param_value_size specifies the size in bytes of memory pointed to
 *  by \a param_value. This size in bytes must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data being
 *  queried by param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *    - CL_INVALID_DEVICE if device is not valid.
 *    - CL_INVALID_VALUE if param_name is not one of the supported values
 *      or if size in bytes specified by \a param_value_size is < size of return
 *      type and \a param_value is not a NULL value.
 *    - CL_SUCCESS if the function is executed successfully.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetDeviceInfo,
              (cl_device_id device, cl_device_info param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

#define CASE(param_name, field_name)                                                               \
  case param_name:                                                                                 \
    return amd::clGetInfo(as_amd(device)->info().field_name, param_value_size, param_value,        \
                          param_value_size_ret);

  switch (param_name) {
    case CL_DEVICE_TYPE: {
      // For cl_device_type, we need to mask out the default bit.
      cl_device_type device_type = as_amd(device)->type();
      return amd::clGetInfo(device_type, param_value_size, param_value, param_value_size_ret);
    }
      CASE(CL_DEVICE_VENDOR_ID, vendorId_);
      CASE(CL_DEVICE_MAX_COMPUTE_UNITS, maxComputeUnits_);
      CASE(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, maxWorkItemDimensions_);
      CASE(CL_DEVICE_MAX_WORK_GROUP_SIZE, preferredWorkGroupSize_);
      CASE(CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD, preferredWorkGroupSize_);
      CASE(CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD, maxWorkGroupSize_);
      CASE(CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemSizes_);
      CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, preferredVectorWidthChar_);
      CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, preferredVectorWidthShort_);
      CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, preferredVectorWidthInt_);
      CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, preferredVectorWidthLong_);
      CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, preferredVectorWidthFloat_);
      CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, preferredVectorWidthDouble_);
      CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, preferredVectorWidthDouble_);
      CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, nativeVectorWidthChar_);
      CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, nativeVectorWidthShort_);
      CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, nativeVectorWidthInt_);
      CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, nativeVectorWidthLong_);
      CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, nativeVectorWidthFloat_);
      CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, nativeVectorWidthDouble_);
      CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, nativeVectorWidthDouble_);
      CASE(CL_DEVICE_MAX_CLOCK_FREQUENCY, maxEngineClockFrequency_);
      CASE(CL_DEVICE_ADDRESS_BITS, addressBits_);
      CASE(CL_DEVICE_MAX_READ_IMAGE_ARGS, maxReadImageArgs_);
      CASE(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, maxWriteImageArgs_);
      CASE(CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, maxReadWriteImageArgs_);
      CASE(CL_DEVICE_MAX_MEM_ALLOC_SIZE, maxMemAllocSize_);
      CASE(CL_DEVICE_IMAGE2D_MAX_WIDTH, image2DMaxWidth_);
      CASE(CL_DEVICE_IMAGE2D_MAX_HEIGHT, image2DMaxHeight_);
      CASE(CL_DEVICE_IMAGE3D_MAX_WIDTH, image3DMaxWidth_);
      CASE(CL_DEVICE_IMAGE3D_MAX_HEIGHT, image3DMaxHeight_);
      CASE(CL_DEVICE_IMAGE3D_MAX_DEPTH, image3DMaxDepth_);
      CASE(CL_DEVICE_IMAGE_SUPPORT, imageSupport_);
      CASE(CL_DEVICE_MAX_PARAMETER_SIZE, maxParameterSize_);
      CASE(CL_DEVICE_MAX_SAMPLERS, maxSamplers_);
      CASE(CL_DEVICE_MEM_BASE_ADDR_ALIGN, memBaseAddrAlign_);
      CASE(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, minDataTypeAlignSize_);
      CASE(CL_DEVICE_HALF_FP_CONFIG, halfFPConfig_);
      CASE(CL_DEVICE_SINGLE_FP_CONFIG, singleFPConfig_);
      CASE(CL_DEVICE_DOUBLE_FP_CONFIG, doubleFPConfig_);
      CASE(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, globalMemCacheType_);
      CASE(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, globalMemCacheLineSize_);
      CASE(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, globalMemCacheSize_);
      CASE(CL_DEVICE_GLOBAL_MEM_SIZE, globalMemSize_);
      CASE(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, maxConstantBufferSize_);
      CASE(CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD, preferredConstantBufferSize_);
      CASE(CL_DEVICE_MAX_CONSTANT_ARGS, maxConstantArgs_);
      CASE(CL_DEVICE_LOCAL_MEM_TYPE, localMemType_);
      CASE(CL_DEVICE_LOCAL_MEM_SIZE, localMemSize_);
      CASE(CL_DEVICE_ERROR_CORRECTION_SUPPORT, errorCorrectionSupport_);
      CASE(CL_DEVICE_HOST_UNIFIED_MEMORY, hostUnifiedMemory_);
      CASE(CL_DEVICE_PROFILING_TIMER_RESOLUTION, profilingTimerResolution_);
      CASE(CL_DEVICE_PROFILING_TIMER_OFFSET_AMD, profilingTimerOffset_);
      CASE(CL_DEVICE_ENDIAN_LITTLE, littleEndian_);
      CASE(CL_DEVICE_AVAILABLE, available_);
      CASE(CL_DEVICE_COMPILER_AVAILABLE, compilerAvailable_);
      CASE(CL_DEVICE_EXECUTION_CAPABILITIES, executionCapabilities_);
      CASE(CL_DEVICE_SVM_CAPABILITIES, svmCapabilities_);
      CASE(CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, preferredPlatformAtomicAlignment_);
      CASE(CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, preferredGlobalAtomicAlignment_);
      CASE(CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT, preferredLocalAtomicAlignment_);
      CASE(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, queueProperties_);
      CASE(CL_DEVICE_PLATFORM, platform_);
      CASE(CL_DEVICE_NAME, name_);
      CASE(CL_DEVICE_VENDOR, vendor_);
      CASE(CL_DRIVER_VERSION, driverVersion_);
      CASE(CL_DEVICE_PROFILE, profile_);
      CASE(CL_DEVICE_VERSION, version_);
      CASE(CL_DEVICE_OPENCL_C_VERSION, oclcVersion_);
      CASE(CL_DEVICE_EXTENSIONS, extensions_);
      CASE(CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT, maxAtomicCounters_);
      CASE(CL_DEVICE_TOPOLOGY_AMD, deviceTopology_);
      CASE(CL_DEVICE_MAX_SEMAPHORE_SIZE_AMD, maxSemaphoreSize_);
      CASE(CL_DEVICE_BOARD_NAME_AMD, boardName_);
      CASE(CL_DEVICE_SPIR_VERSIONS, spirVersions_);
      CASE(CL_DEVICE_IL_VERSION, spirVersions_);
      CASE(CL_DEVICE_MAX_PIPE_ARGS, maxPipeArgs_);
      CASE(CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, maxPipeActiveReservations_);
      CASE(CL_DEVICE_PIPE_MAX_PACKET_SIZE, maxPipePacketSize_);
      CASE(CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, maxGlobalVariableSize_);
      CASE(CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE, globalVariablePreferredTotalSize_);
      CASE(CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES, queueOnDeviceProperties_);
      CASE(CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, queueOnDevicePreferredSize_);
      CASE(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, queueOnDeviceMaxSize_);
      CASE(CL_DEVICE_MAX_ON_DEVICE_QUEUES, maxOnDeviceQueues_);
      CASE(CL_DEVICE_MAX_ON_DEVICE_EVENTS, maxOnDeviceEvents_);
      CASE(CL_DEVICE_LINKER_AVAILABLE, linkerAvailable_);
      CASE(CL_DEVICE_BUILT_IN_KERNELS, builtInKernels_);
      CASE(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, imageMaxBufferSize_);
      CASE(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, imageMaxArraySize_);
    case CL_DEVICE_PARENT_DEVICE: {
      cl_device_id parent = (cl_device_id)0;
      return amd::clGetInfo(parent, param_value_size, param_value, param_value_size_ret);
    }
      CASE(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, maxComputeUnits_);
    case CL_DEVICE_PARTITION_PROPERTIES: {
      cl_device_partition_property cl_property = {};
      return amd::clGetInfo(cl_property, param_value_size, param_value, param_value_size_ret);
    }
    case CL_DEVICE_PARTITION_AFFINITY_DOMAIN: {
      cl_device_affinity_domain deviceAffinity = {};
      return amd::clGetInfo(deviceAffinity, param_value_size, param_value, param_value_size_ret);
    }
    case CL_DEVICE_PARTITION_TYPE: {
      cl_device_partition_property cl_property = {};
      return amd::clGetInfo(cl_property, param_value_size, param_value, param_value_size_ret);
    }
    case CL_DEVICE_REFERENCE_COUNT: {
      cl_uint count = as_amd(device)->referenceCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
      CASE(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, preferredInteropUserSync_);
      CASE(CL_DEVICE_PRINTF_BUFFER_SIZE, printfBufferSize_);
      CASE(CL_DEVICE_IMAGE_PITCH_ALIGNMENT, imagePitchAlignment_);
      CASE(CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT, imageBaseAddressAlignment_);

    default:
      break;
  }
  if (as_amd(device)->type() == CL_DEVICE_TYPE_GPU) {
    switch (param_name) {
      case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD: {
        // Free memory should contain 2 values:
        // total free memory and the biggest free block
        size_t freeMemory[2];
        if (!as_amd(device)->globalFreeMemory(freeMemory)) {
          return CL_INVALID_DEVICE;
        }
        if (param_value_size < sizeof(freeMemory)) {
          // Return just total free memory if the app provided space for one value
          return amd::clGetInfo(freeMemory[0], param_value_size, param_value, param_value_size_ret);
        } else {
          return amd::clGetInfo(freeMemory, param_value_size, param_value, param_value_size_ret);
        }
      }
        CASE(CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD, simdPerCU_);
        CASE(CL_DEVICE_SIMD_WIDTH_AMD, simdWidth_);
        CASE(CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD, simdInstructionWidth_);
        CASE(CL_DEVICE_WAVEFRONT_WIDTH_AMD, wavefrontWidth_);
      case CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD: {
        cl_uint globalMemChannels = as_amd(device)->info().vramBusBitWidth_ / 32;
        return amd::clGetInfo(globalMemChannels, param_value_size, param_value,
                              param_value_size_ret);
      }
        CASE(CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD, globalMemChannelBanks_);
        CASE(CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD, globalMemChannelBankWidth_);
        CASE(CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD, localMemSizePerCU_);
        CASE(CL_DEVICE_LOCAL_MEM_BANKS_AMD, localMemBanks_);
        CASE(CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD, threadTraceEnable_);
      case CL_DEVICE_GFXIP_MAJOR_AMD: {
        cl_uint major = as_amd(device)->isa().versionMajor();
        return amd::clGetInfo(major, param_value_size, param_value, param_value_size_ret);
      }
      case CL_DEVICE_GFXIP_MINOR_AMD: {
        cl_uint minor = as_amd(device)->isa().versionMinor();
        return amd::clGetInfo(minor, param_value_size, param_value, param_value_size_ret);
      }
        CASE(CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD, numAsyncQueues_);
#define CL_DEVICE_MAX_REAL_TIME_COMPUTE_QUEUES_AMD 0x404D
#define CL_DEVICE_MAX_REAL_TIME_COMPUTE_UNITS_AMD 0x404E
#define CL_DEVICE_MAX_REAL_TIME_COMPUTE_UNITS_GRANULARITY_AMD 0x403A
        CASE(CL_DEVICE_MAX_REAL_TIME_COMPUTE_QUEUES_AMD, numRTQueues_);
        CASE(CL_DEVICE_MAX_REAL_TIME_COMPUTE_UNITS_AMD, numRTCUs_);
        CASE(CL_DEVICE_MAX_REAL_TIME_COMPUTE_UNITS_GRANULARITY_AMD, granularityRTCUs_);
      case CL_DEVICE_NUM_P2P_DEVICES_AMD: {
        cl_uint num_p2p_devices = as_amd(device)->p2pDevices_.size();
        return amd::clGetInfo(num_p2p_devices, param_value_size, param_value, param_value_size_ret);
      }
      case CL_DEVICE_P2P_DEVICES_AMD: {
        uint valueSize = as_amd(device)->p2pDevices_.size() * sizeof(cl_device_id);
        if (param_value != NULL) {
          if ((param_value_size < valueSize) || (param_value_size == 0)) {
            return CL_INVALID_VALUE;
          }
        } else {
          return CL_INVALID_VALUE;
        }
        memcpy(param_value, as_amd(device)->p2pDevices_.data(), valueSize);
        *not_null(param_value_size_ret) = valueSize;
        if (param_value != NULL && param_value_size > valueSize) {
          ::memset(static_cast<char*>(param_value) + valueSize, '\0', param_value_size - valueSize);
        }
        return CL_SUCCESS;
      }
        CASE(CL_DEVICE_PCIE_ID_AMD, pcieDeviceId_);
#define CL_DEVICE_NUMERIC_VERSION 0x105E
      case CL_DEVICE_NUMERIC_VERSION: {
        std::string driverVersion = as_amd(device)->info().driverVersion_;
        std::string drvVersion(driverVersion.c_str());
        drvVersion = drvVersion.substr(0, 4);  // calVersion.find_first_of("(") + 4);
        cl_uint numVersion = atoi(drvVersion.c_str());
        return amd::clGetInfo(numVersion, param_value_size, param_value, param_value_size_ret);
      }

      // cl_khr_device_uuid.
      //
      // The UUID must be byte-identical to the one AMD's Vulkan driver reports in
      // VkPhysicalDeviceIDProperties::deviceUUID, otherwise CL-Vulkan interop clients
      // cannot pair up the two handles for the same GPU. That driver builds it from the
      // PCI address as four little-endian uint32s - domain, bus, device, function - so
      // this must too. ROCr's own HSA_AMD_AGENT_INFO_UUID is an ASCII string and would
      // never match, so info_.uuid_ is deliberately not used here.
      case CL_DEVICE_UUID_KHR: {
        const auto& info = as_amd(device)->info();
        const uint32_t pciAddress[4] = {info.pciDomainID,
                                        info.deviceTopology_.pcie.bus,
                                        info.deviceTopology_.pcie.device,
                                        info.deviceTopology_.pcie.function};
        cl_uchar uuid[CL_UUID_SIZE_KHR];
        ::memcpy(uuid, pciAddress, sizeof(uuid));
        return amd::clGetInfo(uuid, param_value_size, param_value, param_value_size_ret);
      }
      case CL_DRIVER_UUID_KHR: {
        // Matches VkPhysicalDeviceIDProperties::driverUUID for this driver: the
        // zero-padded ASCII tag, not a binary UUID.
        cl_uchar driverUuid[CL_UUID_SIZE_KHR] = {0};
#if defined(_WIN32)
        static constexpr char kDriverUuid[] = "AMD-WIN-DRV";
#else
        static constexpr char kDriverUuid[] = "AMD-LINUX-DRV";
#endif
        static_assert(sizeof(kDriverUuid) <= CL_UUID_SIZE_KHR,
                      "driver UUID tag does not fit in CL_UUID_SIZE_KHR");
        ::memcpy(driverUuid, kDriverUuid, sizeof(kDriverUuid) - 1);
        return amd::clGetInfo(driverUuid, param_value_size, param_value, param_value_size_ret);
      }
      case CL_DEVICE_LUID_VALID_KHR: {
        const auto& info = as_amd(device)->info();
        cl_bool luidValid =
            ((info.luidLowPart_ != 0) || (info.luidHighPart_ != 0)) ? CL_TRUE : CL_FALSE;
        return amd::clGetInfo(luidValid, param_value_size, param_value, param_value_size_ret);
      }
      case CL_DEVICE_LUID_KHR: {
        const auto& info = as_amd(device)->info();
        // CL_LUID_SIZE_KHR is the Windows LUID layout: low DWORD first, then high.
        cl_uchar luid[CL_LUID_SIZE_KHR];
        const uint32_t low = info.luidLowPart_;
        const uint32_t high = info.luidHighPart_;
        ::memcpy(&luid[0], &low, sizeof(low));
        ::memcpy(&luid[sizeof(low)], &high, sizeof(high));
        return amd::clGetInfo(luid, param_value_size, param_value, param_value_size_ret);
      }
        CASE(CL_DEVICE_NODE_MASK_KHR, luidDeviceNodeMask_);

      // cl_khr_semaphore. An array, and empty where the backend does not enable the
      // extension, so it agrees with this device's extension string.
      case CL_DEVICE_SEMAPHORE_TYPES_KHR: {
        std::vector<cl_semaphore_type_khr> types;
#ifdef _WIN32
        types.push_back(CL_SEMAPHORE_TYPE_BINARY_KHR);
#endif  //_WIN32
        return clGetInfoVector(types, param_value_size, param_value, param_value_size_ret);
      }

      // cl_khr_external_memory / cl_khr_external_semaphore handle-type lists. All four
      // are plain arrays and may legitimately be empty, which clGetInfoVector reports as
      // a zero size rather than an error.
      case CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR:
        return clGetInfoVector(as_amd(device)->info().externalMemoryImportHandleTypes_,
                               param_value_size, param_value, param_value_size_ret);
      case CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR:
        return clGetInfoVector(
            as_amd(device)->info().externalMemoryImportAssumeLinearImagesHandleTypes_,
            param_value_size, param_value, param_value_size_ret);
      case CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR:
        return clGetInfoVector(as_amd(device)->info().semaphoreImportHandleTypes_, param_value_size,
                               param_value, param_value_size_ret);
      case CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR:
        return clGetInfoVector(as_amd(device)->info().semaphoreExportHandleTypes_, param_value_size,
                               param_value, param_value_size_ret);

      default:
        break;
    }
  }
#undef CASE

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clCreateSubDevices,
              (cl_device_id in_device, const cl_device_partition_property* partition_properties,
               cl_uint num_entries, cl_device_id* out_devices, cl_uint* num_devices)) {
  if (!is_valid(in_device)) {
    return CL_INVALID_DEVICE;
  }
  if (partition_properties == NULL || *partition_properties == 0u) {
    return CL_INVALID_VALUE;
  }
  if ((num_devices == NULL && out_devices == NULL) || (num_entries == 0 && out_devices != NULL)) {
    return CL_INVALID_VALUE;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clRetainDevice, (cl_device_id device)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }
  as_amd(device)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clReleaseDevice, (cl_device_id device)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }
  as_amd(device)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
