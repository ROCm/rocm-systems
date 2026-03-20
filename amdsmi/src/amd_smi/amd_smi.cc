// SPDX-License-Identifier: MIT
/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cassert>
#include <cerrno>
#include <sys/utsname.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>

#include <cstdlib>
#include <cctype>
#include <string>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <limits>
#include <functional>

#include "config/amd_smi_config.h"
#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/scoped_fd.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_cper.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/nic/amd_smi_ainic_device.h"
#include "amdsmi_unified/interface/smi_nic_interface.h"

#ifdef BRCM_NIC
#include "amd_smi/impl/nic/amd_smi_nic_device.h"
#include "amd_smi/impl/nic/amd_smi_switch_device.h"
#include "amd_smi/impl/nic/amd_smi_lspci_commands.h"
#endif//BRCM_NIC
#include "amd_smi/impl/amd_smi_uuid.h"
#include "amd_smi/impl/xf86drm.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_kfd.h"

#include "dxcore_loader.h"
#include "platform.h"
#include "device.h"

// a global instance of std::mutex to protect data passed during threads
std::mutex myMutex;

using namespace wsl::thunk;

static inline amdsmi_status_t translateCodeToSmiStatus(ErrorCode code) {
    switch (code) {
    case ErrorCode::Success:
        return AMDSMI_STATUS_SUCCESS;
    case ErrorCode::UnSupported:
        return AMDSMI_STATUS_NOT_SUPPORTED;
    case ErrorCode::NotReady:
        return AMDSMI_STATUS_RETRY;
    case ErrorCode::InitializationFailed:
        return AMDSMI_STATUS_INIT_ERROR;
    case ErrorCode::Timeout:
        return AMDSMI_STATUS_TIMEOUT;
    case ErrorCode::SyscallFail:
        return AMDSMI_STATUS_API_FAILED;
    case ErrorCode::InvalidParams:
    case ErrorCode::InvalidPointer:
        return AMDSMI_STATUS_INVAL;
    case ErrorCode::OutOfMemory:
    case ErrorCode::OutOfGpuMemory:
        return AMDSMI_STATUS_OUT_OF_RESOURCES;
    case ErrorCode::NotFound:
        return AMDSMI_STATUS_NOT_FOUND;
    default:
        return AMDSMI_STATUS_UNKNOWN_ERROR;
    }
}

// To enable multiple init and shutdown calls, the reference count is used
// to track the number of times the library has been initialized.
static int init_ref_count = 0;

#define	SIZE	10
char proc_id[SIZE] = "\0";

#define AMDSMI_CHECK_INIT() do { \
	if (init_ref_count == 0) { \
		return AMDSMI_STATUS_NOT_INIT; \
	} \
} while (0)

static const std::map<amdsmi_accelerator_partition_type_t, std::string> partition_types_map = {
  { AMDSMI_ACCELERATOR_PARTITION_SPX, "SPX" },
  { AMDSMI_ACCELERATOR_PARTITION_DPX, "DPX" },
  { AMDSMI_ACCELERATOR_PARTITION_TPX, "TPX" },
  { AMDSMI_ACCELERATOR_PARTITION_QPX, "QPX" },
  { AMDSMI_ACCELERATOR_PARTITION_CPX, "CPX" },
  { AMDSMI_ACCELERATOR_PARTITION_MAX, "MAX" },
};
static const std::map<amdsmi_accelerator_partition_type_t,
                     rsmi_compute_partition_type_t> accelerator_to_RSMI = {
  { AMDSMI_ACCELERATOR_PARTITION_SPX, RSMI_COMPUTE_PARTITION_SPX },
  { AMDSMI_ACCELERATOR_PARTITION_DPX, RSMI_COMPUTE_PARTITION_DPX },
  { AMDSMI_ACCELERATOR_PARTITION_TPX, RSMI_COMPUTE_PARTITION_TPX },
  { AMDSMI_ACCELERATOR_PARTITION_QPX, RSMI_COMPUTE_PARTITION_QPX },
  { AMDSMI_ACCELERATOR_PARTITION_CPX, RSMI_COMPUTE_PARTITION_CPX }
};
static const std::map<amdsmi_accelerator_partition_resource_type_t,
    std::string> resource_types_map = {
  { AMDSMI_ACCELERATOR_XCC, "XCC" },
  { AMDSMI_ACCELERATOR_ENCODER, "ENCODER" },
  { AMDSMI_ACCELERATOR_DECODER, "DECODER" },
  { AMDSMI_ACCELERATOR_DMA, "DMA" },
  { AMDSMI_ACCELERATOR_JPEG, "JPEG" },
  { AMDSMI_ACCELERATOR_MAX, "MAX" },
};

static const std::map<amdsmi_memory_partition_type_t,
                     rsmi_memory_partition_type> nps_amdsmi_to_RSMI = {
  { AMDSMI_MEMORY_PARTITION_UNKNOWN, RSMI_MEMORY_PARTITION_UNKNOWN },
  { AMDSMI_MEMORY_PARTITION_NPS1, RSMI_MEMORY_PARTITION_NPS1 },
  { AMDSMI_MEMORY_PARTITION_NPS2, RSMI_MEMORY_PARTITION_NPS2 },
  { AMDSMI_MEMORY_PARTITION_NPS4, RSMI_MEMORY_PARTITION_NPS4 },
  { AMDSMI_MEMORY_PARTITION_NPS8, RSMI_MEMORY_PARTITION_NPS8 }
};

static amdsmi_status_t get_gpu_device_from_handle(amdsmi_processor_handle processor_handle,
            amd::smi::AMDSmiGPUDevice** gpudevice) {
    AMDSMI_CHECK_INIT();
    std::ostringstream ss;

    if (processor_handle == nullptr || gpudevice == nullptr) {
        ss << __PRETTY_FUNCTION__
        << " | processor_handle is NULL; returning: AMDSMI_STATUS_INVAL";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiProcessor* device = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_processor(processor_handle, &device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    if (device->get_processor_type() == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        *gpudevice = static_cast<amd::smi::AMDSmiGPUDevice*>(device);
        return AMDSMI_STATUS_SUCCESS;
    }

    ss << __PRETTY_FUNCTION__
    << " | returning AMDSMI_STATUS_NOT_SUPPORTED";
    LOG_ERROR(ss);
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
template <typename F, typename ...Args>
amdsmi_status_t rsmi_wrapper(F && f,
    amdsmi_processor_handle processor_handle, uint32_t increment_gpu_id, Args &&... args) {

    AMDSMI_CHECK_INIT();

    std::ostringstream ss;
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    ss << __PRETTY_FUNCTION__ << " | get_gpu_device_from_handle status = "
       << smi_amdgpu_get_status_string(r, false);
    LOG_INFO(ss);
    if (r != AMDSMI_STATUS_SUCCESS) return r;


    uint32_t total_num_gpu_processors = 0;
    rsmi_num_monitor_devices(&total_num_gpu_processors);
    uint32_t gpu_index = gpu_device->get_gpu_id() + increment_gpu_id;
    ss << __PRETTY_FUNCTION__ << " | total_num_gpu_processors: " << total_num_gpu_processors
    << "; gpu_index: " << gpu_index;
    LOG_DEBUG(ss);
    if ((gpu_index + 1) > total_num_gpu_processors) {
        ss << __PRETTY_FUNCTION__ << " | returning status = AMDSMI_STATUS_NOT_FOUND";
        LOG_INFO(ss);
        return AMDSMI_STATUS_NOT_FOUND;
    }

    auto rstatus = std::forward<F>(f)(gpu_index,
                    std::forward<Args>(args)...);
    r = amd::smi::rsmi_to_amdsmi_status(rstatus);
    std::string status_string = smi_amdgpu_get_status_string(r, false);
    ss << __PRETTY_FUNCTION__ << " | returning status = " << status_string;
    LOG_INFO(ss);
    return r;
}
static amdsmi_status_t get_ainic_device_from_handle(amdsmi_processor_handle processor_handle,
            amd::smi::AMDSmiAINICDevice **nicdevice) {
    AMDSMI_CHECK_INIT();
    if (processor_handle == nullptr || nicdevice == nullptr) return AMDSMI_STATUS_INVAL;

    amd::smi::AMDSmiProcessor *device = nullptr;
    amdsmi_status_t r =
        amd::smi::AMDSmiSystem::getInstance().handle_to_processor(processor_handle, &device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    if (device->get_processor_type() == AMDSMI_PROCESSOR_TYPE_AMD_NIC) {
        *nicdevice = static_cast<amd::smi::AMDSmiAINICDevice *>(device);
        return AMDSMI_STATUS_SUCCESS;
    }

    return AMDSMI_STATUS_NOT_SUPPORTED;
}
#ifdef BRCM_NIC
static amdsmi_status_t get_nic_device_from_handle(amdsmi_processor_handle processor_handle,
            amd::smi::AMDSmiNICDevice **nicdevice) {
    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || nicdevice == nullptr) return AMDSMI_STATUS_INVAL;

    amd::smi::AMDSmiProcessor *device = nullptr;
    amdsmi_status_t r =
        amd::smi::AMDSmiSystem::getInstance().handle_to_processor(processor_handle, &device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    if (device->get_processor_type() == AMDSMI_PROCESSOR_TYPE_BRCM_NIC) {
        *nicdevice = static_cast<amd::smi::AMDSmiNICDevice *>(device);
        return AMDSMI_STATUS_SUCCESS;
    }

    return AMDSMI_STATUS_NOT_SUPPORTED;
}

static amdsmi_status_t get_switch_device_from_handle(amdsmi_processor_handle processor_handle,
            amd::smi::AMDSmiSWITCHDevice **switchdevice) {
    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || switchdevice == nullptr) return AMDSMI_STATUS_INVAL;

    amd::smi::AMDSmiProcessor *device = nullptr;
    amdsmi_status_t r =
        amd::smi::AMDSmiSystem::getInstance().handle_to_processor(processor_handle, &device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    if (device->get_processor_type() == AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH) {
        *switchdevice = static_cast<amd::smi::AMDSmiSWITCHDevice *>(device);
        return AMDSMI_STATUS_SUCCESS;
    }

    return AMDSMI_STATUS_NOT_SUPPORTED;
}

template <typename F, typename... Args>
static amdsmi_status_t 
rsmi_nic_wrapper(F &&f, amdsmi_processor_handle processor_handle, Args &&... args) {

  std::ostringstream ss;
  const char *status_string = nullptr;

  amd::smi::AMDSmiNICDevice *nic_device = nullptr;
  amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
  if (r != AMDSMI_STATUS_SUCCESS) {
    amdsmi_status_code_to_string(r, &status_string);
    ss << __PRETTY_FUNCTION__ << " | " << status_string;
    LOG_INFO(ss);
    return r;
  }

  uint32_t nic_index = nic_device->get_nic_id();
  auto rstatus = std::forward<F>(f)(nic_index, std::forward<Args>(args)...);
  r = amd::smi::rsmi_to_amdsmi_status(rstatus);
  amdsmi_status_code_to_string(r, &status_string);
  ss << __PRETTY_FUNCTION__ << " | returning status = " << status_string;
  if (r != AMDSMI_STATUS_SUCCESS) {
    LOG_ERROR(ss);
  }
  else {
    LOG_INFO(ss);
  }
  return r;
}

template <typename F, typename... Args>
amdsmi_status_t rsmi_switch_wrapper(F &&f, amdsmi_processor_handle processor_handle, Args &&... args) {

  std::ostringstream ss;
  const char *status_string = nullptr;

  amd::smi::AMDSmiSWITCHDevice *switch_device = nullptr;
  amdsmi_status_t r = get_switch_device_from_handle(processor_handle, &switch_device);
  if (r != AMDSMI_STATUS_SUCCESS) {
    amdsmi_status_code_to_string(r, &status_string);
    ss << __PRETTY_FUNCTION__ << " | " << status_string;
    LOG_INFO(ss);
    return r;
  }

  uint32_t switch_index = switch_device->get_switch_id();
  auto rstatus = std::forward<F>(f)(switch_index, std::forward<Args>(args)...);
  r = amd::smi::rsmi_to_amdsmi_status(rstatus);
  amdsmi_status_code_to_string(r, &status_string);
  ss << __PRETTY_FUNCTION__ << " | returning status = " << status_string;
  if (r != AMDSMI_STATUS_SUCCESS) {
    LOG_ERROR(ss);
  }
  else {
    LOG_INFO(ss);
  }
  return r;
}
#endif//BRCM_NIC

amdsmi_status_t
amdsmi_init(uint64_t flags) {
    if (init_ref_count > 0 ) {
        init_ref_count++;
        return AMDSMI_STATUS_SUCCESS;
    }

    auto &loader = dxcore::DxcoreLoader::Instance();
    if (!loader.Initialize())
        return AMDSMI_STATUS_INIT_ERROR;

    auto code = Platform::instance().Init();
    if (code == ErrorCode::Success) {
        init_ref_count++;
    }
    return translateCodeToSmiStatus(code);
}

amdsmi_status_t
amdsmi_shut_down() {
    if (init_ref_count == 0) {
        return AMDSMI_STATUS_SUCCESS;
    }
    // Decrement the reference count
    init_ref_count--;
    // If the reference count is still greater than 0, return success
    if (init_ref_count > 0) {
        return AMDSMI_STATUS_SUCCESS;
    }
    Platform::instance().Destroy();
    dxcore::DxcoreLoader::Instance().Shutdown();
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_status_code_to_string(amdsmi_status_t status, const char **status_string) {
    switch (status) {
        case AMDSMI_STATUS_SUCCESS:
            *status_string = "AMDSMI_STATUS_SUCCESS: Call succeeded.";
            break;
        case AMDSMI_STATUS_INVAL:
            *status_string = "AMDSMI_STATUS_INVAL: Invalid parameters.";
            break;
        case AMDSMI_STATUS_NOT_SUPPORTED:
            *status_string = "AMDSMI_STATUS_NOT_SUPPORTED: Command not supported.";
            break;
        case AMDSMI_STATUS_NOT_YET_IMPLEMENTED:
            *status_string = "AMDSMI_STATUS_NOT_YET_IMPLEMENTED:  Not implemented yet.";
            break;
        case AMDSMI_STATUS_FAIL_LOAD_MODULE:
            *status_string = "AMDSMI_STATUS_FAIL_LOAD_MODULE: Fail to load lib module.";
            break;
        case AMDSMI_STATUS_FAIL_LOAD_SYMBOL:
            *status_string = "AMDSMI_STATUS_FAIL_LOAD_SYMBOL: Fail to load symbol.";
            break;
        case AMDSMI_STATUS_DRM_ERROR:
            *status_string = "AMDSMI_STATUS_DRM_ERROR: Error when calling libdrm function.";
            break;
        case AMDSMI_STATUS_API_FAILED:
            *status_string = "AMDSMI_STATUS_API_FAILED: API call failed.";
            break;
        case AMDSMI_STATUS_RETRY:
            *status_string = "AMDSMI_STATUS_RETRY: Retry operation.";
            break;
        case AMDSMI_STATUS_NO_PERM:
            *status_string = "AMDSMI_STATUS_NO_PERM: Permission Denied.";
            break;
        case AMDSMI_STATUS_INTERRUPT:
            *status_string = "AMDSMI_STATUS_INTERRUPT: An interrupt occurred during"
                " execution of function.";
            break;
        case AMDSMI_STATUS_IO:
            *status_string = "AMDSMI_STATUS_IO: I/O Error.";
            break;
        case AMDSMI_STATUS_ADDRESS_FAULT:
            *status_string = "AMDSMI_STATUS_ADDRESS_FAULT: Bad address.";
            break;
        case AMDSMI_STATUS_FILE_ERROR:
            *status_string = "AMDSMI_STATUS_FILE_ERROR: Problem accessing a file.";
            break;
        case AMDSMI_STATUS_OUT_OF_RESOURCES:
            *status_string = "AMDSMI_STATUS_OUT_OF_RESOURCES: Not enough memory.";
            break;
        case AMDSMI_STATUS_INTERNAL_EXCEPTION:
            *status_string = "AMDSMI_STATUS_INTERNAL_EXCEPTION: An internal exception was caught.";
            break;
        case AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS:
            *status_string = "AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS: The provided"
                " input is out of allowable or safe range.";
            break;
        case AMDSMI_STATUS_INIT_ERROR:
            *status_string = "AMDSMI_STATUS_INIT_ERROR: An error occurred when"
                " initializing internal data structures.";
            break;
        case AMDSMI_STATUS_REFCOUNT_OVERFLOW:
            *status_string = "AMDSMI_STATUS_REFCOUNT_OVERFLOW: An internal reference"
                " counter exceeded INT32_MAX.";
            break;
        case AMDSMI_STATUS_DIRECTORY_NOT_FOUND:
            *status_string = "AMDSMI_STATUS_DIRECTORY_NOT_FOUND: Error when a"
                " directory is not found, maps to ENOTDIR.";
            break;
        case AMDSMI_STATUS_IPC_ERROR:
            *status_string = "AMDSMI_STATUS_IPC_ERROR: An IPC error occurred.";
            break;
        case AMDSMI_STATUS_BUSY:
            *status_string = "AMDSMI_STATUS_BUSY: Processor busy.";
            break;
        case AMDSMI_STATUS_NOT_FOUND:
            *status_string = "AMDSMI_STATUS_NOT_FOUND: Processor Not found.";
            break;
        case AMDSMI_STATUS_NOT_INIT:
            *status_string = "AMDSMI_STATUS_NOT_INIT: Processor not initialized.";
            break;
        case AMDSMI_STATUS_NO_SLOT:
            *status_string = "AMDSMI_STATUS_NO_SLOT: No more free slot.";
            break;
        case AMDSMI_STATUS_DRIVER_NOT_LOADED:
            *status_string = "AMDSMI_STATUS_DRIVER_NOT_LOADED: Processor driver not loaded.";
            break;
        case AMDSMI_STATUS_NO_DATA:
            *status_string = "AMDSMI_STATUS_NO_DATA: No data was found for a given input.";
            break;
        case AMDSMI_STATUS_INSUFFICIENT_SIZE:
            *status_string = "AMDSMI_STATUS_INSUFFICIENT_SIZE: Not enough resources"
                " were available for the operation.";
            break;
        case AMDSMI_STATUS_UNEXPECTED_SIZE:
            *status_string = "AMDSMI_STATUS_UNEXPECTED_SIZE: An unexpected amount of data"
                " was read.";
            break;
        case AMDSMI_STATUS_UNEXPECTED_DATA:
            *status_string = "AMDSMI_STATUS_UNEXPECTED_DATA: The data read or provided to"
                " function is not what was expected.";
            break;
        case AMDSMI_STATUS_NON_AMD_CPU:
            *status_string = "AMDSMI_STATUS_NON_AMD_CPU: System has different cpu than AMD.";
            break;
        case AMDSMI_STATUS_NO_ENERGY_DRV:
            *status_string = "AMDSMI_STATUS_NO_ENERGY_DRV: Energy driver not found.";
            break;
        case AMDSMI_STATUS_NO_MSR_DRV:
            *status_string = "AMDSMI_STATUS_NO_MSR_DRV: MSR driver not found.";
            break;
        case AMDSMI_STATUS_NO_HSMP_DRV:
            *status_string = "AMDSMI_STATUS_NO_HSMP_DRV: HSMP driver not found.";
            break;
        case AMDSMI_STATUS_NO_HSMP_SUP:
            *status_string = "AMDSMI_STATUS_NO_HSMP_SUP: HSMP not supported.";
            break;
        case AMDSMI_STATUS_NO_HSMP_MSG_SUP:
            *status_string = "AMDSMI_STATUS_NO_HSMP_MSG_SUP: HSMP message/feature not supported.";
            break;
        case AMDSMI_STATUS_HSMP_TIMEOUT:
            *status_string = "AMDSMI_STATUS_HSMP_TIMEOUT: HSMP message timed out.";
            break;
        case AMDSMI_STATUS_NO_DRV:
            *status_string = "AMDSMI_STATUS_NO_DRV: No Energy and HSMP driver present.";
            break;
        case AMDSMI_STATUS_FILE_NOT_FOUND:
            *status_string = "AMDSMI_STATUS_FILE_NOT_FOUND: file or directory not found.";
            break;
        case AMDSMI_STATUS_ARG_PTR_NULL:
            *status_string = "AMDSMI_STATUS_ARG_PTR_NULL: Parsed argument is invalid.";
            break;
        case AMDSMI_STATUS_AMDGPU_RESTART_ERR:
            *status_string = "AMDSMI_STATUS_AMDGPU_RESTART_ERR: AMDGPU restart failed.";
            break;
        case AMDSMI_STATUS_SETTING_UNAVAILABLE:
            *status_string = "AMDSMI_STATUS_SETTING_UNAVAILABLE: Setting is not available.";
            break;
        case AMDSMI_STATUS_CORRUPTED_EEPROM:
            *status_string = "AMDSMI_STATUS_CORRUPTED_EEPROM: EEPROM is corrupted.";
            break;
        case AMDSMI_STATUS_MAP_ERROR:
            *status_string = "AMDSMI_STATUS_MAP_ERROR: The internal library error did"
                " not map to a status code.";
            break;
        case AMDSMI_STATUS_UNKNOWN_ERROR:
            *status_string = "AMDSMI_STATUS_UNKNOWN_ERROR: An unknown error occurred.";
            break;
        default:
            // The case above didn't have a match, so look up the amdsmi status in the rsmi
            // status map
            // If found, get the rsmi status string.  If not, return unknown error string
            for (auto& iter : amd::smi::rsmi_status_map) {
                if (iter.second == status) {
                    rsmi_status_string(iter.first, status_string);
                    return AMDSMI_STATUS_SUCCESS;
                }
            }
            // Not found
            *status_string = "An unknown error occurred";
            return AMDSMI_STATUS_UNKNOWN_ERROR;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_socket_handles(uint32_t *socket_count,
                amdsmi_socket_handle* socket_handles) {

    AMDSMI_CHECK_INIT();

    if (socket_count == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    auto& platform = Platform::instance();

    uint32_t socket_size = platform.GetLdaChainCount();
    if (socket_handles == nullptr) {
        *socket_count = socket_size;
        return AMDSMI_STATUS_SUCCESS;
    }

    // If the socket_handles can hold all sockets, return all of them.
    *socket_count = *socket_count >= socket_size ? socket_size : *socket_count;
    // Copy the socket handles
    for (uint32_t i = 0; i < *socket_count; i++) {
        socket_handles[i] = platform.GetLdaChain(i);
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_socket_info(
                amdsmi_socket_handle socket_handle,
                size_t len, char *name) {
    AMDSMI_CHECK_INIT();

    if (socket_handle == nullptr || name == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    memset(name, 0, sizeof(char) * len);

    return AMDSMI_STATUS_SUCCESS;
}

#ifdef ENABLE_ESMI_LIB
amdsmi_status_t amdsmi_get_processor_info(
                amdsmi_processor_handle processor_handle,
                size_t len, char *name) {
    char proc_id[16] = {0};
    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || name == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiProcessor* processor = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_processor(processor_handle, &processor);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    snprintf(proc_id, sizeof(proc_id), "%d", processor->get_processor_index());
    snprintf(name, len, "%s", proc_id);

    return AMDSMI_STATUS_SUCCESS;
}
#endif

amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle,
                                    uint32_t* processor_count,
                                    amdsmi_processor_handle* processor_handles) {
    AMDSMI_CHECK_INIT();

    if (processor_count == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    auto lda_chain = reinterpret_cast<wsl::thunk::LdaChain *>(socket_handle);
    if (lda_chain == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto device_count = static_cast<uint32_t>(lda_chain->ChainedDeviceCount());

    // Get the processor count only
    if (processor_handles == nullptr) {
        *processor_count = device_count;
        return AMDSMI_STATUS_SUCCESS;
    }

    // If the processor_handles can hold all processors, return all of them.
    *processor_count = *processor_count >= device_count ? device_count : *processor_count;

    // Copy the processor handles
    for (uint32_t i = 0; i < *processor_count; i++) {
        processor_handles[i] = reinterpret_cast<amdsmi_processor_handle>(lda_chain->ChainedDevice(i));
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_nic_processor_handles(amdsmi_socket_handle socket_handle,
    uint32_t* processor_count,
    amdsmi_processor_handle* processor_handles) {
    AMDSMI_CHECK_INIT();

    if (processor_count == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Get the socket object via socket handle.
    amd::smi::AMDSmiSocket* socket = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_socket(socket_handle, &socket);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    std::vector<amd::smi::AMDSmiProcessor*>& processors = socket->get_processors(AMDSMI_PROCESSOR_TYPE_BRCM_NIC);
    uint32_t processor_size = static_cast<uint32_t>(processors.size());
    // Get the processor count only
    if (processor_handles == nullptr) {
        *processor_count = processor_size;
        return AMDSMI_STATUS_SUCCESS;
    }

    // If the processor_handles can hold all processors, return all of them.
    *processor_count = *processor_count >= processor_size ? processor_size : *processor_count;

    // Copy the processor handles
    for (uint32_t i = 0; i < *processor_count; i++) {
        processor_handles[i] = reinterpret_cast<amdsmi_processor_handle>(processors[i]);
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_switch_processor_handles(amdsmi_socket_handle socket_handle,
    uint32_t* processor_count,
    amdsmi_processor_handle* processor_handles) {
    AMDSMI_CHECK_INIT();

    if (processor_count == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Get the socket object via socket handle.
    amd::smi::AMDSmiSocket* socket = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_socket(socket_handle, &socket);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    processor_type_t processor_type = static_cast<processor_type_t>(AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH);
    std::vector<amd::smi::AMDSmiProcessor*>& processors = socket->get_processors(processor_type);
    uint32_t processor_size = static_cast<uint32_t>(processors.size());
    // Get the processor count only
    if (processor_handles == nullptr) {
        *processor_count = processor_size;
        return AMDSMI_STATUS_SUCCESS;
    }

    // If the processor_handles can hold all processors, return all of them.
    *processor_count = *processor_count >= processor_size ? processor_size : *processor_count;

    // Copy the processor handles
    for (uint32_t i = 0; i < *processor_count; i++) {
        processor_handles[i] = reinterpret_cast<amdsmi_processor_handle>(processors[i]);
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_node_handle(amdsmi_processor_handle processor_handle,
                                    amdsmi_node_handle *node_handle) {

    AMDSMI_CHECK_INIT();

    if (node_handle == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Check if OAM ID is 0
    amdsmi_asic_info_t asic_info;
    amdsmi_status_t r = amdsmi_get_gpu_asic_info(processor_handle, &asic_info);
    if (r != AMDSMI_STATUS_SUCCESS) {
        return r;
    }

    if (asic_info.oam_id != 0) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    // Get renderPath
    amdsmi_enumeration_info_t enumeration_info;
    r = amdsmi_get_gpu_enumeration_info(processor_handle, &enumeration_info);
    if (r != AMDSMI_STATUS_SUCCESS) {
        return r;
    }

    namespace fs = std::filesystem;

    // Construct the path from /sys/class/drm/renderD* device
    fs::path drm_device_path = fs::path("/sys/class/drm") / ("renderD" + std::to_string(enumeration_info.drm_render)) / "device";
    fs::path found_board;

    try {
        // Navigate to the board directory from the DRM device path
        fs::path board_dir = drm_device_path / "board";
        fs::path npm_status = board_dir / "npm_status";

        // Check if board directory and npm_status exist
        if (fs::exists(board_dir) && fs::is_directory(board_dir) && fs::exists(npm_status)) {
            found_board = board_dir;
        }
    } catch (...) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    if (found_board.empty()) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    // Store board path so node handle remains valid for library lifetime.
    static std::mutex g_node_mu;
    static std::map<std::string, std::unique_ptr<std::string>> g_node_registry;

    std::string board_path = found_board.string();
    {
        std::lock_guard<std::mutex> lk(g_node_mu);
        auto it = g_node_registry.find(board_path);
        if (it == g_node_registry.end()) {
            auto ptr = std::make_unique<std::string>(board_path);
            amdsmi_node_handle h = reinterpret_cast<amdsmi_node_handle>(ptr.get());
            g_node_registry.emplace(board_path, std::move(ptr));
            *node_handle = h;
        } else {
            *node_handle = reinterpret_cast<amdsmi_node_handle>(it->second.get());
        }
    }

    return AMDSMI_STATUS_SUCCESS;

}

#ifdef ENABLE_ESMI_LIB
amdsmi_status_t amdsmi_get_processor_count_from_handles(amdsmi_processor_handle* processor_handles,
                                                        uint32_t* processor_count, uint32_t* nr_cpusockets,
                                                        uint32_t* nr_cpucores, uint32_t* nr_gpus) {

    AMDSMI_CHECK_INIT();

    uint32_t count_cpusockets = 0;
    uint32_t count_cpucores = 0;
    uint32_t count_gpus = 0;
    processor_type_t processor_type;

    if (processor_count == nullptr || processor_handles == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    for (uint32_t i = 0; i < *processor_count; i++) {
        amdsmi_status_t r = amdsmi_get_processor_type(processor_handles[i], &processor_type);
        if (r != AMDSMI_STATUS_SUCCESS) return r;

        if(processor_type == AMDSMI_PROCESSOR_TYPE_AMD_CPU) {
            count_cpusockets++;
        } else if(processor_type == AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE) {
            count_cpucores++;
        } else if(processor_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
            count_gpus++;
        }
    }
    *nr_cpusockets = count_cpusockets;
    *nr_cpucores = count_cpucores;
    *nr_gpus = count_gpus;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handles_by_type(amdsmi_socket_handle socket_handle,
                                                     processor_type_t processor_type,
                                                     amdsmi_processor_handle* processor_handles,
                                                     uint32_t* processor_count) {
    AMDSMI_CHECK_INIT();
    if (processor_count == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Get the socket object via socket handle.
    amd::smi::AMDSmiSocket* socket = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance().handle_to_socket(socket_handle, &socket);
    if (r != AMDSMI_STATUS_SUCCESS) return r;
    std::vector<amd::smi::AMDSmiProcessor*>& processors = socket->get_processors(processor_type);
    uint32_t processor_size = static_cast<uint32_t>(processors.size());
    // Get the processor count only
    if (processor_handles == nullptr) {
        *processor_count = processor_size;
        return AMDSMI_STATUS_SUCCESS;
    }
    // If the processor_handles can hold all processors, return all of them.
    *processor_count = *processor_count >= processor_size ? processor_size : *processor_count;
    // Copy the processor handles
    for (uint32_t i = 0; i < *processor_count; i++) {
        processor_handles[i] = reinterpret_cast<amdsmi_processor_handle>(processors[i]);
    }

    return AMDSMI_STATUS_SUCCESS;
}

#endif

amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle,
              processor_type_t* processor_type) {

    AMDSMI_CHECK_INIT();

    if (processor_type == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    *processor_type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
    auto device = reinterpret_cast<Device *>(processor_handle);

    const auto& platform = Platform::instance();
    for (size_t i = 0, count = platform.GetDeviceCount(); i < count; ++i) {
        if (platform.GetDevice(i) == device) {
            *processor_type = AMDSMI_PROCESSOR_TYPE_AMD_GPU;
            break;
        }
    }
    return AMDSMI_STATUS_SUCCESS;
}


amdsmi_status_t
amdsmi_get_gpu_device_bdf(amdsmi_processor_handle processor_handle, amdsmi_bdf_t *bdf) {

    AMDSMI_CHECK_INIT();

    if (bdf == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    memset(bdf, 0, sizeof(*bdf));

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_NOT_FOUND;

    wsl::thunk::BdfInfo bi = {};
    auto code = device->QueryBdfInfo(&bi);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    bdf->domain_number   = bi.domain_number;
    bdf->bus_number      = bi.bus_number;
    bdf->device_number   = bi.device_number;
    bdf->function_number = bi.function_number;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_ainic_info(amdsmi_processor_handle processor_handle, amd::smi::AMDSmiAINICDevice::AINICInfo *info) {
  AMDSMI_CHECK_INIT();

  if (!info) {
    return AMDSMI_STATUS_INVAL;
  }

  amd::smi::AMDSmiAINICDevice *nic_device = nullptr;
  amdsmi_status_t r = get_ainic_device_from_handle(processor_handle, &nic_device);
  if (r != AMDSMI_STATUS_SUCCESS || !nic_device) return r;

  nic_device->amd_query_nic_info(*info);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_nic_asic_info(amdsmi_processor_handle processor_handle, amdsmi_nic_asic_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info = {};
    amdsmi_status_t status = amdsmi_get_ainic_info(processor_handle, &ainic_info);
    if(status != AMDSMI_STATUS_SUCCESS){
        return status;
    }
    *info = ainic_info.asic;
    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_nic_bus_info(amdsmi_processor_handle processor_handle, amdsmi_nic_bus_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info = {};
    amdsmi_status_t status = amdsmi_get_ainic_info(processor_handle, &ainic_info);
    if(status != AMDSMI_STATUS_SUCCESS){
        return status;
    }
    *info = ainic_info.bus;
    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_nic_driver_info(amdsmi_processor_handle processor_handle, amdsmi_nic_driver_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info = {};
    amdsmi_status_t status = amdsmi_get_ainic_info(processor_handle, &ainic_info);
    if(status != AMDSMI_STATUS_SUCCESS){
        return status;
    }
    *info = ainic_info.driver;
    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_nic_numa_info(amdsmi_processor_handle processor_handle, amdsmi_nic_numa_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info = {};
    amdsmi_status_t status = amdsmi_get_ainic_info(processor_handle, &ainic_info);
    if(status != AMDSMI_STATUS_SUCCESS){
        return status;
    }
    *info = ainic_info.numa;
    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_nic_port_info(amdsmi_processor_handle processor_handle, amdsmi_nic_port_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info = {};
    amdsmi_status_t status = amdsmi_get_ainic_info(processor_handle, &ainic_info);
    if(status != AMDSMI_STATUS_SUCCESS){
        return status;
    }
    *info = ainic_info.port;
    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_nic_rdma_dev_info(amdsmi_processor_handle processor_handle, amdsmi_nic_rdma_devices_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info = {};
    amdsmi_status_t status = amdsmi_get_ainic_info(processor_handle, &ainic_info);
    if(status != AMDSMI_STATUS_SUCCESS){
        return status;
    }
    *info = ainic_info.rdma_dev;
    return AMDSMI_STATUS_SUCCESS;
}

#ifdef BRCM_NIC
amdsmi_status_t amdsmi_get_nic_info(amdsmi_processor_handle processor_handle, amdsmi_brcm_nic_info_t *info) {
  AMDSMI_CHECK_INIT();

  if (info == NULL) {
    return AMDSMI_STATUS_INVAL;
  }

  amd::smi::AMDSmiNICDevice *nic_device = nullptr;
  amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
  if (r != AMDSMI_STATUS_SUCCESS) return r;

  nic_device->amd_query_nic_info(*info);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_nic_temp_info(amdsmi_processor_handle processor_handle,
                                         amdsmi_brcm_nic_temperature_metric_t *info) {
  AMDSMI_CHECK_INIT();

  if (info == NULL) {
    return AMDSMI_STATUS_INVAL;
  }

  amd::smi::AMDSmiNICDevice *nic_device = nullptr;
  amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
  if (r != AMDSMI_STATUS_SUCCESS) return r;

  nic_device->amd_query_nic_temp_info(*info);
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_nic_power_info(amdsmi_processor_handle processor_handle,
    amdsmi_brcm_nic_hwmon_power_t *info) {
        AMDSMI_CHECK_INIT();
    if (info == NULL) {
      return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiNICDevice *nic_device = nullptr;
    amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    nic_device->amd_query_nic_power_info(*info);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_nic_device_info(amdsmi_processor_handle processor_handle,
    amdsmi_brcm_nic_hwmon_device_t *info) {
    AMDSMI_CHECK_INIT();
    if (info == NULL) {
      return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiNICDevice *nic_device = nullptr;
    amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    nic_device->amd_query_nic_device_info(*info);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_nic_metrics_info(amdsmi_processor_handle processor_handle,
    amdsmi_brcm_nic_hwmon_metrics_t *metrics) {
    AMDSMI_CHECK_INIT();
    if (metrics == NULL) {
      return AMDSMI_STATUS_INVAL;
    }

    amdsmi_status_t ret;

    amd::smi::AMDSmiNICDevice *nic_device = nullptr;
    amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    // Fetch power metrics
    ret = nic_device->amd_query_nic_power_info(metrics->nic_power);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__
            << " | Failed to fetch NIC power metrics: " << ret;
        LOG_INFO(ss);
        return ret;
    }

    // Fetch temperature metrics
    ret = nic_device->amd_query_nic_temp_info(metrics->nic_temperature);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__
           << " | Failed to fetch NIC temperature metrics: " << ret;
        LOG_INFO(ss);
        return ret;
    }

    // Fetch the full device struct
    amdsmi_brcm_nic_hwmon_device_t full_device;
    ret = nic_device->amd_query_nic_device_info(full_device);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__
           << " | Failed to fetch NIC device metrics: " << ret;
        LOG_INFO(ss);
        return ret;
    }

    // Copy only the 3 required fields into metrics
    snprintf(metrics->nic_device_aer_dev_correctable, AMDSMI_MAX_STRING_LENGTH - 1, "%s", full_device.nic_device_aer_dev_correctable);
    snprintf(metrics->nic_device_aer_dev_fatal, AMDSMI_MAX_STRING_LENGTH - 1, "%s", full_device.nic_device_aer_dev_fatal);
    snprintf(metrics->nic_device_aer_dev_nonfatal, AMDSMI_MAX_STRING_LENGTH - 1, "%s", full_device.nic_device_aer_dev_nonfatal);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_switch_device_bdf(amdsmi_processor_handle processor_handle,
                                          amdsmi_bdf_t* bdf) {
  AMDSMI_CHECK_INIT();

  if (bdf == NULL) {
    return AMDSMI_STATUS_INVAL;
  }

  amd::smi::AMDSmiSWITCHDevice* switch_device = nullptr;
  amdsmi_status_t r = get_switch_device_from_handle(processor_handle, &switch_device);
  if (r != AMDSMI_STATUS_SUCCESS) return r;

  // get bdf from sysfs file
  *bdf = switch_device->get_bdf();
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_switch_link_info(amdsmi_processor_handle processor_handle,
    amdsmi_brcm_switch_link_metric_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiSWITCHDevice *switch_device = nullptr;
    amdsmi_status_t r = get_switch_device_from_handle(processor_handle, &switch_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    switch_device->amd_query_switch_link_info(*info);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_switch_power_info(amdsmi_processor_handle processor_handle,
    amdsmi_brcm_switch_power_metric_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiSWITCHDevice *switch_device = nullptr;
    amdsmi_status_t r = get_switch_device_from_handle(processor_handle, &switch_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    switch_device->amd_query_switch_power_info(*info);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_switch_device_info(amdsmi_processor_handle processor_handle,
    amdsmi_brcm_switch_device_metric_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    amdsmi_status_t ret;
    ret = amdsmi_get_switch_power_info(processor_handle, &(info->brcm_device_power));
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " amdsmi_get_switch_device_info - Failed to fetch power metrics";
        LOG_ERROR(ss);
        return ret;
    }

    amd::smi::AMDSmiSWITCHDevice *switch_device = nullptr;
    amdsmi_status_t r = get_switch_device_from_handle(processor_handle, &switch_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    switch_device->amd_query_switch_device_info(*info);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_switch_metrics_info(amdsmi_processor_handle processor_handle, amdsmi_brcm_switch_metric_t *info){
    AMDSMI_CHECK_INIT();

    if (info == NULL) {
        return AMDSMI_STATUS_INVAL;
    }
    amdsmi_status_t ret;
    ret = amdsmi_get_switch_power_info(processor_handle, &(info->brcm_power));
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " amdsmi_get_switch_metrics_info - Failed to fetch power metrics";
        LOG_ERROR(ss);
        return ret;
    }

    // Fetch the full device struct
    amdsmi_brcm_switch_device_metric_t full_device;
    ret = amdsmi_get_switch_device_info(processor_handle, &full_device);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " amdsmi_get_switch_metrics_info - Failed to fetch switch device.";
        LOG_ERROR(ss);
        return ret;
    }

    // Copy only the 3 required fields into metrics
    snprintf(info->brcm_device_aer_dev_correctable, AMDSMI_MAX_STRING_LENGTH - 1, "%s", full_device.brcm_device_aer_dev_correctable);
    snprintf(info->brcm_device_aer_dev_nonfatal, AMDSMI_MAX_STRING_LENGTH - 1, "%s", full_device.brcm_device_aer_dev_nonfatal);
    snprintf(info->brcm_device_aer_dev_fatal, AMDSMI_MAX_STRING_LENGTH - 1, "%s", full_device.brcm_device_aer_dev_fatal);

    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_nic_fw_info(amdsmi_processor_handle processor_handle, 
    amdsmi_brcm_nic_firmware_t *info) {
  AMDSMI_CHECK_INIT();
  if (info == NULL) {
    return AMDSMI_STATUS_INVAL;
  }
  amd::smi::AMDSmiNICDevice *nic_device = nullptr;
  amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
  if (r != AMDSMI_STATUS_SUCCESS) return r;
  nic_device->amd_query_nic_firmware_info(*info);
  return AMDSMI_STATUS_SUCCESS;
}
#endif//BRCM_NIC

amdsmi_status_t amdsmi_get_nic_rdma_port_statistics(
    amdsmi_processor_handle processor_handle, 
    uint32_t rdma_port_index,
    uint32_t *num_stats, 
    amdsmi_nic_stat_t *stats) { 

    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    std::ostringstream ss;
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiAINICDevice *nic_device = nullptr;
    status = get_ainic_device_from_handle(processor_handle, &nic_device);
    if (status != AMDSMI_STATUS_SUCCESS) {
        ss << __PRETTY_FUNCTION__ << smi_amdgpu_get_status_string(status, false);
        LOG_ERROR(ss);
        return status;
    }
    amd::smi::AMDSmiAINICDevice::AINICInfo nic_info = {};
    status = nic_device->amd_query_nic_info(nic_info);
    if (status != AMDSMI_STATUS_SUCCESS) {
        ss << __PRETTY_FUNCTION__ << " | Failed to query NIC info";
        LOG_ERROR(ss);
        return status;
    }
    if(nic_info.rdma_dev.num_rdma_dev < 1) {
        ss << __PRETTY_FUNCTION__ << " | No RDMA devices found";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    else if(rdma_port_index >= nic_info.rdma_dev.num_rdma_dev) {
        ss << __PRETTY_FUNCTION__ << " | NIC ports (" << rdma_port_index << ") is out of range (max ports:" << nic_info.rdma_dev.num_rdma_dev << ")";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    else if(nic_info.rdma_dev.rdma_dev_info[0].num_rdma_ports < 1) {
        ss << __PRETTY_FUNCTION__ << " | No RDMA ports found";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    else if(!num_stats) {
        ss << __PRETTY_FUNCTION__ << " | Invalid num_stats pointer";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }
    else if(!stats && *num_stats > 0) {
        ss << __PRETTY_FUNCTION__ << " | Invalid stats and num_stats pointers";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }

    std::string netdev(amd::smi::trim(nic_info.rdma_dev.rdma_dev_info[0].rdma_port_info[rdma_port_index].netdev));
    std::string rdmadev(nic_info.rdma_dev.rdma_dev_info[0].rdma_dev);
    int port_num = nic_info.rdma_dev.rdma_dev_info[0].rdma_port_info[rdma_port_index].rdma_port;

    std::string directory_path = "/sys/class/net/" + netdev + "/device/infiniband/" + rdmadev + "/subsystem/" + rdmadev + "/subsystem/" + rdmadev + "/ports/" + std::to_string(port_num) + "/hw_counters/";
    if(!std::filesystem::exists(directory_path)) {
        ss << __PRETTY_FUNCTION__ << " | Directory does not exist: " << directory_path;
        LOG_ERROR(ss);
        return AMDSMI_STATUS_FILE_ERROR;
    }
    
    uint32_t idx  = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
        if (std::filesystem::is_regular_file(entry.path())) {
            if(stats && num_stats && idx < *num_stats) {
                snprintf(stats[idx].name, sizeof(stats[idx].name), "%s", entry.path().filename().string().c_str());
                std::ifstream in(entry.path());
                if (!in.is_open()) {
                    ss << __PRETTY_FUNCTION__ << smi_amdgpu_get_status_string(status, false);
                    LOG_ERROR(ss);
                    return AMDSMI_STATUS_FILE_ERROR;
                }
                in  >> stats[idx].value;
            }
            ++idx;
        }
    }
    if(num_stats) {
        *num_stats = idx;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_device_uuid(amdsmi_processor_handle processor_handle,
                           unsigned int *uuid_length,
                           char *uuid) {
    AMDSMI_CHECK_INIT();

    if (uuid_length == nullptr || uuid == nullptr || *uuid_length < AMDSMI_GPU_UUID_SIZE) {
        return AMDSMI_STATUS_INVAL;
    }

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    wsl::thunk::AsicInfo asic_info = {};
    const uint8_t fcn = 0xff;

    auto code = device->QueryAsicInfo(&asic_info);
    if (code == ErrorCode::Success) {
        return amdsmi_uuid_gen(uuid,
                               asic_info.asic_serial,
                               (uint16_t)asic_info.device_id, fcn);
    }
    return translateCodeToSmiStatus(code);
}

// Add a static cache for KFD nodes with initialization flag
static std::once_flag kfd_nodes_initialized;
static std::map<uint64_t, std::shared_ptr<amd::smi::KFDNode>> cached_nodes;
static uint32_t cached_smallest_node_id = std::numeric_limits<uint32_t>::max();

amdsmi_status_t
amdsmi_get_gpu_enumeration_info(amdsmi_processor_handle processor_handle,
                                amdsmi_enumeration_info_t *info){

    // Ensure library initialization
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amdsmi_status_t status;
    std::ostringstream ss;

    // Retrieve GPU device from the processor handle
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    status = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    // Retrieve DRM Card ID
    info->drm_card = gpu_device->get_card_id();

    // Retrieve DRM Render ID
    info->drm_render = gpu_device->get_drm_render_minor();

    // Retrieve HIP ID (difference from the smallest node ID) and HSA ID
    // Initialize KFD nodes once
    std::call_once(kfd_nodes_initialized, []() {
        if (amd::smi::DiscoverKFDNodes(&cached_nodes) == 0) {
            for (const auto& node_pair : cached_nodes) {
                uint32_t node_id = 0;
                if (node_pair.second->get_node_id(&node_id) == 0) {
                    cached_smallest_node_id = std::min(cached_smallest_node_id, node_id);
                }
            }
        }
    });

    // Default to 0xffffffff as not supported
    info->hsa_id = std::numeric_limits<uint32_t>::max();
    info->hip_id = std::numeric_limits<uint32_t>::max();
    amdsmi_kfd_info_t kfd_info;
    status = amdsmi_get_gpu_kfd_info(processor_handle, &kfd_info);
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->hsa_id = kfd_info.node_id;
        info->hip_id = kfd_info.node_id - cached_smallest_node_id;
    }

    // Retrieve HIP UUID
    std::ostringstream ss_uuid;
    uint64_t device_uuid = 0;
    std::string hip_uuid_str;
    status = rsmi_wrapper(rsmi_dev_unique_id_get, processor_handle, 0, &device_uuid);
    ss_uuid << "GPU-" << std::hex << std::setw(16) << std::setfill('0') << device_uuid;
    hip_uuid_str = ss_uuid.str();
    smi_clear_char_and_reinitialize(info->hip_uuid, AMDSMI_MAX_STRING_LENGTH, hip_uuid_str);

    ss << "; device_uuid (dec): " << device_uuid << "\n"
       << "; device_uuid (hex): 0x" << std::hex << std::setw(16) << std::setfill('0') << device_uuid << std::dec << "\n"
       << "; rsmi_dev_unique_id_get() status: "
       << smi_amdgpu_get_status_string(status, false) << "\n";
    LOG_INFO(ss);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_board_info(amdsmi_processor_handle processor_handle, amdsmi_board_info_t *board_info) {
    AMDSMI_CHECK_INIT();

    if (board_info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    wsl::thunk::BoardInfo bi{};
    auto code = device->QueryBoardInfo(&bi);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    memset(board_info, 0, sizeof(*board_info));
    strncpy(board_info->product_name, bi.product_name,
            sizeof(board_info->product_name) - 1);
    strncpy(board_info->manufacturer_name, bi.manufacturer_name,
            sizeof(board_info->manufacturer_name) - 1);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_cache_info(
      amdsmi_processor_handle processor_handle, amdsmi_gpu_cache_info_t *info) {
    AMDSMI_CHECK_INIT();
    if (info == nullptr) return AMDSMI_STATUS_INVAL;

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;

    CacheInfo ci{};
    auto code = device->QueryCacheInfo(&ci);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);

    memset(info, 0, sizeof(*info));
    info->num_cache_types = ci.num_cache_types;
    for (uint32_t i = 0; i < ci.num_cache_types; ++i) {
        info->cache[i].cache_size       = ci.cache[i].cache_size_kb;
        info->cache[i].cache_level      = ci.cache[i].cache_level;
        info->cache[i].cache_properties = ci.cache[i].cache_properties;
        info->cache[i].max_num_cu_shared   = ci.cache[i].max_num_cu_shared;
        info->cache[i].num_cache_instance  = ci.cache[i].num_cache_instance;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t  amdsmi_get_temp_metric(amdsmi_processor_handle processor_handle,
                    amdsmi_temperature_type_t sensor_type,
                    amdsmi_temperature_metric_t metric, int64_t *temperature) {

    AMDSMI_CHECK_INIT();

    if (temperature == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto code = device->QueryTempMetric(static_cast<uint32_t>(sensor_type),
                                        static_cast<uint32_t>(metric),
                                        temperature);
    return translateCodeToSmiStatus(code);
}
amdsmi_status_t amdsmi_get_npm_info(amdsmi_node_handle node_handle,
                            amdsmi_npm_info_t *npm_info) {
    AMDSMI_CHECK_INIT();

    if (node_handle == nullptr || npm_info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Verify board path from node_handle
    auto board_path_str = reinterpret_cast<std::string*>(node_handle);
    if (board_path_str == nullptr || board_path_str->empty()) {
        return AMDSMI_STATUS_INVAL;
    }

    rsmi_npm_info_t rsmi_npm_info;
    rsmi_status_t rstatus = rsmi_dev_npm_info_get(0, reinterpret_cast<uintptr_t>(node_handle), &rsmi_npm_info);
    amdsmi_status_t amdsmi_status = amd::smi::rsmi_to_amdsmi_status(rstatus);
    if (amdsmi_status != AMDSMI_STATUS_SUCCESS) {
        return amdsmi_status;
    }

    if (sizeof(amdsmi_npm_info_t) != sizeof(rsmi_npm_info_t)) {
        return AMDSMI_STATUS_UNEXPECTED_SIZE;
    }
    std::memcpy(npm_info, &rsmi_npm_info, sizeof(amdsmi_npm_info_t));

    return AMDSMI_STATUS_SUCCESS;

}

amdsmi_status_t amdsmi_get_gpu_vram_usage(amdsmi_processor_handle processor_handle,
            amdsmi_vram_usage_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_NOT_FOUND;

    memset(info, 0, sizeof(*info));

    VramUsage usage{};
    auto code = device->QueryVramUsage(&usage);
    if (code == ErrorCode::Success) {
        info->vram_used  = static_cast<uint32_t>(usage.vram_used_mb);
        info->vram_total = static_cast<uint32_t>(usage.vram_total_mb);
    }
    return translateCodeToSmiStatus(code);
}

static void system_wait(int milli_seconds) {
  std::ostringstream ss;
  auto start = std::chrono::high_resolution_clock::now();
  // 1 ms = 1000 us
  int waitTime = milli_seconds * 1000;

  ss << __PRETTY_FUNCTION__ << " | "
     << "** Waiting for " << std::dec << waitTime
     << " us (" << waitTime/1000 << " seconds) **";
  LOG_DEBUG(ss);
  usleep(waitTime);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  ss << __PRETTY_FUNCTION__ << " | "
     << "** Waiting took " << duration.count() / 1000
     << " milli-seconds **";
  LOG_DEBUG(ss);
}

amdsmi_status_t amdsmi_get_violation_status(amdsmi_processor_handle processor_handle,
            amdsmi_violation_status_t *violation_status) {
    AMDSMI_CHECK_INIT();

    std::ostringstream ss;
    if (violation_status == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // 1 sec = 1000 ms = 1000000 us
    // 0.1 sec = 100 ms = 100000 us
    constexpr uint64_t kFASTEST_POLL_TIME_MS = 100;  // fastest SMU FW sample time is 100 ms

    violation_status->reference_timestamp = std::numeric_limits<uint64_t>::max();
    violation_status->violation_timestamp = std::numeric_limits<uint64_t>::max();

    violation_status->acc_counter = std::numeric_limits<uint64_t>::max();
    violation_status->acc_prochot_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->acc_ppt_pwr = std::numeric_limits<uint64_t>::max();
    violation_status->acc_socket_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->acc_vr_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->acc_hbm_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->acc_gfx_clk_below_host_limit = std::numeric_limits<uint64_t>::max();

    violation_status->per_prochot_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->per_ppt_pwr = std::numeric_limits<uint64_t>::max();
    violation_status->per_socket_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->per_vr_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->per_hbm_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->per_gfx_clk_below_host_limit = std::numeric_limits<uint64_t>::max();

    violation_status->active_prochot_thrm = std::numeric_limits<uint8_t>::max();
    violation_status->active_ppt_pwr = std::numeric_limits<uint8_t>::max();
    violation_status->active_socket_thrm = std::numeric_limits<uint8_t>::max();
    violation_status->active_vr_thrm = std::numeric_limits<uint8_t>::max();
    violation_status->active_hbm_thrm = std::numeric_limits<uint8_t>::max();
    violation_status->active_gfx_clk_below_host_limit = std::numeric_limits<uint8_t>::max();

    fill_2d_array(violation_status->acc_gfx_clk_below_host_limit_pwr,
        std::numeric_limits<uint64_t>::max());
    fill_2d_array(violation_status->acc_gfx_clk_below_host_limit_thm,
        std::numeric_limits<uint64_t>::max());
    fill_2d_array(violation_status->acc_low_utilization,
        std::numeric_limits<uint64_t>::max());
    fill_2d_array(violation_status->acc_gfx_clk_below_host_limit_total,
        std::numeric_limits<uint64_t>::max());

    fill_2d_array(violation_status->per_gfx_clk_below_host_limit_pwr,
        std::numeric_limits<uint64_t>::max());
    fill_2d_array(violation_status->per_gfx_clk_below_host_limit_thm,
        std::numeric_limits<uint64_t>::max());
    fill_2d_array(violation_status->per_low_utilization,
        std::numeric_limits<uint64_t>::max());
    fill_2d_array(violation_status->per_gfx_clk_below_host_limit_total,
        std::numeric_limits<uint64_t>::max());

    fill_2d_array(violation_status->active_gfx_clk_below_host_limit_pwr,
        std::numeric_limits<uint8_t>::max());
    fill_2d_array(violation_status->active_gfx_clk_below_host_limit_thm,
        std::numeric_limits<uint8_t>::max());
    fill_2d_array(violation_status->active_low_utilization,
        std::numeric_limits<uint8_t>::max());
    fill_2d_array(violation_status->active_gfx_clk_below_host_limit_total,
        std::numeric_limits<uint8_t>::max());

    const auto p1 = std::chrono::system_clock::now();
    auto current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                                p1.time_since_epoch()).count();
    violation_status->reference_timestamp = current_time;

    amd::smi::AMDSmiProcessor* device = nullptr;
    amdsmi_status_t ret = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_processor(processor_handle, &device);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        return ret;
    }

    if (device->get_processor_type() != AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS) {
        return r;
    }

    // default to 0xffffffff as not supported
    uint32_t partition_id = std::numeric_limits<uint32_t>::max();
    auto tmp_partition_id = uint32_t(0);
    amdsmi_status_t status = rsmi_wrapper(rsmi_dev_partition_id_get, processor_handle, 0,
                                          &(tmp_partition_id));
    // Do not return early if this value fails
    // continue to try getting all info
    if (status == AMDSMI_STATUS_SUCCESS) {
        partition_id = tmp_partition_id;
    }

    amdsmi_gpu_metrics_t metric_info_a = {};
    status =  amdsmi_get_gpu_metrics_info(
                    processor_handle, &metric_info_a);
    if (status != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | amdsmi_get_gpu_metrics_info failed with status = "
        << smi_amdgpu_get_status_string(status, false);
        LOG_ERROR(ss);
        return status;
    }

    // Note: Both XCP and partition_id will default to 0, if gpu_metrics file is not present.
    //       This is why we can check elements in kFIRST_ELEMENT == 0 for both XCP and partition_id.
    const uint32_t kFIRST_ELEMENT = 0;

    // Check if violation status is supported:
    // If all of these values are "undefined" then the feature is not supported on the ASIC
    if (metric_info_a.accumulation_counter == std::numeric_limits<uint64_t>::max()
        && metric_info_a.prochot_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.ppt_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.socket_thm_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.vr_thm_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.hbm_thm_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.xcp_stats[kFIRST_ELEMENT].gfx_below_host_limit_acc[kFIRST_ELEMENT]
        == std::numeric_limits<uint64_t>::max()
        && metric_info_a.xcp_stats[kFIRST_ELEMENT].gfx_below_host_limit_ppt_acc[kFIRST_ELEMENT]
        == std::numeric_limits<uint64_t>::max()
        && metric_info_a.xcp_stats[kFIRST_ELEMENT].gfx_below_host_limit_thm_acc[kFIRST_ELEMENT]
        == std::numeric_limits<uint64_t>::max()
        && metric_info_a.xcp_stats[kFIRST_ELEMENT].gfx_low_utilization_acc[kFIRST_ELEMENT]
        == std::numeric_limits<uint64_t>::max()
        && metric_info_a.xcp_stats[kFIRST_ELEMENT].gfx_below_host_limit_total_acc[kFIRST_ELEMENT]
        == std::numeric_limits<uint64_t>::max()) {
        ss << __PRETTY_FUNCTION__
           << " | ASIC does not support throttle violations!, "
           << "returning AMDSMI_STATUS_NOT_SUPPORTED";
        LOG_INFO(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    // wait 100ms before reading again
    system_wait(static_cast<int>(kFASTEST_POLL_TIME_MS));

    amdsmi_gpu_metrics_t metric_info_b = {};
    status =  amdsmi_get_gpu_metrics_info(
            processor_handle, &metric_info_b);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    // Insert current accumulator counters into struct
    violation_status->violation_timestamp = metric_info_b.firmware_timestamp;
    violation_status->acc_counter = metric_info_b.accumulation_counter;
    violation_status->acc_prochot_thrm = metric_info_b.prochot_residency_acc;
    violation_status->acc_ppt_pwr = metric_info_b.ppt_residency_acc;
    violation_status->acc_socket_thrm = metric_info_b.socket_thm_residency_acc;
    violation_status->acc_vr_thrm = metric_info_b.vr_thm_residency_acc;
    violation_status->acc_hbm_thrm = metric_info_b.hbm_thm_residency_acc;
    violation_status->acc_gfx_clk_below_host_limit  // deprecated
        = metric_info_b.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT];

    // Copy XCP accumulators into 2D array
    auto copy_xcp_metric = [](const auto& src, auto& dst, auto member_ptr) {
        for (size_t i = 0; i < AMDSMI_MAX_NUM_XCP; ++i) {
            std::copy(
                std::begin(src[i].*member_ptr),
                std::end(src[i].*member_ptr),
                dst[i]);
        }
    };
    copy_xcp_metric(metric_info_b.xcp_stats, violation_status->acc_gfx_clk_below_host_limit_pwr,
                    &amdsmi_gpu_xcp_metrics_t::gfx_below_host_limit_ppt_acc);
    copy_xcp_metric(metric_info_b.xcp_stats, violation_status->acc_gfx_clk_below_host_limit_thm,
                    &amdsmi_gpu_xcp_metrics_t::gfx_below_host_limit_thm_acc);
    copy_xcp_metric(metric_info_b.xcp_stats, violation_status->acc_low_utilization,
                    &amdsmi_gpu_xcp_metrics_t::gfx_low_utilization_acc);
    copy_xcp_metric(metric_info_b.xcp_stats, violation_status->acc_gfx_clk_below_host_limit_total,
                    &amdsmi_gpu_xcp_metrics_t::gfx_below_host_limit_total_acc);

    ss << __PRETTY_FUNCTION__ << " | "
       << "[gpu_metrics A] metric_info_a.accumulation_counter: " << std::dec
       << metric_info_a.accumulation_counter << "\n"
       << "; metric_info_a.prochot_residency_acc: " << std::dec
       << metric_info_a.prochot_residency_acc << "\n"
       << "; metric_info_a.ppt_residency_acc (pviol): " << std::dec
       << metric_info_a.ppt_residency_acc << "\n"
       << "; metric_info_a.socket_thm_residency_acc (tviol): " << std::dec
       << metric_info_a.socket_thm_residency_acc << "\n"
       << "; metric_info_a.vr_thm_residency_acc: " << std::dec
       << metric_info_a.vr_thm_residency_acc << "\n"
       << "; metric_info_a.hbm_thm_residency_acc: " << std::dec
       << metric_info_a.hbm_thm_residency_acc << "\n"
       << "; metric_info_a.xcp_stats[" << partition_id << "].gfx_below_host_limit_acc["
       << kFIRST_ELEMENT << "]: " << std::dec  // deprecated
       << metric_info_a.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT] << "\n"
       << " [gpu_metrics B] metric_info_b.accumulation_counter: " << std::dec
       << metric_info_b.accumulation_counter << "\n"
       << "; metric_info_b.prochot_residency_acc: " << std::dec
       << metric_info_b.prochot_residency_acc << "\n"
       << "; metric_info_b.ppt_residency_acc (pviol): " << std::dec
       << metric_info_b.ppt_residency_acc << "\n"
       << "; metric_info_b.socket_thm_residency_acc (tviol): " << std::dec
       << metric_info_b.socket_thm_residency_acc << "\n"
       << "; metric_info_b.vr_thm_residency_acc: " << std::dec
       << metric_info_b.vr_thm_residency_acc << "\n"
       << "; metric_info_b.hbm_thm_residency_acc: " << std::dec
       << metric_info_b.hbm_thm_residency_acc << "\n"
       << "; metric_info_b.xcp_stats[" << partition_id << "].gfx_below_host_limit_acc["
       << kFIRST_ELEMENT << "]: " << std::dec  // deprecated
       << metric_info_b.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT] << "\n";
    LOG_DEBUG(ss);

    if ( (metric_info_b.prochot_residency_acc != std::numeric_limits<uint64_t>::max()
        || metric_info_a.prochot_residency_acc != std::numeric_limits<uint64_t>::max())
        && (metric_info_b.prochot_residency_acc >= metric_info_a.prochot_residency_acc)
        && ((metric_info_b.accumulation_counter - metric_info_a.accumulation_counter) > 0)) {
        violation_status->per_prochot_thrm =
            (((metric_info_b.prochot_residency_acc - metric_info_a.prochot_residency_acc) * 100) /
            (metric_info_b.accumulation_counter - metric_info_a.accumulation_counter));

        if (violation_status->per_prochot_thrm > 0) {
            violation_status->active_prochot_thrm = 1;
        } else {
            violation_status->active_prochot_thrm = 0;
        }
        ss << __PRETTY_FUNCTION__ << " | "
           << "ENTERED prochot_residency_acc | per_prochot_thrm: " << std::dec
           << violation_status->per_prochot_thrm
           << "%; active_prochot_thrm = " << std::dec
           << violation_status->active_prochot_thrm << "\n";
        LOG_DEBUG(ss);
    }
    if ( (metric_info_b.ppt_residency_acc != std::numeric_limits<uint64_t>::max()
        || metric_info_a.ppt_residency_acc != std::numeric_limits<uint64_t>::max())
        && (metric_info_b.ppt_residency_acc >= metric_info_a.ppt_residency_acc)
        && ((metric_info_b.accumulation_counter - metric_info_a.accumulation_counter) > 0)) {
        violation_status->per_ppt_pwr =
            (((metric_info_b.ppt_residency_acc - metric_info_a.ppt_residency_acc) * 100) /
            (metric_info_b.accumulation_counter - metric_info_a.accumulation_counter));

        if (violation_status->per_ppt_pwr > 0) {
            violation_status->active_ppt_pwr = 1;
        } else {
            violation_status->active_ppt_pwr = 0;
        }
        ss << __PRETTY_FUNCTION__ << " | "
           << "ENTERED ppt_residency_acc | per_ppt_pwr: " << std::dec
           << violation_status->per_ppt_pwr
           << "%; active_ppt_pwr = " << std::dec
           << violation_status->active_ppt_pwr << "\n";
        LOG_DEBUG(ss);
    }
    if ( (metric_info_b.socket_thm_residency_acc != std::numeric_limits<uint64_t>::max()
        || metric_info_a.socket_thm_residency_acc != std::numeric_limits<uint64_t>::max())
        && (metric_info_b.socket_thm_residency_acc >= metric_info_a.socket_thm_residency_acc)
        && ((metric_info_b.accumulation_counter - metric_info_a.accumulation_counter) > 0)) {
        violation_status->per_socket_thrm =
            (((metric_info_b.socket_thm_residency_acc -
                metric_info_a.socket_thm_residency_acc) * 100) /
            (metric_info_b.accumulation_counter - metric_info_a.accumulation_counter));

        if (violation_status->per_socket_thrm > 0) {
            violation_status->active_socket_thrm = 1;
        } else {
            violation_status->active_socket_thrm = 0;
        }
        ss << __PRETTY_FUNCTION__ << " | "
           << "ENTERED socket_thm_residency_acc | per_socket_thrm: " << std::dec
           << violation_status->per_socket_thrm
           << "%; active_socket_thrm = " << std::dec
           << violation_status->active_socket_thrm << "\n";
        LOG_DEBUG(ss);
    }
    if ( (metric_info_b.vr_thm_residency_acc != std::numeric_limits<uint64_t>::max()
        || metric_info_a.vr_thm_residency_acc != std::numeric_limits<uint64_t>::max())
        && (metric_info_b.vr_thm_residency_acc >= metric_info_a.vr_thm_residency_acc)
        && ((metric_info_b.accumulation_counter - metric_info_a.accumulation_counter) > 0)) {
        violation_status->per_vr_thrm =
            (((metric_info_b.vr_thm_residency_acc -
                metric_info_a.vr_thm_residency_acc) * 100) /
            (metric_info_b.accumulation_counter - metric_info_a.accumulation_counter));

        if (violation_status->per_vr_thrm > 0) {
            violation_status->active_vr_thrm = 1;
        } else {
            violation_status->active_vr_thrm = 0;
        }
        ss << __PRETTY_FUNCTION__ << " | "
           << "ENTERED vr_thm_residency_acc | per_vr_thrm: " << std::dec
           << violation_status->per_vr_thrm
           << "%; active_ppt_pwr = " << std::dec
           << violation_status->active_vr_thrm << "\n";
        LOG_DEBUG(ss);
    }
    if ( (metric_info_b.hbm_thm_residency_acc != std::numeric_limits<uint64_t>::max()
        || metric_info_a.hbm_thm_residency_acc != std::numeric_limits<uint64_t>::max())
        && (metric_info_b.hbm_thm_residency_acc >= metric_info_a.hbm_thm_residency_acc)
        && ((metric_info_b.accumulation_counter - metric_info_a.accumulation_counter) > 0) ) {
        violation_status->per_hbm_thrm =
            (((metric_info_b.hbm_thm_residency_acc -
                metric_info_a.hbm_thm_residency_acc) * 100) /
            (metric_info_b.accumulation_counter - metric_info_a.accumulation_counter));

        if (violation_status->per_hbm_thrm > 0) {
            violation_status->active_hbm_thrm = 1;
        } else {
            violation_status->active_hbm_thrm = 0;
        }
        ss << __PRETTY_FUNCTION__ << " | "
           << "ENTERED hbm_thm_residency_acc | per_hbm_thrm: " << std::dec
           << violation_status->per_hbm_thrm
           << "%; active_ppt_pwr = " << std::dec
           << violation_status->active_hbm_thrm << "\n";
        LOG_DEBUG(ss);
    }
    // deprecated - design likely needs to include both [XCP][XCC], like the new metrics
    if ((metric_info_b.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT]
        != std::numeric_limits<uint64_t>::max() ||
         metric_info_a.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT]
         != std::numeric_limits<uint64_t>::max()) &&
        (metric_info_b.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT]
            >= metric_info_a.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT]) &&
        ((metric_info_b.accumulation_counter - metric_info_a.accumulation_counter) > 0)) {
        violation_status->per_gfx_clk_below_host_limit =
            (((metric_info_b.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT] -
                metric_info_a.xcp_stats[partition_id].gfx_below_host_limit_acc[kFIRST_ELEMENT])
                * 100) /
                (metric_info_b.accumulation_counter - metric_info_a.accumulation_counter));

        if (violation_status->per_gfx_clk_below_host_limit > 0) {
            violation_status->active_gfx_clk_below_host_limit = 1;
        } else {
            violation_status->active_gfx_clk_below_host_limit = 0;
        }
        ss << __PRETTY_FUNCTION__ << " | "
           << "ENTERED gfx_below_host_limit_acc | per_gfx_clk_below_host_limit: " << std::dec
           << violation_status->per_gfx_clk_below_host_limit
           << "%; active_ppt_pwr = " << std::boolalpha
           << violation_status->active_gfx_clk_below_host_limit << "\n";
        LOG_DEBUG(ss);
    }

    // one-shot processing of all XCP violation metrics
    // using a lambda function to avoid code duplication
    using MetricArrayType = uint64_t[AMDSMI_MAX_NUM_XCC];
    using MetricMemberPtr = MetricArrayType amdsmi_gpu_xcp_metrics_t::*;

    auto process_all_XCP_violation_metrics = [&](
        const std::vector<std::pair<std::string, MetricMemberPtr>>& metric_members,
        std::vector<std::reference_wrapper<
            uint64_t[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC]>> per_arrays,
        std::vector<std::reference_wrapper<
            uint8_t[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC]>> active_arrays) {
        uint64_t counter_delta = static_cast<uint64_t>(metric_info_b.accumulation_counter)
                            - static_cast<uint64_t>(metric_info_a.accumulation_counter);

        ss << __PRETTY_FUNCTION__ << " | Processing all XCP metrics with counter_delta: "
           << std::dec << counter_delta << "\n";
        LOG_DEBUG(ss);

        for (size_t metric_idx = 0; metric_idx < metric_members.size(); ++metric_idx) {
            const auto& member_pair = metric_members[metric_idx];
            const std::string& member_name = member_pair.first;
            MetricMemberPtr member_ptr = member_pair.second;

            auto& per_arr = per_arrays[metric_idx].get();
            auto& active_arr = active_arrays[metric_idx].get();

            ss << "  [Metric] " << member_name << "\n";
            for (uint32_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp) {
                const MetricArrayType& arr_a = metric_info_a.xcp_stats[xcp].*member_ptr;
                const MetricArrayType& arr_b = metric_info_b.xcp_stats[xcp].*member_ptr;
                ss << "    xcp: " << xcp << " (";
                for (uint32_t xcc = 0; xcc < AMDSMI_MAX_NUM_XCC; ++xcc) {
                    uint64_t val_a = arr_a[xcc];
                    uint64_t val_b = arr_b[xcc];

                    if (val_b == std::numeric_limits<uint64_t>::max() ||
                        val_a == std::numeric_limits<uint64_t>::max() ||
                        counter_delta <= 0 ||
                        val_b < val_a) {
                        per_arr[xcp][xcc] = std::numeric_limits<uint64_t>::max();
                        active_arr[xcp][xcc] = std::numeric_limits<uint8_t>::max();
                        ss << "[Invalid] (" << std::dec << per_arr[xcp][xcc]
                           << ", " << static_cast<int>(active_arr[xcp][xcc]) << ") ";
                        continue;
                    }

                    uint64_t percent = ((val_b - val_a) * 100) / counter_delta;
                    per_arr[xcp][xcc] = percent;
                    active_arr[xcp][xcc] = (percent > 0) ? 1 : 0;
                    ss << "[Valid] (" << std::dec << percent << "%, "
                       << std::boolalpha << static_cast<bool>(active_arr[xcp][xcc])
                       << ") | val_b: " << std::dec << val_b
                       << ", val_a: " << std::dec << val_a
                       << ", counter_delta: " << std::dec << counter_delta << " ";
                }
                ss << ")\n";
            }
        }
        LOG_DEBUG(ss);
    };

    // Prepare metric members and arrays for processing
    const std::vector<std::pair<std::string, MetricMemberPtr>> metric_members = {
        {"gfx_below_host_limit_ppt_acc", &amdsmi_gpu_xcp_metrics_t::gfx_below_host_limit_ppt_acc},
        {"gfx_below_host_limit_thm_acc", &amdsmi_gpu_xcp_metrics_t::gfx_below_host_limit_thm_acc},
        {"gfx_low_utilization_acc", &amdsmi_gpu_xcp_metrics_t::gfx_low_utilization_acc},
        {"gfx_below_host_limit_total_acc",
            &amdsmi_gpu_xcp_metrics_t::gfx_below_host_limit_total_acc}
    };

    process_all_XCP_violation_metrics(
        metric_members,
        {
            std::ref(violation_status->per_gfx_clk_below_host_limit_pwr),
            std::ref(violation_status->per_gfx_clk_below_host_limit_thm),
            std::ref(violation_status->per_low_utilization),
            std::ref(violation_status->per_gfx_clk_below_host_limit_total)
        },
        {
            std::ref(violation_status->active_gfx_clk_below_host_limit_pwr),
            std::ref(violation_status->active_gfx_clk_below_host_limit_thm),
            std::ref(violation_status->active_low_utilization),
            std::ref(violation_status->active_gfx_clk_below_host_limit_total)
        });

    ss << __PRETTY_FUNCTION__ << " | "
       << "RETURNING AMDSMI_STATUS_SUCCESS | "
       << "violation_status->reference_timestamp (time since epoch): " << std::dec
       << violation_status->reference_timestamp
       << "; violation_status->violation_timestamp (ms): " << std::dec
       << violation_status->violation_timestamp
       << "; violation_status->per_prochot_thrm (%): " << std::dec
       << violation_status->per_prochot_thrm
       << "; violation_status->per_ppt_pwr (%): " << std::dec
       << violation_status->per_ppt_pwr
       << "; violation_status->per_socket_thrm (%): " << std::dec
       << violation_status->per_socket_thrm
       << "; violation_status->per_vr_thrm (%): " << std::dec
       << violation_status->per_vr_thrm
       << "; violation_status->per_hbm_thrm (%): " << std::dec
       << violation_status->per_hbm_thrm
       << "; violation_status->per_gfx_clk_below_host_limit (%): " << std::dec  // deprecated
       << violation_status->per_gfx_clk_below_host_limit
       << "; violation_status->active_prochot_thrm (bool): " << std::boolalpha
       << static_cast<int>(violation_status->active_prochot_thrm)
       << "; violation_status->active_ppt_pwr (bool): " << std::boolalpha
       << static_cast<int>(violation_status->active_ppt_pwr)
       << "; violation_status->active_socket_thrm (bool): " << std::boolalpha
       << static_cast<int>(violation_status->active_socket_thrm)
       << "; violation_status->active_vr_thrm (bool): " << std::boolalpha
       << static_cast<int>(violation_status->active_vr_thrm)
       << "; violation_status->active_hbm_thrm (bool): " << std::boolalpha
       << static_cast<int>(violation_status->active_hbm_thrm)
       << "; violation_status->active_gfx_clk_below_host_limit (bool): "  // deprecated
       << std::boolalpha << static_cast<int>(violation_status->active_gfx_clk_below_host_limit)
       << "\n";
    LOG_INFO(ss);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_fan_rpms(amdsmi_processor_handle processor_handle,
                            uint32_t sensor_ind, int64_t *speed) {
    if (speed == nullptr) return AMDSMI_STATUS_INVAL;
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    GpuMetricsInfo m{};
    auto code = device->QueryGpuMetricsInfo(&m);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    if (m.current_fan_speed == UINT32_MAX) return AMDSMI_STATUS_NOT_SUPPORTED;
    *speed = static_cast<int64_t>(m.current_fan_speed);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_fan_speed(amdsmi_processor_handle processor_handle,
                                        uint32_t sensor_ind, int64_t *speed) {
    if (speed == nullptr) return AMDSMI_STATUS_INVAL;
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    GpuMetricsInfo m{};
    auto code = device->QueryGpuMetricsInfo(&m);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    if (m.current_fan_speed_percent == UINT32_MAX) return AMDSMI_STATUS_NOT_SUPPORTED;
    // amdsmi fan speed is expressed as a PWM value 0-255 (percentage * 255 / 100)
    *speed = static_cast<int64_t>(m.current_fan_speed_percent) * 255 / 100;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_fan_speed_max(amdsmi_processor_handle processor_handle,
                                    uint32_t sensor_ind, uint64_t *max_speed) {
    if (max_speed == nullptr) return AMDSMI_STATUS_INVAL;
    // Fan speed max is always 255 (full PWM range) when using percentage-based sensors
    *max_speed = 255;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_reset_gpu_fan(amdsmi_processor_handle processor_handle,
                                    uint32_t sensor_ind) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_gpu_fan_speed(amdsmi_processor_handle processor_handle,
                                uint32_t sensor_ind, uint64_t speed) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_id(amdsmi_processor_handle processor_handle,
                                uint16_t *id) {
    AMDSMI_CHECK_INIT();

    if (id == nullptr || processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto device = reinterpret_cast<Device *>(processor_handle);
    wsl::thunk::AsicInfo info{};
    auto code = device->QueryAsicInfo(&info);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);
    *id = static_cast<uint16_t>(info.device_id);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_revision(amdsmi_processor_handle processor_handle,
                                uint16_t *revision) {
    AMDSMI_CHECK_INIT();

    if (revision == nullptr || processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto device = reinterpret_cast<Device *>(processor_handle);
    wsl::thunk::AsicInfo info{};
    auto code = device->QueryAsicInfo(&info);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);
    *revision = static_cast<uint16_t>(info.rev_id);
    return AMDSMI_STATUS_SUCCESS;
}

// TODO(bliu) : add fw info from libdrm
amdsmi_status_t amdsmi_get_fw_info(amdsmi_processor_handle processor_handle,
        amdsmi_fw_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;
    memset(info, 0, sizeof(amdsmi_fw_info_t));

    for (int i = 0; i < AMDSMI_FW_ID__MAX; i++)
        info->fw_info_list[i].fw_id = static_cast<amdsmi_fw_block_t>(AMDSMI_FW_ID__MAX);

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    FwInfo fw{};
    auto code = device->QueryFwInfo(&fw);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    info->num_fw_info = fw.num_fw_info;
    for (uint32_t i = 0; i < fw.num_fw_info; i++) {
        info->fw_info_list[i].fw_id      = static_cast<amdsmi_fw_block_t>(fw.entries[i].fw_id);
        info->fw_info_list[i].fw_version = fw.entries[i].fw_version;
    }
    return AMDSMI_STATUS_SUCCESS;
}

// If similar caches are implemented in the future, make this generic and move it
namespace {
    struct AsicInfoCache {
        amdsmi_asic_info_t info{};
        std::chrono::steady_clock::time_point last_read;
        bool valid = false;
        std::mutex mtx;
    };

    std::unordered_map<std::string, AsicInfoCache> g_asic_info_cache_map;
    std::mutex g_asic_info_cache_map_mu;
    static const std::chrono::milliseconds kAsicInfoCacheDuration(
        read_env_ms("AMDSMI_ASIC_INFO_CACHE_MS", 10000)
    );
}

amdsmi_status_t
amdsmi_get_gpu_asic_info(amdsmi_processor_handle processor_handle, amdsmi_asic_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    wsl::thunk::AsicInfo ai = {};
    auto code = device->QueryAsicInfo(&ai);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    info->device_id    = ai.device_id;
    info->vendor_id    = ai.vendor_id;
    info->subvendor_id = ai.subvendor_id;
    info->subsystem_id = ai.subsystem_id;
    info->rev_id       = ai.rev_id;
    info->num_of_compute_units    = ai.num_of_compute_units;
    info->target_graphics_version = ai.target_graphics_version;
    snprintf(info->asic_serial, sizeof(info->asic_serial),
             "%016llx", (unsigned long long)ai.asic_serial);
    strncpy(info->market_name, ai.market_name, sizeof(info->market_name) - 1);
    info->vendor_name[0] = '\0';
    if (info->vendor_id == 0x1002)
        strncpy(info->vendor_name, "Advanced Micro Devices Inc. [AMD/ATI]",
                sizeof(info->vendor_name) - 1);
    info->oam_id = 0;
    info->flags  = 0;
    return AMDSMI_STATUS_SUCCESS;
}


amdsmi_status_t
amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle processor_handle,
                                amdsmi_xgmi_link_status_t *link_status) {
    AMDSMI_CHECK_INIT();

    if (link_status == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amdsmi_gpu_metrics_t metric_info = {};
    amdsmi_status_t status =  amdsmi_get_gpu_metrics_info(
            processor_handle, &metric_info);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    uint32_t socket_count = 0;
    status = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    // Total number of XGMI links cannot exceed AMDSMI_MAX_NUM_XGMI_LINKS
    link_status->total_links = socket_count <= AMDSMI_MAX_NUM_XGMI_LINKS ?
                                socket_count : AMDSMI_MAX_NUM_XGMI_LINKS;
    // get the status values from the metric info
    // if all links are disabled, return AMDSMI_STATUS_NOT_SUPPORTED
    uint32_t disabled_link_count = 0;
    for (unsigned int i = 0; i < link_status->total_links; i++) {
        if (metric_info.xgmi_link_status[i] == std::numeric_limits<uint16_t>::max()) {
            link_status->status[i] = AMDSMI_XGMI_LINK_DISABLE;
            disabled_link_count++;
        } else if (metric_info.xgmi_link_status[i] == 0) {
            link_status->status[i] = AMDSMI_XGMI_LINK_DOWN;
        } else if (metric_info.xgmi_link_status[i] == 1) {
            link_status->status[i] = AMDSMI_XGMI_LINK_UP;
        } else {
            return AMDSMI_STATUS_UNEXPECTED_DATA;
        }
    }
    if (disabled_link_count == link_status->total_links) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_kfd_info(amdsmi_processor_handle processor_handle,
                                    amdsmi_kfd_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amdsmi_status_t status;
    // default to 0xffffffffffffffff as not supported
    info->kfd_id = std::numeric_limits<uint64_t>::max();
    auto tmp_kfd_id = uint64_t(0);
    status = rsmi_wrapper(rsmi_dev_guid_get, processor_handle, 0,
                          &(tmp_kfd_id));
    // Do not return early if this value fails
    // continue to try getting all info
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->kfd_id = tmp_kfd_id;
    }

    // default to 0xffffffff as not supported
    info->node_id = std::numeric_limits<uint32_t>::max();
    auto tmp_node_id = uint32_t(0);
    status = rsmi_wrapper(rsmi_dev_node_id_get, processor_handle, 0,
                          &(tmp_node_id));
    // Do not return early if this value fails
    // continue to try getting all info
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->node_id = tmp_node_id;
    }

    // default to 0xffffffff as not supported
    info->current_partition_id = std::numeric_limits<uint32_t>::max();
    auto tmp_current_partition_id = uint32_t(0);
    status = rsmi_wrapper(rsmi_dev_partition_id_get, processor_handle, 0,
                          &(tmp_current_partition_id));
    // Do not return early if this value fails
    // continue to try getting all info
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->current_partition_id = tmp_current_partition_id;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_subsystem_id(amdsmi_processor_handle processor_handle,
                                uint16_t *id) {
    AMDSMI_CHECK_INIT();

    if (id == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    wsl::thunk::AsicInfo info{};
    auto code = device->QueryAsicInfo(&info);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    *id = static_cast<uint16_t>(info.subsystem_id);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_subsystem_name(
                                amdsmi_processor_handle processor_handle,
                                char *name, size_t len) {
    return rsmi_wrapper(rsmi_dev_subsystem_name_get, processor_handle, 0,
                        name, len);
}

amdsmi_status_t amdsmi_get_gpu_vendor_name(
            amdsmi_processor_handle processor_handle, char *name, size_t len) {
    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || name == nullptr)
        return AMDSMI_STATUS_INVAL;
    strncpy(name, "Advanced Micro Devices, Inc. [AMD/ATI]", len);
    return AMDSMI_STATUS_SUCCESS;
}


amdsmi_status_t amdsmi_get_gpu_vram_vendor(amdsmi_processor_handle processor_handle,
                                     char *brand, uint32_t len) {
    return rsmi_wrapper(rsmi_dev_vram_vendor_get, processor_handle, 0,
                        brand, len);
}

amdsmi_status_t amdsmi_get_gpu_vram_info(
    amdsmi_processor_handle processor_handle, amdsmi_vram_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    memset(info, 0, sizeof(*info));

    VramInfo raw{};
    auto code = device->QueryVramInfo(&raw);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    // map VIDEO_MEMORY_TYPE -> amdsmi_vram_type_t
    switch (raw.vram_type) {
    case 1:  info->vram_type = AMDSMI_VRAM_TYPE_DDR2;    break;  // DDR2
    case 2:  info->vram_type = AMDSMI_VRAM_TYPE_GDDR5;   break;  // GDDR5
    case 3:  info->vram_type = AMDSMI_VRAM_TYPE_DDR3;    break;  // DDR3
    case 4:  info->vram_type = AMDSMI_VRAM_TYPE_DDR4;    break;  // DDR4
    case 5:  info->vram_type = AMDSMI_VRAM_TYPE_HBM;     break;  // HBM
    case 6:  info->vram_type = AMDSMI_VRAM_TYPE_GDDR6;   break;  // GDDR6
    case 7:  info->vram_type = AMDSMI_VRAM_TYPE_LPDDR4;  break;  // LPDDR4
    case 8:  info->vram_type = AMDSMI_VRAM_TYPE_DDR5;    break;  // DDR5
    case 9:  info->vram_type = AMDSMI_VRAM_TYPE_LPDDR5;  break;  // LPDDR5
    default: info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN; break;
    }
    info->vram_bit_width = raw.vram_bit_width;
    info->vram_size      = raw.vram_size_mb;
    info->vram_max_bandwidth = std::numeric_limits<decltype(info->vram_max_bandwidth)>::max();
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_init_gpu_event_notification(amdsmi_processor_handle processor_handle) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle processor_handle,
          uint64_t mask) {
    return rsmi_wrapper(rsmi_event_notification_mask_set, processor_handle, 0, mask);
}

amdsmi_status_t
amdsmi_get_gpu_event_notification(int timeout_ms,
                    uint32_t *num_elem, amdsmi_evt_notification_data_t *data) {
    AMDSMI_CHECK_INIT();

    if (num_elem == nullptr || data == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Get the rsmi data
    std::vector<rsmi_evt_notification_data_t> r_data(*num_elem);
    rsmi_status_t r = rsmi_event_notification_get(
                        timeout_ms, num_elem, &r_data[0]);
    if (r != RSMI_STATUS_SUCCESS) {
        return amd::smi::rsmi_to_amdsmi_status(r);
    }
    // convert output
    for (uint32_t i=0; i < *num_elem; i++) {
        rsmi_evt_notification_data_t rsmi_data = r_data[i];
        data[i].event = static_cast<amdsmi_evt_notification_type_t>(
                rsmi_data.event);
        // Size is tied max event notification size
        snprintf(data[i].message,
                AMDSMI_MAX_STRING_LENGTH,
                "%s",
                rsmi_data.message);
        amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
            .gpu_index_to_handle(rsmi_data.dv_ind, &(data[i].processor_handle));
        if (r != AMDSMI_STATUS_SUCCESS) return r;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_stop_gpu_event_notification(
                amdsmi_processor_handle processor_handle) {
    return rsmi_wrapper(rsmi_event_notification_stop, processor_handle, 0);
}

amdsmi_status_t amdsmi_gpu_counter_group_supported(
        amdsmi_processor_handle processor_handle, amdsmi_event_group_t group) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_gpu_create_counter(amdsmi_processor_handle processor_handle,
        amdsmi_event_type_t type, amdsmi_event_handle_t *evnt_handle) {
    return rsmi_wrapper(rsmi_dev_counter_create, processor_handle, 0,
                    static_cast<rsmi_event_type_t>(type),
                    static_cast<rsmi_event_handle_t*>(evnt_handle));
}

amdsmi_status_t amdsmi_gpu_destroy_counter(amdsmi_event_handle_t evnt_handle) {
    rsmi_status_t r = rsmi_dev_counter_destroy(
        static_cast<rsmi_event_handle_t>(evnt_handle));
    return amd::smi::rsmi_to_amdsmi_status(r);
}

amdsmi_status_t amdsmi_gpu_control_counter(amdsmi_event_handle_t evt_handle,
                                amdsmi_counter_command_t cmd, void *cmd_args) {
    rsmi_status_t r = rsmi_counter_control(
        static_cast<rsmi_event_handle_t>(evt_handle),
        static_cast<rsmi_counter_command_t>(cmd), cmd_args);
    return amd::smi::rsmi_to_amdsmi_status(r);
}

amdsmi_status_t
amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle,
                            amdsmi_counter_value_t *value) {
    rsmi_status_t r = rsmi_counter_read(
        static_cast<rsmi_event_handle_t>(evt_handle),
        reinterpret_cast<rsmi_counter_value_t*>(value));
    return amd::smi::rsmi_to_amdsmi_status(r);
}

amdsmi_status_t
 amdsmi_get_gpu_available_counters(amdsmi_processor_handle processor_handle,
                            amdsmi_event_group_t grp, uint32_t *available) {
    return rsmi_wrapper(rsmi_counter_available_counters_get, processor_handle, 0,
                    static_cast<rsmi_event_group_t>(grp),
                    available);
}

amdsmi_status_t
amdsmi_topo_get_numa_node_number(amdsmi_processor_handle processor_handle, uint32_t *numa_node) {
    // NUMA node topology requires KFD sysfs, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_topo_get_link_weight(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                          uint64_t *weight) {
    // Link weight requires KFD sysfs topology, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
 amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                          uint64_t *min_bandwidth, uint64_t *max_bandwidth) {
    // Min/max bandwidth requires KFD sysfs topology, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}


amdsmi_status_t amdsmi_get_link_metrics(amdsmi_processor_handle processor_handle,
          amdsmi_link_metrics_t *link_metrics) {
    AMDSMI_CHECK_INIT();
    if (link_metrics == nullptr)  return AMDSMI_STATUS_INVAL;

    amdsmi_gpu_metrics_t metric_info = {};
    for (unsigned int i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i) {
        link_metrics->links[i].max_bandwidth = std::numeric_limits<uint32_t>::max();
        link_metrics->links[i].bit_rate = std::numeric_limits<uint32_t>::max();
        link_metrics->links[i].bdf = amdsmi_bdf_t{};
    }

    amdsmi_status_t status =  amdsmi_get_gpu_metrics_info(
            processor_handle, &metric_info);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;
    link_metrics->num_links = AMDSMI_MAX_NUM_XGMI_LINKS;

    uint16_t link_to_dst_node[AMDSMI_MAX_NUM_XGMI_LINKS];
    std::fill_n(link_to_dst_node, AMDSMI_MAX_NUM_XGMI_LINKS, std::numeric_limits<uint16_t>::max());
    status =  rsmi_wrapper(rsmi_dev_xgmi_port_num_get, processor_handle, 0,
        &link_metrics->num_links, link_to_dst_node);

    for (unsigned int i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; i++) {
        memset(&link_metrics->links[i].bdf, 0xFF, sizeof(amdsmi_bdf_t));
        if (link_to_dst_node[i] != std::numeric_limits<uint16_t>::max()) {
            uint32_t node_id = link_to_dst_node[i];
            std::string node_symlink = "node" + std::to_string(node_id);
            std::string sysfs_base = "/sys/bus/pci/devices/";
            DIR *dir = opendir(sysfs_base.c_str());
            if (dir) {
                struct dirent *entry;
                while ((entry = readdir(dir)) != nullptr) {
                    if (entry->d_type != DT_DIR && entry->d_type != DT_LNK)
                        continue;
                    std::string bdf = entry->d_name;
                    if (bdf == "." || bdf == "..") continue;
                    std::string symlink_path = sysfs_base + bdf + "/xgmi_hive_info/" + node_symlink;
                    char buf[PATH_MAX] = {0};
                    ssize_t len = readlink(symlink_path.c_str(), buf, sizeof(buf)-1);
                    if (len > 0) {
                        buf[len] = '\0';
                        std::string target(buf);
                        size_t last_slash = target.find_last_of('/');
                        std::string bdf_str = (last_slash != std::string::npos) ? target.substr(last_slash + 1) : target;
                        // Parse BDF string: "dddd:bb:dd.f"
                        uint64_t domain = 0;
                        uint32_t bus = 0, device = 0, function = 0;
                        if (sscanf(bdf_str.c_str(), "%4lx:%2x:%2x.%1x", &domain, &bus, &device, &function) == 4) {
                            amdsmi_bdf_t dst_bdf = {};
                            dst_bdf.domain_number = domain & 0xffffffffffff;
                            dst_bdf.bus_number = static_cast<uint8_t>(bus) & 0xff;
                            dst_bdf.device_number = static_cast<uint8_t>(device) & 0x1f;
                            dst_bdf.function_number = static_cast<uint8_t>(function) & 0x07;
                            link_metrics->links[i].bdf = dst_bdf;
                        }
                        break; // Found, stop searching
                    }
                }
                closedir(dir);
            }
        }
        link_metrics->links[i].read = metric_info.xgmi_read_data_acc[i];
        link_metrics->links[i].write = metric_info.xgmi_write_data_acc[i];
        link_metrics->links[i].link_type = AMDSMI_LINK_TYPE_XGMI;
        if (metric_info.xgmi_link_speed != std::numeric_limits<uint16_t>::max()) {
            link_metrics->links[i].bit_rate = metric_info.xgmi_link_speed;
        }
        if ((metric_info.xgmi_link_speed != std::numeric_limits<uint16_t>::max()) &&
            (metric_info.xgmi_link_width != std::numeric_limits<uint16_t>::max()))
            link_metrics->links[i].max_bandwidth = metric_info.xgmi_link_speed * metric_info.xgmi_link_width;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_topo_get_link_type(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                        uint64_t *hops, amdsmi_link_type_t *type) {
    // Link type requires KFD sysfs topology, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_is_P2P_accessible(amdsmi_processor_handle processor_handle_src,
                amdsmi_processor_handle processor_handle_dst,
                       bool *accessible) {
    // P2P accessibility requires KFD sysfs topology, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_topo_get_p2p_status(amdsmi_processor_handle processor_handle_src,
                           amdsmi_processor_handle processor_handle_dst,
                           amdsmi_link_type_t *type, amdsmi_p2p_capability_t *cap) {
    // P2P status requires KFD sysfs topology, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

// Compute Partition functions
amdsmi_status_t
amdsmi_get_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                  char *compute_partition, uint32_t len) {
    if (compute_partition == nullptr || len == 0) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_compute_partition_type_t compute_partition) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

// Memory Partition functions
amdsmi_status_t
amdsmi_get_gpu_memory_partition(amdsmi_processor_handle processor_handle,
                                  char *memory_partition, uint32_t len) {
    if (memory_partition == nullptr || len == 0) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_memory_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_memory_partition_type_t memory_partition) {
    AMDSMI_CHECK_INIT();
    if (memory_partition != AMDSMI_MEMORY_PARTITION_UNKNOWN
        && memory_partition != AMDSMI_MEMORY_PARTITION_NPS1
        && memory_partition != AMDSMI_MEMORY_PARTITION_NPS2
        && memory_partition != AMDSMI_MEMORY_PARTITION_NPS4
        && memory_partition != AMDSMI_MEMORY_PARTITION_NPS8) {
        return AMDSMI_STATUS_INVAL;
    }
    std::ostringstream ss;
    std::lock_guard<std::mutex> g(myMutex);

    const uint32_t k256 = 256;
    char current_partition[k256];
    std::string current_partition_str = "UNKNOWN";
    std::string req_user_partition = "UNKNOWN";

    req_user_partition.clear();
    switch (memory_partition) {
      case AMDSMI_MEMORY_PARTITION_NPS1:
        req_user_partition = "NPS1";
        break;
      case AMDSMI_MEMORY_PARTITION_NPS2:
        req_user_partition = "NPS2";
        break;
      case AMDSMI_MEMORY_PARTITION_NPS4:
        req_user_partition = "NPS4";
        break;
      case AMDSMI_MEMORY_PARTITION_NPS8:
        req_user_partition = "NPS8";
        break;
      default:
        req_user_partition = "UNKNOWN";
        break;
    }
    rsmi_memory_partition_type_t rsmi_type;
    auto it = nps_amdsmi_to_RSMI.find(memory_partition);
    if (it != nps_amdsmi_to_RSMI.end()) {
        rsmi_type = it->second;
    } else if (it == nps_amdsmi_to_RSMI.end()) {
        return AMDSMI_STATUS_INVAL;
    }

    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_memory_partition_config(amdsmi_processor_handle processor_handle,
                                        amdsmi_memory_partition_config_t *config) {
    AMDSMI_CHECK_INIT();
    std::ostringstream ss;
    if (config == nullptr) {
      return AMDSMI_STATUS_INVAL;
    }

    // initialization for devices which do not support partitions
    amdsmi_nps_caps_t flags;
    flags.nps_flags.nps1_cap = 0;
    flags.nps_flags.nps2_cap = 0;
    flags.nps_flags.nps4_cap = 0;
    flags.nps_flags.nps8_cap = 0;
    config->partition_caps = flags;
    config->mp_mode = AMDSMI_MEMORY_PARTITION_UNKNOWN;
    // TODO(amdsmi_team): Will BM/guest VMs have numa ranges?
    config->num_numa_ranges = 0;

    // current memory partition
    constexpr uint32_t kCurrentPartitionSize = 5;
    char current_mem_partition[kCurrentPartitionSize] = {};
    std::string current_mem_partition_str = "N/A";
    amdsmi_status_t status = amdsmi_get_gpu_memory_partition(processor_handle,
                                            current_mem_partition, kCurrentPartitionSize);
    ss << __PRETTY_FUNCTION__ << " | amdsmi_get_gpu_memory_partition() current_partition = |"
       << current_mem_partition << "|";
    LOG_DEBUG(ss);
    current_mem_partition_str = current_mem_partition;
    if (status == AMDSMI_STATUS_SUCCESS) {
        if (current_mem_partition_str == "NPS1") {
            config->mp_mode = AMDSMI_MEMORY_PARTITION_NPS1;
        } else if (current_mem_partition_str == "NPS2") {
            config->mp_mode = AMDSMI_MEMORY_PARTITION_NPS2;
        } else if (current_mem_partition_str == "NPS4") {
            config->mp_mode = AMDSMI_MEMORY_PARTITION_NPS4;
        } else if (current_mem_partition_str == "NPS8") {
            config->mp_mode = AMDSMI_MEMORY_PARTITION_NPS8;
        }
    }

    // Add memory partition capabilities here
    constexpr uint32_t kLenCapsSize = 30;
    char memory_caps[kLenCapsSize] = {};
    auto status_mem_caps = rsmi_wrapper(rsmi_dev_memory_partition_capabilities_get,
                                          processor_handle, 0,
                                          memory_caps, kLenCapsSize);
    ss << __PRETTY_FUNCTION__
       << " | rsmi_dev_memory_partition_capabilities_get Returning: "
       << smi_amdgpu_get_status_string(status, false)
       << " | Type: memory_partition_capabilities"
       << " | Data: " << memory_caps;
    LOG_DEBUG(ss);
    std::string memory_caps_str = "N/A";
    if (status_mem_caps == AMDSMI_STATUS_SUCCESS) {  // older kernels may not support this
        memory_caps_str = std::string(memory_caps);
        if (memory_caps_str.find("NPS1") != std::string::npos) {
            flags.nps_flags.nps1_cap = 1;
        }
        if (memory_caps_str.find("NPS2") != std::string::npos) {
            flags.nps_flags.nps2_cap = 1;
        }
        if (memory_caps_str.find("NPS4") != std::string::npos) {
            flags.nps_flags.nps4_cap = 1;
        }
        if (memory_caps_str.find("NPS8") != std::string::npos) {
            flags.nps_flags.nps8_cap = 1;
        }
    }
    config->partition_caps = flags;
    return status;
}

amdsmi_status_t
amdsmi_set_gpu_memory_partition_mode(amdsmi_processor_handle processor_handle,
                                     amdsmi_memory_partition_type_t mode) {
    AMDSMI_CHECK_INIT();
    return amdsmi_set_gpu_memory_partition(processor_handle, mode);
}

// Accelerator Partition functions
amdsmi_status_t
amdsmi_get_gpu_accelerator_partition_profile_config(amdsmi_processor_handle processor_handle,
                                  amdsmi_accelerator_partition_profile_config_t *profile_config) {
    AMDSMI_CHECK_INIT();
    if (profile_config == nullptr) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__
       << " | START ";
    // std::cout << ss.str() << std::endl;
    LOG_DEBUG(ss);

    if (profile_config == nullptr) {
        ss << __PRETTY_FUNCTION__ << " | profile_config is nullptr" << std::endl;
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }

    // Initialize values
    amdsmi_status_t return_status = AMDSMI_STATUS_NOT_SUPPORTED;
    amdsmi_status_t status = AMDSMI_STATUS_NOT_SUPPORTED;
    profile_config->default_profile_index = 0;
    profile_config->num_profiles = 0;
    profile_config->num_resource_profiles = 0;
    profile_config->resource_profiles->profile_index = 0;
    profile_config->resource_profiles->resource_type = AMDSMI_ACCELERATOR_MAX;
    profile_config->resource_profiles->partition_resource = 0;
    profile_config->resource_profiles->num_partitions_share_resource = 0;
    amdsmi_nps_caps_t flags;
    flags.nps_flags.nps1_cap = 0;
    flags.nps_flags.nps2_cap = 0;
    flags.nps_flags.nps4_cap = 0;
    flags.nps_flags.nps8_cap = 0;

    ss << __PRETTY_FUNCTION__
       << " | 1";
    // std::cout << ss.str() << std::endl;
    LOG_DEBUG(ss);

    // get supported xcp_configs (this will tell use # of profiles/index's)
    // /sys/class/drm/../device/compute_partition_config/supported_xcp_configs
    // otherwise fall back to use /sys/class/drm/../device/available_compute_partition
    // ex. SPX, DPX, QPX, CPX
    std::string accelerator_caps_str = "N/A";
    constexpr uint32_t kLenXCPConfigSize = 30;
    char supported_xcp_configs[kLenXCPConfigSize];
    bool use_xcp_config = false;
    return_status
        = rsmi_wrapper(rsmi_dev_compute_partition_supported_xcp_configs_get, processor_handle, 0,
                          supported_xcp_configs, kLenXCPConfigSize);
    if (return_status == AMDSMI_STATUS_SUCCESS) {
        accelerator_caps_str.clear();
        accelerator_caps_str = std::string(supported_xcp_configs);
        accelerator_caps_str = amd::smi::trimAllWhiteSpace(accelerator_caps_str);
        use_xcp_config = true;
    } else {  // initialize what we can
        ss << __PRETTY_FUNCTION__
           << "\n | rsmi_dev_compute_partition_supported_xcp_configs_get()"
           << " returned: " << smi_amdgpu_get_status_string(return_status, false)
           << "\n | Defaulting to use rsmi_dev_compute_partition_capabilities_get";
        // std::cout << ss.str() << std::endl;
        LOG_DEBUG(ss);
        return_status = rsmi_wrapper(rsmi_dev_compute_partition_capabilities_get,
                                     processor_handle, 0,
                                     supported_xcp_configs, kLenXCPConfigSize);
        if (return_status == AMDSMI_STATUS_SUCCESS) {
            accelerator_caps_str.clear();
            accelerator_caps_str = std::string(supported_xcp_configs);
            accelerator_caps_str = amd::smi::trimAllWhiteSpace(accelerator_caps_str);
        } else {
            ss << __PRETTY_FUNCTION__
               << "\n | rsmi_dev_compute_partition_capabilities_get() failed, "
               << "likely due to feature not supported"
               << "\n | Returning: " << smi_amdgpu_get_status_string(return_status, false);
            // std::cout << ss.str() << std::endl;
            LOG_DEBUG(ss);
            return return_status;
        }
    }

    ss << __PRETTY_FUNCTION__
       << (use_xcp_config ? "\n | Used rsmi_dev_compute_partition_supported_xcp_configs_get()" :
                            "\n | Used rsmi_dev_compute_partition_capabilities_get()")
       << "\n | Returning: " << smi_amdgpu_get_status_string(return_status, false)
       << "\n | Type: "
       << (use_xcp_config ? amd::smi::Device::get_type_string(amd::smi::kDevSupportedXcpConfigs):
            amd::smi::Device::get_type_string(amd::smi::kDevAvailableComputePartition))
       << "\n | Data: " << accelerator_caps_str;
    // std::cout << ss.str() << std::endl;
    LOG_DEBUG(ss);
    if (accelerator_caps_str.find("SPX") != std::string::npos) {
        profile_config->profiles[profile_config->num_profiles].profile_type
            = AMDSMI_ACCELERATOR_PARTITION_SPX;
        profile_config->profiles[profile_config->num_profiles].num_partitions = 1;
        profile_config->profiles[profile_config->num_profiles].profile_index
            = profile_config->num_profiles;
        // default all memory partition caps to 0
        profile_config->profiles[profile_config->num_profiles].memory_caps = flags;
        profile_config->num_profiles++;
    }
    if (accelerator_caps_str.find("DPX") != std::string::npos) {
        profile_config->profiles[profile_config->num_profiles].profile_type
            = AMDSMI_ACCELERATOR_PARTITION_DPX;
        profile_config->profiles[profile_config->num_profiles].num_partitions = 2;
        profile_config->profiles[profile_config->num_profiles].profile_index
            = profile_config->num_profiles;
        // default all memory partition caps to 0
        profile_config->profiles[profile_config->num_profiles].memory_caps = flags;
        profile_config->num_profiles++;
    }
    if (accelerator_caps_str.find("TPX") != std::string::npos) {
        profile_config->profiles[profile_config->num_profiles].profile_type
            = AMDSMI_ACCELERATOR_PARTITION_TPX;
        profile_config->profiles[profile_config->num_profiles].num_partitions = 3;
        profile_config->profiles[profile_config->num_profiles].profile_index
            = profile_config->num_profiles;
        // default all memory partition caps to 0
        profile_config->profiles[profile_config->num_profiles].memory_caps = flags;
        profile_config->num_profiles++;
    }
    if (accelerator_caps_str.find("QPX") != std::string::npos) {
        profile_config->profiles[profile_config->num_profiles].profile_type
            = AMDSMI_ACCELERATOR_PARTITION_QPX;
        profile_config->profiles[profile_config->num_profiles].num_partitions = 4;
        profile_config->profiles[profile_config->num_profiles].profile_index
            = profile_config->num_profiles;
        // default all memory partition caps to 0
        profile_config->profiles[profile_config->num_profiles].memory_caps = flags;
        profile_config->num_profiles++;
    }
    if (accelerator_caps_str.find("CPX") != std::string::npos) {
        profile_config->profiles[profile_config->num_profiles].profile_type
            = AMDSMI_ACCELERATOR_PARTITION_CPX;
        // Note: # of XCDs is max # of partitions CPX supports
        uint16_t tmp_xcd_count = 0;
        status = rsmi_wrapper(rsmi_dev_metrics_xcd_counter_get,
                                            processor_handle, 0, &tmp_xcd_count);
        profile_config->profiles[
                profile_config->num_profiles].num_partitions = 0;  // default to 0
        if (status == AMDSMI_STATUS_SUCCESS) {
            profile_config->profiles[
                profile_config->num_profiles].num_partitions = tmp_xcd_count;
        }
        profile_config->profiles[profile_config->num_profiles].profile_index
            = profile_config->num_profiles;
        // default all memory partition caps to 0
        profile_config->profiles[profile_config->num_profiles].memory_caps = flags;
        profile_config->num_profiles++;
    }

    ss << __PRETTY_FUNCTION__
       << " | 2";
    // std::cout << ss.str() << std::endl;
    LOG_DEBUG(ss);
    auto resource_index = 0;
    // get resource info for each profile
    for (auto i = 0U; i < profile_config->num_profiles; i++) {
        profile_config->profiles[i].num_resources = 0;  // start at 0 resources and increment
        auto it = partition_types_map.find(profile_config->profiles[i].profile_type);
        std::string partition_type_str = "UNKNOWN";
        if (it != partition_types_map.end()) {
            partition_type_str.clear();
            partition_type_str = it->second;
        }
        auto it3 = accelerator_to_RSMI.find(profile_config->profiles[i].profile_type);
        rsmi_compute_partition_type_t rsmi_partition_type = RSMI_COMPUTE_PARTITION_INVALID;
        if (it3 == accelerator_to_RSMI.end()) {
            ss << __PRETTY_FUNCTION__ << " | reached end of map\n";
            LOG_DEBUG(ss);
            continue;
        } else {
            rsmi_partition_type = it3->second;
        }
        status = rsmi_wrapper(rsmi_dev_compute_partition_xcp_config_set, processor_handle, 0,
                              rsmi_partition_type);
        ss << __PRETTY_FUNCTION__
           << "\n | profile_num:  " << i
           << "\n | profile_type: " << partition_type_str
           << "\n | rsmi_dev_compute_partition_xcp_config_set(" << partition_type_str
           << ") Returning: "
           << smi_amdgpu_get_status_string(status, false)
           << "\n | Type: "
           << amd::smi::Device::get_type_string(amd::smi::kDevSupportedXcpConfigs)
           << "\n | Data: " << "N/A";
        // std::cout << ss.str() << std::endl;
        LOG_DEBUG(ss);

        // 1) get memory caps for each profile
        /**
         * rsmi_status_t rsmi_dev_compute_partition_supported_nps_configs_get(uint32_t dv_ind, char *supported_configs,
         * uint32_t len);
         */
        constexpr uint32_t kLenNPSConfigSize = 30;
        char supported_nps_configs[kLenNPSConfigSize];
        std::string supported_nps_caps_str = "N/A";
        status = rsmi_wrapper(rsmi_dev_compute_partition_supported_nps_configs_get,
                              processor_handle, 0,
                              supported_nps_configs, kLenNPSConfigSize);
        if (status == AMDSMI_STATUS_SUCCESS) {
            supported_nps_caps_str.clear();
            supported_nps_caps_str = std::string(supported_nps_configs);
        }
        if (supported_nps_caps_str.find("NPS1") != std::string::npos) {
            profile_config->profiles[i].memory_caps.nps_flags.nps1_cap = 1;
        }
        if (supported_nps_caps_str.find("NPS2") != std::string::npos) {
            profile_config->profiles[i].memory_caps.nps_flags.nps2_cap = 1;
        }
        if (supported_nps_caps_str.find("NPS4") != std::string::npos) {
            profile_config->profiles[i].memory_caps.nps_flags.nps4_cap = 1;
        }
        if (supported_nps_caps_str.find("NPS8") != std::string::npos) {
            profile_config->profiles[i].memory_caps.nps_flags.nps8_cap = 1;
        }
        // 2) get resource profiles
        for (auto r = static_cast<int>(RSMI_ACCELERATOR_XCC);
            r < static_cast<int>(RSMI_ACCELERATOR_MAX); r++) {
            rsmi_accelerator_partition_resource_type_t type
                = static_cast<rsmi_accelerator_partition_resource_type_t>(r);
            rsmi_accelerator_partition_resource_profile_t profile;
            status = rsmi_wrapper(
                rsmi_dev_compute_partition_resource_profile_get, processor_handle, 0,
                &type, &profile);
            if (status == AMDSMI_STATUS_SUCCESS) {
                uint32_t inc_res_profile =
                    profile_config->num_resource_profiles + 1;
                if (inc_res_profile < static_cast<uint32_t>(RSMI_ACCELERATOR_MAX)) {
                    profile_config->num_resource_profiles = inc_res_profile;
                }
                profile_config->resource_profiles[resource_index].profile_index = i;
                profile_config->resource_profiles[resource_index].resource_type
                    = static_cast<amdsmi_accelerator_partition_resource_type_t>(type);
                profile_config->resource_profiles[resource_index].partition_resource
                    = profile.partition_resource;
                profile_config->resource_profiles[resource_index].num_partitions_share_resource
                    = profile.num_partitions_share_resource;
                auto it3 =
                    resource_types_map.find(
                        profile_config->resource_profiles[resource_index].resource_type);
                std::string resource_type_str = "UNKNOWN";
                if (it3 != resource_types_map.end()) {
                    resource_type_str.clear();
                    resource_type_str = it3->second;
                }
                ss << __PRETTY_FUNCTION__ << " | profile_debug 1 "
                << "\n profile type: " << partition_type_str
                << "\n resource_index: " << resource_index
                << "\n profile_index: " << i
                << "\n resource_type: " << resource_type_str
                << "\n partition_resource: " << profile.partition_resource
                << "\n num_partitions_share_resource: " << profile.num_partitions_share_resource
                << std::endl;
                LOG_DEBUG(ss);
                resource_index += 1;

                uint32_t inc_resources =
                    profile_config->profiles[i].num_resources  + 1;
                if (inc_resources < static_cast<uint32_t>(RSMI_ACCELERATOR_MAX)) {
                    profile_config->profiles[i].num_resources = inc_resources;
                }
                ss << __PRETTY_FUNCTION__ << " | profile_debug 2 "
                    << "\n profile_config->profiles[i].num_resources: "
                    << profile_config->profiles[i].num_resources
                    << std::endl;
                // std::cout << ss.str() << std::endl;
                LOG_DEBUG(ss);
            }

            it = partition_types_map.find(profile_config->profiles[i].profile_type);
            partition_type_str = "UNKNOWN";
            if (it != partition_types_map.end()) {
                partition_type_str.clear();
                partition_type_str = it->second;
            }
            auto it2 = resource_types_map.find(
                static_cast<amdsmi_accelerator_partition_resource_type_t>(type));
            std::string resource_type_str = "UNKNOWN";
            if (it2 != resource_types_map.end()) {
                resource_type_str.clear();
                resource_type_str = it2->second;
            }
            auto current_resource_idx = (resource_index >= 1) ? resource_index - 1 : 0;
            std::string nps_caps = "N/A";
            if (profile_config->profiles[i].memory_caps.nps_flags.nps1_cap == 1) {
                if (nps_caps == "N/A") {
                    nps_caps.clear();
                    nps_caps = "NPS1";
                } else {
                    nps_caps += ", NPS1";
                }
            }
            if (profile_config->profiles[i].memory_caps.nps_flags.nps2_cap == 1) {
                if (nps_caps == "N/A") {
                    nps_caps.clear();
                    nps_caps = "NPS2";
                } else {
                    nps_caps += ", NPS2";
                }
            }
            if (profile_config->profiles[i].memory_caps.nps_flags.nps4_cap == 1) {
                if (nps_caps == "N/A") {
                    nps_caps.clear();
                    nps_caps = "NPS4";
                } else {
                    nps_caps += ", NPS4";
                }
            }
            if (profile_config->profiles[i].memory_caps.nps_flags.nps8_cap == 1) {
                if (nps_caps == "N/A") {
                    nps_caps.clear();
                    nps_caps = "NPS8";
                } else {
                    nps_caps += ", NPS8";
                }
            }
            ss << __PRETTY_FUNCTION__
               << " | Detailed output"
               << "\n | profile_config->num_profiles: " << profile_config->num_profiles
               << "\n | profile_num (i):  " << i
               << "\n | resource_num (r): " << r
               << "\n | current_resource_idx: " << current_resource_idx
               << "\n | profile_config->resource_profiles[current_resource_idx].profile_index: "
               << profile_config->resource_profiles[current_resource_idx].profile_index
               << "\n | profile_config->profiles[i].memory_caps: "
               << nps_caps
               << "\n***********************************************"
               << "\n | profile_config->profiles[i].num_resources: "
               << profile_config->profiles[i].num_resources
               << "\n***********************************************"
               << "\n | profile_type: " << partition_type_str
               << "\n | resource_type: " << resource_type_str
               << "\n | partition_resource: " << profile.partition_resource
               << "\n | num_partitions_share_resource: "
               << profile.num_partitions_share_resource
               << "\n | profile_config->num_resource_profiles: "
               << profile_config->num_resource_profiles
               << "\n | rsmi_dev_compute_partition_resource_profile_get("
               << resource_type_str << ") Returning: "
               << smi_amdgpu_get_status_string(status, false)
               << "\n | Type: "
               << amd::smi::Device::get_type_string(amd::smi::kDevSupportedXcpConfigs)
               << "\n";
            // std::cout << ss.str() << std::endl;
            LOG_DEBUG(ss);
        }  // END resources loop
    }  // END profile loop

    int res_ind = 0;
    for (uint32_t i = 0; i < profile_config->num_profiles; i++) {
      auto current_profile = profile_config->profiles[i];
      std::string profile_type_str = "N/A";
      if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_SPX) {
        profile_type_str = "SPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_DPX) {
        profile_type_str = "DPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_TPX) {
        profile_type_str = "TPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_QPX) {
        profile_type_str = "QPX";
      } else if (current_profile.profile_type == AMDSMI_ACCELERATOR_PARTITION_CPX) {
        profile_type_str = "CPX";
      }

      std::string nps_caps_str = "";
      if ((current_profile.memory_caps.nps_flags.nps1_cap == 0
          && current_profile.memory_caps.nps_flags.nps2_cap == 0
          && current_profile.memory_caps.nps_flags.nps4_cap == 0
          && current_profile.memory_caps.nps_flags.nps8_cap == 0)) {
        nps_caps_str = "N/A";
      } else {
        nps_caps_str.clear();
        if (current_profile.memory_caps.nps_flags.nps1_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS1" : nps_caps_str += ", NPS1";
        }
        if (current_profile.memory_caps.nps_flags.nps2_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS2" : nps_caps_str += ", NPS2";
        }
        if (current_profile.memory_caps.nps_flags.nps4_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS4" : nps_caps_str += ", NPS4";
        }
        if (current_profile.memory_caps.nps_flags.nps8_cap) {
          (nps_caps_str.empty()) ? nps_caps_str += "NPS8" : nps_caps_str += ", NPS8";
        }
      }

      ss << __PRETTY_FUNCTION__ << " | profile_debug; after compiling info p1 "
         << "\n\t**profile_config.profiles[" << i << "]:\n"
         << "\t\tprofile_type: " << profile_type_str
         << "\n\t\tnum_partitions: " << current_profile.num_partitions
         << "\n\t\tmemory_caps: " << nps_caps_str
         << "\n\t\tcurrent_profile.num_resources: " << current_profile.num_resources
         << std::endl;
      // std::cout << ss.str() << std::endl;
      LOG_DEBUG(ss);

      for (uint32_t j = 0; j < current_profile.num_resources; j++) {
        auto rp = profile_config->resource_profiles[res_ind];

        auto it2 = resource_types_map.find(rp.resource_type);
        std::string resource_type_str = "UNKNOWN";
        if (it2 != resource_types_map.end()) {
            resource_type_str.clear();
            resource_type_str = it2->second;
        }
        ss << __PRETTY_FUNCTION__ << " | profile_debug; after compiling info p2 "
                  << "\n\t\t\tprofile_index: " << current_profile.profile_index
                  << "\n\t\t\tres_ind: " << res_ind
                  << "\n\t\t\tprofile_config.resource_profiles[" << res_ind
                  << "].resource_type: "
                  << resource_type_str
                  << "\n\t\t\tprofile_config.resource_profiles[" << res_ind
                  << "].partition_resource: "
                  << rp.partition_resource
                  << "\n\t\t\tprofile_config.resource_profiles[" << res_ind
                  << "].num_partitions_share_resource: "
                  << rp.num_partitions_share_resource
                  << std::endl;
        LOG_DEBUG(ss);
        res_ind++;
      }
    }
    ss << __PRETTY_FUNCTION__
       << " | END returning " << smi_amdgpu_get_status_string(return_status, false);
    // std::cout << ss.str() << std::endl;
    LOG_INFO(ss);

    return return_status;
}

amdsmi_status_t
amdsmi_get_gpu_accelerator_partition_profile(amdsmi_processor_handle processor_handle,
                                             amdsmi_accelerator_partition_profile_t *profile,
                                             uint32_t *partition_id) {
    std::ostringstream ss;
    AMDSMI_CHECK_INIT();
    if (profile == nullptr || partition_id == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // initialization for devices which do not support partitions
    profile->num_partitions = std::numeric_limits<uint32_t>::max();
    profile->profile_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
    *partition_id = {0};
    profile->profile_index = std::numeric_limits<uint32_t>::max();
    profile->num_resources = 0;

    amdsmi_nps_caps_t flags;
    flags.nps_flags.nps1_cap = 0;
    flags.nps_flags.nps2_cap = 0;
    flags.nps_flags.nps4_cap = 0;
    flags.nps_flags.nps8_cap = 0;
    profile->memory_caps = flags;

    // TODO(amdsmi_team): add resources here ^
    auto tmp_partition_id = uint32_t(0);
    amdsmi_status_t status = AMDSMI_STATUS_NOT_SUPPORTED;

    // TODO(amdsmi_team): should we do fallback?
    // Info doesn't populate properly if missing other files - CLI FIX?
    // Reason: older kernels do not support xcp_configs

    // get supported xcp_configs (this will tell use # of profiles/index's)
    // /sys/class/drm/../device/compute_partition_config/supported_xcp_configs
    // otherwise fall back to use /sys/class/drm/../device/available_compute_partition
    // ex. SPX, DPX, QPX, CPX
    // Depending on what is available, we can determine the profile index
    // ex. SPX = 0, DPX = 1, QPX = 2, CPX = 3; other devices may have different values
    std::string accelerator_capabilities = "N/A";
    constexpr uint32_t kLenXCPConfigSize = 30;
    char supported_xcp_configs[kLenXCPConfigSize];
    bool use_xcp_config = false;
    status
        = rsmi_wrapper(rsmi_dev_compute_partition_supported_xcp_configs_get, processor_handle, 0,
                          supported_xcp_configs, kLenXCPConfigSize);
    if (status == AMDSMI_STATUS_SUCCESS) {
        accelerator_capabilities.clear();
        accelerator_capabilities = std::string(supported_xcp_configs);
        use_xcp_config = true;
    }

    ss << __PRETTY_FUNCTION__
       << (use_xcp_config ? "\n | Used rsmi_dev_compute_partition_supported_xcp_configs_get()" :
                            "\n | Used rsmi_dev_compute_partition_capabilities_get()")
       << "\n | Returned: " << smi_amdgpu_get_status_string(status, false)
       << "\n | Type: "
       << (use_xcp_config ? amd::smi::Device::get_type_string(amd::smi::kDevSupportedXcpConfigs):
            amd::smi::Device::get_type_string(amd::smi::kDevAvailableComputePartition))
       << "\n | Data: " << accelerator_capabilities;

    // std::cout << ss.str() << std::endl;
    LOG_DEBUG(ss);

    // get index by comma and place into a string vector
    char delimiter = ',';
    std::stringstream ss_obj(accelerator_capabilities);
    std::string temp;
    std::vector<std::string> tokens;
    while (getline(ss_obj, temp, delimiter)) {
        temp = amd::smi::trimAllWhiteSpace(temp);
        tokens.push_back(temp);
    }

    // hold all current available compute partition values within tokens vector
    std::ostringstream ss_1;
    std::copy(std::begin(tokens),
              std::end(tokens),
              amd::smi::make_ostream_joiner(&ss_1, ", "));

    constexpr uint32_t kCurrentPartitionSize = 16;
    char current_partition[kCurrentPartitionSize] = {0};
    std::string current_partition_str = "N/A";
    amdsmi_status_t compute_status = amdsmi_get_gpu_compute_partition(processor_handle,
                                        current_partition, kCurrentPartitionSize);
    ss << __PRETTY_FUNCTION__ << " | amdsmi_get_gpu_compute_partition() current_partition = |"
       << current_partition << "|";
    LOG_DEBUG(ss);
    current_partition_str = current_partition;
    if (status == AMDSMI_STATUS_SUCCESS) {
        // 1) get profile index from
        // /sys/class/drm/../device/compute_partition_config/supported_xcp_configs
        if (current_partition_str == "SPX" || current_partition_str == "DPX"
            || current_partition_str == "TPX" || current_partition_str == "QPX"
            || current_partition_str == "CPX") {
            // get index according to supported_xcp_configs, separated by commas
            if (accelerator_capabilities.find(current_partition_str) != std::string::npos) {
                auto it = std::find(tokens.begin(), tokens.end(), current_partition_str);
                if (it != tokens.end()) {
                    profile->profile_index = static_cast<uint32_t>(std::distance(
                                                tokens.begin(), it));
                }
            }
        }

        // 2) get profile type from /sys/class/drm/../device/current_compute_partition
        if (current_partition_str == "SPX") {
            profile->profile_type = AMDSMI_ACCELERATOR_PARTITION_SPX;
        } else if (current_partition_str == "DPX") {
            profile->profile_type = AMDSMI_ACCELERATOR_PARTITION_DPX;
        } else if (current_partition_str == "TPX") {
            profile->profile_type = AMDSMI_ACCELERATOR_PARTITION_TPX;
        } else if (current_partition_str == "QPX") {
            profile->profile_type = AMDSMI_ACCELERATOR_PARTITION_QPX;
        } else if (current_partition_str == "CPX") {
            profile->profile_type = AMDSMI_ACCELERATOR_PARTITION_CPX;
        } else {
            profile->profile_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
        }
    } else {
        profile->profile_type = AMDSMI_ACCELERATOR_PARTITION_INVALID;
        current_partition_str.clear();
        current_partition_str = "N/A";
    }

    amdsmi_gpu_metrics_t metric_info = {};
    status = amdsmi_get_gpu_metrics_info(processor_handle, &metric_info);
    if (status == AMDSMI_STATUS_SUCCESS
        && metric_info.num_partition != std::numeric_limits<uint16_t>::max()) {
        profile->num_partitions = metric_info.num_partition;
    }

    status = rsmi_wrapper(rsmi_dev_partition_id_get, processor_handle, 0,
                          &tmp_partition_id);
    const uint32_t partition_num = 0;  // Each partition should show the their respective
                                       // partition_id at positon 0 of the array.
                                       // We are no longer populating only the primary partition
                                       // for BM/Guest.

    if (status == AMDSMI_STATUS_SUCCESS) {
        partition_id[partition_num] = tmp_partition_id;
    }

    std::ostringstream ss_2;
    const uint32_t kMaxPartitions = 8;
    uint32_t copy_partition_ids[kMaxPartitions] = {0};  // initialize all to 0s
    std::copy(partition_id, partition_id + kMaxPartitions, copy_partition_ids);
    std::copy(std::begin(copy_partition_ids),
              std::end(copy_partition_ids),
              amd::smi::make_ostream_joiner(&ss_2, ", "));

    auto it_profile_type = partition_types_map.find(profile->profile_type);
    std::string partition_type_str = "N/A";
    if (it_profile_type != partition_types_map.end()) {
        partition_type_str.clear();
        partition_type_str = it_profile_type->second;
    }
    ss << __PRETTY_FUNCTION__
       << " | Num_partitions: " << profile->num_partitions
       << "; profile->profile_type: " << profile->profile_type << " (" << partition_type_str << ")"
       << "; partition_id: " << ss_2.str() << "\n";
    LOG_DEBUG(ss);

    // Add memory partition capabilities here
    constexpr uint32_t kLenCapsSize = 30;
    char memory_caps[kLenCapsSize];
    status = rsmi_wrapper(rsmi_dev_memory_partition_capabilities_get, processor_handle, 0,
                          memory_caps, kLenCapsSize);
    ss << __PRETTY_FUNCTION__
       << " | rsmi_dev_memory_partition_capabilities_get Returning: "
       << smi_amdgpu_get_status_string(status, false)
       << " | Type: memory_partition_capabilities"
       << " | Data: " << memory_caps;
    LOG_DEBUG(ss);
    std::string memory_caps_str = "N/A";
    if (status == AMDSMI_STATUS_SUCCESS) {
        memory_caps_str = std::string(memory_caps);
        if (memory_caps_str.find("NPS1") != std::string::npos) {
            flags.nps_flags.nps1_cap = 1;
        }
        if (memory_caps_str.find("NPS2") != std::string::npos) {
            flags.nps_flags.nps2_cap = 1;
        }
        if (memory_caps_str.find("NPS4") != std::string::npos) {
            flags.nps_flags.nps4_cap = 1;
        }
        if (memory_caps_str.find("NPS8") != std::string::npos) {
            flags.nps_flags.nps8_cap = 1;
        }
    }
    profile->memory_caps = flags;

    ss << __PRETTY_FUNCTION__
       << " | END returning " << smi_amdgpu_get_status_string(compute_status, false) << "\n"
       << " | accelerator_capabilities: " << accelerator_capabilities << "\n"
       << " | current_partition_str: " << current_partition_str << "\n"
       << " | std::vector<std::string> tokens: " << ss_1.str() << "\n"
       << " | profile->num_partitions: " << profile->num_partitions << "\n"
       << " | profile->profile_type: " << partition_type_str << "\n"
       << " | profile->profile_index: " << profile->profile_index << "\n"
       << " | profile->num_resources: " << profile->num_resources << "\n"
       << " | profile->memory_caps: " << "\n"
       << " | nps1_cap: " << profile->memory_caps.nps_flags.nps1_cap << "\n"
       << " | nps2_cap: " << profile->memory_caps.nps_flags.nps2_cap << "\n"
       << " | nps4_cap: " << profile->memory_caps.nps_flags.nps4_cap << "\n"
       << " | nps8_cap: " << profile->memory_caps.nps_flags.nps8_cap << "\n"
       << " | partition_id: " << ss_2.str();
    LOG_INFO(ss);

    return compute_status;  // only return status from amdsmi_get_gpu_compute_partition
                            // as this is the only function that can fail
                            // if the device does not support partitions
}

amdsmi_status_t
amdsmi_set_gpu_accelerator_partition_profile(amdsmi_processor_handle processor_handle,
                                            uint32_t profile_index) {
    AMDSMI_CHECK_INIT();
    std::ostringstream ss;
    amdsmi_accelerator_partition_profile_config_t config;
    amdsmi_status_t status = amdsmi_get_gpu_accelerator_partition_profile_config(
        processor_handle, &config);

    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    std::map<uint32_t, amdsmi_accelerator_partition_type_t> mp_prof_indx_to_accel_type;

    ss << __PRETTY_FUNCTION__ << " | Invalid profile_index: " << profile_index
           << "\n| Max profile_index: " << config.num_profiles - 1
           << "\n| config.num_profiles: " << config.num_profiles
           << "\n| profile_index: " << profile_index
           << "\n| Returning: " << smi_amdgpu_get_status_string(AMDSMI_STATUS_INVAL, false);
    // std::cout << ss.str() << std::endl;
    LOG_DEBUG(ss);
    if (profile_index >= config.num_profiles) {
        ss << __PRETTY_FUNCTION__ << " | Invalid profile_index: " << profile_index
           << "\n| Max profile_index: " << config.num_profiles - 1
           << "\n| Returning: " << smi_amdgpu_get_status_string(AMDSMI_STATUS_INVAL, false);
        // std::cout << ss.str() << std::endl;
        LOG_DEBUG(ss);
        return AMDSMI_STATUS_INVAL;
    }

    for (uint32_t i = 0; i < config.num_profiles; i++) {
        auto it = partition_types_map.find(config.profiles[i].profile_type);
        std::string partition_type_str = "N/A";
        if (it != partition_types_map.end()) {
            partition_type_str.clear();
            partition_type_str = it->second;
        }

        ss << __PRETTY_FUNCTION__ << " | "
        << "config.profiles[" << i << "].profile_type: "
        << static_cast<int>(config.profiles[i].profile_type) << "\n"
        << "| config.profiles[" << i << "].profile_type (str): "
        << partition_type_str << "\n"
        << "| config.profiles[" << i << "].profile_index: "
        << static_cast<int>(config.profiles[i].profile_index)
        << "\n";
        // std::cout << ss.str() << std::endl;
        LOG_DEBUG(ss);
        mp_prof_indx_to_accel_type[config.profiles[i].profile_index]
            = config.profiles[i].profile_type;
    }
    auto return_status = amdsmi_set_gpu_compute_partition(processor_handle,
        static_cast<amdsmi_compute_partition_type_t>(mp_prof_indx_to_accel_type[profile_index]));
    ss << __PRETTY_FUNCTION__ << " | User requested profile_index: " << profile_index
       << "\n| Accelerator Type: "
       << partition_types_map.at(mp_prof_indx_to_accel_type[profile_index])
       << "\n| Returning: " << smi_amdgpu_get_status_string(return_status, false);
    // std::cout << ss.str() << std::endl;
    LOG_INFO(ss);
    return return_status;
}

// TODO(bliu) : other xgmi related information
amdsmi_status_t
amdsmi_get_xgmi_info(amdsmi_processor_handle processor_handle, amdsmi_xgmi_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;
    return rsmi_wrapper(rsmi_dev_xgmi_hive_id_get, processor_handle, 0,
                    &(info->xgmi_hive_id));
}

amdsmi_status_t
amdsmi_gpu_xgmi_error_status(amdsmi_processor_handle processor_handle, amdsmi_xgmi_status_t *status) {
    return rsmi_wrapper(rsmi_dev_xgmi_error_status, processor_handle, 0,
                    reinterpret_cast<rsmi_xgmi_status_t*>(status));
}

amdsmi_status_t
amdsmi_reset_gpu_xgmi_error(amdsmi_processor_handle processor_handle) {
    return rsmi_wrapper(rsmi_dev_xgmi_error_reset, processor_handle, 0);
}

amdsmi_status_t
amdsmi_get_gpu_compute_process_info(amdsmi_process_info_t *procs, uint32_t *num_items) {
    AMDSMI_CHECK_INIT();

    if (num_items == nullptr)
        return AMDSMI_STATUS_INVAL;

    // Enumerate GPU processes across all devices via WSL2 D3DKMTEnumProcesses.
    // Collect unique PIDs (a process may appear on multiple adapters).
    auto& platform = Platform::instance();
    std::unordered_map<uint32_t, uint64_t> pid_vram_map; // pid -> vram bytes
    for (size_t i = 0; i < platform.GetDeviceCount(); ++i) {
        std::vector<thunk_proxy::GpuProcessInfo> gplist;
        if (platform.GetDevice(i)->EnumGpuProcesses(&gplist) != ErrorCode::Success)
            continue;
        for (const auto& gp : gplist) {
            // accumulate VRAM across adapters for the same process
            pid_vram_map[gp.win_pid] += gp.vram_usage_bytes;
        }
    }

    const auto total = static_cast<uint32_t>(pid_vram_map.size());

    if (procs == nullptr) {
        *num_items = total;
        return AMDSMI_STATUS_SUCCESS;
    }

    uint32_t written = 0;
    for (const auto& [pid, vram_bytes] : pid_vram_map) {
        if (written >= *num_items)
            break;
        amdsmi_process_info_t& out = procs[written++];
        out = {};
        out.process_id   = pid;
        out.vram_usage   = vram_bytes / (1024ULL * 1024ULL); // bytes -> MB
        // cu_occupancy / sdma_usage / evicted_time: unavailable on WSL2
    }
    *num_items = total;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_compute_process_info_by_pid(uint32_t pid,
        amdsmi_process_info_t *proc) {
    AMDSMI_CHECK_INIT();

    if (proc == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto& platform = Platform::instance();
    bool found = false;
    uint64_t total_vram = 0;
    for (size_t i = 0; i < platform.GetDeviceCount(); ++i) {
        std::vector<thunk_proxy::GpuProcessInfo> gplist;
        if (platform.GetDevice(i)->EnumGpuProcesses(&gplist) != ErrorCode::Success)
            continue;
        for (const auto& gp : gplist) {
            if (gp.win_pid == pid) {
                total_vram += gp.vram_usage_bytes;
                found = true;
            }
        }
    }
    if (!found)
        return AMDSMI_STATUS_NOT_FOUND;

    *proc = {};
    proc->process_id = pid;
    proc->vram_usage = total_vram / (1024ULL * 1024ULL); // bytes -> MB
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t *dv_indices,
                                                       uint32_t *num_devices) {
    AMDSMI_CHECK_INIT();

    if (num_devices == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto& platform = Platform::instance();
    std::vector<uint32_t> matching_devs;
    for (size_t i = 0; i < platform.GetDeviceCount(); ++i) {
        std::vector<thunk_proxy::GpuProcessInfo> gplist;
        if (platform.GetDevice(i)->EnumGpuProcesses(&gplist) != ErrorCode::Success)
            continue;
        for (const auto& gp : gplist) {
            if (gp.win_pid == pid) {
                matching_devs.push_back(static_cast<uint32_t>(i));
                break; // device already matched
            }
        }
    }

    const auto found = static_cast<uint32_t>(matching_devs.size());
    if (dv_indices == nullptr) {
        *num_devices = found;
        return AMDSMI_STATUS_SUCCESS;
    }

    uint32_t written = 0;
    for (uint32_t idx : matching_devs) {
        if (written >= *num_devices) break;
        dv_indices[written++] = idx;
    }
    *num_devices = found;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t  amdsmi_get_gpu_ecc_count(amdsmi_processor_handle processor_handle,
                        amdsmi_gpu_block_t block, amdsmi_error_count_t *ec) {
    // ECC error counts require KFD sysfs, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t  amdsmi_get_gpu_ecc_enabled(amdsmi_processor_handle processor_handle,
                                                    uint64_t *enabled_blocks) {
    // ECC block enable mask requires KFD sysfs, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t  amdsmi_get_gpu_ecc_status(amdsmi_processor_handle processor_handle,
                                amdsmi_gpu_block_t block,
                                amdsmi_ras_err_state_t *state) {
    // ECC block status requires KFD sysfs, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_metrics_header_info(amdsmi_processor_handle processor_handle,
                amd_metrics_table_header_t *header_value)
{
    AMDSMI_CHECK_INIT();
    // nullptr api supported
    if (header_value != nullptr) {
        *header_value = amd_metrics_table_header_t{};  // Use a default initializer for the struct
        header_value->structure_size = 0x78;//sizeof(gpu_metrics_v1_3);
        header_value->format_revision = 1;
        header_value->content_revision = 3;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t  amdsmi_get_gpu_partition_metrics_info(
        amdsmi_processor_handle processor_handle,
        amdsmi_gpu_metrics_t *pgpu_metrics) {
    // Partition metrics requires KFD/rsmi, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_get_gpu_metrics_info(
        amdsmi_processor_handle processor_handle,
        amdsmi_gpu_metrics_t *pgpu_metrics) {
    AMDSMI_CHECK_INIT();
    if (pgpu_metrics == nullptr)
        return AMDSMI_STATUS_INVAL;

    *pgpu_metrics = amdsmi_gpu_metrics_t{};

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    GpuMetricsInfo m{};
    auto code = device->QueryGpuMetricsInfo(&m);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    // Helper: saturate uint32 to uint16 (UINT32_MAX sentinel → 0 for "unavailable").
    auto to16 = [](uint32_t v) -> uint16_t {
        if (v == UINT32_MAX) return 0;
        return static_cast<uint16_t>(v > 0xFFFFu ? 0xFFFFu : v);
    };

    pgpu_metrics->temperature_edge     = to16(m.temperature_edge);
    pgpu_metrics->temperature_hotspot  = to16(m.temperature_hotspot);
    pgpu_metrics->temperature_mem      = to16(m.temperature_mem);
    pgpu_metrics->average_gfx_activity = to16(m.average_gfx_activity);
    pgpu_metrics->average_umc_activity = to16(m.average_umc_activity);
    pgpu_metrics->current_socket_power = to16(m.current_socket_power);
    pgpu_metrics->current_gfxclk      = to16(m.current_gfxclk);
    pgpu_metrics->current_socclk      = to16(m.current_socclk);
    pgpu_metrics->current_fan_speed   = to16(m.current_fan_speed);
    pgpu_metrics->voltage_soc         = to16(m.voltage_soc);
    pgpu_metrics->voltage_gfx         = to16(m.voltage_gfx);
    pgpu_metrics->voltage_mem         = to16(m.voltage_mem);
    // Propagate to per-instance arrays (index 0 = first/only instance on WDDM).
    pgpu_metrics->current_gfxclks[0]  = pgpu_metrics->current_gfxclk;
    pgpu_metrics->current_socclks[0]  = pgpu_metrics->current_socclk;

    return AMDSMI_STATUS_SUCCESS;
}


amdsmi_status_t amdsmi_get_gpu_pm_metrics_info(
                      amdsmi_processor_handle processor_handle,
                      amdsmi_name_value_t** pm_metrics,
                      uint32_t *num_of_metrics) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_pm_metrics_info_get, processor_handle, 0,
                    reinterpret_cast<rsmi_name_value_t**>(pm_metrics),
                    num_of_metrics);
}

amdsmi_status_t amdsmi_get_gpu_reg_table_info(
                      amdsmi_processor_handle processor_handle,
                      amdsmi_reg_type_t reg_type,
                      amdsmi_name_value_t** reg_metrics,
                      uint32_t *num_of_metrics) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_reg_table_info_get, processor_handle, 0,
                    static_cast<rsmi_reg_type_t>(reg_type),
                    reinterpret_cast<rsmi_name_value_t**>(reg_metrics),
                    num_of_metrics);
}

void amdsmi_free_name_value_pairs(void *p) {
    if (p)
        free(p);
    return;
}

amdsmi_status_t
amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle,
                          uint32_t sensor_ind,
                          amdsmi_power_cap_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;

    amd::smi::AMDSmiGPUDevice* gpudevice = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpudevice);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    amdsmi_status_t status;

    status = get_gpu_device_from_handle(processor_handle, &gpudevice);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    // Ignore errors to get as much as possible info.
    memset(info, 0, sizeof(amdsmi_power_cap_info_t));

    int dpm = 0;
    auto smi_power_cap_status = rsmi_wrapper(rsmi_dev_power_cap_get, processor_handle, 0,
                sensor_ind, &(info->power_cap));

    status = smi_amdgpu_get_ranges(gpudevice, AMDSMI_CLK_TYPE_GFX,
            NULL, NULL, &dpm, NULL);
    info->dpm_cap = dpm;

    // Get other information from rocm-smi
    status = rsmi_wrapper(rsmi_dev_power_cap_default_get, processor_handle, 0,
                          sensor_ind, &(info->default_power_cap));

    status = rsmi_wrapper(rsmi_dev_power_cap_range_get, processor_handle, 0,
                          sensor_ind, &(info->max_power_cap), &(info->min_power_cap));

    return smi_power_cap_status;
}

amdsmi_status_t
amdsmi_set_power_cap(amdsmi_processor_handle processor_handle,
            uint32_t sensor_ind, uint64_t cap) {

    return rsmi_wrapper(rsmi_dev_power_cap_set, processor_handle, 0,
            sensor_ind, cap);
}

amdsmi_status_t
amdsmi_get_supported_power_cap(amdsmi_processor_handle processor_handle, uint32_t *sensor_count,
                                 uint32_t *sensor_inds, amdsmi_power_cap_type_t *sensor_types) {
    AMDSMI_CHECK_INIT();
    if (!sensor_count || !sensor_inds || !sensor_types) {
        return AMDSMI_STATUS_INVAL;
    }

    return rsmi_wrapper(rsmi_dev_supported_power_cap_get, processor_handle, 0,
                    sensor_count, sensor_inds,
                    reinterpret_cast<rsmi_power_cap_type_t*>(sensor_types));
}

amdsmi_status_t
amdsmi_get_gpu_power_profile_presets(amdsmi_processor_handle processor_handle,
                        uint32_t sensor_ind,
                        amdsmi_power_profile_status_t *status) {
    AMDSMI_CHECK_INIT();
    // nullptr api supported

    // Bare Metal and passthrough only feature
    amdsmi_virtualization_mode_t virt_mode;
    if (amdsmi_get_gpu_virtualization_mode(processor_handle, &virt_mode) == AMDSMI_STATUS_SUCCESS) {
        if (virt_mode == AMDSMI_VIRTUALIZATION_MODE_GUEST) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
        }
    }

    return rsmi_wrapper(rsmi_dev_power_profile_presets_get, processor_handle, 0,
                    sensor_ind, reinterpret_cast<rsmi_power_profile_status_t*>(status));
}

amdsmi_status_t amdsmi_set_gpu_perf_determinism_mode(
            amdsmi_processor_handle processor_handle, uint64_t clkvalue) {
    return rsmi_wrapper(rsmi_perf_determinism_mode_set, processor_handle, 0,
                clkvalue);
}

amdsmi_status_t
amdsmi_set_gpu_power_profile(amdsmi_processor_handle processor_handle,
        uint32_t reserved, amdsmi_power_profile_preset_masks_t profile) {

    // Bare Metal and passthrough only feature
    amdsmi_virtualization_mode_t virt_mode;
    if (amdsmi_get_gpu_virtualization_mode(processor_handle, &virt_mode) == AMDSMI_STATUS_SUCCESS) {
        if (virt_mode == AMDSMI_VIRTUALIZATION_MODE_GUEST) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
        }
    }

    return rsmi_wrapper(rsmi_dev_power_profile_set, processor_handle, 0,
                reserved,
                static_cast<rsmi_power_profile_preset_masks_t>(profile));
}

amdsmi_status_t amdsmi_get_gpu_perf_level(amdsmi_processor_handle processor_handle,
                                        amdsmi_dev_perf_level_t *perf) {
    AMDSMI_CHECK_INIT();
    if (!perf) {
        return AMDSMI_STATUS_INVAL;
    }

    return rsmi_wrapper(rsmi_dev_perf_level_get, processor_handle, 0,
                    reinterpret_cast<rsmi_dev_perf_level_t*>(perf));
}

amdsmi_status_t
 amdsmi_set_gpu_perf_level(amdsmi_processor_handle processor_handle,
                amdsmi_dev_perf_level_t perf_lvl) {
    return rsmi_wrapper(rsmi_dev_perf_level_set_v1, processor_handle, 0,
                    static_cast<rsmi_dev_perf_level_t>(perf_lvl));
}

amdsmi_status_t  amdsmi_set_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle,
                uint64_t bw_bitmask) {

    // Bare Metal and passthrough only feature
    amdsmi_virtualization_mode_t virt_mode;
    if (amdsmi_get_gpu_virtualization_mode(processor_handle, &virt_mode) == AMDSMI_STATUS_SUCCESS) {
        if (virt_mode == AMDSMI_VIRTUALIZATION_MODE_GUEST) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
        }
    }

    return rsmi_wrapper(rsmi_dev_pci_bandwidth_set, processor_handle, 0,
                        bw_bitmask);
}

amdsmi_status_t amdsmi_get_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle,
            amdsmi_pcie_bandwidth_t *bandwidth) {
    if (bandwidth == nullptr) return AMDSMI_STATUS_INVAL;
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;

    PCIeInfo pi{};
    auto code = device->QueryPCIeInfo(&pi);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);

    // WSL2 does not expose the full PCIe DPM table (pp_dpm_pcie).
    // Synthesise a 1-or-2-entry table from the static max and current values.
    memset(bandwidth, 0, sizeof(*bandwidth));
    uint32_t n = 0;
    // Entry 0: max speed / max width (always available from KMD caps)
    if (pi.max_pcie_speed > 0) {
        bandwidth->transfer_rate.frequency[n] = static_cast<uint64_t>(pi.max_pcie_speed) * 1000000ULL;
        bandwidth->lanes[n] = pi.max_pcie_width;
        ++n;
    }
    // Entry 1: current speed / current width (from PMLog BUS_SPEED sensor)
    if (pi.pcie_speed > 0 && pi.pcie_speed != pi.max_pcie_speed) {
        bandwidth->transfer_rate.frequency[n] = static_cast<uint64_t>(pi.pcie_speed) * 1000000ULL;
        bandwidth->lanes[n] = pi.pcie_width;
        ++n;
    }
    if (n == 0) return AMDSMI_STATUS_NOT_SUPPORTED;
    bandwidth->transfer_rate.num_supported = n;
    // Mark the current speed entry as active
    bandwidth->transfer_rate.current = (pi.pcie_speed > 0 && pi.pcie_speed != pi.max_pcie_speed) ? 1 : 0;
    return AMDSMI_STATUS_SUCCESS;
}

// TODO(bliu): other frequencies in amdsmi_clk_type_t
amdsmi_status_t  amdsmi_get_clk_freq(amdsmi_processor_handle processor_handle,
                               amdsmi_clk_type_t clk_type, amdsmi_frequencies_t *f) {
    AMDSMI_CHECK_INIT();
    // nullptr is explicitly allowed; caller passes nullptr to check support

    // Map amdsmi_clk_type_t to our internal clk_type integer used by QueryClockInfo.
    // QueryClockInfo supports GFX/SYS (0) and MEM (4) via PMLog; all others
    // are not available on WSL2 (no sysfs pp_dpm_* nodes) → NOT_SUPPORTED.
    uint32_t internal_clk;
    switch (clk_type) {
        case AMDSMI_CLK_TYPE_SYS:   internal_clk = 0; break;  // GFX_CLK
        case AMDSMI_CLK_TYPE_MEM:   internal_clk = 4; break;  // MEM_CLK
        default:
            return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    if (f == nullptr) return AMDSMI_STATUS_INVAL;

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;

    ClockInfo ci{};
    auto code = device->QueryClockInfo(internal_clk, &ci);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    if (ci.clk == 0 && ci.min_clk == 0 && ci.max_clk == 0)
        return AMDSMI_STATUS_NOT_SUPPORTED;

    memset(f, 0, sizeof(*f));
    // Populate a single-entry frequency table with the current clock.
    // WSL2 does not expose the full DPM table; current clock is what PMLog provides.
    f->num_supported = 1;
    f->current       = 0;
    f->has_deep_sleep = 0;
    f->frequency[0]  = static_cast<uint64_t>(ci.clk) * 1000000ULL;  // MHz → Hz
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t  amdsmi_set_clk_freq(amdsmi_processor_handle processor_handle,
                         amdsmi_clk_type_t clk_type, uint64_t freq_bitmask) {
    // Clock frequency DPM table write requires sysfs pp_dpm_*, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_soc_pstate(amdsmi_processor_handle processor_handle,
                         uint32_t policy) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_soc_pstate_set, processor_handle, 0,
                    policy);
}

amdsmi_status_t amdsmi_get_soc_pstate(amdsmi_processor_handle processor_handle,
                         amdsmi_dpm_policy_t* policy) {
    AMDSMI_CHECK_INIT();

    if (policy == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Initialize output structure to zero
    memset(policy, 0, sizeof(*policy));

    // Use rsmi structure with correct size (32-byte description fields)
    rsmi_dpm_policy_t rsmi_policy = {};
    amdsmi_status_t ret = rsmi_wrapper(rsmi_dev_soc_pstate_get, processor_handle, 0,
                    &rsmi_policy);
    
    if (ret != AMDSMI_STATUS_SUCCESS) {
        return ret;
    }

    // Copy data from rsmi structure to amdsmi structure field-by-field
    // to handle the different structure sizes properly
    policy->num_supported = rsmi_policy.num_supported;
    policy->current = rsmi_policy.current;
    
    for (uint32_t i = 0; i < rsmi_policy.num_supported && i < AMDSMI_MAX_NUM_PM_POLICIES; i++) {
        policy->policies[i].policy_id = rsmi_policy.policies[i].policy_id;
        snprintf(policy->policies[i].policy_description, AMDSMI_MAX_STRING_LENGTH - 1, "%s",
                rsmi_policy.policies[i].policy_description);
        policy->policies[i].policy_description[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_xgmi_plpd(amdsmi_processor_handle processor_handle,
                         uint32_t policy) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_xgmi_plpd_set, processor_handle, 0,
                    policy);
}

amdsmi_status_t amdsmi_get_xgmi_plpd(amdsmi_processor_handle processor_handle,
                         amdsmi_dpm_policy_t* policy) {
    AMDSMI_CHECK_INIT();

    if (policy == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Initialize output structure to zero
    memset(policy, 0, sizeof(*policy));

    // Use rsmi structure with correct size (32-byte description fields)
    rsmi_dpm_policy_t rsmi_policy = {};
    amdsmi_status_t ret = rsmi_wrapper(rsmi_dev_xgmi_plpd_get, processor_handle, 0,
                    &rsmi_policy);
    
    if (ret != AMDSMI_STATUS_SUCCESS) {
        return ret;
    }

    // Copy data from rsmi structure to amdsmi structure field-by-field
    // to handle the different structure sizes properly
    policy->num_supported = rsmi_policy.num_supported;
    policy->current = rsmi_policy.current;
    
    for (uint32_t i = 0; i < rsmi_policy.num_supported && i < AMDSMI_MAX_NUM_PM_POLICIES; i++) {
        policy->policies[i].policy_id = rsmi_policy.policies[i].policy_id;
        snprintf(policy->policies[i].policy_description, AMDSMI_MAX_STRING_LENGTH - 1, "%s",
                rsmi_policy.policies[i].policy_description);
        policy->policies[i].policy_description[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_process_isolation(amdsmi_processor_handle processor_handle,
                             uint32_t* pisolate) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_process_isolation_get, processor_handle, 0,
                    pisolate);
}

amdsmi_status_t amdsmi_set_gpu_process_isolation(amdsmi_processor_handle processor_handle,
                             uint32_t pisolate) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_process_isolation_set, processor_handle, 0,
                   pisolate);
}

amdsmi_status_t amdsmi_clean_gpu_local_data(amdsmi_processor_handle processor_handle) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_gpu_run_cleaner_shader, processor_handle, 0);
}

amdsmi_status_t
amdsmi_get_gpu_memory_reserved_pages(amdsmi_processor_handle processor_handle,
                                    uint32_t *num_pages,
                                    amdsmi_retired_page_record_t *records) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t amdsmi_get_gpu_memory_total(amdsmi_processor_handle processor_handle,
                amdsmi_memory_type_t mem_type, uint64_t *total) {
    AMDSMI_CHECK_INIT();

    if (total == nullptr)
        return AMDSMI_STATUS_INVAL;

    if (mem_type < AMDSMI_MEM_TYPE_FIRST || mem_type > AMDSMI_MEM_TYPE_LAST)
        return AMDSMI_STATUS_INVAL;

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto code = device->QueryMemoryTotal(static_cast<uint32_t>(mem_type), total);
    return translateCodeToSmiStatus(code);
}
amdsmi_status_t amdsmi_get_gpu_memory_usage(amdsmi_processor_handle processor_handle,
            amdsmi_memory_type_t mem_type, uint64_t *used) {
    AMDSMI_CHECK_INIT();

    if (used == nullptr)
        return AMDSMI_STATUS_INVAL;

    if (mem_type < AMDSMI_MEM_TYPE_FIRST || mem_type > AMDSMI_MEM_TYPE_LAST)
        return AMDSMI_STATUS_INVAL;

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto code = device->QueryMemoryUsage(static_cast<uint32_t>(mem_type), used);
    return translateCodeToSmiStatus(code);
}

amdsmi_status_t amdsmi_get_gpu_overdrive_level(
            amdsmi_processor_handle processor_handle,
            uint32_t *od) {
    // Overdrive requires sysfs pp_od_clk_voltage, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_mem_overdrive_level(
            amdsmi_processor_handle processor_handle,
            uint32_t *od) {
    // Overdrive requires sysfs pp_od_clk_voltage, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_set_gpu_overdrive_level(
            amdsmi_processor_handle processor_handle, uint32_t od) {
    // Overdrive requires sysfs pp_od_clk_voltage, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t  amdsmi_get_gpu_pci_replay_counter(
            amdsmi_processor_handle processor_handle, uint64_t *counter) {
    return rsmi_wrapper(rsmi_dev_pci_replay_counter_get, processor_handle, 0,
                        counter);
}
amdsmi_status_t amdsmi_get_gpu_pci_throughput(
        amdsmi_processor_handle processor_handle,
        uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz) {
    return rsmi_wrapper(rsmi_dev_pci_throughput_get, processor_handle, 0,
            sent, received, max_pkt_sz);
}

amdsmi_status_t  amdsmi_get_gpu_od_volt_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_od_volt_freq_data_t *odv) {
    // OD voltage curve requires sysfs pp_od_clk_voltage, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_get_gpu_od_volt_curve_regions(
                    amdsmi_processor_handle processor_handle,
                    uint32_t *num_regions, amdsmi_freq_volt_region_t *buffer) {
    // OD voltage curve requires sysfs pp_od_clk_voltage, not available on WSL2
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle,
                            amdsmi_voltage_type_t sensor_type,
                            amdsmi_voltage_metric_t metric, int64_t *voltage) {
    // Check support first so nullptr path returns NOT_SUPPORTED consistently
    if (metric != AMDSMI_VOLT_CURRENT) return AMDSMI_STATUS_NOT_SUPPORTED;
    if (sensor_type != AMDSMI_VOLT_TYPE_VDDGFX) return AMDSMI_STATUS_NOT_SUPPORTED;
    if (voltage == nullptr) return AMDSMI_STATUS_INVAL;
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    GpuMetricsInfo m{};
    auto code = device->QueryGpuMetricsInfo(&m);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    if (m.voltage_gfx == UINT32_MAX) return AMDSMI_STATUS_NOT_SUPPORTED;
    *voltage = static_cast<int64_t>(m.voltage_gfx);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t  amdsmi_set_gpu_od_clk_info(amdsmi_processor_handle processor_handle,
                                        amdsmi_freq_ind_t level,
                                       uint64_t clkvalue,
                                       amdsmi_clk_type_t clkType) {
    return rsmi_wrapper(rsmi_dev_od_clk_info_set, processor_handle, 0,
                static_cast<rsmi_freq_ind_t>(level), clkvalue,
                static_cast<rsmi_clk_type_t>(clkType));
}

amdsmi_status_t  amdsmi_set_gpu_od_volt_info(amdsmi_processor_handle processor_handle,
                    uint32_t vpoint, uint64_t clkvalue, uint64_t voltvalue) {
    return rsmi_wrapper(rsmi_dev_od_volt_info_set, processor_handle, 0,
                vpoint, clkvalue, voltvalue);
}

amdsmi_status_t amdsmi_set_gpu_clk_range(amdsmi_processor_handle processor_handle,
                                    uint64_t minclkvalue,
                                    uint64_t maxclkvalue,
                                    amdsmi_clk_type_t clkType) {

    // Bare Metal and passthrough only feature
    amdsmi_virtualization_mode_t virt_mode;
    if (amdsmi_get_gpu_virtualization_mode(processor_handle, &virt_mode) == AMDSMI_STATUS_SUCCESS) {
        if (virt_mode == AMDSMI_VIRTUALIZATION_MODE_GUEST) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
        }
    }

    return rsmi_wrapper(rsmi_dev_clk_range_set, processor_handle, 0,
                minclkvalue, maxclkvalue,
                static_cast<rsmi_clk_type_t>(clkType));
}

amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle processor_handle,
                                         amdsmi_clk_type_t clk_type,
                                          amdsmi_clk_limit_type_t limit_type,
                                          uint64_t clk_value) {
    return rsmi_wrapper(rsmi_dev_clk_extremum_set, processor_handle, 0,
                static_cast<rsmi_freq_ind_t>(limit_type),
                clk_value,
                static_cast<rsmi_clk_type_t>(clk_type));
}

amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle processor_handle) {
    std::ostringstream ss;
    amdsmi_status_t ret = rsmi_wrapper(rsmi_dev_gpu_reset, processor_handle, 0);
    ss << __PRETTY_FUNCTION__
       << " | Returning: " << smi_amdgpu_get_status_string(ret, false);
    LOG_INFO(ss);
    return ret;
}

amdsmi_status_t amdsmi_gpu_driver_reload(void) {
    std::ostringstream ss;
    AMDSMI_CHECK_INIT();

    // Attempting to speed up processing time
    bool is_logger_enabled = ROCmLogging::Logger::getInstance()->isLoggerEnabled();
    if (is_logger_enabled) {
        ss << __PRETTY_FUNCTION__ << " | ======= start =======";
        LOG_INFO(ss);
    }
    rsmi_status_t ret = rsmi_dev_amdgpu_driver_reload();
    amdsmi_status_t amdsmi_status = amd::smi::rsmi_to_amdsmi_status(ret);
    if (is_logger_enabled) {
        ss << __PRETTY_FUNCTION__
           << " | Returning: " << smi_amdgpu_get_status_string(amdsmi_status, false);
        LOG_INFO(ss);
    }
    return amdsmi_status;
}

amdsmi_status_t amdsmi_get_gpu_busy_percent(amdsmi_processor_handle processor_handle,
                                            uint32_t *gpu_busy_percent) {
    AMDSMI_CHECK_INIT();
    if (gpu_busy_percent == nullptr) return AMDSMI_STATUS_INVAL;
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    GpuActivity activity{};
    auto code = device->QueryGpuActivity(&activity);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    *gpu_busy_percent = activity.gfx_activity;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_utilization_count(amdsmi_processor_handle processor_handle,
                amdsmi_utilization_counter_t utilization_counters[],
                uint32_t count,
                uint64_t *timestamp) {
    AMDSMI_CHECK_INIT();
    if (utilization_counters == nullptr || timestamp == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;

    GpuActivity activity{};
    auto code = device->QueryGpuActivity(&activity);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);

    // Use a monotonic clock as timestamp (microseconds since epoch)
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *timestamp = static_cast<uint64_t>(ts.tv_sec) * 1000000ULL
                 + static_cast<uint64_t>(ts.tv_nsec) / 1000ULL;

    for (uint32_t i = 0; i < count; ++i) {
        utilization_counters[i].fine_value_count = 0;
        switch (utilization_counters[i].type) {
            case AMDSMI_COARSE_GRAIN_GFX_ACTIVITY:
            case AMDSMI_FINE_GRAIN_GFX_ACTIVITY:
                utilization_counters[i].value = activity.gfx_activity;
                break;
            case AMDSMI_COARSE_GRAIN_MEM_ACTIVITY:
            case AMDSMI_FINE_GRAIN_MEM_ACTIVITY:
                utilization_counters[i].value = activity.umc_activity;
                break;
            case AMDSMI_COARSE_DECODER_ACTIVITY:
            case AMDSMI_FINE_DECODER_ACTIVITY:
                utilization_counters[i].value = activity.mm_activity;
                break;
            default:
                utilization_counters[i].value = 0;
                break;
        }
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_energy_count(amdsmi_processor_handle processor_handle,
            uint64_t *energy_accumulator, float *counter_resolution, uint64_t *timestamp) {
    if (energy_accumulator == nullptr || counter_resolution == nullptr || timestamp == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_bdf_id(
        amdsmi_processor_handle processor_handle, uint64_t *bdfid) {
    return rsmi_wrapper(rsmi_dev_pci_id_get, processor_handle, 0,
            bdfid);
}

amdsmi_status_t amdsmi_get_gpu_topo_numa_affinity(
    amdsmi_processor_handle processor_handle, int32_t *numa_node) {
    if (!numa_node) {
        return AMDSMI_STATUS_INVAL;
    }
    return rsmi_wrapper(rsmi_topo_numa_affinity_get, processor_handle, 0,
            numa_node);
}

amdsmi_status_t amdsmi_get_gpu_topo_cpu_affinity(amdsmi_processor_handle processor_handle,
                                           unsigned int *cpu_aff_length, char *cpu_aff_data) {
    AMDSMI_CHECK_INIT();

    if (cpu_aff_length == nullptr || cpu_aff_data == nullptr || cpu_aff_length == nullptr ||
        *cpu_aff_length < AMDSMI_MAX_STRING_LENGTH) {
        return AMDSMI_STATUS_INVAL;
    }

    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    status = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    std::string cpu_affinity;
    status = gpu_device->amdgpu_query_cpu_affinity(cpu_affinity);
    if (status != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__
           << " | Getting cpu_affinity info failed. Return code:: " << status;
        LOG_INFO(ss);
        return status;
    }
    snprintf(cpu_aff_data, *cpu_aff_length - 1,"%s", cpu_affinity.c_str());
    return status;
}

#ifdef BRCM_NIC
amdsmi_status_t amdsmi_get_nic_gpu_topo_info(amdsmi_processor_handle nic_processor_handle, 
                    amdsmi_processor_handle gpu_processor_handle, size_t *topo_info_length, char *topo_info) {
    std::ostringstream ss;
    AMDSMI_CHECK_INIT();
    if (topo_info_length == nullptr || topo_info == nullptr || topo_info_length == nullptr ||
        *topo_info_length < AMDSMI_MAX_STRING_LENGTH) {
        return AMDSMI_STATUS_INVAL;
    }
    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    amd::smi::AMDSmiNICDevice *nic_device = nullptr;
    amdsmi_status_t r = get_nic_device_from_handle(nic_processor_handle, &nic_device);
    if (status != AMDSMI_STATUS_SUCCESS) {
        ss << __PRETTY_FUNCTION__
           << " | Received invalid NIC handler. Return code: " << status;
        LOG_INFO(ss);
        return status;
    }
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    status = get_gpu_device_from_handle(gpu_processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS) {
        ss << __PRETTY_FUNCTION__
           << " | Received invalid GPU handler. Return code: " << status;
        LOG_INFO(ss);
        return status;
    }
    amdsmi_bdf_t nic_switchBdf = {};
    status = amdsmi_get_root_switch(nic_device->get_bdf(), &nic_switchBdf);
    if (status != AMDSMI_STATUS_SUCCESS) {
        ss << __PRETTY_FUNCTION__
           << " | Not able to get nic's switch bdf. Return code: " << status;
        LOG_INFO(ss);
        return status;
    }
    amdsmi_bdf_t gpu_switchBdf = {};
    status = amdsmi_get_root_switch(gpu_device->get_bdf(), &gpu_switchBdf);
    if (status != AMDSMI_STATUS_SUCCESS) {
        ss << __PRETTY_FUNCTION__
           << " | Not able to get gpu's switch bdf. Return code: " << status;
        LOG_INFO(ss);
        return status;
    }
    int32_t gpu_numa_node;
    status = rsmi_wrapper(rsmi_topo_numa_affinity_get, gpu_processor_handle, 0, &gpu_numa_node);
    if (status != AMDSMI_STATUS_SUCCESS) {
        ss << __PRETTY_FUNCTION__
           << " | Not able to get gpu's NUMA. Return code: " << status;
        LOG_INFO(ss);
        return status;
    }
    int32_t nic_numa_node;
    status = nic_device->amd_query_nic_numa_affinity(&nic_numa_node);
    if (nic_numa_node == 65535) {
        ss << __PRETTY_FUNCTION__
           << " | Not able to get nic's NUMA. Return code: " << status;
        LOG_INFO(ss);
        return status;
    }
    if(gpu_numa_node != nic_numa_node) {
        snprintf(topo_info, *topo_info_length - 1, "%s", "X-NUMA");
        return AMDSMI_STATUS_SUCCESS;
    }
    if(gpu_numa_node == nic_numa_node) {
        snprintf(topo_info, *topo_info_length - 1, "%s", "NUMA");
        if ((gpu_switchBdf.bus_number == nic_switchBdf.bus_number) &&
                (gpu_switchBdf.device_number == nic_switchBdf.device_number) &&
                (gpu_switchBdf.domain_number == nic_switchBdf.domain_number) &&
                (gpu_switchBdf.function_number == nic_switchBdf.function_number)) { 
            snprintf(topo_info, *topo_info_length - 1, "%s", "PCIe");
        }
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_root_switch(amdsmi_bdf_t devicehBdf, amdsmi_bdf_t *switchBdf) {
    AMDSMI_CHECK_INIT();
    amdsmi_status_t status = get_lspci_root_switch(devicehBdf, switchBdf);
    return status;
}

amdsmi_status_t amdsmi_get_nic_topo_numa_affinity(
    amdsmi_processor_handle processor_handle, int32_t *numa_node) {
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiNICDevice *nic_device = nullptr;
    amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;
    
    return nic_device->amd_query_nic_numa_affinity(numa_node);
}

amdsmi_status_t amdsmi_get_nic_topo_cpu_affinity(amdsmi_processor_handle processor_handle,
                                           unsigned int *cpu_aff_length, char *cpu_aff_data) {
    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    AMDSMI_CHECK_INIT();
    if (cpu_aff_length == nullptr || cpu_aff_data == nullptr || cpu_aff_length == nullptr ||
        *cpu_aff_length < AMDSMI_MAX_STRING_LENGTH) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiNICDevice *nic_device = nullptr;
    amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    std::string cpu_affinity;
    status = nic_device->amd_query_nic_cpu_affinity(cpu_affinity);
    if (status != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__
           << " | Getting cpu_affinity info failed. Return code: " << status;
        LOG_INFO(ss);
        return status;
    }
    snprintf(cpu_aff_data, *cpu_aff_length - 1, "%s", cpu_affinity.c_str());
    return status;
}

amdsmi_status_t amdsmi_get_switch_topo_numa_affinity(
    amdsmi_processor_handle processor_handle, int32_t *numa_node) {
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiSWITCHDevice *switch_device = nullptr;
    amdsmi_status_t r = get_switch_device_from_handle(processor_handle, &switch_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;
    
    return switch_device->amd_query_switch_numa_affinity(numa_node);
}

amdsmi_status_t amdsmi_get_switch_topo_cpu_affinity(amdsmi_processor_handle processor_handle,
                                           size_t *cpu_aff_length, char *cpu_aff_data) {
    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    AMDSMI_CHECK_INIT();
    if (cpu_aff_length == nullptr || cpu_aff_data == nullptr || cpu_aff_length == nullptr ||
        *cpu_aff_length < AMDSMI_MAX_STRING_LENGTH) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiSWITCHDevice *switch_device = nullptr;
    amdsmi_status_t r = get_switch_device_from_handle(processor_handle, &switch_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    std::string cpu_affinity;
    status = switch_device->amd_query_switch_cpu_affinity(cpu_affinity);
    if (status != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__
           << " | Getting cpu_affinity info failed. Return code: " << status;
        LOG_INFO(ss);
        return status;
    }
    snprintf(cpu_aff_data, *cpu_aff_length - 1, "%s", cpu_affinity.c_str());
    return status;
}
#endif//BRCM_NIC
amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t *version) {
    if (version == nullptr)
        return AMDSMI_STATUS_INVAL;

    version->major = AMDSMI_LIB_VERSION_MAJOR;
    version->minor = AMDSMI_LIB_VERSION_MINOR;
    version->release = AMDSMI_LIB_VERSION_RELEASE;
    version->build = AMDSMI_LIB_VERSION_STRING;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_vbios_info(amdsmi_processor_handle processor_handle, amdsmi_vbios_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    memset(info, 0, sizeof(*info));
    wsl::thunk::VBiosInfo vi = {};
    auto code = device->QueryVBiosInfo(&vi);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    strncpy(info->name,        vi.name,        sizeof(info->name) - 1);
    strncpy(info->build_date,  vi.build_date,  sizeof(info->build_date) - 1);
    strncpy(info->part_number, vi.part_number, sizeof(info->part_number) - 1);
    strncpy(info->version,     vi.version,     sizeof(info->version) - 1);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_activity(amdsmi_processor_handle processor_handle, amdsmi_engine_usage_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    GpuActivity activity{};
    auto code = device->QueryGpuActivity(&activity);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    info->gfx_activity = activity.gfx_activity;
    info->umc_activity = activity.umc_activity;
    info->mm_activity  = activity.mm_activity;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_is_gpu_power_management_enabled(amdsmi_processor_handle processor_handle, bool *enabled) {
    if (enabled == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    *enabled = false;

    amd::smi::AMDSmiGPUDevice * gpu_device = nullptr;
    amdsmi_status_t status;

    status = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    status = smi_amdgpu_is_gpu_power_management_enabled(gpu_device, enabled);

    return status;
}

amdsmi_status_t
amdsmi_get_clock_info(amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    memset(info, 0, sizeof(*info));

    if (clk_type > AMDSMI_CLK_TYPE__MAX) {
        return AMDSMI_STATUS_INVAL;
    }

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    wsl::thunk::ClockInfo clock_info{};
    auto code = device->QueryClockInfo(static_cast<uint32_t>(clk_type), &clock_info);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    info->clk            = clock_info.clk;
    info->min_clk        = clock_info.min_clk;
    info->max_clk        = clock_info.max_clk;
    info->clk_locked     = clock_info.clk_locked;
    info->clk_deep_sleep = clock_info.clk_deep_sleep;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_ras_block_features_enabled(amdsmi_processor_handle processor_handle, amdsmi_gpu_block_t block, amdsmi_ras_err_state_t *state) {
    AMDSMI_CHECK_INIT();

    if (state == nullptr || block > AMDSMI_GPU_BLOCK_LAST) {
        return AMDSMI_STATUS_INVAL;
    }

    uint64_t features_mask = 0;
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    amdsmi_status_t status;
    status = smi_amdgpu_get_enabled_blocks(gpu_device, &features_mask);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    *state = (features_mask & block) ? AMDSMI_RAS_ERR_STATE_ENABLED : AMDSMI_RAS_ERR_STATE_DISABLED;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_bad_page_info(amdsmi_processor_handle processor_handle, uint32_t *num_pages, amdsmi_retired_page_record_t *info) {
    AMDSMI_CHECK_INIT();

    if (num_pages == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    amdsmi_status_t status;
    status = smi_amdgpu_get_bad_page_info(gpu_device, num_pages, info);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_bad_page_threshold(amdsmi_processor_handle processor_handle, uint32_t *threshold) {
    AMDSMI_CHECK_INIT();

    if (threshold == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    amdsmi_status_t status;
    status = smi_amdgpu_get_bad_page_threshold(gpu_device, threshold);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_gpu_validate_ras_eeprom(amdsmi_processor_handle processor_handle) {
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    return smi_amdgpu_validate_ras_eeprom(gpu_device);
}

amdsmi_status_t amdsmi_get_gpu_ras_feature_info(
  amdsmi_processor_handle processor_handle, amdsmi_ras_feature_t *ras_feature) {
    AMDSMI_CHECK_INIT();

    if (ras_feature == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    memset(ras_feature, 0, sizeof(*ras_feature));

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_NOT_FOUND;

    RasFeature info{};
    auto code = device->QueryRasFeature(&info);
    if (code == ErrorCode::Success) {
        ras_feature->ecc_correction_schema_flag =
            (info.dram_ecc  ? 0x1u : 0u) |
            (info.sram_ecc  ? 0x2u : 0u) |
            (info.poisoning ? 0x4u : 0u);
        ras_feature->ras_info.dram_ecc  = info.dram_ecc;
        ras_feature->ras_info.sram_ecc  = info.sram_ecc;
        ras_feature->ras_info.poisoning = info.poisoning;
        ras_feature->needs_reboot       = info.needs_reboot;
    }
    return translateCodeToSmiStatus(code);
}

amdsmi_status_t
amdsmi_get_gpu_total_ecc_count(amdsmi_processor_handle processor_handle, amdsmi_error_count_t *ec) {
    AMDSMI_CHECK_INIT();

    if (ec == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t status = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    amdsmi_ras_err_state_t state = {};
    // Iterate through the ecc blocks
    for (auto block = AMDSMI_GPU_BLOCK_FIRST; block <= AMDSMI_GPU_BLOCK_LAST;
            block = (amdsmi_gpu_block_t)(block * 2)) {
        // Clear the previous ecc block counts
        amdsmi_error_count_t block_ec = {};
        // Check if the current ecc block is enabled
        status = amdsmi_get_gpu_ras_block_features_enabled(processor_handle, block, &state);
        if (status == AMDSMI_STATUS_SUCCESS && state == AMDSMI_RAS_ERR_STATE_ENABLED) {
            // Increment the total ecc counts by the ecc block counts
            status = amdsmi_get_gpu_ecc_count(processor_handle, block, &block_ec);
            if (status == AMDSMI_STATUS_SUCCESS) {
                // Increase the total ecc counts
                ec->correctable_count += block_ec.correctable_count;
                ec->uncorrectable_count += block_ec.uncorrectable_count;
                ec->deferred_count += block_ec.deferred_count;
            }
        }
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_cper_entries(
    amdsmi_processor_handle processor_handle,
    uint32_t severity_mask,
    char *cper_data,
    uint64_t *buf_size,
    amdsmi_cper_hdr_t **cper_hdrs,
    uint64_t *entry_count,
    uint64_t *cursor) {

    std::string path;
    if(amd::smi::FileExists(static_cast<char const *>(processor_handle))) {
        path = std::string(static_cast<char const *>(processor_handle));
    }
    else {

    AMDSMI_CHECK_INIT();
    if (!amd::smi::is_sudo_user()) {
        return AMDSMI_STATUS_NO_PERM;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t status = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    path = std::string("/sys/kernel/debug/dri/") +
        std::to_string(gpu_device->get_card_id()) +
        "/amdgpu_ring_cper";
    }

    return amdsmi_get_gpu_cper_entries_by_path(
        path.c_str(),
        severity_mask,
        cper_data,
        buf_size,
        cper_hdrs,
        entry_count,
        cursor,
        get_product_serial_number(processor_handle)
    );
}

amdsmi_status_t amdsmi_get_afids_from_cper(
            char* cper_buffer, uint32_t buf_size, uint64_t* afids, uint32_t* num_afids) {

    AMDSMI_CHECK_INIT();

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] begin\n";
    LOG_DEBUG(ss);

    if(!cper_buffer) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] cper_buffer should be a valid memory address\n";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }
    else if(!buf_size) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] buf_size should be greater than 0\n";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }
    else if(!afids) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] afids should be a valid memory address\n";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }
    else if(!num_afids) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] num_afids should be a valid memory address\n";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }
    else if(!*num_afids) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] num_afids should be greater than 0\n";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }

    const amdsmi_cper_hdr_t *cper = reinterpret_cast<const amdsmi_cper_hdr_t *>(cper_buffer);
    if(cper->record_length > buf_size) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] cper buffer size " << std::dec << buf_size << " is smaller than cper record length " << std::dec << cper->record_length << "\n";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_UNEXPECTED_SIZE;
    }
    else if(strncmp(cper->signature, "CPER", 4) != 0) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] cper buffer does not have the correct signature\n";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_UNEXPECTED_DATA;
    }
    uint32_t i = 0;
    for(int afid: cper_decode(cper)) {
        if(i < *num_afids) {
            afids[i] = afid;
        }
        ++i;
    }
    *num_afids = i;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_process_list(amdsmi_processor_handle processor_handle, uint32_t *max_processes, amdsmi_proc_info_t *list) {
    AMDSMI_CHECK_INIT();

    if (!max_processes)
        return AMDSMI_STATUS_INVAL;

    // Get the Device object associated with this processor handle.
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (!device)
        return AMDSMI_STATUS_INVAL;

    // Query GPU processes on this adapter via WSL2 D3DKMTEnumProcesses.
    std::vector<thunk_proxy::GpuProcessInfo> gplist;
    ErrorCode ec = device->EnumGpuProcesses(&gplist);
    if (ec == ErrorCode::UnSupported) {
        *max_processes = 0;
        return AMDSMI_STATUS_SUCCESS;
    }
    if (ec != ErrorCode::Success)
        return translateCodeToSmiStatus(ec);

    const auto total = static_cast<uint32_t>(gplist.size());

    // Caller passing max_processes==0 (or list==nullptr) requests the count.
    if (*max_processes == 0 || list == nullptr) {
        *max_processes = total;
        return AMDSMI_STATUS_SUCCESS;
    }

    const auto original_max = *max_processes;
    uint32_t written = 0;
    for (const auto& gp : gplist) {
        if (written >= original_max)
            break;
        amdsmi_proc_info_t& out = list[written++];
        out = {};
        out.pid = gp.win_pid;
        // memory_usage.vram_mem is in bytes; mem is also bytes.
        // On WSL2, cross-process queries are rejected by the kernel;
        // use UINT64_MAX / UINT32_MAX as sentinel so Python shows N/A.
        out.memory_usage.vram_mem = gp.vram_usage_bytes;  // UINT64_MAX = N/A
        out.mem                   = gp.vram_usage_bytes;
        out.memory_usage.gtt_mem  = UINT64_MAX;  // unavailable on WSL2
        out.memory_usage.cpu_mem  = UINT64_MAX;  // unavailable on WSL2
        // engine_usage, cu_occupancy, sdma_usage, evicted_time: unavailable on WSL2
        out.engine_usage.gfx = UINT64_MAX;
        out.engine_usage.enc = UINT64_MAX;
        out.cu_occupancy     = UINT32_MAX;
        out.sdma_usage       = UINT64_MAX;
        out.evicted_time     = UINT32_MAX;
        // Resolve process name via /proc/<pid>/exe (works in WSL2).
        {
            char exe_path[64];
            char exe_real[PATH_MAX] = {};
            std::snprintf(exe_path, sizeof(exe_path), "/proc/%u/exe", gp.win_pid);
            ssize_t n = readlink(exe_path, exe_real, sizeof(exe_real) - 1);
            if (n > 0) {
                // Use basename only (strip directory prefix)
                const char *base = std::strrchr(exe_real, '/');
                const char *name = base ? base + 1 : exe_real;
                std::strncpy(out.name, name, AMDSMI_MAX_STRING_LENGTH - 1);
            } else {
                std::strncpy(out.name, "N/A", AMDSMI_MAX_STRING_LENGTH - 1);
            }
        }
    }
    *max_processes = total;
    return (original_max >= total) ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_OUT_OF_RESOURCES;
}

amdsmi_status_t
amdsmi_get_power_info(amdsmi_processor_handle processor_handle, amdsmi_power_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    info->socket_power         = get_std_num_limit<decltype(info->socket_power)>();
    info->current_socket_power = get_std_num_limit<decltype(info->current_socket_power)>();
    info->average_socket_power = get_std_num_limit<decltype(info->average_socket_power)>();
    info->gfx_voltage          = get_std_num_limit<decltype(info->gfx_voltage)>();
    info->soc_voltage          = get_std_num_limit<decltype(info->soc_voltage)>();
    info->mem_voltage          = get_std_num_limit<decltype(info->mem_voltage)>();
    info->power_limit          = get_std_num_limit<decltype(info->power_limit)>();

    wsl::thunk::PowerInfo pi = {};
    auto code = device->QueryPowerInfo(&pi);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    info->current_socket_power = pi.current_socket_power;
    info->socket_power         = pi.current_socket_power;
    info->gfx_voltage          = pi.gfx_voltage;
    info->soc_voltage          = pi.soc_voltage;
    info->mem_voltage          = pi.mem_voltage;
    info->power_limit          = pi.power_limit;
    return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t amdsmi_get_gpu_driver_info(amdsmi_processor_handle processor_handle,
                amdsmi_driver_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    memset(info, 0, sizeof(*info));
    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    wsl::thunk::DriverInfo dinfo{};
    auto code = device->QueryDriverInfo(&dinfo);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    snprintf(info->driver_version, AMDSMI_MAX_STRING_LENGTH, "%s", dinfo.driver_version);
    snprintf(info->driver_date,    AMDSMI_MAX_STRING_LENGTH, "%s", dinfo.driver_date);
    snprintf(info->driver_name,    AMDSMI_MAX_STRING_LENGTH, "%s", dinfo.driver_name);
    return AMDSMI_STATUS_SUCCESS;
}

#ifdef BRCM_NIC
amdsmi_status_t amdsmi_get_nic_device_uuid(amdsmi_processor_handle processor_handle,
                                           unsigned int *uuid_length, char *uuid) {
  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  AMDSMI_CHECK_INIT();

  if (uuid_length == nullptr || uuid == nullptr || uuid_length == nullptr ||
      *uuid_length < AMDSMI_GPU_UUID_SIZE) {
    return AMDSMI_STATUS_INVAL;
  }

  amd::smi::AMDSmiNICDevice *nic_device = nullptr;
  amdsmi_status_t r = get_nic_device_from_handle(processor_handle, &nic_device);
  if (r != AMDSMI_STATUS_SUCCESS) return r;

  std::string uuidStr;
  status = nic_device->amd_query_nic_uuid(uuidStr);
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__
       << " | Getting NIC UUID failed. Return code: " << status;
    LOG_INFO(ss);
    return status;
  }
  snprintf(uuid, *uuid_length - 1, "%s", uuidStr.c_str());
  return status;
}

amdsmi_status_t amdsmi_get_switch_device_uuid(amdsmi_processor_handle processor_handle,
                                           unsigned int *uuid_length, char *uuid) {
  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  AMDSMI_CHECK_INIT();

  if (uuid_length == nullptr || uuid == nullptr || uuid_length == nullptr ||
      *uuid_length < AMDSMI_GPU_UUID_SIZE) {
    return AMDSMI_STATUS_INVAL;
  }

  amd::smi::AMDSmiSWITCHDevice *switch_device = nullptr;
  amdsmi_status_t r = get_switch_device_from_handle(processor_handle, &switch_device);
  if (r != AMDSMI_STATUS_SUCCESS) return r;

  std::string uuidStr;
  status = switch_device->amd_query_switch_uuid(uuidStr);
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__
       << " | Getting switch UUID failed. Return code: " << status;
    LOG_INFO(ss);
    return status;
  }
  snprintf(uuid, *uuid_length - 1, "%s", uuidStr.c_str());
  return status;
}
#endif//BRCM_NIC
amdsmi_status_t amdsmi_get_pcie_info(amdsmi_processor_handle processor_handle, amdsmi_pcie_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    memset(info, 0, sizeof(*info));
    auto device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    wsl::thunk::PCIeInfo pcie{};
    auto code = device->QueryPCIeInfo(&pcie);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    info->pcie_static.max_pcie_width         = pcie.max_pcie_width;
    info->pcie_static.max_pcie_speed         = pcie.max_pcie_speed;
    info->pcie_static.pcie_interface_version = pcie.pcie_interface_version;
    info->pcie_static.slot_type              = static_cast<amdsmi_card_form_factor_t>(pcie.slot_type);
    info->pcie_metric.pcie_width             = pcie.pcie_width;
    info->pcie_metric.pcie_speed             = pcie.pcie_speed;
    info->pcie_metric.pcie_bandwidth              = pcie.pcie_bandwidth;
    info->pcie_metric.pcie_replay_count           = pcie.pcie_replay_count;
    info->pcie_metric.pcie_l0_to_recovery_count   = pcie.pcie_l0_to_recovery_count;
    info->pcie_metric.pcie_replay_roll_over_count = pcie.pcie_replay_roll_over_count;
    info->pcie_metric.pcie_nak_sent_count         = pcie.pcie_nak_sent_count;
    info->pcie_metric.pcie_nak_received_count     = pcie.pcie_nak_received_count;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_xcd_counter(amdsmi_processor_handle processor_handle,
                                           uint16_t *xcd_count) {
    if (xcd_count == nullptr) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_processor_handle_from_bdf(amdsmi_bdf_t bdf,
                amdsmi_processor_handle* processor_handle)
{
    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto& platform = Platform::instance();
    uint32_t device_count = static_cast<uint32_t>(platform.GetDeviceCount());
    for (uint32_t i = 0; i < device_count; i++) {
        Device *pdev = platform.GetDevice(i);
        if (pdev == nullptr)
            return AMDSMI_STATUS_API_FAILED;

        wsl::thunk::BdfInfo bi = {};
        auto code = pdev->QueryBdfInfo(&bi);
        if (code != ErrorCode::Success)
            continue;

        if (bdf.bus_number    == bi.bus_number    &&
            bdf.device_number == bi.device_number &&
            bdf.domain_number == bi.domain_number &&
            bdf.function_number == bi.function_number) {
            *processor_handle = reinterpret_cast<amdsmi_processor_handle>(pdev);
            return AMDSMI_STATUS_SUCCESS;
        }
    }

    return AMDSMI_STATUS_NOT_FOUND;
}

amdsmi_status_t
amdsmi_get_link_topology_nearest(amdsmi_processor_handle processor_handle,
                                 amdsmi_link_type_t link_type,
                                 amdsmi_topology_nearest_t* topology_nearest_info)
{
    if (topology_nearest_info == nullptr) {
        return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }

    if (link_type < amdsmi_link_type_t::AMDSMI_LINK_TYPE_INTERNAL ||
        link_type > amdsmi_link_type_t::AMDSMI_LINK_TYPE_UNKNOWN) {
        return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }


    auto status(amdsmi_status_t::AMDSMI_STATUS_SUCCESS);

    struct LinkTopolyInfo_t
    {
        amdsmi_processor_handle target_processor_handle;
        amdsmi_link_type_t link_type;
        bool is_accessible;
        uint64_t num_hops;
        uint64_t link_weight;
    };

    /*
     *  Note: The link topology table is sorted by the number of hops and link weight.
     */
    struct LinkTopogyOrderCmp_t {
        constexpr bool operator()(const LinkTopolyInfo_t& left,
                                  const LinkTopolyInfo_t& right) const noexcept
        {
            if (left.num_hops == right.num_hops) {
                return (left.num_hops >= right.num_hops);
            }
            else {
                return (left.link_weight > right.link_weight);
            }
        }
    };
    std::priority_queue<LinkTopolyInfo_t,
                        std::vector<LinkTopolyInfo_t>,
                        LinkTopogyOrderCmp_t> link_topology_order{};
    //


    AMDSMI_CHECK_INIT();
    auto socket_counter = uint32_t(0);
    if (auto api_status = amdsmi_get_socket_handles(&socket_counter, nullptr);
        (api_status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS)) {
        return api_status;
    }

    amdsmi_socket_handle socket_list[socket_counter];
    if (auto api_status = amdsmi_get_socket_handles(&socket_counter, &socket_list[0]);
        (api_status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS)) {
        return api_status;
    }


    uint32_t device_counter(AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP);
    amdsmi_processor_handle device_list[AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP];
    for (auto socket_idx = uint32_t(0); socket_idx < socket_counter; ++socket_idx) {
        if (auto api_status = amdsmi_get_processor_handles(socket_list[socket_idx], &device_counter, device_list);
            (api_status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS)) {
            return api_status;
        }

        for (auto device_idx = uint32_t(0); device_idx < device_counter; ++device_idx) {
            /*  Note: Skip the processor handle that is being queried. */
            if (processor_handle != device_list[device_idx]) {
                // Accessibility?
                auto is_accessible(false);
                if (auto api_status = amdsmi_is_P2P_accessible(processor_handle, device_list[device_idx], &is_accessible);
                    (api_status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) || !is_accessible) {
                    continue;
                }

                // Link type matches what we are searching for?
                auto link_type_new = link_type;
                auto num_hops = uint64_t(0);
                if (auto api_status = amdsmi_topo_get_link_type(processor_handle, device_list[device_idx], &num_hops, &link_type_new);
                    (api_status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) || (link_type_new != link_type)) {
                    continue;
                }

                // Link weights
                auto link_weight = uint64_t(0);
                if (auto api_status = amdsmi_topo_get_link_weight(processor_handle, device_list[device_idx], &link_weight);
                    (api_status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS)) {
                    continue;
                }

                // Topology nearest info
                LinkTopolyInfo_t link_info = {
                    .target_processor_handle = device_list[device_idx],
                    .link_type = link_type,
                    .is_accessible = is_accessible,
                    .num_hops = num_hops,
                    .link_weight = link_weight
                };
                link_topology_order.push(link_info);
            }
        }
    }

    /*
     *  Note: The link topology table is sorted by the number of hops and link weight.
     */
    std::fill(std::begin(topology_nearest_info->processor_list),
              std::end(topology_nearest_info->processor_list), nullptr);
    topology_nearest_info->count = static_cast<uint32_t>(link_topology_order.size());
    auto topology_nearest_counter = uint32_t(0);
    while (!link_topology_order.empty()) {
        auto link_info = link_topology_order.top();
        link_topology_order.pop();

        if (topology_nearest_counter < (AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP)) {
            topology_nearest_info->processor_list[topology_nearest_counter++] = link_info.target_processor_handle;
        }
    }

    return status;
}

static const std::map<amdsmi_virtualization_mode_t, std::string>
virtualization_mode_map = {
  {AMDSMI_VIRTUALIZATION_MODE_UNKNOWN, "UNKNOWN"},
  {AMDSMI_VIRTUALIZATION_MODE_BAREMETAL, "BAREMETAL"},
  { AMDSMI_VIRTUALIZATION_MODE_HOST, "HOST"},
  { AMDSMI_VIRTUALIZATION_MODE_GUEST, "GUEST"},
  {AMDSMI_VIRTUALIZATION_MODE_PASSTHROUGH, "PASSTHROUGH"}
};

amdsmi_status_t
amdsmi_get_gpu_virtualization_mode(amdsmi_processor_handle processor_handle,
                                    amdsmi_virtualization_mode_t *mode) {
    AMDSMI_CHECK_INIT();
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | start";
    LOG_INFO(ss);
    if (mode == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    struct drm_amdgpu_info_device dev_info = {};
    *mode = AMDSMI_VIRTUALIZATION_MODE_UNKNOWN;

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS) {
        return r;
    }

    amdsmi_status_t status;
    SMIGPUDEVICE_MUTEX(gpu_device->get_mutex())

    std::string render_name = gpu_device->get_gpu_path();
    std::string path = "/dev/dri/" + render_name;
    if (render_name.empty()) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    ScopedFD drm_fd(path.c_str(), O_RDWR | O_CLOEXEC);
    if (!drm_fd.valid()) {
        ss << __PRETTY_FUNCTION__
           << " | Failed to open " << path << ": " << strerror(errno)
           << "; Returning: " << smi_amdgpu_get_status_string(AMDSMI_STATUS_FILE_ERROR, false);
        LOG_ERROR(ss);
        return AMDSMI_STATUS_FILE_ERROR;
    }

    amd::smi::AMDSmiLibraryLoader libdrm;
    status = libdrm.load(LIBDRM_AMDGPU_SONAME);
    if (status != AMDSMI_STATUS_SUCCESS) {
        libdrm.unload();
        ss << __PRETTY_FUNCTION__
           << " | Failed to load " LIBDRM_AMDGPU_SONAME ": " << strerror(errno)
           << "; Returning: " << smi_amdgpu_get_status_string(status, false);
        LOG_ERROR(ss);
        return status;
    }

    typedef drmVersionPtr (*drmGetVersion_t)(int fd);
    typedef void (*drmFreeVersion_t)(drmVersionPtr version);

    drmGetVersion_t drm_get_version = nullptr;
    drmFreeVersion_t drm_free_version = nullptr;
    // Load the drmGetVersion symbol
    status = libdrm.load_symbol(reinterpret_cast<drmGetVersion_t *>(&drm_get_version),
                                "drmGetVersion");
    if (status != AMDSMI_STATUS_SUCCESS) {
        libdrm.unload();
        ss << __PRETTY_FUNCTION__
           << " | Failed to load drmGetVersion symbol"
           << "; Returning: " << smi_amdgpu_get_status_string(status, false);
        LOG_ERROR(ss);
        return status;
    }

    // Load the drmFreeVersion symbol
    status = libdrm.load_symbol(reinterpret_cast<drmFreeVersion_t *>(&drm_free_version),
                                "drmFreeVersion");
    if (status != AMDSMI_STATUS_SUCCESS) {
        drm_free_version = nullptr;
        libdrm.unload();
        ss << __PRETTY_FUNCTION__
           << " | Failed to load drmFreeVersion symbol"
           << "; Returning: " << smi_amdgpu_get_status_string(status, false);
        LOG_ERROR(ss);
        return status;
    }

    // get drm version. If it's older than 3.62.0, then say not supported and exit.
    auto drm_version = drm_get_version(drm_fd);
    // minimum version that supports getting of virtualization mode
    int major_version = 3;
    int minor_version = 62;
    int patch_version = 0;
    bool isDRMVersionSupported = false;
    ((drm_version->version_major >= major_version)
        && (drm_version->version_minor >= minor_version)
        && (drm_version->version_patchlevel >= patch_version) ?
        isDRMVersionSupported = true : isDRMVersionSupported = false);
    ss << __PRETTY_FUNCTION__ << " | drm_version: "
       << std::dec << drm_version->version_major << "." << drm_version->version_minor
       << "." << drm_version->version_patchlevel << "\n"
       << " | isDRMVersionSupported: " << (isDRMVersionSupported ? "TRUE" : "FALSE") << "\n"
       << " | Expecting version >= " << major_version << "." << minor_version
       << "." << patch_version << "\n"
       << "; Returning: " << (isDRMVersionSupported ?
            smi_amdgpu_get_status_string(AMDSMI_STATUS_SUCCESS, false):
            smi_amdgpu_get_status_string(AMDSMI_STATUS_NOT_SUPPORTED, false));
    LOG_INFO(ss);

    // Check if the version is supported
    // If not, then return not supported
    if (isDRMVersionSupported == false) {
        drm_free_version(drm_version);
        libdrm.unload();
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    // Get the device info
    typedef int (*drmCommandWrite_t)(int fd, unsigned long drmCommandIndex,
                                    void *data, unsigned long size);
    drmCommandWrite_t drmCommandWrite = nullptr;

    // load symbol from libdrm
    status = libdrm.load_symbol(reinterpret_cast<drmCommandWrite_t *>(&drmCommandWrite),
                                "drmCommandWrite");
    if (status != AMDSMI_STATUS_SUCCESS) {
        drm_free_version(drm_version);
        libdrm.unload();
        ss << __PRETTY_FUNCTION__
           << " | Failed to load drmCommandWrite symbol: " << strerror(errno)
           << "; Returning: " << smi_amdgpu_get_status_string(status, false);
        LOG_ERROR(ss);
        return status;
    }

    // Get the device info
    memset(&dev_info, 0, sizeof(struct drm_amdgpu_info_device));
    struct drm_amdgpu_info request = {};
    memset(&request, 0, sizeof(request));
    request.return_pointer = reinterpret_cast<unsigned long long>(&dev_info);
    request.return_size = sizeof(struct drm_amdgpu_info_device);
    request.query = AMDGPU_INFO_DEV_INFO;
    auto drm_write = drmCommandWrite(drm_fd, DRM_AMDGPU_INFO, &request,
                                     sizeof(struct drm_amdgpu_info));
    ss << __PRETTY_FUNCTION__
       << " | drm_fd: " << std::dec << drm_fd << "\n"
       << " | path: " << path << "\n"
       << " | drmCommandWrite: " << drm_write << "\n"
       << " | drmCommandWrite returned: " << strerror(errno) << "\n"
       << " | dev_info.ids_flags: " << dev_info.ids_flags << "\n"
       << " | dev_info.ids_flags size: " << sizeof(dev_info.ids_flags) << "\n"
       << " | dev_info.pci_rev: 0x"
       << std::setw(4) << std::setfill('0') << std::hex << dev_info.pci_rev << "\n"
       << " | dev_info.device_id: 0x"
       << std::setw(4) << std::setfill('0') << std::hex << dev_info.device_id;
    LOG_INFO(ss);

    if (drm_write == 0) {
        uint32_t ids_flag = ((dev_info.ids_flags & AMDGPU_IDS_FLAGS_MODE_MASK)
                             >> AMDGPU_IDS_FLAGS_MODE_SHIFT);
        switch (ids_flag) {
            case 0: *mode = AMDSMI_VIRTUALIZATION_MODE_BAREMETAL; break;
            case 1: *mode = AMDSMI_VIRTUALIZATION_MODE_GUEST; break;
            case 2: *mode = AMDSMI_VIRTUALIZATION_MODE_PASSTHROUGH; break;
            default: *mode = AMDSMI_VIRTUALIZATION_MODE_UNKNOWN; break;
        }
        std::string mode_str = "UNKNOWN";
        if (virtualization_mode_map.find(*mode) != virtualization_mode_map.end()) {
            mode_str.clear();
            mode_str = virtualization_mode_map.at(*mode);
        }
        ss << __PRETTY_FUNCTION__
           << " | ids_flag: " << std::dec << ids_flag << "\n"
           << " | dev_info.ids_flags: 0x"
           << std::hex << std::setw(8) << std::setfill('0') << dev_info.ids_flags << "\n"
           << " | *mode: " << mode_str << "\n"
           << " | Returning: " << smi_amdgpu_get_status_string(status, false)
           << std::endl;
        LOG_INFO(ss);
    } else {
        ss << __PRETTY_FUNCTION__
           << " | Failed to get device info: " << strerror(errno)
           << " | returning AMDSMI_STATUS_DRM_ERROR";
        LOG_ERROR(ss);
        *mode = AMDSMI_VIRTUALIZATION_MODE_UNKNOWN;
        status = AMDSMI_STATUS_DRM_ERROR;
    }
    drm_free_version(drm_version);
    libdrm.unload();
    return status;
}

// PTL

bool amdsmi_is_supported_format(
    const std::vector<amdsmi_ptl_data_format_t> &supported,
    amdsmi_ptl_data_format_t fmt) {
  return std::find(supported.begin(), supported.end(), fmt) != supported.end();
}

amdsmi_status_t
amdsmi_get_gpu_ptl_state(amdsmi_processor_handle processor_handle, bool *enabled) {
    return rsmi_wrapper(rsmi_get_gpu_ptl_state, processor_handle, 0, enabled);
}

amdsmi_status_t
amdsmi_set_gpu_ptl_state(amdsmi_processor_handle processor_handle, bool enable) {
  return rsmi_wrapper(rsmi_set_gpu_ptl_state, processor_handle, 0, enable);
}

// Mapping for PTL string <-> enum
struct PtlFormatMapEntry {
  const char* token;
  amdsmi_ptl_data_format_t fmt;
};

PtlFormatMapEntry kPtlFormatMap[] = {
  {"I8",   AMDSMI_PTL_DATA_FORMAT_I8},
  {"F16",  AMDSMI_PTL_DATA_FORMAT_F16},
  {"BF16", AMDSMI_PTL_DATA_FORMAT_BF16},
  {"F32",  AMDSMI_PTL_DATA_FORMAT_F32},
  {"F64",  AMDSMI_PTL_DATA_FORMAT_F64},
  {"F8",   AMDSMI_PTL_DATA_FORMAT_F8},
  {"VECTOR",AMDSMI_PTL_DATA_FORMAT_VECTOR},
};
static constexpr size_t kPtlFormatMapSize =
    sizeof(kPtlFormatMap) / sizeof(kPtlFormatMap[0]);

// Given string, return ptl data format enum
amdsmi_ptl_data_format_t token_to_amdsmi_fmt(std::string token) {
  token = amd::smi::trim(token);
  if (token.empty()) {
    return AMDSMI_PTL_DATA_FORMAT_INVALID;
  }

  // Ensure upper case for comparison
  for (auto &c : token) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }

  for (size_t i = 0; i < kPtlFormatMapSize; ++i) {
    if (token == kPtlFormatMap[i].token) {
      return kPtlFormatMap[i].fmt;
    }
  }
  return AMDSMI_PTL_DATA_FORMAT_INVALID;
}

// Given ptl format, return string representation
const char* amdsmi_fmt_to_token(amdsmi_ptl_data_format_t fmt) {
  for (size_t i = 0; i < kPtlFormatMapSize; ++i) {
    if (kPtlFormatMap[i].fmt == fmt) {
      return kPtlFormatMap[i].token;
    }
  }
  return "N/A";
}

// Internal only helper to create supported ptl formats
amdsmi_status_t amdsmi_read_supported_ptl_formats(
    amdsmi_processor_handle processor_handle,
    std::vector<amdsmi_ptl_data_format_t> &out) {

  out.clear();

  std::string line;
  {
    char buf[AMDSMI_MAX_STRING_LENGTH] = {0};
    amdsmi_status_t st = rsmi_wrapper(
        rsmi_read_supported_ptl_formats, processor_handle, 0, buf, AMDSMI_MAX_STRING_LENGTH);

    if (st != AMDSMI_STATUS_SUCCESS) {
        return st;
    }
    line.assign(buf);
  }

  line = amd::smi::trim(line);
  auto tokens = split_string(line, ',');
  if (tokens.empty()) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  for (const auto &t : tokens) {
    amdsmi_ptl_data_format_t f = token_to_amdsmi_fmt(t);
    if (f == AMDSMI_PTL_DATA_FORMAT_INVALID) {
      return AMDSMI_STATUS_UNEXPECTED_DATA;
    }
    out.push_back(f);
  }
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_ptl_formats(amdsmi_processor_handle processor_handle,
                        amdsmi_ptl_data_format_t *data_format1,
                        amdsmi_ptl_data_format_t *data_format2)
{

    if (data_format1 == nullptr || data_format2 == nullptr) {
        return AMDSMI_STATUS_ARG_PTR_NULL;
    }
    *data_format1 = AMDSMI_PTL_DATA_FORMAT_INVALID;
    *data_format2 = AMDSMI_PTL_DATA_FORMAT_INVALID;

    // Ensure PTL enabled
    bool enabled = false;
    amdsmi_status_t st = amdsmi_get_gpu_ptl_state(processor_handle, &enabled);
    if (st != AMDSMI_STATUS_SUCCESS) {
        return st;
    }
    if (!enabled) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    // Read ptl sysfs
    std::string line;
    {
        char buf[AMDSMI_MAX_STRING_LENGTH] = {0};
        st = rsmi_wrapper(rsmi_get_gpu_ptl_formats, processor_handle, 0, buf, AMDSMI_MAX_STRING_LENGTH);
        if (st != AMDSMI_STATUS_SUCCESS) {
            return st;
        }
        line.assign(buf);
    }

    line = amd::smi::trim(line);
    auto tokens = split_string(line, ',');
    if (tokens.empty() || tokens.size() != 2) {
        return AMDSMI_STATUS_UNEXPECTED_SIZE;  // malformed sysfs content
    }

    // Parse tokens
    amdsmi_ptl_data_format_t f1 = token_to_amdsmi_fmt(tokens[0]);
    if (f1 == AMDSMI_PTL_DATA_FORMAT_INVALID) {
        return AMDSMI_STATUS_UNEXPECTED_DATA;
    }

    amdsmi_ptl_data_format_t f2 = token_to_amdsmi_fmt(tokens[1]);
    if (f2 == AMDSMI_PTL_DATA_FORMAT_INVALID) {
        return AMDSMI_STATUS_UNEXPECTED_DATA;
    }

    *data_format1 = f1;
    *data_format2 = f2;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_set_gpu_ptl_formats(amdsmi_processor_handle processor_handle,
                          amdsmi_ptl_data_format_t data_format1,
                          amdsmi_ptl_data_format_t data_format2)
{

    if (data_format1 == AMDSMI_PTL_DATA_FORMAT_INVALID ||
        data_format2 == AMDSMI_PTL_DATA_FORMAT_INVALID ||
        data_format1 == data_format2) {
        return AMDSMI_STATUS_UNEXPECTED_DATA;
    }

    // Ensure PTL enabled
    bool enabled = false;
    amdsmi_status_t st = amdsmi_get_gpu_ptl_state(processor_handle, &enabled);
    if (st != AMDSMI_STATUS_SUCCESS) {
        return st;
    }
    if (!enabled) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    // Read supported formats and check both are allowed
    std::vector<amdsmi_ptl_data_format_t> supported;
    st = amdsmi_read_supported_ptl_formats(processor_handle, supported);
    if (st != AMDSMI_STATUS_SUCCESS) {
        return st;
    }

    if (!amdsmi_is_supported_format(supported, data_format1) ||
        !amdsmi_is_supported_format(supported, data_format2)) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    // Convert enums to string
    std::string format =
        std::string(amdsmi_fmt_to_token(data_format1)) + "," +
                    amdsmi_fmt_to_token(data_format2);

    return rsmi_wrapper(rsmi_set_gpu_ptl_formats, processor_handle, 0, format.c_str());
}

amdsmi_status_t amdsmi_get_cpu_affinity_with_scope(amdsmi_processor_handle processor_handle,
            uint32_t cpu_set_size, uint64_t *cpu_set, amdsmi_affinity_scope_t scope)
{
    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || cpu_set == nullptr || cpu_set_size == 0) {
        return AMDSMI_STATUS_INVAL;
    }

    // Retrieve GPU device from the processor handle
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t status = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    uint32_t numa_node;
    status = amdsmi_topo_get_numa_node_number(processor_handle, &numa_node);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    int32_t node_id = static_cast<int32_t>(numa_node);

    status = amdsmi_get_gpu_topo_numa_affinity(processor_handle, &node_id);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    if(node_id < 0) {
       return AMDSMI_STATUS_NOT_FOUND;
    }

    std::memset(cpu_set, 0, cpu_set_size * sizeof(uint64_t));
    switch(scope) {
        case AMDSMI_AFFINITY_SCOPE_NODE:
        {
            std::vector<uint64_t> bitmask = gpu_device->get_bitmask_from_numa_node(node_id, cpu_set_size);
            if(bitmask[0] == std::numeric_limits<int32_t>::max()){
                return AMDSMI_STATUS_REFCOUNT_OVERFLOW;
            } else {
                std::memcpy(cpu_set, bitmask.data(), cpu_set_size * sizeof(uint64_t));
            }
            break;
        }

        case AMDSMI_AFFINITY_SCOPE_SOCKET:
        {
            uint32_t drm_card = gpu_device->get_card_id();
            std::vector<uint64_t> bitmask = gpu_device->get_bitmask_from_local_cpulist(drm_card, cpu_set_size);
            if(bitmask[0] == std::numeric_limits<int32_t>::max()){
                return AMDSMI_STATUS_REFCOUNT_OVERFLOW;
            } else {
                std::memcpy(cpu_set, bitmask.data(), cpu_set_size * sizeof(uint64_t));
            }
            break;
        }

        default:
            return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
    }

    return AMDSMI_STATUS_SUCCESS;
}

#ifdef ENABLE_ESMI_LIB
static amdsmi_status_t amdsmi_errno_to_esmi_status(amdsmi_status_t status)
{
    for (auto& iter : amd::smi::esmi_status_map) {
        if (iter.first == static_cast<esmi_status_t>(status))
            return iter.second;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_threads_per_core(uint32_t *threads_per_core)
{
    amdsmi_status_t status;
    uint32_t esmi_threads_per_core;

    AMDSMI_CHECK_INIT();

    status = static_cast<amdsmi_status_t>(esmi_threads_per_core_get(&esmi_threads_per_core));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *threads_per_core = esmi_threads_per_core;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_hsmp_proto_ver(amdsmi_processor_handle processor_handle,
                uint32_t *proto_ver)
{
    amdsmi_status_t status;
    uint32_t hsmp_proto_ver;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    status = static_cast<amdsmi_status_t>(esmi_hsmp_proto_ver_get(&hsmp_proto_ver));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *proto_ver = hsmp_proto_ver;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_hsmp_driver_version(amdsmi_processor_handle processor_handle,
                                              amdsmi_hsmp_driver_version_t *amdsmi_hsmp_driver_ver)
{
    amdsmi_status_t status;
    struct hsmp_driver_version hsmp_driver_ver;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    status = static_cast<amdsmi_status_t>(esmi_hsmp_driver_version_get(&hsmp_driver_ver));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    amdsmi_hsmp_driver_ver->major = hsmp_driver_ver.major;
    amdsmi_hsmp_driver_ver->minor = hsmp_driver_ver.minor;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_smu_fw_version(amdsmi_processor_handle processor_handle,
                                              amdsmi_smu_fw_version_t *amdsmi_smu_fw)
{
    amdsmi_status_t status;
    struct smu_fw_version smu_fw;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    status = static_cast<amdsmi_status_t>(esmi_smu_fw_version_get(&smu_fw));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    amdsmi_smu_fw->major = smu_fw.major;
    amdsmi_smu_fw->minor = smu_fw.minor;
    amdsmi_smu_fw->debug = smu_fw.debug;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_core_energy(amdsmi_processor_handle processor_handle,
                                           uint64_t *penergy)
{
    amdsmi_status_t status;
    uint64_t core_input;
    uint32_t core_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    core_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_core_energy_get(core_ind, &core_input));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *penergy = core_input;

    return AMDSMI_STATUS_SUCCESS;

}

amdsmi_status_t amdsmi_get_cpu_socket_energy(amdsmi_processor_handle processor_handle,
                                             uint64_t *penergy)
{
    amdsmi_status_t status;
    uint64_t pkg_input;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_energy_get(sock_ind, &pkg_input));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *penergy = pkg_input;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_prochot_status(amdsmi_processor_handle processor_handle,
                                              uint32_t *prochot)
{
    amdsmi_status_t status;
    uint32_t phot;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_prochot_status_get(sock_ind, &phot));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *prochot = phot;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_fclk_mclk(amdsmi_processor_handle processor_handle,
                                         uint32_t *fclk, uint32_t *mclk)
{
    amdsmi_status_t status;
    uint32_t f_clk, m_clk;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_fclk_mclk_get(sock_ind, &f_clk, &m_clk));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *fclk = f_clk;
    *mclk = m_clk;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_cclk_limit(amdsmi_processor_handle processor_handle,
                                          uint32_t *cclk)
{
    amdsmi_status_t status;
    uint32_t c_clk;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_cclk_limit_get(sock_ind, &c_clk));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *cclk = c_clk;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_socket_current_active_freq_limit(amdsmi_processor_handle processor_handle,
                                                                uint16_t *freq, char **src_type)
{
    amdsmi_status_t status;
    uint16_t limit;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_current_active_freq_limit_get(sock_ind, &limit, src_type));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *freq = limit;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_socket_freq_range(amdsmi_processor_handle processor_handle,
                                                 uint16_t *fmax, uint16_t *fmin)
{
    amdsmi_status_t status;
    uint16_t f_max;
    uint16_t f_min;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_freq_range_get(sock_ind, &f_max, &f_min));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *fmax = f_max;
    *fmin = f_min;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_core_current_freq_limit(amdsmi_processor_handle processor_handle,
                                                       uint32_t *freq)
{
    amdsmi_status_t status;
    uint32_t c_clk;
    uint32_t core_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    core_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_current_freq_limit_core_get(core_ind, &c_clk));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *freq = c_clk;

    return AMDSMI_STATUS_SUCCESS;

}

amdsmi_status_t amdsmi_get_cpu_socket_power(amdsmi_processor_handle processor_handle,
                                            uint32_t *ppower)
{
    amdsmi_status_t status;
    uint32_t avg_power;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_power_get(sock_ind, &avg_power));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *ppower = avg_power;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_socket_power_cap(amdsmi_processor_handle processor_handle,
                                                uint32_t *pcap)
{
    amdsmi_status_t status;
    uint32_t p_cap;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_power_cap_get(sock_ind, &p_cap));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *pcap = p_cap;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_socket_power_cap_max(amdsmi_processor_handle processor_handle,
                                                    uint32_t *pmax)
{
    amdsmi_status_t status;
    uint32_t p_max;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_power_cap_max_get(sock_ind, &p_max));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *pmax = p_max;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_pwr_svi_telemetry_all_rails(amdsmi_processor_handle processor_handle,
                                                           uint32_t *power)
{
    amdsmi_status_t status;
    uint32_t pow;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_pwr_svi_telemetry_all_rails_get(sock_ind, &pow));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *power = pow;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_socket_power_cap(amdsmi_processor_handle processor_handle,
                                                uint32_t pcap)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_power_cap_set(sock_ind, pcap));

    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_pwr_efficiency_mode(amdsmi_processor_handle processor_handle,
                                                   uint8_t mode)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_pwr_efficiency_mode_set(sock_ind, mode));

    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_core_boostlimit(amdsmi_processor_handle processor_handle,
                                               uint32_t *pboostlimit)
{
    amdsmi_status_t status;
    uint32_t boostlimit;
    uint32_t core_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    core_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_core_boostlimit_get(core_ind, &boostlimit));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *pboostlimit = boostlimit;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_socket_c0_residency(amdsmi_processor_handle processor_handle,
                                                   uint32_t *pc0_residency)
{
    amdsmi_status_t status;
    uint32_t res;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_c0_residency_get(sock_ind, &res));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *pc0_residency = res;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_core_boostlimit(amdsmi_processor_handle processor_handle,
                                               uint32_t boostlimit)
{
    amdsmi_status_t status;
    uint32_t core_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    core_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_core_boostlimit_set(core_ind, boostlimit));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_socket_boostlimit(amdsmi_processor_handle processor_handle,
                                                 uint32_t boostlimit)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_boostlimit_set(sock_ind, boostlimit));

    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_ddr_bw(amdsmi_processor_handle processor_handle,
                                      amdsmi_ddr_bw_metrics_t *ddr_bw)
{
    amdsmi_status_t status;
    struct ddr_bw_metrics ddr;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_ddr_bw_get(sock_ind, &ddr));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    ddr_bw->max_bw = ddr.max_bw;
    ddr_bw->utilized_bw = ddr.utilized_bw;
    ddr_bw->utilized_pct = ddr.utilized_pct;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_socket_temperature(amdsmi_processor_handle processor_handle,
                                                  uint32_t *ptmon)
{
    amdsmi_status_t status;
    uint32_t tmon;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_temperature_get(sock_ind, &tmon));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *ptmon = tmon;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(
                   amdsmi_processor_handle processor_handle,
                   uint8_t dimm_addr, amdsmi_temp_range_refresh_rate_t *rate)
{
    amdsmi_status_t status;
    struct temp_range_refresh_rate dimm_rate;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);
    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_dimm_temp_range_and_refresh_rate_get(
                                            sock_ind, dimm_addr, &dimm_rate));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    rate->range = dimm_rate.range;
    rate->ref_rate = dimm_rate.ref_rate;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_dimm_power_consumption(amdsmi_processor_handle processor_handle,
                        uint8_t dimm_addr, amdsmi_dimm_power_t *dimm_pow)
{
    amdsmi_status_t status;
    struct dimm_power d_power;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_dimm_power_consumption_get(sock_ind,
                                                              dimm_addr, &d_power));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    dimm_pow->power = d_power.power;
    dimm_pow->update_rate = d_power.update_rate;
    dimm_pow->dimm_addr = d_power.dimm_addr;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_dimm_thermal_sensor(amdsmi_processor_handle processor_handle,
        uint8_t dimm_addr, amdsmi_dimm_thermal_t *dimm_temp)
{
    amdsmi_status_t status;
    struct dimm_thermal d_sensor;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_dimm_thermal_sensor_get(sock_ind,
                                                              dimm_addr, &d_sensor));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    dimm_temp->temp = d_sensor.temp;
    dimm_temp->update_rate = d_sensor.update_rate;
    dimm_temp->dimm_addr = d_sensor.dimm_addr;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_xgmi_width(amdsmi_processor_handle processor_handle,
        uint8_t min, uint8_t max)
{
    amdsmi_status_t status;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    status = static_cast<amdsmi_status_t>(esmi_xgmi_width_set(min, max));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_gmi3_link_width_range(amdsmi_processor_handle processor_handle,
        uint8_t min_link_width, uint8_t max_link_width)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_gmi3_link_width_range_set(sock_ind,
                                                        min_link_width, max_link_width));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_cpu_apb_enable(amdsmi_processor_handle processor_handle)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_apb_enable(sock_ind));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_cpu_apb_disable(amdsmi_processor_handle processor_handle,
        uint8_t pstate)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_apb_disable(sock_ind, pstate));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_socket_lclk_dpm_level(amdsmi_processor_handle processor_handle,
        uint8_t nbio_id, uint8_t min, uint8_t max)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_lclk_dpm_level_set(sock_ind, nbio_id, min, max));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_socket_lclk_dpm_level(amdsmi_processor_handle processor_handle,
        uint8_t nbio_id, amdsmi_dpm_level_t *nbio)
{
    amdsmi_status_t status;
    struct dpm_level nb;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_socket_lclk_dpm_level_get(sock_ind,
                                                                        nbio_id, &nb));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    nbio->min_dpm_level = nb.min_dpm_level;
    nbio->max_dpm_level = nb.max_dpm_level;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_pcie_link_rate(amdsmi_processor_handle processor_handle,
        uint8_t rate_ctrl, uint8_t *prev_mode)
{
    amdsmi_status_t status;
    uint8_t sock_ind;
    uint8_t p_mode;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_pcie_link_rate_set(sock_ind,
                                                                        rate_ctrl, &p_mode));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *prev_mode = p_mode;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_df_pstate_range(amdsmi_processor_handle processor_handle,
        uint8_t max_pstate, uint8_t min_pstate)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_df_pstate_range_set(sock_ind,
                                                                        max_pstate, min_pstate));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_current_io_bandwidth(amdsmi_processor_handle processor_handle,
        amdsmi_link_id_bw_type_t link, uint32_t *io_bw)
{
    amdsmi_status_t status;
    uint32_t bw;
    struct link_id_bw_type io_link;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    io_link.link_name = link.link_name;
    io_link.bw_type = static_cast<io_bw_encoding>(link.bw_type);

    status = static_cast<amdsmi_status_t>(esmi_current_io_bandwidth_get(sock_ind,
                                                        io_link, &bw));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *io_bw = bw;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_current_xgmi_bw(amdsmi_processor_handle processor_handle,
        amdsmi_link_id_bw_type_t link, uint32_t *xgmi_bw)
{
    amdsmi_status_t status;
    uint32_t bw;
    struct link_id_bw_type io_link;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    io_link.link_name = link.link_name;
    io_link.bw_type= static_cast<io_bw_encoding>(link.bw_type);

    status = static_cast<amdsmi_status_t>(esmi_current_xgmi_bw_get(sock_ind, io_link, &bw));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *xgmi_bw = bw;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_hsmp_metrics_table_version(amdsmi_processor_handle processor_handle,
                uint32_t *metrics_version)
{
    amdsmi_status_t status;
    uint32_t metrics_tbl_ver;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    status = static_cast<amdsmi_status_t>(esmi_metrics_table_version_get(&metrics_tbl_ver));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *metrics_version = metrics_tbl_ver;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_hsmp_metrics_table(amdsmi_processor_handle processor_handle,
                amdsmi_hsmp_metrics_table_t *metrics_table)
{
    amdsmi_status_t status;
    struct hsmp_metric_table metrics_tbl;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    if(sizeof(amdsmi_hsmp_metrics_table_t) != sizeof(struct hsmp_metric_table))
        return AMDSMI_STATUS_UNEXPECTED_SIZE;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_metrics_table_get(sock_ind, &metrics_tbl));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    std::memcpy(metrics_table, &metrics_tbl, sizeof(amdsmi_hsmp_metrics_table_t));

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_first_online_core_on_cpu_socket(amdsmi_processor_handle processor_handle,
        uint32_t *pcore_ind)
{
    amdsmi_status_t status;
    uint32_t online_core;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_first_online_core_on_socket(sock_ind, &online_core));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *pcore_ind = online_core;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_family(uint32_t *cpu_family)
{
    amdsmi_status_t status;
    uint32_t family;

    AMDSMI_CHECK_INIT();

    status = amd::smi::AMDSmiSystem::getInstance().get_cpu_family(&family);
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *cpu_family = family;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_model(uint32_t *cpu_model)
{
    amdsmi_status_t status;
    uint32_t model;

    AMDSMI_CHECK_INIT();

    status = amd::smi::AMDSmiSystem::getInstance().get_cpu_model(&model);
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *cpu_model = model;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_model_name(amdsmi_processor_handle processor_handle, amdsmi_cpu_info_t *cpu_info)
{
    amdsmi_status_t status;
    uint32_t sock_ind;
    std::string model_name;

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = amd::smi::AMDSmiSystem::getInstance().get_cpu_model_name(sock_ind, &model_name);
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    snprintf(cpu_info->model_name, AMDSMI_MAX_STRING_LENGTH, "%s", model_name.c_str());

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_cores_per_socket(uint32_t sock_count, amdsmi_sock_info_t *sock_info)
{
    (void)(sock_count);//unused
    amdsmi_status_t status;
    uint32_t core_num;
    status = amd::smi::AMDSmiSystem::getInstance().get_sys_cpu_cores_per_socket(&core_num);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    sock_info->cores_per_socket = core_num;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_socket_count(uint32_t *sock_count)
{
    amdsmi_status_t status;
    uint32_t sock_num;
    status = amd::smi::AMDSmiSystem::getInstance().get_sys_num_of_cpu_sockets(&sock_num);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    *sock_count = sock_num;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_handles(uint32_t *cpu_count,
                                       amdsmi_processor_handle *processor_handles)
{
    uint32_t soc_count = 0, index = 0, cpu_per_soc = 0;
    processor_type_t processor_type = AMDSMI_PROCESSOR_TYPE_AMD_CPU;
    std::vector<amdsmi_processor_handle> cpu_handles;
    amdsmi_status_t status;

    AMDSMI_CHECK_INIT();
    if (cpu_count == nullptr)
        return AMDSMI_STATUS_INVAL;

    status = amdsmi_get_socket_handles(&soc_count, nullptr);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    // Allocate the memory for the sockets
    std::vector<amdsmi_socket_handle> sockets(soc_count);
    // Get the sockets of the system
    status = amdsmi_get_socket_handles(&soc_count, &sockets[0]);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    for (index = 0 ; index < soc_count; index++)
    {
        cpu_per_soc = 0;
        status = amdsmi_get_processor_handles_by_type(sockets[index], processor_type,
                                                      nullptr, &cpu_per_soc);
        if (status != AMDSMI_STATUS_SUCCESS)
            return status;
        if (cpu_per_soc == 0)
            continue;

        // Allocate the memory for the cpus
        std::vector<amdsmi_processor_handle> plist(cpu_per_soc);
        // Get the cpus for each socket
        status = amdsmi_get_processor_handles_by_type(sockets[index], processor_type,
                                                      &plist[0], &cpu_per_soc);
        if (status != AMDSMI_STATUS_SUCCESS)
            return status;
        cpu_handles.insert(cpu_handles.end(), plist.begin(), plist.end());
    }

    // Get the cpu count
    *cpu_count = static_cast<uint32_t>(cpu_handles.size());
    if (processor_handles == nullptr) {
        return AMDSMI_STATUS_SUCCESS;
    }

    // Copy the cpu socket handles
    for (uint32_t i = 0; i < *cpu_count; i++) {
        processor_handles[i] = reinterpret_cast<amdsmi_processor_handle>(cpu_handles[i]);
    }

    return status;
}

amdsmi_status_t amdsmi_get_cpucore_handles(uint32_t *cores_count,
                                            amdsmi_processor_handle* processor_handles)
{
    uint32_t soc_count = 0, index = 0, cores_per_soc = 0;
    processor_type_t processor_type = AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE;
    std::vector<amdsmi_processor_handle> core_handles;
    amdsmi_status_t status;

    AMDSMI_CHECK_INIT();
    if (cores_count == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Get sockets count
    status = amdsmi_get_socket_handles(&soc_count, nullptr);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    // Allocate the memory for the sockets
    std::vector<amdsmi_socket_handle> sockets(soc_count);
    // Get the sockets of the system
    status = amdsmi_get_socket_handles(&soc_count, &sockets[0]);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    for (index = 0 ; index < soc_count; index++)
    {
        cores_per_soc = 0;
        status = amdsmi_get_processor_handles_by_type(sockets[index], processor_type,
                                                      nullptr, &cores_per_soc);
        if (status != AMDSMI_STATUS_SUCCESS)
            return status;

        // Allocate the memory for the cores
        std::vector<amdsmi_processor_handle> plist(cores_per_soc);
        // Get the coress for each socket
        status = amdsmi_get_processor_handles_by_type(sockets[index], processor_type,
                                                      &plist[0], &cores_per_soc);
        if (status != AMDSMI_STATUS_SUCCESS) {
            return status;
        }

        core_handles.insert(core_handles.end(), plist.begin(), plist.end());
    }

    // Get the cores count
    *cores_count = static_cast<uint32_t>(core_handles.size());
    if (processor_handles == nullptr) {
        return AMDSMI_STATUS_SUCCESS;
    }

    // Copy the core handles
    for (uint32_t i = 0; i < *cores_count; i++) {
        processor_handles[i] = reinterpret_cast<amdsmi_processor_handle>(core_handles[i]);
    }

    return status;
}

amdsmi_status_t amdsmi_get_esmi_err_msg(amdsmi_status_t status, const char **status_string)
{
    for (const auto& iter : amd::smi::esmi_status_map) {
        const amdsmi_status_t _status = status;
        if (static_cast<int>(iter.first) == static_cast<int>(_status)) {
            *status_string = esmi_get_err_msg(static_cast<esmi_status_t>(iter.first));
            return iter.second;
        }
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_cpu_rail_isofreq_policy(amdsmi_processor_handle processor_handle,
                                                   uint8_t input)
{
    amdsmi_status_t status;
    uint8_t sock_ind;
    bool val;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr )
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    val = (bool)input;
    status = static_cast<amdsmi_status_t>(esmi_cpurail_isofreq_policy_set(sock_ind, &val));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_cpu_rail_isofreq_policy(amdsmi_processor_handle processor_handle,
                                                   uint8_t *cpurailiso)
{
    amdsmi_status_t status;
    uint8_t sock_ind;
    bool cpurailisofreq;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || cpurailiso == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_cpurail_isofreq_policy_get(sock_ind, &cpurailisofreq));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *cpurailiso = (uint8_t) cpurailisofreq;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_set_dfc_ctrl(amdsmi_processor_handle processor_handle,
                                    bool dfc_ctrl)
{
    amdsmi_status_t status;
    uint8_t sock_ind;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_dfc_enable_set(sock_ind, &dfc_ctrl));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_dfc_ctrl(amdsmi_processor_handle processor_handle,
                                    uint8_t *dfc_ctrl)
{
    amdsmi_status_t status;
    uint8_t sock_ind;
    bool dfcctrl;

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || dfc_ctrl == nullptr)
        return AMDSMI_STATUS_INVAL;

    amdsmi_status_t r = amdsmi_get_processor_info(processor_handle, SIZE, proc_id);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    sock_ind = (uint8_t)std::stoi(proc_id, NULL, 0);

    status = static_cast<amdsmi_status_t>(esmi_dfc_ctrl_setting_get(sock_ind, &dfcctrl));
    if (status != AMDSMI_STATUS_SUCCESS)
        return amdsmi_errno_to_esmi_status(status);

    *dfc_ctrl = (uint8_t)dfcctrl;

    return AMDSMI_STATUS_SUCCESS;
}

#endif
