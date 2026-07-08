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
#include <unistd.h>

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
#include "amd_smi/impl/amd_smi_common.h"

#ifdef BRCM_NIC
#include "amd_smi/impl/nic/amd_smi_nic_device.h"
#include "amd_smi/impl/nic/amd_smi_switch_device.h"
#include "amd_smi/impl/nic/amd_smi_lspci_commands.h"
#endif//BRCM_NIC
#include "amd_smi/impl/amd_smi_uuid.h"
#include "amd_smi/impl/amd_smi_utils.h"

#include "shared/include/dxcore_loader.h"
#include "shared/include/platform.h"
#include "shared/include/device.h"

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



amdsmi_status_t
amdsmi_init(uint64_t /*flags*/) {
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
amdsmi_status_t amdsmi_get_processor_info(amdsmi_processor_handle /*processor_handle*/,
                size_t /*len*/, char *name) {
    if (name == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t amdsmi_get_nic_processor_handles(amdsmi_socket_handle /*socket_handle*/,
    uint32_t* processor_count,
    amdsmi_processor_handle* /*processor_handles*/) {
    if (processor_count == nullptr) { return AMDSMI_STATUS_INVAL; }
    *processor_count = 0;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_processor_handles(amdsmi_socket_handle /*socket_handle*/,
    uint32_t* processor_count,
    amdsmi_processor_handle* /*processor_handles*/) {
    if (processor_count == nullptr) { return AMDSMI_STATUS_INVAL; }
    *processor_count = 0;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_node_handle(amdsmi_processor_handle /*processor_handle*/,
                                    amdsmi_node_handle *node_handle) {
    if (node_handle == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

#ifdef ENABLE_ESMI_LIB
amdsmi_status_t amdsmi_get_processor_count_from_handles(
        amdsmi_processor_handle* /*processor_handles*/,
        uint32_t* /*processor_count*/, uint32_t* nr_cpusockets,
        uint32_t* nr_cpucores, uint32_t* nr_gpus) {
    if (nr_cpusockets) *nr_cpusockets = 0;
    if (nr_cpucores)   *nr_cpucores   = 0;
    if (nr_gpus)       *nr_gpus       = 0;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_processor_handles_by_type(
        amdsmi_socket_handle /*socket_handle*/,
        processor_type_t /*processor_type*/,
        amdsmi_processor_handle* /*processor_handles*/,
        uint32_t* processor_count) {
    if (processor_count) *processor_count = 0;
    return AMDSMI_STATUS_NOT_SUPPORTED;
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
amdsmi_get_ainic_info(amdsmi_processor_handle /*processor_handle*/,
                      void *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_asic_info(amdsmi_processor_handle /*processor_handle*/,
                                         amdsmi_nic_asic_info_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_bus_info(amdsmi_processor_handle /*processor_handle*/,
                                        amdsmi_nic_bus_info_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_driver_info(amdsmi_processor_handle /*processor_handle*/,
                                           amdsmi_nic_driver_info_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_numa_info(amdsmi_processor_handle /*processor_handle*/,
                                         amdsmi_nic_numa_info_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_port_info(amdsmi_processor_handle /*processor_handle*/,
                                         amdsmi_nic_port_info_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_rdma_dev_info(amdsmi_processor_handle /*processor_handle*/,
                                             amdsmi_nic_rdma_devices_info_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}


#ifdef BRCM_NIC
amdsmi_status_t amdsmi_get_nic_info(amdsmi_processor_handle /*processor_handle*/,
                                    amdsmi_brcm_nic_info_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_temp_info(amdsmi_processor_handle /*processor_handle*/,
                                         amdsmi_brcm_nic_temperature_metric_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_power_info(amdsmi_processor_handle /*processor_handle*/,
                                          amdsmi_brcm_nic_hwmon_power_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_device_info(amdsmi_processor_handle /*processor_handle*/,
                                           amdsmi_brcm_nic_hwmon_device_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_metrics_info(amdsmi_processor_handle /*processor_handle*/,
                                            amdsmi_brcm_nic_hwmon_metrics_t *metrics) {
    if (metrics == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_device_bdf(amdsmi_processor_handle /*processor_handle*/,
                                             amdsmi_bdf_t *bdf) {
    if (bdf == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_link_info(amdsmi_processor_handle /*processor_handle*/,
                                            amdsmi_brcm_switch_link_metric_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_power_info(amdsmi_processor_handle /*processor_handle*/,
                                             amdsmi_brcm_switch_power_metric_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_device_info(amdsmi_processor_handle /*processor_handle*/,
                                              amdsmi_brcm_switch_device_metric_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_metrics_info(amdsmi_processor_handle /*processor_handle*/,
                                               amdsmi_brcm_switch_metric_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_fw_info(amdsmi_processor_handle /*processor_handle*/,
                                       amdsmi_brcm_nic_firmware_t *info) {
    if (info == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
#endif//BRCM_NIC

amdsmi_status_t amdsmi_get_nic_rdma_port_statistics(
    amdsmi_processor_handle /*processor_handle*/,
    uint32_t /*rdma_port_index*/,
    uint32_t *num_stats,
    amdsmi_nic_stat_t *stats) {
    if (num_stats == nullptr) { return AMDSMI_STATUS_INVAL; }
    if (stats == nullptr && *num_stats > 0) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t
amdsmi_get_gpu_enumeration_info(amdsmi_processor_handle /*processor_handle*/,
                                amdsmi_enumeration_info_t *info) {
    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t amdsmi_get_npm_info(amdsmi_node_handle /*node_handle*/,
                            amdsmi_npm_info_t *npm_info) {
    if (npm_info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t amdsmi_get_violation_status(amdsmi_processor_handle /*processor_handle*/,
            amdsmi_violation_status_t *violation_status) {
    if (violation_status == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_fan_rpms(amdsmi_processor_handle processor_handle,
                            uint32_t /*sensor_ind*/, int64_t *speed) {
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    GpuMetricsInfo m{};
    auto code = device->QueryGpuMetricsInfo(&m);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    if (m.current_fan_speed == UINT32_MAX) return AMDSMI_STATUS_NOT_SUPPORTED;
    if (speed == nullptr) return AMDSMI_STATUS_INVAL;
    *speed = static_cast<int64_t>(m.current_fan_speed);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_fan_speed(amdsmi_processor_handle processor_handle,
                                        uint32_t /*sensor_ind*/, int64_t *speed) {
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    GpuMetricsInfo m{};
    auto code = device->QueryGpuMetricsInfo(&m);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    if (m.current_fan_speed_percent == UINT32_MAX) return AMDSMI_STATUS_NOT_SUPPORTED;
    if (speed == nullptr) return AMDSMI_STATUS_INVAL;
    // amdsmi fan speed is expressed as a PWM value 0-255 (percentage * 255 / 100)
    *speed = static_cast<int64_t>(m.current_fan_speed_percent) * 255 / 100;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_fan_speed_max(amdsmi_processor_handle /*processor_handle*/,
                                    uint32_t /*sensor_ind*/, uint64_t *max_speed) {
    if (max_speed == nullptr) return AMDSMI_STATUS_INVAL;
    // Fan speed max is always 255 (full PWM range) when using percentage-based sensors
    *max_speed = 255;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_reset_gpu_fan(amdsmi_processor_handle /*processor_handle*/,
                                    uint32_t /*sensor_ind*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_gpu_fan_speed(amdsmi_processor_handle /*processor_handle*/,
                                uint32_t /*sensor_ind*/, uint64_t /*speed*/) {
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
amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle /*processor_handle*/,
                                amdsmi_xgmi_link_status_t *link_status) {
    if (link_status == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_kfd_info(amdsmi_processor_handle /*processor_handle*/,
                                    amdsmi_kfd_info_t *info) {
    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
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
    AMDSMI_CHECK_INIT();
    if (name == nullptr || len == 0)
        return AMDSMI_STATUS_INVAL;
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;
    wsl::thunk::AsicInfo ai = {};
    auto code = device->QueryAsicInfo(&ai);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);
    strncpy(name, ai.market_name, len - 1);
    name[len - 1] = '\0';
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_vendor_name(
            amdsmi_processor_handle processor_handle, char *name, size_t len) {
    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || name == nullptr)
        return AMDSMI_STATUS_INVAL;
    strncpy(name, "Advanced Micro Devices, Inc. [AMD/ATI]", len);
    return AMDSMI_STATUS_SUCCESS;
}


amdsmi_status_t amdsmi_get_gpu_vram_vendor(amdsmi_processor_handle /*processor_handle*/,
                                     char *brand, uint32_t len) {
    if (brand == nullptr || len == 0) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
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
amdsmi_init_gpu_event_notification(amdsmi_processor_handle /*processor_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle /*processor_handle*/,
          uint64_t /*mask*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_event_notification(int /*timeout_ms*/,
                    uint32_t */*num_elem*/, amdsmi_evt_notification_data_t */*data*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_stop_gpu_event_notification(
                amdsmi_processor_handle /*processor_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_gpu_counter_group_supported(
        amdsmi_processor_handle /*processor_handle*/, amdsmi_event_group_t /*group*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_gpu_create_counter(amdsmi_processor_handle /*processor_handle*/,
        amdsmi_event_type_t /*type*/, amdsmi_event_handle_t */*evnt_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_gpu_destroy_counter(amdsmi_event_handle_t /*evnt_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_gpu_control_counter(amdsmi_event_handle_t /*evt_handle*/,
                                amdsmi_counter_command_t /*cmd*/, void */*cmd_args*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_gpu_read_counter(amdsmi_event_handle_t /*evt_handle*/,
                            amdsmi_counter_value_t */*value*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
 amdsmi_get_gpu_available_counters(amdsmi_processor_handle /*processor_handle*/,
                            amdsmi_event_group_t /*grp*/, uint32_t */*available*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_topo_get_numa_node_number(amdsmi_processor_handle /*processor_handle*/, uint32_t */*numa_node*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_topo_get_link_weight(amdsmi_processor_handle /*processor_handle_src*/, amdsmi_processor_handle /*processor_handle_dst*/,
                          uint64_t */*weight*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
 amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle /*processor_handle_src*/, amdsmi_processor_handle /*processor_handle_dst*/,
                          uint64_t */*min_bandwidth*/, uint64_t */*max_bandwidth*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}


amdsmi_status_t amdsmi_get_link_metrics(amdsmi_processor_handle /*processor_handle*/,
          amdsmi_link_metrics_t *link_metrics) {
    if (link_metrics == nullptr) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t
amdsmi_topo_get_link_type(amdsmi_processor_handle /*processor_handle_src*/, amdsmi_processor_handle /*processor_handle_dst*/,
                        uint64_t */*hops*/, amdsmi_link_type_t */*type*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_is_P2P_accessible(amdsmi_processor_handle /*processor_handle_src*/,
                amdsmi_processor_handle /*processor_handle_dst*/,
                       bool */*accessible*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_topo_get_p2p_status(amdsmi_processor_handle /*processor_handle_src*/,
                           amdsmi_processor_handle /*processor_handle_dst*/,
                           amdsmi_link_type_t */*type*/, amdsmi_p2p_capability_t */*cap*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

// Compute Partition functions
amdsmi_status_t
amdsmi_get_gpu_compute_partition(amdsmi_processor_handle /*processor_handle*/,
                                  char *compute_partition, uint32_t len) {
    if (compute_partition == nullptr || len == 0) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_compute_partition(amdsmi_processor_handle /*processor_handle*/,
                                  amdsmi_compute_partition_type_t /*compute_partition*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

// Memory Partition functions
amdsmi_status_t
amdsmi_get_gpu_memory_partition(amdsmi_processor_handle /*processor_handle*/,
                                  char *memory_partition, uint32_t len) {
    if (memory_partition == nullptr || len == 0) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_memory_partition(amdsmi_processor_handle /*processor_handle*/,
                                  amdsmi_memory_partition_type_t /*memory_partition*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_memory_partition_config(amdsmi_processor_handle /*processor_handle*/,
                                        amdsmi_memory_partition_config_t *config) {
    if (config == nullptr) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_memory_partition_mode(amdsmi_processor_handle processor_handle,
                                     amdsmi_memory_partition_type_t mode) {
    AMDSMI_CHECK_INIT();
    return amdsmi_set_gpu_memory_partition(processor_handle, mode);
}

// Accelerator Partition functions
amdsmi_status_t
amdsmi_get_gpu_accelerator_partition_profile_config(amdsmi_processor_handle /*processor_handle*/,
                                  amdsmi_accelerator_partition_profile_config_t *profile_config) {
    if (profile_config == nullptr) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_accelerator_partition_profile(amdsmi_processor_handle /*processor_handle*/,
                                             amdsmi_accelerator_partition_profile_t *profile,
                                             uint32_t *partition_id) {
    if (profile == nullptr || partition_id == nullptr) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_accelerator_partition_profile(amdsmi_processor_handle /*processor_handle*/,
                                            uint32_t /*profile_index*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t
amdsmi_get_xgmi_info(amdsmi_processor_handle /*processor_handle*/, amdsmi_xgmi_info_t *info) {
    if (info == nullptr) return AMDSMI_STATUS_INVAL;
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_gpu_xgmi_error_status(amdsmi_processor_handle /*processor_handle*/, amdsmi_xgmi_status_t */*status*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_reset_gpu_xgmi_error(amdsmi_processor_handle /*processor_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t  amdsmi_get_gpu_ecc_count(amdsmi_processor_handle /*processor_handle*/,
                        amdsmi_gpu_block_t /*block*/, amdsmi_error_count_t */*ec*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t  amdsmi_get_gpu_ecc_enabled(amdsmi_processor_handle /*processor_handle*/,
                                                    uint64_t */*enabled_blocks*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}
amdsmi_status_t  amdsmi_get_gpu_ecc_status(amdsmi_processor_handle /*processor_handle*/,
                                amdsmi_gpu_block_t /*block*/,
                                amdsmi_ras_err_state_t */*state*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_metrics_header_info(amdsmi_processor_handle /*processor_handle*/,
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
        amdsmi_processor_handle /*processor_handle*/,
        amdsmi_gpu_metrics_t */*pgpu_metrics*/) {
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
                      amdsmi_processor_handle /*processor_handle*/,
                      amdsmi_name_value_t** /*pm_metrics*/,
                      uint32_t */*num_of_metrics*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_reg_table_info(
                      amdsmi_processor_handle /*processor_handle*/,
                      amdsmi_reg_type_t /*reg_type*/,
                      amdsmi_name_value_t** /*reg_metrics*/,
                      uint32_t */*num_of_metrics*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

void amdsmi_free_name_value_pairs(void *p) {
    if (p)
        free(p);
    return;
}

amdsmi_status_t
amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle,
                          uint32_t /*sensor_ind*/,
                          amdsmi_power_cap_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr)
        return AMDSMI_STATUS_INVAL;

    memset(info, 0, sizeof(amdsmi_power_cap_info_t));

    wsl::thunk::PowerInfo pi = {};
    auto code = device->QueryPowerInfo(&pi);
    if (code != ErrorCode::Success)
        return translateCodeToSmiStatus(code);

    info->power_cap     = pi.power_limit;
    info->max_power_cap = pi.power_limit;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_set_power_cap(amdsmi_processor_handle /*processor_handle*/,
            uint32_t /*sensor_ind*/, uint64_t /*cap*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_supported_power_cap(amdsmi_processor_handle /*processor_handle*/, uint32_t */*sensor_count*/,
                                 uint32_t */*sensor_inds*/, amdsmi_power_cap_type_t */*sensor_types*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_power_profile_presets(amdsmi_processor_handle /*processor_handle*/,
                        uint32_t /*sensor_ind*/,
                        amdsmi_power_profile_status_t */*status*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_gpu_perf_determinism_mode(
            amdsmi_processor_handle /*processor_handle*/, uint64_t /*clkvalue*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_power_profile(amdsmi_processor_handle /*processor_handle*/,
        uint32_t /*reserved*/, amdsmi_power_profile_preset_masks_t /*profile*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_perf_level(amdsmi_processor_handle /*processor_handle*/,
                                        amdsmi_dev_perf_level_t */*perf*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
 amdsmi_set_gpu_perf_level(amdsmi_processor_handle /*processor_handle*/,
                amdsmi_dev_perf_level_t /*perf_lvl*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_set_gpu_pci_bandwidth(amdsmi_processor_handle /*processor_handle*/,
                uint64_t /*bw_bitmask*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t  amdsmi_set_clk_freq(amdsmi_processor_handle /*processor_handle*/,
                         amdsmi_clk_type_t /*clk_type*/, uint64_t /*freq_bitmask*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_soc_pstate(amdsmi_processor_handle /*processor_handle*/,
                         uint32_t /*policy*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_soc_pstate(amdsmi_processor_handle /*processor_handle*/,
                         amdsmi_dpm_policy_t* /*policy*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_xgmi_plpd(amdsmi_processor_handle /*processor_handle*/,
                         uint32_t /*policy*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_xgmi_plpd(amdsmi_processor_handle /*processor_handle*/,
                         amdsmi_dpm_policy_t* /*policy*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_process_isolation(amdsmi_processor_handle /*processor_handle*/,
                             uint32_t* /*pisolate*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_gpu_process_isolation(amdsmi_processor_handle /*processor_handle*/,
                             uint32_t /*pisolate*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_clean_gpu_local_data(amdsmi_processor_handle /*processor_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_memory_reserved_pages(amdsmi_processor_handle /*processor_handle*/,
                                    uint32_t */*num_pages*/,
                                    amdsmi_retired_page_record_t */*records*/) {
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
            amdsmi_processor_handle /*processor_handle*/,
            uint32_t */*od*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_mem_overdrive_level(
            amdsmi_processor_handle /*processor_handle*/,
            uint32_t */*od*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_set_gpu_overdrive_level(
            amdsmi_processor_handle /*processor_handle*/, uint32_t /*od*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_get_gpu_pci_replay_counter(
            amdsmi_processor_handle processor_handle, uint64_t *counter) {
    AMDSMI_CHECK_INIT();
    if (counter == nullptr) { return AMDSMI_STATUS_INVAL; }
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    wsl::thunk::PCIeInfo pi = {};
    auto code = device->QueryPCIeInfo(&pi);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    *counter = pi.pcie_replay_count;
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_pci_throughput(
        amdsmi_processor_handle /*processor_handle*/,
        uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz) {
    if (sent == nullptr || received == nullptr || max_pkt_sz == nullptr) { return AMDSMI_STATUS_INVAL; }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_get_gpu_od_volt_info(amdsmi_processor_handle /*processor_handle*/,
                                            amdsmi_od_volt_freq_data_t */*odv*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_get_gpu_od_volt_curve_regions(
                    amdsmi_processor_handle /*processor_handle*/,
                    uint32_t */*num_regions*/, amdsmi_freq_volt_region_t */*buffer*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle,
                            amdsmi_voltage_type_t sensor_type,
                            amdsmi_voltage_metric_t metric, int64_t *voltage) {
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    // Check supported argument combinations and device capability before
    // validating the output pointer so unsupported requests keep returning
    // NOT_SUPPORTED, even when callers probe with a nullptr output buffer.
    if (metric != AMDSMI_VOLT_CURRENT) return AMDSMI_STATUS_NOT_SUPPORTED;
    if (sensor_type != AMDSMI_VOLT_TYPE_VDDGFX) return AMDSMI_STATUS_NOT_SUPPORTED;
    GpuMetricsInfo m{};
    auto code = device->QueryGpuMetricsInfo(&m);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    if (m.voltage_gfx == UINT32_MAX) return AMDSMI_STATUS_NOT_SUPPORTED;
    if (voltage == nullptr) return AMDSMI_STATUS_INVAL;
    *voltage = static_cast<int64_t>(m.voltage_gfx);
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t  amdsmi_set_gpu_od_clk_info(amdsmi_processor_handle /*processor_handle*/,
                                        amdsmi_freq_ind_t /*level*/,
                                       uint64_t /*clkvalue*/,
                                       amdsmi_clk_type_t /*clkType*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t  amdsmi_set_gpu_od_volt_info(amdsmi_processor_handle /*processor_handle*/,
                    uint32_t /*vpoint*/, uint64_t /*clkvalue*/, uint64_t /*voltvalue*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_gpu_clk_range(amdsmi_processor_handle /*processor_handle*/,
                                    uint64_t /*minclkvalue*/,
                                    uint64_t /*maxclkvalue*/,
                                    amdsmi_clk_type_t /*clkType*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle /*processor_handle*/,
                                         amdsmi_clk_type_t /*clk_type*/,
                                          amdsmi_clk_limit_type_t /*limit_type*/,
                                          uint64_t /*clk_value*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle /*processor_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_gpu_driver_reload(void) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t amdsmi_get_energy_count(amdsmi_processor_handle /*processor_handle*/,
            uint64_t *energy_accumulator, float *counter_resolution, uint64_t *timestamp) {
    if (energy_accumulator == nullptr || counter_resolution == nullptr || timestamp == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_bdf_id(
        amdsmi_processor_handle processor_handle, uint64_t *bdfid) {
    AMDSMI_CHECK_INIT();
    if (bdfid == nullptr) { return AMDSMI_STATUS_INVAL; }
    auto *device = reinterpret_cast<Device *>(processor_handle);
    if (device == nullptr) return AMDSMI_STATUS_INVAL;
    wsl::thunk::BdfInfo bi = {};
    auto code = device->QueryBdfInfo(&bi);
    if (code != ErrorCode::Success) return translateCodeToSmiStatus(code);
    *bdfid = ((uint64_t)(bi.domain_number & 0xFFFFFFFFU) << 32)
           | ((uint64_t)(bi.bus_number    & 0xFF) << 8)
           | ((uint64_t)(bi.device_number & 0x1F) << 3)
           | ((uint64_t)(bi.function_number & 0x7));
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_topo_numa_affinity(
    amdsmi_processor_handle /*processor_handle*/, int32_t * /*numa_node*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_topo_cpu_affinity(amdsmi_processor_handle /*processor_handle*/,
                                           unsigned int * /*cpu_aff_length*/, char * /*cpu_aff_data*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

#ifdef BRCM_NIC
amdsmi_status_t amdsmi_get_nic_gpu_topo_info(amdsmi_processor_handle /*nic_processor_handle*/,
                    amdsmi_processor_handle /*gpu_processor_handle*/,
                    size_t * /*topo_info_length*/, char * /*topo_info*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_root_switch(amdsmi_bdf_t /*devicehBdf*/, amdsmi_bdf_t * /*switchBdf*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_topo_numa_affinity(
    amdsmi_processor_handle /*processor_handle*/, int32_t * /*numa_node*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_nic_topo_cpu_affinity(amdsmi_processor_handle /*processor_handle*/,
                                           unsigned int * /*cpu_aff_length*/, char * /*cpu_aff_data*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_topo_numa_affinity(
    amdsmi_processor_handle /*processor_handle*/, int32_t * /*numa_node*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_topo_cpu_affinity(amdsmi_processor_handle /*processor_handle*/,
                                           size_t * /*cpu_aff_length*/, char * /*cpu_aff_data*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t amdsmi_is_gpu_power_management_enabled(
        amdsmi_processor_handle /*processor_handle*/, bool * /*enabled*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
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
amdsmi_get_gpu_ras_block_features_enabled(amdsmi_processor_handle /*processor_handle*/,
        amdsmi_gpu_block_t /*block*/, amdsmi_ras_err_state_t * /*state*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_bad_page_info(amdsmi_processor_handle /*processor_handle*/,
        uint32_t * /*num_pages*/, amdsmi_retired_page_record_t * /*info*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_bad_page_threshold(amdsmi_processor_handle /*processor_handle*/,
        uint32_t * /*threshold*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_gpu_validate_ras_eeprom(amdsmi_processor_handle /*processor_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_gpu_ras_feature_info(
  amdsmi_processor_handle /*processor_handle*/, amdsmi_ras_feature_t * /*ras_feature*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_total_ecc_count(amdsmi_processor_handle /*processor_handle*/, amdsmi_error_count_t * /*ec*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_cper_entries(
    amdsmi_processor_handle /*processor_handle*/,
    uint32_t /*severity_mask*/,
    char * /*cper_data*/,
    uint64_t * /*buf_size*/,
    amdsmi_cper_hdr_t ** /*cper_hdrs*/,
    uint64_t * /*entry_count*/,
    uint64_t * /*cursor*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_afids_from_cper(
            char* /*cper_buffer*/, uint32_t /*buf_size*/, uint64_t* /*afids*/, uint32_t* /*num_afids*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t amdsmi_get_nic_device_uuid(amdsmi_processor_handle /*processor_handle*/,
                                           unsigned int */*uuid_length*/, char */*uuid*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_switch_device_uuid(amdsmi_processor_handle /*processor_handle*/,
                                           unsigned int */*uuid_length*/, char */*uuid*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
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

amdsmi_status_t amdsmi_get_gpu_xcd_counter(amdsmi_processor_handle /*processor_handle*/,
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
amdsmi_get_link_topology_nearest(amdsmi_processor_handle /*processor_handle*/,
                                 amdsmi_link_type_t /*link_type*/,
                                 amdsmi_topology_nearest_t* /*topology_nearest_info*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_virtualization_mode(amdsmi_processor_handle /*processor_handle*/,
                                    amdsmi_virtualization_mode_t * /*mode*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

// PTL

amdsmi_status_t
amdsmi_get_gpu_ptl_state(amdsmi_processor_handle /*processor_handle*/, bool * /*enabled*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_ptl_state(amdsmi_processor_handle /*processor_handle*/, bool /*enable*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_get_gpu_ptl_formats(amdsmi_processor_handle /*processor_handle*/,
                        amdsmi_ptl_data_format_t * /*data_format1*/,
                        amdsmi_ptl_data_format_t * /*data_format2*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
amdsmi_set_gpu_ptl_formats(amdsmi_processor_handle /*processor_handle*/,
                          amdsmi_ptl_data_format_t /*data_format1*/,
                          amdsmi_ptl_data_format_t /*data_format2*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_affinity_with_scope(amdsmi_processor_handle /*processor_handle*/,
            uint32_t /*cpu_set_size*/, uint64_t * /*cpu_set*/, amdsmi_affinity_scope_t /*scope*/)
{
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

#ifdef ENABLE_ESMI_LIB

amdsmi_status_t amdsmi_get_threads_per_core(uint32_t */*threads_per_core*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_hsmp_proto_ver(amdsmi_processor_handle /*processor_handle*/, uint32_t */*proto_ver*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_hsmp_driver_version(amdsmi_processor_handle /*processor_handle*/, amdsmi_hsmp_driver_version_t */*amdsmi_hsmp_driver_ver*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_smu_fw_version(amdsmi_processor_handle /*processor_handle*/, amdsmi_smu_fw_version_t */*amdsmi_smu_fw*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_core_energy(amdsmi_processor_handle /*processor_handle*/, uint64_t */*penergy*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_energy(amdsmi_processor_handle /*processor_handle*/, uint64_t */*penergy*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_prochot_status(amdsmi_processor_handle /*processor_handle*/, uint32_t */*prochot*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_fclk_mclk(amdsmi_processor_handle /*processor_handle*/, uint32_t */*fclk*/, uint32_t */*mclk*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_cclk_limit(amdsmi_processor_handle /*processor_handle*/, uint32_t */*cclk*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_current_active_freq_limit(amdsmi_processor_handle /*processor_handle*/, uint16_t */*freq*/, char **/*src_type*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_freq_range(amdsmi_processor_handle /*processor_handle*/, uint16_t */*fmax*/, uint16_t */*fmin*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_core_current_freq_limit(amdsmi_processor_handle /*processor_handle*/, uint32_t */*freq*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_power(amdsmi_processor_handle /*processor_handle*/, uint32_t */*ppower*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_power_cap(amdsmi_processor_handle /*processor_handle*/, uint32_t */*pcap*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_power_cap_max(amdsmi_processor_handle /*processor_handle*/, uint32_t */*pmax*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_pwr_svi_telemetry_all_rails(amdsmi_processor_handle /*processor_handle*/, uint32_t */*power*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_socket_power_cap(amdsmi_processor_handle /*processor_handle*/, uint32_t /*pcap*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_pwr_efficiency_mode(amdsmi_processor_handle /*processor_handle*/, uint8_t /*mode*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_core_boostlimit(amdsmi_processor_handle /*processor_handle*/, uint32_t */*pboostlimit*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_c0_residency(amdsmi_processor_handle /*processor_handle*/, uint32_t */*pc0_residency*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_core_boostlimit(amdsmi_processor_handle /*processor_handle*/, uint32_t /*boostlimit*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_socket_boostlimit(amdsmi_processor_handle /*processor_handle*/, uint32_t /*boostlimit*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_ddr_bw(amdsmi_processor_handle /*processor_handle*/, amdsmi_ddr_bw_metrics_t */*ddr_bw*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_temperature(amdsmi_processor_handle /*processor_handle*/, uint32_t */*ptmon*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_dimm_temp_range_and_refresh_rate( amdsmi_processor_handle /*processor_handle*/, uint8_t /*dimm_addr*/, amdsmi_temp_range_refresh_rate_t */*rate*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_dimm_power_consumption(amdsmi_processor_handle /*processor_handle*/, uint8_t /*dimm_addr*/, amdsmi_dimm_power_t */*dimm_pow*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_dimm_thermal_sensor(amdsmi_processor_handle /*processor_handle*/, uint8_t /*dimm_addr*/, amdsmi_dimm_thermal_t */*dimm_temp*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_xgmi_width(amdsmi_processor_handle /*processor_handle*/, uint8_t /*min*/, uint8_t /*max*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_gmi3_link_width_range(amdsmi_processor_handle /*processor_handle*/, uint8_t /*min_link_width*/, uint8_t /*max_link_width*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_cpu_apb_enable(amdsmi_processor_handle /*processor_handle*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_cpu_apb_disable(amdsmi_processor_handle /*processor_handle*/, uint8_t /*pstate*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_socket_lclk_dpm_level(amdsmi_processor_handle /*processor_handle*/, uint8_t /*nbio_id*/, uint8_t /*min*/, uint8_t /*max*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_lclk_dpm_level(amdsmi_processor_handle /*processor_handle*/, uint8_t /*nbio_id*/, amdsmi_dpm_level_t */*nbio*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_pcie_link_rate(amdsmi_processor_handle /*processor_handle*/, uint8_t /*rate_ctrl*/, uint8_t */*prev_mode*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_df_pstate_range(amdsmi_processor_handle /*processor_handle*/, uint8_t /*max_pstate*/, uint8_t /*min_pstate*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_current_io_bandwidth(amdsmi_processor_handle /*processor_handle*/, amdsmi_link_id_bw_type_t /*link*/, uint32_t */*io_bw*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_current_xgmi_bw(amdsmi_processor_handle /*processor_handle*/, amdsmi_link_id_bw_type_t /*link*/, uint32_t */*xgmi_bw*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_hsmp_metrics_table_version(amdsmi_processor_handle /*processor_handle*/, uint32_t */*metrics_version*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_hsmp_metrics_table(amdsmi_processor_handle /*processor_handle*/, amdsmi_hsmp_metrics_table_t */*metrics_table*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_first_online_core_on_cpu_socket(amdsmi_processor_handle /*processor_handle*/, uint32_t */*pcore_ind*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_family(uint32_t */*cpu_family*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_model(uint32_t */*cpu_model*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_model_name(amdsmi_processor_handle /*processor_handle*/, amdsmi_cpu_info_t */*cpu_info*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_cores_per_socket(uint32_t /*sock_count*/, amdsmi_sock_info_t */*sock_info*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_socket_count(uint32_t */*sock_count*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_handles(uint32_t */*cpu_count*/, amdsmi_processor_handle */*processor_handles*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpucore_handles(uint32_t */*cores_count*/, amdsmi_processor_handle* /*processor_handles*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_esmi_err_msg(amdsmi_status_t /*status*/, const char **/*status_string*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_cpu_rail_isofreq_policy(amdsmi_processor_handle /*processor_handle*/, uint8_t /*input*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_cpu_rail_isofreq_policy(amdsmi_processor_handle /*processor_handle*/, uint8_t */*cpurailiso*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_set_dfc_ctrl(amdsmi_processor_handle /*processor_handle*/, bool /*dfc_ctrl*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t amdsmi_get_dfc_ctrl(amdsmi_processor_handle /*processor_handle*/, uint8_t */*dfc_ctrl*/) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
}

#endif  // ENABLE_ESMI_LIB
