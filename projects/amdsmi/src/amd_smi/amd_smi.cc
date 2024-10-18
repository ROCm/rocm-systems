/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#include <assert.h>
#include <errno.h>
#include <sys/utsname.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <queue>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <limits>
#include <functional>
#include <xf86drm.h>
#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/fdinfo.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_uuid.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_common.h"
#include "amd_smi/impl/amdgpu_drm.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "rocm_smi/rocm_smi_logger.h"

static bool initialized_lib = false;

#define	SIZE	10
char proc_id[SIZE] = "\0";

#define AMDSMI_CHECK_INIT() do { \
	if (!initialized_lib) { \
		return AMDSMI_STATUS_NOT_INIT; \
	} \
} while (0)

static amdsmi_status_t get_gpu_device_from_handle(amdsmi_processor_handle processor_handle,
            amd::smi::AMDSmiGPUDevice** gpudevice) {

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || gpudevice == nullptr)
        return AMDSMI_STATUS_INVAL;

    amd::smi::AMDSmiProcessor* device = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_processor(processor_handle, &device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    if (device->get_processor_type() == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        *gpudevice = static_cast<amd::smi::AMDSmiGPUDevice*>(processor_handle);
        return AMDSMI_STATUS_SUCCESS;
    }

    return AMDSMI_STATUS_NOT_SUPPORTED;
}

template <typename F, typename ...Args>
amdsmi_status_t rsmi_wrapper(F && f,
    amdsmi_processor_handle processor_handle, Args &&... args) {

    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS) return r;


    uint32_t gpu_index = gpu_device->get_gpu_id();
    auto rstatus = std::forward<F>(f)(gpu_index,
                    std::forward<Args>(args)...);
    r = amd::smi::rsmi_to_amdsmi_status(rstatus);
    std::ostringstream ss;
    const char *status_string;
    amdsmi_status_code_to_string(r, &status_string);
    ss << __PRETTY_FUNCTION__ << " | returning status = " << status_string;
    LOG_INFO(ss);
    return r;
}

amdsmi_status_t
amdsmi_init(uint64_t flags) {
    if (initialized_lib)
        return AMDSMI_STATUS_SUCCESS;

    amdsmi_status_t status = amd::smi::AMDSmiSystem::getInstance().init(flags);
    if (status == AMDSMI_STATUS_SUCCESS) {
        initialized_lib = true;
    }
    return status;
}

amdsmi_status_t
amdsmi_shut_down() {
    if (!initialized_lib)
        return AMDSMI_STATUS_SUCCESS;
    amdsmi_status_t status = amd::smi::AMDSmiSystem::getInstance().cleanup();
    if (status == AMDSMI_STATUS_SUCCESS) {
        initialized_lib = false;
    }
    return status;
}

amdsmi_status_t
amdsmi_status_code_to_string(amdsmi_status_t status, const char **status_string) {
    switch (status) {
        case AMDSMI_STATUS_FAIL_LOAD_MODULE:
            *status_string = "FAIL_LOAD_MODULE: Fail to load module.";
            break;
        case AMDSMI_STATUS_FAIL_LOAD_SYMBOL:
            *status_string = "FAIL_LOAD_SYMBOL: Fail to load symbol.";
            break;
        case AMDSMI_STATUS_DRM_ERROR:
            *status_string = "DRM_ERROR: Fail to run function in libdrm.";
            break;
        default:
            // The case above didn't have a match, so look up the amdsmi status in the rsmi status map
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

    std::vector<amd::smi::AMDSmiSocket*>& sockets
            = amd::smi::AMDSmiSystem::getInstance().get_sockets();
    uint32_t socket_size = static_cast<uint32_t>(sockets.size());
    // Get the socket size
    if (socket_handles == nullptr) {
        *socket_count = socket_size;
        return AMDSMI_STATUS_SUCCESS;
    }

    // If the socket_handles can hold all sockets, return all of them.
    *socket_count = *socket_count >= socket_size ? socket_size : *socket_count;

    // Copy the socket handles
    for (uint32_t i = 0; i < *socket_count; i++) {
        socket_handles[i] = reinterpret_cast<amdsmi_socket_handle>(sockets[i]);
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


    amd::smi::AMDSmiSocket* socket = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_socket(socket_handle, &socket);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    strncpy(name, socket->get_socket_id().c_str(), len);

    return AMDSMI_STATUS_SUCCESS;
}

#ifdef ENABLE_ESMI_LIB
amdsmi_status_t amdsmi_get_processor_info(
                amdsmi_processor_handle processor_handle,
                size_t len, char *name) {
    char proc_id[10];
    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr || name == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiProcessor* processor = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_processor(processor_handle, &processor);
    if (r != AMDSMI_STATUS_SUCCESS) return r;

    sprintf(proc_id, "%d", processor->get_processor_index());
    strncpy(name, proc_id, len);

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

    // Get the socket object via socket handle.
    amd::smi::AMDSmiSocket* socket = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_socket(socket_handle, &socket);
    if (r != AMDSMI_STATUS_SUCCESS) return r;


    std::vector<amd::smi::AMDSmiProcessor*>& processors = socket->get_processors();
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
        amdsmi_status_t r = amdsmi_get_processor_type(processor_handles[i],
                                                      &processor_type);
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
amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle ,
              processor_type_t* processor_type) {

    AMDSMI_CHECK_INIT();

    if (processor_type == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    amd::smi::AMDSmiProcessor* processor = nullptr;
    amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
                    .handle_to_processor(processor_handle, &processor);
    if (r != AMDSMI_STATUS_SUCCESS) return r;
    *processor_type = processor->get_processor_type();

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_device_bdf(amdsmi_processor_handle processor_handle, amdsmi_bdf_t *bdf) {

    AMDSMI_CHECK_INIT();

    if (bdf == NULL) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    // get bdf from sysfs file
    *bdf = gpu_device->get_bdf();

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_board_info(amdsmi_processor_handle processor_handle, amdsmi_board_info_t *board_info) {
    AMDSMI_CHECK_INIT();

    if (board_info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amdsmi_status_t status;
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    if (gpu_device->check_if_drm_is_supported()) {
        // Populate product_serial, product_name, & product_number from sysfs
        status = smi_amdgpu_get_board_info(gpu_device, board_info);
    } else {
        // ignore the errors so that it can populate as many fields as possible.
        // call rocm-smi which search multiple places for device name
        status = rsmi_wrapper(rsmi_dev_name_get, processor_handle,
                        board_info->product_name, AMDSMI_256_LENGTH);

        status = rsmi_wrapper(rsmi_dev_serial_number_get, processor_handle,
                        board_info->product_serial, AMDSMI_NORMAL_STRING_LENGTH);
    }

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "[Before rocm smi correction] "
       << "Returning status = AMDSMI_STATUS_SUCCESS"
       << "\n; info->model_number: |" << board_info->model_number << "|"
       << "\n; info->product_serial: |" << board_info->product_serial << "|"
       << "\n; info->fru_id: |" << board_info->fru_id << "|"
       << "\n; info->manufacturer_name: |" << board_info->manufacturer_name << "|"
       << "\n; info->product_name: |" << board_info->product_name << "|";
    LOG_INFO(ss);

    if (board_info->product_serial[0] == '\0') {
        status = rsmi_wrapper(rsmi_dev_serial_number_get, processor_handle,
                              board_info->product_serial, AMDSMI_NORMAL_STRING_LENGTH);
        if (status != AMDSMI_STATUS_SUCCESS) {
            memset(board_info->product_serial, 0,
                   AMDSMI_NORMAL_STRING_LENGTH * sizeof(board_info->product_serial[0]));
        }
        ss << __PRETTY_FUNCTION__ << " | [rsmi_correction] board_info->product_serial= |"
        << board_info->product_serial << "|";
        LOG_INFO(ss);
    }

    if (board_info->product_name[0] == '\0') {
        status = rsmi_wrapper(rsmi_dev_name_get,
                              processor_handle, board_info->product_name,
                              AMDSMI_256_LENGTH);
        // Check if the value is in hex format
        if (status == AMDSMI_STATUS_SUCCESS) {
            if (board_info->product_name[0] == '0' && board_info->product_name[1] == 'x') {
                memset(board_info->product_name, 0,
                        AMDSMI_256_LENGTH * sizeof(board_info->product_name[0]));
            }
        }
        if (status != AMDSMI_STATUS_SUCCESS) {
            memset(board_info->product_name, 0,
                    AMDSMI_256_LENGTH * sizeof(board_info->product_name[0]));
        }
        ss << __PRETTY_FUNCTION__ << " | [rsmi_correction] board_info->product_name= |"
        << board_info->product_name << "|";
        LOG_INFO(ss);
    }

    if (board_info->manufacturer_name[0] == '\0') {
        status = rsmi_wrapper(rsmi_dev_vendor_name_get,
                              processor_handle, board_info->manufacturer_name,
                              AMDSMI_MAX_STRING_LENGTH);
        if (status != AMDSMI_STATUS_SUCCESS) {
            memset(board_info->manufacturer_name, 0,
                   AMDSMI_MAX_STRING_LENGTH * sizeof(board_info->manufacturer_name[0]));
        }
        ss << __PRETTY_FUNCTION__ << " | [rsmi_correction] board_info->manufacturer_name= |"
        << board_info->manufacturer_name << "|";
        LOG_INFO(ss);
    }

    ss << __PRETTY_FUNCTION__ << " | [After rocm smi correction] "
       << "Returning status = AMDSMI_STATUS_SUCCESS"
       << "\n; info->model_number: |" << board_info->model_number << "|"
       << "\n; info->product_serial: |" << board_info->product_serial << "|"
       << "\n; info->fru_id: |" << board_info->fru_id << "|"
       << "\n; info->manufacturer_name: |" << board_info->manufacturer_name << "|"
       << "\n; info->product_name: |" << board_info->product_name << "|";
    LOG_INFO(ss);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_cache_info(
      amdsmi_processor_handle processor_handle, amdsmi_gpu_cache_info_t *info) {
    AMDSMI_CHECK_INIT();
    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t status = get_gpu_device_from_handle(
                        processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    rsmi_gpu_cache_info_t rsmi_info;
    status = rsmi_wrapper(rsmi_dev_cache_info_get,
                    processor_handle, &rsmi_info);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;
    // Sysfs cache type
    #define  HSA_CACHE_TYPE_DATA     0x00000001
    #define  HSA_CACHE_TYPE_INSTRUCTION  0x00000002
    #define  HSA_CACHE_TYPE_CPU      0x00000004
    #define  HSA_CACHE_TYPE_HSACU    0x00000008

    info->num_cache_types = rsmi_info.num_cache_types;
    for (unsigned int i =0; i < rsmi_info.num_cache_types; i++) {
        // convert from sysfs type to CRAT type(HSA Cache Affinity type)
        info->cache[i].cache_properties = 0;
        if (rsmi_info.cache[i].flags & HSA_CACHE_TYPE_DATA)
            info->cache[i].cache_properties |= AMDSMI_CACHE_PROPERTY_DATA_CACHE;
        if (rsmi_info.cache[i].flags & HSA_CACHE_TYPE_INSTRUCTION)
            info->cache[i].cache_properties |= AMDSMI_CACHE_PROPERTY_INST_CACHE;
        if (rsmi_info.cache[i].flags & HSA_CACHE_TYPE_CPU)
            info->cache[i].cache_properties |= AMDSMI_CACHE_PROPERTY_CPU_CACHE;
        if (rsmi_info.cache[i].flags & HSA_CACHE_TYPE_HSACU)
            info->cache[i].cache_properties |= AMDSMI_CACHE_PROPERTY_SIMD_CACHE;

        info->cache[i].cache_size = rsmi_info.cache[i].cache_size_kb;
        info->cache[i].cache_level = rsmi_info.cache[i].cache_level;
        info->cache[i].max_num_cu_shared = rsmi_info.cache[i].max_num_cu_shared;
        info->cache[i].num_cache_instance = rsmi_info.cache[i].num_cache_instance;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t  amdsmi_get_temp_metric(amdsmi_processor_handle processor_handle,
                    amdsmi_temperature_type_t sensor_type,
                    amdsmi_temperature_metric_t metric, int64_t *temperature) {

    AMDSMI_CHECK_INIT();

    if (temperature == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    // Get the PLX temperature from the gpu_metrics
    if (sensor_type == AMDSMI_TEMPERATURE_TYPE_PLX) {
        amdsmi_gpu_metrics_t metric_info;
        auto r_status =  amdsmi_get_gpu_metrics_info(
                processor_handle, &metric_info);
        if (r_status != AMDSMI_STATUS_SUCCESS)
            return r_status;
        *temperature = metric_info.temperature_vrsoc;
        return r_status;
    }
    amdsmi_status_t amdsmi_status = rsmi_wrapper(rsmi_dev_temp_metric_get, processor_handle,
            static_cast<uint32_t>(sensor_type),
            static_cast<rsmi_temperature_metric_t>(metric), temperature);
    *temperature /= 1000;
    return amdsmi_status;
}

amdsmi_status_t amdsmi_get_gpu_vram_usage(amdsmi_processor_handle processor_handle,
            amdsmi_vram_usage_t *vram_info) {

    AMDSMI_CHECK_INIT();

    if (vram_info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

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

    struct drm_amdgpu_info_vram_gtt gtt;
    uint64_t vram_used = 0;

    r = gpu_device->amdgpu_query_info(AMDGPU_INFO_VRAM_GTT,
                sizeof(struct drm_amdgpu_memory_info), &gtt);
    if (r != AMDSMI_STATUS_SUCCESS)  return r;

    vram_info->vram_total = static_cast<uint32_t>(
        gtt.vram_size / (1024 * 1024));

    r = gpu_device->amdgpu_query_info(AMDGPU_INFO_VRAM_USAGE,
                sizeof(vram_used), &vram_used);
    if (r != AMDSMI_STATUS_SUCCESS) {
        return r;
    }

    vram_info->vram_used = static_cast<uint32_t>(vram_used / (1024 * 1024));

    return AMDSMI_STATUS_SUCCESS;
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
    constexpr uint64_t kFASTEST_POLL_TIME_MS = 1;  // fastest SMU FW sample time is 1ms

    violation_status->reference_timestamp = std::numeric_limits<uint64_t>::max();
    violation_status->violation_timestamp = std::numeric_limits<uint64_t>::max();

    violation_status->acc_counter = std::numeric_limits<uint64_t>::max();
    violation_status->acc_prochot_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->acc_ppt_pwr = std::numeric_limits<uint64_t>::max();
    violation_status->acc_socket_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->acc_vr_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->acc_hbm_thrm = std::numeric_limits<uint64_t>::max();

    violation_status->per_prochot_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->per_ppt_pwr = std::numeric_limits<uint64_t>::max();
    violation_status->per_socket_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->per_vr_thrm = std::numeric_limits<uint64_t>::max();
    violation_status->per_hbm_thrm = std::numeric_limits<uint64_t>::max();

    violation_status->active_prochot_thrm = std::numeric_limits<uint8_t>::max();
    violation_status->active_ppt_pwr = std::numeric_limits<uint8_t>::max();
    violation_status->active_socket_thrm = std::numeric_limits<uint8_t>::max();
    violation_status->active_vr_thrm = std::numeric_limits<uint8_t>::max();
    violation_status->active_hbm_thrm = std::numeric_limits<uint8_t>::max();

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

    amdsmi_gpu_metrics_t metric_info_a = {};
    amdsmi_status_t status =  amdsmi_get_gpu_metrics_info(
            processor_handle, &metric_info_a);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    // if all of these values are "undefined" then the feature is not supported on the ASIC
    if (metric_info_a.accumulation_counter == std::numeric_limits<uint64_t>::max()
        && metric_info_a.prochot_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.ppt_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.socket_thm_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.vr_thm_residency_acc == std::numeric_limits<uint64_t>::max()
        && metric_info_a.hbm_thm_residency_acc == std::numeric_limits<uint64_t>::max()) {
        ss << __PRETTY_FUNCTION__
           << " | ASIC does not support throttle violations!, "
           << "returning AMDSMI_STATUS_NOT_SUPPORTED";
        LOG_INFO(ss);
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    // wait 1ms before reading again
    system_wait(static_cast<int>(kFASTEST_POLL_TIME_MS));

    amdsmi_gpu_metrics_t metric_info_b = {};
    status =  amdsmi_get_gpu_metrics_info(
            processor_handle, &metric_info_b);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    // Insert current accumulator counters into struct
    violation_status->acc_counter = metric_info_b.accumulation_counter;
    violation_status->acc_prochot_thrm = metric_info_b.prochot_residency_acc;
    violation_status->acc_ppt_pwr = metric_info_b.ppt_residency_acc;
    violation_status->acc_socket_thrm = metric_info_b.socket_thm_residency_acc;
    violation_status->acc_vr_thrm = metric_info_b.vr_thm_residency_acc;
    violation_status->acc_hbm_thrm = metric_info_b.hbm_thm_residency_acc;

    ss << __PRETTY_FUNCTION__ << " | "
       << "[gpu_metrics A] metric_info_a.accumulation_counter: " << std::dec
       << metric_info_a.accumulation_counter
       << "; metric_info_a.prochot_residency_acc: " << std::dec
       << metric_info_a.prochot_residency_acc
       << "; metric_info_a.ppt_residency_acc (pviol): " << std::dec
       << metric_info_a.ppt_residency_acc
       << "; metric_info_a.socket_thm_residency_acc (tviol): " << std::dec
       << metric_info_a.socket_thm_residency_acc
       << "; metric_info_a.vr_thm_residency_acc: " << std::dec
       << metric_info_a.vr_thm_residency_acc
       << "; metric_info_a.hbm_thm_residency_acc: " << std::dec
       << metric_info_a.hbm_thm_residency_acc << "\n"
       << " [gpu_metrics B] metric_info_b.accumulation_counter: " << std::dec
       << metric_info_b.accumulation_counter
       << "; metric_info_b.prochot_residency_acc: " << std::dec
       << metric_info_b.prochot_residency_acc
       << "; metric_info_b.ppt_residency_acc (pviol): " << std::dec
       << metric_info_b.ppt_residency_acc
       << "; metric_info_b.socket_thm_residency_acc (tviol): " << std::dec
       << metric_info_b.socket_thm_residency_acc
       << "; metric_info_b.vr_thm_residency_acc: " << std::dec
       << metric_info_b.vr_thm_residency_acc
       << "; metric_info_b.hbm_thm_residency_acc: " << std::dec
       << metric_info_b.hbm_thm_residency_acc
       << "\n";
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
            violation_status->violation_timestamp = kFASTEST_POLL_TIME_MS;
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
            violation_status->violation_timestamp = kFASTEST_POLL_TIME_MS;
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
            violation_status->violation_timestamp = kFASTEST_POLL_TIME_MS;
        } else {
            violation_status->active_socket_thrm = 0;
        }
        ss << __PRETTY_FUNCTION__ << " | "
           << "ENTERED socket_thm_residency_acc | per_socket_thrm: " << std::dec
           << violation_status->per_socket_thrm
           << "%; active_ppt_pwr = " << std::dec
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
            violation_status->violation_timestamp = kFASTEST_POLL_TIME_MS;
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
            violation_status->violation_timestamp = kFASTEST_POLL_TIME_MS;
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
       << "; violation_status->active_prochot_thrm (bool): " << std::dec
       << static_cast<int>(violation_status->active_prochot_thrm)
       << "; violation_status->active_ppt_pwr (bool): " << std::dec
       << static_cast<int>(violation_status->active_ppt_pwr)
       << "; violation_status->active_socket_thrm (bool): " << std::dec
       << static_cast<int>(violation_status->active_socket_thrm)
       << "; violation_status->active_vr_thrm (bool): " << std::dec
       << static_cast<int>(violation_status->active_vr_thrm)
       << "; violation_status->active_hbm_thrm (bool): " << std::dec
       << static_cast<int>(violation_status->active_hbm_thrm)
       << "\n";
    LOG_INFO(ss);

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_fan_rpms(amdsmi_processor_handle processor_handle,
                            uint32_t sensor_ind, int64_t *speed) {
    return rsmi_wrapper(rsmi_dev_fan_rpms_get, processor_handle, sensor_ind,
            speed);
}

amdsmi_status_t amdsmi_get_gpu_fan_speed(amdsmi_processor_handle processor_handle,
                                        uint32_t sensor_ind, int64_t *speed) {
    return rsmi_wrapper(rsmi_dev_fan_speed_get, processor_handle,
                        sensor_ind, speed);
}

amdsmi_status_t amdsmi_get_gpu_fan_speed_max(amdsmi_processor_handle processor_handle,
                                    uint32_t sensor_ind, uint64_t *max_speed) {
    return rsmi_wrapper(rsmi_dev_fan_speed_max_get, processor_handle,
                sensor_ind, max_speed);
}

amdsmi_status_t amdsmi_reset_gpu_fan(amdsmi_processor_handle processor_handle,
                                    uint32_t sensor_ind) {
    return rsmi_wrapper(rsmi_dev_fan_reset, processor_handle, sensor_ind);
}

amdsmi_status_t amdsmi_set_gpu_fan_speed(amdsmi_processor_handle processor_handle,
                                uint32_t sensor_ind, uint64_t speed) {
    return rsmi_wrapper(rsmi_dev_fan_speed_set, processor_handle,
                            sensor_ind, speed);
}

amdsmi_status_t amdsmi_get_gpu_id(amdsmi_processor_handle processor_handle,
                                uint16_t *id) {
    return rsmi_wrapper(rsmi_dev_id_get, processor_handle, id);
}

amdsmi_status_t amdsmi_get_gpu_revision(amdsmi_processor_handle processor_handle,
                                uint16_t *revision) {
    return rsmi_wrapper(rsmi_dev_revision_get, processor_handle, revision);
}

// TODO(bliu) : add fw info from libdrm
amdsmi_status_t amdsmi_get_fw_info(amdsmi_processor_handle processor_handle,
        amdsmi_fw_info_t *info) {
    const std::map<amdsmi_fw_block_t, rsmi_fw_block_t> fw_in_rsmi = {
        { AMDSMI_FW_ID_ASD, RSMI_FW_BLOCK_ASD},
        { AMDSMI_FW_ID_CP_CE, RSMI_FW_BLOCK_CE},
        { AMDSMI_FW_ID_DMCU, RSMI_FW_BLOCK_DMCU},
        { AMDSMI_FW_ID_MC, RSMI_FW_BLOCK_MC},
        { AMDSMI_FW_ID_CP_ME, RSMI_FW_BLOCK_ME},
        { AMDSMI_FW_ID_CP_MEC1, RSMI_FW_BLOCK_MEC},
        { AMDSMI_FW_ID_CP_MEC2, RSMI_FW_BLOCK_MEC2},
        { AMDSMI_FW_ID_CP_PFP, RSMI_FW_BLOCK_PFP},
        { AMDSMI_FW_ID_RLC, RSMI_FW_BLOCK_RLC},
        { AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL, RSMI_FW_BLOCK_RLC_SRLC},
        { AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM, RSMI_FW_BLOCK_RLC_SRLG},
        { AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM, RSMI_FW_BLOCK_RLC_SRLS},
        { AMDSMI_FW_ID_SDMA0, RSMI_FW_BLOCK_SDMA},
        { AMDSMI_FW_ID_SDMA1, RSMI_FW_BLOCK_SDMA2},
        { AMDSMI_FW_ID_PM, RSMI_FW_BLOCK_SMC},
        { AMDSMI_FW_ID_PSP_SOSDRV, RSMI_FW_BLOCK_SOS},
        { AMDSMI_FW_ID_TA_RAS, RSMI_FW_BLOCK_TA_RAS},
        { AMDSMI_FW_ID_TA_XGMI, RSMI_FW_BLOCK_TA_XGMI},
        { AMDSMI_FW_ID_UVD, RSMI_FW_BLOCK_UVD},
        {AMDSMI_FW_ID_VCE, RSMI_FW_BLOCK_VCE},
        { AMDSMI_FW_ID_VCN, RSMI_FW_BLOCK_VCN}
    };

    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;
    memset(info, 0, sizeof(amdsmi_fw_info_t));

    // collect all rsmi supported fw block
    for (auto ite = fw_in_rsmi.begin(); ite != fw_in_rsmi.end(); ite ++) {
        auto status = rsmi_wrapper(rsmi_dev_firmware_version_get, processor_handle,
                (*ite).second,
                &(info->fw_info_list[info->num_fw_info].fw_version));
        if (status == AMDSMI_STATUS_SUCCESS) {
            info->fw_info_list[info->num_fw_info].fw_id = (*ite).first;
            info->num_fw_info++;
        }
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_asic_info(amdsmi_processor_handle processor_handle, amdsmi_asic_info_t *info) {

    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    struct drm_amdgpu_info_device dev_info = {};
    uint16_t vendor_id = 0;
    uint16_t subvendor_id = 0;

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    amdsmi_status_t status;
    if (gpu_device->check_if_drm_is_supported()){
        status = gpu_device->amdgpu_query_info(AMDGPU_INFO_DEV_INFO, sizeof(struct drm_amdgpu_info_device), &dev_info);
        if (status != AMDSMI_STATUS_SUCCESS) return status;

        SMIGPUDEVICE_MUTEX(gpu_device->get_mutex())

        std::string path = "/sys/class/drm/" + gpu_device->get_gpu_path() + "/device/unique_id";
        FILE *fp = fopen(path.c_str(), "r");
        if (fp) {
            fscanf(fp, "%s", info->asic_serial);
            fclose(fp);
        }

        status = smi_amdgpu_get_market_name_from_dev_id(dev_info.device_id, info->market_name);
        if (status != AMDSMI_STATUS_SUCCESS) {
            rsmi_wrapper(rsmi_dev_brand_get, processor_handle,
                info->market_name, AMDSMI_256_LENGTH);
        }

        info->device_id = dev_info.device_id;
        info->rev_id = dev_info.pci_rev;
        info->vendor_id = gpu_device->get_vendor_id();
    }
    else {
        uint64_t dv_uid = 0;
        status = rsmi_wrapper(rsmi_dev_unique_id_get, processor_handle, &dv_uid);
        if (status == AMDSMI_STATUS_SUCCESS) snprintf(info->asic_serial, sizeof(info->asic_serial), "%lu", dv_uid);

        status = rsmi_wrapper(rsmi_dev_brand_get, processor_handle,
                info->market_name, AMDSMI_256_LENGTH);

        status = rsmi_wrapper(rsmi_dev_vendor_id_get, processor_handle,
                                    &vendor_id);
        if (status == AMDSMI_STATUS_SUCCESS) info->vendor_id = vendor_id;
    }
    // For other sysfs related information, get from rocm-smi
    status = rsmi_wrapper(rsmi_dev_subsystem_vendor_id_get, processor_handle,
                &subvendor_id);
    if (status == AMDSMI_STATUS_SUCCESS) info->subvendor_id = subvendor_id;

    status =  rsmi_wrapper(rsmi_dev_pcie_vendor_name_get, processor_handle,
                    info->vendor_name, AMDSMI_MAX_STRING_LENGTH);

    // If vendor name is empty and the vendor id is 0x1002, set vendor name to AMD vendor string
    if ((info->vendor_name != NULL && info->vendor_name[0] == '\0') && info->vendor_id == 0x1002) {
        memset(info->vendor_name, 0, 38);
        strncpy(info->vendor_name, "Advanced Micro Devices Inc. [AMD/ATI]", 37);
    }

    // default to 0xffff as not supported
    info->oam_id = std::numeric_limits<uint16_t>::max();
    uint16_t tmp_oam_id = 0;
    status =  rsmi_wrapper(rsmi_dev_xgmi_physical_id_get, processor_handle,
                    &(tmp_oam_id));
    info->oam_id = tmp_oam_id;

    // default to 0xffffffff as not supported
    info->num_of_compute_units = std::numeric_limits<uint32_t>::max();
    auto tmp_num_of_compute_units = uint32_t(0);
    status = rsmi_wrapper(amd::smi::rsmi_dev_number_of_computes_get, processor_handle,
                          &(tmp_num_of_compute_units));
    if (status == amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
        info->num_of_compute_units = tmp_num_of_compute_units;
    }

    // default to 0xffffffffffffffff as not supported
    info->target_graphics_version = std::numeric_limits<uint64_t>::max();
    auto tmp_target_gfx_version = uint64_t(0);
    status = rsmi_wrapper(rsmi_dev_target_graphics_version_get, processor_handle,
                          &(tmp_target_gfx_version));
    if (status == amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
        info->target_graphics_version = tmp_target_gfx_version;
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
    status = rsmi_wrapper(rsmi_dev_guid_get, processor_handle, &(tmp_kfd_id));
    // Do not return early if this value fails
    // continue to try getting all info
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->kfd_id = tmp_kfd_id;
    }

    // default to 0xffffffff as not supported
    info->node_id = std::numeric_limits<uint32_t>::max();
    auto tmp_node_id = uint32_t(0);
    status = rsmi_wrapper(rsmi_dev_node_id_get, processor_handle, &(tmp_node_id));
    // Do not return early if this value fails
    // continue to try getting all info
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->node_id = tmp_node_id;
    }

    // default to 0xffffffff as not supported
    info->current_partition_id = std::numeric_limits<uint32_t>::max();
    auto tmp_current_partition_id = uint32_t(0);
    status = rsmi_wrapper(rsmi_dev_partition_id_get, processor_handle, &(tmp_current_partition_id));
    // Do not return early if this value fails
    // continue to try getting all info
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->current_partition_id = tmp_current_partition_id;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_subsystem_id(amdsmi_processor_handle processor_handle,
                                uint16_t *id) {
    return rsmi_wrapper(rsmi_dev_subsystem_id_get, processor_handle, id);
}

amdsmi_status_t amdsmi_get_gpu_subsystem_name(
                                amdsmi_processor_handle processor_handle,
                                char *name, size_t len) {
    return rsmi_wrapper(rsmi_dev_subsystem_name_get, processor_handle, name, len);
}

amdsmi_status_t amdsmi_get_gpu_vendor_name(
            amdsmi_processor_handle processor_handle, char *name, size_t len) {
    return rsmi_wrapper(rsmi_dev_vendor_name_get, processor_handle, name, len);
}

amdsmi_status_t amdsmi_get_gpu_vram_vendor(amdsmi_processor_handle processor_handle,
                                     char *brand, uint32_t len) {
    return rsmi_wrapper(rsmi_dev_vram_vendor_get, processor_handle, brand, len);
}

amdsmi_status_t amdsmi_get_gpu_vram_info(
    amdsmi_processor_handle processor_handle, amdsmi_vram_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle,
                            &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    // init the info structure with default value
    info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN;
    info->vram_size = 0;
    info->vram_vendor = AMDSMI_VRAM_VENDOR__PLACEHOLDER0;
    info->vram_bit_width = std::numeric_limits<decltype(info->vram_bit_width)>::max();

    // Only can read vram type from libdrm
    if (gpu_device->check_if_drm_is_supported()) {
        struct drm_amdgpu_info_device dev_info = {};
        r = gpu_device->amdgpu_query_info(
            AMDGPU_INFO_DEV_INFO,
            sizeof(struct drm_amdgpu_info_device), &dev_info);
        if (r == AMDSMI_STATUS_SUCCESS) {
            info->vram_type = amd::smi::vram_type_value(dev_info.vram_type);
            info->vram_bit_width = dev_info.vram_bit_width;
        }
    }

    // if vram type is greater than the max enum set it to unknown
    if (info->vram_type > AMDSMI_VRAM_TYPE__MAX)
        info->vram_type = AMDSMI_VRAM_TYPE_UNKNOWN;

    // map the vendor name to enum
    char brand[256];
    r = rsmi_wrapper(rsmi_dev_vram_vendor_get, processor_handle, brand, 255);
    if (r == AMDSMI_STATUS_SUCCESS) {
        if (strcasecmp(brand, "SAMSUNG") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__SAMSUNG;
        if (strcasecmp(brand, "INFINEON") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__INFINEON;
        if (strcasecmp(brand, "ELPIDA") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__ELPIDA;
        if (strcasecmp(brand, "ETRON") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__ETRON;
        if (strcasecmp(brand, "NANYA") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__NANYA;
        if (strcasecmp(brand, "HYNIX") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__HYNIX;
        if (strcasecmp(brand, "MOSEL") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__MOSEL;
        if (strcasecmp(brand, "WINBOND") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__WINBOND;
        if (strcasecmp(brand, "ESMT") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__ESMT;
        if (strcasecmp(brand, "MICRON") == 0)
            info->vram_vendor = AMDSMI_VRAM_VENDOR__MICRON;
    }
    uint64_t total = 0;
    r = rsmi_wrapper(rsmi_dev_memory_total_get, processor_handle,
                    RSMI_MEM_TYPE_VRAM, &total);
    if (r == AMDSMI_STATUS_SUCCESS) {
        info->vram_size = total / (1024 * 1024);
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_init_gpu_event_notification(amdsmi_processor_handle processor_handle) {
    return rsmi_wrapper(rsmi_event_notification_init, processor_handle);
}

amdsmi_status_t
amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle processor_handle,
          uint64_t mask) {
    return rsmi_wrapper(rsmi_event_notification_mask_set, processor_handle, mask);
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
        strncpy(data[i].message, rsmi_data.message,
                MAX_EVENT_NOTIFICATION_MSG_SIZE);
        amdsmi_status_t r = amd::smi::AMDSmiSystem::getInstance()
            .gpu_index_to_handle(rsmi_data.dv_ind, &(data[i].processor_handle));
        if (r != AMDSMI_STATUS_SUCCESS) return r;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_stop_gpu_event_notification(
                amdsmi_processor_handle processor_handle) {
    return rsmi_wrapper(rsmi_event_notification_stop, processor_handle);
}

amdsmi_status_t amdsmi_gpu_counter_group_supported(
        amdsmi_processor_handle processor_handle, amdsmi_event_group_t group) {
    return rsmi_wrapper(rsmi_dev_counter_group_supported, processor_handle,
                    static_cast<rsmi_event_group_t>(group));
}

amdsmi_status_t amdsmi_gpu_create_counter(amdsmi_processor_handle processor_handle,
        amdsmi_event_type_t type, amdsmi_event_handle_t *evnt_handle) {
    return rsmi_wrapper(rsmi_dev_counter_create, processor_handle,
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
    return rsmi_wrapper(rsmi_counter_available_counters_get, processor_handle,
                    static_cast<rsmi_event_group_t>(grp),
                    available);
}

amdsmi_status_t
amdsmi_topo_get_numa_node_number(amdsmi_processor_handle processor_handle, uint32_t *numa_node) {
    return rsmi_wrapper(rsmi_topo_get_numa_node_number, processor_handle, numa_node);
}

amdsmi_status_t
amdsmi_topo_get_link_weight(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                          uint64_t *weight) {
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiGPUDevice* src_device = nullptr;
    amd::smi::AMDSmiGPUDevice* dst_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle_src, &src_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    r = get_gpu_device_from_handle(processor_handle_dst, &dst_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    auto rstatus = rsmi_topo_get_link_weight(src_device->get_gpu_id(), dst_device->get_gpu_id(),
                weight);
    return amd::smi::rsmi_to_amdsmi_status(rstatus);
}

amdsmi_status_t
 amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                          uint64_t *min_bandwidth, uint64_t *max_bandwidth) {
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiGPUDevice* src_device = nullptr;
    amd::smi::AMDSmiGPUDevice* dst_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle_src, &src_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    r = get_gpu_device_from_handle(processor_handle_dst, &dst_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    auto rstatus = rsmi_minmax_bandwidth_get(src_device->get_gpu_id(), dst_device->get_gpu_id(),
                min_bandwidth, max_bandwidth);
    return amd::smi::rsmi_to_amdsmi_status(rstatus);
}


amdsmi_status_t amdsmi_get_link_metrics(amdsmi_processor_handle processor_handle,
          amdsmi_link_metrics_t *link_metrics) {
    AMDSMI_CHECK_INIT();
    if (link_metrics == nullptr)  return AMDSMI_STATUS_INVAL;

    amdsmi_gpu_metrics_t metric_info = {};
    amdsmi_status_t status =  amdsmi_get_gpu_metrics_info(
            processor_handle, &metric_info);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;
    link_metrics->num_links = AMDSMI_MAX_NUM_XGMI_LINKS;
    for (unsigned int i = 0; i < link_metrics->num_links; i++) {
        link_metrics->links[i].read = metric_info.xgmi_read_data_acc[i];
        link_metrics->links[i].write = metric_info.xgmi_write_data_acc[i];
        link_metrics->links[i].bit_rate = metric_info.xgmi_link_speed;
        link_metrics->links[i].max_bandwidth = metric_info.xgmi_link_width;
        link_metrics->links[i].link_type = AMDSMI_LINK_TYPE_XGMI;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_topo_get_link_type(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                        uint64_t *hops, amdsmi_io_link_type_t *type) {
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiGPUDevice* src_device = nullptr;
    amd::smi::AMDSmiGPUDevice* dst_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle_src, &src_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    r = get_gpu_device_from_handle(processor_handle_dst, &dst_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    auto rstatus = rsmi_topo_get_link_type(src_device->get_gpu_id(), dst_device->get_gpu_id(),
                hops, reinterpret_cast<RSMI_IO_LINK_TYPE*>(type));
    return amd::smi::rsmi_to_amdsmi_status(rstatus);
}

amdsmi_status_t
amdsmi_is_P2P_accessible(amdsmi_processor_handle processor_handle_src,
                amdsmi_processor_handle processor_handle_dst,
                       bool *accessible) {
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiGPUDevice* src_device = nullptr;
    amd::smi::AMDSmiGPUDevice* dst_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle_src, &src_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    r = get_gpu_device_from_handle(processor_handle_dst, &dst_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    auto rstatus = rsmi_is_P2P_accessible(src_device->get_gpu_id(), dst_device->get_gpu_id(),
                accessible);
    return amd::smi::rsmi_to_amdsmi_status(rstatus);
}

amdsmi_status_t
amdsmi_topo_get_p2p_status(amdsmi_processor_handle processor_handle_src,
                           amdsmi_processor_handle processor_handle_dst,
                           amdsmi_io_link_type_t *type, amdsmi_p2p_capability_t *cap) {
    AMDSMI_CHECK_INIT();

    amd::smi::AMDSmiGPUDevice* src_device = nullptr;
    amd::smi::AMDSmiGPUDevice* dst_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle_src, &src_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    r = get_gpu_device_from_handle(processor_handle_dst, &dst_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    auto rstatus = rsmi_topo_get_p2p_status(src_device->get_gpu_id(), dst_device->get_gpu_id(),
                reinterpret_cast<RSMI_IO_LINK_TYPE*>(type),
                reinterpret_cast<rsmi_p2p_capability_t*>(cap));
    return amd::smi::rsmi_to_amdsmi_status(rstatus);
}

// Compute Partition functions
amdsmi_status_t
amdsmi_get_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                  char *compute_partition, uint32_t len) {
    AMDSMI_CHECK_INIT();
    return rsmi_wrapper(rsmi_dev_compute_partition_get, processor_handle,
                          compute_partition, len);
}

amdsmi_status_t
amdsmi_set_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_compute_partition_type_t compute_partition) {
    AMDSMI_CHECK_INIT();
    return rsmi_wrapper(rsmi_dev_compute_partition_set, processor_handle,
                          static_cast<rsmi_compute_partition_type_t>(compute_partition));
}

amdsmi_status_t
amdsmi_reset_gpu_compute_partition(amdsmi_processor_handle processor_handle) {
    AMDSMI_CHECK_INIT();
    return rsmi_wrapper(rsmi_dev_compute_partition_reset, processor_handle);
}

// Memory Partition functions
amdsmi_status_t
amdsmi_get_gpu_memory_partition(amdsmi_processor_handle processor_handle,
                                  char *memory_partition, uint32_t len) {
    AMDSMI_CHECK_INIT();
    return rsmi_wrapper(rsmi_dev_memory_partition_get, processor_handle,
                          memory_partition, len);
}

amdsmi_status_t
amdsmi_set_gpu_memory_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_memory_partition_type_t memory_partition) {
    AMDSMI_CHECK_INIT();
    return rsmi_wrapper(rsmi_dev_memory_partition_set, processor_handle,
                          static_cast<rsmi_memory_partition_type_t>(memory_partition));
}

amdsmi_status_t
amdsmi_reset_gpu_memory_partition(amdsmi_processor_handle processor_handle) {
    AMDSMI_CHECK_INIT();
    return rsmi_wrapper(rsmi_dev_memory_partition_reset, processor_handle);
}

amdsmi_status_t
amdsmi_get_gpu_accelerator_partition_profile(amdsmi_processor_handle processor_handle,
                                             amdsmi_accelerator_partition_profile_t *profile,
                                             uint32_t *partition_id) {
    AMDSMI_CHECK_INIT();
    // TODO: also fill out profile later
    // default to 0xffffffff if not supported
    *partition_id = std::numeric_limits<uint32_t>::max();
    auto tmp_partition_id = uint32_t(0);

    amdsmi_status_t status = rsmi_wrapper(rsmi_dev_partition_id_get, processor_handle, &tmp_partition_id);
    if (status == amdsmi_status_t::AMDSMI_STATUS_SUCCESS){
        *partition_id = tmp_partition_id;
    }

    return status;
}

// TODO(bliu) : other xgmi related information
amdsmi_status_t
amdsmi_get_xgmi_info(amdsmi_processor_handle processor_handle, amdsmi_xgmi_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;
    return rsmi_wrapper(rsmi_dev_xgmi_hive_id_get, processor_handle,
                    &(info->xgmi_hive_id));
}

amdsmi_status_t
amdsmi_gpu_xgmi_error_status(amdsmi_processor_handle processor_handle, amdsmi_xgmi_status_t *status) {
    return rsmi_wrapper(rsmi_dev_xgmi_error_status, processor_handle,
                    reinterpret_cast<rsmi_xgmi_status_t*>(status));
}

amdsmi_status_t
amdsmi_reset_gpu_xgmi_error(amdsmi_processor_handle processor_handle) {
    return rsmi_wrapper(rsmi_dev_xgmi_error_reset, processor_handle);
}

amdsmi_status_t
amdsmi_get_gpu_compute_process_info(amdsmi_process_info_t *procs, uint32_t *num_items) {
    AMDSMI_CHECK_INIT();

    if (num_items == nullptr)
        return AMDSMI_STATUS_INVAL;
    auto r = rsmi_compute_process_info_get(
        reinterpret_cast<rsmi_process_info_t*>(procs),
        num_items);
    return amd::smi::rsmi_to_amdsmi_status(r);
}

amdsmi_status_t amdsmi_get_gpu_compute_process_info_by_pid(uint32_t pid,
        amdsmi_process_info_t *proc) {
    AMDSMI_CHECK_INIT();

    if (proc == nullptr)
        return AMDSMI_STATUS_INVAL;
    auto r = rsmi_compute_process_info_by_pid_get(pid,
        reinterpret_cast<rsmi_process_info_t*>(proc));
    return amd::smi::rsmi_to_amdsmi_status(r);
}

amdsmi_status_t
amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t *dv_indices,
                                                       uint32_t *num_devices) {
    AMDSMI_CHECK_INIT();

    if (dv_indices == nullptr || num_devices == nullptr)
        return AMDSMI_STATUS_INVAL;
    auto r = rsmi_compute_process_gpus_get(pid, dv_indices, num_devices);
    return amd::smi::rsmi_to_amdsmi_status(r);
}

amdsmi_status_t  amdsmi_get_gpu_ecc_count(amdsmi_processor_handle processor_handle,
                        amdsmi_gpu_block_t block, amdsmi_error_count_t *ec) {
    AMDSMI_CHECK_INIT();
    // nullptr api supported

    return rsmi_wrapper(rsmi_dev_ecc_count_get, processor_handle,
                    static_cast<rsmi_gpu_block_t>(block),
                    reinterpret_cast<rsmi_error_count_t*>(ec));
}
amdsmi_status_t  amdsmi_get_gpu_ecc_enabled(amdsmi_processor_handle processor_handle,
                                                    uint64_t *enabled_blocks) {
    AMDSMI_CHECK_INIT();
    // nullptr api supported

    return rsmi_wrapper(rsmi_dev_ecc_enabled_get, processor_handle,
                    enabled_blocks);
}
amdsmi_status_t  amdsmi_get_gpu_ecc_status(amdsmi_processor_handle processor_handle,
                                amdsmi_gpu_block_t block,
                                amdsmi_ras_err_state_t *state) {
    AMDSMI_CHECK_INIT();
    // nullptr api supported

    return rsmi_wrapper(rsmi_dev_ecc_status_get, processor_handle,
                    static_cast<rsmi_gpu_block_t>(block),
                    reinterpret_cast<rsmi_ras_err_state_t*>(state));
}

amdsmi_status_t
amdsmi_get_gpu_metrics_header_info(amdsmi_processor_handle processor_handle,
                amd_metrics_table_header_t *header_value)
{
    AMDSMI_CHECK_INIT();
    // nullptr api supported

    return rsmi_wrapper(rsmi_dev_metrics_header_info_get, processor_handle,
                    reinterpret_cast<metrics_table_header_t*>(header_value));
}

amdsmi_status_t  amdsmi_get_gpu_metrics_info(
        amdsmi_processor_handle processor_handle,
        amdsmi_gpu_metrics_t *pgpu_metrics) {
    AMDSMI_CHECK_INIT();
    // nullptr api supported
    if (pgpu_metrics != nullptr) {
        *pgpu_metrics = {};
    }
    return rsmi_wrapper(rsmi_dev_gpu_metrics_info_get, processor_handle,
                       reinterpret_cast<rsmi_gpu_metrics_t*>(pgpu_metrics));
}


amdsmi_status_t amdsmi_get_gpu_pm_metrics_info(
                      amdsmi_processor_handle processor_handle,
                      amdsmi_name_value_t** pm_metrics,
                      uint32_t *num_of_metrics) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_pm_metrics_info_get, processor_handle,
                    reinterpret_cast<rsmi_name_value_t**>(pm_metrics),
                    num_of_metrics);
}

amdsmi_status_t amdsmi_get_gpu_reg_table_info(
                      amdsmi_processor_handle processor_handle,
                      amdsmi_reg_type_t reg_type,
                      amdsmi_name_value_t** reg_metrics,
                      uint32_t *num_of_metrics) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_reg_table_info_get, processor_handle,
                    static_cast<rsmi_reg_type_t>(reg_type),
                    reinterpret_cast<rsmi_name_value_t**>(reg_metrics),
                    num_of_metrics);
}

void amdsmi_free_name_value_pairs(void *p) {
    free(p);
}

amdsmi_status_t
amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle,
                          uint32_t sensor_ind,
                          amdsmi_power_cap_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr)
        return AMDSMI_STATUS_INVAL;

    bool set_ret_success = false;
    amd::smi::AMDSmiGPUDevice* gpudevice = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpudevice);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    amdsmi_status_t status;

    status = get_gpu_device_from_handle(processor_handle, &gpudevice);
    if (status != AMDSMI_STATUS_SUCCESS)
    {
        return status;
    }
    // Ignore errors to get as much as possible info.
    memset(info, 0, sizeof(amdsmi_power_cap_info_t));

    if (gpudevice->check_if_drm_is_supported()) {
        // Get power_cap and dpm
        int power_cap = 0;
        int dpm = 0;
        status = smi_amdgpu_get_power_cap(gpudevice, &power_cap);
        if ((status == AMDSMI_STATUS_SUCCESS) && !set_ret_success)
            set_ret_success = true;

        info->power_cap = power_cap;
        status = smi_amdgpu_get_ranges(gpudevice, AMDSMI_CLK_TYPE_GFX,
                NULL, NULL, &dpm, NULL);
        if ((status == AMDSMI_STATUS_SUCCESS) && !set_ret_success)
            set_ret_success = true;
        info->dpm_cap = dpm;
    }
    else {
        status = rsmi_wrapper(rsmi_dev_power_cap_get, processor_handle,
                    sensor_ind, &(info->power_cap));
        if ((status == AMDSMI_STATUS_SUCCESS) && !set_ret_success)
            set_ret_success = true;
    }

    // Get other information from rocm-smi
    status = rsmi_wrapper(rsmi_dev_power_cap_default_get, processor_handle,
                        &(info->default_power_cap));

    if ((status == AMDSMI_STATUS_SUCCESS) && !set_ret_success)
        set_ret_success = true;


    status = rsmi_wrapper(rsmi_dev_power_cap_range_get, processor_handle, sensor_ind,
                        &(info->max_power_cap), &(info->min_power_cap));


    if ((status == AMDSMI_STATUS_SUCCESS) && !set_ret_success)
        set_ret_success = true;

    return set_ret_success ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_NOT_SUPPORTED;
}

amdsmi_status_t
 amdsmi_set_power_cap(amdsmi_processor_handle processor_handle,
            uint32_t sensor_ind, uint64_t cap) {
    return rsmi_wrapper(rsmi_dev_power_cap_set, processor_handle,
            sensor_ind, cap);
}

amdsmi_status_t
 amdsmi_get_gpu_power_profile_presets(amdsmi_processor_handle processor_handle,
                        uint32_t sensor_ind,
                        amdsmi_power_profile_status_t *status) {
    AMDSMI_CHECK_INIT();
    // nullptr api supported

    return rsmi_wrapper(rsmi_dev_power_profile_presets_get, processor_handle,
                    sensor_ind,
                    reinterpret_cast<rsmi_power_profile_status_t*>(status));
}

amdsmi_status_t amdsmi_set_gpu_perf_determinism_mode(
            amdsmi_processor_handle processor_handle, uint64_t clkvalue) {
    return rsmi_wrapper(rsmi_perf_determinism_mode_set, processor_handle,
                clkvalue);
}

amdsmi_status_t
 amdsmi_set_gpu_power_profile(amdsmi_processor_handle processor_handle,
        uint32_t reserved, amdsmi_power_profile_preset_masks_t profile) {
    return rsmi_wrapper(rsmi_dev_power_profile_set, processor_handle,
                reserved,
                static_cast<rsmi_power_profile_preset_masks_t>(profile));
}

amdsmi_status_t amdsmi_get_gpu_perf_level(amdsmi_processor_handle processor_handle,
                                        amdsmi_dev_perf_level_t *perf) {
    AMDSMI_CHECK_INIT();
    // nullptr api supported

    return rsmi_wrapper(rsmi_dev_perf_level_get, processor_handle,
                    reinterpret_cast<rsmi_dev_perf_level_t*>(perf));
}

amdsmi_status_t
 amdsmi_set_gpu_perf_level(amdsmi_processor_handle processor_handle,
                amdsmi_dev_perf_level_t perf_lvl) {
    return rsmi_wrapper(rsmi_dev_perf_level_set_v1, processor_handle,
                    static_cast<rsmi_dev_perf_level_t>(perf_lvl));
}

amdsmi_status_t  amdsmi_set_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle,
                uint64_t bw_bitmask) {
    return rsmi_wrapper(rsmi_dev_pci_bandwidth_set, processor_handle,
                    bw_bitmask);
}

amdsmi_status_t amdsmi_get_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle,
            amdsmi_pcie_bandwidth_t *bandwidth) {
    return rsmi_wrapper(rsmi_dev_pci_bandwidth_get, processor_handle,
                    reinterpret_cast<rsmi_pcie_bandwidth_t*>(bandwidth));
}

// TODO(bliu): other frequencies in amdsmi_clk_type_t
amdsmi_status_t  amdsmi_get_clk_freq(amdsmi_processor_handle processor_handle,
                        amdsmi_clk_type_t clk_type, amdsmi_frequencies_t *f) {
    AMDSMI_CHECK_INIT();
    // nullptr api supported

    // Get from gpu_metrics
    if (clk_type == AMDSMI_CLK_TYPE_VCLK0 ||
        clk_type == AMDSMI_CLK_TYPE_VCLK1 ||
        clk_type == AMDSMI_CLK_TYPE_DCLK0 ||
        clk_type == AMDSMI_CLK_TYPE_DCLK1 ) {
        // when f == nullptr -> check if metrics are supported
        amdsmi_gpu_metrics_t metric_info;
        amdsmi_gpu_metrics_t * metric_info_p = nullptr;

        if (f != nullptr) {
            metric_info_p = &metric_info;
        }

        // when metric_info_p == nullptr - this will not return AMDSMI_STATUS_SUCCESS
        auto r_status =  amdsmi_get_gpu_metrics_info(
                processor_handle, metric_info_p);
        if (r_status != AMDSMI_STATUS_SUCCESS)
            return r_status;

        f->num_supported = 1;
        if (clk_type == AMDSMI_CLK_TYPE_VCLK0) {
            f->current = metric_info_p->current_vclk0;
            f->frequency[0] = metric_info_p->average_vclk0_frequency;
        }
        if (clk_type == AMDSMI_CLK_TYPE_VCLK1) {
            f->current = metric_info_p->current_vclk1;
            f->frequency[0] = metric_info_p->average_vclk1_frequency;
        }
        if (clk_type == AMDSMI_CLK_TYPE_DCLK0) {
            f->current = metric_info_p->current_dclk0;
            f->frequency[0] = metric_info_p->average_dclk0_frequency;
        }
        if (clk_type == AMDSMI_CLK_TYPE_DCLK1) {
            f->current = metric_info_p->current_dclk1;
            f->frequency[0] = metric_info_p->average_dclk1_frequency;
        }

        return r_status;
    }

    return rsmi_wrapper(rsmi_dev_gpu_clk_freq_get, processor_handle,
                    static_cast<rsmi_clk_type_t>(clk_type),
                    reinterpret_cast<rsmi_frequencies_t*>(f));
}

amdsmi_status_t  amdsmi_set_clk_freq(amdsmi_processor_handle processor_handle,
                         amdsmi_clk_type_t clk_type, uint64_t freq_bitmask) {
    AMDSMI_CHECK_INIT();

    // Not support the clock type write into gpu_metrics
    if (clk_type == AMDSMI_CLK_TYPE_VCLK0 ||
        clk_type == AMDSMI_CLK_TYPE_VCLK1 ||
        clk_type == AMDSMI_CLK_TYPE_DCLK0 ||
        clk_type == AMDSMI_CLK_TYPE_DCLK1 ) {
            return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    return rsmi_wrapper(rsmi_dev_gpu_clk_freq_set, processor_handle,
                    static_cast<rsmi_clk_type_t>(clk_type), freq_bitmask);
}

amdsmi_status_t amdsmi_set_soc_pstate(amdsmi_processor_handle processor_handle,
                         uint32_t policy) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_soc_pstate_set, processor_handle,
                    policy);
}

amdsmi_status_t amdsmi_get_soc_pstate(amdsmi_processor_handle processor_handle,
                         amdsmi_dpm_policy_t* policy) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_soc_pstate_get, processor_handle,
                    reinterpret_cast<rsmi_dpm_policy_t*>(policy));
}

amdsmi_status_t amdsmi_set_xgmi_plpd(amdsmi_processor_handle processor_handle,
                         uint32_t policy) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_xgmi_plpd_set, processor_handle,
                    policy);
}

amdsmi_status_t amdsmi_get_xgmi_plpd(amdsmi_processor_handle processor_handle,
                         amdsmi_dpm_policy_t* policy) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_xgmi_plpd_get, processor_handle,
                    reinterpret_cast<rsmi_dpm_policy_t*>(policy));
}

amdsmi_status_t amdsmi_get_gpu_process_isolation(amdsmi_processor_handle processor_handle,
                             uint32_t* pisolate) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_process_isolation_get, processor_handle,
                    pisolate);
}

amdsmi_status_t amdsmi_set_gpu_process_isolation(amdsmi_processor_handle processor_handle,
                             uint32_t pisolate) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_process_isolation_set, processor_handle,
                   pisolate);
}

amdsmi_status_t amdsmi_clean_gpu_local_data(amdsmi_processor_handle processor_handle) {
    AMDSMI_CHECK_INIT();

    return rsmi_wrapper(rsmi_dev_gpu_run_cleaner_shader, processor_handle);
}

amdsmi_status_t
amdsmi_get_gpu_memory_reserved_pages(amdsmi_processor_handle processor_handle,
                                    uint32_t *num_pages,
                                    amdsmi_retired_page_record_t *records) {
    return rsmi_wrapper(rsmi_dev_memory_reserved_pages_get, processor_handle,
                    num_pages,
                    reinterpret_cast<rsmi_retired_page_record_t*>(records));
}
amdsmi_status_t amdsmi_get_gpu_memory_total(amdsmi_processor_handle processor_handle,
                amdsmi_memory_type_t mem_type, uint64_t *total) {
    return rsmi_wrapper(rsmi_dev_memory_total_get, processor_handle,
                    static_cast<rsmi_memory_type_t>(mem_type), total);
}
amdsmi_status_t amdsmi_get_gpu_memory_usage(amdsmi_processor_handle processor_handle,
            amdsmi_memory_type_t mem_type, uint64_t *used) {
    return rsmi_wrapper(rsmi_dev_memory_usage_get, processor_handle,
                    static_cast<rsmi_memory_type_t>(mem_type), used);
}

amdsmi_status_t amdsmi_get_gpu_overdrive_level(
            amdsmi_processor_handle processor_handle,
            uint32_t *od) {
    return rsmi_wrapper(rsmi_dev_overdrive_level_get, processor_handle, od);
}

amdsmi_status_t amdsmi_get_gpu_mem_overdrive_level(
            amdsmi_processor_handle processor_handle,
            uint32_t *od) {
    return rsmi_wrapper(rsmi_dev_mem_overdrive_level_get, processor_handle, od);
}

amdsmi_status_t  amdsmi_set_gpu_overdrive_level(
            amdsmi_processor_handle processor_handle, uint32_t od) {
    return rsmi_wrapper(rsmi_dev_overdrive_level_set_v1, processor_handle, od);
}
amdsmi_status_t  amdsmi_get_gpu_pci_replay_counter(
            amdsmi_processor_handle processor_handle, uint64_t *counter) {
    return rsmi_wrapper(rsmi_dev_pci_replay_counter_get,
                processor_handle, counter);
}
amdsmi_status_t amdsmi_get_gpu_pci_throughput(
        amdsmi_processor_handle processor_handle,
        uint64_t *sent, uint64_t *received, uint64_t *max_pkt_sz) {
    return rsmi_wrapper(rsmi_dev_pci_throughput_get, processor_handle,
            sent, received, max_pkt_sz);
}

amdsmi_status_t  amdsmi_get_gpu_od_volt_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_od_volt_freq_data_t *odv) {
    return rsmi_wrapper(rsmi_dev_od_volt_info_get, processor_handle,
                    reinterpret_cast<rsmi_od_volt_freq_data_t*>(odv));
}

amdsmi_status_t  amdsmi_get_gpu_od_volt_curve_regions(
                    amdsmi_processor_handle processor_handle,
                    uint32_t *num_regions, amdsmi_freq_volt_region_t *buffer) {
    return rsmi_wrapper(rsmi_dev_od_volt_curve_regions_get, processor_handle,
        num_regions, reinterpret_cast<rsmi_freq_volt_region_t* >(buffer));
}

amdsmi_status_t  amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle,
                            amdsmi_voltage_type_t sensor_type,
                            amdsmi_voltage_metric_t metric, int64_t *voltage) {
    return rsmi_wrapper(rsmi_dev_volt_metric_get, processor_handle,
                static_cast<rsmi_voltage_type_t>(sensor_type),
                static_cast<rsmi_voltage_metric_t>(metric), voltage);
}

amdsmi_status_t  amdsmi_set_gpu_od_clk_info(amdsmi_processor_handle processor_handle,
                                        amdsmi_freq_ind_t level,
                                       uint64_t clkvalue,
                                       amdsmi_clk_type_t clkType) {
    return rsmi_wrapper(rsmi_dev_od_clk_info_set, processor_handle,
                static_cast<rsmi_freq_ind_t>(level), clkvalue,
                static_cast<rsmi_clk_type_t>(clkType));
}

amdsmi_status_t  amdsmi_set_gpu_od_volt_info(amdsmi_processor_handle processor_handle,
                    uint32_t vpoint, uint64_t clkvalue, uint64_t voltvalue) {
    return rsmi_wrapper(rsmi_dev_od_volt_info_set, processor_handle,
                vpoint, clkvalue, voltvalue);
}

amdsmi_status_t amdsmi_set_gpu_clk_range(amdsmi_processor_handle processor_handle,
                                    uint64_t minclkvalue,
                                    uint64_t maxclkvalue,
                                    amdsmi_clk_type_t clkType) {
    return rsmi_wrapper(rsmi_dev_clk_range_set, processor_handle,
                minclkvalue, maxclkvalue,
                static_cast<rsmi_clk_type_t>(clkType));
}

amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle processor_handle,
                                         amdsmi_clk_type_t clk_type,
                                          amdsmi_clk_limit_type_t limit_type,
                                          uint64_t clk_value) {
    return rsmi_wrapper(rsmi_dev_clk_extremum_set, processor_handle,
                static_cast<rsmi_freq_ind_t>(limit_type),
                clk_value,
                static_cast<rsmi_clk_type_t>(clk_type));
}

amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle processor_handle) {
    return rsmi_wrapper(rsmi_dev_gpu_reset, processor_handle);
}

amdsmi_status_t amdsmi_get_utilization_count(amdsmi_processor_handle processor_handle,
                amdsmi_utilization_counter_t utilization_counters[],
                uint32_t count,
                uint64_t *timestamp) {
    return rsmi_wrapper(rsmi_utilization_count_get, processor_handle,
            reinterpret_cast<rsmi_utilization_counter_t*>(utilization_counters),
            count, timestamp);
}

amdsmi_status_t amdsmi_get_energy_count(amdsmi_processor_handle processor_handle,
            uint64_t *energy_accumulator, float *counter_resolution, uint64_t *timestamp) {
    return rsmi_wrapper(rsmi_dev_energy_count_get, processor_handle,
            energy_accumulator, counter_resolution, timestamp);
}

amdsmi_status_t amdsmi_get_gpu_bdf_id(
        amdsmi_processor_handle processor_handle, uint64_t *bdfid) {
    return rsmi_wrapper(rsmi_dev_pci_id_get, processor_handle,
            bdfid);
}

amdsmi_status_t amdsmi_get_gpu_topo_numa_affinity(
    amdsmi_processor_handle processor_handle, int32_t *numa_node) {
    return rsmi_wrapper(rsmi_topo_numa_affinity_get, processor_handle,
            numa_node);
}

amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t *version) {
    AMDSMI_CHECK_INIT();

    if (version == nullptr)
        return AMDSMI_STATUS_INVAL;

    version->year = AMDSMI_LIB_VERSION_YEAR;
    version->major = AMDSMI_LIB_VERSION_MAJOR;
    version->minor = AMDSMI_LIB_VERSION_MINOR;
    version->release = AMDSMI_LIB_VERSION_RELEASE;
    version->build = AMDSMI_LIB_VERSION_STRING;

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_vbios_info(amdsmi_processor_handle processor_handle, amdsmi_vbios_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    struct drm_amdgpu_info_vbios vbios = {};
    amdsmi_status_t status;

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    status = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;


    if (gpu_device->check_if_drm_is_supported()) {
        status = gpu_device->amdgpu_query_vbios(&vbios);
        if (status == AMDSMI_STATUS_SUCCESS) {
            strncpy(info->name, (char *) vbios.name, AMDSMI_MAX_STRING_LENGTH);
            strncpy(info->build_date, (char *) vbios.date, AMDSMI_MAX_DATE_LENGTH);
            strncpy(info->part_number, (char *) vbios.vbios_pn, AMDSMI_MAX_STRING_LENGTH);
            strncpy(info->version, (char *) vbios.vbios_ver_str, AMDSMI_NORMAL_STRING_LENGTH);
        }
    }
    else {
        // get vbios version string from rocm_smi
        char vbios_version[AMDSMI_NORMAL_STRING_LENGTH];
        status = rsmi_wrapper(rsmi_dev_vbios_version_get, processor_handle,
                vbios_version,
                AMDSMI_NORMAL_STRING_LENGTH);

        // ignore the errors so that it can populate as many fields as possible.
        if (status == AMDSMI_STATUS_SUCCESS) {
            strncpy(info->version,
                vbios_version, AMDSMI_NORMAL_STRING_LENGTH);
        }
    }

    return status;
}

amdsmi_status_t
amdsmi_get_gpu_activity(amdsmi_processor_handle processor_handle, amdsmi_engine_usage_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amdsmi_gpu_metrics_t metrics = {};
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    amdsmi_status_t status;
    status =  amdsmi_get_gpu_metrics_info(processor_handle, &metrics);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    info->gfx_activity = metrics.average_gfx_activity;
    info->mm_activity = metrics.average_mm_activity;
    info->umc_activity = metrics.average_umc_activity;

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

    if (clk_type > AMDSMI_CLK_TYPE__MAX) {
        return AMDSMI_STATUS_INVAL;
    }

    amdsmi_gpu_metrics_t metrics = {};
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;
    amdsmi_status_t status;

    status =  amdsmi_get_gpu_metrics_info(processor_handle, &metrics);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    int max_freq;
    int min_freq;
    int sleep_state_freq;
    status = smi_amdgpu_get_ranges(gpu_device, clk_type,
        &max_freq, &min_freq, NULL, &sleep_state_freq);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    info->max_clk = max_freq;
    info->min_clk = min_freq;
    info->clk_deep_sleep = sleep_state_freq;

    switch (clk_type) {
    case AMDSMI_CLK_TYPE_GFX:
        info->clk = metrics.current_gfxclk;
        break;
    case AMDSMI_CLK_TYPE_MEM:
        info->clk = metrics.current_uclk;
        break;
    case AMDSMI_CLK_TYPE_VCLK0:
        info->clk = metrics.current_vclk0;
        break;
    case AMDSMI_CLK_TYPE_VCLK1:
        info->clk = metrics.current_vclk1;
        break;
    case AMDSMI_CLK_TYPE_DCLK0:
        info->clk = metrics.current_dclk0;
      break;
    case AMDSMI_CLK_TYPE_DCLK1:
        info->clk = metrics.current_dclk1;
        break;
    default:
        return AMDSMI_STATUS_INVAL;
    }

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
    if (gpu_device->check_if_drm_is_supported()){
        status = smi_amdgpu_get_bad_page_info(gpu_device, num_pages, info);
        if (status != AMDSMI_STATUS_SUCCESS) {
            return status;
        }
    }
    else {
        // rocm
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_gpu_ras_feature_info(
  amdsmi_processor_handle processor_handle, amdsmi_ras_feature_t *ras_feature) {
    AMDSMI_CHECK_INIT();

    if (ras_feature == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle,
                                &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    rsmi_ras_feature_info_t rsmi_ras_feature;
    r = rsmi_wrapper(rsmi_ras_feature_info_get, processor_handle,
                &rsmi_ras_feature);

    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    ras_feature->ecc_correction_schema_flag
                = rsmi_ras_feature.ecc_correction_schema_flag;
    ras_feature->ras_eeprom_version = rsmi_ras_feature.ras_eeprom_version;

    return AMDSMI_STATUS_SUCCESS;
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

    if (gpu_device->check_if_drm_is_supported()){
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
    }
    else {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t
amdsmi_get_gpu_process_list(amdsmi_processor_handle processor_handle, uint32_t *max_processes, amdsmi_proc_info_t *list) {
    AMDSMI_CHECK_INIT();
    if (!max_processes) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t status_code = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status_code != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
        return status_code;
    }

    auto compute_process_list = gpu_device->amdgpu_get_compute_process_list();
    if ((*max_processes == 0) || compute_process_list.empty()) {
        *max_processes = static_cast<uint32_t>(compute_process_list.size());
        return amdsmi_status_t::AMDSMI_STATUS_SUCCESS;
    }
    if (!list) {
        return amdsmi_status_t::AMDSMI_STATUS_INVAL;
    }

    const auto max_processes_original_size(*max_processes);
    auto idx = uint32_t(0);
    for (auto& process : compute_process_list) {
        if (idx < *max_processes) {
            list[idx++] = static_cast<amdsmi_proc_info_t>(process.second);
        } else {
            break;
        }
    }

    //  Note: If the reserved size for processes is smaller than the number of
    //        actual processes running. The AMDSMI_STATUS_OUT_OF_RESOURCES is
    //        an indication the caller should handle the situation (resize).
    //        The max_processes is always changed to reflect the actual size of
    //        list of processes running, so the caller knows where it is at.
    //        Holding a copy of max_process before it is passed in will be helpful
    //        for the caller.
    *max_processes = static_cast<uint32_t>(compute_process_list.size());
    return (max_processes_original_size >= static_cast<uint32_t>(compute_process_list.size()))
            ? AMDSMI_STATUS_SUCCESS : amdsmi_status_t::AMDSMI_STATUS_OUT_OF_RESOURCES;
}

amdsmi_status_t
amdsmi_get_power_info(amdsmi_processor_handle processor_handle, amdsmi_power_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    amdsmi_status_t status;

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    status = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    info->current_socket_power = 0xFFFF;
    info->average_socket_power = 0xFFFF;
    info->gfx_voltage = 0xFFFF;
    info->soc_voltage = 0xFFFF;
    info->mem_voltage = 0xFFFF;
    info->power_limit = 0xFFFF;

    amdsmi_gpu_metrics_t metrics = {};
    status = amdsmi_get_gpu_metrics_info(processor_handle, &metrics);
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->current_socket_power = metrics.current_socket_power;
        info->average_socket_power = metrics.average_socket_power;
        info->gfx_voltage = metrics.voltage_gfx;
        info->soc_voltage = metrics.voltage_soc;
        info->mem_voltage = metrics.voltage_mem;
    }

    int power_limit = 0;
    status = smi_amdgpu_get_power_cap(gpu_device, &power_limit);
    if (status == AMDSMI_STATUS_SUCCESS) {
        info->power_limit = power_limit;
    }

    return status;
}

amdsmi_status_t amdsmi_get_gpu_driver_info(amdsmi_processor_handle processor_handle,
                amdsmi_driver_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    int length = AMDSMI_MAX_STRING_LENGTH;

    // Get the driver version
    status = smi_amdgpu_get_driver_version(gpu_device,
                &length, info->driver_version);

    // Get the driver date
    std::string driver_date;
    status = gpu_device->amdgpu_query_driver_date(driver_date);
    if (status != AMDSMI_STATUS_SUCCESS)
        return r;
    // Reformat the driver date from 20150101 to 2015/01/01 00:00
    if (driver_date.length() == 8) {
        driver_date = driver_date.substr(0, 4) + "/" + driver_date.substr(4, 2)
                        + "/" + driver_date.substr(6, 2) + " 00:00";
    }
    strncpy(info->driver_date, driver_date.c_str(), AMDSMI_MAX_STRING_LENGTH-1);

    // Get the driver name
    std::string driver_name;
    status = gpu_device->amdgpu_query_driver_name(driver_name);
    if (status != AMDSMI_STATUS_SUCCESS)
        return r;
    strncpy(info->driver_name, driver_name.c_str(), AMDSMI_MAX_STRING_LENGTH-1);

    return status;
}


amdsmi_status_t
amdsmi_get_gpu_device_uuid(amdsmi_processor_handle processor_handle, unsigned int *uuid_length, char *uuid) {
    AMDSMI_CHECK_INIT();

    if (uuid_length == nullptr || uuid == nullptr || uuid_length == nullptr || *uuid_length < AMDSMI_GPU_UUID_SIZE) {
        return AMDSMI_STATUS_INVAL;
    }

    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    SMIGPUDEVICE_MUTEX(gpu_device->get_mutex())

    size_t len = AMDSMI_GPU_UUID_SIZE;
    amdsmi_asic_info_t asic_info = {};
    const uint8_t fcn = 0xff;

    status = amdsmi_get_gpu_asic_info(processor_handle, &asic_info);
    if (status != AMDSMI_STATUS_SUCCESS) {
        printf("Getting asic info failed. Return code: %d", status);
        return status;
    }

    /* generate random UUID */
    status = amdsmi_uuid_gen(uuid,
                strtoull(asic_info.asic_serial, nullptr, 16),
                (uint16_t)asic_info.device_id, fcn);
    return status;
}

amdsmi_status_t amdsmi_get_pcie_info(amdsmi_processor_handle processor_handle, amdsmi_pcie_info_t *info) {
    AMDSMI_CHECK_INIT();

    if (info == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    amdsmi_status_t r = get_gpu_device_from_handle(processor_handle, &gpu_device);
    if (r != AMDSMI_STATUS_SUCCESS)
        return r;

    SMIGPUDEVICE_MUTEX(gpu_device->get_mutex())

    char buff[AMDSMI_NORMAL_STRING_LENGTH];
    FILE* fp;
    double pcie_speed = 0;
    unsigned pcie_width = 0;

    memset((void *)info, 0, sizeof(*info));

    std::string path_max_link_width = "/sys/class/drm/" +
        gpu_device->get_gpu_path() + "/device/max_link_width";
    fp = fopen(path_max_link_width.c_str(), "r");
    if (fp) {
        fscanf(fp, "%d", &pcie_width);
        fclose(fp);
    } else {
        printf("Failed to open file: %s \n", path_max_link_width.c_str());
        return AMDSMI_STATUS_API_FAILED;
    }
    info->pcie_static.max_pcie_width = (uint16_t)pcie_width;

    std::string path_max_link_speed = "/sys/class/drm/" +
        gpu_device->get_gpu_path() + "/device/max_link_speed";
    fp = fopen(path_max_link_speed.c_str(), "r");
    if (fp) {
        fscanf(fp, "%lf %s", &pcie_speed, buff);
        fclose(fp);
    } else {
        printf("Failed to open file: %s \n", path_max_link_speed.c_str());
        return AMDSMI_STATUS_API_FAILED;
    }

    // pcie speed in sysfs returns in GT/s
    info->pcie_static.max_pcie_speed = pcie_speed * 1000;

    switch (info->pcie_static.max_pcie_speed) {
      case 2500:
        info->pcie_static.pcie_interface_version = 1;
        break;
      case 5000:
        info->pcie_static.pcie_interface_version = 2;
        break;
      case 8000:
        info->pcie_static.pcie_interface_version = 3;
        break;
      case 16000:
        info->pcie_static.pcie_interface_version = 4;
        break;
      case 32000:
        info->pcie_static.pcie_interface_version = 5;
        break;
      case 64000:
        info->pcie_static.pcie_interface_version = 6;
        break;
      default:
        info->pcie_static.pcie_interface_version = 0;
    }

    // default to PCIe
    info->pcie_static.slot_type = AMDSMI_CARD_FORM_FACTOR_PCIE;
    rsmi_pcie_slot_type_t slot_type;
    status = rsmi_wrapper(rsmi_dev_pcie_slot_type_get,
            processor_handle, &slot_type);
    if (status == AMDSMI_STATUS_SUCCESS) {
        switch (slot_type) {
            case RSMI_PCIE_SLOT_PCIE:
                info->pcie_static.slot_type = AMDSMI_CARD_FORM_FACTOR_PCIE;
                break;
            case RSMI_PCIE_SLOT_OAM:
                info->pcie_static.slot_type = AMDSMI_CARD_FORM_FACTOR_OAM;
                break;
            case RSMI_PCIE_SLOT_CEM:
                info->pcie_static.slot_type = AMDSMI_CARD_FORM_FACTOR_CEM;
                break;
            default:
                info->pcie_static.slot_type = AMDSMI_CARD_FORM_FACTOR_UNKNOWN;
        }
    }

    // metrics
    amdsmi_gpu_metrics_t metric_info = {};
    status =  amdsmi_get_gpu_metrics_info(
            processor_handle, &metric_info);
    if (status != AMDSMI_STATUS_SUCCESS)
        return status;

    info->pcie_metric.pcie_width = metric_info.pcie_link_width;
    // gpu metrics is inconsistent with pcie_speed values, if 0-6 then it needs to be translated
    if (metric_info.pcie_link_speed <= 6) {
        status = smi_amdgpu_get_pcie_speed_from_pcie_type(metric_info.pcie_link_speed, &info->pcie_metric.pcie_speed); // mapping to MT/s
    } else {
        // gpu metrics returns pcie link speed in .1 GT/s ex. 160 vs 16
        info->pcie_metric.pcie_speed = translate_umax_or_assign_value<decltype(info->pcie_metric.pcie_speed)>
                                          (metric_info.pcie_link_speed, (metric_info.pcie_link_speed * 100));
    }

    // additional pcie related metrics
    /**
     * pcie_metric.pcie_bandwidth:      MB/s  (uint32_t)
     * metric_info.pcie_bandwidth_inst: GB/s  (uint64_t)
     */
    info->pcie_metric.pcie_bandwidth = translate_umax_or_assign_value<decltype(info->pcie_metric.pcie_bandwidth)>
                                          (metric_info.pcie_bandwidth_inst, metric_info.pcie_bandwidth_inst);
    info->pcie_metric.pcie_replay_count = metric_info.pcie_replay_count_acc;
    info->pcie_metric.pcie_l0_to_recovery_count = metric_info.pcie_l0_to_recov_count_acc;
    info->pcie_metric.pcie_replay_roll_over_count = metric_info.pcie_replay_rover_count_acc;
    /**
     * pcie_metric.pcie_nak_received_count: (uint64_t)
     * metric_info.pcie_nak_rcvd_count_acc: (uint32_t)
     */
    info->pcie_metric.pcie_nak_received_count = translate_umax_or_assign_value<decltype(info->pcie_metric.pcie_nak_received_count)>
                                                  (metric_info.pcie_nak_rcvd_count_acc, (metric_info.pcie_nak_rcvd_count_acc));
    /**
     * pcie_metric.pcie_nak_sent_count:     (uint64_t)
     * metric_info.pcie_nak_sent_count_acc: (uint32_t)
     */
    info->pcie_metric.pcie_nak_sent_count = translate_umax_or_assign_value<decltype(info->pcie_metric.pcie_nak_sent_count)>
                                              (metric_info.pcie_nak_sent_count_acc, (metric_info.pcie_nak_sent_count_acc));
    /**
     * pcie_metric.pcie_lc_perf_other_end_recovery: (uint32_t)
     */
    info->pcie_metric.pcie_lc_perf_other_end_recovery_count =
        translate_umax_or_assign_value<decltype(
            info->pcie_metric.pcie_lc_perf_other_end_recovery_count)> (
                metric_info.pcie_lc_perf_other_end_recovery,
                (metric_info.pcie_lc_perf_other_end_recovery));

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t amdsmi_get_processor_handle_from_bdf(amdsmi_bdf_t bdf,
                amdsmi_processor_handle* processor_handle)
{
    amdsmi_status_t status;
    uint32_t socket_count = 0;

    uint32_t device_count = AMDSMI_MAX_DEVICES;
    amdsmi_processor_handle devs[AMDSMI_MAX_DEVICES];

   AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }

    status = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    amdsmi_socket_handle sockets[socket_count];

    status = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    for (unsigned int i = 0; i < socket_count; i++) {
        status = amdsmi_get_processor_handles(sockets[i], &device_count, devs);
        if (status != AMDSMI_STATUS_SUCCESS) {
            return status;
        }

        for (uint32_t idx = 0; idx < device_count; idx++) {
            amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
            status = get_gpu_device_from_handle(devs[idx], &gpu_device);
            if (status != AMDSMI_STATUS_SUCCESS) {
                return status;
            }
            amdsmi_bdf_t found_bdf = gpu_device->get_bdf();
            if ((bdf.bus_number == found_bdf.bus_number) &&
                (bdf.device_number == found_bdf.device_number) &&
                (bdf.domain_number == found_bdf.domain_number) &&
                (bdf.function_number == found_bdf.function_number)) {
                    *processor_handle = devs[idx];
                    return AMDSMI_STATUS_SUCCESS;
                }
        }
    }

    return AMDSMI_STATUS_API_FAILED;
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
    constexpr auto kKFD_CRAT_INTRA_SOCKET_WEIGHT = uint32_t(13);
    constexpr auto kKFD_CRAT_XGMI_WEIGHT = uint32_t(15);

    /*
     *  Note: This will need to be eventually consolidated within a unique link type.
     */
    static const std::map<amdsmi_link_type_t, amdsmi_io_link_type_t> kLinkToIoLinkTypeTranslationTable =
    {
        {amdsmi_link_type_t::AMDSMI_LINK_TYPE_INTERNAL,       amdsmi_io_link_type_t::AMDSMI_IOLINK_TYPE_UNDEFINED},
        {amdsmi_link_type_t::AMDSMI_LINK_TYPE_XGMI,           amdsmi_io_link_type_t::AMDSMI_IOLINK_TYPE_XGMI},
        {amdsmi_link_type_t::AMDSMI_LINK_TYPE_PCIE,           amdsmi_io_link_type_t::AMDSMI_IOLINK_TYPE_PCIEXPRESS},
        {amdsmi_link_type_t::AMDSMI_LINK_TYPE_NOT_APPLICABLE, amdsmi_io_link_type_t::AMDSMI_IOLINK_TYPE_UNDEFINED},
        {amdsmi_link_type_t::AMDSMI_LINK_TYPE_UNKNOWN,        amdsmi_io_link_type_t::AMDSMI_IOLINK_TYPE_UNDEFINED}
    };

    auto translated_link_type = [&](amdsmi_link_type_t link_type) {
        auto io_link_type(amdsmi_io_link_type_t::AMDSMI_IOLINK_TYPE_UNDEFINED);
        if (kLinkToIoLinkTypeTranslationTable.find(link_type) != kLinkToIoLinkTypeTranslationTable.end()) {
            io_link_type = kLinkToIoLinkTypeTranslationTable.at(link_type);
        }
        return io_link_type;
    };

    auto translated_io_link_type = [&](amdsmi_io_link_type_t io_link_type) {
        auto link_type(amdsmi_link_type_t::AMDSMI_LINK_TYPE_UNKNOWN);
        for (const auto& [key, value] : kLinkToIoLinkTypeTranslationTable) {
            if (value == io_link_type) {
                link_type = key;
                break;
            }
        }
        return link_type;
    };
    //

    struct LinkTopolyInfo_t
    {
        amdsmi_processor_handle target_processor_handle;
        amdsmi_link_type_t link_type;
        bool is_accessible;
        uint64_t num_hops;
        uint64_t link_weight;
    };

    using LinkTopogyOrderPair_t = std::pair<uint64_t, uint64_t>;
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


    uint32_t device_counter(AMDSMI_MAX_DEVICES);
    amdsmi_processor_handle device_list[AMDSMI_MAX_DEVICES];
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
                auto io_link_type = translated_link_type(link_type);
                auto io_link_type_bck(io_link_type);
                auto num_hops = uint64_t(0);
                if (auto api_status = amdsmi_topo_get_link_type(processor_handle, device_list[device_idx], &num_hops, &io_link_type);
                    (api_status != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) || (translated_io_link_type(io_link_type) != link_type)) {
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
                    .link_type = translated_io_link_type(io_link_type),
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
    topology_nearest_info->processor_list[AMDSMI_MAX_DEVICES] = {nullptr};
    topology_nearest_info->count = link_topology_order.size();
    auto topology_nearest_counter = uint32_t(0);
    while (!link_topology_order.empty()) {
        auto link_info = link_topology_order.top();
        link_topology_order.pop();

        if (topology_nearest_counter < AMDSMI_MAX_DEVICES) {
            topology_nearest_info->processor_list[topology_nearest_counter++] = link_info.target_processor_handle;
        }
    }

    return status;
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

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    status = static_cast<amdsmi_status_t>(esmi_ddr_bw_get(&ddr));
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

    AMDSMI_CHECK_INIT();

    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    io_link.link_name = link.link_name;
    io_link.bw_type= static_cast<io_bw_encoding>(link.bw_type);

    status = static_cast<amdsmi_status_t>(esmi_current_xgmi_bw_get(io_link, &bw));
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
    *cpu_count = cpu_handles.size();
    if (processor_handles == nullptr)
        return AMDSMI_STATUS_SUCCESS;

    // Copy the cpu socket handles
    for (uint32_t i = 0; i < *cpu_count; i++)
        processor_handles[i] = reinterpret_cast<amdsmi_processor_handle>(cpu_handles[i]);

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
        if (status != AMDSMI_STATUS_SUCCESS)
            return status;

        core_handles.insert(core_handles.end(), plist.begin(), plist.end());
    }

    // Get the cores count
    *cores_count = core_handles.size();
    if (processor_handles == nullptr)
        return AMDSMI_STATUS_SUCCESS;

    // Copy the core handles
    for (uint32_t i = 0; i < *cores_count; i++)
        processor_handles[i] = reinterpret_cast<amdsmi_processor_handle>(core_handles[i]);

    return status;
}

amdsmi_status_t amdsmi_get_esmi_err_msg(amdsmi_status_t status, const char **status_string)
{
    for (auto& iter : amd::smi::esmi_status_map) {
        if (iter.first == status) {
            *status_string = esmi_get_err_msg(static_cast<esmi_status_t>(iter.first));
            return iter.second;
        }
    }
    return AMDSMI_STATUS_SUCCESS;
}
#endif
