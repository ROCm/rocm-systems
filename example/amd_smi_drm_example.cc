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

#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>
#include <inttypes.h>

#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>

#include "amd_smi/amdsmi.h"


#define CHK_AMDSMI_RET(RET)                                                    \
    {                                                                          \
        if (RET != AMDSMI_STATUS_SUCCESS) {                                    \
            const char *err_str;                                               \
            std::cout << "AMDSMI call returned " << RET << " at line "         \
                      << __LINE__ << std::endl;                                \
            amdsmi_status_code_to_string(RET, &err_str);                       \
            std::cout << err_str << std::endl;                                 \
            return RET;                                                        \
        }                                                                      \
    }


void getFWNameFromId(int id, char *name)
{
    switch (id) {
    case AMDSMI_FW_ID_SMU:
        strcpy(name, "SMU");
        break;
    case AMDSMI_FW_ID_CP_CE:
        strcpy(name, "CP_CE");
        break;
    case AMDSMI_FW_ID_CP_PFP:
        strcpy(name, "CP_PFP");
        break;
    case AMDSMI_FW_ID_CP_ME:
        strcpy(name, "CP_ME");
        break;
    case AMDSMI_FW_ID_CP_MEC_JT1:
        strcpy(name, "CP_MEC_JT1");
        break;
    case AMDSMI_FW_ID_CP_MEC_JT2:
        strcpy(name, "CP_MEC_JT2");
        break;
    case AMDSMI_FW_ID_CP_MEC1:
        strcpy(name, "CP_MEC1");
        break;
    case AMDSMI_FW_ID_CP_MEC2:
        strcpy(name, "CP_MEC2");
        break;
    case AMDSMI_FW_ID_RLC:
        strcpy(name, "RLC");
        break;
    case AMDSMI_FW_ID_SDMA0:
        strcpy(name, "SDMA0");
        break;
    case AMDSMI_FW_ID_SDMA1:
        strcpy(name, "SDMA1");
        break;
    case AMDSMI_FW_ID_SDMA2:
        strcpy(name, "SDMA2");
        break;
    case AMDSMI_FW_ID_SDMA3:
        strcpy(name, "SDMA3");
        break;
    case AMDSMI_FW_ID_SDMA4:
        strcpy(name, "SDMA4");
        break;
    case AMDSMI_FW_ID_SDMA5:
        strcpy(name, "SDMA5");
        break;
    case AMDSMI_FW_ID_SDMA6:
        strcpy(name, "SDMA6");
        break;
    case AMDSMI_FW_ID_SDMA7:
        strcpy(name, "SDMA7");
        break;
    case AMDSMI_FW_ID_VCN:
        strcpy(name, "VCN");
        break;
    case AMDSMI_FW_ID_UVD:
        strcpy(name, "UVD");
        break;
    case AMDSMI_FW_ID_VCE:
        strcpy(name, "VCE");
        break;
    case AMDSMI_FW_ID_ISP:
        strcpy(name, "ISP");
        break;
    case AMDSMI_FW_ID_DMCU_ERAM:
        strcpy(name, "DMCU_ERAM");
        break;
    case AMDSMI_FW_ID_DMCU_ISR:
        strcpy(name, "DMCU_ISR");
        break;
    case AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM:
        strcpy(name, "RLC_RESTORE_LIST_GPM_MEM");
        break;
    case AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM:
        strcpy(name, "RLC_RESTORE_LIST_SRM_MEM");
        break;
    case AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL:
        strcpy(name, "RLC_RESTORE_LIST_CNTL");
        break;
    case AMDSMI_FW_ID_RLC_V:
        strcpy(name, "RLC_V");
        break;
    case AMDSMI_FW_ID_MMSCH:
        strcpy(name, "MMSCH");
        break;
    case AMDSMI_FW_ID_PSP_SYSDRV:
        strcpy(name, "PSP_SYSDRV");
        break;
    case AMDSMI_FW_ID_PSP_SOSDRV:
        strcpy(name, "PSP_SOSDRV");
        break;
    case AMDSMI_FW_ID_PSP_TOC:
        strcpy(name, "PSP_TOC");
        break;
    case AMDSMI_FW_ID_PSP_KEYDB:
        strcpy(name, "PSP_KEYDB");
        break;
    case AMDSMI_FW_ID_DFC:
        strcpy(name, "DFC");
        break;
    case AMDSMI_FW_ID_PSP_SPL:
        strcpy(name, "PSP_SPL");
        break;
    case AMDSMI_FW_ID_DRV_CAP:
        strcpy(name, "DRV_CAP");
        break;
    case AMDSMI_FW_ID_MC:
        strcpy(name, "MC");
        break;
    case AMDSMI_FW_ID_PSP_BL:
        strcpy(name, "PSP_BL");
        break;
    case AMDSMI_FW_ID_CP_PM4:
        strcpy(name, "CP_PM4");
        break;
    case AMDSMI_FW_ID_ASD:
        strcpy(name, "ID_ASD");
        break;
    case AMDSMI_FW_ID_TA_RAS:
        strcpy(name, "ID_TA_RAS");
        break;
    case AMDSMI_FW_ID_TA_XGMI:
        strcpy(name, "ID_TA_XGMI");
        break;
    case AMDSMI_FW_ID_RLC_SRLG:
        strcpy(name, "ID_RLC_SRLG");
        break;
    case AMDSMI_FW_ID_RLC_SRLS:
        strcpy(name, "ID_RLC_SRLS");
        break;
    case AMDSMI_FW_ID_PM:
        strcpy(name, "ID_PM");
        break;
    case AMDSMI_FW_ID_DMCU:
        strcpy(name, "ID_DMCU");
        break;
    default:
        strcpy(name, "");
        break;
    }
}

template <typename T>
std::string print_unsigned_int(T value) {
  std::stringstream ss;
  ss << static_cast<uint64_t>(value | 0);

  return ss.str();
}

static const std::string
computePartitionString(amdsmi_compute_partition_type_t computeParitionType) {
  switch (computeParitionType) {
    case AMDSMI_COMPUTE_PARTITION_SPX:
      return "SPX";
    case AMDSMI_COMPUTE_PARTITION_DPX:
      return "DPX";
    case AMDSMI_COMPUTE_PARTITION_TPX:
      return "TPX";
    case AMDSMI_COMPUTE_PARTITION_QPX:
      return "QPX";
    case AMDSMI_COMPUTE_PARTITION_CPX:
      return "CPX";
    default:
      return "N/A";
  }
}

static const std::map<std::string, amdsmi_compute_partition_type_t>
mapStringToSMIComputePartitionTypes {
  {"SPX", AMDSMI_COMPUTE_PARTITION_SPX},
  {"DPX", AMDSMI_COMPUTE_PARTITION_DPX},
  {"TPX", AMDSMI_COMPUTE_PARTITION_TPX},
  {"QPX", AMDSMI_COMPUTE_PARTITION_QPX},
  {"CPX", AMDSMI_COMPUTE_PARTITION_CPX},
  {"N/A", AMDSMI_COMPUTE_PARTITION_INVALID}
};

static const std::string
memoryPartitionString(amdsmi_memory_partition_type_t memoryParitionType) {
  switch (memoryParitionType) {
    case AMDSMI_MEMORY_PARTITION_NPS1:
      return "NPS1";
    case AMDSMI_MEMORY_PARTITION_NPS2:
      return "NPS2";
    case AMDSMI_MEMORY_PARTITION_NPS4:
      return "NPS4";
    case AMDSMI_MEMORY_PARTITION_NPS8:
      return "NPS8";
    default:
      return "N/A";
  }
}

static const std::map<std::string, amdsmi_memory_partition_type_t>
mapStringToSMIMemoryPartitionTypes {
  {"NPS1", AMDSMI_MEMORY_PARTITION_NPS1},
  {"NPS2", AMDSMI_MEMORY_PARTITION_NPS2},
  {"NPS4", AMDSMI_MEMORY_PARTITION_NPS4},
  {"NPS8", AMDSMI_MEMORY_PARTITION_NPS8},
  {"N/A", AMDSMI_MEMORY_PARTITION_UNKNOWN}
};

int main() {
    amdsmi_status_t ret, ret_set;
    const char *err_str;

    // Init amdsmi for sockets and devices.
    // Here we are only interested in AMD_GPUS.
    ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    CHK_AMDSMI_RET(ret)

    // Get all sockets
    uint32_t socket_count = 0;

    // Get the socket count available for the system.
    ret = amdsmi_get_socket_handles(&socket_count, nullptr);
    CHK_AMDSMI_RET(ret)

    // Allocate the memory for the sockets
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    // Get the sockets of the system
    ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
    CHK_AMDSMI_RET(ret)

    std::cout << "Total Socket: " << socket_count << std::endl;

    // For each socket, get identifier and devices
    for (uint32_t i = 0; i < socket_count; i++) {
        // Get Socket info
        char socket_info[128];
        ret = amdsmi_get_socket_info(sockets[i], 128, socket_info);
        CHK_AMDSMI_RET(ret)
        std::cout << "Socket " << socket_info << std::endl;

        // Get the device count available for the socket.
        uint32_t device_count = 0;
        ret = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);
        CHK_AMDSMI_RET(ret)

        // Allocate the memory for the device handlers on the socket
        std::vector<amdsmi_processor_handle> processor_handles(device_count);
        // Get all devices of the socket
        ret = amdsmi_get_processor_handles(sockets[i],
                                        &device_count, &processor_handles[0]);
        CHK_AMDSMI_RET(ret)

        std::cout << "Processor Count: " << device_count << std::endl;

        // For each device of the socket, get name and temperature.
        for (uint32_t device_index = 0; device_index < device_count; device_index++) {
            std::cout << "Device Index: " << device_index << std::endl;
#ifdef ENABLE_ESMI_LIB
            uint32_t cpu_sockets = 0;
            uint32_t gpus = 0;
            ret = amdsmi_get_processor_count_from_handles(&processor_handles[device_index], &device_count, &cpu_sockets, nullptr, &gpus);
            CHK_AMDSMI_RET(ret)
            std::cout << "CPU socket count: " << cpu_sockets << std::endl;
            std::cout << "GPU count: " << gpus << std::endl;
#endif

            // Get device type. Since the amdsmi is initialized with
            // AMD_SMI_INIT_AMD_GPUS, the processor_type must be AMDSMI_PROCESSOR_TYPE_AMD_GPU.
            processor_type_t processor_type = {};
            ret = amdsmi_get_processor_type(processor_handles[device_index], &processor_type);
            CHK_AMDSMI_RET(ret)
            if (processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
                std::cout << "Expect AMDSMI_PROCESSOR_TYPE_AMD_GPU device type!\n";
                return AMDSMI_STATUS_NOT_SUPPORTED;
            }

            // Get BDF info
            amdsmi_bdf_t bdf = {};
            ret = amdsmi_get_gpu_device_bdf(processor_handles[device_index], &bdf);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_device_bdf:\n");
            printf("\tDevice[%d] BDF %04" PRIx64 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32 "\n\n", i,
                   static_cast<uint64_t>(bdf.domain_number),
                   static_cast<uint32_t>(bdf.bus_number),
                   static_cast<uint32_t>(bdf.device_number),
                   static_cast<uint32_t>(bdf.function_number));

            // Get handle from BDF
            amdsmi_processor_handle dev_handle;
            ret = amdsmi_get_processor_handle_from_bdf(bdf, &dev_handle);
            CHK_AMDSMI_RET(ret)

            // Get ASIC info
            amdsmi_asic_info_t asic_info = {};
            ret = amdsmi_get_gpu_asic_info(processor_handles[device_index], &asic_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_asic_info:\n");
            printf("\tMarket Name: %s\n", asic_info.market_name);
            printf("\tDeviceID: 0x%lx\n", asic_info.device_id);
            printf("\tVendorID: 0x%x\n", asic_info.vendor_id);
            printf("\tRevisionID: 0x%x\n", asic_info.rev_id);
            printf("\tAsic serial: 0x%s\n", asic_info.asic_serial);
            printf("\tNum of Computes: %d\n\n", asic_info.num_of_compute_units);

            bool is_power_management_enabled = false;
            ret = amdsmi_is_gpu_power_management_enabled(processor_handles[device_index],
                                                    &is_power_management_enabled);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_is_gpu_power_management_enabled:\n");
            printf("\tPower Management Enabled: %s\n\n",
                    (is_power_management_enabled ? "TRUE" : "FALSE"));

            std::cout << "    **Version 1: Accelerator/Compute Partition API Examples**\n";
            char original_compute_partition[AMDSMI_MAX_STRING_LENGTH];
            ret = amdsmi_get_gpu_compute_partition(processor_handles[device_index], original_compute_partition,
                                        static_cast<uint32_t>(AMDSMI_MAX_STRING_LENGTH));

            amdsmi_status_code_to_string(ret, &err_str);
            if (ret == AMDSMI_STATUS_SUCCESS) {
                CHK_AMDSMI_RET(ret)
                std::cout << "    Output of amdsmi_get_gpu_compute_partition:\n";
                std::cout << "\tamdsmi_get_gpu_compute_partition(" << device_index << ", "
                << mapStringToSMIComputePartitionTypes.at(original_compute_partition) << "): "
                << err_str << "\n\n";
                std::cout << "\tCompute Partition (original): "
                << original_compute_partition << "\n\n";
            } else {
                std::cout << "\tamdsmi_get_gpu_compute_partition(" << device_index << ", "
                << computePartitionString(AMDSMI_COMPUTE_PARTITION_INVALID) << "): "
                << err_str << "\n\n";
            }

            for (int partition = static_cast<int>(AMDSMI_COMPUTE_PARTITION_SPX);
                    partition <= static_cast<int>(AMDSMI_COMPUTE_PARTITION_CPX);
                    partition++) {
                amdsmi_compute_partition_type_t updatePartition
                    = static_cast<amdsmi_compute_partition_type_t>(partition);
                ret_set = amdsmi_set_gpu_compute_partition(processor_handles[device_index],
                                                                updatePartition);
                amdsmi_status_code_to_string(ret_set, &err_str);
                if (ret_set == AMDSMI_STATUS_SUCCESS) {
                    CHK_AMDSMI_RET(ret_set)
                }
                std::cout << "\tamdsmi_set_gpu_compute_partition(" << device_index << ", "
                << computePartitionString(updatePartition) << "): "
                << err_str << "\n\n";

                // Get the current compute partition
                char current_compute_partition[AMDSMI_MAX_STRING_LENGTH];
                ret = amdsmi_get_gpu_compute_partition(processor_handles[device_index],
                                                current_compute_partition,
                                                static_cast<uint32_t>(AMDSMI_MAX_STRING_LENGTH));
                amdsmi_status_code_to_string(ret, &err_str);
                if (ret == AMDSMI_STATUS_SUCCESS) {
                    CHK_AMDSMI_RET(ret)
                    std::cout << "    Output of amdsmi_get_gpu_compute_partition:\n";
                    std::cout << "\tamdsmi_get_gpu_compute_partition(" << device_index << ", "
                              << computePartitionString(updatePartition) << "): "
                              << err_str << "\n\n";
                    std::cout << "\tCompute Partition (current): "
                              << current_compute_partition << "\n\n";
                } else {
                    std::cout << "\tamdsmi_get_gpu_compute_partition(" << device_index << ", "
                    << computePartitionString(AMDSMI_COMPUTE_PARTITION_INVALID) << "): "
                    << err_str << "\n\n";
                }
            }
            // return to original compute partition
            amdsmi_compute_partition_type_t original_compute_partition_type;
            if (ret == AMDSMI_STATUS_SUCCESS) {
                original_compute_partition_type
                    = mapStringToSMIComputePartitionTypes.at(original_compute_partition);
            } else {
                original_compute_partition_type = AMDSMI_COMPUTE_PARTITION_INVALID;
            }
            std::cout << "    Returning to original compute partition ("
                      << computePartitionString(original_compute_partition_type) << ")\n";
            auto ret_set = amdsmi_set_gpu_compute_partition(processor_handles[device_index],
                                                original_compute_partition_type);
            amdsmi_status_code_to_string(ret_set, &err_str);
            if (ret_set == AMDSMI_STATUS_SUCCESS) {
                CHK_AMDSMI_RET(ret_set)
            }
            std::cout << "\tamdsmi_set_gpu_compute_partition(" << device_index << ", "
            << computePartitionString(original_compute_partition_type) << "): "
            << err_str << "\n\n";

            std::cout << "    **Version 1: Memory Partition API Examples**\n";
            char original_memory_partition[AMDSMI_MAX_STRING_LENGTH];
            ret = amdsmi_get_gpu_memory_partition(processor_handles[device_index], original_memory_partition,
                                        static_cast<uint32_t>(AMDSMI_MAX_STRING_LENGTH));
            amdsmi_status_code_to_string(ret, &err_str);
            if (ret == AMDSMI_STATUS_SUCCESS) {
                CHK_AMDSMI_RET(ret)
                std::cout << "    Output of amdsmi_get_gpu_memory_partition:\n";
                std::cout << "\tamdsmi_get_gpu_memory_partition(" << device_index << ", "
                << mapStringToSMIMemoryPartitionTypes.at(original_memory_partition) << "): "
                << err_str << "\n\n";
                std::cout << "\tMemory Partition (original): "
                << original_memory_partition << "\n\n";
            } else {
                std::cout << "\tamdsmi_get_gpu_memory_partition(" << device_index << ", "
                << memoryPartitionString(AMDSMI_MEMORY_PARTITION_UNKNOWN) << "): "
                << err_str << "\n\n";
            }

            for (int partition = static_cast<int>(AMDSMI_MEMORY_PARTITION_NPS1);
                    partition <= static_cast<int>(AMDSMI_MEMORY_PARTITION_NPS8);
                    partition++) {
                if (partition != static_cast<int>(AMDSMI_MEMORY_PARTITION_NPS1)
                    && partition != static_cast<int>(AMDSMI_MEMORY_PARTITION_NPS2)
                    && partition != static_cast<int>(AMDSMI_MEMORY_PARTITION_NPS4)
                    && partition != static_cast<int>(AMDSMI_MEMORY_PARTITION_NPS8)) {
                    continue;
                }
                amdsmi_memory_partition_type_t updatePartition
                    = static_cast<amdsmi_memory_partition_type_t>(partition);
                auto ret_set = amdsmi_set_gpu_memory_partition(processor_handles[device_index],
                                                                updatePartition);
                amdsmi_status_code_to_string(ret_set, &err_str);
                if (ret_set == AMDSMI_STATUS_SUCCESS) {
                    CHK_AMDSMI_RET(ret_set)
                    std::cout << "    Output of amdsmi_set_gpu_memory_partition:\n";
                }
                std::cout << "\tamdsmi_set_gpu_memory_partition(" << device_index << ", "
                          << memoryPartitionString(updatePartition) << "): "
                          << err_str << "\n\n";

                // Get the current memory partition
                char current_memory_partition[AMDSMI_MAX_STRING_LENGTH];
                ret = amdsmi_get_gpu_memory_partition(processor_handles[device_index],
                                            current_memory_partition,
                                            static_cast<uint32_t>(AMDSMI_MAX_STRING_LENGTH));

                amdsmi_status_code_to_string(ret, &err_str);
                if (ret == AMDSMI_STATUS_SUCCESS) {
                    CHK_AMDSMI_RET(ret)
                    std::cout << "\tamdsmi_get_gpu_memory_partition(" << device_index << ", "
                    << memoryPartitionString(updatePartition) << "): "
                    << err_str << "\n\n";
                    std::cout << "\tMemory Partition (current): "
                    << current_memory_partition << "\n\n";
                } else {
                    std::cout << "\tamdsmi_get_gpu_memory_partition(" << device_index << ", "
                    << memoryPartitionString(AMDSMI_MEMORY_PARTITION_UNKNOWN) << "): "
                    << err_str << "\n\n";
                }
            }
            // return to original compute partition
            amdsmi_memory_partition_type_t original_memory_partition_type;
            if (ret == AMDSMI_STATUS_SUCCESS) {
                original_memory_partition_type
                    = mapStringToSMIMemoryPartitionTypes.at(original_memory_partition);
            } else {
                original_memory_partition_type = AMDSMI_MEMORY_PARTITION_UNKNOWN;
            }
            std::cout << "    Returning to original memory partition ("
                      << memoryPartitionString(original_memory_partition_type)
                      << ")\n";
            ret_set = amdsmi_set_gpu_memory_partition(processor_handles[device_index],
                                                original_memory_partition_type);
            amdsmi_status_code_to_string(ret_set, &err_str);
            if (ret_set == AMDSMI_STATUS_SUCCESS) {
                CHK_AMDSMI_RET(ret_set)
            }
            std::cout << "\tamdsmi_set_gpu_compute_partition(" << device_index << ", "
                      << memoryPartitionString(original_memory_partition_type) << "): "
                      << err_str << "\n\n";

            // TODO(amdsmi_team): Add V2 partiton APIs

            // Get VRAM info
            amdsmi_vram_info_t vram_info = {};
            ret = amdsmi_get_gpu_vram_info(processor_handles[device_index], &vram_info);
            if (ret != amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED) {
                CHK_AMDSMI_RET(ret)
                printf("    Output of amdsmi_get_gpu_vram_info:\n");
                printf("\tVRAM Size: 0x%lx (%ld) \n", vram_info.vram_size, vram_info.vram_size);
                printf("\tBIT Width: 0x%x (%d) \n\n", vram_info.vram_bit_width,
                        vram_info.vram_bit_width);
                printf("\tVRAM max bandwidth: 0x%lx (%lu) \n\n", vram_info.vram_max_bandwidth,
                        vram_info.vram_max_bandwidth);
            } else {
                printf("\t**amdsmi_get_gpu_vram_info() not supported on this system.\n");
            }


            // Get VBIOS info
            amdsmi_vbios_info_t vbios_info = {};
            ret = amdsmi_get_gpu_vbios_info(processor_handles[device_index], &vbios_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_vbios_info:\n");
            printf("\tVBios Name: %s\n", vbios_info.name);
            printf("\tBuild Date: %s\n", vbios_info.build_date);
            printf("\tPart Number: %s\n", vbios_info.part_number);
            printf("\tVBios Version String: %s\n\n",
                   vbios_info.version);

            // Get Cache info
            amdsmi_gpu_cache_info_t cache_info = {};
            ret = amdsmi_get_gpu_cache_info(processor_handles[device_index], &cache_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_cache_info:\n");
            for (unsigned int i = 0 ; i < cache_info.num_cache_types; i++) {
                printf("\tCache Level: %d, Cache Size: %d KB, Cache type: 0x%x\n",
                    cache_info.cache[i].cache_level,
                    cache_info.cache[i].cache_size,
                    cache_info.cache[i].cache_properties);
                printf("\tMax number CU shared: %d, Number of instances: %d\n",
                    cache_info.cache[i].max_num_cu_shared,
                    cache_info.cache[i].num_cache_instance);
            }

            // Get power measure
            amdsmi_power_info_t power_measure = {};
            ret = amdsmi_get_power_info(processor_handles[device_index], &power_measure);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_power_info:\n");
            printf("\tCurrent GFX Voltage: %d\n",
                   power_measure.gfx_voltage);
            printf("\tAverage socket power: %d\n",
                   power_measure.average_socket_power);
            printf("\tGPU Power limit: %d\n\n", power_measure.power_limit);

            // Get driver version
            amdsmi_driver_info_t driver_info;
            ret = amdsmi_get_gpu_driver_info(processor_handles[device_index], &driver_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_driver_info:\n");
            printf("\tDriver name: %s\n", driver_info.driver_name);
            printf("\tDriver version: %s\n", driver_info.driver_version);
            printf("\tDriver date: %s\n\n", driver_info.driver_date);

            // Get device uuid
            unsigned int uuid_length = AMDSMI_GPU_UUID_SIZE;
	        char uuid[AMDSMI_GPU_UUID_SIZE];
            ret = amdsmi_get_gpu_device_uuid(processor_handles[device_index], &uuid_length, uuid);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_device_uuid:\n");
            printf("\tDevice uuid: %s\n\n", uuid);

            // Get engine usage info
            amdsmi_engine_usage_t engine_usage = {};
            ret = amdsmi_get_gpu_activity(processor_handles[device_index], &engine_usage);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_activity:\n");
            printf("\tAverage GFX Activity: %d\n",
                   engine_usage.gfx_activity);
            printf("\tAverage MM Activity: %d\n",
                   engine_usage.mm_activity);
            printf("\tAverage UMC Activity: %d\n\n",
                   engine_usage.umc_activity);

            // Get firmware info
            amdsmi_fw_info_t fw_information = {};
            char ucode_name[AMDSMI_MAX_STRING_LENGTH];
            ret = amdsmi_get_fw_info(processor_handles[device_index], &fw_information);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_fw_info:\n");
            printf("Number of Microcodes: %d\n", fw_information.num_fw_info);
            for (int j = 0; j < fw_information.num_fw_info; j++) {
                getFWNameFromId(fw_information.fw_info_list[j].fw_id, ucode_name);
                printf("        %s: %ld\n", ucode_name, fw_information.fw_info_list[j].fw_version);
            }

            // Get GFX clock measurements
            amdsmi_clk_info_t gfx_clk_values = {};
            ret = amdsmi_get_clock_info(processor_handles[device_index], AMDSMI_CLK_TYPE_GFX,
                                           &gfx_clk_values);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_clock_info:\n");
            printf("\tGPU GFX Max Clock: %d\n", gfx_clk_values.max_clk);
            printf("\tGPU GFX Current Clock: %d\n", gfx_clk_values.clk);

            // Get MEM clock measurements
            amdsmi_clk_info_t mem_clk_values = {};
            ret = amdsmi_get_clock_info(processor_handles[device_index], AMDSMI_CLK_TYPE_MEM,
                                           &mem_clk_values);
            CHK_AMDSMI_RET(ret)
            printf("\tGPU MEM Max Clock: %d\n", mem_clk_values.max_clk);
            printf("\tGPU MEM Current Clock: %d\n\n", mem_clk_values.clk);

            // Get PCIe status
            amdsmi_pcie_info_t pcie_info = {};
            ret = amdsmi_get_pcie_info(processor_handles[device_index], &pcie_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_pcie_info:\n");
            printf("\tCurrent PCIe lanes: %d\n", pcie_info.pcie_metric.pcie_width);
            printf("\tCurrent PCIe speed: %d\n", pcie_info.pcie_metric.pcie_speed);
            printf("\tCurrent PCIe Interface Version: %d\n",
                                    pcie_info.pcie_static.pcie_interface_version);
            printf("\tPCIe slot type: %d\n", pcie_info.pcie_static.slot_type);
            printf("\tPCIe max lanes: %d\n", pcie_info.pcie_static.max_pcie_width);
            printf("\tPCIe max speed: %d\n", pcie_info.pcie_static.max_pcie_speed);

            // additional pcie related metrics
            printf("\tPCIe bandwidth: %u\n", pcie_info.pcie_metric.pcie_bandwidth);
            printf("\tPCIe replay count: %" PRIu64 "\n", pcie_info.pcie_metric.pcie_replay_count);
            printf("\tPCIe L0 recovery count: %" PRIu64 "\n", pcie_info.pcie_metric.pcie_l0_to_recovery_count);
            printf("\tPCIe rollover count: %" PRIu64 "\n", pcie_info.pcie_metric.pcie_replay_roll_over_count);
            printf("\tPCIe nak received count: %" PRIu64 "\n", pcie_info.pcie_metric.pcie_nak_received_count);
            printf("\tPCIe nak sent count: %" PRIu64 "\n", pcie_info.pcie_metric.pcie_nak_sent_count);

            // Get VRAM temperature limit
            int64_t temperature = 0;
            ret = amdsmi_get_temp_metric(
                processor_handles[device_index], AMDSMI_TEMPERATURE_TYPE_VRAM,
                AMDSMI_TEMP_CRITICAL, &temperature);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_temp_metric:\n");
            printf("\tGPU VRAM temp limit: %ld\n", temperature);

            // Get GFX temperature limit
            ret = amdsmi_get_temp_metric(
                processor_handles[device_index], AMDSMI_TEMPERATURE_TYPE_EDGE,
                AMDSMI_TEMP_CRITICAL, &temperature);
            if (ret != amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED) {
                CHK_AMDSMI_RET(ret)
            }
            printf("\tGPU GFX temp limit: %ld\n\n", temperature);

            // Get temperature measurements
            // amdsmi_temperature_t edge_temp, hotspot_temp, vram_temp,
            // plx_temp;
            int64_t temp_measurements[AMDSMI_TEMPERATURE_TYPE__MAX + 1];
            amdsmi_temperature_type_t temp_types[4] = {
                AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                AMDSMI_TEMPERATURE_TYPE_VRAM, AMDSMI_TEMPERATURE_TYPE_PLX};
            for (const auto &temp_type : temp_types) {
                ret = amdsmi_get_temp_metric(
                    processor_handles[device_index], temp_type,
                    AMDSMI_TEMP_CURRENT,
                    &temp_measurements[(int)(temp_type)]);
                if (ret != amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED) {
                    CHK_AMDSMI_RET(ret)
                }
            }
            printf("    Output of amdsmi_get_temp_metric:\n");
            printf("\tGPU Edge temp measurement: %ld\n",
                   temp_measurements[AMDSMI_TEMPERATURE_TYPE_EDGE]);
            printf("\tGPU Hotspot temp measurement: %ld\n",
                   temp_measurements[AMDSMI_TEMPERATURE_TYPE_HOTSPOT]);
            printf("\tGPU VRAM temp measurement: %ld\n",
                   temp_measurements[AMDSMI_TEMPERATURE_TYPE_VRAM]);
            printf("\tGPU PLX temp measurement: %ld\n\n",
                   temp_measurements[AMDSMI_TEMPERATURE_TYPE_PLX]);

            // Get RAS features enabled
            char block_names[14][10] = {"UMC",   "SDMA",     "GFX", "MMHUB",
                                        "ATHUB", "PCIE_BIF", "HDP", "XGMI_WAFL",
                                        "DF",    "SMN",      "SEM", "MP0",
                                        "MP1",   "FUSE"};
            char status_names[7][10] = {"NONE",   "DISABLED", "PARITY",
                                        "SING_C", "MULT_UC",  "POISON",
                                        "ENABLED"};
            amdsmi_ras_err_state_t state = {};
            int index = 0;
            printf("    Output of amdsmi_get_gpu_ras_block_features_enabled:\n");
            for (auto block = AMDSMI_GPU_BLOCK_FIRST;
                 block <= AMDSMI_GPU_BLOCK_LAST;
                 block = (amdsmi_gpu_block_t)(block * 2)) {
                ret = amdsmi_get_gpu_ras_block_features_enabled(processor_handles[device_index], block,
                                                      &state);
                if (ret != AMDSMI_STATUS_API_FAILED && ret != AMDSMI_STATUS_NOT_SUPPORTED) {
                    CHK_AMDSMI_RET(ret)
                }

                printf("\tBlock: %s\n", block_names[index]);
                printf("\tStatus: %s\n", status_names[state]);
                index++;
            }
            printf("\n");

            // Get bad pages
            char bad_page_status_names[3][15] = {"RESERVED", "PENDING",
                                                 "UNRESERVABLE"};
            uint32_t num_pages = 0;
            ret = amdsmi_get_gpu_bad_page_info(processor_handles[device_index], &num_pages,
                                           nullptr);
            if (ret != AMDSMI_STATUS_NOT_SUPPORTED) {
                CHK_AMDSMI_RET(ret)
            }
            printf("    Output of amdsmi_get_gpu_bad_page_info:\n");
            if (!num_pages) {
                printf("\tNo bad pages found.\n");
            } else {
                std::vector<amdsmi_retired_page_record_t> bad_page_info(num_pages);
                ret = amdsmi_get_gpu_bad_page_info(processor_handles[device_index], &num_pages,
                                               bad_page_info.data());
                CHK_AMDSMI_RET(ret)
                for (uint32_t page_it = 0; page_it < num_pages; page_it += 1) {
                    printf("      Page[%d]\n", page_it);
                    printf("\tAddress: %lu\n",
                           bad_page_info[page_it].page_address);
                    printf("\tSize: %lu\n", bad_page_info[page_it].page_size);
                    printf(
                        "\tStatus: %s\n",
                        bad_page_status_names[bad_page_info[page_it].status]);
                }
            }
            printf("\n");

            // Get ECC error counts
            amdsmi_error_count_t err_cnt_info = {};
            ret = amdsmi_get_gpu_total_ecc_count(processor_handles[device_index], &err_cnt_info);
            if (ret != AMDSMI_STATUS_NOT_SUPPORTED) {
                CHK_AMDSMI_RET(ret)
            }
            printf("    Output of amdsmi_get_gpu_total_ecc_count:\n");
            printf("\tCorrectable errors: %lu\n", err_cnt_info.correctable_count);
            printf("\tUncorrectable errors: %lu\n\n",
                   err_cnt_info.uncorrectable_count);

            uint32_t num_process = 0;
            ret = amdsmi_get_gpu_process_list(processor_handles[device_index], &num_process, nullptr);
            CHK_AMDSMI_RET(ret)
            if (!num_process) {
                printf("amdsmi_get_gpu_process_list(): No processes found.\n\n");
            } else {
                std::cout << "Processes found: " << num_process << "\n";
                amdsmi_proc_info_t process_info_list[num_process];
                amdsmi_proc_info_t process = {};
                uint64_t mem = 0, gtt_mem = 0, cpu_mem = 0, vram_mem = 0;
                uint64_t gfx = 0, enc = 0;
                char bdf_str[20];
                sprintf(bdf_str, "%04" PRIx64 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32,
                   static_cast<uint64_t>(bdf.domain_number),
                   static_cast<uint32_t>(bdf.bus_number),
                   static_cast<uint32_t>(bdf.device_number),
                   static_cast<uint32_t>(bdf.function_number));
                ret = amdsmi_get_gpu_process_list(processor_handles[device_index], &num_process, process_info_list);
                std::cout << "Allocation size for process list: " << num_process << "\n";
                CHK_AMDSMI_RET(ret);
                for (auto idx = uint32_t(0); idx < num_process; ++idx) {
                    process = static_cast<amdsmi_proc_info_t>(process_info_list[idx]);
                    printf("\t *Process id: %d / Name: %s / VRAM: %ld \n", process.pid, process.name, process.memory_usage.vram_mem);
                }

                printf("+=======+==================+============+=============="
                       "+=============+=============+=============+============"
                       "==+=========================================+\n");
                printf(
                    "| pid   | name             | user       | gpu bdf      | "
                    "fb usage    | gtt memory  | cpu memory  | vram memory  | "
                    "engine usage (ns)                       |\n");
                printf("|       |                  |            |              "
                       "|             |             |             |            "
                       "  | gfx     enc     |\n");
                printf("+=======+"
                       "+=============+=============+=============+============"
                       "==+=========================================+\n");
                for (int it = 0; it < static_cast<int>(num_process); it++) {
                    char command[30];
                    struct passwd *pwd = nullptr;
                    struct stat st;

                    sprintf(command, "/proc/%d", process_info_list[it].pid);
                    if (stat(command, &st))
                        continue;
                    pwd = getpwuid(st.st_uid);
                    if (!pwd)
                        printf("| %5d | %16s | %10d | %s | %7ld KiB | %7ld KiB "
                               "| %7ld KiB | %7ld KiB  | %lu  %lu |\n",
                               process_info_list[it].pid, process_info_list[it].name, st.st_uid,
                               bdf_str, process_info_list[it].mem / 1024,
                               process_info_list[it].memory_usage.gtt_mem / 1024,
                               process_info_list[it].memory_usage.cpu_mem / 1024,
                               process_info_list[it].memory_usage.vram_mem / 1024,
                               process_info_list[it].engine_usage.gfx,
                               process_info_list[it].engine_usage.enc);
                    else
                        printf("| %5d | %16s | %10s | %s | %7ld KiB | %7ld KiB "
                               "| %7ld KiB | %7ld KiB  | %lu  %lu |\n",
                               process_info_list[it].pid, process_info_list[it].name,
                               pwd->pw_name, bdf_str, process_info_list[it].mem / 1024,
                               process_info_list[it].memory_usage.gtt_mem / 1024,
                               process_info_list[it].memory_usage.cpu_mem / 1024,
                               process_info_list[it].memory_usage.vram_mem / 1024,
                               process_info_list[it].engine_usage.gfx,
                               process_info_list[it].engine_usage.enc);
                    mem += process_info_list[it].mem / 1024;
                    gtt_mem += process_info_list[it].memory_usage.gtt_mem / 1024;
                    cpu_mem += process_info_list[it].memory_usage.cpu_mem / 1024;
                    vram_mem += process_info_list[it].memory_usage.vram_mem / 1024;
                    gfx = process_info_list[it].engine_usage.gfx;
                    enc = process_info_list[it].engine_usage.enc;
                    printf(
                        "+-------+------------------+------------+-------------"
                        "-+-------------+-------------+-------------+----------"
                        "----+-----------------------------------------+\n");
                }
                // TODO: To remove compiler warning, the last 3 values in this printf were
                //       set to 0L.  Need to find out what these values need to be.
                printf("|                                 TOTAL:| %s | %7ld "
                       "KiB | %7ld KiB | %7ld KiB | %7ld KiB | %lu  %lu  "
                       "%lu  %lu  %lu   |\n",
                       bdf_str, mem, gtt_mem, cpu_mem, vram_mem, gfx,
                       enc, 0L, 0L, 0L);
                printf("+=======+==================+============+=============="
                       "+=============+=============+=============+============"
                       "=+==========================================+\n");
            }

            // Get device name
            amdsmi_board_info_t board_info = {};
            ret = amdsmi_get_gpu_board_info(processor_handles[device_index], &board_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_board_info:\n");
            std::cout << "\tdevice [" << device_index
                      << "]\n\t\tProduct name: " << board_info.product_name
                      << "\n"
                      << "\t\tModel Number: " << board_info.model_number
                      << "\n"
                      << "\t\tBoard Serial: " << board_info.product_serial
                      << "\n"
                      << "\t\tManufacturer Name: " << board_info.manufacturer_name
                      << "\n\n";

            // Get temperature
            int64_t val_i64 = 0;
            ret =  amdsmi_get_temp_metric(processor_handles[device_index], AMDSMI_TEMPERATURE_TYPE_EDGE,
                                             AMDSMI_TEMP_CURRENT, &val_i64);
            if (ret != amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED) {
                CHK_AMDSMI_RET(ret)
            }
            printf("    Output of  amdsmi_get_temp_metric:\n");
            std::cout << "\t\tTemperature: " << val_i64 << "C"
                      << "\n\n";

            // Get frame buffer
            amdsmi_vram_usage_t vram_usage = {};
            ret = amdsmi_get_gpu_vram_usage(processor_handles[device_index], &vram_usage);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_vram_usage:\n");
            std::cout << "\t\tFrame buffer usage (MB): " << vram_usage.vram_used
                      << "/" << vram_usage.vram_total << "\n\n";

            amdsmi_power_cap_info_t cap_info = {};
            ret = amdsmi_get_power_cap_info(processor_handles[device_index], 0, &cap_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_power_cap_info:\n");
            std::cout << "\t\t Power Cap: " << cap_info.power_cap
                      << " uW\n";
            std::cout << "\t\t Default Power Cap: " << cap_info.default_power_cap
                      << " uW\n\n";
            std::cout << "\t\t Dpm Cap: " << cap_info.dpm_cap
                      << " MHz\n\n";
            std::cout << "\t\t Min Power Cap: " << cap_info.min_power_cap
                      << " uW\n\n";
            std::cout << "\t\t Max Power Cap: " << cap_info.max_power_cap
                      << " uW\n\n";

            /// Get GPU Metrics info
            std::cout << "\n\n";
            amdsmi_gpu_metrics_t smu;
            ret = amdsmi_get_gpu_metrics_info(processor_handles[device_index], &smu);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_metrics_info:\n");
            printf("\tDevice[%d] BDF %04" PRIx64 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32 "\n\n", i,
                   static_cast<uint64_t>(bdf.domain_number),
                   static_cast<uint32_t>(bdf.bus_number),
                   static_cast<uint32_t>(bdf.device_number),
                   static_cast<uint32_t>(bdf.function_number));

            std::cout << "METRIC TABLE HEADER:\n";
            std::cout << "structure_size=" << std::dec
            << static_cast<uint16_t>(smu.common_header.structure_size) << "\n";
            std::cout << "\tformat_revision=" << std::dec
            << static_cast<uint16_t>(smu.common_header.format_revision) << "\n";
            std::cout << "\tcontent_revision=" << std::dec
            << static_cast<uint16_t>(smu.common_header.content_revision) << "\n";

            std::cout << "\n";
            std::cout << "TIME STAMPS (ns):\n";
            std::cout << std::dec << "\tsystem_clock_counter=" << smu.system_clock_counter << "\n";
            std::cout << "\tfirmware_timestamp (10ns resolution)=" << std::dec << smu.firmware_timestamp
                      << "\n";

            std::cout << "\n";
            std::cout << "TEMPERATURES (C):\n";
            std::cout << std::dec << "\ttemperature_edge= " << smu.temperature_edge << "\n";
            std::cout << std::dec << "\ttemperature_hotspot= " << smu.temperature_hotspot << "\n";
            std::cout << std::dec << "\ttemperature_mem= " << smu.temperature_mem << "\n";
            std::cout << std::dec << "\ttemperature_vrgfx= " << smu.temperature_vrgfx << "\n";
            std::cout << std::dec << "\ttemperature_vrsoc= " << smu.temperature_vrsoc << "\n";
            std::cout << std::dec << "\ttemperature_vrmem= " << smu.temperature_vrmem << "\n";
            std::cout << "\ttemperature_hbm = [";
            auto idx = 0;
            for (const auto& temp : smu.temperature_hbm) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.temperature_hbm))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << "\n";
            std::cout << "UTILIZATION (%):\n";
            std::cout << std::dec << "\taverage_gfx_activity=" << smu.average_gfx_activity << "\n";
            std::cout << std::dec << "\taverage_umc_activity=" << smu.average_umc_activity << "\n";
            std::cout << std::dec << "\taverage_mm_activity=" << smu.average_mm_activity << "\n";
            std::cout << std::dec << "\tvcn_activity= [";
            idx = 0;
            for (const auto& temp : smu.vcn_activity) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.vcn_activity))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << "\n";
            std::cout << std::dec << "\tjpeg_activity= [";
            idx = 0;
            for (const auto& temp : smu.jpeg_activity) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.jpeg_activity))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << "\n";
            std::cout << "POWER (W)/ENERGY (15.259uJ per 1ns):\n";
            std::cout << std::dec << "\taverage_socket_power=" << smu.average_socket_power << "\n";
            std::cout << std::dec << "\tcurrent_socket_power=" << smu.current_socket_power << "\n";
            std::cout << std::dec << "\tenergy_accumulator=" << smu.energy_accumulator << "\n";

            std::cout << "\n";
            std::cout << "AVG CLOCKS (MHz):\n";
            std::cout << std::dec << "\taverage_gfxclk_frequency=" << smu.average_gfxclk_frequency
                      << "\n";
            std::cout << std::dec << "\taverage_gfxclk_frequency=" << smu.average_gfxclk_frequency
                      << "\n";
            std::cout << std::dec << "\taverage_uclk_frequency=" << smu.average_uclk_frequency << "\n";
            std::cout << std::dec << "\taverage_vclk0_frequency=" << smu.average_vclk0_frequency
                      << "\n";
            std::cout << std::dec << "\taverage_dclk0_frequency=" << smu.average_dclk0_frequency
                      << "\n";
            std::cout << std::dec << "\taverage_vclk1_frequency=" << smu.average_vclk1_frequency
                      << "\n";
            std::cout << std::dec << "\taverage_dclk1_frequency=" << smu.average_dclk1_frequency
                      << "\n";

            std::cout << "\n";
            std::cout << "CURRENT CLOCKS (MHz):\n";
            std::cout << std::dec << "\tcurrent_gfxclk=" << smu.current_gfxclk << "\n";
            std::cout << std::dec << "\tcurrent_gfxclks= [";
            idx = 0;
            for (const auto& temp : smu.current_gfxclks) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.current_gfxclks))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << std::dec << "\tcurrent_socclk=" << smu.current_socclk << "\n";
            std::cout << std::dec << "\tcurrent_socclks= [";
            idx = 0;
            for (const auto& temp : smu.current_socclks) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.current_socclks))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << std::dec << "\tcurrent_uclk=" << smu.current_uclk << "\n";
            std::cout << std::dec << "\tcurrent_vclk0=" << smu.current_vclk0 << "\n";
            std::cout << std::dec << "\tcurrent_vclk0s= [";
            idx = 0;
            for (const auto& temp : smu.current_vclk0s) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.current_vclk0s))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << std::dec << "\tcurrent_dclk0=" << smu.current_dclk0 << "\n";
            std::cout << std::dec << "\tcurrent_dclk0s= [";
            idx = 0;
            for (const auto& temp : smu.current_dclk0s) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.current_dclk0s))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << std::dec << "\tcurrent_vclk1=" << smu.current_vclk1 << "\n";
            std::cout << std::dec << "\tcurrent_dclk1=" << smu.current_dclk1 << "\n";

            std::cout << "\n";
            std::cout << "TROTTLE STATUS:\n";
            std::cout << std::dec << "\tthrottle_status=" << smu.throttle_status << "\n";

            std::cout << "\n";
            std::cout << "FAN SPEED:\n";
            std::cout << std::dec << "\tcurrent_fan_speed=" << smu.current_fan_speed << "\n";

            std::cout << "\n";
            std::cout << "LINK WIDTH (number of lanes) /SPEED (0.1 GT/s):\n";
            std::cout << "\tpcie_link_width=" << smu.pcie_link_width << "\n";
            std::cout << "\tpcie_link_speed=" << smu.pcie_link_speed << "\n";
            std::cout << "\txgmi_link_width=" << smu.xgmi_link_width << "\n";
            std::cout << "\txgmi_link_speed=" << smu.xgmi_link_speed << "\n";

            std::cout << "\n";
            std::cout << "Utilization Accumulated(%):\n";
            std::cout << "\tgfx_activity_acc=" << std::dec << smu.gfx_activity_acc << "\n";
            std::cout << "\tmem_activity_acc=" << std::dec << smu.mem_activity_acc  << "\n";

            std::cout << "\n";
            std::cout << "XGMI ACCUMULATED DATA TRANSFER SIZE (KB):\n";
            std::cout << std::dec << "\txgmi_read_data_acc= [";
            idx = 0;
            for (const auto& temp : smu.xgmi_read_data_acc) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.xgmi_read_data_acc))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << std::dec << "\txgmi_write_data_acc= [";
            idx = 0;
            for (const auto& temp : smu.xgmi_write_data_acc) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.xgmi_write_data_acc))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            std::cout << std::dec << "\txgmi_link_status= [";
            idx = 0;
            for (const auto& temp : smu.xgmi_link_status) {
              std::cout << temp;
              if ((idx + 1) != static_cast<int>(std::size(smu.xgmi_link_status))) {
                std::cout << ", ";
              } else {
                std::cout << "]\n";
              }
              ++idx;
            }

            // Voltage (mV)
            std::cout << "\tvoltage_soc = " << std::dec << smu.voltage_soc << "\n";
            std::cout << "\tvoltage_gfx = " << std::dec << smu.voltage_gfx << "\n";
            std::cout << "\tvoltage_mem = " << std::dec << smu.voltage_mem << "\n";

            std::cout << "\tindep_throttle_status = " << std::dec << smu.indep_throttle_status << "\n";

            // Clock Lock Status. Each bit corresponds to clock instance
            std::cout << "\tgfxclk_lock_status (in hex) = " << std::hex
                      << smu.gfxclk_lock_status << std::dec <<"\n";

            // Bandwidth (GB/sec)
            std::cout << "\tpcie_bandwidth_acc=" << std::dec << smu.pcie_bandwidth_acc << "\n";
            std::cout << "\tpcie_bandwidth_inst=" << std::dec << smu.pcie_bandwidth_inst << "\n";

            // VRAM max bandwidth at max memory clock
            std::cout << "\tvram_max_bandwidth=" << std::dec << smu.vram_max_bandwidth << "\n";

            // Counts
            std::cout << "\tpcie_l0_to_recov_count_acc= " << std::dec << smu.pcie_l0_to_recov_count_acc
                      << "\n";
            std::cout << "\tpcie_replay_count_acc= " << std::dec << smu.pcie_replay_count_acc << "\n";
            std::cout << "\tpcie_replay_rover_count_acc= " << std::dec
                      << smu.pcie_replay_rover_count_acc << "\n";
            std::cout << "\tpcie_nak_sent_count_acc= " << std::dec << smu.pcie_nak_sent_count_acc
                      << "\n";
            std::cout << "\tpcie_nak_rcvd_count_acc= " << std::dec << smu.pcie_nak_rcvd_count_acc
                      << "\n";

            // Accumulation cycle counter
            // Accumulated throttler residencies
            std::cout << "\n";
            std::cout << "RESIDENCY ACCUMULATION / COUNTER:\n";
            std::cout << "\taccumulation_counter = " << std::dec << smu.accumulation_counter << "\n";
            std::cout << "\tprochot_residency_acc = " << std::dec << smu.prochot_residency_acc << "\n";
            std::cout << "\tppt_residency_acc = " << std::dec << smu.ppt_residency_acc << "\n";
            std::cout << "\tsocket_thm_residency_acc = " << std::dec << smu.socket_thm_residency_acc
                      << "\n";
            std::cout << "\tvr_thm_residency_acc = " << std::dec << smu.vr_thm_residency_acc
                      << "\n";
            std::cout << "\thbm_thm_residency_acc = " << std::dec << smu.hbm_thm_residency_acc << "\n";

            // Number of current partitions
            std::cout << "\tnum_partition = " << std::dec << smu.num_partition << "\n";

            // PCIE other end recovery counter
            std::cout << "\tpcie_lc_perf_other_end_recovery = "
                      << std::dec << smu.pcie_lc_perf_other_end_recovery << "\n";

            idx = 0;
            auto idy = 0;
            std::cout  << "\txcp_stats.gfx_busy_inst: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.gfx_busy_inst) {
                if ((idy + 1) != static_cast<int>(std::size(row.gfx_busy_inst))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            idx = 0;
            idy = 0;
            std::cout  << "\txcp_stats.vcn_busy: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.vcn_busy) {
                if ((idy + 1) != static_cast<int>(std::size(row.vcn_busy))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            idx = 0;
            idy = 0;
            std::cout  << "\txcp_stats.jpeg_busy: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.jpeg_busy) {
                if ((idy + 1) != static_cast<int>(std::size(row.jpeg_busy))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            idx = 0;
            idy = 0;
            std::cout  << "\txcp_stats.gfx_busy_acc: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.gfx_busy_acc) {
                if ((idy + 1) != static_cast<int>(std::size(row.gfx_busy_acc))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            idx = 0;
            idy = 0;
            std::cout  << "\txcp_stats.gfx_below_host_limit_acc: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.gfx_below_host_limit_acc) {
                if ((idy + 1) != static_cast<int>(std::size(row.gfx_below_host_limit_acc))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            /*New scp stats v1.8*/
            idx = 0;
            idy = 0;
            std::cout  << "\txcp_stats.gfx_below_host_limit_ppt_acc: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.gfx_below_host_limit_ppt_acc) {
                if ((idy + 1) != static_cast<int>(std::size(row.gfx_below_host_limit_ppt_acc))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            idx = 0;
            idy = 0;
            std::cout  << "\txcp_stats.gfx_below_host_limit_thm_acc: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.gfx_below_host_limit_thm_acc) {
                if ((idy + 1) != static_cast<int>(std::size(row.gfx_below_host_limit_thm_acc))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            idx = 0;
            idy = 0;
            std::cout  << "\txcp_stats.gfx_low_utilization_acc: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.gfx_low_utilization_acc) {
                if ((idy + 1) != static_cast<int>(std::size(row.gfx_low_utilization_acc))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            idx = 0;
            idy = 0;
            std::cout  << "\txcp_stats.gfx_below_host_limit_total_acc: " << "\n";
            for (auto& row : smu.xcp_stats) {
              std::cout << "\t XCP [" << idx << "] : [";
              for (auto& col : row.gfx_below_host_limit_total_acc) {
                if ((idy + 1) != static_cast<int>(std::size(row.gfx_below_host_limit_total_acc))) {
                    std::cout << col << ", ";
                } else {
                    std::cout << col;
                }
                idy++;
              }
              std::cout << "]\n";
              idy = 0;
              idx++;
            }

            std::cout << "\n\n";
            std::cout << "\t ** -> Checking metrics with constant changes ** " << "\n";
            constexpr uint16_t kMAX_ITER_TEST = 10;
            amdsmi_gpu_metrics_t gpu_metrics_check = {};
            for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
                amdsmi_get_gpu_metrics_info(processor_handles[device_index], &gpu_metrics_check);
                std::cout << "\t\t -> firmware_timestamp [" << idx << "/" << kMAX_ITER_TEST << "]: "
                          << gpu_metrics_check.firmware_timestamp << "\n";
            }

            std::cout << "\n";
            for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
                amdsmi_get_gpu_metrics_info(processor_handles[device_index], &gpu_metrics_check);
                std::cout << "\t\t -> system_clock_counter [" << idx << "/" << kMAX_ITER_TEST << "]: "
                          << gpu_metrics_check.system_clock_counter << "\n";
            }

            std::cout << "\n";
            std::cout << " ** Note: Values MAX'ed out "
                      << "(UINTX MAX are unsupported for the version in question) ** " << "\n\n";

            // Get nearest GPUs
            char *topology_link_type_str[] = {
                "AMDSMI_LINK_TYPE_INTERNAL",
                "AMDSMI_LINK_TYPE_XGMI",
                "AMDSMI_LINK_TYPE_PCIE",
                "AMDSMI_LINK_TYPE_NOT_APPLICABLE",
                "AMDSMI_LINK_TYPE_UNKNOWN",
            };
            printf("\tOutput of amdsmi_get_link_topology_nearest:\n");
            for (uint32_t topo_link_type = AMDSMI_LINK_TYPE_INTERNAL; topo_link_type <= AMDSMI_LINK_TYPE_UNKNOWN; topo_link_type++) {
                auto topology_nearest_info = amdsmi_topology_nearest_t();
                ret = amdsmi_get_link_topology_nearest(processor_handles[device_index],
                                                       static_cast<amdsmi_link_type_t>(topo_link_type),
                                                       nullptr);
                if (ret != AMDSMI_STATUS_INVAL) {
                    CHK_AMDSMI_RET(ret);
                }

                ret = amdsmi_get_link_topology_nearest(processor_handles[device_index],
                                                       static_cast<amdsmi_link_type_t>(topo_link_type),
                                                       &topology_nearest_info);
                if (ret != AMDSMI_STATUS_INVAL) {
                    CHK_AMDSMI_RET(ret);
                }

                printf("\tNearest GPUs found at %s\n", topology_link_type_str[topo_link_type]);
                printf("\tNearest Count: %d\n", topology_nearest_info.count);
                for (uint32_t k = 0; k < topology_nearest_info.count; k++) {
                    amdsmi_bdf_t bdf = {};
                    ret = amdsmi_get_gpu_device_bdf(topology_nearest_info.processor_list[k], &bdf);
                    CHK_AMDSMI_RET(ret)
                    printf("\t\tGPU BDF %04" PRIx64 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32 "\n",
                        static_cast<uint64_t>(bdf.domain_number),
                        static_cast<uint32_t>(bdf.bus_number),
                        static_cast<uint32_t>(bdf.device_number),
                        static_cast<uint32_t>(bdf.function_number));
                }
            }
      }
    }

    // Clean up resources allocated at amdsmi_init. It will invalidate sockets
    // and devices pointers
    ret = amdsmi_shut_down();
    CHK_AMDSMI_RET(ret)

    return 0;
}
