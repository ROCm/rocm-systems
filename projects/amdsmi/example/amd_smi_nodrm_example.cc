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
#include <cinttypes>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "amd_smi/amdsmi.h"

#define CHK_AMDSMI_RET(RET)                                                    \
    {                                                                          \
        if (RET != AMDSMI_STATUS_SUCCESS) {                                    \
            const char *err_str;                                               \
            amdsmi_status_code_to_string(RET, &err_str);                       \
            std::cout << "AMDSMI call returned " << RET << " at line "         \
                      << __LINE__ << ": " << err_str << std::endl;             \
            if (RET != AMDSMI_STATUS_NOT_SUPPORTED && RET != AMDSMI_STATUS_INVAL) { \
                return RET;                                                    \
            }                                                                  \
        }                                                                      \
    }

int main() {
    amdsmi_status_t ret;

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

        // For each device of the socket, get name and temperature.
        for (uint32_t j = 0; j < device_count; j++) {
            // Get device type. Since the amdsmi is initialized with
            // AMD_SMI_INIT_AMD_GPUS, the processor_type must be AMDSMI_PROCESSOR_TYPE_AMD_GPU.
            processor_type_t processor_type = {};
            ret = amdsmi_get_processor_type(processor_handles[j], &processor_type);
            CHK_AMDSMI_RET(ret)
            if (processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
                std::cout << "Expect AMDSMI_PROCESSOR_TYPE_AMD_GPU device type!\n";
                return AMDSMI_STATUS_NOT_SUPPORTED;
            }

            amdsmi_ras_feature_t ras_feature;
            ret = amdsmi_get_gpu_ras_feature_info(
                processor_handles[j] ,&ras_feature);
            CHK_AMDSMI_RET(ret)
            if (ret != AMDSMI_STATUS_NOT_SUPPORTED) {
                printf("\tras_feature: version: %x, schema: %x\n",
                       ras_feature.ras_eeprom_version, ras_feature.ecc_correction_schema_flag);
            }


            amdsmi_bdf_t bdf = {};
            ret = amdsmi_get_gpu_device_bdf(processor_handles[j], &bdf);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_device_bdf:\n");
            printf("\tDevice[%d] BDF %04" PRIx64 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32 "\n\n", i,
                   static_cast<uint64_t>(bdf.domain_number),
                   static_cast<uint32_t>(bdf.bus_number),
                   static_cast<uint32_t>(bdf.device_number),
                   static_cast<uint32_t>(bdf.function_number));

            amdsmi_asic_info_t asic_info = {};
            ret = amdsmi_get_gpu_asic_info(processor_handles[j], &asic_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_asic_info:\n");
            printf("\tMarket Name: %s\n", asic_info.market_name);
            printf("\tDeviceID: 0x%lx\n", asic_info.device_id);
            printf("\tVendorID: 0x%x\n", asic_info.vendor_id);
            printf("\tRevisionID: 0x%x\n", asic_info.rev_id);
            printf("\tSubSystemID: 0x%x\n", asic_info.subsystem_id);
            printf("\tAsic serial: 0x%s\n", asic_info.asic_serial);
            printf("\tOAM id: 0x%x\n", asic_info.oam_id);
            printf("\tNum of Computes: %d\n\n", asic_info.num_of_compute_units);

            // Get VBIOS info
            amdsmi_vbios_info_t vbios_info = {};
            ret = amdsmi_get_gpu_vbios_info(processor_handles[j], &vbios_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_vbios_info:\n");
            printf("\tVBIOS/IFWI Name: %s\n", vbios_info.name);
            printf("\tVBIOS/IFWI Build Date: %s\n", vbios_info.build_date);
            printf("\tVBIOS/IFWI Part Number: %s\n", vbios_info.part_number);
            printf("\tVBIOS/IFWI Version String: %s\n\n", vbios_info.version);
            printf("\tVBIOS/IFWI Boot Firmware: %s\n\n", vbios_info.boot_firmware);

            // Get engine usage info
            amdsmi_engine_usage_t engine_usage = {};
            ret = amdsmi_get_gpu_activity(processor_handles[j], &engine_usage);
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
            ret = amdsmi_get_fw_info(processor_handles[j], &fw_information);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_fw_info:\n");
            printf("\tFirmware version: %d\n", fw_information.num_fw_info);
            printf("\tSMU: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_SMU]
                       .fw_version);
            printf("\tPM: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_PM]
                       .fw_version);
            printf("\tVCN: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_VCN]
                       .fw_version);
            printf("\tCP_ME: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_CP_ME]
                       .fw_version);
            printf("\tCP_PFP: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_CP_PFP]
                       .fw_version);
            printf("\tCP_CE: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_CP_CE]
                       .fw_version);
            printf("\tRLC: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_RLC]
                       .fw_version);
            printf("\tCP_MEC1: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_CP_MEC1]
                       .fw_version);
            printf("\tCP_MEC2: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_CP_MEC2]
                       .fw_version);
            printf("\tSDMA0: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_SDMA0]
                       .fw_version);
            printf("\tMC: %ld\n",
                   fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_MC]
                       .fw_version);
            printf("\tRLC RESTORE LIST CNTL: %ld\n",
                   fw_information
                       .fw_info_list
                           [amdsmi_fw_block_t::AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL]
                       .fw_version);
            printf("\tRLC RESTORE LIST GPM MEM: %ld\n",
                   fw_information
                       .fw_info_list
                           [amdsmi_fw_block_t::AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM]
                       .fw_version);
            printf("\tRLC RESTORE LIST SRM MEM: %ld\n",
                   fw_information
                       .fw_info_list
                           [amdsmi_fw_block_t::AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM]
                       .fw_version);
            printf(
                "\tPSP SOSDRV: %ld\n\n",
                fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_PSP_SOSDRV]
                    .fw_version);
            printf(
                "\tPLDM BUNDLE: %ld\n\n",
                fw_information.fw_info_list[amdsmi_fw_block_t::AMDSMI_FW_ID_PLDM_BUNDLE]
                    .fw_version);

            // Get temperature measurements
            int64_t temp_measurements[AMDSMI_TEMPERATURE_TYPE__MAX + 1];
            amdsmi_temperature_type_t temp_types[4] = {
                AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                AMDSMI_TEMPERATURE_TYPE_VRAM, AMDSMI_TEMPERATURE_TYPE_PLX};
            for (const auto &temp_type : temp_types) {
                ret = amdsmi_get_temp_metric(
                    processor_handles[j], temp_type,
                    AMDSMI_TEMP_CURRENT,
                    &temp_measurements[(int)(temp_type)]);
                CHK_AMDSMI_RET(ret)
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

            // Get bad pages
            char bad_page_status_names[3][15] = {"RESERVED", "PENDING",
                                                 "UNRESERVABLE"};
            uint32_t num_pages = 0;
            std::vector<amdsmi_retired_page_record_t> bad_page_info(num_pages);
            ret = amdsmi_get_gpu_bad_page_info(processor_handles[j], &num_pages,
                                           bad_page_info.data());
            std::cout << "num_pages = " << num_pages << "\n";
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_bad_page_info:\n");
            if (!num_pages) {
                printf("\tNo bad pages found.\n");
            } else {
                std::vector<amdsmi_retired_page_record_t> bad_page_info(num_pages);
                ret = amdsmi_get_gpu_bad_page_info(processor_handles[j], &num_pages,
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
            ret = amdsmi_get_gpu_total_ecc_count(processor_handles[j], &err_cnt_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_total_ecc_count:\n");
            printf("\tCorrectable errors: %lu\n", err_cnt_info.correctable_count);
            printf("\tUncorrectable errors: %lu\n\n",
                   err_cnt_info.uncorrectable_count);

            // Get device name
            amdsmi_board_info_t board_info = {};
            ret = amdsmi_get_gpu_board_info(processor_handles[j], &board_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_board_info:\n");
            std::cout << "\tdevice [" << j
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
            ret =  amdsmi_get_temp_metric(processor_handles[j], AMDSMI_TEMPERATURE_TYPE_EDGE,
                                             AMDSMI_TEMP_CURRENT, &val_i64);
            CHK_AMDSMI_RET(ret)
            printf("    Output of  amdsmi_get_temp_metric:\n");
            std::cout << "\t\tTemperature: " << val_i64 << "C"
                      << "\n\n";

            // Get frame buffer
            amdsmi_vram_usage_t vram_usage = {};
            ret = amdsmi_get_gpu_vram_usage(processor_handles[j], &vram_usage);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_gpu_vram_usage:\n");
            std::cout << "\t\tFrame buffer usage (MB): " << vram_usage.vram_used
                      << "/" << vram_usage.vram_total << "\n\n";

            amdsmi_power_cap_info_t cap_info = {};
            ret = amdsmi_get_power_cap_info(processor_handles[j], 0, &cap_info);
            CHK_AMDSMI_RET(ret)
            printf("    Output of amdsmi_get_power_cap_info:\n");
            std::cout << "\t\t Power Cap: " << cap_info.power_cap / 1000000
                      << "W\n\n";

            amdsmi_dpm_policy_t policy;
            ret = amdsmi_get_soc_pstate(processor_handles[j], &policy);
            if (ret != AMDSMI_STATUS_NOT_SUPPORTED) {
                CHK_AMDSMI_RET(ret)
                std::cout << "\t amdsmi_get_soc_pstate total:" << policy.num_supported
                        <<" current:" << policy.current << "\n";
                for (uint32_t x=0; x < policy.num_supported; x++) {
                    std::cout << x <<": (" << policy.policies[x].policy_id
                    <<"," << policy.policies[x].policy_description << ")\n";
                }
            }

            // Get nearest GPUs
            const char *topology_link_type_str[] = {
                "AMDSMI_LINK_TYPE_INTERNAL",
                "AMDSMI_LINK_TYPE_PCIE",
                "AMDSMI_LINK_TYPE_XGMI",
                "AMDSMI_LINK_TYPE_NOT_APPLICABLE",
                "AMDSMI_LINK_TYPE_UNKNOWN",
            };
            printf("\tOutput of amdsmi_get_link_topology_nearest:\n");
            for (uint32_t topo_link_type = AMDSMI_LINK_TYPE_INTERNAL; topo_link_type <= AMDSMI_LINK_TYPE_UNKNOWN; topo_link_type++) {
                auto topology_nearest_info = amdsmi_topology_nearest_t();
                ret = amdsmi_get_link_topology_nearest(processor_handles[j],
                                                       static_cast<amdsmi_link_type_t>(topo_link_type),
                                                       nullptr);
                CHK_AMDSMI_RET(ret);
                ret = amdsmi_get_link_topology_nearest(processor_handles[j],
                                                       static_cast<amdsmi_link_type_t>(topo_link_type),
                                                       &topology_nearest_info);
                CHK_AMDSMI_RET(ret);
                printf("\tNearest GPUs found at %s\n", topology_link_type_str[topo_link_type]);
                for (uint32_t k = 0; k < topology_nearest_info.count; k++) {
                    amdsmi_bdf_t bdf = {};
                    ret = amdsmi_get_gpu_device_bdf(topology_nearest_info.processor_list[k], &bdf);
                    CHK_AMDSMI_RET(ret)
                    printf("\tGPU BDF %04" PRIx64 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32 "\n",
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
