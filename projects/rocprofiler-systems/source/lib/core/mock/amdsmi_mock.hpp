// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// Mock AMDSMI header for AINIC development
// This header provides a passthrough to the real AMDSMI when ROCm is enabled,
// plus AINIC types that may not be in the system AMDSMI library yet

#include <cstdint>

// When ROCm is enabled, include the real AMDSMI header first
#include <amd_smi/amdsmi.h>

// Then provide AINIC types if not already defined
#ifndef AMDSMI_PROCESSOR_TYPE_AMD_NIC

#    ifdef __cplusplus
extern "C"
{
#    endif

// AINIC initialization flags
#    define AMDSMI_INIT_AMD_NICS (1 << 2)

// AINIC processor type
#    define AMDSMI_PROCESSOR_TYPE_AMD_NIC ((processor_type_t) 4)

// AINIC constants
#    ifndef AMDSMI_MAX_STRING_LENGTH
#        define AMDSMI_MAX_STRING_LENGTH 64
#    endif
#    define AMDSMI_MAX_NUM_RDMA_DEVS     8
#    define AMDSMI_MAX_NUM_RDMA_PORTS    2
#    define AMDSMI_NIC_PORT_NAME_LEN     16
#    define AMDSMI_NIC_RDMA_DEV_NAME_LEN 16

    // NIC driver information
    typedef struct
    {
        char driver_name[AMDSMI_MAX_STRING_LENGTH];
        char driver_version[AMDSMI_MAX_STRING_LENGTH];
    } amdsmi_nic_driver_info_t;

    // NIC ASIC information
    typedef struct
    {
        uint16_t vendor_id;
        char     vendor_name[AMDSMI_MAX_STRING_LENGTH];
        char     product_name[AMDSMI_MAX_STRING_LENGTH];
        char     part_number[AMDSMI_MAX_STRING_LENGTH];
        char     serial_number[AMDSMI_MAX_STRING_LENGTH];
    } amdsmi_nic_asic_info_t;

    // NIC bus information
    typedef struct
    {
        uint32_t pcie_generation;
        uint32_t pcie_lanes;
        uint32_t link_speed_mbps;
    } amdsmi_nic_bus_info_t;

    // NIC NUMA information
    typedef struct
    {
        uint32_t numa_node;
    } amdsmi_nic_numa_info_t;

    // NIC port information
    typedef struct
    {
        uint8_t num_ports;
        struct
        {
            char     netdev[AMDSMI_NIC_PORT_NAME_LEN];
            uint8_t  port_num;
            uint8_t  port_state;           // 0 = DOWN, 1 = UP
            uint8_t  physical_link_state;  // 0 = DOWN, 1 = UP
            uint8_t  mac_addr[6];
            uint32_t link_speed_mbps;
            uint8_t  roce_enabled;
        } ports[AMDSMI_MAX_NUM_RDMA_PORTS];
    } amdsmi_nic_port_info_t;

    // NIC RDMA port information
    typedef struct
    {
        uint8_t rdma_port_num;
        char    netdev[AMDSMI_NIC_PORT_NAME_LEN];
    } amdsmi_nic_rdma_port_info_t;

    // NIC RDMA device information
    typedef struct
    {
        char                        rdma_dev[AMDSMI_NIC_RDMA_DEV_NAME_LEN];
        uint8_t                     num_rdma_ports;
        amdsmi_nic_rdma_port_info_t rdma_port_info[AMDSMI_MAX_NUM_RDMA_PORTS];
    } amdsmi_nic_rdma_dev_info_t;

    typedef struct
    {
        uint8_t                    num_rdma_dev;
        amdsmi_nic_rdma_dev_info_t rdma_dev_info[AMDSMI_MAX_NUM_RDMA_DEVS];
    } amdsmi_nic_rdma_devices_info_t;

    // NIC statistics
    typedef struct
    {
        char     name[AMDSMI_MAX_STRING_LENGTH];
        uint64_t value;
    } amdsmi_nic_stat_t;

    // AINIC API function declarations
    amdsmi_status_t amdsmi_get_nic_driver_info(amdsmi_processor_handle   processor_handle,
                                               amdsmi_nic_driver_info_t* info);
    amdsmi_status_t amdsmi_get_nic_asic_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_nic_asic_info_t* info);
    amdsmi_status_t amdsmi_get_nic_bus_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_nic_bus_info_t*  info);
    amdsmi_status_t amdsmi_get_nic_numa_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_nic_numa_info_t* info);
    amdsmi_status_t amdsmi_get_nic_port_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_nic_port_info_t* info);
    amdsmi_status_t amdsmi_get_nic_rdma_dev_info(amdsmi_processor_handle processor_handle,
                                                 amdsmi_nic_rdma_devices_info_t* info);
    amdsmi_status_t amdsmi_get_nic_rdma_port_statistics(
        amdsmi_processor_handle processor_handle, uint32_t rdma_port_index,
        uint32_t* num_stats, amdsmi_nic_stat_t* stats);

    // Mock-specific function to get AINIC processor handles when ROCm is enabled
    // Use this instead of amdsmi_get_processor_handles_by_type for AINIC discovery
    amdsmi_status_t rocprofsys_mock_ainic_get_processor_handles(
        amdsmi_socket_handle socket_handle, amdsmi_processor_handle* processor_handles,
        uint32_t* processor_count);

#    ifdef __cplusplus
}
#    endif

#endif  // AMDSMI_PROCESSOR_TYPE_AMD_NIC
