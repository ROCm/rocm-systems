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

#include "amdsmi_mock.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace
{
// Mock state for AINIC simulation
struct MockAINICState
{
    uint32_t   num_nics             = 2;   // Default 2 mock NICs
    uint64_t   stat_counters[8][32] = {};  // [nic_index][stat_index]
    std::mutex mutex;
    bool       initialized = false;
};

MockAINICState&
get_ainic_state()
{
    static MockAINICState state;
    return state;
}

// Initialize mock AINIC state from environment variable
void
ensure_ainic_initialized()
{
    auto& state = get_ainic_state();
    if(!state.initialized)
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if(!state.initialized)  // Double-check after acquiring lock
        {
            const char* mock_count_env = std::getenv("ROCPROFSYS_MOCK_AINIC_COUNT");
            if(mock_count_env != nullptr)
            {
                int count = std::atoi(mock_count_env);
                if(count >= 0 && count <= 8)
                {
                    state.num_nics = static_cast<uint32_t>(count);
                }
            }
            state.initialized = true;
        }
    }
}

// Create mock processor handle from index
amdsmi_processor_handle
make_ainic_handle(uint32_t index)
{
    // Use a base address that's unlikely to conflict with real handles
    return reinterpret_cast<void*>(static_cast<uintptr_t>(0xA1C00000 + index));
}

// Extract index from mock processor handle
uint32_t
get_ainic_index(amdsmi_processor_handle handle)
{
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(handle) - 0xA1C00000);
}

}  // namespace

extern "C"
{
    // Mock AINIC processor handle discovery
    // Use this instead of amdsmi_get_processor_handles_by_type for AINIC
    amdsmi_status_t rocprofsys_mock_ainic_get_processor_handles(
        amdsmi_socket_handle socket_handle, amdsmi_processor_handle* processor_handles,
        uint32_t* processor_count)
    {
        (void) socket_handle;  // Unused in mock
        ensure_ainic_initialized();
        auto&                       state = get_ainic_state();
        std::lock_guard<std::mutex> lock(state.mutex);

        if(processor_count == nullptr)
        {
            return AMDSMI_STATUS_INVAL;
        }

        if(processor_handles == nullptr)
        {
            // First call: return count
            *processor_count = state.num_nics;
            return AMDSMI_STATUS_SUCCESS;
        }

        // Second call: fill handles
        for(uint32_t i = 0; i < state.num_nics; ++i)
        {
            processor_handles[i] = make_ainic_handle(i);
        }
        *processor_count = state.num_nics;

        return AMDSMI_STATUS_SUCCESS;
    }

    // Mock AINIC driver info
    amdsmi_status_t amdsmi_get_nic_driver_info(amdsmi_processor_handle   processor_handle,
                                               amdsmi_nic_driver_info_t* info)
    {
        (void) processor_handle;  // Unused in simple mock
        ensure_ainic_initialized();

        if(info == nullptr)
        {
            return AMDSMI_STATUS_INVAL;
        }

        std::strncpy(info->driver_name, "ionic", AMDSMI_MAX_STRING_LENGTH - 1);
        info->driver_name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        std::strncpy(info->driver_version, "v25.08.2.001", AMDSMI_MAX_STRING_LENGTH - 1);
        info->driver_version[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';

        return AMDSMI_STATUS_SUCCESS;
    }

    // Mock AINIC ASIC info
    amdsmi_status_t amdsmi_get_nic_asic_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_nic_asic_info_t* info)
    {
        (void) processor_handle;  // Unused in simple mock
        ensure_ainic_initialized();

        if(info == nullptr)
        {
            return AMDSMI_STATUS_INVAL;
        }

        info->vendor_id = 0x1dd8;  // AMD Pensando Systems
        std::strncpy(info->vendor_name, "AMD Pensando Systems",
                     AMDSMI_MAX_STRING_LENGTH - 1);
        info->vendor_name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        std::strncpy(info->product_name, "Pensando DSC2-200 50/100/200G 2p QSFP56 Card",
                     AMDSMI_MAX_STRING_LENGTH - 1);
        info->product_name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        std::strncpy(info->part_number, "DSC2-2Q200-32R32F64P-R4",
                     AMDSMI_MAX_STRING_LENGTH - 1);
        info->part_number[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        std::strncpy(info->serial_number, "FPF2316002EEC0V2",
                     AMDSMI_MAX_STRING_LENGTH - 1);
        info->serial_number[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';

        return AMDSMI_STATUS_SUCCESS;
    }

    // Mock AINIC bus info
    amdsmi_status_t amdsmi_get_nic_bus_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_nic_bus_info_t*  info)
    {
        (void) processor_handle;  // Unused in simple mock
        ensure_ainic_initialized();

        if(info == nullptr)
        {
            return AMDSMI_STATUS_INVAL;
        }

        info->pcie_generation = 5;
        info->pcie_lanes      = 16;
        info->link_speed_mbps = 32000;  // 32 GT/s

        return AMDSMI_STATUS_SUCCESS;
    }

    // Mock AINIC NUMA info
    amdsmi_status_t amdsmi_get_nic_numa_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_nic_numa_info_t* info)
    {
        (void) processor_handle;  // Unused in simple mock
        ensure_ainic_initialized();

        if(info == nullptr)
        {
            return AMDSMI_STATUS_INVAL;
        }

        info->numa_node = 1;

        return AMDSMI_STATUS_SUCCESS;
    }

    // Mock AINIC port info
    amdsmi_status_t amdsmi_get_nic_port_info(amdsmi_processor_handle processor_handle,
                                             amdsmi_nic_port_info_t* info)
    {
        ensure_ainic_initialized();

        if(info == nullptr)
        {
            return AMDSMI_STATUS_INVAL;
        }

        uint32_t nic_index = get_ainic_index(processor_handle);

        // Each NIC has 2 ports
        info->num_ports = 2;

        for(uint8_t i = 0; i < 2; ++i)
        {
            // Device names: enp226s0, enp226s1, enp227s0, enp227s1, etc.
            snprintf(info->ports[i].netdev, AMDSMI_NIC_PORT_NAME_LEN, "enp%us%u",
                     226 + nic_index, i);

            info->ports[i].port_num = i;

            // First port UP, second port DOWN for variety
            if(i == 0)
            {
                info->ports[i].port_state          = 1;       // UP
                info->ports[i].physical_link_state = 1;       // UP
                info->ports[i].link_speed_mbps     = 100000;  // 100 Gbps
                info->ports[i].roce_enabled        = 1;
            }
            else
            {
                info->ports[i].port_state          = 0;  // DOWN
                info->ports[i].physical_link_state = 0;  // DOWN
                info->ports[i].link_speed_mbps     = 0;
                info->ports[i].roce_enabled        = 0;
            }

            // MAC addresses: 04:90:81:01:09:38, 04:90:81:01:09:39, etc.
            info->ports[i].mac_addr[0] = 0x04;
            info->ports[i].mac_addr[1] = 0x90;
            info->ports[i].mac_addr[2] = 0x81;
            info->ports[i].mac_addr[3] = 0x01;
            info->ports[i].mac_addr[4] = 0x09;
            info->ports[i].mac_addr[5] = 0x38 + (nic_index * 2) + i;
        }

        return AMDSMI_STATUS_SUCCESS;
    }

    // Mock AINIC RDMA device info
    amdsmi_status_t amdsmi_get_nic_rdma_dev_info(amdsmi_processor_handle processor_handle,
                                                 amdsmi_nic_rdma_devices_info_t* info)
    {
        ensure_ainic_initialized();

        if(info == nullptr)
        {
            return AMDSMI_STATUS_INVAL;
        }

        uint32_t nic_index = get_ainic_index(processor_handle);

        // One RDMA device per NIC
        info->num_rdma_dev = 1;

        // RDMA device name: rdma0, rdma1, etc.
        snprintf(info->rdma_dev_info[0].rdma_dev, AMDSMI_NIC_RDMA_DEV_NAME_LEN, "rdma%u",
                 nic_index);

        // 2 RDMA ports per device
        info->rdma_dev_info[0].num_rdma_ports = 2;

        for(uint8_t i = 0; i < 2; ++i)
        {
            info->rdma_dev_info[0].rdma_port_info[i].rdma_port_num = i;
            snprintf(info->rdma_dev_info[0].rdma_port_info[i].netdev,
                     AMDSMI_NIC_PORT_NAME_LEN, "enp%us%u", 226 + nic_index, i);
        }

        return AMDSMI_STATUS_SUCCESS;
    }

    // Mock AINIC RDMA port statistics
    amdsmi_status_t amdsmi_get_nic_rdma_port_statistics(
        amdsmi_processor_handle processor_handle, uint32_t rdma_port_index,
        uint32_t* num_stats, amdsmi_nic_stat_t* stats)
    {
        ensure_ainic_initialized();
        auto&                       state = get_ainic_state();
        std::lock_guard<std::mutex> lock(state.mutex);

        if(num_stats == nullptr)
        {
            return AMDSMI_STATUS_INVAL;
        }

        uint32_t nic_index = get_ainic_index(processor_handle);

        if(stats == nullptr)
        {
            // First call: return count
            *num_stats = 6;
            return AMDSMI_STATUS_SUCCESS;
        }

        // Second call: fill statistics
        // Increment counters to simulate activity (only for port 0 which is UP)
        if(rdma_port_index == 0)
        {
            for(int i = 0; i < 6; ++i)
            {
                state.stat_counters[nic_index][i] += 1000 * (i + 1);
            }
        }

        std::strncpy(stats[0].name, "rx_rdma_ucast_bytes", AMDSMI_MAX_STRING_LENGTH - 1);
        stats[0].name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        stats[0].value                              = state.stat_counters[nic_index][0];

        std::strncpy(stats[1].name, "rx_rdma_ucast_pkts", AMDSMI_MAX_STRING_LENGTH - 1);
        stats[1].name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        stats[1].value                              = state.stat_counters[nic_index][1];

        std::strncpy(stats[2].name, "tx_rdma_ucast_bytes", AMDSMI_MAX_STRING_LENGTH - 1);
        stats[2].name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        stats[2].value                              = state.stat_counters[nic_index][2];

        std::strncpy(stats[3].name, "tx_rdma_ucast_pkts", AMDSMI_MAX_STRING_LENGTH - 1);
        stats[3].name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        stats[3].value                              = state.stat_counters[nic_index][3];

        std::strncpy(stats[4].name, "rx_rdma_cnp_pkts", AMDSMI_MAX_STRING_LENGTH - 1);
        stats[4].name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        stats[4].value                              = state.stat_counters[nic_index][4];

        std::strncpy(stats[5].name, "tx_rdma_cnp_pkts", AMDSMI_MAX_STRING_LENGTH - 1);
        stats[5].name[AMDSMI_MAX_STRING_LENGTH - 1] = '\0';
        stats[5].value                              = state.stat_counters[nic_index][5];

        return AMDSMI_STATUS_SUCCESS;
    }

}  // extern "C"
