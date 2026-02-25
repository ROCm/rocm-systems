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
#include <iomanip>
#include <functional>
#include <optional>

#include <amd_smi/amdsmi.h>
#include <amd_smi/impl/amd_smi_system.h>
#include <amd_smi/impl/amd_smi_utils.h>
#include <amd_smi/impl/amd_smi_utils.h>
#include <amd_smi/impl/nic/amd_smi_ainic_device.h>

namespace {
struct RAII {
    RAII(std::function<void()> init, std::function<void()> finish) 
    : _finish(finish) {
        init();
    }
    ~RAII() {
        _finish();
    }
    std::function<void()> _finish;
};


void dump_ainic_info(int idx, const amd::smi::AMDSmiAINICDevice::AINICInfo &ainic_info) {
#if 0//Expected Output:
NIC: 0
   ASIC:
       VENDOR_ID: 0x1dd8
       SUBVENDOR_ID: 0x1dd8
       DEVICE_ID: 0x8
       SUBSYSTEM_ID: 0x5201
       REVISION: 0x0
       PERMANENT_ADDRESS: 04:90:81:01:09:38
       PRODUCT_NAME: ,Pensando DSC2-200 50/100/200G 2p QSFP56 Card
       PART_NUMBER: DSC2-2Q200-32R32F64P-R4
       SERIAL_NUMBER: FPF2316002EEC0V2
       VENDOR_NAME: AMD Pensando Systems, Inc.
   BUS:
      BDF: 0000:e2:00.0
      MAX_PCIE_WIDTH: 16
      MAX_PCIE_SPEED: 32
      PCIE_INTERFACE_VERSION: N/A
      SLOT_TYPE: N/A
   DRIVER:
      NAME: ionic
      VERSION: 25.08.2.001
      FW_VERSION: 1.97.0-A-5
      BUS_INFO: 0000:83:00.0
   NUMA:
      NODE: 1
      AFFINITY: 16-31,48-63
   VERSIONS:
      RUNNING:
          fw: ??
          fw.heartbeat: ??
          fw.status: ??
   PORTS:
      PORT_0:
            BDF: 0
            TYPE: Ethernet
            FLAVOUR: N/A
            NETDEV: enp131s0
            IFINDEX: 11
            MAC_ADDRESS: 04:90:81:01:09:38
            CARRIER: 0
            MTU: 1500B
            LINK_STATE: down
            LINK_SPEED: 0 Mb/s
            ACTIVE_FEC: Off
            AUTONEG: off
            PAUSE_AUTONEG: off
            PAUSE_RX: on
            PAUSE_TX: on
      PORT_1:
            BDF: 0
            TYPE: Ethernet
            FLAVOUR: N/A
            NETDEV: enp132s0
            IFINDEX: 12
            MAC_ADDRESS: 04:90:81:01:09:39
            CARRIER: 0
            MTU: 1500B
            LINK_STATE: down
            LINK_SPEED: 0 Mb/s
            ACTIVE_FEC: Off
            AUTONEG: off
            PAUSE_AUTONEG: off
            PAUSE_RX: on
            PAUSE_TX: on
      PORT_2:
            BDF: 0
            TYPE: Ethernet
            FLAVOUR: N/A
            NETDEV: enp229s0
            IFINDEX: 14
            MAC_ADDRESS: 04:90:81:2c:77:b0
            CARRIER: 0
            MTU: 1500B
            LINK_STATE: down
            LINK_SPEED: 0 Mb/s
            ACTIVE_FEC: Off
            AUTONEG: off
            PAUSE_AUTONEG: off
            PAUSE_RX: off
            PAUSE_TX: off
   RDMA_DEVICES:
      RDMA_DEVICES:
            RDMA_DEVICE_0:
               NAME: rocep131s0
               NODE_GUID: 0690:81ff:fe01:0938
               NODE_TYPE: 1: CA
               SYS_IMAGE_GUID: 0690:81ff:fe01:0938
               FW_VER: 1.97.0-A-5
               PORTS:
                  PORT_0:
                     NETDEV:
                     PORT_NUM: 1
                     STATE: DOWN
                     MAX_MTU: 65535
                     ACTIVE_MTU: 65535
      RDMA_DEVICES:
            RDMA_DEVICE_0:
               NAME: rocep132s0
               NODE_GUID: 0690:81ff:fe01:0939
               NODE_TYPE: 1: CA
               SYS_IMAGE_GUID: 0690:81ff:fe01:0939
               FW_VER: 1.97.0-A-5
               PORTS:
                  PORT_0:
                     NETDEV:
                     PORT_NUM: 1
                     STATE: DOWN
                     MAX_MTU: 65535
                     ACTIVE_MTU: 65535
      RDMA_DEVICES:
            RDMA_DEVICE_0:
               NAME: rocep229s0
               NODE_GUID: 0690:81ff:fe2c:77b0
               NODE_TYPE: 1: CA
               SYS_IMAGE_GUID: 0690:81ff:fe2c:77b0
               FW_VER: 1.110.1-a-1
               PORTS:
                  PORT_0:
                     NETDEV:
                     PORT_NUM: 1
                     STATE: DOWN
                     MAX_MTU: 65535
                     ACTIVE_MTU: 65535
#endif//0
    std::ostringstream oss;
    oss << std::hex << 
        std::setw(4) << std::setfill('0') << ainic_info.bus.bdf.domain_number << ":" << 
        std::setw(2) << std::setfill('0') << ainic_info.bus.bdf.bus_number << ":" << 
        std::setw(2) << std::setfill('0') << ainic_info.bus.bdf.device_number << "." << 
        std::setw(2) << std::setfill('0') << ainic_info.bus.bdf.function_number;
    std::string bdf_str = oss.str();
    oss.str("");
    oss << "===============================================\n" <<
    "NIC: " << idx << "\n" <<
    "   ASIC:" << "\n" <<
    "       VENDOR_ID: 0x" << std::hex << ainic_info.asic.vendor_id << std::dec << "\n" <<
    "       SUBVENDOR_ID: 0x" << std::hex << ainic_info.asic.subvendor_id << std::dec << "\n" <<
    "       DEVICE_ID: 0x" << std::hex << ainic_info.asic.device_id << std::dec << "\n" <<
    "       SUBSYSTEM_ID: 0x" << std::hex << ainic_info.asic.subsystem_id << std::dec << "\n" <<
    "       REVISION: 0x" << std::hex << static_cast<int>(ainic_info.asic.revision) << std::dec << "\n" <<
    "       PERMANENT_ADDRESS: " << ainic_info.asic.permanent_address << "\n" <<
    "       PRODUCT_NAME: " << ainic_info.asic.product_name << "\n" <<
    "       PART_NUMBER: " << ainic_info.asic.part_number << "\n" <<
    "       SERIAL_NUMBER: " << ainic_info.asic.serial_number << "\n" <<
    "       VENDOR_NAME: " << ainic_info.asic.vendor_name << "\n" <<
    "   BUS:" << "\n" <<
    "      BDF: " << bdf_str << "\n" <<
    "      MAX_PCIE_WIDTH: " << static_cast<int>(ainic_info.bus.max_pcie_width) << "\n" <<
    "      MAX_PCIE_SPEED: " << ainic_info.bus.max_pcie_speed << "\n" <<
    "      PCIE_INTERFACE_VERSION: " << ainic_info.bus.pcie_interface_version << "\n" <<
    "      SLOT_TYPE: " << ainic_info.bus.slot_type << "\n" <<
    "   DRIVER:" << "\n" <<
    "      NAME: " << ainic_info.driver.name << "\n" <<
    "      VERSION: " << ainic_info.driver.version << "\n" <<
    // "      FW_VERSION: " << ainic_info.driver.version << "\n" <<
    // "      BUS_INFO: " << ainic_info.driver.bus_info << "\n" <<
    "   NUMA:" << "\n" <<
    "      NODE: " << static_cast<int>(ainic_info.numa.node) << "\n" <<
    "      AFFINITY: " << ainic_info.numa.affinity << "\n" <<
    // "   LIMIT:" << "\n" << //can be fetched through https://www.kernel.org/doc/Documentation/hwmon/sysfs-interface or https://github.com/lm-sensors/lm-sensors
    // "      MAX_POWER: ?? W" << "\n" <<
    // "      MAX_TEMPERATURE: ?? C" << "\n" <<
    //     "   VERSIONS:" << "\n" << //these can be obtained only through netlink interface which we currently don't have.            It requires 3rd party library dependency or manually netlink hdr/msg construction which is hard to write and maintain/test. So, netlink interface implementation is postponed.
    // "      RUNNING:" << "\n" <<
    //     "          fw: ??" << ainic_info.versions.running.fw << "\n" <<
    //     "          fw.heartbeat: ??" << ainic_info.versions.running.fw_heartbeat << "\n" <<
    //     "          fw.status: ??" << ainic_info.versions.running.fw_status << "\n"
    "   PORTS:" << "\n";
    for(int port_idx = 0; port_idx < ainic_info.port.num_ports; ++port_idx) {
        oss <<
    "      PORT_" << static_cast<int>(port_idx)  << ":\n" <<
    "            BDF: " << 0 << "\n" <<
    "            TYPE: " << ainic_info.port.ports[port_idx].type << "\n" <<
    "            FLAVOUR: " << ainic_info.port.ports[port_idx].flavour << "\n" <<
    "            NETDEV: " << ainic_info.port.ports[port_idx].netdev << "\n" <<
    "            IFINDEX: " << static_cast<int>(ainic_info.port.ports[port_idx].ifindex) << "\n" <<
    "            MAC_ADDRESS: " << ainic_info.port.ports[port_idx].mac_address << "\n" <<
    "            CARRIER: " << static_cast<int>(ainic_info.port.ports[port_idx].carrier) << "\n" <<
    "            MTU: " << ainic_info.port.ports[port_idx].mtu << "B\n" <<
    "            LINK_STATE: " << ainic_info.port.ports[port_idx].link_state << "\n" <<
    "            LINK_SPEED: " << ainic_info.port.ports[port_idx].link_speed << " Mb/s\n" <<
    "            ACTIVE_FEC: " << ainic_info.port.ports[port_idx].active_fec  << "\n" <<
    "            AUTONEG: " << ainic_info.port.ports[port_idx].autoneg << "\n" <<
    "            PAUSE_AUTONEG: " << ainic_info.port.ports[port_idx].pause_autoneg << "\n" <<
    "            PAUSE_RX: " << ainic_info.port.ports[port_idx].pause_rx << "\n" <<
    "            PAUSE_TX: " << ainic_info.port.ports[port_idx].pause_tx << "\n";
    }//port
    oss << 
    "   RDMA_DEVICES: " << "\n";
    int rdma_dev_idx = 0;
    for(int port_idx = 0; port_idx < ainic_info.port.num_ports; ++port_idx) {
        oss << 
    "      RDMA_DEVICES: " << "\n";
        for(uint8_t rdma_dev_idx = 0; rdma_dev_idx < ainic_info.rdma_dev.num_rdma_dev; ++rdma_dev_idx) {
            oss <<
    "            RDMA_DEVICE_" << static_cast<int>(rdma_dev_idx) << ":\n" <<
    "               NAME: " <<ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].rdma_dev << "\n" <<
    "               NODE_GUID: " <<ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].node_guid << "\n" <<
    "               NODE_TYPE: " <<ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].node_type << "\n" <<
    "               SYS_IMAGE_GUID: " <<ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].sys_image_guid << "\n" <<
    "               FW_VER: " <<ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].fw_ver << "\n"
    "               PORTS:\n";
            for(uint8_t rdma_port_idx = 0; rdma_port_idx <ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].num_rdma_ports; ++rdma_port_idx) {
                oss <<
    "                  PORT_" << static_cast<int>(rdma_port_idx) << ":\n" <<
    "                     NETDEV: " <<ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].rdma_port_info[rdma_port_idx].netdev << "\n" <<
    // "                     PORT_NUM: " << static_cast<int>(ainic_info.port.ports[port_idx].rdma_dev[rdma_dev_idx].rdma_port_info[rdma_port_idx].port_num) << "\n" <<
    "                     STATE: " <<ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].rdma_port_info[rdma_port_idx].state << "\n" <<
    // "                     MAX_MTU: " << static_cast<int>(ainic_info.port.ports[port_idx].rdma_dev[rdma_dev_idx].rdma_port_info[rdma_port_idx].max_mtu) << "\n" <<
    // "                     ACTIVE_MTU: " << static_cast<int>(ainic_info.port.ports[port_idx].rdma_dev[rdma_dev_idx].rdma_port_info[rdma_port_idx].active_mtu) << 
                        "\n";
            }
        }//num_infiniband

    }
    std::cout << oss.str();
}

std::optional<std::vector<amd::smi::AMDSmiAINICDevice::AINICInfo>> get_nics() {

    auto &amdsmi = amd::smi::AMDSmiSystem::getInstance();
    uint32_t soc_count = 10;
    std::vector<amdsmi_socket_handle> sockets(soc_count);
    // Get the sockets of the system
    amdsmi_status_t status = amdsmi_get_socket_handles(&soc_count, &sockets[0]);
    if (status != AMDSMI_STATUS_SUCCESS){
        return std::nullopt;
    }
    std::cout << "Got " << soc_count << " socket(s)\n";

    std::vector<amd::smi::AMDSmiAINICDevice::AINICInfo> nics;
    for (uint32_t index = 0 ; index < soc_count; index++){
        uint32_t processor_count = 0;
        status = amdsmi_get_processor_handles_by_type(
            sockets[index], 
            AMDSMI_PROCESSOR_TYPE_AMD_NIC,
            nullptr, &processor_count);
        if (status != AMDSMI_STATUS_SUCCESS){
            return std::nullopt;
        }
        std::cout << "Got " << processor_count << " processors for socket " << index << ":\n";
        std::vector<amdsmi_processor_handle> processor_handles(processor_count);
        status = amdsmi_get_processor_handles_by_type(
            sockets[index], 
            AMDSMI_PROCESSOR_TYPE_AMD_NIC,
            processor_handles.data(), &processor_count);
        if (status != AMDSMI_STATUS_SUCCESS){
            return std::nullopt;
        }

        for(uint32_t idx = 0; idx < processor_count; ++idx){
            amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info = {0};
            amdsmi_status_t status = amdsmi_get_ainic_info(processor_handles[idx], &ainic_info);
            if(status == AMDSMI_STATUS_SUCCESS){
                dump_ainic_info(idx, ainic_info);
                nics.emplace_back(ainic_info);
            }
        }
    }
    return nics;
}

std::map<std::string, uint64_t> port_stats() {
    amdsmi_processor_handle processor_handle = nullptr;
    amdsmi_status_t status = smi_amdgpu_get_ainic_processor_handle_by_index(0, &processor_handle);

    uint32_t rdma_port_index = 0;
    uint32_t num_stats = 0;
    std::unique_ptr<amdsmi_nic_stat_t[]> stats;
    status = amdsmi_get_nic_rdma_port_statistics(
        processor_handle, 
        rdma_port_index,
        &num_stats, 
        nullptr);
    if(status != AMDSMI_STATUS_SUCCESS) {
        return {};
    }

    std::cout << "[" << __FILE__ << ":" << __LINE__ << "]\nnum_stats: " << num_stats << "\n";
    stats = std::make_unique<amdsmi_nic_stat_t[]>(num_stats);
    status = amdsmi_get_nic_rdma_port_statistics(
        processor_handle,
        rdma_port_index,
        &num_stats,
        stats.get());
    if(status != AMDSMI_STATUS_SUCCESS) {
        return {};
    }

    std::map<std::string, uint64_t> values;
    for(uint32_t idx = 0; idx < num_stats; ++idx) {
        std::cout << "Stat " << idx << ": " << stats[idx].name << " = " << stats[idx].value << std::endl;
        values[stats[idx].name] = stats[idx].value;
    }
    return values;
}
}

int main(int argc, char *argv[]) {
    RAII _([]() {amdsmi_init(AMDSMI_INIT_AMD_NICS);}, []() { amdsmi_shut_down(); });
    auto nics = get_nics();
    for(const auto& [key, value]: port_stats()) {
        std::cout << key << ":" << value << "\n";
    }

    return 0;
}