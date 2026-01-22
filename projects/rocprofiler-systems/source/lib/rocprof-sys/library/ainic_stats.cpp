#include "ainic_stats.hpp"

#include <memory>

std::string
NICData::to_string() const
{
    std::ostringstream stream;

    stream << "[_name=" << _name << ", _netdev=" << _netdev
        << ", _rx_rdma_ucast_bytes=" << _rx_rdma_ucast_bytes
        << ", _rx_rdma_ucast_pkts="  << _rx_rdma_ucast_pkts
        << ", _tx_rdma_ucast_bytes=" << _tx_rdma_ucast_bytes
        << ", _tx_rdma_ucast_pkts="  << _tx_rdma_ucast_pkts
        << ", _rx_rdma_cnp_pkts=" << _rx_rdma_cnp_pkts
        << ", _tx_rdma_cnp_pkts=" << _tx_rdma_cnp_pkts << "]";
    return stream.str();
}

const char* NICData::RX_RDMA_UCAST_BYTES = "rx_rdma_ucast_bytes";
const char* NICData::RX_RDMA_UCAST_PKTS  = "rx_rdma_ucast_pkts";
const char* NICData::TX_RDMA_UCAST_BYTES = "tx_rdma_ucast_bytes";
const char* NICData::TX_RDMA_UCAST_PKTS  = "tx_rdma_ucast_pkts";
const char* NICData::RX_RDMA_CNP_PKTS    = "rx_rdma_cnp_pkts";
const char* NICData::TX_RDMA_CNP_PKTS    = "tx_rdma_cnp_pkts";

AINICStatsCollector::AINICStatsCollector()
#ifdef USE_AINIC
    :
    _amdsmi(amd::smi::AMDSmiSystem::getInstance())
#endif
{}

bool AINICStatsCollector::find_nic(const std::string& nic, NICData& data)
{
    auto pair = _nic_params.find(nic);
    if(pair == _nic_params.end())
    {
        return false;
    }
    data = pair->second;
    return true;
}

void AINICStatsCollector::update_stats()
{
#ifdef USE_AINIC
    uint32_t soc_count{};
    std::unique_ptr<amdsmi_socket_handle[]> sockets;
    // Call amdsmi_get_socket_handles with second parameter (socket_handles)
    // nullptr to get the number of socket handles.
    amdsmi_status_t status = amdsmi_get_socket_handles(&soc_count, nullptr);
    if (status != AMDSMI_STATUS_SUCCESS){
        return;
    }

    if(soc_count == 0) // Nothing to do.
        return;

    // Allocate a buffer for soc_count socket handles.
    sockets = std::make_unique<amdsmi_socket_handle[]>(soc_count);
    // Get the socket handles.
    status = amdsmi_get_socket_handles(&soc_count, sockets.get());
    if (status != AMDSMI_STATUS_SUCCESS){
        return;
    }

    // Iterate through all socket handles to find all AI NIC
    // processor handles and update the statistics for each of them.
    std::vector<amd::smi::AMDSmiAINICDevice::AINICInfo> nics;
    for (uint32_t index = 0 ; index < soc_count; index++)
    {
        uint32_t processor_count = 0;
        status = amdsmi_get_processor_handles_by_type(
            sockets[index],
            AMDSMI_PROCESSOR_TYPE_AMD_NIC,
            nullptr, &processor_count);
        if (status != AMDSMI_STATUS_SUCCESS){
            return;
        }
        std::vector<amdsmi_processor_handle> processor_handles(processor_count);
        status = amdsmi_get_processor_handles_by_type(
            sockets[index],
            AMDSMI_PROCESSOR_TYPE_AMD_NIC,
            processor_handles.data(), &processor_count);
        if (status != AMDSMI_STATUS_SUCCESS)
        {
            return;
        }
        for(uint32_t idx = 0; idx < processor_count; ++idx)
        {
            amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info {};
            amdsmi_get_ainic_info(processor_handles[idx], &ainic_info);

            // Update info and stats.
            auto processor_handle = processor_handles[idx];
            update_data_for_one_nic(processor_handle);
            nics.emplace_back(ainic_info);
        }
    }
#endif  // USE_AINIC
}

size_t
AINICStatsCollector::get_nic_count()
{
#ifdef USE_AINIC
    auto &amdsmi = amd::smi::AMDSmiSystem::getInstance();
    uint32_t soc_count = 10;
    std::vector<amdsmi_socket_handle> sockets(soc_count);
    // Get the sockets of the system
    amdsmi_status_t status = amdsmi_get_socket_handles(&soc_count, &sockets[0]);
    if (status != AMDSMI_STATUS_SUCCESS){
        return 0;
    }

    // For all sockets, find all NIC processor handles.
    size_t nic_count{};
    for (uint32_t index = 0 ; index < soc_count; index++){
        uint32_t processor_count = 0;
        status = amdsmi_get_processor_handles_by_type(
            sockets[index],
            AMDSMI_PROCESSOR_TYPE_AMD_NIC,
            nullptr, &processor_count);
        if (status != AMDSMI_STATUS_SUCCESS){
            continue;
        }
        nic_count += processor_count;
    }
    return nic_count;
#else
    return 0;
#endif  // USE_AINIC
}

void AINICStatsCollector::update_data_for_one_nic(amdsmi_processor_handle processor_handle)
{
#ifdef USE_AINIC
    amd::smi::AMDSmiAINICDevice::AINICInfo ainic_info;
    amdsmi_get_ainic_info(processor_handle, &ainic_info);

    for(int port_idx = 0; port_idx < ainic_info.port.num_ports; ++port_idx) {
        for(uint8_t rdma_dev_idx = 0; rdma_dev_idx < ainic_info.rdma_dev.num_rdma_dev; ++rdma_dev_idx) {
            for(uint8_t rdma_port_idx = 0; rdma_port_idx < ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].num_rdma_ports; ++rdma_port_idx) {
                NICData data;
                data._name = ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].rdma_dev;
                data._netdev = ainic_info.rdma_dev.rdma_dev_info[rdma_dev_idx].rdma_port_info[rdma_port_idx].netdev;

                std::unique_ptr<amdsmi_nic_stat_t[]> stats;

                // Call *_statistics the first time to get the number of statistics.
                uint32_t num_stats{};
                amdsmi_get_nic_rdma_port_statistics(
                    processor_handle,
                    rdma_port_idx,
                    &num_stats,
                    nullptr
                );

                // Allocate stats.
                stats = std::make_unique<amdsmi_nic_stat_t[]>(num_stats);

                // Call *_statistics the second time to get the statistics.
                amdsmi_get_nic_rdma_port_statistics(
                    processor_handle,
                    rdma_port_idx,
                    &num_stats,
                    stats.get()
                );

                // Retrieve relevant stats.
                for (uint32_t stat_idx{}; stat_idx < num_stats; ++stat_idx)
                {
                    if (strcmp(stats[stat_idx].name, NICData::RX_RDMA_UCAST_BYTES) == 0)
                    {
                        data._rx_rdma_ucast_bytes = static_cast<std::uint32_t>(stats[stat_idx].value);
                    }
                    else if (strcmp(stats[stat_idx].name, NICData::RX_RDMA_UCAST_PKTS) == 0)
                    {
                        data._rx_rdma_ucast_pkts = static_cast<std::uint32_t>(stats[stat_idx].value);
                    }
                    else if (strcmp(stats[stat_idx].name, NICData::TX_RDMA_UCAST_BYTES) == 0)
                    {
                        data._tx_rdma_ucast_bytes = static_cast<std::uint32_t>(stats[stat_idx].value);
                    }
                    else if (strcmp(stats[stat_idx].name, NICData::TX_RDMA_UCAST_PKTS) == 0)
                    {
                        data._tx_rdma_ucast_pkts = static_cast<std::uint32_t>(stats[stat_idx].value);
                    }
                    else if (strcmp(stats[stat_idx].name, NICData::RX_RDMA_CNP_PKTS) == 0)
                    {
                        data._rx_rdma_cnp_pkts = static_cast<std::uint32_t>(stats[stat_idx].value);
                    }
                    else if (strcmp(stats[stat_idx].name, NICData::TX_RDMA_CNP_PKTS) == 0)
                    {
                        data._tx_rdma_cnp_pkts = static_cast<std::uint32_t>(stats[stat_idx].value);
                    }
                }

                // We have filled in the fields of data. Now update _nic_params and _nic_delta_params.
                auto it = _nic_params.find(data._netdev);
                if(it == _nic_params.end())  // not found
                {
                    NICData new_delta;
                    new_delta._name   = data._name;
                    new_delta._netdev = data._netdev;

                    new_delta._rx_rdma_ucast_bytes = 0;
                    new_delta._tx_rdma_ucast_bytes = 0;
                    new_delta._rx_rdma_ucast_pkts  = 0;
                    new_delta._tx_rdma_ucast_pkts  = 0;

                    new_delta._rx_rdma_cnp_pkts     = 0;
                    new_delta._tx_rdma_cnp_pkts     = 0;
                    _nic_params[data._netdev]       = data;
                    _nic_delta_params[data._netdev] = new_delta;
                }
                else
                {
                    NICData  new_delta;
                    NICData& old_data = it->second;

                    new_delta._name   = data._name;
                    new_delta._netdev = data._netdev;

                    new_delta._rx_rdma_ucast_bytes =
                        data._rx_rdma_ucast_bytes - old_data._rx_rdma_ucast_bytes;
                    new_delta._tx_rdma_ucast_bytes =
                        data._tx_rdma_ucast_bytes - old_data._tx_rdma_ucast_bytes;
                    new_delta._rx_rdma_ucast_pkts =
                        data._rx_rdma_ucast_pkts - old_data._rx_rdma_ucast_pkts;
                    new_delta._tx_rdma_ucast_pkts =
                        data._tx_rdma_ucast_pkts - old_data._tx_rdma_ucast_pkts;

                    new_delta._rx_rdma_cnp_pkts = data._rx_rdma_cnp_pkts - old_data._rx_rdma_cnp_pkts;
                    new_delta._tx_rdma_cnp_pkts = data._tx_rdma_cnp_pkts - old_data._tx_rdma_cnp_pkts;

                    _nic_params[data._netdev]   = data;
                    _nic_delta_params[data._netdev] = new_delta;
                }
            }
        }
    }
#endif //  USE_AINIC
}

void
AINICStatsCollector::get_data(const std::string& nic, NICData& data) const
{
    auto it = _nic_delta_params.find(nic);
    if(it == _nic_delta_params.end()) // not found
    {
        data._netdev              = nic;
        data._name                = "";
        data._rx_rdma_ucast_bytes = 0;
        data._tx_rdma_ucast_bytes = 0;
        data._rx_rdma_ucast_pkts  = 0;
        data._tx_rdma_ucast_pkts  = 0;

        data._rx_rdma_cnp_pkts = 0;
        data._tx_rdma_cnp_pkts = 0;
    }
    else
    {
        data = it->second;
    }
}
