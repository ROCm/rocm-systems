#include "ainic_stats.hpp"

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

const AINICStatsCollector::nic_params_t&
AINICStatsCollector::params()
{
    return _nic_params;
}

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
    amdsmi_status_t                          status;
    const std::vector<amdsmi_ai_nic_info_t>& ai_nic_infos(_amdsmi.get_ai_nic_info());

    auto size = ai_nic_infos.size();

    for(unsigned nic_index {}; nic_index < size; ++nic_index)
    {
        auto& ai_nic_info = ai_nic_infos[nic_index];

        for (unsigned port_idx {}; port_idx < ai_nic_info.num_ports; ++port_idx)
        {
            auto&    port          = ai_nic_info.port[port_idx];
            unsigned num_rdma_devs = port.num_rdma_dev;

            for (unsigned rdma_dev_idx {}; rdma_dev_idx < num_rdma_devs; ++rdma_dev_idx)
            {
                auto&    rdma_dev  = port.rdma_dev[rdma_dev_idx];
                unsigned num_ports = rdma_dev.rdma_port;

                for (unsigned rdma_port_idx {}; rdma_port_idx < num_ports; ++rdma_port_idx)
                {
                    auto& rdma_port = rdma_dev.rdma_port_info[rdma_port_idx];

                    NICData data {};
                    data._name = rdma_dev.rdma_dev;
                    data._netdev = rdma_port.netdev;

                    amdsmi_processor_handle processor_handle{};

                    status = smi_amdgpu_get_ainic_processor_handle_by_index(
                        nic_index, &processor_handle);
                    if(status != AMDSMI_STATUS_SUCCESS)
                    {
                        std::cerr << "Error: for index " << nic_index
                                  << ", smi_amdgpu_get_ainic_processor_handle_by_index returned "
                                  << status << std::endl;
                        continue;
                    }

                    std::unique_ptr<amdsmi_nic_stat_t[]> stats;

                    // Call *_statistics the first time to get the number of statistics.
                    if (data._num_stats == 0)
                    {
                        amdsmi_get_nic_rdma_port_statistics(
                            processor_handle, rdma_port_idx, &data._num_stats, nullptr);
                    }

                    // Allocate stats.
                    stats = std::make_unique<amdsmi_nic_stat_t[]>(data._num_stats);

                    // Call *_statistics the second time to get the statistics.
                    amdsmi_get_nic_rdma_port_statistics(
                        processor_handle, rdma_port_idx, &data._num_stats, stats.get());

                    for(uint32_t stat_idx{}; stat_idx < data._num_stats; ++stat_idx)
                    {
                        if(strcmp(stats[stat_idx].name, NICData::RX_RDMA_UCAST_BYTES) ==
                            0)
                        {
                            data._rx_rdma_ucast_bytes =
                                static_cast<std::uint32_t>(stats[stat_idx].value);
                        }
                        else if(strcmp(stats[stat_idx].name, NICData::RX_RDMA_UCAST_PKTS) == 0)
                        {
                            data._rx_rdma_ucast_pkts =
                                static_cast<std::uint32_t>(stats[stat_idx].value);
                        }
                        else if(strcmp(stats[stat_idx].name, NICData::TX_RDMA_UCAST_BYTES) == 0)
                        {
                            data._tx_rdma_ucast_bytes =
                                static_cast<std::uint32_t>(stats[stat_idx].value);
                        }
                        else if(strcmp(stats[stat_idx].name, NICData::TX_RDMA_UCAST_PKTS) == 0)
                        {
                            data._tx_rdma_ucast_pkts =
                                static_cast<std::uint32_t>(stats[stat_idx].value);
                        }
                        else if(strcmp(stats[stat_idx].name, NICData::RX_RDMA_CNP_PKTS) == 0)
                        {
                            data._rx_rdma_cnp_pkts =
                                static_cast<std::uint32_t>(stats[stat_idx].value);
                        }
                        else if(strcmp(stats[stat_idx].name, NICData::TX_RDMA_CNP_PKTS) == 0)
                        {
                            data._tx_rdma_cnp_pkts =
                                static_cast<std::uint32_t>(stats[stat_idx].value);
                        }
                    }

                    update_data_for_one_nic(data);
                }
            }
        }
    }
#endif  // USE_AINIC
}

void
AINICStatsCollector::update_data_for_one_nic(NICData& data)
{
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

void \
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
