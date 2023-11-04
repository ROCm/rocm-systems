#pragma once

#include <functional>
#include <map>
#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"
#include "lib/rocprofiler/hsa/queue.hpp"

namespace rocprofiler
{
namespace aql
{
/**
 * Class to construct AQL Packets for a specific agent and metric set.
 * Thie class checks that the counters supplied are collectable on the
 * agent in question (including making sure that they stay within block
 * limits). construct_packet returns an AQLPacket class containing the
 * consturcted start/stop/read packets along with allocated buffers needed
 * to collect the counter data.
 */
class AQLPacketConstruct
{
public:
    AQLPacketConstruct(const hsa::AgentCache& agent, const std::vector<counters::Metric>& metrics);
    std::unique_ptr<hsa::AQLPacket> construct_packet(const AmdExtTable&) const;

    const counters::Metric* event_to_metric(const hsa_ven_amd_aqlprofile_event_t& event) const;

private:
    struct AQLProfileMetric
    {
        counters::Metric                            metric;
        std::vector<hsa_ven_amd_aqlprofile_event_t> instances;
    };

    std::vector<hsa_ven_amd_aqlprofile_event_t> get_all_events() const;
    void                                        can_collect();

    const hsa::AgentCache&                      _agent;
    std::vector<AQLProfileMetric>               _metrics;
    std::vector<hsa_ven_amd_aqlprofile_event_t> _events;
    std::map<std::tuple<hsa_ven_amd_aqlprofile_block_name_t, uint32_t, uint32_t>, counters::Metric>
        _event_to_metric;
};

}  // namespace aql
}  // namespace rocprofiler
