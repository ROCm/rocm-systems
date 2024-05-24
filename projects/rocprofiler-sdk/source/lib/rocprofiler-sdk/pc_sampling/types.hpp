#pragma once

#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/cid_manager.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_pc_sampling.h>

#include <memory>

namespace rocprofiler
{
namespace pc_sampling
{
// forward declaration to avoid circular dependency
class PCSCIDManager;

struct PCSAgentSession
{
    const rocprofiler_agent_t*       agent;
    rocprofiler_pc_sampling_method_t method;
    rocprofiler_pc_sampling_unit_t   unit;
    uint64_t                         interval;
    rocprofiler_buffer_id_t          buffer_id;
    // hsa relevant information
    std::optional<hsa_agent_t> hsa_agent = std::nullopt;
    hsa_ven_amd_pcs_t          hsa_pc_sampling;
    hsa::ClientID              intercept_cb_id{-1};
    // ioctl relevant information
    uint32_t ioctl_pcs_id;
    // PC sampling parser
    std::unique_ptr<PCSamplingParserContext> parser;
    // Manager responsible for retiring CIDs
    std::unique_ptr<PCSCIDManager> cid_manager;
};

// TODO static assertions

}  // namespace pc_sampling
}  // namespace rocprofiler
