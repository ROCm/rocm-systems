#pragma once

#include <rocprofiler/agent.h>
#include <rocprofiler/dispatch_profile.h>

#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/aql/packet_construct.hpp"
#include "lib/rocprofiler/counters/evaluate_ast.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"

namespace rocprofiler
{
namespace counters
{
// Stores counter profiling information such as the agent
// to collect counters on, the metrics to collect, the hw
// counters needed to evaluate the metrics, and the ASTs.
// This profile can be shared among many rocprof contexts.
struct profile_config
{
    rocprofiler_agent_t           agent{};
    std::vector<counters::Metric> metrics{};
    // HW counters that must be collected to compute the above
    // metrics (derived metrics are broken down into hw counters
    // in this vector).
    std::set<counters::Metric> reqired_hw_counters{};
    // ASTs to evaluate
    std::vector<counters::EvaluateAST> asts{};
    rocprofiler_profile_config_id_t    id{.handle = 0};
};

// Internal counter struct that stores the state needed to handle an intercepted
// HSA kernel packet.
struct counter_callback_info
{
    // Packet generator to create AQL packets for insertion
    std::unique_ptr<rocprofiler::aql::AQLPacketConstruct> pkt_generator{nullptr};
    // A packet cache of AQL packets. This allows reuse of AQL packets (preventing costly
    // allocation of new packets/destruction).
    rocprofiler::common::Synchronized<std::vector<std::unique_ptr<rocprofiler::hsa::AQLPacket>>>
        packets{};
    // User callback
    rocprofiler_profile_counting_dispatch_callback_t user_cb{nullptr};
    // Profile configuration used for this callback containing the counters
    // to collect and the evaluation ASTs
    profile_config profile_cfg{};
    // User id
    void* callback_args{nullptr};
    // Link to the context this is associated with
    rocprofiler_context_id_t context{.handle = 0};
    // HSA Queue ClientID. This is an ID we get when we insert a callback into the
    // HSA queue interceptor. This ID can be used to disable the callback.
    rocprofiler::hsa::ClientID queue_id{-1};
};

uint64_t
     create_counter_profile(profile_config&& config);
void destroy_counter_profile(uint64_t);
bool
     configure_dispatch(rocprofiler_context_id_t                         context_id,
                        uint64_t                                         profile_id,
                        rocprofiler_profile_counting_dispatch_callback_t callback,
                        void*                                            callback_args);
void start_context(rocprofiler_context_id_t);

void stop_context(rocprofiler_context_id_t);
}  // namespace counters
}  // namespace rocprofiler
