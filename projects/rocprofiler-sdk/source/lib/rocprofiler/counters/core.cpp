#include "lib/rocprofiler/counters/core.hpp"

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/aql/packet_construct.hpp"
#include "lib/rocprofiler/context/context.hpp"
#include "lib/rocprofiler/hsa/queue_controller.hpp"
#include "lib/rocprofiler/registration.hpp"

#include <rocprofiler/rocprofiler.h>

namespace rocprofiler
{
namespace counters
{
/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 *
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
std::unique_ptr<rocprofiler::hsa::AQLPacket>
queue_cb(const std::shared_ptr<rocprofiler::counters::counter_callback_info>& info,
         const hsa::Queue&                                                    queue,
         hsa::ClientID,
         const hsa_ext_amd_aql_pm4_packet_t&)
{
    if(!info) return nullptr;

    std::unique_ptr<rocprofiler::hsa::AQLPacket> ret_pkt;

    // Check packet cache
    info->packets.wlock([&](auto& pkt_vector) {
        // Delay packet generator construction until first HSA packet is processed
        // This ensures that HSA exists
        if(!info->pkt_generator)
        {
            info->pkt_generator = std::make_unique<rocprofiler::aql::AQLPacketConstruct>(
                queue.get_agent(),
                std::vector<counters::Metric>{info->profile_cfg.reqired_hw_counters.begin(),
                                              info->profile_cfg.reqired_hw_counters.end()});
        }

        if(!pkt_vector.empty())
        {
            ret_pkt = std::move(pkt_vector.back());
            pkt_vector.pop_back();
        }
    });

    if(!ret_pkt)
    {
        // If we do not have a packet in the cache, create one.
        ret_pkt =
            info->pkt_generator->construct_packet(hsa::get_queue_controller().get_ext_table());
    }
    return ret_pkt;
}

/**
 * Callback called by HSA interceptor when the kernel has completed processing.
 */
void
completed_cb(const std::shared_ptr<rocprofiler::counters::counter_callback_info>& info,
             const hsa::Queue&                                                    queue,
             hsa::ClientID,
             const hsa_ext_amd_aql_pm4_packet_t&          kernel,
             std::unique_ptr<rocprofiler::hsa::AQLPacket> pkt)
{
    if(!info) return;

    // auto out_buf = pkt->profile.output_buffer.ptr;
    // Read data and create user return....

    // return AQL packet for reuse.

    info->packets.wlock([&](auto& pkt_vector) {
        if(pkt)
        {
            pkt_vector.emplace_back(std::move(pkt));
        }
    });

    if(!info->user_cb) return;

    info->user_cb(queue.get_id(),
                  info->profile_cfg.agent,
                  rocprofiler_correlation_id_t{},
                  reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(&kernel),
                  info->callback_args,
                  nullptr,  // Date pointer does here.
                  0,        // Number of objects
                  info->profile_cfg.id);
}

class CounterController
{
public:
    // Adds a counter collection profile to our global cache.
    // Note: these profiles can be used across multiple contexts
    //       and are independent of the context.
    uint64_t add_profile(profile_config&& config)
    {
        static std::atomic<uint64_t> profile_val = 1;
        uint64_t                     ret         = 0;
        _configs.wlock([&](auto& data) {
            config.id = rocprofiler_profile_config_id_t{.handle = profile_val};
            data.emplace(profile_val, std::move(config));
            ret = profile_val;
            profile_val++;
        });
        return ret;
    }

    void destroy_profile(uint64_t id)
    {
        _configs.wlock([&](auto& data) { data.erase(id); });
    }

    // Setup the counter collection service. counter_callback_info is created here
    // to contain the counters that need to be collected (specified in profile_id) and
    // the AQL packet generator for injecting packets. Note: the service is created
    // in the stop state.
    bool configure_dispatch(rocprofiler_context_id_t                         context_id,
                            uint64_t                                         profile_id,
                            rocprofiler_profile_counting_dispatch_callback_t callback,
                            void*                                            callback_args) const
    {
        auto& ctx = *rocprofiler::context::get_registered_contexts().at(context_id.handle);

        // Note: A single profile config could be used on multiple contexts
        profile_config cfg;
        _configs.rlock([&](const auto& map) { cfg = map.at(profile_id); });

        if(!ctx.counter_collection)
        {
            ctx.counter_collection =
                std::make_unique<rocprofiler::context::counter_collection_service>();
        }

        auto& cb = *ctx.counter_collection->callbacks.emplace_back(
            std::make_shared<rocprofiler::counters::counter_callback_info>());

        cb.user_cb = callback;

        // Secondary copy of the config to be shared with async callback
        cb.profile_cfg   = cfg;
        cb.callback_args = callback_args;
        cb.context       = context_id;
        return true;
    }

private:
    rocprofiler::common::Synchronized<std::unordered_map<uint64_t, profile_config>> _configs;
};

CounterController&
get_controller()
{
    static CounterController controller;
    return controller;
}

uint64_t
create_counter_profile(profile_config&& config)
{
    return get_controller().add_profile(std::move(config));
}

void
destroy_counter_profile(uint64_t id)
{
    get_controller().destroy_profile(id);
}

void
start_context(rocprofiler_context_id_t context_id)
{
    auto& ctx        = *rocprofiler::context::get_registered_contexts().at(context_id.handle);
    auto& controller = hsa::get_queue_controller();
    if(!ctx.counter_collection) return;

    // Only one thread should be attempting to enable/disable this context
    ctx.counter_collection->enabled.wlock([&](auto& enabled) {
        if(enabled) return;
        for(auto& cb : ctx.counter_collection->callbacks)
        {
            // Insert our callbacks into HSA Interceptor. This
            // turns on counter instrumentation.
            cb->queue_id = controller.add_callback(
                cb->profile_cfg.agent,
                [=](const hsa::Queue&                   q,
                    hsa::ClientID                       c,
                    const hsa_ext_amd_aql_pm4_packet_t& kern_pkt) {
                    return queue_cb(cb, q, c, kern_pkt);
                },
                // Completion CB
                [=](const hsa::Queue&                   q,
                    hsa::ClientID                       c,
                    const hsa_ext_amd_aql_pm4_packet_t& kern_pkt,
                    std::unique_ptr<hsa::AQLPacket>     aql) {
                    completed_cb(cb, q, c, kern_pkt, std::move(aql));
                });
        }
        enabled = true;
    });
}

void
stop_context(rocprofiler_context_id_t context_id)
{
    auto& controller = hsa::get_queue_controller();
    auto& ctx        = *rocprofiler::context::get_registered_contexts().at(context_id.handle);
    if(!ctx.counter_collection) return;

    ctx.counter_collection->enabled.wlock([&](auto& enabled) {
        if(!enabled) return;
        for(auto& cb : ctx.counter_collection->callbacks)
        {
            // Remove our callbacks from HSA's queue controller
            controller.remove_callback(cb->queue_id);
            cb->queue_id = -1;
        }
        enabled = false;
    });
}

bool
configure_dispatch(rocprofiler_context_id_t                         context_id,
                   uint64_t                                         profile_id,
                   rocprofiler_profile_counting_dispatch_callback_t callback,
                   void*                                            callback_args)
{
    return get_controller().configure_dispatch(context_id, profile_id, callback, callback_args);
}

}  // namespace counters
}  // namespace rocprofiler
