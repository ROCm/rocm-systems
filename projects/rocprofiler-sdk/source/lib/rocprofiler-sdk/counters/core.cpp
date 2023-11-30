// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/core.hpp"

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofiler
{
namespace counters
{
class CounterController
{
public:
    // Adds a counter collection profile to our global cache.
    // Note: these profiles can be used across multiple contexts
    //       and are independent of the context.
    uint64_t add_profile(std::shared_ptr<profile_config>&& config)
    {
        static std::atomic<uint64_t> profile_val = 1;
        uint64_t                     ret         = 0;
        _configs.wlock([&](auto& data) {
            config->id = rocprofiler_profile_config_id_t{.handle = profile_val};
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
    static bool configure_dispatch(rocprofiler_context_id_t                         context_id,
                                   rocprofiler_buffer_id_t                          buffer,
                                   rocprofiler_profile_counting_dispatch_callback_t callback,
                                   void*                                            callback_args)
    {
        auto& ctx = *rocprofiler::context::get_registered_contexts().at(context_id.handle);

        if(!ctx.counter_collection)
        {
            ctx.counter_collection =
                std::make_unique<rocprofiler::context::counter_collection_service>();
        }

        auto& cb = *ctx.counter_collection->callbacks.emplace_back(
            std::make_shared<counter_callback_info>());

        cb.user_cb       = callback;
        cb.callback_args = callback_args;
        cb.context       = context_id;
        cb.buffer        = buffer;
        cb.internal_context =
            rocprofiler::context::get_registered_contexts().at(context_id.handle).get();

        return true;
    }

    std::shared_ptr<profile_config> get_profile_cfg(rocprofiler_profile_config_id_t id)
    {
        std::shared_ptr<profile_config> cfg;
        _configs.rlock([&](const auto& map) { cfg = map.at(id.handle); });
        return cfg;
    }

private:
    rocprofiler::common::Synchronized<std::unordered_map<uint64_t, std::shared_ptr<profile_config>>>
        _configs;
};

CounterController&
get_controller()
{
    static CounterController controller;
    return controller;
}

uint64_t
create_counter_profile(std::shared_ptr<profile_config>&& config)
{
    return get_controller().add_profile(std::move(config));
}

void
destroy_counter_profile(uint64_t id)
{
    get_controller().destroy_profile(id);
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 *
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
std::unique_ptr<rocprofiler::hsa::AQLPacket>
queue_cb(const std::shared_ptr<counter_callback_info>& info,
         const hsa::Queue&                             queue,
         hsa::ClientID,
         const hsa::rocprofiler_packet&                                  pkt,
         const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
         const context::correlation_id*                                  correlation_id)
{
    if(!info || !info->user_cb) return nullptr;

    auto _corr_id_v =
        rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};
    if(const auto* _corr_id = correlation_id)
    {
        _corr_id_v.internal = _corr_id->internal;
        if(const auto* extrenal =
               rocprofiler::common::get_val(extern_corr_ids, info->internal_context))
        {
            _corr_id_v.external = *extrenal;
        }
    }

    rocprofiler_profile_config_id_t req_profile = {.handle = 0};
    info->user_cb(queue.get_id(),
                  queue.get_agent().get_rocp_agent(),
                  _corr_id_v,
                  &pkt.kernel_dispatch,
                  info->callback_args,
                  &req_profile);
    if(req_profile.handle == 0) return nullptr;

    auto prof_config = get_controller().get_profile_cfg(req_profile);
    CHECK(prof_config);

    std::unique_ptr<rocprofiler::hsa::AQLPacket> ret_pkt;

    // Check packet cache
    prof_config->packets.wlock([&](auto& pkt_vector) {
        // Delay packet generator construction until first HSA packet is processed
        // This ensures that HSA exists
        if(!prof_config->pkt_generator)
        {
            // One time setup of profile config
            if(prof_config->reqired_hw_counters.empty())
            {
                auto& config     = *prof_config;
                auto  agent_name = std::string(config.agent.name);
                for(const auto& metric : config.metrics)
                {
                    auto req_counters =
                        get_required_hardware_counters(get_ast_map(), agent_name, metric);

                    if(!req_counters)
                    {
                        throw std::runtime_error(
                            fmt::format("Could not find counter {}", metric.name()));
                    }

                    // Special metrics are those that are not hw counters but other
                    // constants like MAX_WAVE_SIZE
                    for(const auto& req_metric : *req_counters)
                    {
                        if(req_metric.special().empty())
                        {
                            config.reqired_hw_counters.insert(req_metric);
                        }
                        else
                        {
                            config.required_special_counters.insert(req_metric);
                        }
                    }

                    const auto& asts      = get_ast_map();
                    const auto* agent_map = rocprofiler::common::get_val(asts, agent_name);
                    if(!agent_map)
                        throw std::runtime_error(
                            fmt::format("Coult not build AST for {}", agent_name));
                    const auto* counter_ast =
                        rocprofiler::common::get_val(*agent_map, metric.name());
                    if(!counter_ast)
                    {
                        throw std::runtime_error(
                            fmt::format("Coult not find AST for {}", metric.name()));
                    }
                    config.asts.push_back(*counter_ast);
                    config.asts.back().set_dimensions();
                }
            }

            prof_config->pkt_generator = std::make_unique<rocprofiler::aql::AQLPacketConstruct>(
                queue.get_agent(),
                std::vector<counters::Metric>{prof_config->reqired_hw_counters.begin(),
                                              prof_config->reqired_hw_counters.end()});
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
        ret_pkt = prof_config->pkt_generator->construct_packet(
            hsa::get_queue_controller().get_ext_table());
    }

    info->packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), prof_config); });

    return ret_pkt;
}

/**
 * Callback called by HSA interceptor when the kernel has completed processing.
 */
void
completed_cb(const std::shared_ptr<counter_callback_info>& info,
             const hsa::Queue&,
             hsa::ClientID,
             hsa::rocprofiler_packet,
             const hsa::Queue::queue_info_session_t&      session,
             std::unique_ptr<rocprofiler::hsa::AQLPacket> pkt)
{
    if(!info || !pkt) return;

    std::shared_ptr<profile_config> prof_config;
    // Get the Profile Config
    info->packet_return_map.wlock([&](auto& data) {
        prof_config = data.at(pkt.get());
        data.erase(pkt.get());
    });

    auto decoded_pkt = EvaluateAST::read_pkt(prof_config->pkt_generator.get(), *pkt);
    EvaluateAST::read_special_counters(
        prof_config->agent, prof_config->required_special_counters, decoded_pkt);

    prof_config->packets.wlock([&](auto& pkt_vector) {
        if(pkt)
        {
            pkt_vector.emplace_back(std::move(pkt));
        }
    });

    if(!info->buffer) return;

    std::vector<rocprofiler_record_counter_t> out;
    rocprofiler::buffer::instance*            buf = nullptr;

    buf = CHECK_NOTNULL(buffer::get_buffer(info->buffer->handle));

    auto _corr_id_v =
        rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};
    if(const auto* _corr_id = session.correlation_id)
    {
        _corr_id_v.internal = _corr_id->internal;
        if(const auto* extrenal =
               rocprofiler::common::get_val(session.extern_corr_ids, info->internal_context))
        {
            _corr_id_v.external = *extrenal;
        }
    }

    for(auto& ast : prof_config->asts)
    {
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        auto* ret = ast.evaluate(decoded_pkt, cache);
        CHECK(ret);
        ast.set_out_id(*ret);

        for(auto& val : *ret)
        {
            val.corr_id = _corr_id_v;
            buf->emplace(ROCPROFILER_BUFFER_CATEGORY_COUNTERS, 0, val);
        }
    }
}

void
start_context(context::context* ctx)
{
    if(!ctx || !ctx->counter_collection) return;

    auto& controller = hsa::get_queue_controller();

    // Only one thread should be attempting to enable/disable this context
    ctx->counter_collection->enabled.wlock([&](auto& enabled) {
        if(enabled) return;
        for(auto& cb : ctx->counter_collection->callbacks)
        {
            // Insert our callbacks into HSA Interceptor. This
            // turns on counter instrumentation.
            cb->queue_id = controller.add_callback(
                std::nullopt,
                [=](const hsa::Queue&                                               q,
                    hsa::ClientID                                                   c,
                    const hsa::rocprofiler_packet&                                  kern_pkt,
                    const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
                    const context::correlation_id* correlation_id) {
                    return queue_cb(cb, q, c, kern_pkt, extern_corr_ids, correlation_id);
                },
                // Completion CB
                [=](const hsa::Queue&                       q,
                    hsa::ClientID                           c,
                    hsa::rocprofiler_packet                 kern_pkt,
                    const hsa::Queue::queue_info_session_t& session,
                    std::unique_ptr<hsa::AQLPacket>         aql) {
                    completed_cb(cb, q, c, kern_pkt, session, std::move(aql));
                });
        }
        enabled = true;
    });
}

void
stop_context(context::context* ctx)
{
    if(!ctx || !ctx->counter_collection) return;

    auto& controller = hsa::get_queue_controller();

    ctx->counter_collection->enabled.wlock([&](auto& enabled) {
        if(!enabled) return;
        for(auto& cb : ctx->counter_collection->callbacks)
        {
            // Remove our callbacks from HSA's queue controller
            controller.remove_callback(cb->queue_id);
            cb->queue_id = -1;
        }
        enabled = false;
    });
}

bool
configure_buffered_dispatch(rocprofiler_context_id_t                         context_id,
                            rocprofiler_buffer_id_t                          buffer,
                            rocprofiler_profile_counting_dispatch_callback_t callback,
                            void*                                            callback_args)
{
    return get_controller().configure_dispatch(context_id, buffer, callback, callback_args);
}

}  // namespace counters
}  // namespace rocprofiler
