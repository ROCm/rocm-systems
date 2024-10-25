

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

#include "lib/rocprofiler-sdk/counters/dispatch_handlers.hpp"

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

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
queue_cb(const context::context*                                         ctx,
         const std::shared_ptr<counter_callback_info>&                   info,
         const hsa::Queue&                                               queue,
         const hsa::rocprofiler_packet&                                  pkt,
         rocprofiler_kernel_id_t                                         kernel_id,
         rocprofiler_dispatch_id_t                                       dispatch_id,
         rocprofiler_user_data_t*                                        user_data,
         const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
         const context::correlation_id*                                  correlation_id)
{
    CHECK(info && ctx);

    // Maybe adds serialization packets to the AQLPacket (if serializer is enabled)
    // and maybe adds barrier packets if the state is transitioning from serialized <->
    // unserialized
    auto maybe_add_serialization = [&](auto& gen_pkt) {
        CHECK_NOTNULL(hsa::get_queue_controller())
            ->serializer(&queue)
            .rlock([&](const auto& serializer) {
                for(auto& s_pkt : serializer.kernel_dispatch(queue))
                {
                    gen_pkt->before_krn_pkt.push_back(s_pkt.ext_amd_aql_pm4);
                }
            });
    };

    // Packet generated when no instrumentation is performed. May contain serialization
    // packets/barrier packets (and can be empty).
    auto no_instrumentation = [&]() {
        auto ret_pkt = std::make_unique<rocprofiler::hsa::EmptyAQLPacket>();
        // If we have a counter collection context but it is not enabled, we still might need
        // to add barrier packets to transition from serialized -> unserialized execution. This
        // transition is coordinated by the serializer.
        maybe_add_serialization(ret_pkt);
        info->packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), nullptr); });
        return ret_pkt;
    };

    if(!ctx || !ctx->counter_collection) return nullptr;

    bool is_enabled = false;

    ctx->counter_collection->enabled.rlock(
        [&](const auto& collect_ctx) { is_enabled = collect_ctx; });

    if(!is_enabled || !info->user_cb)
    {
        return no_instrumentation();
    }

    auto _corr_id_v =
        rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};
    if(const auto* _corr_id = correlation_id)
    {
        _corr_id_v.internal = _corr_id->internal;
        if(const auto* external =
               rocprofiler::common::get_val(extern_corr_ids, info->internal_context))
        {
            _corr_id_v.external = *external;
        }
    }

    auto req_profile = rocprofiler_profile_config_id_t{.handle = 0};
    auto dispatch_data =
        common::init_public_api_struct(rocprofiler_dispatch_counting_service_data_t{});

    dispatch_data.correlation_id = _corr_id_v;
    {
        auto dispatch_info = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
        dispatch_info.kernel_id            = kernel_id;
        dispatch_info.dispatch_id          = dispatch_id;
        dispatch_info.agent_id             = CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id;
        dispatch_info.queue_id             = queue.get_id();
        dispatch_info.private_segment_size = pkt.kernel_dispatch.private_segment_size;
        dispatch_info.group_segment_size   = pkt.kernel_dispatch.group_segment_size;
        dispatch_info.workgroup_size       = {pkt.kernel_dispatch.workgroup_size_x,
                                        pkt.kernel_dispatch.workgroup_size_y,
                                        pkt.kernel_dispatch.workgroup_size_z};
        dispatch_info.grid_size            = {pkt.kernel_dispatch.grid_size_x,
                                   pkt.kernel_dispatch.grid_size_y,
                                   pkt.kernel_dispatch.grid_size_z};
        dispatch_data.dispatch_info        = dispatch_info;
    }

    info->user_cb(dispatch_data, &req_profile, user_data, info->callback_args);

    if(req_profile.handle == 0)
    {
        return no_instrumentation();
    }

    auto prof_config = get_controller().get_profile_cfg(req_profile);
    CHECK(prof_config);

    std::unique_ptr<rocprofiler::hsa::AQLPacket> ret_pkt;
    auto                                         status = info->get_packet(ret_pkt, prof_config);
    CHECK_EQ(status, ROCPROFILER_STATUS_SUCCESS) << rocprofiler_get_status_string(status);

    maybe_add_serialization(ret_pkt);
    if(ret_pkt->empty)
    {
        return ret_pkt;
    }

    ret_pkt->populate_before();
    ret_pkt->populate_after();
    for(auto& aql_pkt : ret_pkt->after_krn_pkt)
        aql_pkt.completion_signal.handle = 0;

    return ret_pkt;
}

/**
 * Callback called by HSA interceptor when the kernel has completed processing.
 */
void
completed_cb(const context::context*                       ctx,
             const std::shared_ptr<counter_callback_info>& info,
             const hsa::Queue& /*queue*/,
             hsa::rocprofiler_packet /*packet*/,
             const hsa::Queue::queue_info_session_t& session,
             inst_pkt_t&                             pkts,
             kernel_dispatch::profiling_time         dispatch_time)
{
    CHECK(info && ctx);

    std::shared_ptr<profile_config> prof_config;
    // Get the Profile Config
    std::unique_ptr<rocprofiler::hsa::AQLPacket> pkt = nullptr;
    info->packet_return_map.wlock([&](auto& data) {
        for(auto& [aql_pkt, _] : pkts)
        {
            const auto* profile = rocprofiler::common::get_val(data, aql_pkt.get());
            if(profile)
            {
                prof_config = *profile;
                data.erase(aql_pkt.get());
                pkt = std::move(aql_pkt);
                return;
            }
        }
    });

    if(!pkt) return;

    CHECK_NOTNULL(hsa::get_queue_controller())
        ->serializer(&session.queue)
        .wlock([&](auto& serializer) { serializer.kernel_completion_signal(session.queue); });

    // We have no profile config, nothing to output.
    if(!prof_config) return;

    auto decoded_pkt = EvaluateAST::read_pkt(prof_config->pkt_generator.get(), *pkt);
    EvaluateAST::read_special_counters(
        *prof_config->agent, prof_config->required_special_counters, decoded_pkt);

    prof_config->packets.wlock([&](auto& pkt_vector) {
        if(pkt)
        {
            pkt_vector.emplace_back(std::move(pkt));
        }
    });

    common::container::small_vector<rocprofiler_record_counter_t, 128> out;
    rocprofiler::buffer::instance*                                     buf = nullptr;

    if(info->buffer)
    {
        buf = CHECK_NOTNULL(buffer::get_buffer(info->buffer->handle));
    }

    auto _corr_id_v =
        rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};
    if(const auto* _corr_id = session.correlation_id)
    {
        _corr_id_v.internal = _corr_id->internal;
        if(const auto* external = rocprofiler::common::get_val(
               session.tracing_data.external_correlation_ids, info->internal_context))
        {
            _corr_id_v.external = *external;
        }
    }

    auto _dispatch_id = session.callback_record.dispatch_info.dispatch_id;
    for(auto& ast : prof_config->asts)
    {
        std::vector<std::unique_ptr<std::vector<rocprofiler_record_counter_t>>> cache;
        auto* ret = ast.evaluate(decoded_pkt, cache);
        CHECK(ret);
        ast.set_out_id(*ret);

        out.reserve(out.size() + ret->size());
        for(auto& val : *ret)
        {
            val.agent_id    = prof_config->agent->id;
            val.dispatch_id = _dispatch_id;
            out.emplace_back(val);
        }
    }

    if(!out.empty())
    {
        if(buf)
        {
            auto _header =
                common::init_public_api_struct(rocprofiler_dispatch_counting_service_record_t{});
            _header.num_records    = out.size();
            _header.correlation_id = _corr_id_v;
            if(dispatch_time.status == HSA_STATUS_SUCCESS)
            {
                _header.start_timestamp = dispatch_time.start;
                _header.end_timestamp   = dispatch_time.end;
            }
            _header.dispatch_info = session.callback_record.dispatch_info;
            buf->emplace(ROCPROFILER_BUFFER_CATEGORY_COUNTERS,
                         ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER,
                         _header);

            for(auto itr : out)
                buf->emplace(
                    ROCPROFILER_BUFFER_CATEGORY_COUNTERS, ROCPROFILER_COUNTER_RECORD_VALUE, itr);
        }
        else
        {
            CHECK(info->record_callback);

            auto dispatch_data =
                common::init_public_api_struct(rocprofiler_dispatch_counting_service_data_t{});

            dispatch_data.dispatch_info  = session.callback_record.dispatch_info;
            dispatch_data.correlation_id = _corr_id_v;
            if(dispatch_time.status == HSA_STATUS_SUCCESS)
            {
                dispatch_data.start_timestamp = dispatch_time.start;
                dispatch_data.end_timestamp   = dispatch_time.end;
            }

            info->record_callback(dispatch_data,
                                  out.data(),
                                  out.size(),
                                  session.user_data,
                                  info->record_callback_args);
        }
    }
}
}  // namespace counters
}  // namespace rocprofiler
