

// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/sample_processing.hpp"

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/sample_consumer.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace counters
{
std::mutex&
get_buffer_mut()
{
    static auto*& mut = common::static_object<std::mutex>::construct();
    return *CHECK_NOTNULL(mut);
}

/**
 * Callback called by HSA interceptor when the kernel has completed processing.
 */
void
proccess_completed_cb(completed_cb_params_t&& params)
{
    auto& info          = params.info;
    auto& session       = *params.session;
    auto& dispatch_time = params.dispatch_time;
    auto& prof_config   = params.prof_config;
    auto& pkt           = params.pkt;

    ROCP_FATAL_IF(pkt == nullptr) << "AQL packet is a nullptr!";

    auto decoded_pkt = EvaluateAST::read_pkt(prof_config->pkt_generator.get(), *pkt);
    EvaluateAST::read_special_counters(
        *prof_config->agent, prof_config->required_special_counters, decoded_pkt);

    prof_config->packets.wlock([&](auto& pkt_vector) { pkt_vector.emplace_back(std::move(pkt)); });

    common::container::small_vector<rocprofiler_counter_record_t, 128> out;
    rocprofiler::buffer::instance*                                     buf = nullptr;

    if(info->buffer)
    {
        buf = CHECK_NOTNULL(buffer::get_buffer(info->buffer->handle));
    }

    auto _corr_id_v =
        rocprofiler_async_correlation_id_t{.internal = 0, .external = context::null_user_data};

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
        std::vector<std::unique_ptr<std::vector<rocprofiler_counter_record_t>>> cache;
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

            auto _lk = std::unique_lock{get_buffer_mut()};  // Buffer records need to be in order

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

auto&
callback_thread_get()
{
    using consumer_t = consumer_thread_t<completed_cb_params_t>;
    static auto*& _v = common::static_object<consumer_t>::construct(proccess_completed_cb);
    return *CHECK_NOTNULL(_v);
}

void
callback_thread_start()
{
    callback_thread_get().start();
}

void
callback_thread_stop()
{
    callback_thread_get().exit();
}

void
process_callback_data(completed_cb_params_t&& params)
{
    callback_thread_get().add(std::move(params));
}

}  // namespace counters
}  // namespace rocprofiler
