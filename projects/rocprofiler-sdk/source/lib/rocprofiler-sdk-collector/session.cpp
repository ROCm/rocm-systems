// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk-collector/session.hpp"

#include "lib/common/logging.hpp"

#include <lttng/lttng.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace rocprofiler
{
namespace collector
{
namespace
{
constexpr auto channel_name = "rocprofiler_collector";

lttng_domain
ust_domain()
{
    auto domain = lttng_domain{};
    std::memset(&domain, 0, sizeof(domain));
    domain.type     = LTTNG_DOMAIN_UST;
    domain.buf_type = LTTNG_BUFFER_PER_PID;
    return domain;
}

const char*
error_string(int code)
{
    return ::lttng_strerror(code < 0 ? code : -code);
}

struct handle_deleter
{
    void operator()(lttng_handle* handle) const { ::lttng_destroy_handle(handle); }
};

struct channel_deleter
{
    void operator()(lttng_channel* channel) const { ::lttng_channel_destroy(channel); }
};

struct event_deleter
{
    void operator()(lttng_event* event) const { ::lttng_event_destroy(event); }
};

struct descriptor_deleter
{
    void operator()(lttng_session_descriptor* desc) const
    {
        ::lttng_session_descriptor_destroy(desc);
    }
};

rocprofiler_collector_status_t
configure_session(const trace_session_config& config)
{
    auto domain = ust_domain();
    auto handle = std::unique_ptr<lttng_handle, handle_deleter>{
        ::lttng_create_handle(config.name.c_str(), &domain)};
    if(!handle)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] lttng_create_handle failed";
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
    }

    auto channel = std::unique_ptr<lttng_channel, channel_deleter>{::lttng_channel_create(&domain)};
    if(!channel)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] lttng_channel_create failed";
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
    }
    std::strncpy(channel->name, channel_name, sizeof(channel->name) - 1);
    channel->enabled        = 1;
    channel->attr.overwrite = 0;  // a full buffer drops new events and counts them
    channel->attr.output    = LTTNG_EVENT_MMAP;
    if(config.subbuffer_size > 0) channel->attr.subbuf_size = config.subbuffer_size;
    if(config.num_subbuffers > 0) channel->attr.num_subbuf = config.num_subbuffers;
    // Writers wait for space first, if the target allows it (LTTNG_UST_ALLOW_BLOCKING).
    if(config.blocking_timeout_us != 0)
    {
        if(auto ret =
               ::lttng_channel_set_blocking_timeout(channel.get(), config.blocking_timeout_us);
           ret < 0)
        {
            ROCP_ERROR << "[rocprofiler-sdk-collector] setting the blocking timeout failed: "
                       << error_string(ret);
            return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
        }
    }

    if(auto ret = ::lttng_enable_channel(handle.get(), channel.get()); ret < 0)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] enabling channel " << channel_name
                   << " failed: " << error_string(ret);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
    }

    for(const auto& provider : config.providers)
    {
        auto event = std::unique_ptr<lttng_event, event_deleter>{::lttng_event_create()};
        if(!event) return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
        auto pattern = provider + ":*";
        if(pattern.size() >= sizeof(event->name))
        {
            ROCP_ERROR << "[rocprofiler-sdk-collector] provider name too long: " << provider;
            return ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_ARGUMENT;
        }
        std::strncpy(event->name, pattern.c_str(), sizeof(event->name) - 1);
        event->type          = LTTNG_EVENT_TRACEPOINT;
        event->loglevel_type = LTTNG_EVENT_LOGLEVEL_ALL;
        if(auto ret = ::lttng_enable_event(handle.get(), event.get(), channel_name); ret < 0)
        {
            ROCP_ERROR << "[rocprofiler-sdk-collector] enabling " << pattern
                       << " failed: " << error_string(ret);
            return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
        }
    }

    // The join pairs events per process and thread.
    for(auto type : {LTTNG_EVENT_CONTEXT_VPID, LTTNG_EVENT_CONTEXT_VTID})
    {
        auto context = lttng_event_context{};
        std::memset(&context, 0, sizeof(context));
        context.ctx = type;
        if(auto ret = ::lttng_add_context(handle.get(), &context, nullptr, channel_name); ret < 0)
        {
            ROCP_ERROR << "[rocprofiler-sdk-collector] adding context " << type
                       << " failed: " << error_string(ret);
            return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
        }
    }
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}
}  // namespace

rocprofiler_collector_status_t
create_trace_session(const trace_session_config& config, trace_session* session)
{
    auto desc = std::unique_ptr<lttng_session_descriptor, descriptor_deleter>{
        ::lttng_session_descriptor_local_create(config.name.c_str(), config.trace_path.c_str())};
    if(!desc)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] invalid session name or trace path: "
                   << config.name << ", " << config.trace_path;
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
    }
    if(auto ret = ::lttng_create_session_ext(desc.get()); ret != LTTNG_OK)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] creating session " << config.name
                   << " failed: " << error_string(ret);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
    }
    session->name    = config.name;
    session->created = true;
    session->started = false;

    auto status = configure_session(config);
    if(status != ROCPROFILER_COLLECTOR_STATUS_SUCCESS) destroy_trace_session(session);
    return status;
}

rocprofiler_collector_status_t
start_trace_session(trace_session* session)
{
    if(auto ret = ::lttng_start_tracing(session->name.c_str()); ret < 0)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] starting session " << session->name
                   << " failed: " << error_string(ret);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
    }
    session->started = true;
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

rocprofiler_collector_status_t
stop_trace_session(trace_session* session)
{
    // lttng_stop_tracing waits until the consumer daemon has written the buffers.
    if(auto ret = ::lttng_stop_tracing(session->name.c_str()); ret < 0)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] stopping session " << session->name
                   << " failed: " << error_string(ret);
        return ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
    }
    session->started = false;
    return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
}

rocprofiler_collector_status_t
destroy_trace_session(trace_session* session)
{
    if(!session->created) return ROCPROFILER_COLLECTOR_STATUS_SUCCESS;

    auto status = ROCPROFILER_COLLECTOR_STATUS_SUCCESS;
    if(session->started) status = stop_trace_session(session);
    if(auto ret = ::lttng_destroy_session(session->name.c_str()); ret < 0)
    {
        ROCP_ERROR << "[rocprofiler-sdk-collector] destroying session " << session->name
                   << " failed: " << error_string(ret);
        status = ROCPROFILER_COLLECTOR_STATUS_ERROR_SESSION;
    }
    session->created = false;
    session->started = false;
    return status;
}
}  // namespace collector
}  // namespace rocprofiler
