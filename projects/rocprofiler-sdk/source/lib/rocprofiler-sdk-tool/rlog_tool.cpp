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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "rlog_tool.hpp"
#include "lib/output/domain_type.hpp"
#include "lib/output/tmp_file_buffer.hpp"

#include <rlog/Logger.h>
#include <Hub.h>

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <stack>
#include <string>

namespace
{
std::string
make_message(const char* domain, const char* category, const char* apiname, const char* args)
{
    std::string msg;

    bool has_domain   = domain && domain[0] != '\0';
    bool has_category = category && category[0] != '\0';

    if(has_domain || has_category)
    {
        msg += '[';
        if(has_domain) msg += domain;
        if(has_domain && has_category) msg += '/';
        if(has_category) msg += category;
        msg += "] ";
    }

    if(apiname && apiname[0] != '\0') msg += apiname;

    if(args && args[0] != '\0')
    {
        msg += ": ";
        msg += args;
    }

    return msg;
}

uint64_t
next_synthetic_id()
{
    // High bit set to avoid colliding with SDK-assigned correlation ids.
    static std::atomic<uint64_t> counter{1ULL << 48};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

struct RangeEntry
{
    rocprofiler_timestamp_t ts;
    uint64_t                corr_internal;
};

thread_local std::stack<RangeEntry> t_range_stack;

class RlogToolBridge : public ::rlog::Logger
{
public:
    void mark(const char* domain,
              const char* category,
              const char* apiname,
              const char* args) override
    {
        auto ts  = rocprofiler_timestamp_t{};
        auto tid = rocprofiler_thread_id_t{};
        rocprofiler_get_timestamp(&ts);
        rocprofiler_get_thread_id(&tid);

        auto corr_id           = rocprofiler_correlation_id_t{};
        corr_id.internal       = next_synthetic_id();
        corr_id.external.value = 0;

        ::add_rlog_marker_message(corr_id.internal,
                                  make_message(domain, category, apiname, args));

        auto rec            = rocprofiler_buffer_tracing_marker_api_record_t{};
        rec.size            = sizeof(rocprofiler_buffer_tracing_marker_api_record_t);
        rec.kind            = ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API;
        rec.operation       = ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxMarkA;
        rec.thread_id       = tid;
        rec.correlation_id  = corr_id;
        rec.start_timestamp = ts;
        rec.end_timestamp   = ts;
        ::rocprofiler::tool::write_ring_buffer(rec, domain_type::MARKER);
    }

    void rangePush(const char* domain,
                   const char* category,
                   const char* apiname,
                   const char* args) override
    {
        auto ts  = rocprofiler_timestamp_t{};
        auto tid = rocprofiler_thread_id_t{};
        rocprofiler_get_timestamp(&ts);
        rocprofiler_get_thread_id(&tid);
        (void) tid;

        auto corr_id           = rocprofiler_correlation_id_t{};
        corr_id.internal       = next_synthetic_id();
        corr_id.external.value = 0;

        ::add_rlog_marker_message(corr_id.internal,
                                  make_message(domain, category, apiname, args));

        t_range_stack.push({ts, corr_id.internal});
    }

    void rangePop() override
    {
        if(t_range_stack.empty()) return;

        auto entry = t_range_stack.top();
        t_range_stack.pop();

        auto ts  = rocprofiler_timestamp_t{};
        auto tid = rocprofiler_thread_id_t{};
        rocprofiler_get_timestamp(&ts);
        rocprofiler_get_thread_id(&tid);

        auto corr_id           = rocprofiler_correlation_id_t{};
        corr_id.internal       = entry.corr_internal;
        corr_id.external.value = 0;

        auto rec            = rocprofiler_buffer_tracing_marker_api_record_t{};
        rec.size            = sizeof(rocprofiler_buffer_tracing_marker_api_record_t);
        rec.kind            = ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API;
        rec.operation       = ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxThreadRangeA;
        rec.thread_id       = tid;
        rec.correlation_id  = corr_id;
        rec.start_timestamp = entry.ts;
        rec.end_timestamp   = ts;
        ::rocprofiler::tool::write_ring_buffer(rec, domain_type::MARKER);
    }
};

RlogToolBridge&
get_bridge()
{
    // Intentionally never destroyed: vtable lives in librlog.so; destructor ordering unsafe.
    static RlogToolBridge* instance = new RlogToolBridge{};
    return *instance;
}
}  // namespace

namespace rocprofiler
{
namespace tool
{
namespace rlog
{
void
enable()
{
    ::rlog::Hub::singleton().addLogger(get_bridge());
}

void
disable()
{
    ::rlog::Hub::singleton().removeLogger(get_bridge());
}
}  // namespace rlog
}  // namespace tool
}  // namespace rocprofiler
