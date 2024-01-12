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

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <tuple>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "rocprofiler-sdk/registration.h"

using namespace rocprofiler::counters;

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

struct metric_map_order
{
    metric_map_order() = default;
    ~metric_map_order() { check_copy(); }

    metric_map_order(const metric_map_order&) = delete;
    metric_map_order& operator=(const metric_map_order&) = delete;

    metric_map_order(metric_map_order&&) noexcept = delete;
    metric_map_order& operator=(metric_map_order&&) noexcept = delete;

    void check_copy()
    {
        ASSERT_TRUE(!copy_.empty());

        const auto* metricIdMap = rocprofiler::counters::getMetricIdMap();
        int         fini_status = 0;
        ROCPROFILER_CALL(rocprofiler_is_finalized(&fini_status), "get finalization state");

        if(fini_status > 0)
        {
            // this should only be true in the destructor of the static metric_map_order instance
            ASSERT_TRUE(metricIdMap != nullptr) << "rocprofiler finalization state: " << fini_status
                                                << ", metricIdMap: " << metricIdMap;

            // this should ensure the metric id map is destroyed
            rocprofiler::common::destroy_static_objects();
            metricIdMap = rocprofiler::counters::getMetricIdMap();

            ASSERT_TRUE(metricIdMap == nullptr) << "rocprofiler finalization state: " << fini_status
                                                << ", metricIdMap: " << metricIdMap;
        }
        else
        {
            for(const auto& [id, actual] : copy_)
            {
                // Assert because this is getting triggered on shutdown and
                // we want to fail the test if the values in both maps are not equal.
                const auto* val = rocprofiler::common::get_val(*metricIdMap, id);
                ASSERT_TRUE(val != nullptr) << "metricIdMap: " << metricIdMap;
                ASSERT_TRUE(*val == actual) << "metricIdMap: " << metricIdMap;
            }
        }
    }

private:
    MetricIdMap copy_ = *CHECK_NOTNULL(rocprofiler::counters::getMetricIdMap());
};

metric_map_order&
get_metric_map()
{
    static metric_map_order order = {};
    return order;
}

void
buffered_callback(rocprofiler_context_id_t,
                  rocprofiler_buffer_id_t,
                  rocprofiler_record_header_t**,
                  size_t,
                  void*,
                  uint64_t)
{}

void
dispatch_callback(rocprofiler_queue_id_t,
                  const rocprofiler_agent_t*,
                  rocprofiler_correlation_id_t,
                  const hsa_kernel_dispatch_packet_t*,
                  uint64_t,
                  void*,
                  rocprofiler_profile_config_id_t*)
{}

rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx;
    return ctx;
}

rocprofiler_buffer_id_t&
get_buffer()
{
    static rocprofiler_buffer_id_t buf = {};
    return buf;
}

// Test that metrics map remains in scope at exit
TEST(counters_init_order, metric_map_order)
{
    rocprofiler::registration::init_logging();
    // do not call rocprofiler::registration::initialize()!
    // doing so will add an atexit call which might invoke
    // rocprofiler::common::destroy_static_objects() before
    // the get_metric_map() instance is destroyed

    rocprofiler::registration::set_init_status(-1);
    rocprofiler::context::push_client(1);
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               4096,
                                               2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               buffered_callback,
                                               nullptr,
                                               &get_buffer()),
                     "buffer creation failed");
    ROCPROFILER_CALL(rocprofiler_configure_buffered_dispatch_profile_counting_service(
                         get_client_ctx(), get_buffer(), dispatch_callback, nullptr),
                     "Could not setup buffered service");
    rocprofiler::registration::set_init_status(1);

    auto& global_metric_map = get_metric_map();
    global_metric_map.check_copy();

    auto local_metric_map = metric_map_order{};
    local_metric_map.check_copy();

    rocprofiler::registration::finalize();
}
