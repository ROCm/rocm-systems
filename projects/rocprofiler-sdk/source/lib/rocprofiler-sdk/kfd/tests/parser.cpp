// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <string_view>

#include "lib/rocprofiler-sdk/kfd/defines.hpp"
#include "lib/rocprofiler-sdk/kfd/utils.hpp"
#include "rocprofiler-sdk/kfd/kfd_id.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <unistd.h>

namespace rocprofiler
{
namespace kfd
{
#define ROCPROFILER_LIB_ROCPROFILER_SDK_KFD_CPP_IMPL 1
#include "lib/rocprofiler-sdk/kfd/kfd.def.cpp"
#undef ROCPROFILER_LIB_ROCPROFILER_SDK_KFD_CPP_IMPL

}  // namespace kfd
}  // namespace rocprofiler

namespace
{
using namespace rocprofiler::kfd;

// Internally, get_node_map is called for a given gpu ID
// We are testing the parser, so use an identity mapping
const agent_id_map_t agent_map = {
    {0, {0}},
    {0x1ea0, {0x1ea0}},
    {0xfc7d, {0xfc7d}},
    {0xad5c, {0xad5c}},
};

kfd_event_record
parse(std::string_view arg)
{
    size_t kfd_id = std::numeric_limits<size_t>::max();

    const auto scan_count = std::sscanf(arg.data(), "%lx ", &kfd_id);

    EXPECT_EQ(scan_count, 1);
    EXPECT_GE(kfd_id, KFD_SMI_EVENT_MIGRATE_START);
    EXPECT_LE(kfd_id, KFD_SMI_EVENT_DROPPED_EVENT);

    auto event_id = to_rocprofiler_kfd_event_id_func(static_cast<rocprofiler_kfd_event_id>(kfd_id),
                                                     std::make_index_sequence<KFD_EVENT_LAST>{});

    EXPECT_GE(event_id, KFD_EVENT_PAGE_MIGRATE_START);
    EXPECT_LE(event_id, KFD_EVENT_DROPPED_EVENT);

    return parse_event(event_id, agent_map, arg);
}

}  // namespace

/* The following tests ensure that we can correctly parse and convert an event to the structured
   equivalent
*/

TEST(rocprofiler_lib, parse_kfd_event_page_migrate_start)
{
    using namespace rocprofiler::kfd;

    const auto event         = parse("5 14601738586508 -152990 @7f0f15200(2c02) 0->1ea0 1ea0:0 1");
    const auto migrate_start = event.data.page_migrate_event;

    EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE);
    EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_GPU);
    EXPECT_EQ(migrate_start.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE);
    EXPECT_EQ(migrate_start.operation, ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_GPU);
    EXPECT_EQ(migrate_start.timestamp, 14601738586508);
    EXPECT_EQ(migrate_start.pid, 152990);
    EXPECT_EQ(migrate_start.start_address.handle, 0x7f0f15200 << 12);
    EXPECT_EQ(migrate_start.end_address.handle, (0x7f0f15200 + 0x2c02) << 12);
    EXPECT_EQ(migrate_start.src_agent.handle, 0);
    EXPECT_EQ(migrate_start.dst_agent.handle, 0x1ea0);
    EXPECT_EQ(migrate_start.prefetch_agent.handle, 0x1ea0);
    EXPECT_EQ(migrate_start.preferred_agent.handle, 0);
    EXPECT_EQ(migrate_start.error_code, 0);
}

TEST(rocprofiler_lib, parse_kfd_event_page_migrate_end)
{
    using namespace rocprofiler::kfd;

    const auto event       = parse("6 15202910515905 -152990 @7f0f15200(2c02) 0->1ea0 1 75");
    const auto migrate_end = event.data.page_migrate_event;

    EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE);
    EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_END);
    EXPECT_EQ(migrate_end.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE);
    EXPECT_EQ(migrate_end.operation, ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_END);
    EXPECT_EQ(migrate_end.timestamp, 15202910515905);
    EXPECT_EQ(migrate_end.pid, 152990);
    EXPECT_EQ(migrate_end.start_address.handle, 0x7f0f15200 << 12);
    EXPECT_EQ(migrate_end.end_address.handle, (0x7f0f15200 + 0x2c02) << 12);
    EXPECT_EQ(migrate_end.src_agent.handle, 0);
    EXPECT_EQ(migrate_end.dst_agent.handle, 0x1ea0);
    EXPECT_EQ(migrate_end.prefetch_agent.handle, 0);   // Not generated for end event
    EXPECT_EQ(migrate_end.preferred_agent.handle, 0);  // Not generated for end event
    EXPECT_EQ(migrate_end.error_code, 75);
}

TEST(rocprofiler_lib, parse_kfd_event_page_fault_start)
{
    using namespace rocprofiler::kfd;

    const auto event      = parse("7 40507688414784 -156893 @7fa608127(ad5c) R");
    const auto page_fault = event.data.page_fault_event;

    EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT);
    EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_READ_FAULT);
    EXPECT_EQ(page_fault.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT);
    EXPECT_EQ(page_fault.operation, ROCPROFILER_KFD_EVENT_PAGE_FAULT_START_READ_FAULT);
    EXPECT_EQ(page_fault.timestamp, 40507688414784);
    EXPECT_EQ(page_fault.pid, 156893);
    EXPECT_EQ(page_fault.agent_id.handle, 0xad5c);
    EXPECT_EQ(page_fault.address.handle, 0x7fa608127 << 12);
}

TEST(rocprofiler_lib, parse_kfd_event_page_fault_end)
{
    using namespace rocprofiler::kfd;

    {
        const auto event      = parse("8 40507688514376 -156893 @7fa608127(ad5c) U");
        const auto page_fault = event.data.page_fault_event;

        EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT);
        EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_UPDATED);
        EXPECT_EQ(page_fault.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT);
        EXPECT_EQ(page_fault.operation, ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_UPDATED);
        EXPECT_EQ(page_fault.timestamp, 40507688514376);
        EXPECT_EQ(page_fault.pid, 156893);
        EXPECT_EQ(page_fault.agent_id.handle, 0xad5c);
        EXPECT_EQ(page_fault.address.handle, 0x7fa608127 << 12);
    }

    {
        const auto event      = parse("8 40507688516386 -156898 @7fa698127(ad5c) M");
        const auto page_fault = event.data.page_fault_event;

        EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT);
        EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_MIGRATED);
        EXPECT_EQ(page_fault.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT);
        EXPECT_EQ(page_fault.operation, ROCPROFILER_KFD_EVENT_PAGE_FAULT_END_PAGE_MIGRATED);
        EXPECT_EQ(page_fault.timestamp, 40507688516386);
        EXPECT_EQ(page_fault.pid, 156898);
        EXPECT_EQ(page_fault.agent_id.handle, 0xad5c);
        EXPECT_EQ(page_fault.address.handle, 0x7fa698127 << 12);
    }
}

TEST(rocprofiler_lib, parse_kfd_event_queue_eviction)
{
    using namespace rocprofiler::kfd;

    {
        const auto event = parse("9 38086928279363 -125752 1ea0 1");

        const auto queue_event = event.data.queue_event;
        EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
        EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR);
        EXPECT_EQ(queue_event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
        EXPECT_EQ(queue_event.operation, ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR);
        EXPECT_EQ(queue_event.timestamp, 38086928279363);
        EXPECT_EQ(queue_event.pid, 125752);
        EXPECT_EQ(queue_event.agent_id.handle, 0x1ea0);
    }
    {
        const auto event = parse("9 38086928279363 -125752 1ea0 3");

        const auto queue_event = event.data.queue_event;
        EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
        EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND);
        EXPECT_EQ(queue_event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
        EXPECT_EQ(queue_event.operation, ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND);
        EXPECT_EQ(queue_event.timestamp, 38086928279363);
        EXPECT_EQ(queue_event.pid, 125752);
        EXPECT_EQ(queue_event.agent_id.handle, 0x1ea0);
    }
}

TEST(rocprofiler_lib, parse_kfd_event_queue_restore)
{
    using namespace rocprofiler::kfd;
    {
        const auto event = parse("a 38652512365099 -131057 fc7d");

        const auto queue_event = event.data.queue_event;
        EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
        EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_QUEUE_RESTORE);
        EXPECT_EQ(queue_event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
        EXPECT_EQ(queue_event.operation, ROCPROFILER_KFD_EVENT_QUEUE_RESTORE);
        EXPECT_EQ(queue_event.timestamp, 38652512365099);
        EXPECT_EQ(queue_event.pid, 131057);
        EXPECT_EQ(queue_event.agent_id.handle, 0xfc7d);
    }

    {
        const auto event = parse("a 40082605896929 -148516 fc7d R");

        const auto queue_event = event.data.queue_event;
        EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
        EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED);
        EXPECT_EQ(queue_event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE);
        EXPECT_EQ(queue_event.operation, ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED);
        EXPECT_EQ(queue_event.timestamp, 40082605896929);
        EXPECT_EQ(queue_event.pid, 148516);
        EXPECT_EQ(queue_event.agent_id.handle, 0xfc7d);
    }
}

TEST(rocprofiler_lib, parse_kfd_event_unmap_from_gpu)
{
    using namespace rocprofiler::kfd;

    const auto event = parse("b 15203214457186 -152990 @7f93ac200(200) 1ea0 2");
    const auto unmap = event.data.unmap_event;

    EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU);
    EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM);
    EXPECT_EQ(unmap.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU);
    EXPECT_EQ(unmap.operation, ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM);
    EXPECT_EQ(unmap.timestamp, 15203214457186);
    EXPECT_EQ(unmap.pid, 152990);
    EXPECT_EQ(unmap.start_address.handle, 0x7f93ac200 << 12);
    EXPECT_EQ(unmap.end_address.handle, (0x7f93ac200 + 0x200) << 12);
    EXPECT_EQ(unmap.agent_id.handle, 0x1ea0);
}

TEST(rocprofiler_lib, parse_kfd_event_dropped_event)
{
    using namespace rocprofiler::kfd;

    const auto event   = parse("e 18203445217186 -323990 5");
    const auto dropped = event.data.dropped_event;

    EXPECT_EQ(event.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS);
    EXPECT_EQ(event.operation, ROCPROFILER_KFD_EVENT_DROPPED_EVENTS);
    EXPECT_EQ(dropped.kind, ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS);
    EXPECT_EQ(dropped.operation, ROCPROFILER_KFD_EVENT_DROPPED_EVENTS);
    EXPECT_EQ(dropped.timestamp, 18203445217186);
    EXPECT_EQ(dropped.pid, 323990);
    EXPECT_EQ(dropped.count, 5);
}
