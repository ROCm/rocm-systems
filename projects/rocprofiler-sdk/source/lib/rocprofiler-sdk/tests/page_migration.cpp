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

#include "lib/common/container/small_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/mpl.hpp"
#include "lib/rocprofiler-sdk/page_migration/utils.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <sstream>
#include <string_view>
#include <utility>

#define ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL 1
#include "lib/rocprofiler-sdk/page_migration/page_migration.def.cpp"
#undef ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL

namespace
{
constexpr std::string_view MULTILINE_STRING = "This is 0 Line 0\n"
                                              "This is 10 Line 1\n"
                                              "This is 20 Line 2\n"
                                              "This is 30 Line 3\n"
                                              "This is 40 Line 4\n";
}

void
return_line(const std::string_view line)
{
    static int        line_no = 0;
    std::stringstream strs{};
    strs << fmt::format("This is {} Line {}", line_no * 10, line_no);
    EXPECT_EQ(strs.str(), line);
    line_no++;
}

auto
parse_lines()
{
    rocprofiler::page_migration::kfd_readlines(MULTILINE_STRING, return_line);
}

TEST(page_migration, readlines)
{
    // Ensure all lines are read
    parse_lines();
}

TEST(page_migration, rocprof_kfd_map)
{
    using namespace rocprofiler::page_migration;
    using namespace rocprofiler::common::container;

    using rocprofiler_page_migration_seq_t =
        std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>;

    const small_vector<size_t> vec{ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_END,
                                   ROCPROFILER_PAGE_MIGRATION_QUEUE_EVICTION,
                                   ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU};

    EXPECT_EQ((page_migration_info<ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_END>::kfd_bitmask |
               page_migration_info<ROCPROFILER_PAGE_MIGRATION_QUEUE_EVICTION>::kfd_bitmask |
               page_migration_info<ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU>::kfd_bitmask),
              kfd_bitmask(vec, rocprofiler_page_migration_seq_t{}));

    const auto to_kfd_str = [](kfd_smi_event e) {
        std::string str = fmt::format("{:x} ", static_cast<size_t>(e));
        return rocprofiler::page_migration::get_rocprof_op({str});
    };

    // clang-format off
    EXPECT_EQ(to_kfd_str(KFD_SMI_EVENT_MIGRATE_START),    ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_START);
    EXPECT_EQ(to_kfd_str(KFD_SMI_EVENT_MIGRATE_END),      ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_END);
    EXPECT_EQ(to_kfd_str(KFD_SMI_EVENT_PAGE_FAULT_START), ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_START);
    EXPECT_EQ(to_kfd_str(KFD_SMI_EVENT_PAGE_FAULT_END),   ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_END);
    EXPECT_EQ(to_kfd_str(KFD_SMI_EVENT_QUEUE_EVICTION),   ROCPROFILER_PAGE_MIGRATION_QUEUE_EVICTION);
    EXPECT_EQ(to_kfd_str(KFD_SMI_EVENT_QUEUE_RESTORE),    ROCPROFILER_PAGE_MIGRATION_QUEUE_RESTORE);
    EXPECT_EQ(to_kfd_str(KFD_SMI_EVENT_UNMAP_FROM_GPU),   ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU);
    EXPECT_EQ(to_kfd_str(KFD_SMI_EVENT_DROPPED_EVENT),    ROCPROFILER_PAGE_MIGRATION_DROPPED_EVENT);
    // clang-format on
}
