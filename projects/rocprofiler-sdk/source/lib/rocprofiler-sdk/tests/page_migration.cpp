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

#include "lib/common/defines.hpp"
#include "lib/rocprofiler-sdk/page_migration/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/page_migration/utils.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <sstream>
#include <string_view>

#define ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL 1
#include "lib/rocprofiler-sdk/page_migration/page_migration.def.cpp"
#undef ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL

#define ASSERT_SAME(A, B) static_assert(static_cast<size_t>(A) == static_cast<size_t>(B))

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
    KFD_EVENT_PARSE_EVENTS(MULTILINE_STRING, return_line);
}

TEST(page_migration, readlines)
{
    // Ensure all lines are read
    parse_lines();
}

TEST(page_migration, parse_kvm_events)
{
    // Ensure all lines are read
    parse_lines();
}

TEST(page_migtation, rocprof_kfd_map)
{
    using namespace ::rocprofiler::page_migration;

    // clang-format off
    static_assert(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT,      ROCPROFILER_UVM_EVENT_PAGE_FAULT_START >() );
    static_assert(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT,      ROCPROFILER_UVM_EVENT_PAGE_FAULT_END   >() );
    static_assert(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE,    ROCPROFILER_UVM_EVENT_MIGRATE_START    >() );
    static_assert(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE,    ROCPROFILER_UVM_EVENT_MIGRATE_END      >() );
    static_assert(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND,   ROCPROFILER_UVM_EVENT_QUEUE_EVICTION   >() );
    static_assert(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND,   ROCPROFILER_UVM_EVENT_QUEUE_RESTORE    >() );
    static_assert(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU,  ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU   >() );

    EXPECT_TRUE(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT         >(ROCPROFILER_UVM_EVENT_PAGE_FAULT_START) );
    EXPECT_TRUE(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT         >(ROCPROFILER_UVM_EVENT_PAGE_FAULT_END  ) );
    EXPECT_TRUE(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE       >(ROCPROFILER_UVM_EVENT_MIGRATE_START   ) );
    EXPECT_TRUE(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE       >(ROCPROFILER_UVM_EVENT_MIGRATE_END     ) );
    EXPECT_TRUE(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND      >(ROCPROFILER_UVM_EVENT_QUEUE_EVICTION  ) );
    EXPECT_TRUE(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND      >(ROCPROFILER_UVM_EVENT_QUEUE_RESTORE   ) );
    EXPECT_TRUE(  is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU     >(ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU  ) );


    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE,   ROCPROFILER_UVM_EVENT_QUEUE_EVICTION   >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE,   ROCPROFILER_UVM_EVENT_QUEUE_RESTORE    >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE,   ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU   >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE,   ROCPROFILER_UVM_EVENT_PAGE_FAULT_START >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE,   ROCPROFILER_UVM_EVENT_PAGE_FAULT_END   >() );

    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT,     ROCPROFILER_UVM_EVENT_MIGRATE_START    >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT,     ROCPROFILER_UVM_EVENT_MIGRATE_END      >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT,     ROCPROFILER_UVM_EVENT_QUEUE_EVICTION   >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT,     ROCPROFILER_UVM_EVENT_QUEUE_RESTORE    >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT,     ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU   >() );

    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND,  ROCPROFILER_UVM_EVENT_MIGRATE_START    >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND,  ROCPROFILER_UVM_EVENT_MIGRATE_END      >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND,  ROCPROFILER_UVM_EVENT_PAGE_FAULT_START >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND,  ROCPROFILER_UVM_EVENT_PAGE_FAULT_END   >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND,  ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU   >() );

    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU, ROCPROFILER_UVM_EVENT_MIGRATE_START    >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU, ROCPROFILER_UVM_EVENT_MIGRATE_END      >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU, ROCPROFILER_UVM_EVENT_PAGE_FAULT_START >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU, ROCPROFILER_UVM_EVENT_PAGE_FAULT_END   >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU, ROCPROFILER_UVM_EVENT_QUEUE_EVICTION   >() );
    static_assert( ! is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU, ROCPROFILER_UVM_EVENT_QUEUE_RESTORE    >() );

    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE       >(ROCPROFILER_UVM_EVENT_QUEUE_EVICTION  ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE       >(ROCPROFILER_UVM_EVENT_QUEUE_RESTORE   ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE       >(ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU  ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE       >(ROCPROFILER_UVM_EVENT_PAGE_FAULT_START) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE       >(ROCPROFILER_UVM_EVENT_PAGE_FAULT_END  ) );

    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT         >(ROCPROFILER_UVM_EVENT_MIGRATE_START   ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT         >(ROCPROFILER_UVM_EVENT_MIGRATE_END     ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT         >(ROCPROFILER_UVM_EVENT_QUEUE_EVICTION  ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT         >(ROCPROFILER_UVM_EVENT_QUEUE_RESTORE   ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT         >(ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU  ) );

    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND      >(ROCPROFILER_UVM_EVENT_MIGRATE_START   ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND      >(ROCPROFILER_UVM_EVENT_MIGRATE_END     ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND      >(ROCPROFILER_UVM_EVENT_PAGE_FAULT_START) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND      >(ROCPROFILER_UVM_EVENT_PAGE_FAULT_END  ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND      >(ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU  ) );

    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU     >(ROCPROFILER_UVM_EVENT_MIGRATE_START   ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU     >(ROCPROFILER_UVM_EVENT_MIGRATE_END     ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU     >(ROCPROFILER_UVM_EVENT_PAGE_FAULT_START) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU     >(ROCPROFILER_UVM_EVENT_PAGE_FAULT_END  ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU     >(ROCPROFILER_UVM_EVENT_QUEUE_EVICTION  ) );
    EXPECT_FALSE( is_rocprof_uvm_map < ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU     >(ROCPROFILER_UVM_EVENT_QUEUE_RESTORE   ) );

    static_assert(to_rocprof_op(ROCPROFILER_UVM_EVENT_MIGRATE_START)  == ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE    );
    static_assert(to_rocprof_op(ROCPROFILER_UVM_EVENT_PAGE_FAULT_END) == ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT      );
    static_assert(to_rocprof_op(ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU) == ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU  );
    static_assert(to_rocprof_op(ROCPROFILER_UVM_EVENT_QUEUE_EVICTION) == ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND   );

    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_TRIGGER_PREFETCH,                            KFD_MIGRATE_TRIGGER_PREFETCH            );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_TRIGGER_PAGEFAULT_GPU,                       KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU       );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_TRIGGER_PAGEFAULT_CPU,                       KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU       );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_TRIGGER_TTM_EVICTION,                        KFD_MIGRATE_TRIGGER_TTM_EVICTION        );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_SVM,                   KFD_QUEUE_EVICTION_TRIGGER_SVM          );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_USERPTR,               KFD_QUEUE_EVICTION_TRIGGER_USERPTR      );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_TTM,                   KFD_QUEUE_EVICTION_TRIGGER_TTM          );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_SUSPEND,               KFD_QUEUE_EVICTION_TRIGGER_SUSPEND      );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_CRIU_CHECKPOINT,       KFD_QUEUE_EVICTION_CRIU_CHECKPOINT      );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND_TRIGGER_CRIU_RESTORE,          KFD_QUEUE_EVICTION_CRIU_RESTORE         );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_MMU_NOTIFY,           KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY        );
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_MMU_NOTIFY_MIGRATE,   KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE);
    ASSERT_SAME(ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU_TRIGGER_UNMAP_FROM_CPU,       KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU    );

    static_assert(to_kfd_bitmask(std::index_sequence<
        ROCPROFILER_UVM_EVENT_PAGE_FAULT_START, ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU>()) ==
          (KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_PAGE_FAULT_START)
          | KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_UNMAP_FROM_GPU)));

    // clang-format on
}
