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

#include <fmt/core.h>

#include "lib/common/container/small_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/mpl.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/page_migration/utils.hpp"

#define ASSERT_SAME(A, B) static_assert(static_cast<size_t>(A) == static_cast<size_t>(B))

#define ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL 1
#include "lib/rocprofiler-sdk/page_migration/page_migration.def.cpp"
#undef ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL

namespace rocprofiler
{
namespace page_migration
{
using namespace rocprofiler::page_migration;
using namespace rocprofiler::common::container;
using rocprofiler_page_migration_seq_t = std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>;

static_assert(KFD_SMI_EVENT_NONE == 0);
static_assert(KFD_SMI_EVENT_MIGRATE_START == 5);
static_assert(KFD_SMI_EVENT_MIGRATE_END == 6);
static_assert(KFD_SMI_EVENT_PAGE_FAULT_START == 7);
static_assert(KFD_SMI_EVENT_PAGE_FAULT_END == 8);
static_assert(KFD_SMI_EVENT_QUEUE_EVICTION == 9);
static_assert(KFD_SMI_EVENT_QUEUE_RESTORE == 10);
static_assert(KFD_SMI_EVENT_UNMAP_FROM_GPU == 11);
static_assert(KFD_SMI_EVENT_ALL_PROCESS == 64);

// Update page_migration.def.cpp with event mappings
// Update page_migration.cpp to parse and report new event
static_assert(ROCPROFILER_PAGE_MIGRATION_LAST == 8,
              "New event added, update KFD to ROCPROFILER mappings");

// clang-format off
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
// clang-format on

static_assert(kfd_bitmask(std::index_sequence<ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_START,
                                              ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_END,
                                              ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU>()) ==
              (KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_PAGE_FAULT_START) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_MIGRATE_END) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_UNMAP_FROM_GPU)));

static_assert((page_migration_info<ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_END>::kfd_bitmask |
               page_migration_info<ROCPROFILER_PAGE_MIGRATION_QUEUE_EVICTION>::kfd_bitmask |
               page_migration_info<ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU>::kfd_bitmask) ==
              (KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_MIGRATE_END) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_QUEUE_EVICTION) |
               KFD_SMI_EVENT_MASK_FROM_INDEX(KFD_SMI_EVENT_UNMAP_FROM_GPU)));

static_assert(kfd_to_rocprof_op(KFD_SMI_EVENT_MIGRATE_START, rocprofiler_page_migration_seq_t{}) ==
              ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_START);
static_assert(kfd_to_rocprof_op(KFD_SMI_EVENT_MIGRATE_END, rocprofiler_page_migration_seq_t{}) ==
              ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE_END);
static_assert(kfd_to_rocprof_op(KFD_SMI_EVENT_PAGE_FAULT_START,
                                rocprofiler_page_migration_seq_t{}) ==
              ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_START);
static_assert(kfd_to_rocprof_op(KFD_SMI_EVENT_PAGE_FAULT_END, rocprofiler_page_migration_seq_t{}) ==
              ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT_END);
static_assert(kfd_to_rocprof_op(KFD_SMI_EVENT_QUEUE_EVICTION, rocprofiler_page_migration_seq_t{}) ==
              ROCPROFILER_PAGE_MIGRATION_QUEUE_EVICTION);
static_assert(kfd_to_rocprof_op(KFD_SMI_EVENT_QUEUE_RESTORE, rocprofiler_page_migration_seq_t{}) ==
              ROCPROFILER_PAGE_MIGRATION_QUEUE_RESTORE);
static_assert(kfd_to_rocprof_op(KFD_SMI_EVENT_UNMAP_FROM_GPU, rocprofiler_page_migration_seq_t{}) ==
              ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU);

}  // namespace page_migration
}  // namespace rocprofiler
