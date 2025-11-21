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

#include "lib/common/container/small_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/mpl.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/kfd/defines.hpp"
#include "lib/rocprofiler-sdk/kfd/utils.hpp"

#include <fmt/core.h>

#define ASSERT_SAME(A, B) static_assert(static_cast<size_t>(A) == static_cast<size_t>(B))

namespace rocprofiler
{
namespace kfd
{
static_assert(KFD_SMI_EVENT_NONE == 0);
static_assert(KFD_SMI_EVENT_MIGRATE_START == 5);
static_assert(KFD_SMI_EVENT_MIGRATE_END == 6);
static_assert(KFD_SMI_EVENT_PAGE_FAULT_START == 7);
static_assert(KFD_SMI_EVENT_PAGE_FAULT_END == 8);
static_assert(KFD_SMI_EVENT_QUEUE_EVICTION == 9);
static_assert(KFD_SMI_EVENT_QUEUE_RESTORE == 10);
static_assert(KFD_SMI_EVENT_UNMAP_FROM_GPU == 11);
static_assert(KFD_SMI_EVENT_DROPPED_EVENT == 14);
static_assert(KFD_SMI_EVENT_ALL_PROCESS == 64);

// If any of these changes, we can no longer static_cast the new triggers
// from the KFD trigger to the record operation type
static_assert(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_LAST == 5);
static_assert(ROCPROFILER_KFD_PAGE_MIGRATE_LAST == 4);
static_assert(ROCPROFILER_KFD_EVENT_PAGE_FAULT_LAST == 5);
static_assert(ROCPROFILER_KFD_PAGE_FAULT_LAST == 4);
static_assert(ROCPROFILER_KFD_EVENT_QUEUE_LAST == 8);
static_assert(ROCPROFILER_KFD_QUEUE_LAST == 6);
static_assert(ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_LAST == 3);
static_assert(ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED == 6);
static_assert(ROCPROFILER_KFD_EVENT_QUEUE_RESTORE == 7);

// clang-format off
ASSERT_SAME(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PREFETCH,             KFD_MIGRATE_TRIGGER_PREFETCH            );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_GPU,        KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU       );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_PAGEFAULT_CPU,        KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU       );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_PAGE_MIGRATE_TTM_EVICTION,         KFD_MIGRATE_TRIGGER_TTM_EVICTION        );
ASSERT_SAME(ROCPROFILER_KFD_PAGE_MIGRATE_PREFETCH,                   KFD_MIGRATE_TRIGGER_PREFETCH            );
ASSERT_SAME(ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_GPU,              KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU       );
ASSERT_SAME(ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_CPU,              KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU       );
ASSERT_SAME(ROCPROFILER_KFD_PAGE_MIGRATE_TTM_EVICTION,               KFD_MIGRATE_TRIGGER_TTM_EVICTION        );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SVM,                   KFD_QUEUE_EVICTION_TRIGGER_SVM          );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR,               KFD_QUEUE_EVICTION_TRIGGER_USERPTR      );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM,                   KFD_QUEUE_EVICTION_TRIGGER_TTM          );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND,               KFD_QUEUE_EVICTION_TRIGGER_SUSPEND      );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT,       KFD_QUEUE_EVICTION_CRIU_CHECKPOINT      );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE,          KFD_QUEUE_EVICTION_CRIU_RESTORE         );
ASSERT_SAME(ROCPROFILER_KFD_QUEUE_EVICT_SVM,                         KFD_QUEUE_EVICTION_TRIGGER_SVM          );
ASSERT_SAME(ROCPROFILER_KFD_QUEUE_EVICT_USERPTR,                     KFD_QUEUE_EVICTION_TRIGGER_USERPTR      );
ASSERT_SAME(ROCPROFILER_KFD_QUEUE_EVICT_TTM,                         KFD_QUEUE_EVICTION_TRIGGER_TTM          );
ASSERT_SAME(ROCPROFILER_KFD_QUEUE_EVICT_SUSPEND,                     KFD_QUEUE_EVICTION_TRIGGER_SUSPEND      );
ASSERT_SAME(ROCPROFILER_KFD_QUEUE_EVICT_CRIU_CHECKPOINT,             KFD_QUEUE_EVICTION_CRIU_CHECKPOINT      );
ASSERT_SAME(ROCPROFILER_KFD_QUEUE_EVICT_CRIU_RESTORE,                KFD_QUEUE_EVICTION_CRIU_RESTORE         );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY,         KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY        );
ASSERT_SAME(ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE, KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE);
ASSERT_SAME(ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_UNMAP_FROM_CPU,     KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU    );
// clang-format on

}  // namespace kfd
}  // namespace rocprofiler
