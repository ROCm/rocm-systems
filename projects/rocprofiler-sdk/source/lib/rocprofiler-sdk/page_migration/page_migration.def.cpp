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

#include "lib/rocprofiler-sdk/page_migration/defines.hpp"
#include "lib/rocprofiler-sdk/page_migration/page_migration.hpp"

#if defined(ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL) &&             \
    ROCPROFILER_LIB_ROCPROFILER_SDK_PAGE_MIGRATION_PAGE_MIGRATION_CPP_IMPL == 1

namespace rocprofiler
{
namespace page_migration
{
using namespace rocprofiler::common;

using kfd_event_id_t           = decltype(KFD_SMI_EVENT_NONE);
using migrate_trigger_t        = rocprofiler_page_migration_trigger_t;
using page_migration_record_t  = rocprofiler_buffer_tracing_page_migration_record_t;
using queue_suspend_trigger_t  = rocprofiler_page_migration_queue_suspend_trigger_t;
using unmap_from_gpu_trigger_t = rocprofiler_page_migration_unmap_from_gpu_trigger_t;

using trigger_type_list_t = common::mpl::type_list<rocprofiler_page_migration_trigger_t,
                                                   queue_suspend_trigger_t,
                                                   unmap_from_gpu_trigger_t>;
// clang-format off
// Map ROCPROF UVM enums to KFD enums
SPECIALIZE_PAGE_MIGRATION_INFO(NONE,                NONE,               "Error: Invalid UVM event from KFD"     );
SPECIALIZE_PAGE_MIGRATION_INFO(PAGE_MIGRATE_START,  MIGRATE_START,      "%x %ld -%d @%lx(%lx) %x->%x %x:%x %d\n");
SPECIALIZE_PAGE_MIGRATION_INFO(PAGE_MIGRATE_END,    MIGRATE_END,        "%x %ld -%d @%lx(%lx) %x->%x %d\n"      );
SPECIALIZE_PAGE_MIGRATION_INFO(PAGE_FAULT_START,    PAGE_FAULT_START,   "%x %ld -%d @%lx(%x) %c\n"              );
SPECIALIZE_PAGE_MIGRATION_INFO(PAGE_FAULT_END,      PAGE_FAULT_END,     "%x %ld -%d @%lx(%x) %c\n"              );
SPECIALIZE_PAGE_MIGRATION_INFO(QUEUE_EVICTION,      QUEUE_EVICTION,     "%x %ld -%d %x %d\n"                    );
SPECIALIZE_PAGE_MIGRATION_INFO(QUEUE_RESTORE,       QUEUE_RESTORE,      "%x %ld -%d %x\n"                       );
SPECIALIZE_PAGE_MIGRATION_INFO(UNMAP_FROM_GPU,      UNMAP_FROM_GPU,     "%x %ld -%d @%lx(%lx) %x %d\n"          );
#undef SPECIALIZE_PAGE_MIGRATION_INFO
// clang-format on

}  // namespace page_migration
}  // namespace rocprofiler
#endif
