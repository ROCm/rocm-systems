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
// clang-format off
// Map ROCPROF UVM enums to KFD enums
SPECIALIZE_UVM_KFD_EVENT(ROCPROFILER_UVM_EVENT_NONE,             KFD_SMI_EVENT_NONE,             "Error: Invalid UVM event from KFD" );
SPECIALIZE_UVM_KFD_EVENT(ROCPROFILER_UVM_EVENT_MIGRATE_START,    KFD_SMI_EVENT_MIGRATE_START,    "%x %ld -%d @%lx(%lx) %x->%x %x:%x %d\n" );
SPECIALIZE_UVM_KFD_EVENT(ROCPROFILER_UVM_EVENT_MIGRATE_END,      KFD_SMI_EVENT_MIGRATE_END,      "%x %ld -%d @%lx(%lx) %x->%x %d\n"       );
SPECIALIZE_UVM_KFD_EVENT(ROCPROFILER_UVM_EVENT_PAGE_FAULT_START, KFD_SMI_EVENT_PAGE_FAULT_START, "%x %ld -%d @%lx(%x) %c\n"               );
SPECIALIZE_UVM_KFD_EVENT(ROCPROFILER_UVM_EVENT_PAGE_FAULT_END,   KFD_SMI_EVENT_PAGE_FAULT_END,   "%x %ld -%d @%lx(%x) %c\n"               );
SPECIALIZE_UVM_KFD_EVENT(ROCPROFILER_UVM_EVENT_QUEUE_EVICTION,   KFD_SMI_EVENT_QUEUE_EVICTION,   "%x %ld -%d %x %d\n"                     );
SPECIALIZE_UVM_KFD_EVENT(ROCPROFILER_UVM_EVENT_QUEUE_RESTORE,    KFD_SMI_EVENT_QUEUE_RESTORE,    "%x %ld -%d %x\n"                        );
SPECIALIZE_UVM_KFD_EVENT(ROCPROFILER_UVM_EVENT_UNMAP_FROM_GPU,   KFD_SMI_EVENT_UNMAP_FROM_GPU,   "%x %ld -%d @%lx(%lx) %x %d\n"          );
// clang-format on
#    undef SPECIALIZE_UVM_KFD_EVENT

SPECIALIZE_PAGE_MIGRATION_INFO(NONE, NONE);
SPECIALIZE_PAGE_MIGRATION_INFO(PAGE_MIGRATE, MIGRATE_START, MIGRATE_END);
SPECIALIZE_PAGE_MIGRATION_INFO(PAGE_FAULT, PAGE_FAULT_START, PAGE_FAULT_END);
SPECIALIZE_PAGE_MIGRATION_INFO(QUEUE_SUSPEND, QUEUE_EVICTION, QUEUE_RESTORE);
SPECIALIZE_PAGE_MIGRATION_INFO(UNMAP_FROM_GPU, UNMAP_FROM_GPU);

template <size_t UvmOpInx, size_t... OpInxs>
constexpr size_t to_rocprof_op_impl(std::index_sequence<OpInxs...>)
{
    return ((((bitmask(UvmOpInx) & page_migration_info<OpInxs>::uvm_bitmask) != 0) * OpInxs) + ...);
}

template <size_t... OpInxs>
constexpr auto _to_rocprof_op_impl(std::index_sequence<OpInxs...>)
{
    return std::array{
        to_rocprof_op_impl<OpInxs>(std::make_index_sequence<ROCPROFILER_PAGE_MIGRATION_LAST>{})...};
}

constexpr auto
to_rocprof_op(size_t pos)
{
    using rop = rocprofiler_page_migration_operation_t;
    return static_cast<rop>(
        _to_rocprof_op_impl(std::make_index_sequence<ROCPROFILER_UVM_EVENT_LAST>{})[pos]);
}
}  // namespace page_migration
}  // namespace rocprofiler
#endif
