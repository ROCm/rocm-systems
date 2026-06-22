/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "rocshmem/api_trace.h"
#include "rocshmem/rocshmem_common.hpp"
#include <cstddef>
#include <cstdint>

#if defined(ROCSHMEM_ROCPROFILER_REGISTER)
#    include <rocprofiler-register/rocprofiler-register.h>
#    include "rocshmem/rocshmem_config.h"

#    define ROCP_REG_VERSION                                                             \
        ROCPROFILER_REGISTER_COMPUTE_VERSION_3(ROCSHMEM_VENDOR_MAJOR_VERSION,            \
                                               ROCSHMEM_VENDOR_MINOR_VERSION,            \
                                               ROCSHMEM_VENDOR_PATCH_VERSION)

ROCPROFILER_REGISTER_DEFINE_IMPORT(rocshmem, ROCP_REG_VERSION)
#endif

#include "context.hpp"
#include "log.hpp"

// Forward declarations for implementation functions with C linkage
extern "C" {
__host__ void rocshmem_barrier_all_on_stream_impl(hipStream_t stream);

__host__ void rocshmem_quiet_on_stream_impl(hipStream_t stream);

__host__ void rocshmem_sync_all_on_stream_impl(hipStream_t stream);

__host__ void rocshmem_alltoallmem_on_stream_impl(rocshmem_team_t team, void *dest,
                                                   const void *source, size_t size,
                                                   hipStream_t stream);

__host__ void rocshmem_broadcastmem_on_stream_impl(rocshmem_team_t team, void *dest,
                                                    const void *source, size_t nelems,
                                                    int pe_root, hipStream_t stream);

__host__ void rocshmem_getmem_on_stream_impl(void *dest, const void *source, size_t nelems,
                                              int pe, hipStream_t stream);

__host__ void rocshmem_putmem_on_stream_impl(void *dest, const void *source, size_t nelems,
                                              int pe, hipStream_t stream);

__host__ void rocshmem_putmem_signal_on_stream_impl(void *dest, const void *source,
                                                     size_t nelems, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe,
                                                     hipStream_t stream);

__host__ void rocshmem_signal_wait_until_on_stream_impl(uint64_t *sig_addr, int cmp,
                                                         uint64_t cmp_value,
                                                         hipStream_t stream);
}  // extern "C"

namespace rocshmem
{
extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

namespace
{
constexpr size_t
compute_table_offset(size_t n)
{
    return (sizeof(uint64_t) + (n * sizeof(void*)));
}

constexpr size_t
compute_table_size(size_t nmembers)
{
    return (sizeof(uint64_t) + (nmembers * sizeof(void*)));
}

#define ROCSHMEM_ASSERT_OFFSET(TABLE, MEMBER, IDX)                                           \
    static_assert(offsetof(TABLE, MEMBER) == compute_table_offset(IDX),                  \
                  "Do not re-arrange the table members")


// DO NOT REORDER, ADD NEW ITEMS TO BOTTOM
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, barrier_all_on_stream_fn, 0);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, quiet_on_stream_fn, 1);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, sync_all_on_stream_fn, 2);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, alltoallmem_on_stream_fn, 3);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, broadcastmem_on_stream_fn, 4);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, getmem_on_stream_fn, 5);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, putmem_on_stream_fn, 6);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, putmem_signal_on_stream_fn, 7);
ROCSHMEM_ASSERT_OFFSET(rocshmemApiFuncTable, signal_wait_until_on_stream_fn, 8);

#undef ROCSHMEM_ASSERT_OFFSET

static_assert(sizeof(rocshmemApiFuncTable) == compute_table_size(9),
              "Update table major/step version and add a new offset assertion if this "
              "fails to compile");

// Use alignas to ensure proper alignment for placement new
alignas(rocshmemApiFuncTable) std::array<unsigned char, sizeof(rocshmemApiFuncTable)> m_buffer = {};

rocshmemApiFuncTable*
RocshmemGetFunctionTable_impl()
{
    static auto* tbl =
        new(m_buffer.data()) rocshmemApiFuncTable{
            sizeof(rocshmemApiFuncTable),
            &rocshmem_barrier_all_on_stream_impl,
            &rocshmem_quiet_on_stream_impl,
            &rocshmem_sync_all_on_stream_impl,
            &rocshmem_alltoallmem_on_stream_impl,
            &rocshmem_broadcastmem_on_stream_impl,
            &rocshmem_getmem_on_stream_impl,
            &rocshmem_putmem_on_stream_impl,
            &rocshmem_putmem_signal_on_stream_impl,
            &rocshmem_signal_wait_until_on_stream_impl
        };

#if defined(ROCSHMEM_ROCPROFILER_REGISTER) && ROCSHMEM_ROCPROFILER_REGISTER > 0
    std::array<void*, 1>                       table_array{ tbl };
    rocprofiler_register_library_indentifier_t lib_id =
        rocprofiler_register_library_indentifier_t{};
    rocprofiler_register_error_code_t rocp_reg_status =
        rocprofiler_register_library_api_table(
            "rocshmem", &ROCPROFILER_REGISTER_IMPORT_FUNC(rocshmem), ROCP_REG_VERSION,
            table_array.data(), table_array.size(), &lib_id);

    if(rocp_reg_status != ROCP_REG_SUCCESS && rocp_reg_status != ROCP_REG_NO_TOOLS) {
        fprintf(stderr, "[rocprofiler-sdk-rocshmem][%d] rocprofiler-register failed with error code %d : %s\n",
                getpid(), rocp_reg_status,
                rocprofiler_register_error_string(rocp_reg_status));
    }
#endif

    return tbl;
}

}  // end of anonymous namespace

}  // end of namespace rocshmem

extern "C" {
const rocshmemApiFuncTable*
RocshmemGetFunctionTable()
{
    static const auto* tbl = rocshmem::RocshmemGetFunctionTable_impl();
    return tbl;
}
}


// =============================================================================
// IMPLEMENTATION FUNCTIONS WITH ROCPROFILER TRACING
// =============================================================================

extern "C" {

__host__ void rocshmem_barrier_all_on_stream_impl(hipStream_t stream)
{
    LOG_API("host::barrier_all_on_stream ()");
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->barrier_all_on_stream(stream);
}

__host__ void rocshmem_quiet_on_stream_impl(hipStream_t stream)
{
    LOG_API("rocshmem_quiet_on_stream");
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->quiet_on_stream(stream);
}

__host__ void rocshmem_sync_all_on_stream_impl(hipStream_t stream)
{
    LOG_API("rocshmem_sync_all_on_stream");
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->sync_all_on_stream(stream);
}

__host__ void rocshmem_alltoallmem_on_stream_impl(rocshmem_team_t team, void *dest,
                                                   const void *source, size_t size,
                                                   hipStream_t stream)
{
     LOG_API("host::alltoallmem_on_stream (dest=%p, source=%p, size=%zd)", dest, source, size);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->alltoallmem_on_stream(team, dest, source, size, stream);
}

__host__ void rocshmem_broadcastmem_on_stream_impl(rocshmem_team_t team, void *dest,
                                                    const void *source, size_t nelems,
                                                    int pe_root, hipStream_t stream)
{
    LOG_API("host::broadcastmem_on_stream (dest=%p, source=%p, nelems=%zd, pe_root=%d)", dest, source, nelems, pe_root);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->broadcastmem_on_stream(team, dest, source, nelems, pe_root, stream);
}

__host__ void rocshmem_getmem_on_stream_impl(void *dest, const void *source, size_t nelems,
                                              int pe, hipStream_t stream)
{
    LOG_API("host::getmem_on_stream (dest=%p, source=%p, nelems=%zd, pe=%d)", dest, source, nelems, pe);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->getmem_on_stream(dest, source, nelems, pe, stream);
}

__host__ void rocshmem_putmem_on_stream_impl(void *dest, const void *source, size_t nelems,
                                              int pe, hipStream_t stream)
{
    LOG_API("host::putmem_on_stream (dest=%p, source=%p, nelems=%zd, pe=%d)", dest, source, nelems, pe);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->putmem_on_stream(dest, source, nelems, pe, stream);
}

__host__ void rocshmem_putmem_signal_on_stream_impl(void *dest, const void *source,
                                                     size_t nelems, uint64_t *sig_addr,
                                                     uint64_t signal, int sig_op, int pe,
                                                     hipStream_t stream)
{
    LOG_API("host::putmem_signal_on_stream (dest=%p, source=%p, nelems=%zd, pe=%d)", dest, source, nelems, pe);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->putmem_signal_on_stream(dest, source, nelems, sig_addr, signal, sig_op, pe, stream);
}

__host__ void rocshmem_signal_wait_until_on_stream_impl(uint64_t *sig_addr, int cmp,
                                                         uint64_t cmp_value,
                                                         hipStream_t stream)
{
    LOG_API("host::signal_wait_until_on_stream (sig_addr=%p, cmp=%d)", sig_addr, cmp);
    rocshmem::Context* ctx = reinterpret_cast<rocshmem::Context*>(rocshmem::ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque);
    ctx->signal_wait_until_on_stream(sig_addr, cmp, cmp_value, stream);
}

}  // extern "C"

