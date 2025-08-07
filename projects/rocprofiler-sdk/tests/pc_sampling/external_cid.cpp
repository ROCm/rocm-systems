// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/pc_sampling_library/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "utils.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <stdint.h>
#include <atomic>
#include <iostream>
#include <memory>
#include <sstream>

namespace client
{
namespace external_cid
{
namespace
{
template <typename Arg, typename... Args>
auto
make_array(Arg arg, Args&&... args)
{
    constexpr auto N = 1 + sizeof...(Args);
    return std::array<Arg, N>{std::forward<Arg>(arg), std::forward<Args>(args)...};
}
}  // namespace

/**
 * @brief Must be called at the beginning of the `tool_ini`.
 */
void
init()
{}

/**
 * @brief Should be called at the of the `tool_fini`
 */
void
fini()
{}

int
set_external_correlation_id(rocprofiler_thread_id_t /*thr_id*/,
                            rocprofiler_context_id_t /*ctx_id*/,
                            rocprofiler_external_correlation_id_request_kind_t /*kind*/,
                            rocprofiler_tracing_operation_t /*op*/,
                            uint64_t                 internal_corr_id,
                            rocprofiler_user_data_t* external_corr_id,
                            void* /*user_data*/)
{
    // In multi-queues (devices) scenario, incrementing external correlation IDs
    // might not always match with incrementing internal correlation IDs.
    // Thus, use the value of internal correlation ID and verify that both
    // externall correlation IDs and internal correlation IDs are the same
    // in delivered PC samples.
    external_corr_id->value = internal_corr_id;
    return 0;
}

void
configure_external_correlation_service(rocprofiler_context_id_t context)
{
    auto external_corr_id_request_kinds =
        make_array(ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);

    ROCPROFILER_CHECK(rocprofiler_configure_external_correlation_id_request_service(
        context,
        external_corr_id_request_kinds.data(),
        external_corr_id_request_kinds.size(),
        set_external_correlation_id,
        nullptr));
}

}  // namespace external_cid
}  // namespace client
