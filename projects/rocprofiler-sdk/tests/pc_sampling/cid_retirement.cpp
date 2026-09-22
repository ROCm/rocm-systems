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

#include <iostream>
#include <memory>
#include <sstream>

namespace client
{
namespace cid_retirement
{
constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);

rocprofiler_buffer_id_t cid_retirement_buffer;

void
cid_retirement_tracing_buffered(rocprofiler_context_id_t /*context*/,
                                rocprofiler_buffer_id_t /*buffer_id*/,
                                rocprofiler_record_header_t** headers,
                                size_t                        num_headers,
                                void* /*user_data*/,
                                uint64_t /*drop_count*/)
{
    std::stringstream ss;

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(header->hash !=
                rocprofiler_record_header_compute_hash(header->category, header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
        {
            if(header->kind == ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT)
            {
                auto* cid_record =
                    static_cast<rocprofiler_buffer_tracing_correlation_id_retirement_record_t*>(
                        header->payload);
                ss << "... The retired internal correlation id is: "
                   << cid_record->internal_correlation_id;
                ss << ", the timestamp is: " << cid_record->timestamp;
                ss << std::endl;
                // TODO: assert that the retiring timestamp is greater than
                // the greatest timestamp of PC samples matching the retired CID.
            }
        }
    }

    *utils::get_output_stream() << ss.str();
}

void
configure_cid_retirement_tracing(rocprofiler_context_id_t context)
{
    ROCPROFILER_CALL(rocprofiler_create_buffer(context,
                                               BUFFER_SIZE_BYTES,
                                               WATERMARK,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               cid_retirement_tracing_buffered,
                                               nullptr,
                                               &cid_retirement_buffer),
                     "buffer creation");

    ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                         context,
                         ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT,
                         nullptr,
                         0,
                         cid_retirement_buffer),
                     "buffer tracing service for cid retirement configure");
}

void
flush_retired_cids()
{
    ROCPROFILER_CALL(rocprofiler_flush_buffer(cid_retirement_buffer),
                     "Cannot flush retired CIDs buffer");
    *utils::get_output_stream() << "Retired CIDs flushed..." << std::endl;
}

}  // namespace cid_retirement
}  // namespace client
