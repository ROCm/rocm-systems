// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hipfile/defines.hpp"
#include "lib/rocprofiler-sdk/hipfile/hipfile.hpp"

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hipfile.h>
#include <rocprofiler-sdk/hipfile/table_id.h>

namespace rocprofiler
{
namespace hipfile
{
template <>
struct hipfile_domain_info<ROCPROFILER_HIPFILE_TABLE_ID_LAST>
{
    using args_type              = rocprofiler_hipfile_api_args_t;
    using retval_type            = rocprofiler_hipfile_api_retval_t;
    using callback_data_type     = rocprofiler_callback_tracing_hipfile_api_data_t;
    using buffer_data_type       = rocprofiler_buffer_tracing_hipfile_api_record_t;
    using buffered_ext_data_type = rocprofiler_buffer_tracing_hipfile_api_ext_record_t;
};

template <>
struct hipfile_domain_info<ROCPROFILER_HIPFILE_TABLE_ID_CORE>
: hipfile_domain_info<ROCPROFILER_HIPFILE_TABLE_ID_LAST>
{
    using enum_type                               = rocprofiler_hipfile_api_id_t;
    static constexpr auto callback_domain_idx     = ROCPROFILER_CALLBACK_TRACING_HIPFILE_API;
    static constexpr auto buffered_domain_idx     = ROCPROFILER_BUFFER_TRACING_HIPFILE_API;
    static constexpr auto buffered_ext_domain_idx = ROCPROFILER_BUFFER_TRACING_HIPFILE_API_EXT;
    static constexpr auto none                    = ROCPROFILER_HIPFILE_API_ID_NONE;
    static constexpr auto last                    = ROCPROFILER_HIPFILE_API_ID_LAST;
    static constexpr auto external_correlation_id_domain_idx =
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIPFILE_API;
};

}  // namespace hipfile
}  // namespace rocprofiler

#if defined(ROCPROFILER_LIB_ROCPROFILER_SDK_HIPFILE_HIPFILE_CPP_IMPL) &&                           \
    ROCPROFILER_LIB_ROCPROFILER_SDK_HIPFILE_HIPFILE_CPP_IMPL == 1

// clang-format off
HIPFILE_API_TABLE_LOOKUP_DEFINITION(ROCPROFILER_HIPFILE_TABLE_ID_CORE, hipfile_api_func_table_t)

HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileGetOpErrorString, hipFileGetOpErrorString, pfn_hipfile_get_op_error_string, status)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileHandleRegister, hipFileHandleRegister, pfn_hipfile_handle_register, fh, descr)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileHandleDeregister, hipFileHandleDeregister, pfn_hipfile_handle_deregister, fh)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileBufRegister, hipFileBufRegister, pfn_hipfile_buf_register, buffer_base, length, flags)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileBufDeregister, hipFileBufDeregister, pfn_hipfile_buf_deregister, buffer_base)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileRead, hipFileRead, pfn_hipfile_read, fh, buffer_base, size, file_offset, buffer_offset)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileWrite, hipFileWrite, pfn_hipfile_write, fh, buffer_base, size, file_offset, buffer_offset)
HIPFILE_API_INFO_DEFINITION_0(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileDriverOpen, hipFileDriverOpen, pfn_hipfile_driver_open)
HIPFILE_API_INFO_DEFINITION_0(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileDriverClose, hipFileDriverClose, pfn_hipfile_driver_close)
HIPFILE_API_INFO_DEFINITION_0(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileUseCount, hipFileUseCount, pfn_hipfile_use_count)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileDriverGetProperties, hipFileDriverGetProperties, pfn_hipfile_driver_get_properties, props)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileDriverSetPollMode, hipFileDriverSetPollMode, pfn_hipfile_driver_set_poll_mode, poll, poll_threshold_size)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileDriverSetMaxDirectIOSize, hipFileDriverSetMaxDirectIOSize, pfn_hipfile_driver_set_max_direct_io_size, max_direct_io_size)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileDriverSetMaxCacheSize, hipFileDriverSetMaxCacheSize, pfn_hipfile_driver_set_max_cache_size, max_cache_size)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileDriverSetMaxPinnedMemSize, hipFileDriverSetMaxPinnedMemSize, pfn_hipfile_driver_set_max_pinned_mem_size, max_pinned_size)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileBatchIOSetUp, hipFileBatchIOSetUp, pfn_hipfile_batch_io_set_up, batch_idp, max_nr)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileBatchIOSubmit, hipFileBatchIOSubmit, pfn_hipfile_batch_io_submit, batch_idp, nr, iocbp, flags)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileBatchIOGetStatus, hipFileBatchIOGetStatus, pfn_hipfile_batch_io_get_status, batch_idp, min_nr, nr, iocbp, timeout)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileBatchIOCancel, hipFileBatchIOCancel, pfn_hipfile_batch_io_cancel, batch_idp)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileBatchIODestroy, hipFileBatchIODestroy, pfn_hipfile_batch_io_destroy, batch_idp)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileReadAsync, hipFileReadAsync, pfn_hipfile_read_async, fh, buffer_base, size_p, file_offset_p, buffer_offset_p, bytes_read_p, stream)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileWriteAsync, hipFileWriteAsync, pfn_hipfile_write_async, fh, buffer_base, size_p, file_offset_p, buffer_offset_p, bytes_written_p, stream)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileStreamRegister, hipFileStreamRegister, pfn_hipfile_stream_register, stream, flags)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileStreamDeregister, hipFileStreamDeregister, pfn_hipfile_stream_deregister, stream)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileGetVersion, hipFileGetVersion, pfn_hipfile_get_version, major, minor, patch)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileGetParameterSizeT, hipFileGetParameterSizeT, pfn_hipfile_get_parameter_size_t, param, value)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileGetParameterBool, hipFileGetParameterBool, pfn_hipfile_get_parameter_bool, param, value)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileGetParameterString, hipFileGetParameterString, pfn_hipfile_get_parameter_string, param, desc_str, len)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileSetParameterSizeT, hipFileSetParameterSizeT, pfn_hipfile_set_parameter_size_t, param, value)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileSetParameterBool, hipFileSetParameterBool, pfn_hipfile_set_parameter_bool, param, value)
HIPFILE_API_INFO_DEFINITION_V(ROCPROFILER_HIPFILE_TABLE_ID_CORE, ROCPROFILER_HIPFILE_API_ID_hipFileSetParameterString, hipFileSetParameterString, pfn_hipfile_set_parameter_string, param, desc_str)

#else
#    error                                                                                         \
        "Do not compile this file directly. It is included by lib/rocprofiler-sdk/hipfile/hipfile.cpp"
#endif
