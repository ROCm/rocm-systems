/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "api_trace/api-trace-internal.h"

const char *
hipFileGetOpErrorString(hipFileOpError_t status)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_get_op_error_string(status);
}
hipFileError_t
hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_handle_register(fh, descr);
}
void
hipFileHandleDeregister(hipFileHandle_t fh)
{
    hipFile::GetHipFileDispatchTable()->pfn_hipfile_handle_deregister(fh);
}
hipFileError_t
hipFileBufRegister(const void *buffer_base, size_t length, int flags)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_buf_register(buffer_base, length, flags);
}
hipFileError_t
hipFileBufDeregister(const void *buffer_base)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_buf_deregister(buffer_base);
}
ssize_t
hipFileRead(hipFileHandle_t fh, void *buffer_base, size_t size, hoff_t file_offset, hoff_t buffer_offset)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_read(fh, buffer_base, size, file_offset,
                                                                buffer_offset);
}
ssize_t
hipFileWrite(hipFileHandle_t fh, const void *buffer_base, size_t size, hoff_t file_offset,
             hoff_t buffer_offset)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_write(fh, buffer_base, size, file_offset,
                                                                 buffer_offset);
}
hipFileError_t
hipFileDriverOpen(void)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_driver_open();
}
hipFileError_t
hipFileDriverClose(void)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_driver_close();
}
int64_t
hipFileUseCount(void)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_use_count();
}
hipFileError_t
hipFileDriverGetProperties(hipFileDriverProps_t *props)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_driver_get_properties(props);
}
hipFileError_t
hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_driver_set_poll_mode(poll, poll_threshold_size);
}
hipFileError_t
hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_driver_set_max_direct_io_size(max_direct_io_size);
}
hipFileError_t
hipFileDriverSetMaxCacheSize(size_t max_cache_size)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_driver_set_max_cache_size(max_cache_size);
}
hipFileError_t
hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_driver_set_max_pinned_mem_size(max_pinned_size);
}
hipFileError_t
hipFileBatchIOSetUp(hipFileBatchHandle_t *batch_idp, unsigned max_nr)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_batch_io_set_up(batch_idp, max_nr);
}
hipFileError_t
hipFileBatchIOSubmit(hipFileBatchHandle_t batch_idp, unsigned nr, hipFileIOParams_t *iocbp, unsigned flags)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_batch_io_submit(batch_idp, nr, iocbp, flags);
}
hipFileError_t
hipFileBatchIOGetStatus(hipFileBatchHandle_t batch_idp, unsigned min_nr, unsigned *nr,
                        hipFileIOEvents_t *iocbp, struct timespec *timeout)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_batch_io_get_status(batch_idp, min_nr, nr, iocbp,
                                                                               timeout);
}
hipFileError_t
hipFileBatchIOCancel(hipFileBatchHandle_t batch_idp)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_batch_io_cancel(batch_idp);
}
void
hipFileBatchIODestroy(hipFileBatchHandle_t batch_idp)
{
    hipFile::GetHipFileDispatchTable()->pfn_hipfile_batch_io_destroy(batch_idp);
}
hipFileError_t
hipFileReadAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                 hoff_t *buffer_offset_p, ssize_t *bytes_read_p, hipStream_t stream)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_read_async(fh, buffer_base, size_p, file_offset_p,
                                                                      buffer_offset_p, bytes_read_p, stream);
}
hipFileError_t
hipFileWriteAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                  hoff_t *buffer_offset_p, ssize_t *bytes_written_p, hipStream_t stream)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_write_async(
        fh, buffer_base, size_p, file_offset_p, buffer_offset_p, bytes_written_p, stream);
}
hipFileError_t
hipFileStreamRegister(hipStream_t stream, unsigned flags)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_stream_register(stream, flags);
}
hipFileError_t
hipFileStreamDeregister(hipStream_t stream)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_stream_deregister(stream);
}
hipFileError_t
hipFileGetVersion(unsigned *major, unsigned *minor, unsigned *patch)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_get_version(major, minor, patch);
}
hipFileError_t
hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t *value)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_get_parameter_size_t(param, value);
}
hipFileError_t
hipFileGetParameterBool(hipFileBoolConfigParameter_t param, bool *value)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_get_parameter_bool(param, value);
}
hipFileError_t
hipFileGetParameterString(hipFileStringConfigParameter_t param, char *desc_str, int len)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_get_parameter_string(param, desc_str, len);
}
hipFileError_t
hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_set_parameter_size_t(param, value);
}
hipFileError_t
hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_set_parameter_bool(param, value);
}
hipFileError_t
hipFileSetParameterString(hipFileStringConfigParameter_t param, const char *desc_str)
{
    return hipFile::GetHipFileDispatchTable()->pfn_hipfile_set_parameter_string(param, desc_str);
}
