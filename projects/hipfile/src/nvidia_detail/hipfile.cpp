/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile.h"
#include "hipfile-cufile.h"

#include <cuda.h>
#include <cufile.h>
#include <driver_types.h>
#include <hip/hip_runtime_api.h>
#include <sys/types.h>
#include <vector>

using namespace std;

/*
 * AIS Team Note:
 *
 * Typically the HIP->CUDA call has the following components:
 *
 * 1) Type conversions (of some variety).
 * 2) Call the CUDA function.
 * 3) Convert the error code back to HIP.
 *
 * Released HIP code typically combines this all into one line when possible.
 */

hipFileError_t
hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr)
try {
    CUfileHandle_t *cu_fh = fh;
    CUfileError_t   status;

    if (descr) {
        auto cu_descr = toCUfileDescr(*descr);
        status        = cuFileHandleRegister(cu_fh, &cu_descr);
        // Pass back the FSOps pointer to hipFileDescr_t as it may have been
        // modified by the cuFile library.
        descr->fs_ops = reinterpret_cast<const hipFileFSOps_t *>(cu_descr.fs_ops);
    }
    else {
        status = cuFileHandleRegister(cu_fh, nullptr);
    }

    return toHipFileError(status);
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

void
hipFileHandleDeregister(hipFileHandle_t fh)
{
    CUfileHandle_t cu_fh = fh;
    cuFileHandleDeregister(cu_fh);
}

hipFileError_t
hipFileBufRegister(const void *buffer_base, size_t length, int flags)
try {
    CUfileError_t status = cuFileBufRegister(buffer_base, length, flags);
    return toHipFileError(status);
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileBufDeregister(const void *buffer_base)
try {
    return toHipFileError(cuFileBufDeregister(buffer_base));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

ssize_t
hipFileRead(hipFileHandle_t fh, void *buffer_base, size_t size, hoff_t file_offset, hoff_t buffer_offset)
{
    CUfileHandle_t cu_fh     = fh;
    ssize_t        bytesRead = cuFileRead(cu_fh, buffer_base, size, file_offset, buffer_offset);
    // If -1, check errno
    // If <=-5000, maps to -hipFileOpError_t
    return bytesRead;
}

ssize_t
hipFileWrite(hipFileHandle_t fh, const void *buffer_base, size_t size, hoff_t file_offset,
             hoff_t buffer_offset)
{
    CUfileHandle_t cu_fh        = fh;
    ssize_t        bytesWritten = cuFileWrite(cu_fh, buffer_base, size, file_offset, buffer_offset);
    // If -1, check errno
    // If <=-5000, maps to -hipFileOpError_t
    return bytesWritten;
}

hipFileError_t
hipFileDriverOpen()
try {
    return toHipFileError(cuFileDriverOpen());
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileDriverClose()
try {
    return toHipFileError(cuFileDriverClose());
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

int64_t
hipFileUseCount()
{
    return cuFileUseCount();
}

hipFileError_t
hipFileDriverGetProperties(hipFileDriverProps_t *props)
try {
    CUfileError_t status;

    if (props) {
        CUfileDrvProps_t cu_props;
        status = cuFileDriverGetProperties(&cu_props);
        if (status.err == CU_FILE_SUCCESS) {
            *props = toHipFileDriverProps(cu_props);
        }
    }
    else {
        status = cuFileDriverGetProperties(nullptr);
    }

    return toHipFileError(status);
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size)
try {
    return toHipFileError(cuFileDriverSetPollMode(poll, poll_threshold_size));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)
try {
    return toHipFileError(cuFileDriverSetMaxDirectIOSize(max_direct_io_size));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileDriverSetMaxCacheSize(size_t max_cache_size)
try {
    return toHipFileError(cuFileDriverSetMaxCacheSize(max_cache_size));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)
try {
    return toHipFileError(cuFileDriverSetMaxPinnedMemSize(max_pinned_size));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileBatchIOSetUp(hipFileBatchHandle_t *batch_idp, unsigned max_nr)
try {
    CUfileBatchHandle_t *cu_batch_idp = batch_idp;
    return toHipFileError(cuFileBatchIOSetUp(cu_batch_idp, max_nr));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileBatchIOSubmit(hipFileBatchHandle_t batch_idp, unsigned nr, hipFileIOParams_t *iocbp, unsigned flags)
try {
    CUfileBatchHandle_t cu_batch_idp = batch_idp;
    CUfileError_t       status;

    if (iocbp) {
        vector<CUfileIOParams_t> io_params(nr);

        for (unsigned i = 0; i < nr; i++) {
            io_params[i] = toCUfileIOParams(iocbp[i]);
        }

        status = cuFileBatchIOSubmit(cu_batch_idp, nr, io_params.data(), flags);
    }
    else {
        status = cuFileBatchIOSubmit(cu_batch_idp, nr, nullptr, flags);
    }

    return toHipFileError(status);
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileBatchIOGetStatus(hipFileBatchHandle_t batch_idp, unsigned min_nr, unsigned *nr,
                        hipFileIOEvents_t *iocbp, struct timespec *timeout)
try {
    CUfileBatchHandle_t cu_batch_idp = batch_idp;
    CUfileError_t       status;

    if (iocbp) {
        vector<CUfileIOEvents> io_events(*nr);

        status = cuFileBatchIOGetStatus(cu_batch_idp, min_nr, nr, io_events.data(), timeout);

        if (status.err == CU_FILE_SUCCESS) {
            for (unsigned i = 0; i < *nr; i++) {
                iocbp[i] = toHipFileIOEvents(io_events[i]);
            }
        }
    }
    else {
        status = cuFileBatchIOGetStatus(cu_batch_idp, min_nr, nr, nullptr, timeout);
    }

    return toHipFileError(status);
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileBatchIOCancel(hipFileBatchHandle_t batch_idp)
try {
    CUfileBatchHandle_t cu_batch_idp = batch_idp;
    return toHipFileError(cuFileBatchIOCancel(cu_batch_idp));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

void
hipFileBatchIODestroy(hipFileBatchHandle_t batch_idp)
{
    CUfileBatchHandle_t cu_batch_idp = batch_idp;

    cuFileBatchIODestroy(cu_batch_idp);
}

hipFileError_t
hipFileReadAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                 hoff_t *buffer_offset_p, ssize_t *bytes_read_p, hipStream_t stream)
try {
    CUfileHandle_t cu_fh     = fh;
    CUstream       cu_stream = stream;

    return toHipFileError(
        cuFileReadAsync(cu_fh, buffer_base, size_p, file_offset_p, buffer_offset_p, bytes_read_p, cu_stream));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileWriteAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                  hoff_t *buffer_offset_p, ssize_t *bytes_written_p, hipStream_t stream)
try {
    CUfileHandle_t cu_fh     = fh;
    CUstream       cu_stream = stream;

    return toHipFileError(cuFileWriteAsync(cu_fh, buffer_base, size_p, file_offset_p, buffer_offset_p,
                                           bytes_written_p, cu_stream));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileStreamRegister(hipStream_t stream, unsigned flags)
try {
    CUstream cu_stream = stream;
    return toHipFileError(cuFileStreamRegister(cu_stream, flags));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileStreamDeregister(hipStream_t stream)
try {
    CUstream cu_stream = stream;
    return toHipFileError(cuFileStreamDeregister(cu_stream));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t *value)
try {
    return toHipFileError(cuFileGetParameterSizeT(toCUFileSizeTConfigParameter(param), value));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileGetParameterBool(hipFileBoolConfigParameter_t param, bool *value)
try {
    return toHipFileError(cuFileGetParameterBool(toCUFileBoolConfigParameter(param), value));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileGetParameterString(hipFileStringConfigParameter_t param, char *desc_str, int len)
try {
    return toHipFileError(cuFileGetParameterString(toCUFileStringConfigParameter(param), desc_str, len));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value)
try {
    return toHipFileError(cuFileSetParameterSizeT(toCUFileSizeTConfigParameter(param), value));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value)
try {
    return toHipFileError(cuFileSetParameterBool(toCUFileBoolConfigParameter(param), value));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}

hipFileError_t
hipFileSetParameterString(hipFileStringConfigParameter_t param, const char *desc_str)
try {
    return toHipFileError(cuFileSetParameterString(toCUFileStringConfigParameter(param), desc_str));
}
catch (...) {
    return {hipFileInternalError, hipSuccess};
}
