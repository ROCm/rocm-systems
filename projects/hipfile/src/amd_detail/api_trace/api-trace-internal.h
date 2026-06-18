/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile-api-trace.h"

// Internal header for the dispatch-table layer in src/amd_detail/api_trace.
//
// Do NOT include this from anything other than the api_trace .cpp files
// (and, where needed, from .cpp files that *define* one of these runtime
// implementations, such as hipfile.cpp). The in-namespace declarations of
// the public hipFile API collide with the extern "C" declarations from
// <hipfile.h> when both become visible through `using namespace hipFile;`,
// which is common in tests.

namespace hipFile {

// Accessor for the singleton dispatch table; defined in api-trace.cpp.
const hipFileDispatchTable *GetHipFileDispatchTable(void);

// Runtime implementations of the public hipFile API. The extern "C" entry
// points in api-dispatch-interface.cpp dispatch into these via the
// hipFileDispatchTable populated in api-trace.cpp.
const char    *hipFileGetOpErrorString(hipFileOpError_t status);
hipFileError_t hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr);
void           hipFileHandleDeregister(hipFileHandle_t fh);
hipFileError_t hipFileBufRegister(const void *buffer_base, size_t length, int flags);
hipFileError_t hipFileBufDeregister(const void *buffer_base);
ssize_t        hipFileRead(hipFileHandle_t fh, void *buffer_base, size_t size, hoff_t file_offset,
                           hoff_t buffer_offset);
ssize_t        hipFileWrite(hipFileHandle_t fh, const void *buffer_base, size_t size, hoff_t file_offset,
                            hoff_t buffer_offset);
hipFileError_t hipFileDriverOpen(void);
hipFileError_t hipFileDriverClose(void);
int64_t        hipFileUseCount(void);
hipFileError_t hipFileDriverGetProperties(hipFileDriverProps_t *props);
hipFileError_t hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size);
hipFileError_t hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size);
hipFileError_t hipFileDriverSetMaxCacheSize(size_t max_cache_size);
hipFileError_t hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size);
hipFileError_t hipFileBatchIOSetUp(hipFileBatchHandle_t *batch_idp, unsigned max_nr);
hipFileError_t hipFileBatchIOSubmit(hipFileBatchHandle_t batch_idp, unsigned nr, hipFileIOParams_t *iocbp,
                                    unsigned flags);
hipFileError_t hipFileBatchIOGetStatus(hipFileBatchHandle_t batch_idp, unsigned min_nr, unsigned *nr,
                                       hipFileIOEvents_t *iocbp, struct timespec *timeout);
hipFileError_t hipFileBatchIOCancel(hipFileBatchHandle_t batch_idp);
void           hipFileBatchIODestroy(hipFileBatchHandle_t batch_idp);
hipFileError_t hipFileReadAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                                hoff_t *buffer_offset_p, ssize_t *bytes_read_p, hipStream_t stream);
hipFileError_t hipFileWriteAsync(hipFileHandle_t fh, void *buffer_base, size_t *size_p, hoff_t *file_offset_p,
                                 hoff_t *buffer_offset_p, ssize_t *bytes_written_p, hipStream_t stream);
hipFileError_t hipFileStreamRegister(hipStream_t stream, unsigned flags);
hipFileError_t hipFileStreamDeregister(hipStream_t stream);
hipFileError_t hipFileGetVersion(unsigned *major, unsigned *minor, unsigned *patch);
hipFileError_t hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t *value);
hipFileError_t hipFileGetParameterBool(hipFileBoolConfigParameter_t param, bool *value);
hipFileError_t hipFileGetParameterString(hipFileStringConfigParameter_t param, char *desc_str, int len);
hipFileError_t hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value);
hipFileError_t hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value);
hipFileError_t hipFileSetParameterString(hipFileStringConfigParameter_t param, const char *desc_str);

} // namespace hipFile
