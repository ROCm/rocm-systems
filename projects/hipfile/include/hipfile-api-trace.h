/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

// The major version number should ideally remain unchanged. Increment the
// major version only for fundamental changes to the hipFileDispatchTable
// struct, such as altering the type or name of an existing member variable.
#define HIPFILE_RUNTIME_API_TABLE_MAJOR_VERSION 0

// Increment the step version when new runtime API functions are added.
// If the corresponding table version increases, reset the step version
// to zero.
#define HIPFILE_RUNTIME_API_TABLE_STEP_VERSION 0

// hipFile API interface
typedef const char *(*PfnHipFileGetOpErrorString)(hipFileOpError_t status);
typedef hipFileError_t (*PfnHipFileHandleRegister)(hipFileHandle_t *fh, hipFileDescr_t *descr);
typedef void (*PfnHipFileHandleDeregister)(hipFileHandle_t fh);
typedef hipFileError_t (*PfnHipFileBufRegister)(const void *buffer_base, size_t length, int flags);
typedef hipFileError_t (*PfnHipFileBufDeregister)(const void *buffer_base);
typedef ssize_t (*PfnHipFileRead)(hipFileHandle_t fh, void *buffer_base, size_t size, hoff_t file_offset,
                                  hoff_t buffer_offset);
typedef ssize_t (*PfnHipFileWrite)(hipFileHandle_t fh, const void *buffer_base, size_t size,
                                   hoff_t file_offset, hoff_t buffer_offset);
typedef hipFileError_t (*PfnHipFileDriverOpen)(void);
typedef hipFileError_t (*PfnHipFileDriverClose)(void);
typedef int64_t (*PfnHipFileUseCount)(void);
typedef hipFileError_t (*PfnHipFileDriverGetProperties)(hipFileDriverProps_t *props);
typedef hipFileError_t (*PfnHipFileDriverSetPollMode)(bool poll, size_t poll_threshold_size);
typedef hipFileError_t (*PfnHipFileDriverSetMaxDirectIOSize)(size_t max_direct_io_size);
typedef hipFileError_t (*PfnHipFileDriverSetMaxCacheSize)(size_t max_cache_size);
typedef hipFileError_t (*PfnHipFileDriverSetMaxPinnedMemSize)(size_t max_pinned_size);
typedef hipFileError_t (*PfnHipFileBatchIOSetUp)(hipFileBatchHandle_t *batch_idp, unsigned max_nr);
typedef hipFileError_t (*PfnHipFileBatchIOSubmit)(hipFileBatchHandle_t batch_idp, unsigned nr,
                                                  hipFileIOParams_t *iocbp, unsigned flags);
typedef hipFileError_t (*PfnHipFileBatchIOGetStatus)(hipFileBatchHandle_t batch_idp, unsigned min_nr,
                                                     unsigned *nr, hipFileIOEvents_t *iocbp,
                                                     struct timespec *timeout);
typedef hipFileError_t (*PfnHipFileBatchIOCancel)(hipFileBatchHandle_t batch_idp);
typedef void (*PfnHipFileBatchIODestroy)(hipFileBatchHandle_t batch_idp);
typedef hipFileError_t (*PfnHipFileReadAsync)(hipFileHandle_t fh, void *buffer_base, size_t *size_p,
                                              hoff_t *file_offset_p, hoff_t *buffer_offset_p,
                                              ssize_t *bytes_read_p, hipStream_t stream);
typedef hipFileError_t (*PfnHipFileWriteAsync)(hipFileHandle_t fh, void *buffer_base, size_t *size_p,
                                               hoff_t *file_offset_p, hoff_t *buffer_offset_p,
                                               ssize_t *bytes_written_p, hipStream_t stream);
typedef hipFileError_t (*PfnHipFileStreamRegister)(hipStream_t stream, unsigned flags);
typedef hipFileError_t (*PfnHipFileStreamDeregister)(hipStream_t stream);
typedef hipFileError_t (*PfnHipFileGetVersion)(unsigned *major, unsigned *minor, unsigned *patch);
typedef hipFileError_t (*PfnHipFileGetParameterSizeT)(hipFileSizeTConfigParameter_t param, size_t *value);
typedef hipFileError_t (*PfnHipFileGetParameterBool)(hipFileBoolConfigParameter_t param, bool *value);
typedef hipFileError_t (*PfnHipFileGetParameterString)(hipFileStringConfigParameter_t param, char *desc_str,
                                                       int len);
typedef hipFileError_t (*PfnHipFileSetParameterSizeT)(hipFileSizeTConfigParameter_t param, size_t value);
typedef hipFileError_t (*PfnHipFileSetParameterBool)(hipFileBoolConfigParameter_t param, bool value);
typedef hipFileError_t (*PfnHipFileSetParameterString)(hipFileStringConfigParameter_t param,
                                                       const char                    *desc_str);

// hipFile API dispatch table
struct hipFileDispatchTable {
    // HIPFILE_RUNTIME_API_TABLE_STEP_VERSION == 0
    size_t                              size;
    PfnHipFileGetOpErrorString          pfn_hipfile_get_op_error_string;
    PfnHipFileHandleRegister            pfn_hipfile_handle_register;
    PfnHipFileHandleDeregister          pfn_hipfile_handle_deregister;
    PfnHipFileBufRegister               pfn_hipfile_buf_register;
    PfnHipFileBufDeregister             pfn_hipfile_buf_deregister;
    PfnHipFileRead                      pfn_hipfile_read;
    PfnHipFileWrite                     pfn_hipfile_write;
    PfnHipFileDriverOpen                pfn_hipfile_driver_open;
    PfnHipFileDriverClose               pfn_hipfile_driver_close;
    PfnHipFileUseCount                  pfn_hipfile_use_count;
    PfnHipFileDriverGetProperties       pfn_hipfile_driver_get_properties;
    PfnHipFileDriverSetPollMode         pfn_hipfile_driver_set_poll_mode;
    PfnHipFileDriverSetMaxDirectIOSize  pfn_hipfile_driver_set_max_direct_io_size;
    PfnHipFileDriverSetMaxCacheSize     pfn_hipfile_driver_set_max_cache_size;
    PfnHipFileDriverSetMaxPinnedMemSize pfn_hipfile_driver_set_max_pinned_mem_size;
    PfnHipFileBatchIOSetUp              pfn_hipfile_batch_io_set_up;
    PfnHipFileBatchIOSubmit             pfn_hipfile_batch_io_submit;
    PfnHipFileBatchIOGetStatus          pfn_hipfile_batch_io_get_status;
    PfnHipFileBatchIOCancel             pfn_hipfile_batch_io_cancel;
    PfnHipFileBatchIODestroy            pfn_hipfile_batch_io_destroy;
    PfnHipFileReadAsync                 pfn_hipfile_read_async;
    PfnHipFileWriteAsync                pfn_hipfile_write_async;
    PfnHipFileStreamRegister            pfn_hipfile_stream_register;
    PfnHipFileStreamDeregister          pfn_hipfile_stream_deregister;
    PfnHipFileGetVersion                pfn_hipfile_get_version;
    PfnHipFileGetParameterSizeT         pfn_hipfile_get_parameter_size_t;
    PfnHipFileGetParameterBool          pfn_hipfile_get_parameter_bool;
    PfnHipFileGetParameterString        pfn_hipfile_get_parameter_string;
    PfnHipFileSetParameterSizeT         pfn_hipfile_set_parameter_size_t;
    PfnHipFileSetParameterBool          pfn_hipfile_set_parameter_bool;
    PfnHipFileSetParameterString        pfn_hipfile_set_parameter_string;
    // PLEASE DO NOT EDIT ABOVE!

    // HIPFILE_RUNTIME_API_TABLE_STEP_VERSION == 1

    // *******************************************************************************************
    //                                            READ BELOW
    // *******************************************************************************************
    // Please keep this text at the end of the structure:

    // 1. Do not reorder any existing members
    // 2. Increase the step version definition before adding new members
    // 3. Insert new members under the appropriate step version comment
    // 4. Generate a comment for the next step version
    // 5. Add a "PLEASE DO NOT EDIT ABOVE!" comment
    // *******************************************************************************************
};
