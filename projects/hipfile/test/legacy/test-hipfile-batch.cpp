/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* Test hipFile batch operations
 *
 * This is a temporary test until we get proper testing
 * set up.
 */

#include "hipfile.h"

#include "hipfile-warnings.h"

#include <hip/hip_runtime_api.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#define MAX_BATCH_IO_OPS 16
#define MAX_BUFFER_SIZE  4096

#define LOG_DEBUG 1

// Not fixing this already buggy piece of test software.
HIPFILE_WARN_FORMAT_NONLITERAL_OFF
void static debug_printf(const char *format, ...)
{
#if LOG_DEBUG
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
#endif
}
HIPFILE_WARN_FORMAT_NONLITERAL_ON

int
main(int argc, char *argv[])
{
    void                *dbufs[MAX_BATCH_IO_OPS];
    hipFileHandle_t      fh[MAX_BATCH_IO_OPS];
    hipFileDescr_t       fd[MAX_BATCH_IO_OPS];
    hipFileIOParams_t    batch_params[MAX_BATCH_IO_OPS];
    hipFileIOEvents_t    batch_events[MAX_BATCH_IO_OPS];
    hipFileBatchHandle_t batch_handle;
    hipFileError_t       err;
    hipError_t           hip_err;
    unsigned             batch_flags = 0;
    unsigned             batch_size;
    int                  input_batch_size;
    const char          *fin_path;
    const char          *fout_path;
    struct stat          fstat;
    int                  rc               = 0;
    unsigned             events_completed = 0;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s FILE_IN FILE_OUT BATCH_SIZE\n", argv[0]);
        exit(1);
    }

    fin_path = argv[1];
    if (stat(fin_path, &fstat)) {
        fprintf(stderr, "Could not stat %s: %s\n", fin_path, strerror(errno));
        return 1;
    }

    fout_path = argv[2];

    input_batch_size = atoi(argv[3]);
    if (input_batch_size <= 0 || input_batch_size > MAX_BATCH_IO_OPS) {
        fprintf(stderr, "batch_size must be in the range [1,%d], not %d.\n", MAX_BATCH_IO_OPS,
                input_batch_size);
        return 1;
    }
    batch_size = static_cast<unsigned>(input_batch_size);

    // Initialize Data Structs to make cleanup more reliable.
    memset(dbufs, 0, MAX_BATCH_IO_OPS * sizeof(void *));
    memset(fh, 0, MAX_BATCH_IO_OPS * sizeof(hipFileHandle_t));
    memset(fd, 0, MAX_BATCH_IO_OPS * sizeof(hipFileDescr_t));
    memset(batch_params, 0, MAX_BATCH_IO_OPS * sizeof(hipFileIOParams_t));
    memset(batch_events, 0, MAX_BATCH_IO_OPS * sizeof(hipFileIOEvents_t));
    for (unsigned i = 0; i < batch_size; i++) {
        fd[i].handle.fd = -1;
    }

    // Allocate buffers for GPU IO
    for (unsigned i = 0; i < batch_size; i++) {
        debug_printf("hipMalloc %d\n", i);
        hip_err = hipMalloc(&dbufs[i], MAX_BUFFER_SIZE);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "Could not allocate buffer: %d.\n", hip_err);
            return 1;
        }
        hip_err = hipMemset(dbufs[i], 0, MAX_BUFFER_SIZE);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "Could not zero-out buffer: %d\n", hip_err);
            rc = 1;
            goto free_dbufs;
        }
        debug_printf("hipFileBufRegister %d\n", i);
        err = hipFileBufRegister(dbufs[i], MAX_BUFFER_SIZE, 0);
        if (err.err != hipFileSuccess) {
            fprintf(stderr, "Failed to register GPU buffer.\n");
            fprintf(stderr, "hipFileError: %d\n", err.err);
            fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
            rc = 1;
            goto free_dbufs;
        }
    }

    debug_printf("hipFileDriverOpen\n");
    err = hipFileDriverOpen();
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to open GPU IO Driver.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto free_dbufs;
    }

    // Setup Batch FD's.
    for (unsigned i = 0; i < batch_size; i++) {
        debug_printf("open %d\n", i);
        fd[i].handle.fd = open(fin_path, O_RDONLY | O_DIRECT);
        if (fd[i].handle.fd < 0) {
            fprintf(stderr, "Could not open %s: %s\n", fin_path, strerror(errno));
            rc = 1;
            goto close_files;
        }
        fd[i].type = hipFileHandleTypeOpaqueFD;

        debug_printf("hipFileHandleRegister %d\n", i);
        err = hipFileHandleRegister(&fh[i], &fd[i]);
        if (err.err != hipFileSuccess) {
            fprintf(stderr, "Failed to register fd[%u].\n", i);
            fprintf(stderr, "hipFileError: %d\n", err.err);
            fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
            rc = 1;
            goto close_files;
        }

        // Setup the Batch IO Params to first read from the file.
        batch_params[i].mode                  = hipFileBatch;
        batch_params[i].fh                    = fh[i];
        batch_params[i].u.batch.devPtr_base   = dbufs[i];
        batch_params[i].u.batch.file_offset   = i * 16;
        batch_params[i].u.batch.devPtr_offset = 0;
        batch_params[i].u.batch.size          = 16;
        batch_params[i].opcode                = hipFileBatchRead;
        batch_params[i].cookie                = &batch_params[i];
    }

    // Setup and Execute our Batch Read operation.
    debug_printf("hipFileBatchIOSetup\n");
    err = hipFileBatchIOSetUp(&batch_handle, batch_size);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to setup Read Batch IO.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto close_files;
    }
    debug_printf("hipFileBatchIOSubmit\n");
    err = hipFileBatchIOSubmit(batch_handle, batch_size, batch_params, batch_flags);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to submit Read Batch IO.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto close_files;
    }

    // Get Batch Events to confirm data read successfully.
    events_completed = 0;
    while (events_completed != batch_size) {
        memset(batch_events, 0, MAX_BATCH_IO_OPS * sizeof(hipFileIOEvents_t));
        unsigned num_events_reported = batch_size;
        err = hipFileBatchIOGetStatus(batch_handle, batch_size, &num_events_reported, batch_events, nullptr);
        if (err.err != hipFileSuccess) {
            fprintf(stderr, "Failed to fetch Read Batch IO status.\n");
            fprintf(stderr, "hipFileError: %d\n", err.err);
            fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
            goto finish_batch_io;
        }
        debug_printf("Received %d Batch IO Events.\n", num_events_reported);
        events_completed += num_events_reported;
        for (unsigned i = 0; i < num_events_reported; i++) {
            // At this point the example copies the data into a Host buffer.
            // For this, we are just going to perform a nominal check on the
            // members within hipFileIOEvents_t.
            hipFileIOEvents_t event = batch_events[i];
            fprintf(stderr, "hipFileIOEvent #%u\n", i);
            fprintf(stderr, "Status: %d -- Return Value: %zu\n", event.status, event.ret);
            fprintf(stderr, "Cookie: %p\n", event.cookie);
            bool match = (event.cookie == &batch_params[i]);
            fprintf(stderr, "Cookie ?= hipFileIOParams: %p ?= %p : %s\n", event.cookie,
                    reinterpret_cast<void *>(&batch_params[i]), match ? "true" : "false");
        }
    }

    hipFileBatchIODestroy(batch_handle);
    // Ideally we can write-back the data here to an output file.
    // Setup Write Batch FD's.

    // Setup Write Batch FD's.
    // We are reusing our previous data structures. So we will free and
    // reinitialize for the write step.
    for (unsigned i = 0; i < batch_size; i++) {
        debug_printf("hipFileHandleDeregister %d\n", i);
        hipFileHandleDeregister(fh[i]);

        debug_printf("close %d\n", i);
        close(fd[i].handle.fd);

        debug_printf("open output file %d\n", i);
        fd[i].handle.fd = open(fout_path, O_RDWR | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        if (fd[i].handle.fd < 0) {
            fprintf(stderr, "Could not open %s: %s\n", fout_path, strerror(errno));
            rc = 1;
            goto close_files;
        }
        fd[i].type = hipFileHandleTypeOpaqueFD;

        debug_printf("hipFileHandleRegister %d\n", i);
        err = hipFileHandleRegister(&fh[i], &fd[i]);
        if (err.err != hipFileSuccess) {
            fprintf(stderr, "Failed to register fd[%u].\n", i);
            fprintf(stderr, "hipFileError: %d\n", err.err);
            fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
            rc = 1;
            goto close_files;
        }

        // Setup the Batch IO Params to write to the file.
        batch_params[i].mode                  = hipFileBatch;
        batch_params[i].fh                    = fh[i];
        batch_params[i].u.batch.devPtr_base   = dbufs[i];
        batch_params[i].u.batch.file_offset   = i * 16;
        batch_params[i].u.batch.devPtr_offset = 0;
        batch_params[i].u.batch.size          = 16;
        batch_params[i].opcode                = hipFileBatchWrite;
    }

    // Setup and Execute our Batch Write operation.
    debug_printf("hipFileBatchIOSetup\n");
    err = hipFileBatchIOSetUp(&batch_handle, batch_size);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to setup Write Batch IO.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto close_files;
    }
    debug_printf("hipFileBatchIOSubmit\n");
    err = hipFileBatchIOSubmit(batch_handle, batch_size, batch_params, batch_flags);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to submit Write Batch IO.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto close_files;
    }

    // Get Batch Events to confirm data read successfully.
    events_completed = 0;
    while (events_completed != batch_size) {
        memset(batch_events, 0, MAX_BATCH_IO_OPS * sizeof(hipFileIOEvents_t));
        unsigned num_events_reported = batch_size;
        err = hipFileBatchIOGetStatus(batch_handle, batch_size, &num_events_reported, batch_events, nullptr);
        if (err.err != hipFileSuccess) {
            fprintf(stderr, "Failed to fetch Write Batch IO status.\n");
            fprintf(stderr, "hipFileError: %d\n", err.err);
            fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
            goto finish_batch_io;
        }
        debug_printf("Received %d Batch IO Events.\n", num_events_reported);
        events_completed += num_events_reported;
        for (unsigned i = 0; i < num_events_reported; i++) {
            // At this point the example copies the data into a Host buffer.
            // For this, we are just going to perform a nominal check on the
            // members within hipFileIOEvents_t.
            hipFileIOEvents_t event = batch_events[i];
            fprintf(stderr, "hipFileIOEvent #%u\n", i);
            fprintf(stderr, "Status: %d -- Return Value: %zu\n", event.status, event.ret);
        }
    }

finish_batch_io:
    // Only needs to be run if hipFileBatchIOSubmit was successful.
    debug_printf("hipFileBatchIODestroy\n");
    hipFileBatchIODestroy(batch_handle);

close_files:
    for (unsigned i = 0; i < batch_size; i++) {
        if (fd[i].handle.fd >= 0) {
            if (fh[i] != nullptr) {
                debug_printf("hipFileHandleDeregister %u\n", i);
                hipFileHandleDeregister(fh[i]);
                if (err.err != hipFileSuccess) {
                    fprintf(stderr, "Failed to deregister fd[%u].\n", i);
                    fprintf(stderr, "hipFileError: %d\n", err.err);
                    fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
                    rc = 1;
                }
            }
            debug_printf("close %d\n", i);
            int close_status = close(fd[i].handle.fd);
            if (close_status) {
                fprintf(stderr, "Failed to close fd[%u]: %s\n", i, strerror(errno));
                rc = 1;
            }
        }
    }

    debug_printf("hipFileDriverClose\n");
    err = hipFileDriverClose();
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to close GPU IO Driver.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
    }

free_dbufs:
    for (unsigned i = 0; i < batch_size; i++) {
        debug_printf("hipFileBufDeregister %u\n", i);
        err = hipFileBufDeregister(dbufs[i]);
        if (err.err != hipFileSuccess) {
            fprintf(stderr, "Failed to deregister GPU buffer %u.\n", i);
            fprintf(stderr, "hipFileError: %d\n", err.err);
            fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
            rc = 1;
        }

        if (dbufs[i] != nullptr) {
            debug_printf("hipFree %u\n", i);
            hip_err = hipFree(dbufs[i]);
            if (hip_err != hipSuccess) {
                fprintf(stderr, "Could not free device buffer %u\n", i);
                rc = 1;
            }
        }
    }

    return rc;
}
