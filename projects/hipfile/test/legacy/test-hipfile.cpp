/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* Test basic hipFile operations
 *
 * This is a temporary test until we get proper testing
 * set up.
 */

#include "hipfile.h"

#include <hip/hip_runtime_api.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

void static print_drv_props(hipFileDriverProps_t hipfile_props)
{
    fprintf(stderr, "Fetching hipFile Driver Properties.\n");
    fprintf(stderr, "    Version (Major.Minor): %u.%u\n", hipfile_props.nvfs.major_version,
            hipfile_props.nvfs.minor_version);
    fprintf(stderr, "    poll_thresh_size: %" PRIu64 "\n", hipfile_props.nvfs.poll_thresh_size);
    fprintf(stderr, "    max_direct_io_size: %" PRIu64 "\n", hipfile_props.nvfs.max_direct_io_size);
    fprintf(stderr, "    driver_status_flags: %u\n", hipfile_props.nvfs.driver_status_flags);
    fprintf(stderr, "    driver_status_flags: %u\n", hipfile_props.nvfs.driver_control_flags);
    fprintf(stderr, "  feature_flags: %u\n", hipfile_props.feature_flags);
    fprintf(stderr, "  max_device_cache_size: %" PRIu64 "\n", hipfile_props.max_device_cache_size);
    fprintf(stderr, "  per_buffer_cache_size: %" PRIu64 "\n", hipfile_props.per_buffer_cache_size);
    fprintf(stderr, "  max_device_pinned_mem_size: %" PRIu64 "\n", hipfile_props.max_device_pinned_mem_size);
    fprintf(stderr, "  max_batch_io_count: %u\n", hipfile_props.max_batch_io_count);
    fprintf(stderr, "  max_batch_io_timeout_msecs: %u\n", hipfile_props.max_batch_io_timeout_msecs);
}

int
main(int argc, char *argv[])
{
    hipFileHandle_t fh_in;
    hipFileHandle_t fh_out;
    hipFileDescr_t  fd_in;
    hipFileDescr_t  fd_out;
    hipFileError_t  err;
    const char     *fpath_out;
    const char     *fpath_in;
    struct stat     fstat_in;
    size_t          file_size; // hip functions use unsigned parameters
    ssize_t         bytes_written;
    ssize_t         bytes_read;
    void           *dbuf;
    int             rc = 0;

    // HIP Driver Properties
    int64_t  hipfile_used;
    uint64_t start_poll_thresh_size;
    uint64_t start_max_direct_io_size;
    uint64_t start_max_device_cache_size;
    uint64_t start_max_device_pinned_mem_size;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s FILE_IN FILE_OUT\n", argv[0]);
        exit(1);
    }

    fpath_in = argv[1];
    if (stat(fpath_in, &fstat_in)) {
        fprintf(stderr, "Could not stat %s: %s\n", fpath_in, strerror(errno));
        return 1;
    }
    file_size = static_cast<size_t>(fstat_in.st_size);

    memset(&fd_in, 0, sizeof(hipFileDescr_t));
    fd_in.type      = hipFileHandleTypeOpaqueFD;
    fd_in.handle.fd = open(fpath_in, O_RDONLY | O_DIRECT);
    if (fd_in.handle.fd < 0) {
        fprintf(stderr, "Could not open %s: %s\n", fpath_in, strerror(errno));
        return 1;
    }

    fpath_out = argv[2];
    memset(&fd_out, 0, sizeof(hipFileDescr_t));
    fd_out.type      = hipFileHandleTypeOpaqueFD;
    fd_out.handle.fd = open(fpath_out, O_WRONLY | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd_out.handle.fd < 0) {
        fprintf(stderr, "Could not open %s: %s\n", fpath_out, strerror(errno));
        rc = 1;
        goto close_fd_in;
    }
    /**
     * Begin GPU interactions.
     */
    if (hipMalloc(&dbuf, file_size) != hipSuccess) {
        fprintf(stderr, "Could not allocate buffer\n");
        rc = 1;
        goto close_fd_out;
    }

    err = hipFileDriverOpen();
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "hipFile IO Driver failed to open.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto free_dbuf;
    }

    err = hipFileHandleRegister(&fh_in, &fd_in);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to register IN file.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto free_dbuf;
    }

    err = hipFileHandleRegister(&fh_out, &fd_out);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to register OUT file.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto deregister_fh_in;
    }

    err = hipFileBufRegister(dbuf, file_size, 0);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to register GPU buffer.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto deregister_fh_out;
    }

    bytes_read = hipFileRead(fh_in, dbuf, file_size, 0, 0);
    if (bytes_read != fstat_in.st_size) {
        fprintf(stderr, "hipFileRead return value: %ld\n", bytes_read);
        fprintf(stderr, "Errno: %d - %s\n", errno, strerror(errno));
        rc = 1;
        goto deregister_dbuf;
    }

    bytes_written = hipFileWrite(fh_out, dbuf, file_size, 0, 0);
    if (bytes_written != fstat_in.st_size) {
        fprintf(stderr, "hipFileWrite return value: %ld\n", bytes_written);
        fprintf(stderr, "Errno: %d - %s\n", errno, strerror(errno));
        rc = 1;
        goto deregister_dbuf;
    }

    hipfile_used = hipFileUseCount();
    fprintf(stderr, "hipFileUseCount: %ld\n", hipfile_used);

    hipFileDriverProps_t hipfile_props;
    err = hipFileDriverGetProperties(&hipfile_props);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "hipFileDriverGetProperties failed.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto deregister_dbuf;
    }
    print_drv_props(hipfile_props);

    // Should be a multiple of 4K - value is in kb
    start_poll_thresh_size = hipfile_props.nvfs.poll_thresh_size;
    // Should be a multiple of 64K - value is in kb
    start_max_direct_io_size = hipfile_props.nvfs.max_direct_io_size;
    // Should be a multiple of 4K - value is in kb
    start_max_device_cache_size = hipfile_props.max_device_cache_size;
    // Should be a multiple of 4K - value is in kb
    start_max_device_pinned_mem_size = hipfile_props.max_device_pinned_mem_size;

    err = hipFileDriverSetPollMode(true, hipfile_props.nvfs.poll_thresh_size + 4);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "hipFileDriverSetPollMode failed.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto deregister_dbuf;
    }

    err = hipFileDriverSetMaxDirectIOSize(hipfile_props.nvfs.max_direct_io_size - 64);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "hipFileDriverSetMaxDirectIOSize failed.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto deregister_dbuf;
    }

    err = hipFileDriverSetMaxCacheSize(hipfile_props.max_device_cache_size + 4);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "hipFileDriverSetMaxCacheSize failed.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto deregister_dbuf;
    }

    err = hipFileDriverSetMaxPinnedMemSize(hipfile_props.max_device_pinned_mem_size + 4);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "hipFileDriverSetMaxPinnedMemSize failed.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto deregister_dbuf;
    }

    // At least on cgy-rowlet, the properties provided from cuFile do not
    // get updated with the new values as modified above.
    err = hipFileDriverGetProperties(&hipfile_props);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "2nd hipFileDriverGetProperties failed.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto deregister_dbuf;
    }

    if (hipfile_props.nvfs.poll_thresh_size == start_poll_thresh_size) {
        fprintf(stderr, "WARN: nvfs.poll_thresh_size unchanged.\n");
        fprintf(stderr, "      This is unusual, but not unexpected.\n");
    }
    if (hipfile_props.nvfs.max_direct_io_size == start_max_direct_io_size) {
        fprintf(stderr, "WARN: nvfs.max_direct_io_size unchanged.\n");
        fprintf(stderr, "      This is unusual, but not unexpected.\n");
    }
    if (hipfile_props.max_device_cache_size == start_max_device_cache_size) {
        fprintf(stderr, "WARN: max_device_cache_size unchanged.\n");
        fprintf(stderr, "      This is unusual, but not unexpected.\n");
    }
    if (hipfile_props.max_device_pinned_mem_size == start_max_device_pinned_mem_size) {
        fprintf(stderr, "WARN: max_device_pinned_mem_size unchanged.\n");
        fprintf(stderr, "      This is unusual, but not unexpected.\n");
    }

/**
 * Test cleanup.
 */
deregister_dbuf:
    err = hipFileBufDeregister(dbuf);
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to deregister GPU buffer.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
    }

deregister_fh_out:
    hipFileHandleDeregister(fh_out);

deregister_fh_in:
    hipFileHandleDeregister(fh_in);

    err = hipFileDriverClose();
    if (err.err != hipFileSuccess) {
        fprintf(stderr, "Failed to close hipFile IO Driver.\n");
        fprintf(stderr, "hipFileError: %d\n", err.err);
        fprintf(stderr, "hipError: %d\n", err.hip_drv_err);
        rc = 1;
        goto free_dbuf;
    }

free_dbuf:
    if (hipFree(dbuf) != hipSuccess) {
        rc = 1;
        fprintf(stderr, "Could not free device buffer\n");
    }

close_fd_out:
    if (0 != close(fd_out.handle.fd)) {
        rc = 1;
        fprintf(stderr, "Could not close %s: %s\n", fpath_out, strerror(errno));
    }

close_fd_in:
    if (0 != close(fd_in.handle.fd)) {
        rc = 1;
        fprintf(stderr, "Could not close %s: %s\n", fpath_in, strerror(errno));
    }
    return rc;
}
