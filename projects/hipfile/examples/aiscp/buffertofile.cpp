/*
 * Copyright of Aristotle TODO.  All rights reserved.

 */

/*
 * Gpu to File memcpy.
 * This is an example program moving data from GPU memory to a file


//TODO 
 * For verification, input data has a pattern.
 * User can verify the output file's data after write using:
 * hexdump -C <dst_path>
 * 0000000 abab abab abab abab abab abab abab abab  |................|
 *
 * ./bufregister_write <dst_path> <gpu_id>
 *
 * | Output |
 * cuFileWrite with device memory registration
 * Open file: <dst_path> for writing
 * Allocate device memory of size: 131072 on GPU ID: <gpu_id>
 * Register device memory of size: 131072
 * Write from device memory
 * Written bytes: 131072
 * Deregister device memory
 */


#include <hipfile.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#define CHECK_HIP(condition) {                                            \
    hipError_t error = condition;                                         \
    if(error != hipSuccess){                                              \
        fprintf(stderr, "HIP error: %d line: %s:%d\n",                    \
                error, __FILE__,  __LINE__);                              \
        exit(error);                                                      \
    }                                                                     \
}

/// @brief Open and register a file
/// @param path [in] Path to the file
/// @param flags [in] flags: Flags to pass to open (2)
/// @param mode [in] mode: Mode to pass to open (2)
/// @param fd [out] fd: The file descriptor of the opened file
/// @param handle [out] handle: The handle to use with hipFile APIs
/// @return zero on success, non-zero on failure
static int
open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
{
    hipFileError_t hipfile_err;
    hipFileDescr_t descr;

    *fd = open(path, flags | O_DIRECT, mode);
    if (-1 == *fd) {
        fprintf(stderr, "Could not open %s (%s)\n", path, strerror(errno));
        return 1;
    }

    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = *fd;

    hipfile_err = hipFileHandleRegister(handle, &descr);
    if (hipFileSuccess != hipfile_err.err) {
        fprintf(stderr, "Could not register %s (%s)\n", path, hipFileGetOpErrorString(hipfile_err.err));
        close(*fd);
        return 1;
    }

    return 0;
}

/// @brief Unregister and close a file
/// @param path [in] Path to the file
/// @param fd [in] The file descriptor of the opened file
/// @param handle [in] The handle of the opened file
/// @return zero on success, non-zero on failure
static int
close_file(const char *path, int fd, hipFileHandle_t handle)
{
    hipFileHandleDeregister(handle);
    if (-1 == close(fd)) {
        fprintf(stderr, "Could not close %s (%s)\n", path, strerror(errno));
        return 1;
    }
    return 0;
}


//main TODO DOC
static constexpr size_t SIZE = 8192;

int
main(int argc, char *argv[])
{
    const char     *dst_path;
    int             dst_fd, gpu_id=-1;
    hipFileHandle_t dst_handle;
    // void           *devbuf;
    // hipError_t      hip_err;
    hipFileError_t  exit_status;
    // size_t          buffer_size, file_size, block_size;
    ssize_t         nread{};

	int ret = EXIT_SUCCESS;
	void *devPtr = nullptr;
    // hoff_t          file_offset{};

    if (argc != 3) {
        fprintf(stderr, "Usage: %s FILE_DEST GPU\n", argv[0]);
        exit(1);
    }

    dst_path = argv[1];
    gpu_id = std::atoi(argv[2]);
   	CHECK_HIP(hipSetDevice(gpu_id));




	// Opens a file to write
	if (open_file(dst_path, O_WRONLY | O_CREAT, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH, &dst_fd,
                  &dst_handle)) {
		ret = EXIT_FAILURE;
        goto program_exit;
    }
	
	// Allocate device memory and fill with 0xab

	fprintf(stdout, "Allocate device memory\n");
	CHECK_HIP(hipMalloc(&devPtr, SIZE));
	CHECK_HIP(hipMemset(static_cast<void*>(devPtr), 0xab, SIZE));
	CHECK_HIP(hipStreamSynchronize(nullptr));

	// Registers device memory
	exit_status = hipFileBufRegister(devPtr, SIZE, 0);
	if (exit_status.err != hipFileSuccess) {
		fprintf(stderr, "Buffer register failed: %s\n", HIPFILE_ERRSTR(exit_status.err));
		ret = EXIT_FAILURE;
		goto close_file;
	}

	// Writes device memory contents to a file
	fprintf(stdout, "Write from device memory\n");
	nread = hipFileWrite(dst_handle, devPtr, SIZE, 0, 0);
	if (nread < 0) {
		fprintf(stderr, "Could not write in %s (%s)\n", dst_path, IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : strerror(errno));
		ret = EXIT_FAILURE;
	} else {

		fprintf(stdout, "Written bytes %ld\n", nread);
		ret = EXIT_SUCCESS;
	}

	// Deregister the device memory
	std::cout << "Deregister device memory" << std::endl;
	exit_status = hipFileBufDeregister(devPtr);
	if (exit_status.err != hipFileSuccess) {
		fprintf(stderr, "CBuffer deregister failed:  %s\n", IS_HIPFILE_ERR(exit_status.err) ? HIPFILE_ERRSTR(exit_status.err) : strerror(errno));
		ret = EXIT_FAILURE;
	}

// Cleanup labels
close_file:
    if (close_file(dst_path, dst_fd, dst_handle)) {
        ret = EXIT_FAILURE;
    }
	// Free the device memory
	if (devPtr) CHECK_HIP(hipFree(devPtr));
program_exit:

	return ret;
}
