/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* Compiled as C11, and as C++11 from a copy in the build directory, to
 * check that hipfile.h is valid in both languages (see CMakeLists.txt).
 * hipfile.h is included first to check that it's self-contained.
 */

#include <hipfile.h>

/* A macro is only checked where it's expanded, so this uses every
 * macro hipfile.h defines
 */
int hipfile_header_check_macros[] = {
    HIPFILE_VERSION_MAJOR,
    HIPFILE_VERSION_MINOR,
    HIPFILE_VERSION_PATCH,
    HIPFILE_BASE_ERR,
    HIPFILE_RDMA_REGISTER,
    HIPFILE_RDMA_RELAXED_ORDERING,
    HIPFILE_STREAM_FIXED_BUF_OFFSET,
    HIPFILE_STREAM_FIXED_FILE_OFFSET,
    HIPFILE_STREAM_FIXED_FILE_SIZE,
    HIPFILE_STREAM_PAGE_ALIGNED_INPUTS,
    HIPFILE_STREAM_FLAGS_MASK,
    HIPFILE_MAX_GPUS,
    HIPFILE_GPU_UUID_LEN,
};

/* The function-like macros, which take arguments, so they can't go in
 * the array.
 *
 * op_err is an int, since IS_HIPFILE_ERR() and HIPFILE_ERRSTR() call
 * abs(), which takes an int in C. A hipFileOpError_t is unsigned, so
 * gcc and clang warn that abs() has no effect, and a ssize_t would be
 * truncated.
 */
int hipfile_header_check_function_macros(int op_err, hipFileError_t err);

int
hipfile_header_check_function_macros(int op_err, hipFileError_t err)
{
    return IS_HIPFILE_ERR(op_err) && HIPFILE_ERRSTR(op_err) != NULL && IS_HIP_DRV_ERR(err) &&
           HIP_DRV_ERR(err) != hipSuccess;
}
