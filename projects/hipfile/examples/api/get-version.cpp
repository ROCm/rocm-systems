/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

/* get-version - Get hipFile version info
 *
 * Usage: ./get-version
 */

#include <hipfile.h>

#include <cstdio>
#include <cstdlib>

int
main(void)
{
    unsigned       major = 0;
    unsigned       minor = 0;
    unsigned       patch = 0;
    hipFileError_t err;

    printf("\n");

    /* hipFile #defines symbols for the major, minor, and version numbers
     * which are useful for #ifdefs at compile time.
     */
    printf("Version from the header symbols (major.minor.patch): %d.%d.%d\n", HIPFILE_VERSION_MAJOR,
           HIPFILE_VERSION_MINOR, HIPFILE_VERSION_PATCH);
    printf("\n");

    /* You can also get the version numbers at runtime */
    err = hipFileGetVersion(&major, &minor, &patch);
    if (err.err != hipFileSuccess || err.hip_drv_err != hipSuccess)
        return EXIT_FAILURE;
    printf("Version from hipFileGetVersion() (major.minor.patch): %u.%u.%u\n", major, minor, patch);
    printf("\n");

    return EXIT_SUCCESS;
}
