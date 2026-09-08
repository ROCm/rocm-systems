/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPIHostUtils.cc
 * @brief Host utilities for MPI test binaries linking librccl.so.
 *
 * getHostName() lives in librccl but is not exported from the shared library,
 * while MPI test headers call it directly. Provide a local copy for slim MPI
 * test executables (e.g. rccl-HostApiCustomGinPluginMPI).
 */

#ifdef MPI_TESTS_ENABLED

#include "utils.h"

#include <cstring>
#include <unistd.h>

ncclResult_t getHostName(char* hostname, int maxlen, const char delim)
{
    if(gethostname(hostname, maxlen) != 0)
    {
        strncpy(hostname, "unknown", maxlen);
        return ncclSystemError;
    }
    int i = 0;
    while((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1))
        i++;
    hostname[i] = '\0';
    return ncclSuccess;
}

#endif // MPI_TESTS_ENABLED
