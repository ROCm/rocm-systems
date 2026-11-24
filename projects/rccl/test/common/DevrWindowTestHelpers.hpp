/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef RCCL_TEST_DEVR_WINDOW_TEST_HELPERS_HPP
#define RCCL_TEST_DEVR_WINDOW_TEST_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include "MPIEnvironment.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "dev_runtime_internal.h"

// Segment count and memory kind from the registered window's memory record.
inline void expectDevrWindowSegments(ncclComm_t comm, void* userPtr, int numSegments, bool hasSysmem)
{
    struct ncclDevrWindow* devrWin = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclDevrFindWindow(reinterpret_cast<struct ncclComm*>(comm), userPtr, &devrWin));
    ASSERT_MPI_NE(devrWin, nullptr);
    ASSERT_MPI_NE(devrWin->memory, nullptr);
    ASSERT_MPI_EQ(numSegments, devrWin->memory->numSegments);
    ASSERT_MPI_EQ(hasSysmem, devrWin->memory->hasSysmemSegment);
}

#endif // MPI_TESTS_ENABLED
#endif // RCCL_TEST_DEVR_WINDOW_TEST_HELPERS_HPP
