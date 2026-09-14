/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_SOCKET_FAKES_H_
#define RCCL_TEST_HOST_SOCKET_FAKES_H_

#include <functional>

#include "nccl.h"
#include "socket.h"

extern std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, int*)> g_socketProgress;

void ResetSocketFakes();

#endif  // RCCL_TEST_HOST_SOCKET_FAKES_H_
