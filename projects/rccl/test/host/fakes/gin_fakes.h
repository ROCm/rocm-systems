/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the GIN entry points src/plugin/gin.cc and src/gin/gin_host.cc own.

#ifndef RCCL_TEST_HOST_GIN_FAKES_H_
#define RCCL_TEST_HOST_GIN_FAKES_H_

#include <functional>

#include "nccl.h"
#include "nccl_device/core.h"  // ncclGinType_t

struct ncclComm;
struct ncclGinState;

extern ncclResult_t g_ncclGinInitResult;
extern bool g_ginHasError;
// What ncclGetRailedGinType reports; NCCL_GIN_TYPE_NONE by default.
extern std::function<ncclGinType_t(ncclComm*)> g_railedGinType;

void ResetGinFakes();

#endif  // RCCL_TEST_HOST_GIN_FAKES_H_
