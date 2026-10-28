/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_CHANNEL_FAKES_H_
#define RCCL_TEST_HOST_FAKES_CHANNEL_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;

// The fake initChannel does not allocate ring storage; callers must supply it.
extern ncclResult_t g_initChannelResult;
extern int g_initChannelLastId;

// src/channel.cc CollNet per-channel setup. Fail-loud by default.
extern std::function<ncclResult_t(struct ncclComm*, int /*channelId*/, struct ncclComm* /*parent*/, bool /*share*/)>
    g_initCollnetChannel;

void ResetChannelFakes();

#endif  // RCCL_TEST_HOST_FAKES_CHANNEL_FAKES_H_
