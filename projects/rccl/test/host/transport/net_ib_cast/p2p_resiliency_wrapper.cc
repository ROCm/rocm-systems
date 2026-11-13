/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The real p2p_resiliency.cc, compiled as its own test compiles it, for the p2p_resiliency_recovery test's binary.
#include "P2pResiliencyScene.h"
#include "fakes/param_redirect.h"

#define clockNano P2pResiliencyTestClockNano
#include NET_IB_CAST_P2P_RESILIENCY_CC_PATH
#undef clockNano
