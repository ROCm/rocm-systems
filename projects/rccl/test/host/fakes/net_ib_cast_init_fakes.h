/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols src/transport/net_ib_cast/init.cc defines and other
// net_ib_cast TUs read: the transport-wide flags as plain globals that tests
// flip directly (common_cast.h declares them), and the params through
// g_loadParam. Nothing to reset beyond the flags.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_INIT_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_INIT_FAKES_H_

// Restores every flag to its production initial value.
void ResetNetIbCastInitFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_INIT_FAKES_H_
