/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_DDA_IPC_NRANKS_FAKES_H_
#define RCCL_TEST_HOST_DDA_IPC_NRANKS_FAKES_H_

// For the binaries that #include a DDA IPC launcher (dda_*_ipc.cu) but not
// ipc_init.cu, which owns ncclDdaIpcNranksSupported(). Default false = relax off,
// the stock behaviour: only the full 8-rank clique is supported. Set true to model
// RCCL_DDA_NRANKS_RELAX=1, which admits any 2..8 ranks.
extern bool g_ddaIpcNranksRelaxed;

#endif  // RCCL_TEST_HOST_DDA_IPC_NRANKS_FAKES_H_
