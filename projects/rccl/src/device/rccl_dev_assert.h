/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_DEV_ASSERT_H_
#define RCCL_DEV_ASSERT_H_

// HIP's assert()/__assert_fail is implemented with ockl fprintf (device
// hostcall). That attaches the hostcall ABI to every kernel containing one,
// which requires PCIe atomics at launch even when the condition never fails,
// so collectives die on hosts without AtomicOp support (ROCM-23518).
#if defined(NDEBUG)
#define RCCL_DEV_ASSERT(cond) ((void)0)
#else
#define RCCL_DEV_ASSERT(cond) \
  do { \
    if (!(cond)) __builtin_trap(); \
  } while (0)
#endif

#endif
