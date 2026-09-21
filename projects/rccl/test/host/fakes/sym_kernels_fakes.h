/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for src/sym_kernels.cc.

#ifndef RCCL_TEST_HOST_SYM_KERNELS_FAKES_H_
#define RCCL_TEST_HOST_SYM_KERNELS_FAKES_H_

#include <cstddef>
#include <functional>

#include "nccl.h"
#include "sym_kernels.h"  // ncclSymRegType_t

// Symmetric-registration query (sym_kernels.cc:762). Defaults to "neither side
// registered", but that is a CHOICE that steers production down one arm, so it
// is an explicit, counted, overridable seam rather than a fixed stub result.
extern ncclSymRegType_t g_symRegType;
extern ncclResult_t g_getSymRegTypeResult;
extern int g_getSymRegTypeCalls;  // no test currently reads this counter
extern std::function<ncclResult_t(struct ncclDevrWindow*, struct ncclDevrWindow*, ncclSymRegType_t*)>
    g_getSymRegType;

extern std::function<ncclResult_t(struct ncclComm*)> g_symkInitOnce;
extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t)> g_symkAvailable;
extern std::function<bool(int)> g_symkKernelIdIsLL;
extern std::function<ncclResult_t(struct ncclComm*)> g_symkFinalize;

// sym_kernels.cc's ncclSymkLLKernelMask: default has no bits set (the LL-kernel-init-once check never fires).
extern std::function<int()> g_symkLLKernelMask;

// sym_kernels.cc's ncclSymkDynamicSmemKernelMask: default has no bits set (kernelDynSmem stays 0).
extern std::function<int()> g_symkDynamicSmemKernelMask;

// sym_kernels.cc's ncclSymkGetKernelIndex: default always selects index 0.
extern std::function<int(ncclSymkKernelId, int, ncclDataType_t)> g_symkGetKernelIndex;

// sym_kernels.cc's ncclSymkKernelIdToString: default returns a fixed, recognizable name.
extern std::function<const char*(int)> g_symkKernelIdToString;

// sym_kernels.cc's ncclSymkMakeDevWork: trivial success; does not populate *outDevWork.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*)>
    g_symkMakeDevWork;

void ResetSymKernelsFakes();

#endif  // RCCL_TEST_HOST_SYM_KERNELS_FAKES_H_
