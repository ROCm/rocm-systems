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

// ncclSymkLLKernelMask/ncclSymkDynamicSmemKernelMask/ncclSymkGetKernelIndex/ncclSymkKernelIdToString/
// ncclSymkMakeDevWork seams live in sym_kernels_index_fakes.h, not here (rccl-UnitTestsMicro links the
// real sym_kernels_host.cc, which defines those symbols for real -- a duplicate-symbol error if faked here too).

void ResetSymKernelsFakes();

#endif  // RCCL_TEST_HOST_SYM_KERNELS_FAKES_H_
