/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams utils_fakes.cc owns, for the binaries that do not compile the real src/misc/utils.cc.

#ifndef RCCL_TEST_HOST_UTILS_FAKES_H_
#define RCCL_TEST_HOST_UTILS_FAKES_H_

#include <cstdint>
#include <functional>

#include "nccl.h"

// Defaults to bus id 0 for every device and ncclSuccess, as the plain stub did.
extern std::function<ncclResult_t(int /*cudaDev*/, int64_t* /*busId*/)> g_getBusId;

void ResetUtilsFakes();

#endif  // RCCL_TEST_HOST_UTILS_FAKES_H_
