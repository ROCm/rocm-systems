/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/misc/kernel_config.cc. No state, so no reset.

#include "nccl.h"

bool ncclIommuPassthroughOk(const char*) { return true; }
