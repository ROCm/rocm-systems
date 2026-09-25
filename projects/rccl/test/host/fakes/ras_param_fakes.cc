/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ras/ras_param.h"

// client.cc is the unit under test; keep its timeout calculations independent
// of NCCL_RAS_TIMEOUT_FACTOR inherited from the test runner's environment.
double rasTimeoutFactorSec(int baseSeconds) { return static_cast<double>(baseSeconds); }
