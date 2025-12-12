/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "backend/asyncop-fallback.h"

#include <hip/hip_runtime.h>

namespace hipFile {
__global__ void hipFileMemcpyKernel(AsyncOpFallback *op);
}
