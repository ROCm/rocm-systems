/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device-malloc-kernel.h"

#include <cstdlib>

__global__ void
hipFileTestDeviceMallocKernel(int *result)
{
    void *allocation = malloc(1);
    *result          = allocation != nullptr;
    free(allocation);
}
