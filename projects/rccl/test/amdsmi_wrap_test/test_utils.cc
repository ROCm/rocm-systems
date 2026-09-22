/*************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file test_utils.cc
 * @brief Stub implementations of utility functions for standalone testing
 */

#include "test_stubs.h"
#include <cstring>
#include <cstdio>

// Implementation of busIdToInt64 for parsing PCI bus IDs
// Format: DDDD:BB:DD.F (domain:bus:device.function)
ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
    if (busId == nullptr || id == nullptr) {
        return ncclInvalidArgument;
    }

    // Parse format: DDDD:BB:DD.F
    unsigned int domain, bus, device, function;
    int parsed = sscanf(busId, "%x:%x:%x.%x", &domain, &bus, &device, &function);

    if (parsed != 4) {
        // Try alternative format without domain
        parsed = sscanf(busId, "%x:%x.%x", &bus, &device, &function);
        if (parsed != 3) {
            return ncclInvalidArgument;
        }
        domain = 0;
    }

    // Pack into int64_t in RCCL format:
    // Bits [63:20] = domain
    // Bits [19:12] = bus
    // Bits [11:4]  = device
    // Bits [3:0]   = function
    *id = ((int64_t)domain << 20) | ((int64_t)bus << 12) | ((int64_t)device << 4) | function;

    return ncclSuccess;
}
