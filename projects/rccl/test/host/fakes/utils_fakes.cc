/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/misc/utils.cc, for the microtest binaries that do NOT
// compile the real utils.cc.
//
// Kept out of comm_fakes.cc for exactly that reason: the init and enqueue
// targets link the real hipified utils.cc as an oracle TU, so a fake definition
// there is a duplicate symbol at link time rather than an unused one.

#include <cstdlib>

#include "utils.h"

// Per-thread wait signal referenced by the inline MPSC-callback drain helpers
// in utils.h.
thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance = {};

// busId helpers referenced by transports (e.g. p2p.cc) but only faked here for
// microtest binaries that do not link the real utils.cc. Sensible defaults.
// Parse the PCI bus-id string the same way src/misc/utils.cc does, so tests
// that hand p2pCanConnect device-distinct bus strings (via the
// g_hipDeviceGetPCIBusId hook) resolve distinct busIdToCudaDev indices.
// (The old stub always returned 0, collapsing every device to index 0.)
ncclResult_t busIdToInt64(const char* busId, int64_t* id)
{
    if (!id) return ncclSuccess;
    char hexStr[17];
    int hexOffset = 0;
    for (int i = 0; busId && hexOffset < (int)sizeof(hexStr) - 1; i++) {
        char c = busId[i];
        if (c == '\0') break;
        if (c == '.' || c == ':') continue;
        if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f'))
            hexStr[hexOffset++] = c;
        else
            break;
    }
    hexStr[hexOffset] = '\0';
    *id = std::strtol(hexStr, nullptr, 16);
    return ncclSuccess;
}

ncclResult_t getBusId(int /*cudaDev*/, int64_t* busId)
{
    if (busId) *busId = 0;
    return ncclSuccess;
}
