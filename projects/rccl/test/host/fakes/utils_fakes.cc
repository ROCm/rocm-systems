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

#include "utils.h"

// Per-thread wait signal referenced by the inline MPSC-callback drain helpers
// in utils.h.
thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance = {};

// ncclMemoryStack::allocateSpilled is already faked in fakes/dev_runtime_micro_fakes.cc; reused, not duplicated.

// busId helpers referenced by transports (e.g. p2p.cc) but only faked here for
// microtest binaries that do not link the real utils.cc. Sensible defaults.
ncclResult_t busIdToInt64(const char* /*busId*/, int64_t* id)
{
    if (id) *id = 0;
    return ncclSuccess;
}

ncclResult_t getBusId(int /*cudaDev*/, int64_t* busId)
{
    if (busId) *busId = 0;
    return ncclSuccess;
}
