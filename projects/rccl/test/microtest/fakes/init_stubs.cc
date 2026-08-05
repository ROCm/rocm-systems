/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor + data-symbol definitions for the init.cc microtest
// (AICOMRCCL-1685). The init.cc TU references the full communicator-init symbol
// surface (~180 nccl/rccl/bootstrap/hip internals). The Tier A-D tests never
// call the deep path (initTransportsRank / transport / proxy / topology / nvls /
// gin / ras / net), but every referenced symbol must still be DEFINED for the
// binary to link. Those get one-line fail-loud stubs here (abort if ever hit, so
// a test that strays into the deep path fails immediately and visibly). The
// controllable D5 / Tier A-D subset lives in init_fakes.cc / nccl_fakes.cc /
// hip_fakes.cc instead.
//
// This file is populated iteratively from the reduced undefined-symbol set after
// the real oracle TUs (argcheck/archinfo/utils) are linked with --gc-sections.

#include <cstdlib>

// (stubs added in the next bring-up iteration)
