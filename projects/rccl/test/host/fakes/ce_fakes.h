/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Copy-engine availability gates defined by src/ce_coll.cc. All default to
// "unavailable" so the CE arms stay off unless a test asks for them.

#ifndef RCCL_TEST_HOST_CE_FAKES_H_
#define RCCL_TEST_HOST_CE_FAKES_H_

extern bool g_ceImplemented;  // UNDRIVEN
extern bool g_ceAvailable;  // UNDRIVEN
extern bool g_ceScratchAvailable;  // UNDRIVEN
extern bool g_hierCeAvailable;  // UNDRIVEN

void ResetCeFakes();

#endif  // RCCL_TEST_HOST_CE_FAKES_H_
