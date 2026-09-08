/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_WRAP_FAKES_H_
#define RCCL_TEST_HOST_WRAP_FAKES_H_

#include <cstdint>
#include <functional>

#include "nccl.h"  // ncclResult_t

// The env-var test-control API (SetMicroEnv/SetMicroEnvAbsent/ClearMicroEnv)
// is declared by fakes/env_fakes.h, the shared owner of getenv interposition
// for every microtest binary -- include that directly rather than this file
// for those. See MICROTEST_README.md's "Where a fake belongs".

// getFirmwareVersion()'s sole dependency, made settable so a test can script
// a canned firmware response or a failure.
extern std::function<ncclResult_t(uint32_t, uint64_t*)> g_amdSmiGetFirmwareVersion;

#endif  // RCCL_TEST_HOST_WRAP_FAKES_H_
