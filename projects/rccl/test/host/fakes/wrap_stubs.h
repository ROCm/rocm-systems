/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_WRAP_STUBS_H_
#define RCCL_TEST_HOST_WRAP_STUBS_H_

#include <cstdint>
#include <functional>

#include "nccl.h"  // ncclResult_t

// Test-control API for wrap_stubs.cc's env-var fake. Same shape as
// fakes/init_fakes.h's micro_getenv/SetMicroEnv family, replicated here
// rather than shared, since this file's fakes stay self-contained (see
// wrap_stubs.cc's header comment). micro_getenv itself has no caller
// outside wrap_stubs.cc (ncclGetEnv, same TU), so it isn't declared here.

// Returns a pointer into the map's own std::string: re-scripting a name
// invalidates a pointer a caller may still hold.
void SetMicroEnv(const char* name, const char* value);
void SetMicroEnvAbsent(const char* name);
void ClearMicroEnv();

// getFirmwareVersion()'s sole dependency, made settable so a test can script
// a canned firmware response or a failure.
extern std::function<ncclResult_t(uint32_t, uint64_t*)> g_amdSmiGetFirmwareVersion;

#endif  // RCCL_TEST_HOST_WRAP_STUBS_H_
