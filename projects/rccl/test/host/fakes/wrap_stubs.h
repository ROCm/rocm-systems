/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_WRAP_STUBS_H_
#define RCCL_TEST_HOST_WRAP_STUBS_H_

// Test-control API for wrap_stubs.cc's env-var fake. Same shape as
// fakes/init_fakes.h's micro_getenv/SetMicroEnv family, replicated here
// rather than shared, since this file's fakes stay self-contained (see
// wrap_stubs.cc's header comment).

// Returns a pointer into the map's own std::string: re-scripting a name
// invalidates a pointer a caller may still hold.
const char* micro_getenv(const char* name);
void SetMicroEnv(const char* name, const char* value);
void SetMicroEnvAbsent(const char* name);
void ClearMicroEnv();

#endif  // RCCL_TEST_HOST_WRAP_STUBS_H_
