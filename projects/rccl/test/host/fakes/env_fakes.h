/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The environment seam shared by every host-only microtest binary: ncclGetEnv
// (src/misc/param.cc:107) plus process-wide getenv/std::getenv interposition.
//
// Hoisted out of init_fakes.cc so there is exactly ONE env implementation. A
// second, map-only copy cannot intercept the raw getenv() call sites that
// production still has (e.g. enqueue.cc:2687-2688 read NCCL_PROTO/NCCL_ALGO
// through libc getenv, not ncclGetEnv), which leaves results dependent on the
// ambient environment of the machine running the suite.
//
// Note the interposer is strict only for names the map KNOWS: an unmapped name
// falls through to the real getenv, deliberately, so gtest and libstdc++ still
// see the real environment. A fixture that wants hermeticity for a given name
// must map it, with SetMicroEnvAbsent if the wanted value is "unset".

#ifndef RCCL_TEST_HOST_ENV_FAKES_H_
#define RCCL_TEST_HOST_ENV_FAKES_H_

// Strict: a name the fixture has not scripted reads as unset, so no test can
// depend on the ambient environment. ncclGetEnv routes here.
const char* micro_getenv(const char* name);

// A null value means "absent", NOT "leave unmapped" -- leaving it unmapped would
// fall through to the real getenv.
void SetMicroEnv(const char* name, const char* value);
void SetMicroEnvAbsent(const char* name);
void ClearMicroEnv();

// Named like every other per-TU reset so a Reset*Fakes() chain reads as one list
// and "did I add mine?" is checkable by pattern. Currently just ClearMicroEnv().
void ResetEnvFakes();

#endif  // RCCL_TEST_HOST_ENV_FAKES_H_
