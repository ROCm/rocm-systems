/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NCCL_PARAM redirector shared by the microtest TUs. Pull in param.h, then
// replace NCCL_PARAM with a body that routes every generated ncclParamXxx()
// through g_loadParam on each call (no caching), so tests can flip a param's
// value between cases. The redirected body matches the real signature
// (int64_t ncclParam<name>()) but skips the cache / uninitialized-sentinel
// machinery. g_loadParam (nccl_fakes.h) must be in scope wherever the generated
// bodies are emitted -- i.e. include this before the unit-under-test.

#ifndef RCCL_TEST_HOST_PARAM_REDIRECT_H_
#define RCCL_TEST_HOST_PARAM_REDIRECT_H_

#include "param.h"

#undef NCCL_PARAM
#define NCCL_PARAM(name, env, deftVal) \
    int64_t ncclParam##name() { return g_loadParam((env), (deftVal)); }

// The real RCCL_PARAM macros cache and declare a pthread_mutex_t global; redirect
// them the same way so params stay per-test controllable. RCCL_PARAM_NCCL_ALIAS
// differs only in also accepting the NCCL_ name in production, which a redirected
// body does not model, so it DELEGATES rather than repeating the expansion -- a
// future edit to one cannot silently skip the other.
#undef RCCL_PARAM
#define RCCL_PARAM(name, env, deftVal) \
    int64_t rcclParam##name() { return g_loadParam(("RCCL_" env), (deftVal)); }
#undef RCCL_PARAM_NCCL_ALIAS
#define RCCL_PARAM_NCCL_ALIAS(name, env, deftVal) RCCL_PARAM(name, env, deftVal)

#endif  // RCCL_TEST_HOST_PARAM_REDIRECT_H_
