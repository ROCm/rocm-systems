/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/register/register.cc's param, for binaries that compile coll_reg.cc or sendrecv_reg.cc but not
// register.cc. Read through g_loadParam on each call, as param_redirect.h does for the real file.

#include <cstdint>

#include "register.h"
#include "rocmwrap.h"

#include "nccl_fakes.h"

// register.cc splits the default on the cuMem version; mirrored rather than pinned to one arm.
#if NCCL_CUMEM_VERSION_SUPPORTED(HIP_VERSION)
int64_t ncclParamLocalRegister() { return g_loadParam("LOCAL_REGISTER", 1); }
#else
int64_t ncclParamLocalRegister() { return g_loadParam("LOCAL_REGISTER", 0); }
#endif
