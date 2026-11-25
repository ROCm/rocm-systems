/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Renames src/register/register.cc's definitions of the three names nccl_fakes.cc keeps as seams for
// p2p-test.cc, so the real file can be compiled into the same binary; tests call the micro_* names. Include it
// before the #include REGISTER_CC_PATH and register_seam_undef.h after it. No include guard on purpose.

// The header declaring the renamed names, so a rename never rewrites a declaration.
#include "register.h"

#define ncclRegLocalIsValid micro_ncclRegLocalIsValid
#define ncclCommGraphRegister micro_ncclCommGraphRegister
#define ncclCommGraphDeregister micro_ncclCommGraphDeregister
