/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Compiles the real src/sym_kernels.cc for rccl-UnitTestsMicroTuningSymModel: the symmetric cost model
// calls its kernel masks and Tma gates. Same prelude as sym-kernels-test.cc, which owns this unit's tests.

#include <functional>

#include "fakes/devcomm_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "sym_kernels.h"

#include "fakes/sym_kernels_host_include.h"

#include SYM_KERNELS_CC_PATH
