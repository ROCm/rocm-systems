/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The real rccl_wrap.cc, compiled as wrap-test.cc compiles it, for rccl-UnitTestsMicroTuningGeneral:
// tuning_general.cc sizes blocks through rcclOptThreadBlockSize.

#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/wrap_fakes.h"
#include "graph/topo.h"

#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/rccl_arch_thresholds_fakes.h"

#include WRAP_CC_PATH
