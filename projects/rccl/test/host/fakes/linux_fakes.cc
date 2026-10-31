/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The one non-libc, non-HIP symbol rccl-UnitTestsMicroLinux needs that no shared fakes file on its
// link line provides: src/init.cc's SetCpuStackSize param, which os/linux.cc reads. init_fakes.cc
// cannot be linked there. Defined with init.cc's real int64_t type, not os/linux.cc's int declaration.

#include <cstdint>

#include "nccl_fakes.h"

int64_t ncclParamSetCpuStackSize() { return g_loadParam("SET_CPU_STACK_SIZE", 1); }
