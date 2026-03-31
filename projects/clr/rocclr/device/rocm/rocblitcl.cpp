/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

namespace amd::roc {

#define BLIT_KERNEL(...) #__VA_ARGS__

const char* SchedulerSourceCode = BLIT_KERNEL(

    extern void __amd_scheduler_rocm(__global void*);

    __kernel void __amd_rocclr_scheduler(__global void* params) { __amd_scheduler_rocm(params); });

}  // namespace amd::roc
