/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __CL_SEMAPHORE_AMD_H
#define __CL_SEMAPHORE_AMD_H
/*******************************************
 * AMD Extension cl_amd_semaphore
 *******************************************/
#define cl_amd_semaphore 1

#if cl_amd_semaphore

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* cl_device_info */
#define CL_DEVICE_MAX_SEMAPHORES_AMD 0xF050
#define CL_DEVICE_MAX_SEMAPHORE_SIZE_AMD 0xF051

/* cl_kernel_work_group_info */
#define CL_KERNEL_MAX_SEMAPHORE_SIZE_AMD 0xF052

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* cl_amd_semaphore */

#endif /* __CL_SEMAPHORE_AMD_H */
