/* Copyright (c) 2012 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
