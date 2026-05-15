/* Copyright (c) 2009 - 2021 Advanced Micro Devices, Inc.

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

#include "cl_common.hpp"
#include "cl_kernel_info_amd.h"
#include "platform/kernel.hpp"
#include "platform/ndrange.hpp"
#include "platform/command.hpp"

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup AMD_Extensions
 *  @{
 *
 */

/*! \brief Retrieves the kernel information.
 *
 *  \param kernel specifies the kernel object being queried.
 *
 *  \param device identifies a specific device in the list of devices associated
 *  with \a kernel. The list of devices is the list of devices in the OpenCL
 *  context that is associated with \a kernel. If the list of devices associated
 *  with kernel is a single device, \a device can be a NULL value.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result
 *  being queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL
 *  - CL_INVALID_KERNEL if \a kernel is a not a valid program object
 */
RUNTIME_ENTRY(cl_int, clGetKernelInfoAMD,
              (cl_kernel kernel, cl_device_id device, cl_kernel_info_amd param_name,
               size_t param_value_size, void* param_value, size_t* param_value_size_ret)) {
  // Check if we have a valid device
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  // Check if we have a valid performance counter
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }

  // Find the kernel, associated with the specified device
  const device::Kernel* devKernel = as_amd(kernel)->getDeviceKernel(*as_amd(device));

  // Make sure we found a valid kernel
  if (devKernel == NULL) {
    return CL_INVALID_KERNEL;
  }

  // Get the corresponded parameters
  switch (param_name) {
    case CL_KERNELINFO_SCRATCH_REGS:
      return amd::clGetInfo(devKernel->workGroupInfo()->scratchRegs_, param_value_size, param_value,
                            param_value_size_ret);
    case CL_KERNELINFO_WAVEFRONT_PER_SIMD:
      return amd::clGetInfo(devKernel->workGroupInfo()->wavefrontPerSIMD_, param_value_size,
                            param_value, param_value_size_ret);
    case CL_KERNELINFO_WAVEFRONT_SIZE:
      return amd::clGetInfo(devKernel->workGroupInfo()->wavefrontSize_, param_value_size,
                            param_value, param_value_size_ret);
    case CL_KERNELINFO_AVAILABLE_GPRS:
      return amd::clGetInfo(devKernel->workGroupInfo()->availableGPRs_, param_value_size,
                            param_value, param_value_size_ret);
    case CL_KERNELINFO_USED_GPRS:
      return amd::clGetInfo(devKernel->workGroupInfo()->usedGPRs_, param_value_size, param_value,
                            param_value_size_ret);
    case CL_KERNELINFO_AVAILABLE_SGPRS:
      return amd::clGetInfo(devKernel->workGroupInfo()->availableSGPRs_, param_value_size,
                            param_value, param_value_size_ret);
    case CL_KERNELINFO_USED_SGPRS:
      return amd::clGetInfo(devKernel->workGroupInfo()->usedSGPRs_, param_value_size, param_value,
                            param_value_size_ret);
    case CL_KERNELINFO_AVAILABLE_VGPRS:
      return amd::clGetInfo(devKernel->workGroupInfo()->availableVGPRs_, param_value_size,
                            param_value, param_value_size_ret);
    case CL_KERNELINFO_USED_VGPRS:
      return amd::clGetInfo(devKernel->workGroupInfo()->usedVGPRs_, param_value_size, param_value,
                            param_value_size_ret);
    case CL_KERNELINFO_AVAILABLE_LDS_SIZE:
      return amd::clGetInfo(devKernel->workGroupInfo()->availableLDSSize_, param_value_size,
                            param_value, param_value_size_ret);
    case CL_KERNELINFO_USED_LDS_SIZE:
      return amd::clGetInfo(devKernel->workGroupInfo()->usedLDSSize_, param_value_size, param_value,
                            param_value_size_ret);
    case CL_KERNELINFO_AVAILABLE_STACK_SIZE:
      return amd::clGetInfo(devKernel->workGroupInfo()->availableStackSize_, param_value_size,
                            param_value, param_value_size_ret);
    case CL_KERNELINFO_USED_STACK_SIZE:
      return amd::clGetInfo(devKernel->workGroupInfo()->usedStackSize_, param_value_size,
                            param_value, param_value_size_ret);
    default:
      return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
