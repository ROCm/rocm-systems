/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_NVIDIA_HIP_GL_INTEROP_H
#define HIP_INCLUDE_NVIDIA_HIP_GL_INTEROP_H

#include <cuda_gl_interop.h>

typedef enum cudaGLDeviceList hipGLDeviceList;
#define hipGLDeviceListAll cudaGLDeviceListAll
#define hipGLDeviceListCurrentFrame cudaGLDeviceListCurrentFrame
#define hipGLDeviceListNextFrame cudaGLDeviceListNextFrame

inline static hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                                         unsigned int hipDeviceCount, hipGLDeviceList deviceList) {
  return hipCUDAErrorTohipError(
      cudaGLGetDevices(pHipDeviceCount, pHipDevices, hipDeviceCount, deviceList));
}

inline static hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                                     unsigned int flags) {
  return hipCUDAErrorTohipError(cudaGraphicsGLRegisterBuffer(resource, buffer, flags));
}

inline static hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image,
                                                    GLenum target, unsigned int flags) {
  return hipCUDAErrorTohipError(cudaGraphicsGLRegisterImage(resource, image, target, flags));
}
#endif