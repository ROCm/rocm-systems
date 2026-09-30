/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"
#include <hip/surface_types.h>

struct __hip_surface {
  uint32_t imageSRD[HIP_IMAGE_OBJECT_SIZE_DWORD];
  amd::Image* image;
  hipResourceDesc resDesc;

  __hip_surface(amd::Image* image_, const hipResourceDesc& resDesc_)
      : image(image_), resDesc(resDesc_) {
    amd::Context& context = *hip::getCurrentDevice()->asContext();
    amd::Device& device = *context.devices()[0];

    device::Memory* imageMem = image->getDeviceMemory(device);
    std::memcpy(imageSRD, imageMem->cpuSrd(), sizeof(imageSRD));
  }
};

namespace hip {

hipError_t ihipFree(void* ptr);
hipError_t ihipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                   const hipResourceDesc* pResDesc) {
  amd::Device* device = hip::getCurrentDevice()->devices()[0];
  const device::Info& info = device->info();
  if (!info.imageSupport_) {
    LogPrintfError("Texture not supported on the device %s", info.name_);
    return hipErrorNotSupported;
  }

  // Validate input params
  if (pSurfObject == nullptr || pResDesc == nullptr) {
    return hipErrorInvalidValue;
  }

  // the type of resource must be a HIP array
  // hipResourceDesc::res::array::array must be set to a valid HIP array handle.
  if ((pResDesc->resType != hipResourceTypeArray) || (pResDesc->res.array.array == nullptr)) {
    return hipErrorInvalidValue;
  }

  if (pResDesc->res.array.array->flags != hipArrayDefault &&
      (pResDesc->res.array.array->flags & hipArraySurfaceLoadStore) == 0) {
    return hipErrorInvalidValue;
  }

  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  amd::Image* image = nullptr;
  cl_mem memObj = reinterpret_cast<cl_mem>(pResDesc->res.array.array->data);
  if (!is_valid(memObj)) {
    return hipErrorInvalidValue;
  }
  image = as_amd(memObj)->asImage();

  void* surfObjectBuffer = nullptr;
  hipError_t err =
      ihipMalloc(&surfObjectBuffer, sizeof(__hip_surface), CL_MEM_SVM_FINE_GRAIN_BUFFER);
  if (surfObjectBuffer == nullptr || err != hipSuccess) {
    return hipErrorOutOfMemory;
  }
  *pSurfObject = new (surfObjectBuffer) __hip_surface{image, *pResDesc};

  return hipSuccess;
}

hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                  const hipResourceDesc* pResDesc) {
  HIP_INIT_API(hipCreateSurfaceObject, pSurfObject, pResDesc);

  HIP_RETURN(ihipCreateSurfaceObject(pSurfObject, pResDesc));
}

hipError_t ihipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
  if (surfaceObject == nullptr) {
    return hipSuccess;
  }

  if (!hip::tls.capture_streams_.empty() || !g_captureStreams.empty()) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  return ihipFree(surfaceObject);
}

hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject) {
  HIP_INIT_API(hipDestroySurfaceObject, surfaceObject);

  HIP_RETURN(ihipDestroySurfaceObject(surfaceObject));
}
}  // namespace hip
