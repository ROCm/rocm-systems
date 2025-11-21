/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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
#include "vdi_common.hpp"
#ifdef _WIN32
#include <d3d10_1.h>
#include "cl_d3d9_amd.hpp"
#include "cl_d3d10_amd.hpp"
#include "cl_d3d11_amd.hpp"
#endif  //_WIN32

#include <mutex>

static cl_int CL_API_CALL icdGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                                             size_t param_value_size, void* param_value,
                                             size_t* param_value_size_ret) {
  if (platform != reinterpret_cast<cl_platform_id>(&amd::PlatformID::Platform)) {
    return CL_INVALID_PLATFORM;
  }

  return clGetPlatformInfo(NULL, param_name, param_value_size, param_value, param_value_size_ret);
}

static cl_int CL_API_CALL icdGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
                                          cl_uint num_entries, cl_device_id* devices,
                                          cl_uint* num_devices) {
  return clGetDeviceIDs(NULL, device_type, num_entries, devices, num_devices);
}

static cl_int CL_API_CALL icdGetDeviceInfo(cl_device_id device, cl_device_info param_name,
                                           size_t param_value_size, void* param_value,
                                           size_t* param_value_size_ret) {
  if (param_name == CL_DEVICE_PLATFORM) {
    // Return the ICD platform instead of the default NULL platform.
    cl_platform_id platform = reinterpret_cast<cl_platform_id>(&amd::PlatformID::Platform);
    return amd::clGetInfo(platform, param_value_size, param_value, param_value_size_ret);
  }

  return clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_icd_dispatch amd::ICDDispatchedObject::icdVendorDispatch_[] = {
    {NULL /* should not get called */, icdGetPlatformInfo, icdGetDeviceIDs, icdGetDeviceInfo,
     clCreateContext, clCreateContextFromType, clRetainContext, clReleaseContext, clGetContextInfo,
     clCreateCommandQueue, clRetainCommandQueue, clReleaseCommandQueue, clGetCommandQueueInfo,
     clSetCommandQueueProperty, clCreateBuffer, clCreateImage2D, clCreateImage3D, clRetainMemObject,
     clReleaseMemObject, clGetSupportedImageFormats, clGetMemObjectInfo, clGetImageInfo,
     clCreateSampler, clRetainSampler, clReleaseSampler, clGetSamplerInfo,
     clCreateProgramWithSource, clCreateProgramWithBinary, clRetainProgram, clReleaseProgram,
     clBuildProgram, clUnloadCompiler, clGetProgramInfo, clGetProgramBuildInfo, clCreateKernel,
     clCreateKernelsInProgram, clRetainKernel, clReleaseKernel, clSetKernelArg, clGetKernelInfo,
     clGetKernelWorkGroupInfo, clWaitForEvents, clGetEventInfo, clRetainEvent, clReleaseEvent,
     clGetEventProfilingInfo, clFlush, clFinish, clEnqueueReadBuffer, clEnqueueWriteBuffer,
     clEnqueueCopyBuffer, clEnqueueReadImage, clEnqueueWriteImage, clEnqueueCopyImage,
     clEnqueueCopyImageToBuffer, clEnqueueCopyBufferToImage, clEnqueueMapBuffer, clEnqueueMapImage,
     clEnqueueUnmapMemObject, clEnqueueNDRangeKernel, clEnqueueTask, clEnqueueNativeKernel,
     clEnqueueMarker, clEnqueueWaitForEvents, clEnqueueBarrier, clGetExtensionFunctionAddress,
     clCreateFromGLBuffer, clCreateFromGLTexture2D, clCreateFromGLTexture3D,
     clCreateFromGLRenderbuffer, clGetGLObjectInfo, clGetGLTextureInfo, clEnqueueAcquireGLObjects,
     clEnqueueReleaseGLObjects, clGetGLContextInfoKHR,
     WINDOWS_SWITCH(clGetDeviceIDsFromD3D10KHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D10BufferKHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D10Texture2DKHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D10Texture3DKHR, NULL),
     WINDOWS_SWITCH(clEnqueueAcquireD3D10ObjectsKHR, NULL),
     WINDOWS_SWITCH(clEnqueueReleaseD3D10ObjectsKHR, NULL), clSetEventCallback, clCreateSubBuffer,
     clSetMemObjectDestructorCallback, clCreateUserEvent, clSetUserEventStatus,
     clEnqueueReadBufferRect, clEnqueueWriteBufferRect, clEnqueueCopyBufferRect, NULL, NULL, NULL,
     clCreateEventFromGLsyncKHR,

     /* OpenCL 1.2*/
     clCreateSubDevices, clRetainDevice, clReleaseDevice, clCreateImage,
     clCreateProgramWithBuiltInKernels, clCompileProgram, clLinkProgram, clUnloadPlatformCompiler,
     clGetKernelArgInfo, clEnqueueFillBuffer, clEnqueueFillImage, clEnqueueMigrateMemObjects,
     clEnqueueMarkerWithWaitList, clEnqueueBarrierWithWaitList,
     clGetExtensionFunctionAddressForPlatform, clCreateFromGLTexture,

     WINDOWS_SWITCH(clGetDeviceIDsFromD3D11KHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D11BufferKHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D11Texture2DKHR, NULL),
     WINDOWS_SWITCH(clCreateFromD3D11Texture3DKHR, NULL),
     WINDOWS_SWITCH(clCreateFromDX9MediaSurfaceKHR, NULL),
     WINDOWS_SWITCH(clEnqueueAcquireD3D11ObjectsKHR, NULL),
     WINDOWS_SWITCH(clEnqueueReleaseD3D11ObjectsKHR, NULL),

     WINDOWS_SWITCH(clGetDeviceIDsFromDX9MediaAdapterKHR,
                    NULL),  // KHRpfn_clGetDeviceIDsFromDX9MediaAdapterKHR
                            // clGetDeviceIDsFromDX9MediaAdapterKHR;
     WINDOWS_SWITCH(
         clEnqueueAcquireDX9MediaSurfacesKHR,
         NULL),  // KHRpfn_clEnqueueAcquireDX9MediaSurfacesKHR clEnqueueAcquireDX9MediaSurfacesKHR;
     WINDOWS_SWITCH(
         clEnqueueReleaseDX9MediaSurfacesKHR,
         NULL),  // KHRpfn_clEnqueueReleaseDX9MediaSurfacesKHR clEnqueueReleaseDX9MediaSurfacesKHR;

     NULL, NULL, NULL, NULL,

     clCreateCommandQueueWithProperties, clCreatePipe, clGetPipeInfo, clSVMAlloc, clSVMFree,
     clEnqueueSVMFree, clEnqueueSVMMemcpy, clEnqueueSVMMemFill, clEnqueueSVMMap, clEnqueueSVMUnmap,
     clCreateSamplerWithProperties, clSetKernelArgSVMPointer, clSetKernelExecInfo,
     clGetKernelSubGroupInfo, clCloneKernel, clCreateProgramWithIL, clEnqueueSVMMigrateMem,
     clGetDeviceAndHostTimer, clGetHostTimer, clGetKernelSubGroupInfo,
     clSetDefaultDeviceCommandQueue,

     clSetProgramReleaseCallback, clSetProgramSpecializationConstant}};

CL_API_ENTRY cl_int CL_API_CALL clIcdGetPlatformIDsKHR(cl_uint num_entries,
                                                       cl_platform_id* platforms,
                                                       cl_uint* num_platforms) {
  if (((num_entries > 0 || num_platforms == NULL) && platforms == NULL) ||
      (num_entries == 0 && platforms != NULL)) {
    return CL_INVALID_VALUE;
  }

  if (!amd::Runtime::initialized()) {
    amd::Runtime::init();
  }

  if (num_platforms != NULL && platforms == NULL) {
    *num_platforms = 1;
    return CL_SUCCESS;
  }

  assert(platforms != NULL && "check the code above");
  *platforms = reinterpret_cast<cl_platform_id>(&amd::PlatformID::Platform);

  *not_null(num_platforms) = 1;
  return CL_SUCCESS;
}
