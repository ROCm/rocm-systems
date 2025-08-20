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
#include "platform/context.hpp"
#include "device/device.hpp"
#include "platform/runtime.hpp"
#include "platform/agent.hpp"
#ifdef _WIN32
#include "cl_d3d9_amd.hpp"
#include "cl_d3d10_amd.hpp"
#include "cl_d3d11_amd.hpp"
#endif  // _WIN32
#include "cl_kernel_info_amd.h"
#include "cl_profile_amd.h"
#include "cl_platform_amd.h"
#include "cl_sdi_amd.h"
#include "cl_thread_trace_amd.h"
#include "cl_p2p_amd.h"

#include <GL/gl.h>
#include <GL/glext.h>
#include "CL/cl_gl.h"

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_Contexts
 *  @{
 */

/*! \brief Create an OpenCL context.
 *
 *  An OpenCL context is created with one or more devices. Contexts are used by
 *  the OpenCL runtime for managing objects such as command-queues, memory,
 *  program and kernel objects and for executing kernels on one or more devices
 *  specified in the context.
 *
 *  \param properties is reserved and must be zero.
 *
 *  \param num_devices is the number of devices specified in the \a devices
 *  argument.
 *
 *  \param devices is a pointer to a list of unique devices returned by
 *  clGetDevices. If more than one device is specified in devices,
 *  a selection criteria may be applied to determine if the list of devices
 *  specified can be used together to create a context.
 *
 *  \param pfn_notify is a callback function that can be registered by the
 *  application. This callback function will be used by the runtime to report
 *  information on errors that occur in this context. This callback function
 *  may be called asynchronously by the runtime. If \a pfn_notify is NULL,
 *  no callback function is registered.
 *
 *  \param user_data will be passed as the user_data argument when \a pfn_notify
 *  is called. \a user_data can be NULL.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero context and errcode_ret is set to CL_SUCCESS
 *  if the context is created successfully or NULL with the following
 *  error values stored in \a errcode_ret:
 *    - CL_INVALID_VALUE if \a properties is not zero.
 *    - CL_INVALID_VALUE if \a devices is NULL.
 *    - CL_INVALID_VALUE if \a num_devices is equal to zero.
 *    - CL_INVALID_DEVICE if \a devices contains an invalid device.
 *    - CL_INVALID_DEVICE_LIST if more than one device is specified in
 *      \a devices and the list of devices specified cannot be used together
 *      to create a context.
 *    - CL_DEVICE_NOT_AVAILABLE if a device in \a devices is currently not
 *      available even though the device was returned by clGetDevices.
 *    - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *      required by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_context, clCreateContext,
                  (const cl_context_properties* properties, cl_uint num_devices,
                   const cl_device_id* devices,
                   void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
                   void* user_data, cl_int* errcode_ret)) {
  cl_int errcode;
  amd::Context::Info info;

  errcode = amd::Context::checkProperties(properties, &info);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_context)0;
  }

  if (num_devices == 0 || devices == NULL) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_context)0;
  }

  std::vector<amd::Device*> devices_;
  for (cl_uint i = 0; i < num_devices; ++i) {
    // FIXME_lmoriche: Set errcode_ret to CL_DEVICE_NOT_AVAILABLE if a
    // device in devices is no longer available.
    cl_device_id device = devices[i];

    if (!is_valid(device)) {
      *not_null(errcode_ret) = CL_INVALID_DEVICE;
      return (cl_context)0;
    }
    devices_.push_back(as_amd(device));
  }

  amd::Context* context = new amd::Context(devices_, info);
  if (context == NULL) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_context)0;
  }

  if (CL_SUCCESS != (errcode = context->create(properties))) {
    context->release();
    *not_null(errcode_ret) = errcode;
    return (cl_context)0;
  }

  if (amd::Agent::shouldPostContextEvents()) {
    amd::Agent::postContextCreate(as_cl(context));
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(context);
}
RUNTIME_EXIT

/*! \brief Create an OpenCL context from a device type that identifies the
 *  specific device(s) to use.
 *
 *  \param properties is reserved and must be zero.
 *
 *  \param device_type is a bit-field that identifies the type of device.
 *
 *  \param pfn_notify described in clCreateContext.
 *
 *  \param user_data described in clCreateContext.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero context and errcode_ret is set to CL_SUCCESS
 *  if the context is created successfully or NULL with the following error
 *  values stored in errcode_ret:
 *    - CL_INVALID_VALUE if \a properties is not zero.
 *    - CL_INVALID_DEVICE_TYPE if \a device_type is not a valid value.
 *    - CL_DEVICE_NOT_AVAILABLE if no devices that match \a device_type
 *      are currently available.
 *    - CL_DEVICE_NOT_FOUND if no devices that match \a device_type were found.
 *    - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *      required by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_context, clCreateContextFromType,
                  (const cl_context_properties* properties, cl_device_type device_type,
                   void(CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*),
                   void* user_data, cl_int* errcode_ret)) {
  amd::Context::Info info;
  cl_int errcode = amd::Context::checkProperties(properties, &info);
  if (errcode != CL_SUCCESS) {
    *not_null(errcode_ret) = errcode;
    return (cl_context)0;
  }

  // Get the devices of the given type.
  cl_uint num_devices;
  bool offlineDevices = (info.flags_ & amd::Context::OfflineDevices) ? true : false;
  if (!amd::Device::getDeviceIDs(device_type, 0, NULL, &num_devices, offlineDevices)) {
    *not_null(errcode_ret) = CL_DEVICE_NOT_FOUND;
    return (cl_context)0;
  }

  assert(num_devices > 0 && "Should have returned an error!");
  cl_device_id* devices = (cl_device_id*)alloca(num_devices * sizeof(cl_device_id));

  if (!amd::Device::getDeviceIDs(device_type, num_devices, devices, NULL, offlineDevices)) {
    *not_null(errcode_ret) = CL_DEVICE_NOT_FOUND;
    return (cl_context)0;
  }

  // Create a new context with the devices
  cl_context context =
      clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);

  return context;
}
RUNTIME_EXIT

/*! \brief Increment the context reference count.
 *
 *  \return One of the following values:
 *    - CL_INVALID_CONTEXT if context is not a valid OpenCL context.
 *    - CL_SUCCESS if the function is executed successfully.
 *
 *  clCreateContext and clCreateContextFromType perform an implicit retain.
 *  This is very helpful for 3rd party libraries, which typically get a context
 *  passed to them by the application.
 *  However, it is possible that the application may delete the context without
 *  informing the library. Allowing functions to attach to (i.e. retain) and
 *  release a context solves the problem of a context being used by a library
 *  no longer being valid.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clRetainContext, (cl_context context)) {
  if (!is_valid(context)) {
    return CL_INVALID_CONTEXT;
  }
  as_amd(context)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrement the context reference count.
 *
 *  \return One of the following values:
 *    - CL_INVALID_CONTEXT if context is not a valid OpenCL context.
 *    - CL_SUCCESS if the function is executed successfully.
 *
 *  After the context reference count becomes zero and all the objects attached
 *  to context (such as memory objects, command-queues) are released,
 *  the context is deleted.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clReleaseContext, (cl_context context)) {
  if (!is_valid(context)) {
    return CL_INVALID_CONTEXT;
  }
  as_amd(context)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Query information about a context.
 *
 *  \param context specifies the OpenCL context being queried.
 *
 *  \param param_name is an enum that specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result being
 *  queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size specifies the size in bytes of memory pointed to by
 *  \a param_value. This size must be greater than or equal to the size of
 *  return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data being
 *  queried by \a param_value. If \a param_value_size_ret is NULL,
 *  it is ignored.
 *
 *  \return One of the following values:
 *    - CL_INVALID_CONTEXT if context is not a valid context.
 *    - CL_INVALID_VALUE if \a param_name is not one of the supported values
 *      or if size in bytes specified by \a param_value_size is < size of return
 *      type and \a param_value is not a NULL value.
 *    - CL_SUCCESS if the function is executed successfully.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetContextInfo,
              (cl_context context, cl_context_info param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(context)) {
    return CL_INVALID_CONTEXT;
  }

  switch (param_name) {
    case CL_CONTEXT_REFERENCE_COUNT: {
      cl_uint count = as_amd(context)->referenceCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_CONTEXT_NUM_DEVICES: {
      cl_uint numDevices = (cl_uint)as_amd(context)->devices().size();
      return amd::clGetInfo(numDevices, param_value_size, param_value, param_value_size_ret);
    }
    case CL_CONTEXT_DEVICES: {
      const std::vector<amd::Device*>& devices = as_amd(context)->devices();
      size_t numDevices = devices.size();
      size_t valueSize = numDevices * sizeof(cl_device_id*);

      if (param_value != NULL && param_value_size < valueSize) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = valueSize;
      if (param_value != NULL) {
        cl_device_id* device_list = (cl_device_id*)param_value;
        for (const auto& it : devices) {
          *device_list++ = const_cast<cl_device_id>(as_cl(it));
        }
      }
      return CL_SUCCESS;
    }
    case CL_CONTEXT_PROPERTIES: {
      const amd::Context* amdContext = as_amd(context);
      size_t valueSize = amdContext->info().propertiesSize_;

      if (param_value != NULL && param_value_size < valueSize) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = valueSize;
      if ((param_value != NULL) && (valueSize != 0)) {
        ::memcpy(param_value, amdContext->properties(), valueSize);
      }
      return CL_SUCCESS;
    }
#ifdef _WIN32
    case CL_CONTEXT_D3D10_DEVICE_KHR: {
      // Not defined in the ext.spec, but tested in the conf.test
      // Guessing functionality from the test...
      if (param_value != NULL && param_value_size < sizeof(void*)) {
        return CL_INVALID_VALUE;
      }
      const amd::Context* amdContext = as_amd(context);
      if (!(amdContext->info().flags_ & amd::Context::D3D10DeviceKhr)) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = sizeof(intptr_t);
      if (param_value != NULL) {
        *(intptr_t*)param_value =
            reinterpret_cast<intptr_t>(amdContext->info().hDev_[amd::Context::D3D10DeviceKhrIdx]);
      }
      return CL_SUCCESS;
    }
    case CL_CONTEXT_D3D10_PREFER_SHARED_RESOURCES_KHR: {
      if (param_value != NULL && param_value_size < sizeof(cl_bool)) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = sizeof(cl_bool);
      if (param_value != NULL) {
        *(cl_bool*)param_value = CL_TRUE;
      }
      return CL_SUCCESS;
    }
    case CL_CONTEXT_D3D11_DEVICE_KHR: {
      // Not defined in the ext.spec, but tested in the conf.test
      // Guessing functionality from the test...
      if (param_value != NULL && param_value_size < sizeof(void*)) {
        return CL_INVALID_VALUE;
      }
      const amd::Context* amdContext = as_amd(context);
      if (!(amdContext->info().flags_ & amd::Context::D3D11DeviceKhr)) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = sizeof(intptr_t);
      if (param_value != NULL) {
        *(intptr_t*)param_value =
            reinterpret_cast<intptr_t>(amdContext->info().hDev_[amd::Context::D3D11DeviceKhrIdx]);
      }
      return CL_SUCCESS;
    }
    case CL_CONTEXT_D3D11_PREFER_SHARED_RESOURCES_KHR: {
      if (param_value != NULL && param_value_size < sizeof(cl_bool)) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = sizeof(cl_bool);
      if (param_value != NULL) {
        *(cl_bool*)param_value = CL_TRUE;
      }
      return CL_SUCCESS;
    }
    case CL_CONTEXT_ADAPTER_D3D9_KHR: {
      if (param_value != NULL && param_value_size < sizeof(void*)) {
        return CL_INVALID_VALUE;
      }
      const amd::Context* amdContext = as_amd(context);
      if (!(amdContext->info().flags_ & amd::Context::D3D9DeviceKhr)) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = sizeof(intptr_t);
      if (param_value != NULL) {
        *(intptr_t*)param_value =
            reinterpret_cast<intptr_t>(amdContext->info().hDev_[amd::Context::D3D9DeviceKhrIdx]);
      }
      return CL_SUCCESS;
    }
    case CL_CONTEXT_ADAPTER_D3D9EX_KHR: {
      if (param_value != NULL && param_value_size < sizeof(void*)) {
        return CL_INVALID_VALUE;
      }
      const amd::Context* amdContext = as_amd(context);
      if (!(amdContext->info().flags_ & amd::Context::D3D9DeviceEXKhr)) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = sizeof(intptr_t);
      if (param_value != NULL) {
        *(intptr_t*)param_value =
            reinterpret_cast<intptr_t>(amdContext->info().hDev_[amd::Context::D3D9DeviceEXKhrIdx]);
      }
      return CL_SUCCESS;
    }
    case CL_CONTEXT_ADAPTER_DXVA_KHR: {
      if (param_value != NULL && param_value_size < sizeof(void*)) {
        return CL_INVALID_VALUE;
      }
      const amd::Context* amdContext = as_amd(context);
      if (!(amdContext->info().flags_ & amd::Context::D3D9DeviceVAKhr)) {
        return CL_INVALID_VALUE;
      }
      *not_null(param_value_size_ret) = sizeof(intptr_t);
      if (param_value != NULL) {
        *(intptr_t*)param_value =
            reinterpret_cast<intptr_t>(amdContext->info().hDev_[amd::Context::D3D9DeviceVAKhrIdx]);
      }
      return CL_SUCCESS;
    }
#endif  //_WIN32
    default:
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief returns the address of the extension function named by
 *  funcname for a given platform. The pointer returned should be cast
 *  to a function pointer type matching the extension functions definition
 *  defined in the appropriate extension specification and header file.
 *  A return value of NULL indicates that the specified function does not
 *  exist for the implementation or platform is not a valid platform.
 *  A non-NULL return value for \a clGetExtensionFunctionAddressForPlatform
 *  does not guarantee that an extension function is actually supported by
 *  the platform. The application must also make a corresponding query using
 *  \a clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, ... ) or
 *  \a clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ... ) to determine if
 *  an extension is supported by the OpenCL implementation.
 *
 *  \version 1.2r07
 */
CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddressForPlatform(cl_platform_id platform,
                                                                        const char* funcname) {
  if (platform != NULL && platform != AMD_PLATFORM) {
    return NULL;
  }

  return clGetExtensionFunctionAddress(funcname);
}

CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddress(const char* func_name) {
#define CL_EXTENSION_ENTRYPOINT_CHECK(name)                                                        \
  if (!strcmp(func_name, #name)) return reinterpret_cast<void*>(name);
#define CL_EXTENSION_ENTRYPOINT_CHECK2(name1, name2)                                               \
  if (!strcmp(func_name, #name1)) return reinterpret_cast<void*>(name2);

  switch (func_name[2]) {
    case 'C':
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateEventFromGLsyncKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreatePerfCounterAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateThreadTraceAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromGLBuffer);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromGLTexture);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromGLTexture2D);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromGLTexture3D);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromGLRenderbuffer);
#ifdef _WIN32
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromD3D10BufferKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromD3D10Texture2DKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromD3D10Texture3DKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateFromDX9MediaSurfaceKHR);
#endif  //_WIN32
      CL_EXTENSION_ENTRYPOINT_CHECK(clConvertImageAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateBufferFromImageAMD);
#if defined(cl_khr_il_program) || defined(CL_VERSION_2_1)
      CL_EXTENSION_ENTRYPOINT_CHECK2(clCreateProgramWithILKHR, clCreateProgramWithIL);
#endif  // defined(cl_khr_il_program) || defined(CL_VERSION_2_1)
#if cl_amd_assembly_program
      CL_EXTENSION_ENTRYPOINT_CHECK(clCreateProgramWithAssemblyAMD);
#endif  // cl_amd_assembly_program
      break;
    case 'D':
      break;
    case 'E':
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueBeginPerfCounterAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueEndPerfCounterAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueAcquireGLObjects);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueReleaseGLObjects);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueBindThreadTraceBufferAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueThreadTraceCommandAMD);
#ifdef _WIN32
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueAcquireD3D10ObjectsKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueReleaseD3D10ObjectsKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueAcquireDX9MediaSurfacesKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueReleaseDX9MediaSurfacesKHR);
#endif  //_WIN32
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueWaitSignalAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueWriteSignalAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueMakeBuffersResidentAMD);
#if cl_amd_copy_buffer_p2p
      CL_EXTENSION_ENTRYPOINT_CHECK(clEnqueueCopyBufferP2PAMD);
#endif  // cl_amd_copy_buffer_p2p
      break;
    case 'G':
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetKernelInfoAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetPerfCounterInfoAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetGLObjectInfo);
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetGLTextureInfo);
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetGLContextInfoKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetThreadTraceInfoAMD);
#ifdef _WIN32
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetDeviceIDsFromD3D10KHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetDeviceIDsFromDX9MediaAdapterKHR);
      CL_EXTENSION_ENTRYPOINT_CHECK(clGetPlaneFromImageAMD);
#endif  //_WIN32
#if defined(cl_khr_sub_groups) || defined(CL_VERSION_2_1)
      CL_EXTENSION_ENTRYPOINT_CHECK2(clGetKernelSubGroupInfoKHR, clGetKernelSubGroupInfo);
#endif  // defined(cl_khr_sub_groups) || defined(CL_VERSION_2_1)
      break;
    case 'I':
      CL_EXTENSION_ENTRYPOINT_CHECK(clIcdGetPlatformIDsKHR);
      break;
    case 'R':
      CL_EXTENSION_ENTRYPOINT_CHECK(clReleasePerfCounterAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clRetainPerfCounterAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clReleaseThreadTraceAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clRetainThreadTraceAMD);
      break;
    case 'S':
      CL_EXTENSION_ENTRYPOINT_CHECK(clSetThreadTraceParamAMD);
      CL_EXTENSION_ENTRYPOINT_CHECK(clSetDeviceClockModeAMD);
      break;
    case 'U':
      CL_EXTENSION_ENTRYPOINT_CHECK(clUnloadPlatformAMD);
    default:
      break;
  }

  return NULL;
}

RUNTIME_ENTRY(cl_int, clTerminateContextKHR, (cl_context context)) { return CL_INVALID_CONTEXT; }
RUNTIME_EXIT


/*! @}
 *  @}
 */
