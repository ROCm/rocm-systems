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

#ifdef _WIN32

#include "top.hpp"

#include "cl_common.hpp"
#include "cl_d3d10_amd.hpp"
#include "platform/command.hpp"

#include <cstring>
#include <utility>


/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_D3D10_Interops
 *
 *  This section discusses OpenCL functions that allow applications to use Direct3D 10
 * resources (buffers/textures) as OpenCL memory objects. This allows efficient sharing of
 * data between OpenCL and Direct3D 10. The OpenCL API can be used to execute kernels that
 * read and/or write memory objects that are also the Direct3D resources.
 * An OpenCL image object can be created from a D3D10 texture object. An
 * OpenCL buffer object can be created from a D3D10 buffer object (index/vertex).
 *
 *  @}
 *  \addtogroup clGetDeviceIDsFromD3D10KHR
 *  @{
 */

RUNTIME_ENTRY(cl_int, clGetDeviceIDsFromD3D10KHR,
              (cl_platform_id platform, cl_d3d10_device_source_khr d3d_device_source,
               void* d3d_object, cl_d3d10_device_set_khr d3d_device_set, cl_uint num_entries,
               cl_device_id* devices, cl_uint* num_devices)) {
  cl_int errcode;
  ID3D10Device* d3d10_device = NULL;
  cl_device_id* gpu_devices;
  cl_uint num_gpu_devices = 0;
  bool create_d3d10Device = false;
  static const bool VALIDATE_ONLY = true;
  HMODULE d3d10Module = NULL;

  if (platform != NULL && platform != AMD_PLATFORM) {
    LogWarning("\"platrform\" is not a valid AMD platform");
    return CL_INVALID_PLATFORM;
  }
  if (((num_entries > 0 || num_devices == NULL) && devices == NULL) ||
      (num_entries == 0 && devices != NULL)) {
    return CL_INVALID_VALUE;
  }
  // Get GPU devices
  errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpu_devices);
  if (errcode != CL_SUCCESS && errcode != CL_DEVICE_NOT_FOUND) {
    return CL_INVALID_VALUE;
  }

  if (!num_gpu_devices) {
    *not_null(num_devices) = 0;
    return CL_DEVICE_NOT_FOUND;
  }

  switch (d3d_device_source) {
    case CL_D3D10_DEVICE_KHR:
      d3d10_device = static_cast<ID3D10Device*>(d3d_object);
      break;
    case CL_D3D10_DXGI_ADAPTER_KHR: {
      typedef HRESULT(WINAPI * LPD3D10CREATEDEVICE)(IDXGIAdapter*, D3D10_DRIVER_TYPE, HMODULE, UINT,
                                                    UINT32, ID3D10Device**);
      static LPD3D10CREATEDEVICE dynamicD3D10CreateDevice = NULL;

      d3d10Module = LoadLibrary("D3D10.dll");
      if (d3d10Module == NULL) {
        return CL_INVALID_PLATFORM;
      }

      dynamicD3D10CreateDevice =
          (LPD3D10CREATEDEVICE)GetProcAddress(d3d10Module, "D3D10CreateDevice");

      IDXGIAdapter* dxgi_adapter = static_cast<IDXGIAdapter*>(d3d_object);
      HRESULT hr = dynamicD3D10CreateDevice(dxgi_adapter, D3D10_DRIVER_TYPE_HARDWARE, NULL, 0,
                                            D3D10_SDK_VERSION, &d3d10_device);
      if (SUCCEEDED(hr) && (NULL != d3d10_device)) {
        create_d3d10Device = true;
      } else {
        FreeLibrary(d3d10Module);
        return CL_INVALID_VALUE;
      }
    } break;
    default:
      LogWarning("\"d3d_device_source\" is invalid");
      return CL_INVALID_VALUE;
  }

  switch (d3d_device_set) {
    case CL_PREFERRED_DEVICES_FOR_D3D10_KHR:
    case CL_ALL_DEVICES_FOR_D3D10_KHR: {
      gpu_devices = (cl_device_id*)alloca(num_gpu_devices * sizeof(cl_device_id));

      errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, num_gpu_devices, gpu_devices, NULL);
      if (errcode != CL_SUCCESS) {
        break;
      }

      void* external_device[amd::Context::DeviceFlagIdx::LastDeviceFlagIdx] = {};
      external_device[amd::Context::DeviceFlagIdx::D3D10DeviceKhrIdx] = d3d10_device;

      std::vector<amd::Device*> compatible_devices;
      for (cl_uint i = 0; i < num_gpu_devices; ++i) {
        cl_device_id device = gpu_devices[i];
        if (is_valid(device) &&
            as_amd(device)->bindExternalDevice(amd::Context::Flags::D3D10DeviceKhr, external_device,
                                               NULL, VALIDATE_ONLY)) {
          compatible_devices.push_back(as_amd(device));
        }
      }
      if (compatible_devices.size() == 0) {
        *not_null(num_devices) = 0;
        errcode = CL_DEVICE_NOT_FOUND;
        break;
      }

      auto it = compatible_devices.cbegin();
      cl_uint compatible_count = std::min(num_entries, (cl_uint)compatible_devices.size());

      while (compatible_count--) {
        *devices++ = as_cl(*it++);
        --num_entries;
      }
      while (num_entries--) {
        *devices++ = (cl_device_id)0;
      }

      *not_null(num_devices) = (cl_uint)compatible_devices.size();
    } break;

    default:
      LogWarning("\"d3d_device_set\" is invalid");
      errcode = CL_INVALID_VALUE;
  }

  if (create_d3d10Device) {
    d3d10_device->Release();
    FreeLibrary(d3d10Module);
  }
  return errcode;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D10BufferKHR
 *  @{
 */

/*! \brief Creates an OpenCL buffer object from a Direct3D 10 resource.
 *
 *  \param context is a valid OpenCL context.
 *
 *  \param flags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param pD3DResource is a valid pointer to a D3D10 resource of type ID3D10Buffer.
 *
 *  \return valid non-zero OpenCL buffer object and \a errcode_ret is set
 *  to CL_SUCCESS if the buffer object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context or if Direct3D 10
 *    interoperatbility has not been initialized between context and the ID3D10Device
 *    from which pD3DResource was created.
 *  - CL_INVALID_VALUE if values specified in \a clFlags are not valid.
 *  - CL_INVALID_D3D_RESOURCE if \a pD3DResource is not of type ID3D10Buffer.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33?
 */

RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D10BufferKHR,
                  (cl_context context, cl_mem_flags flags, ID3D10Buffer* pD3DResource,
                   cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }
  if (!flags) flags = CL_MEM_READ_WRITE;
  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }
  if (!pD3DResource) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("parameter \"pD3DResource\" is a NULL pointer");
    return clMemObj;
  }
  return (
      amd::clCreateBufferFromD3D10ResourceAMD(*as_amd(context), flags, pD3DResource, errcode_ret));
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateImageFromD3D10Resource
 *  @{
 */

/*! \brief Create an OpenCL 2D or 3D image object from a D3D10 resource.
 *
 *  \param context is a valid OpenCL context.
 *
 *  \param flags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param pD3DResource is a valid pointer to a D3D10 resource of type
 *  ID3D10Texture2D, ID3D10Texture2D, or ID3D10Texture3D.
 * If pD3DResource is of type ID3D10Texture1D then the created image object
 * will be a 1D mipmapped image object.
 * If pD3DResource is of type ID3D10Texture2D and was not created with flag
 * D3D10_RESOURCE_MISC_TEXTURECUBE then the created image object will be a
 * 2D mipmapped image object.
 * If pD3DResource is of type ID3D10Texture2D and was created with flag
 * D3D10_RESOURCE_MISC_TEXTURECUBE then the created image object will be
 * a cubemap mipmapped image object.
 * errocde_ret returns CL_INVALID_D3D_RESOURCE if an OpenCL memory object has
 * already been created from pD3DResource in context.
 * If pD3DResource is of type ID3D10Texture3D then the created image object will
 * be a 3D mipmapped imageobject.
 *
 *  \return valid non-zero OpenCL image object and \a errcode_ret is set
 *  to CL_SUCCESS if the image object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context or if Direct3D 10
 *    interoperatbility has not been initialized between context and the ID3D10Device
 *    from which pD3DResource was created.
 *  - CL_INVALID_VALUE if values specified in \a flags are not valid.
 *  - CL_INVALID_D3D_RESOURCE if \a pD3DResource is not of type ID3D10Texture1D,
 *    ID3D10Texture2D, or ID3D10Texture3D.
 *  - CL_INVALID_D3D_RESOURCE if an OpenCL memory object has already been created
 *    from \a pD3DResource in context.
 *  - CL_INVALID_IMAGE_FORMAT if the Direct3D 10 texture format does not map
 *    to an appropriate OpenCL image format.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r48?
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateImageFromD3D10Resource,
                  (cl_context context, cl_mem_flags flags, ID3D10Resource* pD3DResource,
                   UINT subresource, int* errcode_ret, UINT dimension)) {
  cl_mem clMemObj = NULL;

  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return clMemObj;
  }
  if (!flags) flags = CL_MEM_READ_WRITE;
  if (!(((flags & CL_MEM_READ_ONLY) == CL_MEM_READ_ONLY) ||
        ((flags & CL_MEM_WRITE_ONLY) == CL_MEM_WRITE_ONLY) ||
        ((flags & CL_MEM_READ_WRITE) == CL_MEM_READ_WRITE))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("invalid parameter \"flags\"");
    return clMemObj;
  }
  if (!pD3DResource) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("parameter \"pD3DResource\" is a NULL pointer");
    return clMemObj;
  }

  // Verify context init'ed for interop
  ID3D10Device* pDev;
  pD3DResource->GetDevice(&pDev);
  if (pDev == NULL) {
    *not_null(errcode_ret) = CL_INVALID_D3D10_DEVICE_KHR;
    LogWarning("Cannot retrieve D3D10 device from D3D10 resource");
    return (cl_mem)0;
  }
  pDev->Release();
  if (!((*as_amd(context)).info().flags_ & amd::Context::D3D10DeviceKhr)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("\"amdContext\" is not created from D3D10 device");
    return (cl_mem)0;
  }

  // Check for image support
  const std::vector<amd::Device*>& devices = as_amd(context)->devices();
  bool supportPass = false;
  bool sizePass = false;
  for (const auto& it : devices) {
    if (it->info().imageSupport_) {
      supportPass = true;
    }
  }
  if (!supportPass) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    LogWarning("there are no devices in context to support images");
    return (cl_mem)0;
  }

  switch (dimension) {
#if 0
    case 1:
        return(amd::clCreateImage1DFromD3D10ResourceAMD(
            *as_amd(context),
            flags,
            pD3DResource,
            subresource,
            errcode_ret));
#endif  // 0
    case 2:
      return (amd::clCreateImage2DFromD3D10ResourceAMD(*as_amd(context), flags, pD3DResource,
                                                       subresource, errcode_ret));
    case 3:
      return (amd::clCreateImage3DFromD3D10ResourceAMD(*as_amd(context), flags, pD3DResource,
                                                       subresource, errcode_ret));
    default:
      break;
  }

  *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
  return (cl_mem)0;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D10Texture2DKHR
 *  @{
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D10Texture2DKHR,
                  (cl_context context, cl_mem_flags flags, ID3D10Texture2D* resource,
                   UINT subresource, cl_int* errcode_ret)) {
  return clCreateImageFromD3D10Resource(context, flags, resource, subresource, errcode_ret, 2);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D10Texture3DKHR
 *  @{
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D10Texture3DKHR,
                  (cl_context context, cl_mem_flags flags, ID3D10Texture3D* resource,
                   UINT subresource, cl_int* errcode_ret)) {
  return clCreateImageFromD3D10Resource(context, flags, resource, subresource, errcode_ret, 3);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clEnqueueAcquireD3D10ObjectsKHR
 *  @{
 */
RUNTIME_ENTRY(cl_int, clEnqueueAcquireD3D10ObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clDXTranslateErrorCode(amd::clEnqueueAcquireExtObjectsAMD(
      command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event,
      CL_COMMAND_ACQUIRE_D3D10_OBJECTS_KHR));
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clEnqueueReleaseD3D10ObjectsKHR
 *  @{
 */
RUNTIME_ENTRY(cl_int, clEnqueueReleaseD3D10ObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clDXTranslateErrorCode(amd::clEnqueueReleaseExtObjectsAMD(
      command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event,
      CL_COMMAND_RELEASE_D3D10_OBJECTS_KHR));
}
RUNTIME_EXIT


/*! @}
 *  \addtogroup CL-D3D10 interop helper functions
 *  @{
 */


//*******************************************************************
//
// Internal implementation of CL API functions
//
//*******************************************************************
//
//      clCreateBufferFromD3D10ResourceAMD
//
cl_mem amd::clCreateBufferFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                               ID3D10Resource* pD3DResource, int* errcode_ret) {
  // Verify pD3DResource is a buffer
  D3D10_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D10_RESOURCE_DIMENSION_BUFFER) {
    *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D10Object obj;
  int errcode = D3D10Object::initD3D10Object(amdContext, pD3DResource, 0, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  BufferD3D10* pBufferD3D10 = new (amdContext) BufferD3D10(amdContext, flags, obj);
  if (!pBufferD3D10) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pBufferD3D10->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pBufferD3D10->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pBufferD3D10);
}
#if 0
// There is no support for 1D images in the base imagee code
//
//      clCreateImage1DFromD3D10ResourceAMD
//
cl_mem amd::clCreateImage1DFromD3D10ResourceAMD(
    Context& amdContext,
    cl_mem_flags flags,
    ID3D10Resource* pD3DResource,
    UINT subresource,
    int* errcode_ret)
{

    // Verify the resource is a 1D texture
    D3D10_RESOURCE_DIMENSION rType;
    pD3DResource->GetType(&rType);
    if(rType != D3D10_RESOURCE_DIMENSION_TEXTURE1D) {
        *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
        return (cl_mem) 0;
    }

    D3D10Object obj;
    int errcode = D3D10Object::initD3D10Object(pD3DResource, subresource, obj);
    if(CL_SUCCESS != errcode)
    {
        *not_null(errcode_ret) = errcode;
        return (cl_mem) 0;
    }

    Image1DD3D10 *pImage1DD3D10 = new Image1DD3D10(amdContext, flags, obj);
    if(!pImage1DD3D10) {
        *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
        return (cl_mem) 0;
    }
    if (!pImage1DD3D10->create()) {
        *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
        pImage1DD3D10->release();
        return (cl_mem) 0;
    }

    *not_null(errcode_ret) = CL_SUCCESS;
    return as_cl<Memory>(pImage1DD3D10);
}
#endif

//
//      clCreateImage2DFromD3D10ResourceAMD
//
cl_mem amd::clCreateImage2DFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                                ID3D10Resource* pD3DResource, UINT subresource,
                                                int* errcode_ret) {
  // Verify the resource is a 2D texture
  D3D10_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D10_RESOURCE_DIMENSION_TEXTURE2D) {
    *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D10Object obj;
  int errcode = D3D10Object::initD3D10Object(amdContext, pD3DResource, subresource, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  Image2DD3D10* pImage2DD3D10 = new (amdContext) Image2DD3D10(amdContext, flags, obj);
  if (!pImage2DD3D10) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage2DD3D10->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage2DD3D10->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pImage2DD3D10);
}

//
//      clCreateImage2DFromD3D10ResourceAMD
//
cl_mem amd::clCreateImage3DFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                                ID3D10Resource* pD3DResource, UINT subresource,
                                                int* errcode_ret) {
  // Verify the resource is a 2D texture
  D3D10_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D10_RESOURCE_DIMENSION_TEXTURE3D) {
    *not_null(errcode_ret) = CL_INVALID_D3D10_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D10Object obj;
  int errcode = D3D10Object::initD3D10Object(amdContext, pD3DResource, subresource, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  Image3DD3D10* pImage3DD3D10 = new (amdContext) Image3DD3D10(amdContext, flags, obj);
  if (!pImage3DD3D10) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage3DD3D10->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage3DD3D10->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<amd::Memory>(pImage3DD3D10);
}

//
// Helper function SyncD3D10Objects
//
void amd::SyncD3D10Objects(std::vector<amd::Memory*>& memObjects) {
  Memory*& mem = memObjects.front();
  if (!mem) {
    LogWarning("\nNULL memory object\n");
    return;
  }
  InteropObject* interop = mem->getInteropObj();
  if (!interop) {
    LogWarning("\nNULL interop object\n");
    return;
  }
  D3D10Object* d3d10Obj = interop->asD3D10Object();
  if (!d3d10Obj) {
    LogWarning("\nNULL D3D10 object\n");
    return;
  }
  ID3D10Query* query = d3d10Obj->getQuery();
  if (!query) {
    LogWarning("\nNULL ID3D10Query\n");
    return;
  }
  query->End();
  BOOL data = FALSE;
  while (S_OK != query->GetData(&data, sizeof(BOOL), 0)) {
  }
}


#endif  //_WIN32
