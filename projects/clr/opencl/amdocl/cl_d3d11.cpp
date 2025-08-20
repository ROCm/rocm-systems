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

#include "cl_d3d11_amd.hpp"
#include "platform/command.hpp"

#include <cstring>
#include <utility>

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_D3D11_Interops
 *
 *  This section discusses OpenCL functions that allow applications to use Direct3D 11
 * resources (buffers/textures) as OpenCL memory objects. This allows efficient sharing of
 * data between OpenCL and Direct3D 11. The OpenCL API can be used to execute kernels that
 * read and/or write memory objects that are also the Direct3D resources.
 * An OpenCL image object can be created from a D3D11 texture object. An
 * OpenCL buffer object can be created from a D3D11 buffer object (index/vertex).
 *
 *  @}
 *  \addtogroup clGetDeviceIDsFromD3D11KHR
 *  @{
 */

RUNTIME_ENTRY(cl_int, clGetDeviceIDsFromD3D11KHR,
              (cl_platform_id platform, cl_d3d11_device_source_khr d3d_device_source,
               void* d3d_object, cl_d3d11_device_set_khr d3d_device_set, cl_uint num_entries,
               cl_device_id* devices, cl_uint* num_devices)) {
  cl_int errcode;
  ID3D11Device* d3d11_device = NULL;
  cl_device_id* gpu_devices;
  cl_uint num_gpu_devices = 0;
  bool create_d3d11Device = false;
  static const bool VALIDATE_ONLY = true;
  HMODULE d3d11Module = NULL;

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
    case CL_D3D11_DEVICE_KHR:
      d3d11_device = static_cast<ID3D11Device*>(d3d_object);
      break;
    case CL_D3D11_DXGI_ADAPTER_KHR: {
      static PFN_D3D11_CREATE_DEVICE dynamicD3D11CreateDevice = NULL;

      d3d11Module = LoadLibrary("D3D11.dll");
      if (d3d11Module == NULL) {
        return CL_INVALID_PLATFORM;
      }

      dynamicD3D11CreateDevice =
          (PFN_D3D11_CREATE_DEVICE)GetProcAddress(d3d11Module, "D3D11CreateDevice");

      IDXGIAdapter* dxgi_adapter = static_cast<IDXGIAdapter*>(d3d_object);
      D3D_FEATURE_LEVEL requestedFeatureLevels[] = {D3D_FEATURE_LEVEL_10_0};
      D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
      HRESULT hr = dynamicD3D11CreateDevice(dxgi_adapter, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0,
                                            requestedFeatureLevels, 1, D3D11_SDK_VERSION,
                                            &d3d11_device, &featureLevel, NULL);
      if (SUCCEEDED(hr) && (NULL != d3d11_device)) {
        create_d3d11Device = true;
      } else {
        FreeLibrary(d3d11Module);
        return CL_INVALID_VALUE;
      }
    } break;
    default:
      LogWarning("\"d3d_device_source\" is invalid");
      return CL_INVALID_VALUE;
  }

  switch (d3d_device_set) {
    case CL_PREFERRED_DEVICES_FOR_D3D11_KHR:
    case CL_ALL_DEVICES_FOR_D3D11_KHR: {
      gpu_devices = (cl_device_id*)alloca(num_gpu_devices * sizeof(cl_device_id));

      errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, num_gpu_devices, gpu_devices, NULL);
      if (errcode != CL_SUCCESS) {
        break;
      }

      std::vector<amd::Device*> compatible_devices;
      for (cl_uint i = 0; i < num_gpu_devices; ++i) {
        void* external_device[amd::Context::DeviceFlagIdx::LastDeviceFlagIdx] = {};
        external_device[amd::Context::DeviceFlagIdx::D3D11DeviceKhrIdx] = d3d11_device;

        cl_device_id device = gpu_devices[i];
        if (is_valid(device) &&
            as_amd(device)->bindExternalDevice(amd::Context::Flags::D3D11DeviceKhr, external_device,
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

  if (create_d3d11Device) {
    d3d11_device->Release();
    FreeLibrary(d3d11Module);
  }
  return errcode;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D11BufferKHR
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
 *  \param pD3DResource is a valid pointer to a D3D11 resource of type ID3D11Buffer.
 *
 *  \return valid non-zero OpenCL buffer object and \a errcode_ret is set
 *  to CL_SUCCESS if the buffer object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context or if Direct3D 10
 *    interoperatbility has not been initialized between context and the ID3D11Device
 *    from which pD3DResource was created.
 *  - CL_INVALID_VALUE if values specified in \a clFlags are not valid.
 *  - CL_INVALID_D3D_RESOURCE if \a pD3DResource is not of type ID3D11Buffer.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33?
 */

RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D11BufferKHR,
                  (cl_context context, cl_mem_flags flags, ID3D11Buffer* pD3DResource,
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
      amd::clCreateBufferFromD3D11ResourceAMD(*as_amd(context), flags, pD3DResource, errcode_ret));
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateImageFromD3D11Resource
 *  @{
 */

/*! \brief Create an OpenCL 2D or 3D image object from a D3D11 resource.
 *
 *  \param context is a valid OpenCL context.
 *
 *  \param flags is a bit-field that is used to specify usage information.
 *  Only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE values
 *  can be used.
 *
 *  \param pD3DResource is a valid pointer to a D3D11 resource of type
 *  ID3D11Texture2D, ID3D11Texture2D, or ID3D11Texture3D.
 * If pD3DResource is of type ID3D11Texture1D then the created image object
 * will be a 1D mipmapped image object.
 * If pD3DResource is of type ID3D11Texture2D and was not created with flag
 * D3D11_RESOURCE_MISC_TEXTURECUBE then the created image object will be a
 * 2D mipmapped image object.
 * If pD3DResource is of type ID3D11Texture2D and was created with flag
 * D3D11_RESOURCE_MISC_TEXTURECUBE then the created image object will be
 * a cubemap mipmapped image object.
 * errocde_ret returns CL_INVALID_D3D_RESOURCE if an OpenCL memory object has
 * already been created from pD3DResource in context.
 * If pD3DResource is of type ID3D11Texture3D then the created image object will
 * be a 3D mipmapped imageobject.
 *
 *  \return valid non-zero OpenCL image object and \a errcode_ret is set
 *  to CL_SUCCESS if the image object is created successfully. It returns a NULL
 *  value with one of the following error values returned in \a errcode_ret:
 *  - CL_INVALID_CONTEXT if \a context is not a valid context or if Direct3D 11
 *    interoperatbility has not been initialized between context and the ID3D11Device
 *    from which pD3DResource was created.
 *  - CL_INVALID_VALUE if values specified in \a flags are not valid.
 *  - CL_INVALID_D3D_RESOURCE if \a pD3DResource is not of type ID3D11Texture1D,
 *    ID3D11Texture2D, or ID3D11Texture3D.
 *  - CL_INVALID_D3D_RESOURCE if an OpenCL memory object has already been created
 *    from \a pD3DResource in context.
 *  - CL_INVALID_IMAGE_FORMAT if the Direct3D 11 texture format does not map
 *    to an appropriate OpenCL image format.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r48?
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateImageFromD3D11Resource,
                  (cl_context context, cl_mem_flags flags, ID3D11Resource* pD3DResource,
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
  ID3D11Device* pDev;
  pD3DResource->GetDevice(&pDev);
  if (pDev == NULL) {
    *not_null(errcode_ret) = CL_INVALID_D3D11_DEVICE_KHR;
    LogWarning("Cannot retrieve D3D11 device from D3D11 resource");
    return (cl_mem)0;
  }
  pDev->Release();
  if (!((*as_amd(context)).info().flags_ & amd::Context::D3D11DeviceKhr)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("\"amdContext\" is not created from D3D11 device");
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
        return(amd::clCreateImage1DFromD3D11ResourceAMD(
            *as_amd(context),
            flags,
            pD3DResource,
            subresource,
            errcode_ret));
#endif  // 0
    case 2:
      return (amd::clCreateImage2DFromD3D11ResourceAMD(*as_amd(context), flags, pD3DResource,
                                                       subresource, errcode_ret));
    case 3:
      return (amd::clCreateImage3DFromD3D11ResourceAMD(*as_amd(context), flags, pD3DResource,
                                                       subresource, errcode_ret));
    default:
      break;
  }

  *not_null(errcode_ret) = CL_INVALID_D3D11_RESOURCE_KHR;
  return (cl_mem)0;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D11Texture2DKHR
 *  @{
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D11Texture2DKHR,
                  (cl_context context, cl_mem_flags flags, ID3D11Texture2D* resource,
                   UINT subresource, cl_int* errcode_ret)) {
  return clCreateImageFromD3D11Resource(context, flags, resource, subresource, errcode_ret, 2);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clCreateFromD3D11Texture3DKHR
 *  @{
 */
RUNTIME_ENTRY_RET(cl_mem, clCreateFromD3D11Texture3DKHR,
                  (cl_context context, cl_mem_flags flags, ID3D11Texture3D* resource,
                   UINT subresource, cl_int* errcode_ret)) {
  return clCreateImageFromD3D11Resource(context, flags, resource, subresource, errcode_ret, 3);
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clEnqueueAcquireD3D11ObjectsKHR
 *  @{
 */
RUNTIME_ENTRY(cl_int, clEnqueueAcquireD3D11ObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clDXTranslateErrorCode(amd::clEnqueueAcquireExtObjectsAMD(
      command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event,
      CL_COMMAND_ACQUIRE_D3D11_OBJECTS_KHR));
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clEnqueueReleaseD3D11ObjectsKHR
 *  @{
 */
RUNTIME_ENTRY(cl_int, clEnqueueReleaseD3D11ObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clDXTranslateErrorCode(amd::clEnqueueReleaseExtObjectsAMD(
      command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event,
      CL_COMMAND_RELEASE_D3D11_OBJECTS_KHR));
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup clGetPlaneFromImageAMD
 *  @{
 */
RUNTIME_ENTRY_RET(cl_mem, clGetPlaneFromImageAMD,
                  (cl_context context, cl_mem mem, cl_uint plane, cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    LogWarning("invalid parameter \"context\"");
    return 0;
  }
  if (mem == 0) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return 0;
  }
  if (!is_valid(mem)) {
    *not_null(errcode_ret) = CL_INVALID_MEM_OBJECT;
    return 0;
  }
  amd::Memory* amdMem = as_amd(mem);
  amd::Context& amdContext = *as_amd(context);
  if (amdMem->getInteropObj() == NULL) {
    *not_null(errcode_ret) = CL_INVALID_MEM_OBJECT;
    return 0;
  }
  amd::Image2DD3D11* pImage = reinterpret_cast<amd::Image2DD3D11*>(amdMem);
  ID3D11Resource* pD3DResource = pImage->getD3D11Resource();
  // Verify the resource is a 2D texture
  D3D11_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
    *not_null(errcode_ret) = CL_INVALID_D3D11_RESOURCE_KHR;
    return (cl_mem)0;
  }

  amd::D3D11Object obj;
  int errcode = amd::D3D11Object::initD3D11Object(amdContext, pD3DResource, 0, obj, plane);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  amd::Image2DD3D11* pImage2DD3D11 =
      new (amdContext) amd::Image2DD3D11(amdContext, pImage->getMemFlags(), obj);
  if (!pImage2DD3D11) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage2DD3D11->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage2DD3D11->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<amd::Memory>(pImage2DD3D11);
}
RUNTIME_EXIT


/*! @}
 *  \addtogroup CL-D3D11 interop helper functions
 *  @{
 */


//*******************************************************************
//
// Internal implementation of CL API functions
//
//*******************************************************************
//
//      clCreateBufferFromD3D11ResourceAMD
//
cl_mem amd::clCreateBufferFromD3D11ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                               ID3D11Resource* pD3DResource, int* errcode_ret) {
  // Verify pD3DResource is a buffer
  D3D11_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D11_RESOURCE_DIMENSION_BUFFER) {
    *not_null(errcode_ret) = CL_INVALID_D3D11_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D11Object obj;
  int errcode = D3D11Object::initD3D11Object(amdContext, pD3DResource, 0, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  BufferD3D11* pBufferD3D11 = new (amdContext) BufferD3D11(amdContext, flags, obj);
  if (!pBufferD3D11) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pBufferD3D11->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pBufferD3D11->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pBufferD3D11);
}

//
//      clCreateImage2DFromD3D11ResourceAMD
//
cl_mem amd::clCreateImage2DFromD3D11ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                                ID3D11Resource* pD3DResource, UINT subresource,
                                                int* errcode_ret) {
  // Verify the resource is a 2D texture
  D3D11_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
    *not_null(errcode_ret) = CL_INVALID_D3D11_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D11Object obj;
  int errcode = D3D11Object::initD3D11Object(amdContext, pD3DResource, subresource, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  Image2DD3D11* pImage2DD3D11 = new (amdContext) Image2DD3D11(amdContext, flags, obj);
  if (!pImage2DD3D11) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage2DD3D11->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage2DD3D11->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pImage2DD3D11);
}

//
//      clCreateImage2DFromD3D11ResourceAMD
//
cl_mem amd::clCreateImage3DFromD3D11ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                                ID3D11Resource* pD3DResource, UINT subresource,
                                                int* errcode_ret) {
  // Verify the resource is a 2D texture
  D3D11_RESOURCE_DIMENSION rType;
  pD3DResource->GetType(&rType);
  if (rType != D3D11_RESOURCE_DIMENSION_TEXTURE3D) {
    *not_null(errcode_ret) = CL_INVALID_D3D11_RESOURCE_KHR;
    return (cl_mem)0;
  }

  D3D11Object obj;
  int errcode = D3D11Object::initD3D11Object(amdContext, pD3DResource, subresource, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  Image3DD3D11* pImage3DD3D11 = new (amdContext) Image3DD3D11(amdContext, flags, obj);
  if (!pImage3DD3D11) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage3DD3D11->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage3DD3D11->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<Memory>(pImage3DD3D11);
}

//
// Helper function SyncD3D11Objects
//
void amd::SyncD3D11Objects(std::vector<amd::Memory*>& memObjects) {
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
  D3D11Object* d3dObj = interop->asD3D11Object();
  if (!d3dObj) {
    LogWarning("\nNULL D3D11 object\n");
    return;
  }
  ID3D11Query* query = d3dObj->getQuery();
  if (!query) {
    LogWarning("\nNULL ID3D11Query\n");
    return;
  }
  ID3D11Device* d3dDev;
  query->GetDevice(&d3dDev);
  if (!d3dDev) {
    LogError("\nCannot get D3D11 device from D3D11 resource\n");
    return;
  }
  ID3D11DeviceContext* pImmediateContext = NULL;
  d3dDev->GetImmediateContext(&pImmediateContext);
  if (!pImmediateContext) {
    LogError("\nCannot get D3D11 device context");
    return;
  }
  pImmediateContext->Release();

  // Flush D3D queues and make sure D3D stuff is finished
  {
    ScopedLock sl(d3dObj->getResLock());
    pImmediateContext->End(query);
    BOOL data = FALSE;
    while ((S_OK != pImmediateContext->GetData(query, &data, sizeof(BOOL), 0)) || (data != TRUE)) {
    }
  }

  d3dDev->Release();
}


#endif  //_WIN32
