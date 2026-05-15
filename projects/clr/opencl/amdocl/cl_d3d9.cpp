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

#ifdef _WIN32

#include "top.hpp"

#include "cl_d3d9_amd.hpp"
#include "platform/command.hpp"

#include <cstring>
#include <utility>


RUNTIME_ENTRY(cl_int, clGetDeviceIDsFromDX9MediaAdapterKHR,
              (cl_platform_id platform, cl_uint num_media_adapters,
               cl_dx9_media_adapter_type_khr* media_adapters_type, void* media_adapters,
               cl_dx9_media_adapter_set_khr media_adapter_set, cl_uint num_entries,
               cl_device_id* devices, cl_uint* num_devices)) {
  cl_int errcode;
  // Accept an array of DX9 devices here as the spec mention of array of num_media_adapters size.
  IDirect3DDevice9Ex** d3d9_device = static_cast<IDirect3DDevice9Ex**>(media_adapters);
  cl_device_id* gpu_devices = NULL;
  cl_uint num_gpu_devices = 0;
  static const bool VALIDATE_ONLY = true;

  if (platform != NULL && platform != AMD_PLATFORM) {
    LogWarning("\"platform\" is not a valid AMD platform");
    return CL_INVALID_PLATFORM;
  }
  // check if input parameter are correct
  if ((num_media_adapters == 0) || (media_adapters_type == NULL) || (media_adapters == NULL) ||
      (media_adapter_set != CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR &&
       media_adapter_set != CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR) ||
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

  switch (media_adapter_set) {
    case CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR:
    case CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR: {
      gpu_devices = new cl_device_id[num_gpu_devices];
      errcode = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, num_gpu_devices, gpu_devices, NULL);
      if (errcode != CL_SUCCESS) {
        break;
      }

      std::vector<amd::Device*> compatible_devices;
      for (cl_uint i = 0; i < num_gpu_devices; ++i) {
        cl_device_id device = gpu_devices[i];
        amd::Context::Flags context_flag;
        amd::Context::DeviceFlagIdx devIdx;
        switch (media_adapters_type[i]) {
          case CL_ADAPTER_D3D9_KHR:
            context_flag = amd::Context::Flags::D3D9DeviceKhr;
            devIdx = amd::Context::DeviceFlagIdx::D3D9DeviceKhrIdx;
            break;
          case CL_ADAPTER_D3D9EX_KHR:
            context_flag = amd::Context::Flags::D3D9DeviceEXKhr;
            devIdx = amd::Context::DeviceFlagIdx::D3D9DeviceEXKhrIdx;
            break;
          case CL_ADAPTER_DXVA_KHR:
            context_flag = amd::Context::Flags::D3D9DeviceVAKhr;
            devIdx = amd::Context::DeviceFlagIdx::D3D9DeviceVAKhrIdx;
            break;
        }

        for (cl_uint j = 0; j < num_media_adapters; ++j) {
          // Since there can be multiple DX9 adapters passed in the array we need to validate
          // interopability with each.
          void* external_device[amd::Context::DeviceFlagIdx::LastDeviceFlagIdx] = {};
          external_device[devIdx] = d3d9_device[j];

          if (is_valid(device) && (media_adapters_type[j] == CL_ADAPTER_D3D9EX_KHR) &&
              as_amd(device)->bindExternalDevice(context_flag, external_device, NULL,
                                                 VALIDATE_ONLY)) {
            compatible_devices.push_back(as_amd(device));
          }
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
      LogWarning("\"d3d9_device_set\" is invalid");
      errcode = CL_INVALID_VALUE;
  }

  delete[] gpu_devices;
  return errcode;
}
RUNTIME_EXIT

RUNTIME_ENTRY_RET(cl_mem, clCreateFromDX9MediaSurfaceKHR,
                  (cl_context context, cl_mem_flags flags,
                   cl_dx9_media_adapter_type_khr adapter_type, void* surface_info, cl_uint plane,
                   cl_int* errcode_ret)) {
  cl_mem clMemObj = NULL;

  cl_dx9_surface_info_khr* cl_surf_info = NULL;

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

  if ((adapter_type != CL_ADAPTER_D3D9_KHR) && (adapter_type != CL_ADAPTER_D3D9EX_KHR) &&
      (adapter_type != CL_ADAPTER_DXVA_KHR)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return clMemObj;
  }

  if (!surface_info) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("parameter \"pD3DResource\" is a NULL pointer");
    return clMemObj;
  }

  cl_surf_info = (cl_dx9_surface_info_khr*)surface_info;
  IDirect3DSurface9* pD3D9Resource = cl_surf_info->resource;
  HANDLE shared_handle = cl_surf_info->shared_handle;

  if (!pD3D9Resource) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("parameter \"surface_info\" is a NULL pointer");
    return clMemObj;
  }

  D3DSURFACE_DESC Desc;
  pD3D9Resource->GetDesc(&Desc);

  if ((Desc.Format != D3DFMT_NV_12) && (Desc.Format != D3DFMT_P010) &&
      (Desc.Format != D3DFMT_YV_12) && (plane != 0)) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    LogWarning("The plane has to be Zero if the surface format is non-planar !");
    return clMemObj;
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
  // Verify the resource is a 2D image
  return amd::clCreateImage2DFromD3D9ResourceAMD(*as_amd(context), flags, adapter_type,
                                                 cl_surf_info, plane, errcode_ret);
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clEnqueueAcquireDX9MediaSurfacesKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clDXTranslateErrorCode(amd::clEnqueueAcquireExtObjectsAMD(
      command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event,
      CL_COMMAND_ACQUIRE_DX9_MEDIA_SURFACES_KHR));
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clEnqueueReleaseDX9MediaSurfacesKHR,
              (cl_command_queue command_queue, cl_uint num_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clDXTranslateErrorCode(amd::clEnqueueReleaseExtObjectsAMD(
      command_queue, num_objects, mem_objects, num_events_in_wait_list, event_wait_list, event,
      CL_COMMAND_RELEASE_DX9_MEDIA_SURFACES_KHR));
}
RUNTIME_EXIT

//
//      clCreateImage2DFromD3D9ResourceAMD
//
cl_mem amd::clCreateImage2DFromD3D9ResourceAMD(amd::Context& amdContext, cl_mem_flags flags,
                                               cl_dx9_media_adapter_type_khr adapter_type,
                                               cl_dx9_surface_info_khr* surface_info, cl_uint plane,
                                               int* errcode_ret) {
  cl_dx9_surface_info_khr* cl_surf_info = reinterpret_cast<cl_dx9_surface_info_khr*>(surface_info);
  IDirect3DSurface9* pD3D9Resource = cl_surf_info->resource;
  HANDLE shared_handle = cl_surf_info->shared_handle;

  amd::D3D9Object obj;
  cl_int errcode =
      amd::D3D9Object::initD3D9Object(amdContext, adapter_type, surface_info, plane, obj);
  if (CL_SUCCESS != errcode) {
    *not_null(errcode_ret) = errcode;
    return (cl_mem)0;
  }

  amd::Image2DD3D9* pImage2DD3D9 = new (amdContext) amd::Image2DD3D9(amdContext, flags, obj);
  if (!pImage2DD3D9) {
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_mem)0;
  }
  if (!pImage2DD3D9->create()) {
    *not_null(errcode_ret) = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    pImage2DD3D9->release();
    return (cl_mem)0;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl<amd::Memory>(pImage2DD3D9);
}

//
// Helper function SyncD3D9Objects
//
void amd::SyncD3D9Objects(std::vector<amd::Memory*>& memObjects) {
  amd::Memory*& mem = memObjects.front();
  if (!mem) {
    LogWarning("\nNULL memory object\n");
    return;
  }
  amd::InteropObject* interop = mem->getInteropObj();
  if (!interop) {
    LogWarning("\nNULL interop object\n");
    return;
  }
  amd::D3D9Object* d3d9Obj = interop->asD3D9Object();
  if (!d3d9Obj) {
    LogWarning("\nNULL D3D9 object\n");
    return;
  }
  IDirect3DQuery9* query = d3d9Obj->getQuery();
  if (!query) {
    LogWarning("\nNULL IDirect3DQuery9\n");
    return;
  }
  amd::ScopedLock sl(d3d9Obj->getResLock());
  query->Issue(D3DISSUE_END);
  BOOL data = FALSE;
  while (S_OK != query->GetData(&data, sizeof(BOOL), D3DGETDATA_FLUSH)) {
  }
}


#endif  //_WIN32
