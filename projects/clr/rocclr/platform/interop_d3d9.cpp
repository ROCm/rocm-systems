/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef _WIN32

#include "top.hpp"

#include "platform/interop_d3d9.hpp"
#include "platform/command.hpp"

#include <cstring>
#include <utility>

namespace amd {
std::vector<std::pair<TD3D9RESINFO, TD3D9RESINFO>> D3D9Object::resources_;
Monitor D3D9Object::resLock_;

size_t D3D9Object::getElementBytes(D3DFORMAT d3d9Format, cl_uint plane) {
  size_t bytesPerPixel;

  switch (d3d9Format) {
    case D3DFMT_UNKNOWN:
    case D3DFMT_UYVY:
    case D3DFMT_DXT1:
    case D3DFMT_DXT2:
    case D3DFMT_DXT3:
    case D3DFMT_DXT4:
    case D3DFMT_DXT5:
    case D3DFMT_VERTEXDATA:
    case D3DFMT_D32:
    case D3DFMT_D15S1:
    case D3DFMT_D24S8:
    case D3DFMT_D24X8:
    case D3DFMT_D24X4S4:
    case D3DFMT_D16:
    case D3DFMT_INDEX16:
    case D3DFMT_INDEX32:
    case D3DFMT_MULTI2_ARGB8:
    case D3DFMT_CxV8U8:
      // Less than 1 byte per pixel - needs special consideration
      bytesPerPixel = 0;
      break;

    case D3DFMT_R3G3B2:
    case D3DFMT_P8:
    case D3DFMT_A8:
    case D3DFMT_L8:
    case D3DFMT_A4L4:
      bytesPerPixel = 1;
      break;

    case D3DFMT_R16F:
    case D3DFMT_R5G6B5:
    case D3DFMT_X1R5G5B5:
    case D3DFMT_A1R5G5B5:
    case D3DFMT_A4R4G4B4:
    case D3DFMT_A8R3G3B2:
    case D3DFMT_X4R4G4B4:
    case D3DFMT_A8P8:
    case D3DFMT_A8L8:
    case D3DFMT_V8U8:
    case D3DFMT_L6V5U5:
    case D3DFMT_D16_LOCKABLE:
    case D3DFMT_L16:
      bytesPerPixel = 2;
      break;

    case D3DFMT_R8G8B8:
    case D3DFMT_D24FS8:
      bytesPerPixel = 3;
      break;

    case D3DFMT_D32F_LOCKABLE:
    case D3DFMT_A8R8G8B8:
    case D3DFMT_R32F:
    case D3DFMT_X8R8G8B8:
    case D3DFMT_A2B10G10R10:
    case D3DFMT_A8B8G8R8:
    case D3DFMT_X8B8G8R8:
    case D3DFMT_G16R16:
    case D3DFMT_A2R10G10B10:
    case D3DFMT_Q8W8V8U8:
    case D3DFMT_X8L8V8U8:
    case D3DFMT_V16U16:
    case D3DFMT_A2W10V10U10:
    case D3DFMT_R8G8_B8G8:
    case D3DFMT_G8R8_G8B8:
    case D3DFMT_G16R16F:
    case D3DFMT_YUY2:
      bytesPerPixel = 4;
      break;

    case D3DFMT_G32R32F:
    case D3DFMT_A16B16G16R16:
    case D3DFMT_A16B16G16R16F:
    case D3DFMT_Q16W16V16U16:
      bytesPerPixel = 8;
      break;
    case D3DFMT_A32B32G32R32F:
      bytesPerPixel = 16;
      break;
    // #if !defined(D3D_DISABLE_9EX)
    //  case D3DFMT_D32_LOCKABLE:
    //  case D3DFMT_S8_LOCKABLE:
    // #endif // !D3D_DISABLE_9EX
    case D3DFMT_NV_12:
      if (plane == 0) {
        bytesPerPixel = 1;
      } else if (plane == 1) {
        bytesPerPixel = 2;
      }  // plane != 0 or != 1 shouldn't happen here
      break;
    case D3DFMT_P010:
      if (plane == 0) {
        bytesPerPixel = 2;
      } else if (plane == 1) {
        bytesPerPixel = 4;
      }  // plane != 0 or != 1 shouldn't happen here
      break;
    case D3DFMT_YV_12:
      bytesPerPixel = 1;
      break;

    default:
      bytesPerPixel = 0;
      _ASSERT(FALSE);
      break;
  }
  return bytesPerPixel;
}

void setObjDesc(amd::D3D9ObjDesc_t& objDesc, D3DSURFACE_DESC& resDesc, cl_uint plane) {
  objDesc.d3dPool_ = resDesc.Pool;
  objDesc.resType_ = resDesc.Type;
  objDesc.usage_ = resDesc.Usage;
  objDesc.d3dFormat_ = resDesc.Format;
  switch (resDesc.Format) {
    case D3DFMT_NV_12:
    case D3DFMT_P010:
      objDesc.surfRect_.left = 0;
      objDesc.surfRect_.top = 0;
      if (plane == 0) {
        objDesc.objSize_.Height = resDesc.Height;
        objDesc.objSize_.Width = resDesc.Width;
        objDesc.surfRect_.right = resDesc.Width;  // resDesc.Width/2-1;
        objDesc.surfRect_.bottom = 3 * resDesc.Height / 2;
        ;  // 3*resDesc.Height/2-1;
      } else if (plane == 1) {
        objDesc.objSize_.Height = resDesc.Height / 2;
        objDesc.objSize_.Width = resDesc.Width / 2;
        objDesc.surfRect_.right = resDesc.Width;  // resDesc.Width/2-1;
        objDesc.surfRect_.bottom = 3 * resDesc.Height / 2;
        ;  // 3*resDesc.Height/2-1;
      }  // plane != 0 or != 1 shouldn't happen here
      break;
    case D3DFMT_YV_12:
      objDesc.surfRect_.left = 0;
      if (plane == 0) {
        objDesc.objSize_.Height = resDesc.Height;
        objDesc.objSize_.Width = resDesc.Width;
        objDesc.surfRect_.top = 0;
        objDesc.surfRect_.right = resDesc.Width - 1;
        objDesc.surfRect_.bottom = resDesc.Height - 1;
      } else if (plane == 1) {
        objDesc.objSize_.Height = resDesc.Height / 2;
        objDesc.objSize_.Width = resDesc.Width / 2;
        objDesc.surfRect_.top = resDesc.Height;
        objDesc.surfRect_.right = resDesc.Width / 2 - 1;
        objDesc.surfRect_.bottom = 3 * resDesc.Height / 2 - 1;
      } else if (plane == 2) {
        objDesc.objSize_.Height = resDesc.Height / 2;
        objDesc.objSize_.Width = resDesc.Width / 2;
        objDesc.surfRect_.top = 3 * resDesc.Height / 2;
        objDesc.surfRect_.right = resDesc.Width / 2 - 1;
        objDesc.surfRect_.bottom = 2 * resDesc.Height - 1;
      }  // plane > 0 or > 2 shouldn't happen here
      break;
    default:
      objDesc.objSize_.Height = resDesc.Height;
      objDesc.objSize_.Width = resDesc.Width;
      objDesc.surfRect_.left = 0;
      objDesc.surfRect_.top = 0;
      objDesc.surfRect_.right = resDesc.Width - 1;
      objDesc.surfRect_.bottom = resDesc.Height - 1;
      if (resDesc.Format == D3DFMT_YUY2) {
        objDesc.objSize_.Width >>= 1;
      }
      break;
  }
}

bool D3D9Object::isAdapterTypeSupported(cl_dx9_media_adapter_type_khr adapter_type) {
  // CL_ADAPTER_DXVA_KHR test need a DXVA-HD capable device where IDXVAHD_Device::CreateVideoSurface
  // hands back an ordinary IDirect3DSurface9 in D3DPOOL_DEFAULT.
  return (adapter_type == CL_ADAPTER_D3D9EX_KHR) || (adapter_type == CL_ADAPTER_DXVA_KHR);
}

int D3D9Object::initD3D9Object(const Context& amdContext,
                               cl_dx9_media_adapter_type_khr adapter_type,
                               cl_dx9_surface_info_khr* cl_surf_info, cl_uint plane,
                               D3D9Object& obj) {
  ScopedLock sl(resLock_);

  cl_int errcode = CL_SUCCESS;

  // Backstop only: clGetDeviceIDsFromDX9MediaAdapterKHR reports CL_DEVICE_NOT_FOUND for the
  // unsupported adapter types, so an application that follows the spec never arrives here with
  // one.
  if (!isAdapterTypeSupported(adapter_type)) {
    return CL_INVALID_DX9_MEDIA_ADAPTER_KHR;
  }

  // Check if this ressource has already been used for interop
  IDirect3DSurface9* pD3D9res = cl_surf_info->resource;
  HANDLE shared_handle = cl_surf_info->shared_handle;

  for (const auto& it : resources_) {
    if (it.first.surfInfo.resource == cl_surf_info->resource && it.first.surfPlane == plane) {
      LogPrintfError("D3D9 interop: surface %p plane %u is already bound to a cl_mem",
                     cl_surf_info->resource, plane);
      return CL_INVALID_D3D9_RESOURCE_KHR;
    }
  }

  HRESULT hr;
  D3DQUERYTYPE desc = D3DQUERYTYPE_EVENT;

  D3DSURFACE_DESC resDesc;
  if (D3D_OK != pD3D9res->GetDesc(&resDesc)) {
    LogError("D3D9 interop: IDirect3DSurface9::GetDesc failed");
    return CL_INVALID_D3D9_RESOURCE_KHR;
  }

  // GetDevice() rather than GetContainer(): a surface that is a level of a texture
  // reports the texture as its container, whereas the owning device is always
  // reachable. The base IDirect3DDevice9 is enough here - CreateQuery and
  // CreateOffscreenPlainSurface both live on it - so plain D3D9 and DXVA adapters
  // work alongside D3D9Ex.
  IDirect3DDevice9* pDev9 = nullptr;
  if (D3D_OK != pD3D9res->GetDevice(&pDev9) || nullptr == pDev9) {
    LogError("D3D9 interop: IDirect3DSurface9::GetDevice failed");
    return CL_INVALID_D3D9_RESOURCE_KHR;
  }
  pDev9->CreateQuery(desc, &(obj.pQuery_));

  obj.handleShared_ = shared_handle;
  obj.surfPlane_ = plane;
  obj.surfInfo_ = *cl_surf_info;
  obj.adapterType_ = adapter_type;

  // Init defaults
  setObjDesc(obj.objDescOrig_, resDesc, plane);
  obj.objDesc_ = obj.objDescOrig_;

  // A plain offscreen surface cannot be described by CLQueryResource whatever its origin, so a
  // shared handle the application supplies for one is not enough on its own. For non-planar
  // formats, stage into a texture in that case too - unless what we were handed is already a
  // texture level, which the driver does describe and which therefore needs no copy.
  bool stageIntoTexture = false;
  if (shared_handle != nullptr && resDesc.Format != D3DFMT_NV_12 &&
      resDesc.Format != D3DFMT_P010 && resDesc.Format != D3DFMT_YV_12) {
    IDirect3DTexture9* container = nullptr;
    if (FAILED(pD3D9res->GetContainer(IID_IDirect3DTexture9,
                                      reinterpret_cast<void**>(&container))) ||
        container == nullptr) {
      stageIntoTexture = true;
    } else {
      container->Release();
    }
  }

  // shared handle cases if the shared_handle is nullptr
  // first check if the format is NV12 or YV12, which we need special handling
  if (nullptr == shared_handle || stageIntoTexture) {
    bool found = false;
    for (const auto& it : resources_) {
      if (it.first.surfInfo.resource == cl_surf_info->resource && it.first.surfPlane != plane) {
        obj.handleShared_ = it.second.surfInfo.shared_handle;
        obj.pD3D9Res_ = it.second.surfInfo.resource;
        obj.pD3D9Res_->AddRef();
        obj.objDesc_ = obj.objDescOrig_;
        found = true;
        break;
      }
    }
    if (!found) {
      obj.handleShared_ = 0;
      // Stage into a texture, mirroring the D3D11 backend's CreateTexture2D
      // (interop_d3d11.cpp:252). The UMD's CLQueryResource only describes textures and YUV
      // surfaces - a plain offscreen surface is refused as "neither texture nor YUV" - so
      // staging into a texture level is what makes an SRD available for the non-YUV formats.
      // Planar YUV keeps the offscreen-plain path, where the query already works, and any
      // format that cannot back a render target falls back to it too.
      const bool planarYuv = (resDesc.Format == D3DFMT_NV_12) ||
                             (resDesc.Format == D3DFMT_P010) || (resDesc.Format == D3DFMT_YV_12);
      hr = D3DERR_INVALIDCALL;
      if (!planarYuv) {
        // D3DUSAGE_RENDERTARGET first, because StretchRect needs a render-target destination.
        // Formats that cannot be render targets (A8L8 on this driver) still make usable
        // textures, so try plain usage before falling back to an unqueryable surface.
        for (DWORD usage : {static_cast<DWORD>(D3DUSAGE_RENDERTARGET), static_cast<DWORD>(0)}) {
          IDirect3DTexture9* stagingTex = nullptr;
          obj.handleShared_ = 0;
          hr = pDev9->CreateTexture(resDesc.Width, resDesc.Height, 1, usage, resDesc.Format,
                                    D3DPOOL_DEFAULT, &stagingTex, &obj.handleShared_);
          if (SUCCEEDED(hr) && stagingTex != nullptr) {
            hr = stagingTex->GetSurfaceLevel(0, &obj.pD3D9Res_);
            // A child surface keeps its container alive, so this reference can go now.
            stagingTex->Release();
          }
          if (SUCCEEDED(hr)) {
            break;
          }
          obj.handleShared_ = 0;
          obj.pD3D9Res_ = nullptr;
        }
      }

      if (FAILED(hr)) {
        hr = pDev9->CreateOffscreenPlainSurface(resDesc.Width, resDesc.Height, resDesc.Format,
                                                resDesc.Pool, &obj.pD3D9Res_, &obj.handleShared_);
      }

      if (D3D_OK != hr) {
        LogPrintfError(
            "D3D9 interop: staging CreateOffscreenPlainSurface(%ux%u fmt=0x%x pool=%u) failed: "
            "0x%xh",
            resDesc.Width, resDesc.Height, resDesc.Format, resDesc.Pool, hr);
        errcode = CL_INVALID_D3D9_RESOURCE_KHR;
      }
    }

    // put the original info into the obj
    obj.pD3D9ResOrig_ = pD3D9res;
    obj.pD3D9ResOrig_->AddRef();  // addRef in case lost the resource
  } else {
    // Share the original resource
    obj.pD3D9ResOrig_ = nullptr;
    obj.pD3D9Res_ = pD3D9res;
    obj.pD3D9Res_->AddRef();
  }

  // Release the device interface obtained from the surface
  if (pDev9) pDev9->Release();

  // Check for CL format compatibilty
  if (obj.objDesc_.resType_ == D3DRTYPE_SURFACE) {
    cl_image_format clFmt = obj.getCLFormatFromD3D9(obj.objDesc_.d3dFormat_, plane);
    amd::Image::Format imageFormat(clFmt);
    if (!imageFormat.isSupported(amdContext)) {
      return CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
    }
  }

  TD3D9RESINFO d3d9ObjOri = {*cl_surf_info, plane};
  TD3D9RESINFO d3d9ObjShared = {{obj.pD3D9Res_, obj.handleShared_}, plane};

  if (errcode == CL_SUCCESS) {
    resources_.push_back({d3d9ObjOri, d3d9ObjShared});
  }

  return errcode;
}
cl_uint D3D9Object::getMiscFlag() {
  switch (objDescOrig_.d3dFormat_) {
    case D3DFMT_NV_12:
    case D3DFMT_P010:
      return 1;
      break;
    case D3DFMT_YV_12:
      return 2;
      break;
    case D3DFMT_YUY2:
      return 3;
      break;
    default:
      return 0;
      break;
  }
}

cl_image_format D3D9Object::getCLFormatFromD3D9() {
  return getCLFormatFromD3D9(objDesc_.d3dFormat_, surfPlane_);
}

cl_image_format D3D9Object::getCLFormatFromD3D9(D3DFORMAT d3d9Fmt, cl_uint plane) {
  cl_image_format fmt;

  fmt.image_channel_order = 0;      // CL_RGBA;
  fmt.image_channel_data_type = 0;  // CL_UNSIGNED_INT8;

  switch (d3d9Fmt) {
    case D3DFMT_R32F:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case D3DFMT_R16F:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case D3DFMT_L16:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case D3DFMT_A8:
      fmt.image_channel_order = CL_A;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_L8:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_G32R32F:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case D3DFMT_G16R16F:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case D3DFMT_G16R16:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case D3DFMT_A8L8:
      fmt.image_channel_order = CL_RG;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_A32B32G32R32F:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_FLOAT;
      break;

    case D3DFMT_A16B16G16R16F:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;

    case D3DFMT_A16B16G16R16:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;

    case D3DFMT_A8B8G8R8:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_X8B8G8R8:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_A8R8G8B8:
      fmt.image_channel_order = CL_BGRA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;

    case D3DFMT_X8R8G8B8:
      fmt.image_channel_order = CL_BGRA;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;
    case D3DFMT_NV_12:
      fmt.image_channel_data_type = CL_UNORM_INT8;
      if (plane == 0) {
        fmt.image_channel_order = CL_R;
      } else if (plane == 1) {
        fmt.image_channel_order = CL_RG;
      }
      break;
    case D3DFMT_P010:
      fmt.image_channel_data_type = CL_UNORM_INT16;
      if (plane == 0) {
        fmt.image_channel_order = CL_R;
      } else if (plane == 1) {
        fmt.image_channel_order = CL_RG;
      }
      break;
    case D3DFMT_YV_12:
      fmt.image_channel_order = CL_R;
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;
    case D3DFMT_YUY2:
      fmt.image_channel_order = CL_RGBA;
      fmt.image_channel_data_type = CL_UNSIGNED_INT8;
      break;
    case D3DFMT_UNKNOWN:
    case D3DFMT_R8G8B8:
    case D3DFMT_R5G6B5:
    case D3DFMT_X1R5G5B5:
    case D3DFMT_A1R5G5B5:
    case D3DFMT_A4R4G4B4:
    case D3DFMT_R3G3B2:
    case D3DFMT_A8R3G3B2:
    case D3DFMT_X4R4G4B4:
    case D3DFMT_A2B10G10R10:
    case D3DFMT_A2R10G10B10:
    case D3DFMT_A8P8:
    case D3DFMT_P8:
    case D3DFMT_A4L4:
    case D3DFMT_V8U8:
    case D3DFMT_L6V5U5:
    case D3DFMT_X8L8V8U8:
    case D3DFMT_Q8W8V8U8:
    case D3DFMT_V16U16:
    case D3DFMT_A2W10V10U10:
    case D3DFMT_UYVY:
    case D3DFMT_R8G8_B8G8:
    case D3DFMT_G8R8_G8B8:
    case D3DFMT_DXT1:
    case D3DFMT_DXT2:
    case D3DFMT_DXT3:
    case D3DFMT_DXT4:
    case D3DFMT_DXT5:
    case D3DFMT_D16_LOCKABLE:
    case D3DFMT_D32:
    case D3DFMT_D15S1:
    case D3DFMT_D24S8:
    case D3DFMT_D24X8:
    case D3DFMT_D24X4S4:
    case D3DFMT_D16:
    case D3DFMT_D32F_LOCKABLE:
    case D3DFMT_D24FS8:
    // #if !defined(D3D_DISABLE_9EX)
    case D3DFMT_D32_LOCKABLE:
    case D3DFMT_S8_LOCKABLE:
    // #endif // !D3D_DISABLE_9EX
    case D3DFMT_VERTEXDATA:
    case D3DFMT_INDEX16:
    case D3DFMT_INDEX32:
    case D3DFMT_Q16W16V16U16:
    case D3DFMT_MULTI2_ARGB8:
    case D3DFMT_CxV8U8:
    // #if !defined(D3D_DISABLE_9EX)
    case D3DFMT_A1:
    case D3DFMT_A2B10G10R10_XR_BIAS:
    case D3DFMT_BINARYBUFFER:
      _ASSERT(FALSE);  // NOT SURPPORTED
      break;
    // #endif // !D3D_DISABLE_9EX
    default:
      _ASSERT(FALSE);
      break;
  }

  return fmt;
}

bool D3D9Object::copyOrigToShared() {
  // Don't copy if there is no orig
  if (nullptr == getD3D9ResOrig()) return true;

  IDirect3DDevice9* d3dDev = nullptr;
  HRESULT hr;
  ScopedLock sl(getResLock());

  IDirect3DSurface9* srcSurf = getD3D9ResOrig();
  IDirect3DSurface9* dstSurf = getD3D9Resource();

  hr = getD3D9Resource()->GetDevice(&d3dDev);
  if (hr != D3D_OK || !d3dDev) {
    LogError("\nCannot get D3D9 device from D3D9 surface\n");
    return false;
  }

  hr = d3dDev->StretchRect(srcSurf, nullptr, dstSurf, nullptr, D3DTEXF_NONE);
  d3dDev->Release();
  if (hr != D3D_OK) {
    LogError("\ncopy original surface to shared surface failed\n");
    return false;
  }
  // Flush D3D queues and make sure D3D stuff is finished.
  //
  // GetData returns S_FALSE while the query is still pending and leaves the output buffer
  // untouched, so only its return value can end this wait. The previous condition also tested an
  // uninitialised `data`: whenever that stack slot happened to hold a non-zero value the && went
  // false on the first iteration and the wait returned with the StretchRect still in flight,
  // handing OpenCL a surface whose copy had not landed. That is an intermittent
  // wrong-data failure, not a hang.
  pQuery_->Issue(D3DISSUE_END);
  BOOL data = FALSE;
  HRESULT queryHr;
  while (S_FALSE == (queryHr = pQuery_->GetData(&data, sizeof(BOOL), D3DGETDATA_FLUSH))) {
  }
  if (FAILED(queryHr)) {
    // A lost device ends the wait here rather than spinning on an error forever.
    LogPrintfError("D3D9 interop: IDirect3DQuery9::GetData failed: 0x%xh", queryHr);
    return false;
  }

  return true;
}

bool D3D9Object::copySharedToOrig() {
  // Don't copy if there is no orig
  if (nullptr == getD3D9ResOrig()) return true;

  IDirect3DDevice9* d3dDev = nullptr;
  HRESULT hr;
  ScopedLock sl(getResLock());

  hr = getD3D9Resource()->GetDevice(&d3dDev);
  if (hr != D3D_OK || !d3dDev) {
    LogError("\nCannot get D3D9 device from D3D9 surface\n");
    return false;
  }

  hr = d3dDev->StretchRect(getD3D9Resource(), nullptr, getD3D9ResOrig(), nullptr, D3DTEXF_NONE);
  d3dDev->Release();
  if (hr != D3D_OK) {
    LogError("\ncopy shared surface to original surface failed\n");
    return false;
  }

  return true;
}

void Image2DD3D9::initDeviceMemory() {
  deviceMemories_ =
      reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(Image2DD3D9));
  memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
}

}  // namespace amd

#endif  //_WIN32
