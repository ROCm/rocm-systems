/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "device/rocm/rocglinterop.hpp"
#include "GL/gl_interop.h"
#include "platform/interop_gl.hpp"

namespace amd::roc {
namespace GlInterop {

// ================================================================================================
bool glCanInterop(Device* device, void* GLplatformContext, void* GLdeviceContext) {
  if (!amd::GLFunctions::wglGetContextGPUInfoAMD_s) return false;

  bool canInteroperate = false;

  LUID glAdapterLuid = {0, 0};
  UINT glChainBitMask = 0;
  HGLRC hRC = static_cast<HGLRC>(GLplatformContext);

  // get GL context's LUID and chainBitMask from UGL
  if (amd::GLFunctions::wglGetContextGPUInfoAMD_s(hRC, &glAdapterLuid, &glChainBitMask)) {
    // match the adapter
    canInteroperate = device->info().luidLowPart_ == glAdapterLuid.LowPart &&
                      device->info().luidHighPart_ == glAdapterLuid.HighPart &&
                      (((1 << device->index()) & glChainBitMask) != 0);
  }

  return canInteroperate;
}

// ================================================================================================
bool glAssociate(Device* device, uint flags, void* GLplatformContext, void* GLdeviceContext) {
  static_cast<void>(flags); // unused

  return glCanInterop(device, GLplatformContext, GLdeviceContext);
}

// ================================================================================================
bool glDissociate(Device* device, void* GLplatformContext, void* GLdeviceContext) {
  return true;
}

// ================================================================================================
bool Export(amd::Memory* mem, GLenum targetType, int miplevel, hsa_handle_t* handle, int* offset) {
  assert(mem->getInteropObj() != nullptr);
  assert(mem->getInteropObj()->asGLObject() != nullptr);

  const auto* obj = mem->getInteropObj()->asGLObject();
  const auto GLContext = mem->getContext().info().hCtx_;
  const auto name = static_cast<uint>(obj->getGLName());

  GLenum type;
  switch (obj->getCLGLObjectType()) {
    case CL_GL_OBJECT_BUFFER:
      type = GL_RESOURCE_ATTACH_VERTEXBUFFER_AMD;
      break;
    case CL_GL_OBJECT_RENDERBUFFER:
      type = GL_RESOURCE_ATTACH_RENDERBUFFER_AMD;
      break;
    case CL_GL_OBJECT_TEXTURE_BUFFER:
    case CL_GL_OBJECT_TEXTURE1D:
    case CL_GL_OBJECT_TEXTURE1D_ARRAY:
    case CL_GL_OBJECT_TEXTURE2D:
    case CL_GL_OBJECT_TEXTURE2D_ARRAY:
    case CL_GL_OBJECT_TEXTURE3D:
      type = GL_RESOURCE_ATTACH_TEXTURE_AMD;
      break;
    default:
      LogError("Unknown OpenGL interop type: 0x%x", obj->getCLGLObjectType());
      return false;
  }

  const auto glRenderContext = reinterpret_cast<HGLRC>(GLContext);
  GLResource glResource = {.type = type, .name = name};
  GLResourceData glResourceData = {.version = GL_RESOURCE_DATA_VERSION};

  if (!amd::GLFunctions::wglResourceAttachAMD_s(glRenderContext, static_cast<GLvoid*>(&glResource), &glResourceData))
    return false;
  *handle = reinterpret_cast<hsa_handle_t>(glResourceData.handle);
  *offset = static_cast<int>(glResourceData.offset);

  return true;
}

} // namespace GlInterop
} // namespace amd::roc
