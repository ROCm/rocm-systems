/* Copyright (c) 2016 - 2025 Advanced Micro Devices, Inc.

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

#pragma once

#ifndef _WIN32
#include <GL/glx.h>
#include <EGL/egl.h>
#else
#include <windows.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <EGL/egl.h>
#ifndef GLX_H
struct _XDisplay;
struct __GLXcontextRec;
#endif
typedef _XDisplay Display;
typedef __GLXcontextRec* GLXContext;
#endif

#include "device/rocm/rocdevice.hpp"
#include "device/rocm/mesa_glinterop.h"
#include "device/rocm/rocregisters.hpp"

namespace amd::roc {

// Specific typed container for version 1
typedef struct metadata_amd_ci_vi_s {
  uint32_t version;   // Must be 1
  uint32_t vendorID;  // AMD | CZ
  SQ_IMG_RSRC_WORD0 word0;
  SQ_IMG_RSRC_WORD1 word1;
  SQ_IMG_RSRC_WORD2 word2;
  SQ_IMG_RSRC_WORD3 word3;
  SQ_IMG_RSRC_WORD4 word4;
  SQ_IMG_RSRC_WORD5 word5;
  SQ_IMG_RSRC_WORD6 word6;
  SQ_IMG_RSRC_WORD7 word7;
  uint32_t mip_offsets[0];  // Mip level offset bits [39:8] for each level (if any)
} metadata_amd_ci_vi_t;

class image_metadata {
 private:
  metadata_amd_ci_vi_t* data;

  image_metadata(const image_metadata&) = delete;
  image_metadata& operator=(const image_metadata&) = delete;

 public:
  image_metadata() : data(nullptr) {}
  ~image_metadata() { data = nullptr; }

  bool create(hsa_amd_image_descriptor_t* image_desc) {
    if ((image_desc->version != 1) || ((image_desc->deviceID >> 16) != 0x1002)) return false;
    data = reinterpret_cast<metadata_amd_ci_vi_t*>(image_desc);
    return true;
  }

  bool setMipLevel(uint32_t level) {
    if (level > data->word3.bits.last_level) return false;
    data->word3.bits.base_level = level;
    data->word3.bits.last_level = level;
    return true;
  }


  bool setFace(GLenum face, int32_t gfxipMajor) {
    int index = face - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
    if (index < 0 || index > 5) {
      return false;
    }
    if (data->word3.bits.type != SQ_RSRC_IMG_CUBE) {
      return false;
    }
    data->word3.bits.type = SQ_RSRC_IMG_2D_ARRAY;
    if (gfxipMajor >= 10) {
      // The base array is moved to word4 for gfx10
      data->word4.u32_all = index << 16;
    } else {
      data->word5.bits.last_array = index;
      data->word5.bits.base_array = index;
    }
    return true;
  }
};

namespace MesaInterop {
enum MESA_INTEROP_KIND { MESA_INTEROP_NONE = 0, MESA_INTEROP_GLX = 1, MESA_INTEROP_EGL = 2 };

union DisplayHandle {
  Display* glxDisplay;
  EGLDisplay eglDisplay;
  DisplayHandle() {}
  DisplayHandle(Display* display) : glxDisplay(display) {}
  DisplayHandle(EGLDisplay display) : eglDisplay(display) {}
};

union ContextHandle {
  GLXContext glxContext;
  EGLContext eglContext;
  ContextHandle() {}
  ContextHandle(GLXContext context) : glxContext(context) {}
  ContextHandle(EGLContext context) : eglContext(context) {}
};

// True if the build supports Mesa interop.
bool Supported();

// Returns true if the required subsystem is supported on the GL device.
// Must be called at least once, may be called multiple times.
bool Init(MESA_INTEROP_KIND Kind);

bool GetInfo(mesa_glinterop_device_info& info, MESA_INTEROP_KIND Kind, const DisplayHandle display,
             const ContextHandle context);

bool Export(mesa_glinterop_export_in& in, mesa_glinterop_export_out& out, MESA_INTEROP_KIND Kind,
            const DisplayHandle display, const ContextHandle context);
}  // namespace MesaInterop
}  // namespace amd::roc

