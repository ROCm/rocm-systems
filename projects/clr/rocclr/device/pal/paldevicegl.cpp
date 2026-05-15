/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#include "platform/context.hpp"
#include "device/device.hpp"
#include "platform/runtime.hpp"
#include "platform/agent.hpp"
#ifdef _WIN32
#include <d3d10_1.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#endif  // _WIN32

#include <GL/gl.h>
#include <GL/glext.h>
#include "CL/cl_gl.h"
#include "paldevice.hpp"
// #include "cwddeci.h"
#include <GL/gl.h>
#include "GL/gl_interop.h"
#ifdef ATI_OS_LINUX
#include <stdlib.h>
#include <dlfcn.h>
#include "GL/glx.h"
#include "GL/glxext.h"
#endif

/**
 * Device information returned by Mesa/Orca.
 */
typedef struct _mesa_glinterop_device_info {
  uint32_t size; /* size of this structure */

  /* PCI location */
  uint32_t pci_segment_group;
  uint32_t pci_bus;
  uint32_t pci_device;
  uint32_t pci_function;

  /* Device identification */
  uint32_t vendor_id;
  uint32_t device_id;
} mesa_glinterop_device_info;

#ifdef ATI_OS_LINUX
typedef void* (*PFNGlxGetProcAddress)(const GLubyte* procName);
static PFNGlxGetProcAddress pfnGlxGetProcAddress = nullptr;
typedef int(APIENTRYP PFNMesaGLInteropGLXQueryDeviceInfo)(Display* dpy, GLXContext context,
                                                          mesa_glinterop_device_info* out);
static PFNMesaGLInteropGLXQueryDeviceInfo pfnMesaGLInteropGLXQueryDeviceInfo = nullptr;
typedef Bool(APIENTRYP PFNGLXBEGINCLINTEROPAMD)(GLXContext context, GLuint flags);
typedef Bool(APIENTRYP PFNGLXENDCLINTEROPAMD)(GLXContext context, GLuint flags);
typedef Bool(APIENTRYP PFNGLXRESOURCEATTACHAMD)(GLXContext context, GLvoid* resource,
                                                GLvoid* pResourceData);
typedef Bool(APIENTRYP PFNGLXRESOURCEDETACHAMD)(GLXContext context, GLvoid* resource);
typedef Bool(APIENTRYP PFNGLXRESOURCEDETACHAMD)(GLXContext context, GLvoid* resource);
typedef Bool(APIENTRYP PFNGLXRESOURCEDETACHAMD)(GLXContext context, GLvoid* resource);
typedef Bool(APIENTRYP PFNGLXGETCONTEXTMVPUINFOAMD)(GLXContext context, GLuint* deviceId,
                                                    GLuint* chainMask);
static PFNGLXBEGINCLINTEROPAMD glXBeginCLInteropAMD = nullptr;
static PFNGLXENDCLINTEROPAMD glXEndCLInteropAMD = nullptr;
static PFNGLXRESOURCEATTACHAMD glXResourceAttachAMD = nullptr;
static PFNGLXRESOURCEDETACHAMD glxResourceAcquireAMD = nullptr;
static PFNGLXRESOURCEDETACHAMD glxResourceReleaseAMD = nullptr;
static PFNGLXRESOURCEDETACHAMD glXResourceDetachAMD = nullptr;
static PFNGLXGETCONTEXTMVPUINFOAMD glXGetContextMVPUInfoAMD = nullptr;
#else
typedef PROC(WINAPI* PFNWGLGETPROCADDRESS)(LPCSTR name);
typedef HGLRC(WINAPI* PFNWGLGETCURRENTCONTEXT)(void);
typedef HGLRC(WINAPI* PFNWGLCREATECONTEXT)(HDC hdc);
typedef BOOL(WINAPI* PFNWGLDELETECONTEXT)(HGLRC hglrc);
typedef BOOL(WINAPI* PFNWGLMAKECURRENT)(HDC hdc, HGLRC hglrc);
static PFNWGLGETPROCADDRESS pfnWglGetProcAddress = nullptr;
static PFNWGLGETCURRENTCONTEXT pfnWglGetCurrentContext = nullptr;
static PFNWGLCREATECONTEXT pfnWglCreateContext = nullptr;
static PFNWGLDELETECONTEXT pfnWglDeleteContext = nullptr;
static PFNWGLMAKECURRENT pfnWglMakeCurrent = nullptr;
static PFNWGLBEGINCLINTEROPAMD wglBeginCLInteropAMD = nullptr;
static PFNWGLENDCLINTEROPAMD wglEndCLInteropAMD = nullptr;
static PFNWGLRESOURCEATTACHAMD wglResourceAttachAMD = nullptr;
static PFNWGLRESOURCEDETACHAMD wglResourceAcquireAMD = nullptr;
static PFNWGLRESOURCEDETACHAMD wglResourceReleaseAMD = nullptr;
static PFNWGLRESOURCEDETACHAMD wglResourceDetachAMD = nullptr;
static PFNWGLGETCONTEXTGPUINFOAMD wglGetContextGPUInfoAMD = nullptr;
#endif
bool gGlFuncInit = false;

namespace amd::pal {

//
/// GSL Surface Formats as per defined in cmSurfFmtEnum enum in
/// //depot/stg/ugl/drivers/ugl/src/include/cm_enum.h
//
typedef enum cmSurfFmtEnum {
  CM_SURF_FMT_NOOVERRIDE = -1,
  CM_SURF_FMT_LUMINANCE8,    ///< Luminance,  8 bits per element packed as (@c LLLLLLLL)
  CM_SURF_FMT_LUMINANCE16,   ///< Luminance, 16 bits per element packed as (@c LLLLLLLLLLLLLLLL)
  CM_SURF_FMT_LUMINANCE16F,  ///< Luminance, 16 bits per element packed as (@c LLLLLLLLLLLLLLLL)
  CM_SURF_FMT_LUMINANCE32F,  ///< Luminance, 32 bits per element packed as (@c
                             ///< LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL)
  CM_SURF_FMT_INTENSITY8,    ///< Intensity,  8 bits per element packed as (@c IIIIIIII)
  CM_SURF_FMT_INTENSITY16,   ///< Intensity, 16 bits per element packed as (@c IIIIIIIIIIIIIIII)
  CM_SURF_FMT_INTENSITY16F,  ///< Intensity, 16 bits per element packed as (@c IIIIIIIIIIIIIIII)
  CM_SURF_FMT_INTENSITY32F,  ///< Intensity, 32 bits per element packed as (@c
                             ///< IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII)
  CM_SURF_FMT_ALPHA8,        ///< Alpha,      8 bits per element packed as (@c AAAAAAAA)
  CM_SURF_FMT_ALPHA16,       ///< Alpha,     16 bits per element packed as (@c AAAAAAAAAAAAAAAA)
  CM_SURF_FMT_ALPHA16F,      ///< Alpha,     16 bits per element packed as (@c AAAAAAAAAAAAAAAA)
  CM_SURF_FMT_ALPHA32F,      ///< Alpha,     32 bits per element packed as (@c
                             ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
  CM_SURF_FMT_LUMINANCE8_ALPHA8,      ///< Luminance Alpha, 16 bits per element packed as (@c
                                      ///< AAAAAAAALLLLLLLL)
  CM_SURF_FMT_LUMINANCE16_ALPHA16,    ///< Luminance Alpha, 32 bits per element packed as (@c
                                      ///< AAAAAAAAAAAAAAAALLLLLLLLLLLLLLLL)
  CM_SURF_FMT_LUMINANCE16F_ALPHA16F,  ///< Luminance Alpha, 32 bits per element packed as (@c
                                      ///< AAAAAAAAAAAAAAAALLLLLLLLLLLLLLLL)
  CM_SURF_FMT_LUMINANCE32F_ALPHA32F,  ///< Luminance Alpha, 64 bits per element packed as (@c
                                      ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL)
  CM_SURF_FMT_B2_G3_R3,  ///< RGB,    8 bits per element packed as (@c RRRGGGBB)
  CM_SURF_FMT_B5_G6_R5,  ///< RGB,   16 bits per element packed as (@c RRRRRGGGGGGBBBBB)
  CM_SURF_FMT_BGRX4,     ///< RGB,   16 bits per element packed as (@c XXXXRRRRGGGGBBBB)
  CM_SURF_FMT_BGR5_X1,   ///< RGB,   16 bits per element packed as (@c XRRRRRGGGGGBBBBB)
  CM_SURF_FMT_BGRX8,     ///< RGB,   32 bits per element packed as (@c
                         ///< XXXXXXXXRRRRRRRRGGGGGGGGBBBBBBBB) - XXX unused by current driver
  CM_SURF_FMT_BGR10_X2,  ///< RGB,   32 bits per element packed as (@c
                         ///< XXRRRRRRRRRRGGGGGGGGGGBBBBBBBBBB)
  CM_SURF_FMT_BGRX16,    ///< RGB,   64 bits per element packed as (@c
                         ///< XXXXXXXXXXXXXXXXRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_BGRX16F,   ///< RGB,   64 bits per element packed as (@c
                         ///< XXXXXXXXXXXXXXXXRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_BGRX32F,   ///< RGB,  128 bits per element packed as (@c
                        ///< XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_RGBX4,     ///< RGB,   16 bits per element packed as (@c XXXXBBBBGGGGRRRR)
  CM_SURF_FMT_RGB5_X1,   ///< RGB,   16 bits per element packed as (@c XBBBBBGGGGGRRRRR)
  CM_SURF_FMT_RGBX8,     ///< RGB,   32 bits per element packed as (@c
                         ///< XXXXXXXXBBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_RGB10_X2,  ///< RGB,   32 bits per element packed as (@c
                         ///< XXBBBBBBBBBBGGGGGGGGGGRRRRRRRRRR)
  CM_SURF_FMT_RGBX16,    ///< RGB,   64 bits per element packed as (@c
                         ///< XXXXXXXXXXXXXXXXBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGBX16F,   ///< RGB,   64 bits per element packed as (@c
                         ///< XXXXXXXXXXXXXXXXBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGBX32F,   ///< RGB,  128 bits per element packed as (@c
                        ///< XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_BGRA4,     ///< RGBA,  16 bits per element packed as (@c AAAARRRRGGGGBBBB)
  CM_SURF_FMT_BGR5_A1,   ///< RGBA,  16 bits per element packed as (@c ARRRRRGGGGGBBBBB)
  CM_SURF_FMT_BGRA8,     ///< RGBA,  32 bits per element packed as (@c
                         ///< AAAAAAAARRRRRRRRGGGGGGGGBBBBBBBB)
  CM_SURF_FMT_BGR10_A2,  ///< RGBA,  32 bits per element packed as (@c
                         ///< AARRRRRRRRRRGGGGGGGGGGBBBBBBBBBB)
  CM_SURF_FMT_BGRA16,    ///< RGBA,  64 bits per element packed as (@c
                         ///< AAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_BGRA16F,   ///< RGBA,  64 bits per element packed as (@c
                         ///< AAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_BGRA32F,   ///< RGBA, 128 bits per element packed as (@c
                        ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_RGBA4,     ///< RGBA,  16 bits per element packed as (@c AAAABBBBGGGGRRRR)
  CM_SURF_FMT_RGB5_A1,   ///< RGBA,  16 bits per element packed as (@c ABBBBBGGGGGRRRRR)
  CM_SURF_FMT_RGBA8,     ///< RGBA,  32 bits per element packed as (@c
                         ///< AAAAAAAABBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_RGB10_A2,  ///< RGBA,  32 bits per element packed as (@c
                         ///< AABBBBBBBBBBGGGGGGGGGGRRRRRRRRRR)
  CM_SURF_FMT_RGBA16,    ///< RGBA,  64 bits per element packed as (@c
                         ///< AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGBA16F,   ///< RGBA,  64 bits per element packed as (@c
                         ///< AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGBA32I,   ///< RGBA, 128 bits per element packed as (@c
                        ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGBA32F,  ///< RGBA, 128 bits per element packed as (@c
                        ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_DUDV8,    ///< DUDV   16 bits per element packed as (@c VVVVVVVVUUUUUUUU)
  CM_SURF_FMT_DXT1,     ///< compressed, DXT1
  CM_SURF_FMT_DXT2_3,   ///< compressed, DXT2_3
  CM_SURF_FMT_DXT4_5,   ///< compressed, DXT4_5
  CM_SURF_FMT_ATI1N,    ///< compressed, 1 component
  CM_SURF_FMT_ATI2N,    ///< compressed, 2 component
  CM_SURF_FMT_DEPTH16,  ///< depth, 16 bits per element packed as (@c DDDDDDDDDDDDDDDD)
  CM_SURF_FMT_DEPTH16F,            ///< depth, 16 bits per element packed as (@c DDDDDDDDDDDDDDDD)
  CM_SURF_FMT_DEPTH24_X8,          ///< depth, 32 bits per element packed as (@c
                                   ///< XXXXXXXXDDDDDDDDDDDDDDDDDDDDDDDD)
  CM_SURF_FMT_DEPTH24F_X8,         ///< depth, 32 bits per element packed as (@c
                                   ///< SSSSSSSSDDDDDDDDDDDDDDDDDDDDDDDD)
  CM_SURF_FMT_DEPTH24_STEN8,       ///< depth + stencil, 32 bits per element packed as (@c
                                   ///< SSSSSSSSDDDDDDDDDDDDDDDDDDDDDDDD)
  CM_SURF_FMT_DEPTH24F_STEN8,      ///< depth + stencil, 32 bits per element packed as (@c
                                   ///< SSSSSSSSDDDDDDDDDDDDDDDDDDDDDDDD)
  CM_SURF_FMT_DEPTH32F_X24_STEN8,  ///< depth + stencil, 64 bits per element packed as (@c
                                   ///< XXXXXXXXXXXXXXXXXXXXXXXXSSSSSSSSDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD)
  CM_SURF_FMT_DEPTH32F,        ///< depth, 32 bits per element packed as (@c
                               ///< DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD)
  CM_SURF_FMT_sR11_sG11_sB10,  ///< RGB,   32 bits per element packed as (@c
                               ///< RRRRRRRRRRRGGGGGGGGGGGBBBBBBBBBB)
  CM_SURF_FMT_sU16,            ///<
  CM_SURF_FMT_sUV16,           ///<
  CM_SURF_FMT_sUVWQ16,         ///<
  CM_SURF_FMT_RG16,  ///< RG,    32 bits per element packed as (@c RRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGG)
  CM_SURF_FMT_RG16F,     ///< RG,    32 bits per element packed as (@c
                         ///< RRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGG)
  CM_SURF_FMT_RG32F,     ///< RG,    64 bits per element packed as (@c
                         ///< RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG)
  CM_SURF_FMT_ABGR4,     ///< RGBA,  16 bits per element packed as (@c RRRRGGGGBBBBAAAA)
  CM_SURF_FMT_A1_BGR5,   ///< RGBA,  16 bits per element packed as (@c RRRRRGGGGGBBBBBA)
  CM_SURF_FMT_ABGR8,     ///< RGBA,  32 bits per element packed as (@c
                         ///< RRRRRRRRGGGGGGGGBBBBBBBBAAAAAAAA)
  CM_SURF_FMT_A2_BGR10,  ///< RGBA,  32 bits per element packed as (@c
                         ///< RRRRRRRRRRGGGGGGGGGGBBBBBBBBBBAA)
  CM_SURF_FMT_ABGR16,    ///< RGBA,  64 bits per element packed as (@c
                         ///< RRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAA)
  CM_SURF_FMT_ABGR16F,   ///< RGBA,  64 bits per element packed as (@c
                         ///< RRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAA)
  CM_SURF_FMT_ABGR32F,   ///< RGBA, 128 bits per element packed as (@c
                        ///< RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
  CM_SURF_FMT_DXT1A,
  CM_SURF_FMT_sRGB10_A2,  ///< RGBA,  32  bits per element packed as signed (@c
                          ///< AABBBBBBBBBBGGGGGGGGGGRRRRRRRRRR)
  CM_SURF_FMT_sR8,        ///< R,     8   bits per element packed as signed (@c RRRRRRRR)
  CM_SURF_FMT_sRG8,       ///< RG,    16  bits per element packed as signed (@c RRRRRRRRGGGGGGGG)
  CM_SURF_FMT_sR32I,      ///< R,     32  bits per element packed as signed (@c
                          ///< RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_sRG32I,     ///< RG,    64  bits per element packed as signed (@c
                          ///< RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG)
  CM_SURF_FMT_sRGBA32I,   ///< RGBA,  128 bits per element packed as signed (@c
                         ///< RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
  CM_SURF_FMT_R32I,    ///< R,     32  bits per element packed as (@c
                       ///< RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RG32I,   ///< RG,    64  bits per element packed as (@c
                       ///< RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG)
  CM_SURF_FMT_RG8,     ///< RG8,   16 bits per element packed as (@c RRRRRRRRGGGGGGGG)
  CM_SURF_FMT_sRGBA8,  ///< RGBA8, 32 bits per element packed as signed (@c
                       ///< RRRRRRRRGGGGGGGGBBBBBBBBAAAAAAAA)
  CM_SURF_FMT_R11F_G11F_B10F,                ///< RGB,   32 bits per element packed as (@c
                                             ///< BBBBBBBBBBGGGGGGGGGGGRRRRRRRRRRR)
  CM_SURF_FMT_RGB9_E5,                       ///< RGB,   32 bits per element packed as (@c
                                             ///< EEEEEBBBBBBBBBGGGGGGGGGRRRRRRRRR)
  CM_SURF_FMT_LUMINANCE_LATC1,               ///< compressed LATC1
  CM_SURF_FMT_SIGNED_LUMINANCE_LATC1,        ///< compressed signed LATC1
  CM_SURF_FMT_LUMINANCE_ALPHA_LATC2,         ///< compressed LATC2
  CM_SURF_FMT_SIGNED_LUMINANCE_ALPHA_LATC2,  ///< compressed signed LATC2
  CM_SURF_FMT_RED_RGTC1,                     ///< compressed RGTC1
  CM_SURF_FMT_SIGNED_RED_RGTC1,              ///< compressed signed RGTC1
  CM_SURF_FMT_RED_GREEN_RGTC2,               ///< compressed RGTC2
  CM_SURF_FMT_SIGNED_RED_GREEN_RGTC2,        ///< compressed signed RGTC2
  CM_SURF_FMT_R8,                            ///< R,     8   bits per element packed (@c RRRRRRRR)
  CM_SURF_FMT_R16,     ///< R,    16   bits per element packed (@c RRRRRRRRRRRRRRRR)
  CM_SURF_FMT_R16F,    ///< R,    16   bits per element packed (@c RRRRRRRRRRRRRRRR)
  CM_SURF_FMT_R32F,    ///< R,    32   bits per element packed (@c RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_R8I,     ///< R,     8   bits per element packed (@c RRRRRRRR)
  CM_SURF_FMT_sR8I,    ///< R,     8   bits per element packed as signed (@c RRRRRRRR)
  CM_SURF_FMT_RG8I,    ///< RG,   16   bits per element packed (@c RRRRRRRRGGGGGGGG)
  CM_SURF_FMT_sRG8I,   ///< RG,   16   bits per element packed as signed (@c RRRRRRRRGGGGGGGG)
  CM_SURF_FMT_R16I,    ///< R,    16   bits per element packed (@c RRRRRRRRRRRRRRRR)
  CM_SURF_FMT_sR16I,   ///< R,    16   bits per element packed as signed (@c RRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RG16I,   ///< RG,   32   bits per element packed (@c RRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGG)
  CM_SURF_FMT_sRG16I,  ///< RG,   32   bits per element packed as signed (@c
                       ///< RRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGG)
  CM_SURF_FMT_RGBA32UI,  ///< RGBA, 128 bits per element packed as (@c
                         ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_RGBX32UI,  ///< RGBX,  128 bits per element packed as(@c
                         ///< XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_ALPHA32UI,            ///< Alpha, 32 bits per element packed as (@c
                                    ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
  CM_SURF_FMT_INTENSITY32UI,        ///< Intensity, 32 bits per element packed as (@c
                                    ///< IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII)
  CM_SURF_FMT_LUMINANCE32UI,        ///< Luminance, 32 bits per element packed as (@c
                                    ///< LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL)
  CM_SURF_FMT_LUMINANCE_ALPHA32UI,  ///< Luminance Alpha, 64 bits per element packed as (@c
                                    ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL)
  CM_SURF_FMT_RGBA16UI,       ///< RGBA,  64 bits per element packed as (@c
                              ///< AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGBX16UI,       ///< RGB,   64 bits per element packed as (@c
                              ///< XXXXXXXXXXXXXXXXBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_ALPHA16UI,      ///< Alpha, 16 bits per element packed as (@c AAAAAAAAAAAAAAAA)
  CM_SURF_FMT_INTENSITY16UI,  ///< Intensity, 16 bits per element packed as (@c IIIIIIIIIIIIIIII)
  CM_SURF_FMT_LUMINANCE16UI,  ///< Luminance, 16 bits per element packed as (@c LLLLLLLLLLLLLLLL)
  CM_SURF_FMT_LUMINANCE_ALPHA16UI,  ///< Luminance Alpha, 32 bits per element packed as (@c
                                    ///< AAAAAAAAAAAAAAAALLLLLLLLLLLLLLLL)
  CM_SURF_FMT_RGBA8UI,              ///< RGBA,  32 bits per element packed as (@c
                                    ///< AAAAAAAABBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_RGBX8UI,              ///< RGB,   32 bits per element packed as (@c
                                    ///< XXXXXXXXBBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_ALPHA8UI,             ///< Alpha, 8 bits per element packed as (@c AAAAAAAA)
  CM_SURF_FMT_INTENSITY8UI,         ///< Intensity, 8 bits per element packed as (@c IIIIIIII)
  CM_SURF_FMT_LUMINANCE8UI,         ///< Luminance, 8 bits per element packed as (@c LLLLLLLL)
  CM_SURF_FMT_LUMINANCE_ALPHA8UI,   ///< Luminance Alpha, 32 bits per element packed as (@c
                                    ///< AAAAAAAALLLLLLLL)
  CM_SURF_FMT_sRGBX32I,             ///< RGBX,  128 bits per element packed as(@c
                         ///< XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB)
  CM_SURF_FMT_sALPHA32I,            ///< Alpha, 32 bits per element packed as (@c
                                    ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
  CM_SURF_FMT_sINTENSITY32I,        ///< Intensity, 32 bits per element packed as (@c
                                    ///< IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII)
  CM_SURF_FMT_sLUMINANCE32I,        ///< Luminance, 32 bits per element packed as (@c
                                    ///< LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL)
  CM_SURF_FMT_sLUMINANCE_ALPHA32I,  ///< Luminance Alpha, 64 bits per element packed as (@c
                                    ///< AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL)
  CM_SURF_FMT_sRGBA16I,       ///< RGBA,  64 bits per element packed as (@c
                              ///< AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_sRGBX16I,       ///< RGB,   64 bits per element packed as (@c
                              ///< XXXXXXXXXXXXXXXXBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_sALPHA16I,      ///< Alpha, 16 bits per element packed as (@c AAAAAAAAAAAAAAAA)
  CM_SURF_FMT_sINTENSITY16I,  ///< Intensity, 16 bits per element packed as (@c IIIIIIIIIIIIIIII)
  CM_SURF_FMT_sLUMINANCE16I,  ///< Luminance, 16 bits per element packed as (@c LLLLLLLLLLLLLLLL)
  CM_SURF_FMT_sLUMINANCE_ALPHA16I,  ///< Luminance Alpha, 32 bits per element packed as (@c
                                    ///< AAAAAAAAAAAAAAAALLLLLLLLLLLLLLLL)
  CM_SURF_FMT_sRGBA8I,              ///< RGBA,  32 bits per element packed as (@c
                                    ///< AAAAAAAABBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_sRGBX8I,              ///< RGB,   32 bits per element packed as (@c
                                    ///< XXXXXXXXBBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_sALPHA8I,             ///< Alpha, 8 bits per element packed as (@c AAAAAAAA)
  CM_SURF_FMT_sINTENSITY8I,         ///< Intensity, 8 bits per element packed as (@c IIIIIIII)
  CM_SURF_FMT_sLUMINANCE8I,         ///< Luminance, 8 bits per element packed as (@c LLLLLLLL)
  CM_SURF_FMT_sLUMINANCE_ALPHA8I,   ///< Alpha, 8 bits per element packed as (@c AAAAAAAA)
  CM_SURF_FMT_sDXT6,                ///< compressed, CM_SURF_FMT_sDXT6
  CM_SURF_FMT_DXT6,                 ///< compressed, CM_SURF_FMT_DXT6
  CM_SURF_FMT_DXT7,                 ///< compressed, DXT7
  CM_SURF_FMT_LUMINANCE8_SNORM,   ///< Luminance,  8 bits per element packed as signed (@c LLLLLLLL)
  CM_SURF_FMT_LUMINANCE16_SNORM,  ///< Luminance, 16 bits per element packed as signed (@c
                                  ///< LLLLLLLLLLLLLLLL)
  CM_SURF_FMT_INTENSITY8_SNORM,   ///< Intensity,  8 bits per element packed as signed (@c IIIIIIII)
  CM_SURF_FMT_INTENSITY16_SNORM,  ///< Intensity, 16 bits per element packed as signed (@c
                                  ///< IIIIIIIIIIIIIIII)
  CM_SURF_FMT_ALPHA8_SNORM,       ///< Alpha,      8 bits per element packed as signed (@c AAAAAAAA)
  CM_SURF_FMT_ALPHA16_SNORM,      ///< Alpha,     16 bits per element packed as signed (@c
                                  ///< AAAAAAAAAAAAAAAA)
  CM_SURF_FMT_LUMINANCE_ALPHA8_SNORM,   ///< Luminance Alpha, 16 bits per element packed as signed
                                        ///< (@c AAAAAAAALLLLLLLL)
  CM_SURF_FMT_LUMINANCE_ALPHA16_SNORM,  ///< Luminance Alpha, 32 bits per element packed as signed
                                        ///< (@c AAAAAAAAAAAAAAAALLLLLLLLLLLLLLLL)
  CM_SURF_FMT_R8_SNORM,      ///< R,     8   bits per element packed as signed (@c RRRRRRRR)
  CM_SURF_FMT_R16_SNORM,     ///< R,    16   bits per element packed as signed (@c RRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RG8_SNORM,     ///< RG8,   16 bits per element packed as signed (@c RRRRRRRRGGGGGGGG)
  CM_SURF_FMT_RG16_SNORM,    ///< RG,    32 bits per element packed as signed (@c
                             ///< RRRRRRRRRRRRRRRRGGGGGGGGGGGGGGGG)
  CM_SURF_FMT_RGBX8_SNORM,   ///< RGB,   32 bits per element packed as signed (@c
                             ///< XXXXXXXXBBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_RGBX16_SNORM,  ///< RGB,   64 bits per element packed as signed (@c
                             ///< XXXXXXXXXXXXXXXXBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGBA8_SNORM,   ///< RGBA,  32 bits per element packed as signed (@c
                             ///< AAAAAAAABBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_RGBA16_SNORM,  ///< RGBA,  64 bits per element packed as signed (@c
                             ///< AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGB10_A2UI,    ///< RGBA,  32 bits per element packed as (@c
                             ///< AABBBBBBBBBBGGGGGGGGGGRRRRRRRRRR)
  CM_SURF_FMT_RGB32F,        ///< RGB, float, 96 bits per element packed as (@c
                       ///< BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGB32I,  ///< RGB, unnormalized int, 96 bits per element packed as (@c
                       ///< BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGB32UI,  ///< RGB, unnormalized uint, 96 bits per element packed as (@c
                        ///< BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR)
  CM_SURF_FMT_RGBX8_SRGB,             ///< RGB,   32 bits per element packed as (@c
                                      ///< XXXXXXXXBBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_RGBA8_SRGB,             ///< RGBA,  32 bits per element packed as (@c
                                      ///< AAAAAAAABBBBBBBBGGGGGGGGRRRRRRRR)
  CM_SURF_FMT_DXT1_SRGB,              ///< compressed, DXT1
  CM_SURF_FMT_DXT1A_SRGB,             ///<
  CM_SURF_FMT_DXT2_3_SRGB,            ///< compressed, DXT2_3
  CM_SURF_FMT_DXT4_5_SRGB,            ///< compressed, DXT4_5
  CM_SURF_FMT_DXT7_SRGB,              ///< compressed, DXT7
  CM_SURF_FMT_RGB8_ETC2,              ///< ETC2 compressed, RGB8 in 64 bits
  CM_SURF_FMT_SRGB8_ETC2,             ///< ETC2 compressed, SRGB8 in 64 bits
  CM_SURF_FMT_RGB8_PT_ALPHA1_ETC2,    ///< ETC2 compressed, RGB8 in 64 bits
  CM_SURF_FMT_SRGB8_PT_ALPHA1_ETC2,   ///< ETC2 compressed, sRGB8A1 in 64 bits
  CM_SURF_FMT_RGBA8_ETC2_EAC,         ///< ETC2 compressed, RGBA8 in 128 bits
  CM_SURF_FMT_SRGB8_ALPHA8_ETC2_EAC,  ///< ETC2 compressed, sRGBA8 in 128 bits
  CM_SURF_FMT_R11_EAC,                ///< EAC compressed, R11 in 64 bits
  CM_SURF_FMT_SIGNED_R11_EAC,         ///< EAC compressed, signed R11 in 64 bits
  CM_SURF_FMT_RG11_EAC,               ///< EAC compressed, RG11 in 128 bits
  CM_SURF_FMT_SIGNED_RG11_EAC,        ///< EAC compressed, signed RG11 in 128 bits

  CM_SURF_FMT_RGBA8_ASTC_4x4,    ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_5x4,    ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_5x5,    ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_6x5,    ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_6x6,    ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_8x5,    ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_8x6,    ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_8x8,    ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_10x5,   ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_10x6,   ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_10x8,   ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_10x10,  ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_12x10,  ///< ASTC compressed RGBA8 in 128 bits block
  CM_SURF_FMT_RGBA8_ASTC_12x12,  ///< ASTC compressed RGBA8 in 128 bits block

  CM_SURF_FMT_SRGBA8_ASTC_4x4,    ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_5x4,    ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_5x5,    ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_6x5,    ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_6x6,    ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_8x5,    ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_8x6,    ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_8x8,    ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_10x5,   ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_10x6,   ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_10x8,   ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_10x10,  ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_12x10,  ///< ASTC compressed SRGBA8 in 128 bits block
  CM_SURF_FMT_SRGBA8_ASTC_12x12,  ///< ASTC compressed SRGBA8 in 128 bits block

  CM_SURF_FMT_BGR10_A2UI,  ///< RGBA,  32 bits per element packed as (@c
                           ///< AARRRRRRRRRRGGGGGGGGGGBBBBBBBBBB)
  CM_SURF_FMT_A2_BGR10UI,  ///< RGBA,  32 bits per element packed as (@c
                           ///< RRRRRRRRRRGGGGGGGGGGBBBBBBBBBBAA)
  CM_SURF_FMT_A2_RGB10UI,  ///< RGBA,  32 bits per element packed as (@c
                           ///< BBBBBBBBBBGGGGGGGGGGRRRRRRRRRRAA)
  CM_SURF_FMT_B5_G6_R5UI,  ///< RGB,   16 bits per element packed as (@c BBBBBGGGGGGRRRRR)
  CM_SURF_FMT_R5_G6_B5UI,  ///< RGB,   16 bits per element packed as (@c RRRRRGGGGGGBBBBB)

  CM_SURF_FMT_DEPTH32F_X24_STEN8_UNCLAMPED,  ///< depth + stencil, 64 bits per element packed as (@c
                                             ///< XXXXXXXXXXXXXXXXXXXXXXXXSSSSSSSSDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD)
  CM_SURF_FMT_DEPTH32F_UNCLAMPED,  ///< depth, 32 bits per element packed as (@c
                                   ///< DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD)

  CM_SURF_FMT_L8_X16_A8_SRGB,  ///< Sluminance Alpha,  32 bits per element packed as (@c
                               ///< AAAAAAAAXXXXXXXXXXXXXXXXLLLLLLLL)
  CM_SURF_FMT_L8_X24_SRGB,     ///< Sluminance,        32 bits per element packed as (@c
                               ///< XXXXXXXXXXXXXXXXXXXXXXXXLLLLLLLL)

  CM_SURF_FMT_STENCIL8,  ///< stencil, 32 bits per element packed as (@c
                         ///< SSSSSSSSXXXXXXXXXXXXXXXXXXXXXXXX)


  // non-native surface formats after this line, will be ignored by HWL
  // all non-native surface formats should use the _NN suffix to distinguish
  // them from potential corresponding native formats added in the future
  CM_SURF_FMT_I420_NN,                              ///< 4:2:0 Planar Y-U-V format
  CM_SURF_FMT_YV12_NN,                              ///< 4:2:0 Planar Y-V-U format
  CM_SURF_FMT_NV12_NN,                              ///< 4:2:0 Semi-planar Y-UV format
  CM_SURF_FMT_NV21_NN,                              ///< 4:2:0 Semi-planar Y-VU format
  cmSurfFmt_FIRST = CM_SURF_FMT_LUMINANCE8,         ///< First surface format
  cmSurfFmt_LAST = CM_SURF_FMT_STENCIL8,            ///< Last native surface format
  cmSurfFmt_LAST_NON_NATIVE = CM_SURF_FMT_NV21_NN,  ///< Last non-native surface format
} cmSurfFmt;

typedef struct cmFormatXlateRec {
  cmSurfFmt raw_cmFormat;
  cl_channel_type image_channel_data_type;
  cl_channel_order image_channel_order;
} cmFormatXlateParams;

// relates full range of cm surface formats to those supported by CAL
static const cmFormatXlateParams cmFormatXlateTable[] = {
    {CM_SURF_FMT_LUMINANCE8, CL_UNORM_INT8, CL_LUMINANCE},
    {CM_SURF_FMT_LUMINANCE16, CL_UNORM_INT16, CL_LUMINANCE},
    {CM_SURF_FMT_LUMINANCE16F, CL_HALF_FLOAT, CL_LUMINANCE},
    {CM_SURF_FMT_LUMINANCE32F, CL_FLOAT, CL_LUMINANCE},
    {CM_SURF_FMT_INTENSITY8, CL_UNORM_INT8, CL_INTENSITY},
    {CM_SURF_FMT_INTENSITY16, CL_UNORM_INT16, CL_INTENSITY},
    {CM_SURF_FMT_INTENSITY16F, CL_HALF_FLOAT, CL_INTENSITY},
    {CM_SURF_FMT_INTENSITY32F, CL_FLOAT, CL_INTENSITY},
    {CM_SURF_FMT_ALPHA8, CL_UNSIGNED_INT8, CL_A},
    {CM_SURF_FMT_ALPHA16, CL_UNORM_INT16, CL_A},
    {CM_SURF_FMT_ALPHA16F, CL_HALF_FLOAT, CL_A},
    {CM_SURF_FMT_ALPHA32F, CL_FLOAT, CL_A},
    {CM_SURF_FMT_LUMINANCE8_ALPHA8, CL_UNSIGNED_INT8, CL_RG},
    {CM_SURF_FMT_LUMINANCE16_ALPHA16, CL_UNSIGNED_INT16, CL_RG},
    {CM_SURF_FMT_LUMINANCE16F_ALPHA16F, CL_HALF_FLOAT, CL_RG},
    {CM_SURF_FMT_LUMINANCE32F_ALPHA32F, CL_FLOAT, CL_RG},
    {CM_SURF_FMT_B2_G3_R3, 500, CL_R},
    {CM_SURF_FMT_B5_G6_R5, CL_UNSIGNED_INT16, CL_RGB},
    {CM_SURF_FMT_BGRX4, 500, CL_BGRA},
    {CM_SURF_FMT_BGR5_X1, CL_UNSIGNED_INT16, CL_RGB},
    {CM_SURF_FMT_BGRX8, CL_UNORM_INT8, CL_BGRA},
    {CM_SURF_FMT_BGR10_X2, CL_UNORM_INT_101010, CL_RGB},
    {CM_SURF_FMT_BGRX16, CL_UNORM_INT16, CL_BGRA},
    {CM_SURF_FMT_BGRX16F, CL_HALF_FLOAT, CL_BGRA},
    {CM_SURF_FMT_BGRX32F, CL_FLOAT, CL_BGRA},
    {CM_SURF_FMT_RGBX4, 500, CL_RGB},
    {CM_SURF_FMT_RGB5_X1, CL_UNORM_INT16, CL_BGRA},
    {CM_SURF_FMT_RGBX8, CL_UNORM_INT8, CL_RGBA},
    {CM_SURF_FMT_RGB10_X2, CL_UNORM_INT_101010, CL_RGBA},
    {CM_SURF_FMT_RGBX16, CL_UNORM_INT16, CL_RGBA},
    {CM_SURF_FMT_RGBX16F, CL_HALF_FLOAT, CL_RGBA},
    {CM_SURF_FMT_RGBX32F, CL_FLOAT, CL_RGBA},
    {CM_SURF_FMT_BGRA4, 500, CL_BGRA},
    {CM_SURF_FMT_BGR5_A1, CL_UNSIGNED_INT16, CL_BGRA},
    {CM_SURF_FMT_BGRA8, CL_UNORM_INT8, CL_BGRA},
    {CM_SURF_FMT_BGR10_A2, 500, CL_BGRA},
    {CM_SURF_FMT_BGRA16, CL_UNORM_INT16, CL_BGRA},
    {CM_SURF_FMT_BGRA16F, CL_UNORM_INT16, CL_BGRA},
    {CM_SURF_FMT_BGRA32F, CL_FLOAT, CL_BGRA},
    {CM_SURF_FMT_RGBA4, 500, CL_RGBA},
    {CM_SURF_FMT_RGB5_A1, CL_UNSIGNED_INT16, CL_RGBA},
    {CM_SURF_FMT_RGBA8, CL_UNORM_INT8, CL_RGBA},
    {CM_SURF_FMT_RGB10_A2, CL_UNORM_INT_101010, CL_RGB},
    {CM_SURF_FMT_RGBA16, CL_UNORM_INT16, CL_RGBA},
    {CM_SURF_FMT_RGBA16F, CL_HALF_FLOAT, CL_RGBA},
    {CM_SURF_FMT_RGBA32I, CL_UNSIGNED_INT32, CL_RGBA},
    {CM_SURF_FMT_RGBA32F, CL_FLOAT, CL_RGBA},
    {CM_SURF_FMT_DUDV8, CL_UNSIGNED_INT8, CL_RG},
    {CM_SURF_FMT_DXT1, 500, CL_R},
    {CM_SURF_FMT_DXT2_3, 500, CL_R},
    {CM_SURF_FMT_DXT4_5, 500, CL_R},
    {CM_SURF_FMT_ATI1N, 500, CL_R},
    {CM_SURF_FMT_ATI2N, 500, CL_R},
    {CM_SURF_FMT_DEPTH16, CL_UNORM_INT16, CL_DEPTH},
    {CM_SURF_FMT_DEPTH16F, CL_HALF_FLOAT, CL_DEPTH},
    {CM_SURF_FMT_DEPTH24_X8, 500, CL_DEPTH},
    {CM_SURF_FMT_DEPTH24F_X8, 500, CL_DEPTH},
    {CM_SURF_FMT_DEPTH24_STEN8, CL_UNORM_INT24, CL_DEPTH_STENCIL},
    {CM_SURF_FMT_DEPTH24F_STEN8, 500, CL_DEPTH_STENCIL},
    {CM_SURF_FMT_DEPTH32F_X24_STEN8, CL_FLOAT, CL_DEPTH_STENCIL},
    {CM_SURF_FMT_DEPTH32F, CL_FLOAT, CL_DEPTH},
    {CM_SURF_FMT_sR11_sG11_sB10, 500, CL_R},
    {CM_SURF_FMT_sU16, CL_SNORM_INT16, CL_R},
    {CM_SURF_FMT_sUV16, CL_SNORM_INT16, CL_RG},
    {CM_SURF_FMT_sUVWQ16, CL_SNORM_INT16, CL_RGBA},
    {CM_SURF_FMT_RG16, CL_UNORM_INT16, CL_RG},
    {CM_SURF_FMT_RG16F, CL_HALF_FLOAT, CL_RG},
    {CM_SURF_FMT_RG32F, CL_FLOAT, CL_RG},
    {CM_SURF_FMT_ABGR4, 500, CL_ARGB},
    {CM_SURF_FMT_A1_BGR5, CL_UNSIGNED_INT16, CL_ARGB},
    {CM_SURF_FMT_ABGR8, CL_UNORM_INT8, CL_ARGB},
    {CM_SURF_FMT_A2_BGR10, CL_UNORM_INT_101010, CL_RGB},
    {CM_SURF_FMT_ABGR16, CL_UNORM_INT16, CL_ARGB},
    {CM_SURF_FMT_ABGR16F, CL_HALF_FLOAT, CL_ARGB},
    {CM_SURF_FMT_ABGR32F, CL_FLOAT, CL_ARGB},
    {CM_SURF_FMT_DXT1A, 500, CL_R},
    {CM_SURF_FMT_sRGB10_A2, 500, CL_RGBA},
    {CM_SURF_FMT_sR8, CL_SNORM_INT8, CL_R},
    {CM_SURF_FMT_sRG8, CL_SNORM_INT8, CL_RG},
    {CM_SURF_FMT_sR32I, CL_SIGNED_INT32, CL_R},
    {CM_SURF_FMT_sRG32I, CL_SIGNED_INT32, CL_RG},
    {CM_SURF_FMT_sRGBA32I, CL_SIGNED_INT32, CL_RGBA},
    {CM_SURF_FMT_R32I, CL_UNSIGNED_INT32, CL_R},
    {CM_SURF_FMT_RG32I, CL_UNSIGNED_INT32, CL_RG},
    {CM_SURF_FMT_RG8, CL_UNORM_INT8, CL_RG},
    {CM_SURF_FMT_sRGBA8, CL_SNORM_INT8, CL_RGBA},
    {CM_SURF_FMT_R11F_G11F_B10F, 500, CL_RGBA},
    {CM_SURF_FMT_RGB9_E5, CL_UNORM_INT8, CL_ARGB},
    {CM_SURF_FMT_LUMINANCE_LATC1, 500, CL_RGBA},
    {CM_SURF_FMT_SIGNED_LUMINANCE_LATC1, 500, CL_RGBA},
    {CM_SURF_FMT_LUMINANCE_ALPHA_LATC2, 500, CL_RGBA},
    {CM_SURF_FMT_SIGNED_LUMINANCE_ALPHA_LATC2, 500, CL_RGBA},
    {CM_SURF_FMT_RED_RGTC1, 500, CL_RGBA},
    {CM_SURF_FMT_SIGNED_RED_RGTC1, 500, CL_RGBA},
    {CM_SURF_FMT_RED_GREEN_RGTC2, 500, CL_RGBA},
    {CM_SURF_FMT_SIGNED_RED_GREEN_RGTC2, 500, CL_RGBA},
    {CM_SURF_FMT_R8, CL_UNORM_INT8, CL_R},
    {CM_SURF_FMT_R16, CL_UNORM_INT16, CL_R},
    {CM_SURF_FMT_R16F, CL_HALF_FLOAT, CL_R},
    {CM_SURF_FMT_R32F, CL_FLOAT, CL_R},
    {CM_SURF_FMT_R8I, CL_UNSIGNED_INT8, CL_R},
    {CM_SURF_FMT_sR8I, CL_SIGNED_INT8, CL_R},
    {CM_SURF_FMT_RG8I, CL_UNSIGNED_INT8, CL_RG},
    {CM_SURF_FMT_sRG8I, CL_SIGNED_INT8, CL_RG},
    {CM_SURF_FMT_R16I, CL_UNSIGNED_INT16, CL_R},
    {CM_SURF_FMT_sR16I, CL_SIGNED_INT16, CL_R},
    {CM_SURF_FMT_RG16I, CL_UNSIGNED_INT16, CL_RG},
    {CM_SURF_FMT_sRG16I, CL_SIGNED_INT16, CL_RG},
    {CM_SURF_FMT_RGBA32UI, CL_UNSIGNED_INT32, CL_RGBA},
    {CM_SURF_FMT_RGBX32UI, CL_UNSIGNED_INT32, CL_RGBA},
    {CM_SURF_FMT_ALPHA32UI, CL_UNSIGNED_INT32, CL_R},
    {CM_SURF_FMT_INTENSITY32UI, CL_UNSIGNED_INT32, CL_R},
    {CM_SURF_FMT_LUMINANCE32UI, CL_UNSIGNED_INT32, CL_R},
    {CM_SURF_FMT_LUMINANCE_ALPHA32UI, CL_UNSIGNED_INT32, CL_RG},
    {CM_SURF_FMT_RGBA16UI, CL_UNSIGNED_INT16, CL_RGBA},
    {CM_SURF_FMT_RGBX16UI, CL_UNSIGNED_INT16, CL_RGBA},
    {CM_SURF_FMT_ALPHA16UI, CL_UNSIGNED_INT16, CL_R},
    {CM_SURF_FMT_INTENSITY16UI, CL_UNSIGNED_INT16, CL_R},
    {CM_SURF_FMT_LUMINANCE16UI, CL_UNSIGNED_INT16, CL_R},
    {CM_SURF_FMT_LUMINANCE_ALPHA16UI, CL_UNSIGNED_INT32, CL_RG},
    {CM_SURF_FMT_RGBA8UI, CL_UNSIGNED_INT8, CL_RGBA},
    {CM_SURF_FMT_RGBX8UI, CL_UNORM_INT8, CL_RGBA},
    {CM_SURF_FMT_ALPHA8UI, CL_UNSIGNED_INT8, CL_R},
    {CM_SURF_FMT_INTENSITY8UI, CL_UNSIGNED_INT8, CL_R},
    {CM_SURF_FMT_LUMINANCE8UI, CL_UNSIGNED_INT8, CL_R},
    {CM_SURF_FMT_LUMINANCE_ALPHA8UI, CL_UNSIGNED_INT8, CL_RG},
    {CM_SURF_FMT_sRGBX32I, CL_SIGNED_INT32, CL_RGBA},
    {CM_SURF_FMT_sALPHA32I, CL_SIGNED_INT32, CL_R},
    {CM_SURF_FMT_sINTENSITY32I, CL_SIGNED_INT32, CL_R},
    {CM_SURF_FMT_sLUMINANCE32I, CL_SIGNED_INT32, CL_R},
    {CM_SURF_FMT_sLUMINANCE_ALPHA32I, CL_SIGNED_INT32, CL_RG},
    {CM_SURF_FMT_sRGBA16I, CL_SIGNED_INT16, CL_RGBA},
    {CM_SURF_FMT_sRGBX16I, CL_SIGNED_INT16, CL_RGBA},
    {CM_SURF_FMT_sALPHA16I, CL_SIGNED_INT16, CL_R},
    {CM_SURF_FMT_sINTENSITY16I, CL_SIGNED_INT16, CL_R},
    {CM_SURF_FMT_sLUMINANCE16I, CL_SIGNED_INT16, CL_R},
    {CM_SURF_FMT_sLUMINANCE_ALPHA16I, CL_SIGNED_INT16, CL_RG},
    {CM_SURF_FMT_sRGBA8I, CL_SIGNED_INT8, CL_RGBA},
    {CM_SURF_FMT_sRGBX8I, CL_SIGNED_INT8, CL_RGBA},
    {CM_SURF_FMT_sALPHA8I, CL_SIGNED_INT8, CL_R},
    {CM_SURF_FMT_sINTENSITY8I, CL_SIGNED_INT8, CL_R},
    {CM_SURF_FMT_sLUMINANCE8I, CL_SIGNED_INT8, CL_R},
    {CM_SURF_FMT_sLUMINANCE_ALPHA8I, CM_SURF_FMT_sRG8I, CL_RG},
    {CM_SURF_FMT_sDXT6, 500, CL_R},
    {CM_SURF_FMT_DXT6, 500, CL_R},
    {CM_SURF_FMT_DXT7, 500, CL_R},
    {CM_SURF_FMT_LUMINANCE8_SNORM, CL_SNORM_INT8, CL_R},
    {CM_SURF_FMT_LUMINANCE16_SNORM, CL_SNORM_INT16, CL_R},
    {CM_SURF_FMT_INTENSITY8_SNORM, CL_SNORM_INT8, CL_R},
    {CM_SURF_FMT_INTENSITY16_SNORM, CL_SNORM_INT16, CL_R},
    {CM_SURF_FMT_ALPHA8_SNORM, CL_SNORM_INT8, CL_R},
    {CM_SURF_FMT_ALPHA16_SNORM, CL_SNORM_INT16, CL_R},
    {CM_SURF_FMT_LUMINANCE_ALPHA8_SNORM, CL_SNORM_INT8, CL_RG},
    {CM_SURF_FMT_LUMINANCE_ALPHA16_SNORM, CL_SNORM_INT16, CL_RG},
    {CM_SURF_FMT_R8_SNORM, CL_SNORM_INT8, CL_R},
    {CM_SURF_FMT_R16_SNORM, CL_SNORM_INT16, CL_R},
    {CM_SURF_FMT_RG8_SNORM, CL_SNORM_INT8, CL_RG},
    {CM_SURF_FMT_RG16_SNORM, CL_SNORM_INT16, CL_RG},
    {CM_SURF_FMT_RGBX8_SNORM, CL_SNORM_INT8, CL_RGBA},
    {CM_SURF_FMT_RGBX16_SNORM, CL_SNORM_INT16, CL_RGBA},
    {CM_SURF_FMT_RGBA8_SNORM, CL_SNORM_INT8, CL_RGBA},
    {CM_SURF_FMT_RGBA16_SNORM, CL_SNORM_INT16, CL_RGBA},
    {CM_SURF_FMT_RGB10_A2UI, 500, CL_RGBA},
    {CM_SURF_FMT_RGB32F, 500, CL_RGBA},
    {CM_SURF_FMT_RGB32I, 500, CL_RGBA},
    {CM_SURF_FMT_RGB32UI, 500, CL_RGBA},
    {CM_SURF_FMT_RGBX8_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_DXT1_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_DXT1A_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_DXT2_3_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_DXT4_5_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_DXT7_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_RGB8_ETC2, 500, CL_RGB},
    {CM_SURF_FMT_SRGB8_ETC2, 500, CL_RGB},
    {CM_SURF_FMT_RGB8_PT_ALPHA1_ETC2, 500, CL_RGBA},
    {CM_SURF_FMT_SRGB8_PT_ALPHA1_ETC2, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ETC2_EAC, 500, CL_RGBA},
    {CM_SURF_FMT_SRGB8_ALPHA8_ETC2_EAC, 500, CL_RGBA},
    {CM_SURF_FMT_R11_EAC, 500, CL_R},
    {CM_SURF_FMT_SIGNED_R11_EAC, 500, CL_R},
    {CM_SURF_FMT_RG11_EAC, 500, CL_RG},
    {CM_SURF_FMT_SIGNED_RG11_EAC, 500, CL_RG},
    {CM_SURF_FMT_RGBA8_ASTC_4x4, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_5x4, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_5x5, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_6x5, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_6x6, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_8x5, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_8x6, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_8x8, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_10x5, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_10x6, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_10x8, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_10x10, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_12x10, 500, CL_RGBA},
    {CM_SURF_FMT_RGBA8_ASTC_12x12, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_4x4, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_5x4, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_5x5, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_6x5, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_6x6, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_8x5, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_8x6, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_8x8, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_10x5, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_10x6, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_10x8, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_10x10, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_12x10, 500, CL_RGBA},
    {CM_SURF_FMT_SRGBA8_ASTC_12x12, 500, CL_RGBA},
    {CM_SURF_FMT_BGR10_A2UI, 500, CL_BGRA},
    {CM_SURF_FMT_A2_BGR10UI, 500, CL_ARGB},
    {CM_SURF_FMT_A2_RGB10UI, 500, CL_ABGR},
    {CM_SURF_FMT_B5_G6_R5UI, 500, CL_BGRA},
    {CM_SURF_FMT_R5_G6_B5UI, 500, CL_RGBA},
    {CM_SURF_FMT_DEPTH32F_X24_STEN8_UNCLAMPED, CL_UNSIGNED_INT32, CL_R},
    {CM_SURF_FMT_DEPTH32F_UNCLAMPED, CL_FLOAT, CL_R},
    {CM_SURF_FMT_L8_X16_A8_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_L8_X24_SRGB, 500, CL_RGBA},
    {CM_SURF_FMT_STENCIL8, CL_UNSIGNED_INT8, CL_R},
};

bool Device::initGLInteropPrivateExt(void* GLplatformContext, void* GLdeviceContext) const {
#ifdef ATI_OS_LINUX
  GLXContext ctx = (GLXContext)GLplatformContext;
  void* pModule = dlopen("libGL.so.1", RTLD_NOW);

  if (nullptr == pModule) {
    return false;
  }
  pfnGlxGetProcAddress = (PFNGlxGetProcAddress)dlsym(pModule, "glXGetProcAddress");
  if (nullptr == pfnGlxGetProcAddress) {
    return false;
  }

  pfnMesaGLInteropGLXQueryDeviceInfo =
      (PFNMesaGLInteropGLXQueryDeviceInfo)dlsym(pModule, "MesaGLInteropGLXQueryDeviceInfo");
  if (nullptr == pfnMesaGLInteropGLXQueryDeviceInfo) {
    return false;
  }

  if (!glXBeginCLInteropAMD || !glXEndCLInteropAMD || !glXResourceAttachAMD ||
      !glXResourceDetachAMD || !glXGetContextMVPUInfoAMD) {
    glXBeginCLInteropAMD = (PFNGLXBEGINCLINTEROPAMD)pfnGlxGetProcAddress(
        (const GLubyte*)"glXBeginCLInteroperabilityAMD");
    glXEndCLInteropAMD =
        (PFNGLXENDCLINTEROPAMD)pfnGlxGetProcAddress((const GLubyte*)"glXEndCLInteroperabilityAMD");
    glXResourceAttachAMD =
        (PFNGLXRESOURCEATTACHAMD)pfnGlxGetProcAddress((const GLubyte*)"glXResourceAttachAMD");
    glxResourceAcquireAMD =
        (PFNGLXRESOURCEDETACHAMD)pfnGlxGetProcAddress((const GLubyte*)"glXResourceAcquireAMD");
    glxResourceReleaseAMD =
        (PFNGLXRESOURCEDETACHAMD)pfnGlxGetProcAddress((const GLubyte*)"glXResourceReleaseAMD");
    glXResourceDetachAMD =
        (PFNGLXRESOURCEDETACHAMD)pfnGlxGetProcAddress((const GLubyte*)"glXResourceDetachAMD");
    glXGetContextMVPUInfoAMD = (PFNGLXGETCONTEXTMVPUINFOAMD)pfnGlxGetProcAddress(
        (const GLubyte*)"glXGetContextMVPUInfoAMD");
  }

  if (!glXBeginCLInteropAMD || !glXEndCLInteropAMD || !glXResourceAttachAMD ||
      !glXResourceDetachAMD || !glXGetContextMVPUInfoAMD) {
    return false;
  }
#else
  if (!gGlFuncInit) {
    HMODULE h = static_cast<HMODULE>(amd::Os::loadLibrary("opengl32.dll"));
    if (nullptr != h) {
      pfnWglGetProcAddress =
          reinterpret_cast<PFNWGLGETPROCADDRESS>(GetProcAddress(h, "wglGetProcAddress"));
      pfnWglGetCurrentContext =
          reinterpret_cast<PFNWGLGETCURRENTCONTEXT>(GetProcAddress(h, "wglGetCurrentContext"));
      pfnWglCreateContext =
          reinterpret_cast<PFNWGLCREATECONTEXT>(GetProcAddress(h, "wglCreateContext"));
      pfnWglDeleteContext =
          reinterpret_cast<PFNWGLDELETECONTEXT>(GetProcAddress(h, "wglDeleteContext"));
      pfnWglMakeCurrent = reinterpret_cast<PFNWGLMAKECURRENT>(GetProcAddress(h, "wglMakeCurrent"));
      if (!pfnWglGetProcAddress || !pfnWglGetCurrentContext || !pfnWglCreateContext ||
          !pfnWglDeleteContext || !pfnWglMakeCurrent) {
        LogError("Couldn't obtain WGL context API");
        return false;
      }
    }
    HGLRC fakeRC = nullptr;
    // @note: For unclear reason runtime can't get AMD API entry points without a context...
    if (!pfnWglGetCurrentContext()) {
      fakeRC = pfnWglCreateContext((HDC)GLdeviceContext);
      pfnWglMakeCurrent((HDC)GLdeviceContext, fakeRC);
    }
    wglBeginCLInteropAMD =
        (PFNWGLBEGINCLINTEROPAMD)pfnWglGetProcAddress("wglBeginCLInteroperabilityAMD");
    wglEndCLInteropAMD = (PFNWGLENDCLINTEROPAMD)pfnWglGetProcAddress("wglEndCLInteroperabilityAMD");
    wglResourceAttachAMD = (PFNWGLRESOURCEATTACHAMD)pfnWglGetProcAddress("wglResourceAttachAMD");
    wglResourceAcquireAMD = (PFNWGLRESOURCEDETACHAMD)pfnWglGetProcAddress("wglResourceAcquireAMD");
    wglResourceReleaseAMD = (PFNWGLRESOURCEDETACHAMD)pfnWglGetProcAddress("wglResourceReleaseAMD");
    wglResourceDetachAMD = (PFNWGLRESOURCEDETACHAMD)pfnWglGetProcAddress("wglResourceDetachAMD");
    wglGetContextGPUInfoAMD =
        (PFNWGLGETCONTEXTGPUINFOAMD)pfnWglGetProcAddress("wglGetContextGPUInfoAMD");
    if (fakeRC) {
      pfnWglMakeCurrent(nullptr, nullptr);
      pfnWglDeleteContext(fakeRC);
    }
  }
  if (!wglBeginCLInteropAMD || !wglEndCLInteropAMD || !wglResourceAttachAMD ||
      !wglResourceDetachAMD || !wglGetContextGPUInfoAMD) {
    LogError("Couldn't obtain WGL AMD API");
    return false;
  }
#endif
  gGlFuncInit = true;
  return true;
}

bool Device::glCanInterop(void* GLplatformContext, void* GLdeviceContext) const {
  bool canInteroperate = false;

#ifdef ATI_OS_WIN
  LUID glAdapterLuid = {0, 0};
  UINT glChainBitMask = 0;
  HGLRC hRC = (HGLRC)GLplatformContext;

  // get GL context's LUID and chainBitMask from UGL
  if (wglGetContextGPUInfoAMD(hRC, &glAdapterLuid, &glChainBitMask)) {
    // match the adapter
    canInteroperate = (properties().osProperties.luidHighPart == glAdapterLuid.HighPart) &&
                      (properties().osProperties.luidLowPart == glAdapterLuid.LowPart) &&
                      ((1 << properties().gpuIndex) == glChainBitMask);
  }
#else
  GLuint glDeviceId = 0;
  GLuint glChainMask = 0;
  GLXContext ctx = static_cast<GLXContext>(GLplatformContext);
  Display* disp = static_cast<Display*>(GLdeviceContext);


  if (glXGetContextMVPUInfoAMD(ctx, &glDeviceId, &glChainMask)) {
    mesa_glinterop_device_info info = {};
    if (pfnMesaGLInteropGLXQueryDeviceInfo(disp, ctx, &info) == 0) {
      // match the adapter
      canInteroperate = (properties().pciProperties.busNumber == info.pci_bus) &&
                        (properties().pciProperties.deviceNumber == info.pci_device) &&
                        (properties().pciProperties.functionNumber == info.pci_function) &&
                        (static_cast<GLuint>(1 << properties().gpuIndex) == glChainMask);
    }
  }
#endif
  return canInteroperate;
}

bool Device::glAssociate(void* GLplatformContext, void* GLdeviceContext) const {
  // initialize pointers to the gl extension that supports interoperability
  if (!initGLInteropPrivateExt(GLplatformContext, GLdeviceContext) ||
      !glCanInterop(GLplatformContext, GLdeviceContext)) {
    return false;
  }
#ifdef ATI_OS_LINUX
  GLXContext ctx = (GLXContext)GLplatformContext;
  return (glXBeginCLInteropAMD(ctx, 0)) ? true : false;
#else
  HGLRC hRC = (HGLRC)GLplatformContext;
  return (wglBeginCLInteropAMD(hRC, 0)) ? true : false;
#endif
}

bool Device::glDissociate(void* GLplatformContext, void* GLdeviceContext) const {
#ifdef ATI_OS_LINUX
  GLXContext ctx = (GLXContext)GLplatformContext;
  if (glXEndCLInteropAMD == nullptr) {
    return false;
  } else {
    return (glXEndCLInteropAMD(ctx, 0)) ? true : false;
  }
#else
  HGLRC hRC = (HGLRC)GLplatformContext;
  if (wglEndCLInteropAMD == nullptr) {
    return false;
  } else {
    return (wglEndCLInteropAMD(hRC, 0)) ? true : false;
  }
#endif
}

bool Device::resGLAssociate(void* GLContext, uint name, uint type, Pal::OsExternalHandle* handle,
                            void** mbResHandle, size_t* offset, cl_image_format& newClFormat
#ifdef ATI_OS_WIN
                            ,
                            Pal::DoppDesktopInfo& doppDesktopInfo
#endif
) const {
  amd::ScopedLock lk(lockPAL());

  GLResource hRes = {};
  GLResourceData hData = {};

  bool status = false;

  hRes.type = type;
  hRes.name = name;

  hData.version = GL_RESOURCE_DATA_VERSION;
#ifdef ATI_OS_LINUX
  GLXContext ctx = (GLXContext)GLContext;
  if (glXResourceAttachAMD(ctx, &hRes, &hData)) {
    status = true;
  }
#else
  HGLRC hRC = (HGLRC)GLContext;
  if (wglResourceAttachAMD(hRC, &hRes, &hData)) {
    status = true;
  }
#endif

  if (!status) {
    return false;
  }

  *mbResHandle = reinterpret_cast<void*>(hData.mbResHandle);
  *offset = static_cast<size_t>(hData.offset);
#ifdef ATI_OS_WIN
  *handle = reinterpret_cast<Pal::OsExternalHandle>(hData.handle);
  if (hData.isDoppDesktopTexture) {
    doppDesktopInfo.gpuVirtAddr = hData.cardAddr;
    doppDesktopInfo.vidPnSourceId = hData.vidpnSourceId;
  } else {
    doppDesktopInfo.gpuVirtAddr = 0;
    doppDesktopInfo.vidPnSourceId = 0;
  }
#else
  *handle = static_cast<Pal::OsExternalHandle>(hData.sharedBufferID);
#endif

  // OCL supports only a limited number of cm_surf formats, so we
  // have to translate incoming cm_surf formats
  uint index = hData.format - (uint)CM_SURF_FMT_LUMINANCE8;
  if (index >= sizeof(cmFormatXlateTable) / sizeof(cmFormatXlateParams)) {
    LogError("\nInvalid GL surface reported in hData\n");
    return status;
  }
  assert(static_cast<cmSurfFmt>(hData.format) == cmFormatXlateTable[index].raw_cmFormat);
  cl_channel_type imageDataType;
  imageDataType = cmFormatXlateTable[index].image_channel_data_type;
  if (imageDataType == 500) {
    LogError("\nGL surface is not supported by OCL\n");
    return status;
  }

  newClFormat.image_channel_data_type = cmFormatXlateTable[index].image_channel_data_type;
  newClFormat.image_channel_order = cmFormatXlateTable[index].image_channel_order;

  return status;
}

bool Device::resGLAcquire(void* GLplatformContext, void* mbResHandle, uint type) const {
  amd::ScopedLock lk(lockPAL());

  GLResource hRes = {};
  hRes.mbResHandle = (GLuintp)mbResHandle;
  hRes.type = type;

#ifdef ATI_OS_LINUX
  GLXContext ctx = (GLXContext)GLplatformContext;
  return (glxResourceAcquireAMD(ctx, &hRes)) ? true : false;
#else
  HGLRC hRC = pfnWglGetCurrentContext();
  //! @todo A temporary workaround for MT issue in conformance fence_sync
  if (0 == hRC) {
    return true;
  }
  return (wglResourceAcquireAMD(hRC, &hRes)) ? true : false;
#endif
}

bool Device::resGLRelease(void* GLplatformContext, void* mbResHandle, uint type) const {
  amd::ScopedLock lk(lockPAL());

  GLResource hRes = {};
  hRes.mbResHandle = (GLuintp)mbResHandle;
  hRes.type = type;
#ifdef ATI_OS_LINUX
  // TODO : make sure the application GL context is current. if not no
  // point calling into the GL RT.
  GLXContext ctx = (GLXContext)GLplatformContext;
  return (glxResourceReleaseAMD(ctx, &hRes)) ? true : false;
#else
  // Make the call into the GL driver only if the application GL context is current
  HGLRC hRC = pfnWglGetCurrentContext();
  //! @todo A temporary workaround for MT issue in conformance fence_sync
  if (0 == hRC) {
    return true;
  }
  return (wglResourceReleaseAMD(hRC, &hRes)) ? true : false;
#endif
}

bool Device::resGLFree(void* GLplatformContext, void* mbResHandle, uint type) const {
  amd::ScopedLock lk(lockPAL());

  GLResource hRes = {};
  hRes.mbResHandle = (GLuintp)mbResHandle;
  hRes.type = type;
#ifdef ATI_OS_LINUX
  GLXContext ctx = (GLXContext)GLplatformContext;
  return (glXResourceDetachAMD(ctx, &hRes)) ? true : false;
#else
  HGLRC hRC = (HGLRC)GLplatformContext;
  return (wglResourceDetachAMD(hRC, &hRes)) ? true : false;
#endif
}

}  // namespace amd::pal
