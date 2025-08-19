/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include "OCLGLCommon.h"

struct OCLGLHandle_ {
  HDC hdc;
  HGLRC hglrc;
};

OCLGLCommon::OCLGLCommon() {
  hGL_ = new OCLGLHandle_;

  hGL_->hdc = NULL;
  hGL_->hglrc = NULL;
}

OCLGLCommon::~OCLGLCommon() { destroyGLContext(hGL_); }

void OCLGLCommon::destroyGLContext(OCLGLHandle& hGL) {
  deleteGLContext(hGL);
  delete hGL;
  hGL = NULL;
}

void OCLGLCommon::deleteGLContext(OCLGLHandle& hGL) {
  wglMakeCurrent(NULL, NULL);
  if (hGL->hglrc) {
    wglDeleteContext(hGL->hglrc);
    hGL->hglrc = NULL;
  }
  if (hGL->hdc) {
    DeleteDC(hGL->hdc);
    hGL->hdc = NULL;
  }
}

bool OCLGLCommon::createGLContext(OCLGLHandle& hGL) {
  hGL = new OCLGLHandle_;
  return initializeGLContext(hGL);
}

bool OCLGLCommon::initializeGLContext(OCLGLHandle& hGL) {
  BOOL glErr = FALSE;
  DISPLAY_DEVICE dispDevice;
  DWORD deviceNum;
  int pfmt;
  PIXELFORMATDESCRIPTOR pfd;
  pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
  pfd.nVersion = 1;
  pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
  pfd.iPixelType = PFD_TYPE_RGBA;
  pfd.cColorBits = 24;
  pfd.cRedBits = 8;
  pfd.cRedShift = 0;
  pfd.cGreenBits = 8;
  pfd.cGreenShift = 0;
  pfd.cBlueBits = 8;
  pfd.cBlueShift = 0;
  pfd.cAlphaBits = 8;
  pfd.cAlphaShift = 0;
  pfd.cAccumBits = 0;
  pfd.cAccumRedBits = 0;
  pfd.cAccumGreenBits = 0;
  pfd.cAccumBlueBits = 0;
  pfd.cAccumAlphaBits = 0;
  pfd.cDepthBits = 24;
  pfd.cStencilBits = 8;
  pfd.cAuxBuffers = 0;
  pfd.iLayerType = PFD_MAIN_PLANE;
  pfd.bReserved = 0;
  pfd.dwLayerMask = 0;
  pfd.dwVisibleMask = 0;
  pfd.dwDamageMask = 0;

  dispDevice.cb = sizeof(DISPLAY_DEVICE);
  for (deviceNum = 0; EnumDisplayDevices(NULL, deviceNum, &dispDevice, 0); deviceNum++) {
    if (dispDevice.StateFlags & DISPLAY_DEVICE_MIRRORING_DRIVER) {
      continue;
    }

    hGL->hdc = CreateDC(NULL, dispDevice.DeviceName, NULL, NULL);
    if (!hGL->hdc) {
      continue;
    }

    pfmt = ChoosePixelFormat(hGL->hdc, &pfd);
    if (pfmt == 0) {
      printf("Failed choosing the requested PixelFormat.\n");
      return false;
    }

    glErr = SetPixelFormat(hGL->hdc, pfmt, &pfd);
    if (glErr == FALSE) {
      printf("Failed to set the requested PixelFormat.\n");
      return false;
    }

    hGL->hglrc = wglCreateContext(hGL->hdc);
    if (NULL == hGL->hglrc) {
      printf("wglCreateContext() failed\n");
      return false;
    }

    glErr = wglMakeCurrent(hGL->hdc, hGL->hglrc);
    if (FALSE == glErr) {
      printf("wglMakeCurrent() failed\n");
      return false;
    }

    if (!checkAssociationDeviceWithGLContext(hGL)) {
      deleteGLContext(hGL);
      return false;
    }

    return true;
  }  //  for (deviceNum = 0; EnumDisplayDevices(NULL, deviceNum, &dispDevice,
     //  0); deviceNum++) {

  return false;
}

bool OCLGLCommon::checkAssociationDeviceWithGLContext(OCLGLHandle& hGL) {
  bool ret = false;
  size_t devicesSize = 0;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_,
                                        CL_GL_CONTEXT_KHR,
                                        (cl_context_properties)hGL->hglrc,
                                        CL_WGL_HDC_KHR,
                                        (cl_context_properties)hGL->hdc,
                                        0};

  error_ = _wrapper->clGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR, 0, NULL,
                                           &devicesSize);
  if (error_ != CL_SUCCESS) {
    printf("clGetGLContextInfoKHR failed (%d)\n", error_);
    return false;
  }

  cl_uint numDevices = (cl_uint)devicesSize / sizeof(cl_device_id);
  cl_device_id* interopDevices = (cl_device_id*)malloc(devicesSize);

  error_ = _wrapper->clGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR, devicesSize,
                                           interopDevices, NULL);
  if (error_ != CL_SUCCESS) {
    printf("clGetGLContextInfoKHR failed (%d)\n", error_);
    free(interopDevices);
    return false;
  }

  // Check that current device can be associated with OpenGL context
  for (unsigned int i = 0; i < numDevices; i++) {
    if (interopDevices[i] == devices_[_deviceId]) {
      ret = true;
      break;
    }
  }

  free(interopDevices);
  return ret;
}

void OCLGLCommon::createCLContextFromGLContext(OCLGLHandle& hGL) {
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_,
                                        CL_GL_CONTEXT_KHR,
                                        (cl_context_properties)hGL->hglrc,
                                        CL_WGL_HDC_KHR,
                                        (cl_context_properties)hGL->hdc,
                                        0};

  // Release current command queue
  if (cmdQueues_[_deviceId]) {
    error_ = _wrapper->clReleaseCommandQueue(cmdQueues_[_deviceId]);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseCommandQueue() failed");
  }

  // Release current context
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN((error_ != CL_SUCCESS), "clReleaseContext() failed");
  }

  // Create new CL context from GL context
  context_ = clCreateContext(properties, 1, &devices_[_deviceId], NULL, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateContext() failed (%d)", error_);

  // Create command queue for new context
  cmdQueues_[_deviceId] = _wrapper->clCreateCommandQueue(context_, devices_[_deviceId], 0, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed (%d)", error_);

  GLenum glErr = glewInit();
  CHECK_RESULT((glErr != GLEW_OK), "glewInit() failed");
}

void OCLGLCommon::makeCurrent(OCLGLHandle hGL) {
  if (hGL == NULL) {
    wglMakeCurrent(NULL, NULL);
  } else {
    wglMakeCurrent(hGL->hdc, hGL->hglrc);
  }
}

void OCLGLCommon::getCLContextPropertiesFromGLContext(const OCLGLHandle hGL,
                                                      cl_context_properties properties[7]) {
  if (!properties) return;

  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties)platform_;
  properties[2] = CL_GL_CONTEXT_KHR;
  properties[3] = (cl_context_properties)hGL->hglrc;
  properties[4] = CL_WGL_HDC_KHR;
  properties[5] = (cl_context_properties)hGL->hdc;
  properties[6] = 0;
}
