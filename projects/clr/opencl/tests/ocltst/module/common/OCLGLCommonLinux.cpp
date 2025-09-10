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
  static Display* display;
  static XVisualInfo* vInfo;
  static int referenceCount;
  GLXContext context;
  Window window;
  Colormap cmap;
};

Display* OCLGLHandle_::display = NULL;
XVisualInfo* OCLGLHandle_::vInfo = NULL;
int OCLGLHandle_::referenceCount = 0;

OCLGLCommon::OCLGLCommon() {
  hGL_ = new OCLGLHandle_;

  hGL_->context = NULL;
  hGL_->window = 0;
  hGL_->cmap = 0;
}

OCLGLCommon::~OCLGLCommon() { destroyGLContext(hGL_); }

void OCLGLCommon::destroyGLContext(OCLGLHandle& hGL) {
  deleteGLContext(hGL);
  delete hGL;
  hGL = NULL;
}

void OCLGLCommon::deleteGLContext(OCLGLHandle& hGL) {
  if (hGL->display != NULL) {
    glXMakeCurrent(hGL->display, None, NULL);
    if (hGL->cmap) {
      XFreeColormap(hGL->display, hGL->cmap);
      hGL->cmap = 0;
    }
    if (hGL->window) {
      XDestroyWindow(hGL->display, hGL->window);
      hGL->window = 0;
    }
    if (hGL->context) {
      glXDestroyContext(hGL->display, hGL->context);
      hGL->context = NULL;
    }

    hGL->referenceCount--;
    if (hGL->referenceCount == 0) {
      XCloseDisplay(hGL->display);
      hGL->display = NULL;

      XFree(hGL->vInfo);
      hGL->vInfo = NULL;
    }
  }
}

bool OCLGLCommon::createGLContext(OCLGLHandle& hGL) {
  hGL = new OCLGLHandle_;
  return initializeGLContext(hGL);
}

bool OCLGLCommon::initializeGLContext(OCLGLHandle& hGL) {
  if (hGL->display == NULL) {
    hGL->display = XOpenDisplay(NULL);
    if (hGL->display == NULL) {
      printf("XOpenDisplay() failed\n");
      return false;
    }
  }
  if (hGL->vInfo == NULL) {
    int dblBuf[] = {GLX_RGBA, GLX_RED_SIZE,   1,  GLX_GREEN_SIZE,   1,   GLX_BLUE_SIZE,
                    1,        GLX_DEPTH_SIZE, 12, GLX_DOUBLEBUFFER, None};

    hGL->vInfo = glXChooseVisual(hGL->display, DefaultScreen(hGL->display), dblBuf);
    if (hGL->vInfo == NULL) {
      printf("glXChooseVisual() failed\n");
      return false;
    }
  }
  hGL->referenceCount++;

  hGL->context = glXCreateContext(hGL->display, hGL->vInfo, None, True);
  if (hGL->context == NULL) {
    printf("glXCreateContext() failed\n");
    return false;
  }

  XSetWindowAttributes swa = {0};
  hGL->cmap = XCreateColormap(hGL->display, RootWindow(hGL->display, hGL->vInfo->screen),
                              hGL->vInfo->visual, AllocNone);
  swa.colormap = hGL->cmap;
  hGL->window = XCreateWindow(hGL->display, RootWindow(hGL->display, hGL->vInfo->screen), 0, 0, 640,
                              480, 0, hGL->vInfo->depth, InputOutput, hGL->vInfo->visual,
                              CWBorderPixel | CWColormap | CWEventMask, &swa);

  Bool glErr = glXMakeCurrent(hGL->display, hGL->window, hGL->context);
  if (False == glErr) {
    return false;
  }

  if (!checkAssociationDeviceWithGLContext(hGL)) {
    deleteGLContext(hGL);
    return false;
  }
  return true;
}

bool OCLGLCommon::checkAssociationDeviceWithGLContext(OCLGLHandle& hGL) {
  bool ret = false;
  size_t devicesSize = 0;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_,
                                        CL_GL_CONTEXT_KHR,
                                        (cl_context_properties)hGL->context,
                                        CL_GLX_DISPLAY_KHR,
                                        (cl_context_properties)hGL->display,
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

  // If the current device id does not work for CL GL interop,
  // fall back to other devices.
  if (!ret) {
    for (unsigned int i = 0; i < numDevices; i++) {
      for (unsigned int j = 0; j < deviceCount_; j++) {
        if (interopDevices[i] == devices_[j]) {
          // Found an OpenCL device work with current GL context.
          _deviceId = j;
          ret = true;
          break;
        }
      }
    }
  }

  free(interopDevices);
  return ret;
}

void OCLGLCommon::createCLContextFromGLContext(OCLGLHandle& hGL) {
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_,
                                        CL_GL_CONTEXT_KHR,
                                        (cl_context_properties)hGL->context,
                                        CL_GLX_DISPLAY_KHR,
                                        (cl_context_properties)hGL->display,
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

  // GLEW versions 1.13.0 and earlier do not fetch all GL function pointers
  // without glewExperimental set.
  glewExperimental = GL_TRUE;
  GLenum glErr = glewInit();
  CHECK_RESULT((glErr != GLEW_OK), "glewInit() failed: %s", glewGetErrorString(glErr));
}

void OCLGLCommon::makeCurrent(OCLGLHandle hGL) {
  if (hGL == NULL) {
    if (hGL_ != NULL) {
      glXMakeCurrent(hGL_->display, None, NULL);
    }
  } else {
    bool ret = glXMakeCurrent(hGL->display, hGL->window, hGL->context);
    assert(ret && "glXMakeCurrent failed!");
  }
}

void OCLGLCommon::getCLContextPropertiesFromGLContext(const OCLGLHandle hGL,
                                                      cl_context_properties properties[7]) {
  if (!properties) return;

  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties)platform_;
  properties[2] = CL_GL_CONTEXT_KHR;
  properties[3] = (cl_context_properties)hGL->context;
  properties[4] = CL_GLX_DISPLAY_KHR;
  properties[5] = (cl_context_properties)hGL->display;
  properties[6] = 0;
}
