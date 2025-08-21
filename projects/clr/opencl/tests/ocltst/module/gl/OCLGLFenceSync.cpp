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

#include "OCLGLFenceSync.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "Timer.h"
#ifndef WIN_OS
#include <GL/glx.h>
#endif

const static char* strKernel =
    "__kernel void glmulticontext_test( __global uint4 *source, __global uint4 "
    "*dest)   \n"
    "{                                                                         "
    "         \n"
    "    int  tid = get_global_id(0);                                          "
    "         \n"
    "    dest[ tid ] = source [ tid ] + (uint4)(1);                            "
    "         \n"
    "}                                                                         "
    "         \n";

OCLGLFenceSync::OCLGLFenceSync() {
  memset(contextData_, 0, sizeof(contextData_));
  _numSubTests = 2;
}

OCLGLFenceSync::~OCLGLFenceSync() {}

#ifdef WIN_OS
typedef GLsync(__stdcall* glFenceSyncPtr)(GLenum condition, GLbitfield flags);
typedef bool(__stdcall* glIsSyncPtr)(GLsync sync);
typedef void(__stdcall* glDeleteSyncPtr)(GLsync sync);
typedef GLenum(__stdcall* glClientWaitSyncPtr)(GLsync sync, GLbitfield flags, GLuint64 timeout);
typedef void(__stdcall* glWaitSyncPtr)(GLsync sync, GLbitfield flags, GLuint64 timeout);
typedef void(__stdcall* glGetInteger64vPtr)(GLenum pname, GLint64* params);
typedef void(__stdcall* glGetSyncivPtr)(GLsync sync, GLenum pname, GLsizei bufSize, GLsizei* length,
                                        GLint* values);
#else
typedef GLsync (*glFenceSyncPtr)(GLenum condition, GLbitfield flags);
typedef bool (*glIsSyncPtr)(GLsync sync);
typedef void (*glDeleteSyncPtr)(GLsync sync);
typedef GLenum (*glClientWaitSyncPtr)(GLsync sync, GLbitfield flags, GLuint64 timeout);
typedef void (*glWaitSyncPtr)(GLsync sync, GLbitfield flags, GLuint64 timeout);
typedef void (*glGetInteger64vPtr)(GLenum pname, GLint64* params);
typedef void (*glGetSyncivPtr)(GLsync sync, GLenum pname, GLsizei bufSize, GLsizei* length,
                               GLint* values);
#endif

typedef struct __GLsync* GLsync;

glFenceSyncPtr glFenceSyncFunc;

glIsSyncPtr glIsSyncFunc;

glDeleteSyncPtr glDeleteSyncFunc;

glClientWaitSyncPtr glClientWaitSyncFunc;

glWaitSyncPtr glWaitSyncFunc;

glGetInteger64vPtr glGetInteger64vFunc;

glGetSyncivPtr glGetSyncivFunc;

#define CHK_GL_ERR() printf("%s\n", gluErrorString(glGetError()))

#define cl_khr_gl_event 1

static void InitSyncFns() {
#ifdef WIN_OS
  glFenceSyncFunc = (glFenceSyncPtr)wglGetProcAddress("glFenceSync");
  glIsSyncFunc = (glIsSyncPtr)wglGetProcAddress("glIsSync");
  glDeleteSyncFunc = (glDeleteSyncPtr)wglGetProcAddress("glDeleteSync");
  glClientWaitSyncFunc = (glClientWaitSyncPtr)wglGetProcAddress("glClientWaitSync");
  glWaitSyncFunc = (glWaitSyncPtr)wglGetProcAddress("glWaitSync");
  glGetInteger64vFunc = (glGetInteger64vPtr)wglGetProcAddress("glGetInteger64v");
  glGetSyncivFunc = (glGetSyncivPtr)wglGetProcAddress("glGetSynciv");
#else
  glFenceSyncFunc = (glFenceSyncPtr)glXGetProcAddress((GLubyte*)"glFenceSync");
  glIsSyncFunc = (glIsSyncPtr)glXGetProcAddress((GLubyte*)"glIsSync");
  glDeleteSyncFunc = (glDeleteSyncPtr)glXGetProcAddress((GLubyte*)"glDeleteSync");
  glClientWaitSyncFunc = (glClientWaitSyncPtr)glXGetProcAddress((GLubyte*)"glClientWaitSync");
  glWaitSyncFunc = (glWaitSyncPtr)glXGetProcAddress((GLubyte*)"glWaitSync");
  glGetInteger64vFunc = (glGetInteger64vPtr)glXGetProcAddress((GLubyte*)"glGetInteger64v");
  glGetSyncivFunc = (glGetSyncivPtr)glXGetProcAddress((GLubyte*)"glGetSynciv");
#endif
}

#define USING_ARB_sync 1

typedef cl_event(CL_API_CALL* clCreateEventFromGLsyncKHR_fn)(cl_context context, GLsync sync,
                                                             cl_int* errCode_ret);

clCreateEventFromGLsyncKHR_fn clCreateEventFromGLsyncKHR_ptr;

/* Helper to determine if an extension is supported by a device */
int is_extension_available(cl_device_id device, const char* extensionName) {
  char* extString;
  size_t size = 0;
  int err;
  int result = -1;

  if ((err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &size))) {
    printf(
        "Error: failed to determine size of device extensions string (err = "
        "%d)\n",
        err);
    return -2;
  }

  if (0 == size) return -3;

  extString = (char*)malloc(size);
  if (NULL == extString) {
    printf(
        "Error: unable to allocate %ld byte buffer for extension string (err = "
        "%d)\n",
        (long)size, err);
    return -40;
  }

  if ((err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, size, extString, NULL))) {
    printf("Error: failed to obtain device extensions string (err = %d)\n", err);
    free(extString);
    return -5;
  }

  if (strstr(extString, extensionName)) result = 0;

  free(extString);
  return result;
}

void OCLGLFenceSync::open(unsigned int test, char* units, double& conversion,
                          unsigned int deviceId) {
  _openTest = test;

  // Initialize random number seed
  srand((unsigned int)time(NULL));

  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  cl_context_properties properties[7] = {0};
  for (unsigned int i = 0; i < c_glContextCount; i++) {
    error_ = is_extension_available(devices_[_deviceId], "cl_khr_gl_event");
    if (error_ != CL_SUCCESS) {
      printf("Silent failure: cl_khr_gl_event extension not available (%d)\n", error_);
      extensionSupported_ = false;
      return;
    }
    extensionSupported_ = true;

    createGLContext(contextData_[i].glContext);
    getCLContextPropertiesFromGLContext(contextData_[i].glContext, properties);

    // Create new CL context from GL context
    contextData_[i].clContext =
        _wrapper->clCreateContext(properties, 1, &devices_[_deviceId], NULL, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateContext() failed (%d)", error_);

    // Create command queue for new context
    contextData_[i].clCmdQueue =
        _wrapper->clCreateCommandQueue(contextData_[i].clContext, devices_[_deviceId], 0, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateCommandQueue() failed (%d)", error_);

    // Build the kernel
    contextData_[i].clProgram = _wrapper->clCreateProgramWithSource(contextData_[i].clContext, 1,
                                                                    &strKernel, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed (%d)", error_);

    error_ = _wrapper->clBuildProgram(contextData_[i].clProgram, 1, &devices_[deviceId], NULL, NULL,
                                      NULL);
    if (error_ != CL_SUCCESS) {
      char programLog[1024];
      _wrapper->clGetProgramBuildInfo(contextData_[i].clProgram, devices_[deviceId],
                                      CL_PROGRAM_BUILD_LOG, 1024, programLog, 0);
      printf("\n%s\n", programLog);
      fflush(stdout);
    }
    CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed (%d)", error_);

    contextData_[i].clKernel =
        _wrapper->clCreateKernel(contextData_[i].clProgram, "glmulticontext_test", &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed (%d)", error_);
  }
}

void OCLGLFenceSync::run() {
  if (_errorFlag || !extensionSupported_) {
    return;
  }

  CPerfCounter timer;
  double sec;
  float perf;
  cl_uint4 inOutData[c_numOfElements] = {{{0}}};
  cl_uint4 expectedData[c_numOfElements] = {{{0}}};
  unsigned int m = sizeof(cl_uint4) / sizeof(cl_uint);
  int count = 0;
  // Initialize input data with random values
  for (unsigned int i = 0; i < c_numOfElements; i++) {
    for (unsigned int j = 0; j < m; j++) {
      inOutData[i].s[j] = (unsigned int)i;
      expectedData[i].s[j] = inOutData[i].s[j] + c_glContextCount;
    }
  }

  cl_event fenceEvent0 = NULL, fenceEvent = NULL;
  GLsync glFence0 = NULL, glFence = NULL;
  InitSyncFns();

  clCreateEventFromGLsyncKHR_ptr =
      (clCreateEventFromGLsyncKHR_fn)clGetExtensionFunctionAddress("clCreateEventFromGLsyncKHR");
  if (clCreateEventFromGLsyncKHR_ptr == NULL) {
    printf(
        "ERROR: Unable to run fence_sync test (clCreateEventFromGLsyncKHR "
        "function not discovered!)\n");
    return;
  }

  for (unsigned int i = 0; i < c_glContextCount; i++) {
    makeCurrent(contextData_[i].glContext);

    // Generate and Bind in & out OpenGL buffers
    GLuint inGLBuffer = 0, outGLBuffer = 0;
    glGenBuffers(1, &inGLBuffer);
    glGenBuffers(1, &outGLBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, inGLBuffer);
    glBufferData(GL_ARRAY_BUFFER, c_numOfElements * sizeof(cl_uint4), inOutData, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, outGLBuffer);
    glBufferData(GL_ARRAY_BUFFER, c_numOfElements * sizeof(cl_uint4), NULL, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glFinish();

    // Checking if clWaitForEvents works
    switch (_openTest) {
      case 0:  // Using fence sync
        glFence0 = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        CHECK_RESULT((glFence0 == NULL), "Unable to create GL fence");

        fenceEvent0 = clCreateEventFromGLsyncKHR_ptr(contextData_[i].clContext, glFence0, &error_);
        CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create CL event from GL fence (%d)",
                     error_);

        error_ = clWaitForEvents(1, &fenceEvent0);
        CHECK_RESULT((error_ != CL_SUCCESS), "clWaitForEvents() failed (%d)", error_);
        break;
      default:
        glFinish();
        break;
    }

    if (fenceEvent != NULL) {
      clReleaseEvent(fenceEvent0);
      glDeleteSync(glFence0);
    }

    cl_event acqEvent1 = 0, acqEvent2 = 0, kernelEvent = 0, relEvent1 = 0, relEvent2 = 0;

    // Create input buffer from GL input buffer
    contextData_[i].inputBuffer = _wrapper->clCreateFromGLBuffer(
        contextData_[i].clContext, CL_MEM_READ_ONLY, inGLBuffer, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create input GL buffer (%d)", error_);

    // Create output buffer from GL output buffer
    contextData_[i].outputBuffer = _wrapper->clCreateFromGLBuffer(
        contextData_[i].clContext, CL_MEM_WRITE_ONLY, outGLBuffer, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create output GL buffer (%d)", error_);

    timer.Reset();
    switch (_openTest) {
      case 0:  // Using fence sync
        timer.Start();
        glFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        timer.Stop();
        CHECK_RESULT((glFence == NULL), "Unable to create GL fence");

        timer.Start();
        fenceEvent = clCreateEventFromGLsyncKHR_ptr(contextData_[i].clContext, glFence, &error_);
        timer.Stop();
        CHECK_RESULT((error_ != CL_SUCCESS), "Unable to create CL event from GL fence (%d)",
                     error_);
        break;
      default:
        break;
    }

    error_ = _wrapper->clSetKernelArg(contextData_[i].clKernel, 0, sizeof(cl_mem),
                                      &(contextData_[i].inputBuffer));
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed (%d)", error_);

    error_ = _wrapper->clSetKernelArg(contextData_[i].clKernel, 1, sizeof(cl_mem),
                                      &(contextData_[i].outputBuffer));
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed (%d)", error_);

    switch (_openTest) {
      case 0:  // Using fence sync
        timer.Start();
        error_ = _wrapper->clEnqueueAcquireGLObjects(contextData_[i].clCmdQueue, 1,
                                                     &(contextData_[i].inputBuffer), 1, &fenceEvent,
                                                     &acqEvent1);
        timer.Stop();
        CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);

        timer.Start();
        error_ = _wrapper->clEnqueueAcquireGLObjects(contextData_[i].clCmdQueue, 1,
                                                     &(contextData_[i].outputBuffer), 1,
                                                     &fenceEvent, &acqEvent2);
        timer.Stop();
        CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);
        break;
      case 1:  // Using glFinish
        timer.Start();
        glFinish();
        timer.Stop();

        timer.Start();
        error_ = _wrapper->clEnqueueAcquireGLObjects(
            contextData_[i].clCmdQueue, 1, &(contextData_[i].inputBuffer), 0, NULL, &acqEvent1);
        timer.Stop();
        CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);

        timer.Start();
        error_ = _wrapper->clEnqueueAcquireGLObjects(
            contextData_[i].clCmdQueue, 1, &(contextData_[i].outputBuffer), 0, NULL, &acqEvent2);
        timer.Stop();
        CHECK_RESULT((error_ != CL_SUCCESS), "Unable to acquire GL objects (%d)", error_);
        break;
      default:
        break;
    }

    size_t gws[1] = {c_numOfElements};
    cl_event evts[2] = {acqEvent1, acqEvent2};
    error_ = _wrapper->clEnqueueNDRangeKernel(contextData_[i].clCmdQueue, contextData_[i].clKernel,
                                              1, NULL, gws, NULL, 2, evts, &kernelEvent);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed (%d)", error_);

    error_ = _wrapper->clEnqueueReleaseGLObjects(
        contextData_[i].clCmdQueue, 1, &(contextData_[i].inputBuffer), 1, &kernelEvent, &relEvent1);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReleaseGLObjects failed (%d)", error_);

    error_ = _wrapper->clEnqueueReleaseGLObjects(contextData_[i].clCmdQueue, 1,
                                                 &(contextData_[i].outputBuffer), 1, &kernelEvent,
                                                 &relEvent2);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReleaseGLObjects failed (%d)", error_);

    evts[0] = relEvent1;
    evts[1] = relEvent2;
    error_ = clWaitForEvents(2, evts);
    CHECK_RESULT((error_ != CL_SUCCESS), "clWaitForEvents() failed (%d)", error_);

    glBindBuffer(GL_ARRAY_BUFFER, outGLBuffer);
    void* glMem = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    memcpy(inOutData, glMem, c_numOfElements * sizeof(cl_uint4));
    glUnmapBuffer(GL_ARRAY_BUFFER);

    _wrapper->clReleaseMemObject(contextData_[i].inputBuffer);
    _wrapper->clReleaseMemObject(contextData_[i].outputBuffer);

    // Delete GL buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &inGLBuffer);
    inGLBuffer = 0;
    glDeleteBuffers(1, &outGLBuffer);
    outGLBuffer = 0;
  }

  sec = timer.GetElapsedTime();
  perf = (float)sec * 1000000;  // in microseconds
  _perfInfo = (float)perf;

  if (fenceEvent != NULL) {
    clReleaseEvent(fenceEvent);
    glDeleteSync(glFence);
  }

  // Compare expected output with actual data received
  for (unsigned int i = 0; i < c_numOfElements; i++) {
    for (unsigned int j = 0; j < m; j++) {
      if (inOutData[i].s[j] != expectedData[i].s[j]) {
        printf(
            "Element %u is incorrect!\t expected:[ %u, %u, %u, %u ] differs "
            "from actual:{%u, %u, %u, %u}\n",
            i, expectedData[i].s[0], expectedData[i].s[1], expectedData[i].s[2],
            expectedData[i].s[3], inOutData[i].s[0], inOutData[i].s[1], inOutData[i].s[2],
            inOutData[i].s[3]);

        count++;
      }
    }
  }
  if (count) printf("Number of elements wrong: %d\n", count);
}

unsigned int OCLGLFenceSync::close() {
  error_ = is_extension_available(devices_[_deviceId], "cl_khr_gl_event");
  if (error_ == CL_SUCCESS) {
    for (unsigned int i = 0; i < c_glContextCount; i++) {
      makeCurrent(contextData_[i].glContext);
      _wrapper->clReleaseKernel(contextData_[i].clKernel);
      _wrapper->clReleaseProgram(contextData_[i].clProgram);
      _wrapper->clReleaseCommandQueue(contextData_[i].clCmdQueue);
      _wrapper->clReleaseContext(contextData_[i].clContext);
      destroyGLContext(contextData_[i].glContext);
    }
  }

  return OCLGLCommon::close();
}
