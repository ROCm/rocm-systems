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

#include "OCLGLPerfSepia.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define WIDTH 1024
#define HEIGHT 1024

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define MAX(a, b) (a > b ? a : b)

const char* sepiaVertexProgram =
    "!!ARBvp1.0\n"
    "\n"
    "\n"
    "OPTION ARB_position_invariant;\n"
    "\n"
    "PARAM p0 = program.local[2];\n"
    "PARAM p1 = program.local[3];\n"
    "ATTRIB a0 = vertex.texcoord[0];\n"
    "OUTPUT o0 = result.texcoord[0];\n"
    "OUTPUT o1 = result.texcoord[1];\n"
    "TEMP r0, r1;\n"
    "\n"
    "MOV o0, a0;\n"
    "#SWZ r1, a0, x, y, 0, 0;\n"
    "#DPH r0.x, r1, p0;\n"
    "#DPH r0.y, r1, p1;\n"
    "#MOV o1, r0;\n"
    "MOV o1, a0;\n"
    "\n"
    "END\n";

const char* sepiaFragmentProgram =
    "!!ARBfp1.0\n"
    "\n"
    "\n"
    "PARAM p0 = {1e-4, 0.085, 0.0, 0.0};\n"
    "PARAM p1 = {0.2125, 0.7154, 0.0721, 0.0};\n"
    "PARAM p2 = {-3605.984, 0.1323156, 0.0, -0.1991615};\n"
    "PARAM p3 = {708.7939, -0.3903106, -0.05854013, 0.6621023};\n"
    "PARAM p4 = {-50.93341, 0.4654831, 1.027555, -0.9069088};\n"
    "PARAM p5 = {3.116672, 0.7926372, 0.03219686, 1.411847};\n"
    "PARAM p6 = {8.95663e-4, -0.001104567, -6.0827e-4, 0.03277428};\n"
    "PARAM p7 = program.local[0];\n"
    "PARAM p8 = program.local[1];\n"
    "ATTRIB a0 = fragment.texcoord[1];\n"
    "OUTPUT o0 = result.color;\n"
    "TEMP r0, r1, r2, r3;\n"
    "\n"
    "TEX r1, a0, texture[0], RECT;\n"
    "#MAX r0, p0.x, r1.w;\n"
    "#RCP r2, r0.x;\n"
    "#DP3 r3, r1, p1;\n"
    "#MUL r0, r3, r2;\n"
    "#MAD r2, r0, p2, p3;\n"
    "#MAD r2, r2, r0, p4;\n"
    "#MAD r0, r2, r0, p5;\n"
    "#MUL r2, r1.w, p6;\n"
    "#MAD r2, r0, r3, r2;\n"
    "#MAD r0, r1.w, p0.y, -r3;\n"
    "#CMP r2.x, -r0, r2.x, r2.w;\n"
    "#MAD r0, r3, r3, -r3;\n"
    "#CMP r0, r0.x, r2, r3;\n"
    "#MOV r0.w, r1;\n"
    "#MUL r0, r0, p7;\n"
    "#LRP o0, p8.x, r0, r1;\n"
    "MOV o0, r1;\n"
    "\n"
    "END\n";

const static char* strKernel =
    "\n"
    "__kernel void program(write_only image2d_t dest, int flipped, int4 dim, "
    "float2 st_origin, float4 st_delta, float4 l0, float4 l1, float4 l2, "
    "float4 l3, read_only image2d_t t0, sampler_t t_sampler0)\n"
    "{\n"
    "      const sampler_t sam = CLK_NORMALIZED_COORDS_FALSE | "
    "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
    "//    const float4 p0  = (float4)( 0x1.b33334p-3, 0x1.6e48e8p-1, "
    "0x1.275254p-4, 0x0p+0 );\n"
    "//    const float4 p1  = (float4)( 0x1.a36e2ep-14, 0x1.5c28f6p-4, 0x0p+0, "
    "0x0p+0 );\n"
    "//    const float4 p2  = (float4)( 0x1.d595dap-11, -0x1.218e3cp-10, "
    "-0x1.3ee89ep-11, 0x1.0c7ca6p-5 );\n"
    "//    const float4 p3  = (float4)( -0x1.c2bf7cp+11, 0x1.0efb7cp-3, "
    "0x0p+0, -0x1.97e1fcp-3 );\n"
    "//    const float4 p4  = (float4)( 0x1.62659ep+9, -0x1.8fad94p-2, "
    "-0x1.df8f8cp-5, 0x1.52ff12p-1 );\n"
    "//   const float4 p5  = (float4)( -0x1.9777ap+5, 0x1.dca79ap-2, "
    "0x1.070dd8p+0, -0x1.d0565ap-1 );\n"
    "//    const float4 p6  = (float4)( 0x1.8eef1cp+1, 0x1.95d48cp-1, "
    "0x1.07c1b6p-5, 0x1.696ecep+0 );\n"
    "//    int          dest_width = dim.x;\n"
    "//    int          dest_height = dim.y;\n"
    "    float4       o0, r0, r1, r2, r3, r4;\n"
    "//    float4       false_vector = (float4) 0.0f;\n"
    "//    float4       true_vector = (float4) 1.0f;\n"
    "    int2         loc = (int2)( get_global_id(0), get_global_id(1) );\n"
    "//    if ((loc.x >= dim.x) || loc.y >= dim.y) return;\n"
    "//    float4 f0 = (float4)( st_origin.x + ((float)loc.x + 0.5f) * "
    "st_delta.x + ((float)loc.y + 0.5f) * st_delta.z, st_origin.y + "
    "((float)loc.x + 0.5f) * st_delta.y + ((float)loc.y + 0.5f) * st_delta.w, "
    "0.0f, 0.0f );\n"
    "//    r2 = f0;\n"
    "//    r0.x = dot(r2.xy,l2.xy) + l2.w;\n"
    "//    r0.y = dot(r2.xy,l3.xy) + l3.w;\n"
    "//    r4 = r0;\n"
    "    r1 = read_imagef(t0, sam/*t_sampler0*/, r4.xy);\n"
    "//    r3 = dot(r1.xyz,p0.xyz);\n"
    "//    r2 = max(p1.xxxx, r1.wwww);\n"
    "//    r0 = native_recip(r2.xxxx);\n"
    "//    r4 = r3*r0;\n"
    "//    r2 = r1.wwww*p2;\n"
    "//    r0 = mad(r4,p3,p4);\n"
    "//    r0 = mad(r0,r4,p5);\n"
    "//    r0 = mad(r0,r4,p6);\n"
    "//    r2 = mad(r0,r3,r2);\n"
    "//    r0 = mad(r1.wwww,p1.yyyy,-r3);\n"
    "//    r2.x = select(r2.w,r2.x, isless(-r0.x, 0.0f));\n"
    "//    r0 = mad(r3,r3,-r3);\n"
    "//    r0 = select(r3,r2, isless(r0.xxxx, 0.0f));\n"
    "//    r0.w = r1.w;\n"
    "//    r0 = r0*l0;\n"
    "//    r0 = mix(r1,r0, l1.xxxx);\n"
    "//    r0.xyz = min(r0.xyz, r0.www);\n"
    "//    o0 = r0;\n"
    "    write_imagef(dest, loc /*(int2)( loc.x + dim.z , flipped ? "
    "get_image_height(dest) - (loc.y + dim.w + 1) : loc.y + dim.w )*/, r1 "
    "/*o0*/);\n"
    "}\n";

OCLGLPerfSepia::OCLGLPerfSepia() { _numSubTests = 2; }

OCLGLPerfSepia::~OCLGLPerfSepia() {}

void OCLGLPerfSepia::open(unsigned int test, char* units, double& conversion,
                          unsigned int deviceId) {
  bVerify_ = false;
  silentFailure_ = false;
  iterations_ = 50000;
  bpr_ = 0;
  data_ = 0;
  result_ = 0;
  width_ = 0;
  height_ = 0;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;
  texId = 0;
  format_.image_channel_order = CL_RGBA;
  format_.image_channel_data_type = CL_UNORM_INT8;

  srand(0x8956);  // some constant instead of time() so that we get same random
                  // numbers

  if (!IsGLEnabled(test, units, conversion, deviceId)) {
    silentFailure_ = true;
    return;
  }
  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;
  if (test == 0) {
    // Build the kernel
    program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed (%d)", error_);
    const char* optionsGPU = "-cl-denorms-are-zero -cl-mad-enable";
    error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], optionsGPU, NULL, NULL);
    if (error_ != CL_SUCCESS) {
      char programLog[1024];
      _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                      programLog, 0);
      printf("\n%s\n", programLog);
      fflush(stdout);
    }
    CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed (%d)", error_);

    kernel_ = _wrapper->clCreateKernel(program_, "program", &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed (%d)", error_);
  }
}

void OCLGLPerfSepia::populateData(void) {
  width_ = WIDTH;
  height_ = HEIGHT;
  bpr_ = 4 * width_;
  data_ = (cl_uchar*)malloc(height_ * bpr_);
  for (unsigned int n = 0; n < (height_ * bpr_); n++) {
    data_[n] = (n & 3) ? (rand() % 256) : 0xFF;
  }
}

void OCLGLPerfSepia::runGL(void) {
  glDisable(GL_ALPHA_TEST);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_SCISSOR_TEST);
  glDisable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glDisable(GL_DITHER);
  glDisable(GL_CULL_FACE);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  glDepthMask(GL_FALSE);
  glStencilMask(0);

  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  // let's create the textures we need

  glEnable(GL_TEXTURE_RECTANGLE_EXT);
  glGenTextures(1, &texId);
  glBindTexture(GL_TEXTURE_RECTANGLE_EXT, texId);

  // have GL alloc memory for us for our destination texture which we will be
  // rendering into
  glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA, width_, height_, 0, GL_BGRA /*RGBA*/,
               GL_UNSIGNED_INT_8_8_8_8_REV, NULL);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // for the source texture we will provide a data ptr and hang on to it
  GLuint srcTexture;

  glGenTextures(1, &srcTexture);
  glBindTexture(GL_TEXTURE_RECTANGLE_EXT, srcTexture);

  glPixelStorei(GL_UNPACK_ROW_LENGTH, width_);
  glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, height_);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 8);

  // XXX Alex -- use optimal texture upload format.
  glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA, width_, height_, 0, GL_BGRA, /* GL_RGBA,*/
               format_.image_channel_order == CL_RGBA ? GL_UNSIGNED_INT_8_8_8_8
                                                      : GL_UNSIGNED_INT_8_8_8_8_REV,
               data_);

  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_SWAP_BYTES, 0);
  glPixelStorei(GL_UNPACK_LSB_FIRST, 0);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei(GL_UNPACK_SKIP_IMAGES, 0);
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

  GLuint vertexProgram;
  GLuint fragmentProgram;

  glGenProgramsARB(1, &vertexProgram);
  glGenProgramsARB(1, &fragmentProgram);

  glBindProgramARB(GL_VERTEX_PROGRAM_ARB, vertexProgram);
  glProgramStringARB(GL_VERTEX_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
                     (GLsizei)strlen(sepiaVertexProgram), sepiaVertexProgram);

  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, fragmentProgram);
  glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
                     (GLsizei)strlen(sepiaFragmentProgram), sepiaFragmentProgram);

  GLfloat l0[] = {1.0f, 0.99f, 0.92f, 1.0f};
  GLfloat l1[] = {0.5, 0, 0, 0};
  GLfloat l2[] = {1, 0, 0, 0};
  GLfloat l3[] = {0, -1, 0, (GLfloat)height_};

  glProgramLocalParameter4fvARB(GL_VERTEX_PROGRAM_ARB, 0, l0);
  glProgramLocalParameter4fvARB(GL_VERTEX_PROGRAM_ARB, 1, l1);
  glProgramLocalParameter4fvARB(GL_VERTEX_PROGRAM_ARB, 2, l2);
  glProgramLocalParameter4fvARB(GL_VERTEX_PROGRAM_ARB, 3, l3);

  glProgramLocalParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 0, l0);
  glProgramLocalParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 1, l1);
  glProgramLocalParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 2, l2);
  glProgramLocalParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 3, l3);

  GLuint fbo;

  glGenFramebuffersEXT(1, &fbo);

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB,
                            texId, 0);
  glViewport(0, 0, width_, height_);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, width_, 0, height_, -1, 1);
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDisable(GL_BLEND);

  glEnable(GL_VERTEX_PROGRAM_ARB);
  glEnable(GL_FRAGMENT_PROGRAM_ARB);

  // warm up
  for (unsigned int k = 0; k < (iterations_ / 10); k++) {
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, (GLfloat)height_);
    glTexCoord2f((GLfloat)width_, 0);
    glVertex2f((GLfloat)width_, (GLfloat)height_);
    glTexCoord2f((GLfloat)width_, (GLfloat)height_);
    glVertex2f((GLfloat)width_, 0);
    glTexCoord2f(0, (GLfloat)height_);
    glVertex2f(0, 0);
    glEnd();
    glFlush();
    glFinish();
  }

  // actual test
  for (unsigned int k = 0; k < iterations_; k++) {
    if (k == 1) {
      timer_.Reset();
      timer_.Start();
    }

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, (GLfloat)height_);
    glTexCoord2f((GLfloat)width_, 0);
    glVertex2f((GLfloat)width_, (GLfloat)height_);
    glTexCoord2f((GLfloat)width_, (GLfloat)height_);
    glVertex2f((GLfloat)width_, 0);
    glTexCoord2f(0, (GLfloat)height_);
    glVertex2f(0, 0);
    glEnd();
  }

  glFlush();
  glFinish();

  timer_.Stop();

  glDisable(GL_VERTEX_PROGRAM_ARB);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);

  // now let's read back the pixels
  result_ = (cl_uchar*)malloc(width_ * height_ * 4);

  glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, result_);

  // bind back default frame buffer
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

  glDeleteFramebuffersEXT(1, &fbo);
  glDeleteTextures(1, &srcTexture);
  glDeleteProgramsARB(1, &vertexProgram);
  glDeleteProgramsARB(1, &fragmentProgram);
}

void OCLGLPerfSepia::runCL(void) {
  cl_mem dst, src;
  cl_sampler nearestZero;

  glEnable(GL_TEXTURE_RECTANGLE_EXT);
  glGenTextures(1, &texId);
  glBindTexture(GL_TEXTURE_RECTANGLE_EXT, texId);
  // XXX Alex: have GL alloc memory for us ...
  glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA, width_, height_, 0, GL_RGBA /*BGRA*/,
               GL_UNSIGNED_INT_8_8_8_8_REV, NULL);

  dst = _wrapper->clCreateFromGLTexture2D(context_, CL_MEM_READ_WRITE, GL_TEXTURE_RECTANGLE_EXT, 0,
                                          texId, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateFromGLTexture2D error (%d)", error_);
  nearestZero =
      _wrapper->clCreateSampler(context_, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateSampler error (%d)", error_);
  src = _wrapper->clCreateImage2D(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format_,
                                  width_, height_, bpr_, data_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateImage2D error (%d)", error_);

  int numArgs = 0;
  int dim[2] = {(int)width_, (int)height_};
  int flipped[] = {1};
  int dims[] = {(int)width_, (int)height_, 0, 0};
  float st_origin[] = {0, 0};
  float st_delta[] = {1, 0, 0, 1};

  _wrapper->clSetKernelArg(kernel_, numArgs++, sizeof(cl_mem),
                           &dst);  // arg is a image2DGL named "dst"
  _wrapper->clSetKernelArg(kernel_, numArgs++, sizeof(int),
                           &flipped);  // arg is a int1 named "flipped"
  _wrapper->clSetKernelArg(kernel_, numArgs++, 4 * sizeof(int),
                           &dims);  // arg is a int4 named "dim"
  _wrapper->clSetKernelArg(kernel_, numArgs++, 2 * sizeof(float),
                           &st_origin);  // arg is a float2 named "st_origin"
  _wrapper->clSetKernelArg(kernel_, numArgs++, 4 * sizeof(float),
                           &st_delta);  // arg is a float4 named "st_delta"

  float l0[] = {1.0f, 0.99f, 0.92f, 1.0f};
  float l1[] = {0.5f, 0.0f, 0.0f, 0.0f};
  float l2[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float l3[] = {0.0f, -1.0f, 0.0f, (float)height_};

  _wrapper->clSetKernelArg(kernel_, numArgs++, 4 * sizeof(float),
                           &l0);  // arg is a float4 named "l0"
  _wrapper->clSetKernelArg(kernel_, numArgs++, 4 * sizeof(float),
                           &l1);  // arg is a float4 named "l1"
  _wrapper->clSetKernelArg(kernel_, numArgs++, 4 * sizeof(float),
                           &l2);  // arg is a float4 named "l2"
  _wrapper->clSetKernelArg(kernel_, numArgs++, 4 * sizeof(float),
                           &l3);  // arg is a float4 named "l3"
  _wrapper->clSetKernelArg(kernel_, numArgs++, sizeof(cl_mem),
                           &src);  // arg is a image2D named "t0"
  _wrapper->clSetKernelArg(kernel_, numArgs++, sizeof(cl_sampler),
                           &nearestZero);  // arg is a sampler named "t_sampler0"

  size_t execution_threads[2];
  size_t execution_local[2];
  cl_uint work_dim = 2;
  error_ =
      _wrapper->clGetKernelWorkGroupInfo(kernel_, devices_[_deviceId], CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof(execution_local[0]), &execution_local[0], 0);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetKernelWorkGroupInfo error (%d)", error_);
  execution_local[1] = 1;
  work_dim = 2;
  GetKernelExecDimsForImage((unsigned int)execution_local[0], dim[0], dim[1], execution_threads,
                            execution_local);
  result_ = (cl_uchar*)malloc(height_ * bpr_);

  const size_t origin[] = {0, 0, 0};
  const size_t region[] = {width_, height_, 1};

  // warm up
  for (unsigned int k = 0; k < (iterations_ / 10); k++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, work_dim, NULL,
                                              execution_threads, execution_local, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel error (%d)", error_);
    error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
    CHECK_RESULT((error_ != CL_SUCCESS), "clFinish error (%d)", error_);
  }

  // actual test
  for (unsigned int k = 0; k < iterations_; k++) {
    if (k == 1) {
      timer_.Reset();
      timer_.Start();
    }
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, work_dim, NULL,
                                              execution_threads, execution_local, 0, NULL, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel error (%d)", error_);
  }
  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clFinish error (%d)", error_);

  timer_.Stop();

  error_ = _wrapper->clEnqueueReadImage(cmdQueues_[_deviceId], dst, true, origin, region, bpr_, 0,
                                        result_, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadImage error (%d)", error_);
  _wrapper->clFinish(cmdQueues_[_deviceId]);

  _wrapper->clReleaseMemObject(src), src = NULL;
  _wrapper->clReleaseSampler(nearestZero);
  _wrapper->clReleaseMemObject(dst), dst = NULL;
}

void OCLGLPerfSepia::GetKernelExecDimsForImage(unsigned int work_group_size, unsigned int w,
                                               unsigned int h, size_t* global, size_t* local) {
  unsigned int a, b;
  static const unsigned int tile_size = 16;

  // local[0] and local[1] must be at least 1
  local[0] = tile_size < work_group_size ? tile_size : work_group_size;
  local[1] =
      work_group_size / tile_size > tile_size ? tile_size : MAX(work_group_size / tile_size, 1);

  a = w;
  b = (unsigned int)local[0];

  global[0] = ((a % b) != 0) ? (a / b + 1) : (a / b);
  global[0] *= local[0];

  a = h;
  b = (unsigned int)local[1];

  global[1] = ((a % b) != 0) ? (a / b + 1) : (a / b);
  global[1] *= local[1];
}

void OCLGLPerfSepia::run(void) {
  if (_errorFlag || silentFailure_) {
    return;
  }
  populateData();
  if (_openTest == 0) {
    runCL();
  } else {
    runGL();
  }
  if (bVerify_) {
    verifyResult();
  }
  char buf[100];
  SNPRINTF(buf, sizeof(buf), "%s iterations# %d", (_openTest == 0) ? "CL" : "GL", iterations_);
  testDescString = buf;
  _perfInfo = (float)timer_.GetElapsedTime();
}

void OCLGLPerfSepia::verifyResult(void) {
  int r = 0, g = 0, b = 0, a = 0, d = 0;
  for (unsigned int k = 0; k < height_ * bpr_; k += 4) {
    a = a + result_[k + 0];
    r = r + result_[k + 1];
    g = g + result_[k + 2];
    b = b + result_[k + 3];
  }
  d = abs(r - 152797810) + abs(g - 125868080) + abs(b - 76147833) + abs(a - 267386880);
  CHECK_RESULT(d > 20000, "wrong result");
}
unsigned int OCLGLPerfSepia::close(void) {
  if (silentFailure_) {
    return 0;
  }

  if (data_) {
    free(data_);
  }

  if (result_) {
    free(result_);
  }

  if (texId) {
    glDeleteTextures(1, &texId);
  }

  return OCLGLCommon::close();
}
