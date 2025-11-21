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

#include "OCLDX11YUY2.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define DXGI_FORMAT_NV12 103
#define DXGI_FORMAT_P010 104
#define GROUP_SIZE 256

const static char strKernel[] =
    "__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | "
    "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; \n"
    "__kernel void image2imageCopy(                                            "
    "                             \n"
    "   __read_only image2d_t input,                                           "
    "                             \n"
    "   __write_only image2d_t output)                                         "
    "                             \n"
    "{                                                                         "
    "                             \n"
    "   int2 coord = (int2)(get_global_id(0), get_global_id(1));               "
    "                             \n"
    "   uint4 temp = read_imageui(input, imageSampler, coord);                 "
    "                             \n"
    "   write_imageui(output, coord, temp);                                    "
    "                             \n"
    "}                                                                         "
    "                             \n";

OCLDX11YUY2::OCLDX11YUY2() : OCLDX11Common() {
  _numSubTests = 4;
  blockSizeX = GROUP_SIZE;
  blockSizeY = 1;
}

OCLDX11YUY2::~OCLDX11YUY2() {}

void OCLDX11YUY2::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  dxDX11Texture = 0;
  clImage2DOut = 0;
  _openTest = test;
  // Initialize random number seed
  srand((unsigned int)time(NULL));

  OCLDX11Common::open(test, units, conversion, deviceId);
  if (_errorFlag) return;
  if (!extensionsAvailable) {
    return;
  }

  if (_openTest < 2) {
    dxFormat = (DXGI_FORMAT)DXGI_FORMAT_NV12;
    extensionsAvailable = formatSupported();
    if (!extensionsAvailable) {
      printf("DXGI_FORMAT_NV12 is required for this test!\n");
      return;
    }
  } else {
    dxFormat = (DXGI_FORMAT)DXGI_FORMAT_P010;
    extensionsAvailable = formatSupported();
    if (!extensionsAvailable) {
      printf("DXGI_FORMAT_P010 is required for this test!\n");
      return;
    }
  }

  CompileKernel();
  AllocateOpenCLImage();
}

void OCLDX11YUY2::run(void) {
  if (_errorFlag) return;
  if (!extensionsAvailable) return;

  D3D11_TEXTURE2D_DESC Desc = {0};

  Desc.ArraySize = 1;
  Desc.BindFlags = 0;
  Desc.Format = dxFormat;
  Desc.Width = OCLDX11YUY2::WIDTH;
  Desc.Height = OCLDX11YUY2::HEIGHT;
  Desc.MipLevels = 1;
  Desc.SampleDesc.Count = 1;
  // Desc.MiscFlags=D3D11_RESOURCE_MISC_SHARED; //MM for fast GPU interop
  // MM: these flags are incompatible with D3D11_RESOURCE_MISC_SHARED
  // now we allocate texture without CPU access and if needed use temp texture
  // (see FromSystemToDX11 and FromDX11ToSystem)

  Desc.Usage = D3D11_USAGE_STAGING;
  Desc.BindFlags = 0;
  Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE | D3D11_CPU_ACCESS_READ;

  ID3D11Texture2D* pTextureTmp;
  HRESULT hr = dxD3D11Device->CreateTexture2D(&Desc, NULL, &pTextureTmp);

  // fill memory
  D3D11_MAPPED_SUBRESOURCE LockedRectD11;
  if (SUCCEEDED(hr)) {
    hr = dxD3D11Context->Map(pTextureTmp, 0, D3D11_MAP_WRITE, 0, &LockedRectD11);
  }
  if (SUCCEEDED(hr)) {
    // fill memory with something
    for (int y = 0; y < OCLDX11YUY2::HEIGHT; y++) {
      BYTE* pLine = (BYTE*)LockedRectD11.pData + y * LockedRectD11.RowPitch;

      BYTE* pLineUV = (BYTE*)LockedRectD11.pData + y * LockedRectD11.RowPitch +
                      OCLDX11YUY2::HEIGHT * LockedRectD11.RowPitch;

      for (int x = 0; x < OCLDX11YUY2::WIDTH; x++) {
        *pLine++ = 0x7F;  // Y
        if (y < OCLDX11YUY2::HEIGHT / 2 && x < OCLDX11YUY2::WIDTH / 2) {
          *pLineUV++ = 0x1F;  // U
          *pLineUV++ = 0x2F;  // V
        }
      }
    }

    dxD3D11Context->Unmap(pTextureTmp, 0);
  }
  Desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
  Desc.Usage = D3D11_USAGE_DEFAULT;
  Desc.CPUAccessFlags = 0;
  Desc.MiscFlags = (_openTest == 0) ? 0 : D3D11_RESOURCE_MISC_SHARED;  // MM for fast GPU interop

  hr = dxD3D11Device->CreateTexture2D(&Desc, NULL, &dxDX11Texture);

  if (pTextureTmp != NULL) {
    dxD3D11Context->CopySubresourceRegion(dxDX11Texture, 0, 0, 0, 0, pTextureTmp, 0, NULL);
    pTextureTmp->Release();
  }
  testInterop();
}

void OCLDX11YUY2::AllocateOpenCLImage() {
  cl_int status = 0;

  cl_image_format format{};
  format.image_channel_order = CL_R;
  format.image_channel_data_type =
      (dxFormat == DXGI_FORMAT_NV12) ? CL_UNSIGNED_INT8 : CL_UNSIGNED_INT16;
  cl_image_desc descr{};
  descr.image_type = CL_MEM_OBJECT_IMAGE2D;
  descr.image_width = WIDTH;
  descr.image_height = HEIGHT + HEIGHT / 2;

  clImage2DOut = clCreateImage(context_, CL_MEM_WRITE_ONLY, &format, &descr, NULL, &status);
  CHECK_RESULT((status != CL_SUCCESS), "AllocateOpenCLImage() failed");
}

void OCLDX11YUY2::testInterop() {
  // alloc
  cl_int clStatus = 0;
  cl_mem clImage2D = clCreateFromD3D11Texture2DKHR(context_, 0, dxDX11Texture, 0, &clStatus);
  CHECK_RESULT((clStatus != CL_SUCCESS), "clCreateFromD3D11Texture2DKHR() failed");

  // bring objects to the queue
  cl_event clEvent = NULL;
  clEnqueueAcquireD3D11ObjectsKHR(_queue, 1, &clImage2D, 0, NULL, &clEvent);
  clStatus = clWaitForEvents(1, &clEvent);
  clReleaseEvent(clEvent);

  CopyOpenCLImage(clImage2D);
  bool ImageReadWorks = CheckCLImage(clImage2D);
  bool bKernelWorks = CheckCLImage(clImage2DOut);
  CHECK_RESULT_NO_RETURN((ImageReadWorks != true), "CheckCLImage(clImage2D) failed");
  CHECK_RESULT_NO_RETURN((bKernelWorks != true), "CheckCLImage(clImage2DOut) failed");

  cl_mem planeY = clGetPlaneFromImageAMD(context_, clImage2D, 0, &clStatus);
  CHECK_RESULT((clStatus != CL_SUCCESS),
               "clGetPlaneFromImageAMD(context_,clImage2D,0,&clStatus) failed");

  cl_mem planeUV = clGetPlaneFromImageAMD(context_, clImage2D, 1, &clStatus);
  CHECK_RESULT((clStatus != CL_SUCCESS),
               "clGetPlaneFromImageAMD(context_,clImage2D,1,&clStatus) failed");

  bool ImageWorksY = CheckCLImageY(planeY);
  bool ImageWorksUV = CheckCLImageUV(planeUV);

  clReleaseMemObject(planeY);
  clReleaseMemObject(planeUV);

  // release
  clEvent = NULL;
  // release object from the queue
  clStatus = clEnqueueReleaseD3D11ObjectsKHR(_queue, 1, &clImage2D, 0, NULL, &clEvent);
  clStatus = clWaitForEvents(1, &clEvent);
  clReleaseEvent(clEvent);

  // release mem object
  clReleaseMemObject(clImage2D);

  CHECK_RESULT_NO_RETURN((ImageWorksY != true), "CheckCLImageY() failed");
  CHECK_RESULT_NO_RETURN((ImageWorksUV != true), "CheckCLImageUV() failed");
}

unsigned int OCLDX11YUY2::close(void) {
  if (clImage2DOut) clReleaseMemObject(clImage2DOut);
  if (dxDX11Texture) dxDX11Texture->Release();
  return OCLDX11Common::close();
}

bool OCLDX11YUY2::CheckCLImage(cl_mem clImage) {
  cl_int clStatus = 0;

  size_t pitch = 0;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_ROW_PITCH, sizeof(pitch), &pitch, NULL);
  pitch *= 2;

  cl_image_format format;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);

  size_t height;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);

  CHECK_RESULT_NO_RETURN(height != (HEIGHT + HEIGHT / 2),
                         "CheckCLImage: height!=(HEIGHT+HEIGHT/2)");

  char* pTempBuffer = new char[(HEIGHT + HEIGHT / 2) * pitch];

  size_t origin[] = {0, 0, 0};
  size_t region[] = {WIDTH, HEIGHT + HEIGHT / 2, 1};
  clStatus = clEnqueueReadImage(_queue, clImage, 1, origin, region, pitch, 0, pTempBuffer, 0, 0, 0);

  ::clFinish(_queue);

  // test

  bool bBreak = false;
  for (int y = 0; y < HEIGHT && !bBreak; y++) {
    char* pLine = (char*)pTempBuffer + y * pitch;
    char* pLineUV = (char*)pTempBuffer + y * pitch + HEIGHT * pitch;

    for (int x = 0; x < WIDTH; x++) {
      if (*pLine != 0x7F)  // Y
      {
        bBreak = true;
        break;
      }
      pLine++;
      if (y < HEIGHT / 2 && x < WIDTH / 2) {
        if (*pLineUV != 0x1F)  // U
        {
          bBreak = true;
          break;
        }
        pLineUV++;
        if (*pLineUV != 0x2F)  // V
        {
          bBreak = true;
          break;
        }
        pLineUV++;
      }
    }
  }
  delete[] pTempBuffer;

  return !bBreak;
}

bool OCLDX11YUY2::CheckCLImageY(cl_mem clImage) {
  cl_int clStatus = 0;

  size_t pitch = 0;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_ROW_PITCH, sizeof(pitch), &pitch, NULL);
  pitch *= 2;

  cl_image_format format;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);

  size_t height;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);

  CHECK_RESULT_NO_RETURN(height != HEIGHT, "CheckCLImageY: height!=HEIGHT");

  char* pTempBuffer = new char[HEIGHT * pitch];

  size_t origin[] = {0, 0, 0};
  size_t region[] = {WIDTH, HEIGHT, 1};
  clStatus = clEnqueueReadImage(_queue, clImage, 1, origin, region, pitch, 0, pTempBuffer, 0, 0, 0);

  ::clFinish(_queue);

  // test

  bool bBreak = false;
  for (int y = 0; y < HEIGHT && !bBreak; y++) {
    char* pLine = (char*)pTempBuffer + y * pitch;
    for (int x = 0; x < WIDTH; x++) {
      if (*pLine != 0x7F)  // Y
      {
        bBreak = true;
        break;
      }
      pLine++;
    }
  }

  delete[] pTempBuffer;

  return !bBreak;
}

bool OCLDX11YUY2::CheckCLImageUV(cl_mem clImage) {
  cl_int clStatus = 0;

  size_t pitch = 0;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_ROW_PITCH, sizeof(pitch), &pitch, NULL);
  pitch *= 2;
  size_t width = 0;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);

  cl_image_format format;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);

  size_t height;
  clStatus = clGetImageInfo(clImage, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);

  CHECK_RESULT_NO_RETURN(height != HEIGHT / 2, "CheckCLImageUV: height!=HEIGHT/2");

  char* pTempBuffer = new char[(HEIGHT / 2) * pitch];

  size_t origin[] = {0, 0, 0};
  size_t region[] = {WIDTH / 2, HEIGHT / 2, 1};
  clStatus = clEnqueueReadImage(_queue, clImage, 1, origin, region, pitch, 0, pTempBuffer, 0, 0, 0);

  ::clFinish(_queue);

  bool bBreak = false;
  for (int y = 0; y < HEIGHT / 2 && !bBreak; y++) {
    char* pLineUV = (char*)pTempBuffer + y * pitch;
    for (int x = 0; x < WIDTH / 2; x++) {
      if (*pLineUV != 0x1F)  // U
      {
        bBreak = true;
        break;
      }
      pLineUV++;
      if (*pLineUV != 0x2F)  // V
      {
        bBreak = true;
        break;
      }
      pLineUV++;
    }
  }
  delete[] pTempBuffer;

  return !bBreak;
}

void OCLDX11YUY2::CopyOpenCLImage(cl_mem clImageSrc) {
  cl_int status = 0;

  // Set appropriate arguments to the kernel2D

  // input buffer image
  status = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &clImageSrc);
  CHECK_RESULT((status != CL_SUCCESS),
               "CopyOpenCLImage() failed at "
               "clSetKernelArg(kernel_,0,sizeof(cl_mem),&clImageSrc)");
  status = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &clImage2DOut);
  CHECK_RESULT((status != CL_SUCCESS),
               "CopyOpenCLImage() failed at "
               "clSetKernelArg(kernel_,1,sizeof(cl_mem),&clImage2DOut)");

  // Enqueue a kernel run call.
  size_t global_work_offset[] = {0, 0};
  size_t globalThreads[] = {WIDTH, HEIGHT + HEIGHT / 2};
  size_t localThreads[] = {blockSizeX, blockSizeY};

  // status =
  // clEnqueueNDRangeKernel(_queue,kernel_,2,NULL,globalThreads,localThreads,0,NULL,0);
  status = clEnqueueNDRangeKernel(_queue, kernel_, 2, NULL, globalThreads, NULL, 0, NULL, 0);
  CHECK_RESULT((status != CL_SUCCESS), "CopyOpenCLImage() failed at clEnqueueNDRangeKernel");

  status = clFinish(_queue);
  CHECK_RESULT((status != CL_SUCCESS), "CopyOpenCLImage() failed at clFinish");
}

void OCLDX11YUY2::CompileKernel() {
  cl_int status = 0;

  size_t kernelSize = sizeof(strKernel);
  const char* strs = (const char*)&strKernel[0];

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strs, &kernelSize, &status);

  status = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], NULL, NULL, NULL);
  if (status != CL_SUCCESS) {
    if (status == CL_BUILD_PROGRAM_FAILURE) {
      cl_int logStatus;
      size_t buildLogSize = 0;
      logStatus = clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG,
                                        buildLogSize, NULL, &buildLogSize);
      std::string buildLog;
      buildLog.resize(buildLogSize);

      logStatus = clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG,
                                        buildLogSize, &buildLog[0], NULL);
      printf("%s", buildLog.c_str());
    }
    return;
  }
  // get a kernel object handle for a kernel with the given name
  kernel_ = _wrapper->clCreateKernel(program_, "image2imageCopy", &status);

  size_t kernel2DWorkGroupSize = 0;
  status = clGetKernelWorkGroupInfo(kernel_, devices_[_deviceId], CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(size_t), &kernel2DWorkGroupSize, 0);

  if ((blockSizeX * blockSizeY) > kernel2DWorkGroupSize) {
    if (blockSizeX > kernel2DWorkGroupSize) {
      blockSizeX = kernel2DWorkGroupSize;
      blockSizeY = 1;
    }
  }
}

bool OCLDX11YUY2::formatSupported() {
  UINT supported = 0u;
  dxD3D11Device->CheckFormatSupport(dxFormat, (UINT*)&supported);
  return supported & D3D11_FORMAT_SUPPORT_TEXTURE2D;
}
