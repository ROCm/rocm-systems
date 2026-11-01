/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLDX11YUY2.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define DXGI_FORMAT_NV12 103
#define DXGI_FORMAT_P010 104
#define GROUP_SIZE 256

OCLDX11YUY2::OCLDX11YUY2() : OCLDX11Common() {
  _numSubTests = 4;
  blockSizeX = GROUP_SIZE;
  blockSizeY = 1;
}

OCLDX11YUY2::~OCLDX11YUY2() {}

void OCLDX11YUY2::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  dxDX11Texture = 0;
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
  if (dxDX11Texture) dxDX11Texture->Release();
  return OCLDX11Common::close();
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

bool OCLDX11YUY2::formatSupported() {
  UINT supported = 0u;
  dxD3D11Device->CheckFormatSupport(dxFormat, (UINT*)&supported);
  return supported & D3D11_FORMAT_SUPPORT_TEXTURE2D;
}
