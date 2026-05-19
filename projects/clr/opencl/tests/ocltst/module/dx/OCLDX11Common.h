/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_DX11_COMMON_H_
#define _OCL_DX11_COMMON_H_

#include <CL/cl.h>
#include <CL/cl_d3d11.h>

#include "OCLTestImp.h"
#include "d3d11.h"

typedef CL_API_ENTRY cl_mem(CL_API_CALL* clGetPlaneFromImageAMD_fn)(cl_context /* context */,
                                                                    cl_mem /* mem */,
                                                                    cl_uint /* plane */,
                                                                    cl_int* /* errcode_ret */);

class OCLDX11Common : public OCLTestImp {
 public:
  // S///////////////////////////////////////
  // private initialization and clean-up //
  /////////////////////////////////////////
  OCLDX11Common();
  virtual ~OCLDX11Common();
  ///////////////////////
  // virtual interface //
  ///////////////////////
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual unsigned int close(void);

 protected:
  bool extensionsAvailable;

  ID3D11Device* dxD3D11Device;
  ID3D11DeviceContext* dxD3D11Context;
  ID3D11Texture2D* dxDX11Texture;
  cl_command_queue _queue;

  clGetDeviceIDsFromD3D11KHR_fn clGetDeviceIDsFromD3D11KHR;
  clCreateFromD3D11BufferKHR_fn clCreateFromD3D11BufferKHR;
  clCreateFromD3D11Texture2DKHR_fn clCreateFromD3D11Texture2DKHR;
  clCreateFromD3D11Texture3DKHR_fn clCreateFromD3D11Texture3DKHR;
  clEnqueueAcquireD3D11ObjectsKHR_fn clEnqueueAcquireD3D11ObjectsKHR;
  clEnqueueReleaseD3D11ObjectsKHR_fn clEnqueueReleaseD3D11ObjectsKHR;
  clGetPlaneFromImageAMD_fn clGetPlaneFromImageAMD;

 private:
  void ExtensionCheck();
};

#endif  // _OCL_DX11_COMMON_H_
