/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_DX11_YUY2_H_
#define _OCL_DX11_YUY2_H_

#include "OCLDX11Common.h"

class OCLDX11YUY2 : public OCLDX11Common {
 public:
  OCLDX11YUY2();
  virtual ~OCLDX11YUY2();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 protected:
  static const unsigned int WIDTH = 1280;
  static const unsigned int HEIGHT = 720;

  void testInterop();
  void AllocateOpenCLImage();
  bool CheckCLImage(cl_mem clImage);
  bool CheckCLImageY(cl_mem clImage);
  bool CheckCLImageUV(cl_mem clImage);
  void CopyOpenCLImage(cl_mem clImageSrc);
  void CompileKernel();
  bool formatSupported();
  void testFormat();

  size_t blockSizeX; /**< Work-group size in x-direction */
  size_t blockSizeY; /**< Work-group size in y-direction */
  cl_mem clImage2DOut;
  DXGI_FORMAT dxFormat;
};

#endif  // _OCL_DX11_YUY2_H_
