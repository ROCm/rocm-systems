/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCLImage2DFromBuffer_H_
#define _OCLImage2DFromBuffer_H_

#include "OCLTestImp.h"

class OCLImage2DFromBuffer : public OCLTestImp {
 public:
  OCLImage2DFromBuffer();
  virtual ~OCLImage2DFromBuffer();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 protected:
  static const unsigned int imageWidth;
  static const unsigned int imageHeight;

  void testReadImage(cl_mem image);
  void testKernel();
  void AllocateOpenCLImage();
  void CopyOpenCLImage(cl_mem clImageSrc);
  void CompileKernel();

  bool done_;
  size_t blockSizeX; /**< Work-group size in x-direction */
  size_t blockSizeY; /**< Work-group size in y-direction */
  cl_mem buffer;
  cl_mem clImage2DOriginal;
  cl_mem clImage2D;
  cl_mem clImage2DOut;
  cl_uint pitchAlignment;
};

#endif  // _OCLImage2DFromBuffer_H_
