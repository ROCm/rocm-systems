/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCLBufferFromImage_H_
#define _OCLBufferFromImage_H_

#include "OCLTestImp.h"

class OCLBufferFromImage : public OCLTestImp {
 public:
  OCLBufferFromImage();
  virtual ~OCLBufferFromImage();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 protected:
  static const unsigned int imageWidth = 1920;
  static const unsigned int imageHeight = 1080;

  void testReadBuffer(cl_mem buffer);
  void testKernel();
  void AllocateOpenCLBuffer();
  void CopyOpenCLBuffer(cl_mem buffer);
  void CompileKernel();

  bool done;
  size_t blockSizeX; /**< Work-group size in x-direction */
  size_t blockSizeY; /**< Work-group size in y-direction */
  size_t bufferSize;
  cl_mem buffer;
  cl_mem clImage2D;
  cl_mem bufferImage;
  cl_mem bufferOut;
  cl_uint pitchAlignment;
};

#endif  // _OCLBufferFromImage_H_
