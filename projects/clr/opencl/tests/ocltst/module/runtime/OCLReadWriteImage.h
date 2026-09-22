/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_READ_WRITE_IMAGE_H_
#define _OCL_READ_WRITE_IMAGE_H_

#include "OCLTestImp.h"

class OCLReadWriteImage : public OCLTestImp {
 public:
  OCLReadWriteImage();
  virtual ~OCLReadWriteImage();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool done_;
  unsigned int testID_;
  size_t maxSize_;
  size_t imageWidth;
  size_t imageHeight;
  size_t imageDepth;
  size_t bufferSize;
  cl_sampler sampler;
  bool verifyImageData(unsigned char* inputImageData, unsigned char* output, size_t width,
                       size_t height);
};

#endif  // _OCL_READ_WRITE_IMAGE_H_
