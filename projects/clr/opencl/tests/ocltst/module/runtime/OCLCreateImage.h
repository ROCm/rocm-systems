/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_CREATE_IMAGE_H_
#define _OCL_CREATE_IMAGE_H_

#include "OCLTestImp.h"

class OCLCreateImage : public OCLTestImp {
 public:
  OCLCreateImage();
  virtual ~OCLCreateImage();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool done_;
  unsigned int testID_;
  size_t maxSize_;
  size_t ImageSizeX;
  size_t ImageSizeY;
  size_t ImageSizeZ;

  bool is64BitApp() { return sizeof(int*) == 8; }
};

#endif  // _OCL_CREATE_IMAGE_H_
