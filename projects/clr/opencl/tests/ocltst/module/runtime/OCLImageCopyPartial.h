/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_ImageCopyCorners_H_
#define _OCL_ImageCopyCorners_H_

#include "OCLTestImp.h"

class OCLImageCopyPartial : public OCLTestImp {
 public:
  OCLImageCopyPartial();
  virtual ~OCLImageCopyPartial();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 1;

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_mem srcBuffer_;
  cl_mem dstBuffer_;
  cl_int error_;

  unsigned int bufSizeW_;
  unsigned int bufSizeH_;
  unsigned int bufnum_;
  bool srcImage_;
  bool dstImage_;
  unsigned int numIter;
  void setData(void* ptr, unsigned int pitch, unsigned int size, unsigned int value);
  void checkData(void* ptr, unsigned int pitch, unsigned int size, unsigned int value);
};

#endif  // _OCL_ImageCopyPartial_H_
