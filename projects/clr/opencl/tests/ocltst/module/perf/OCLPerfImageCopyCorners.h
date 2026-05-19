/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_ImageCopyCorners_H_
#define _OCL_ImageCopyCorners_H_

#include "OCLTestImp.h"

class OCLPerfImageCopyCorners : public OCLTestImp {
 public:
  OCLPerfImageCopyCorners();
  virtual ~OCLPerfImageCopyCorners();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 10;

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_mem srcBuffer_;
  cl_mem dstBuffer_;
  cl_int error_;
  bool skip_;

  unsigned int bufSizeW_;
  unsigned int bufSizeH_;
  unsigned int bufnum_;
  bool srcImage_;
  bool dstImage_;
  unsigned int numIter;
  void setData(void* ptr, unsigned int pitch, unsigned int size);
  void checkData(void* ptr, unsigned int pitch, unsigned int size);
};

#endif  // _OCL_ImageCopyCorners_H_
