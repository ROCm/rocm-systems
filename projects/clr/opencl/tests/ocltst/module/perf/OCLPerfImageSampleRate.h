/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_IMAGESAMPLERATE_H_
#define _OCL_IMAGESAMPLERATE_H_

#include "OCLTestImp.h"

class OCLPerfImageSampleRate : public OCLTestImp {
 public:
  OCLPerfImageSampleRate();
  virtual ~OCLPerfImageSampleRate();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void setData(cl_mem buffer, unsigned int data);
  void checkData(cl_mem buffer);
  void setKernel(void);

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_mem* inBuffer_;
  cl_mem outBuffer_;
  cl_int error_;

  unsigned int width_;
  unsigned int outBufWidth_;
  unsigned int outBufSize_;
  static const unsigned int MAX_ITERATIONS = 25;
  unsigned int numBufs_;
  unsigned int typeIdx_;
  bool skip_;
};

#endif  // _OCL_IMAGESAMPLERATE_H_
