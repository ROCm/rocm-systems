/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_AES256_H_
#define _OCL_AES256_H_

#include "OCLTestImp.h"

class OCLPerfAES256 : public OCLTestImp {
 public:
  OCLPerfAES256();
  virtual ~OCLPerfAES256();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void setData(cl_mem buffer, unsigned int data);
  void checkData(cl_mem buffer);

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_mem inBuffer_;
  cl_mem outBuffer_;
  cl_mem tableBuffer_;
  cl_mem keyBuffer_;
  cl_int error_;

  unsigned int width_;
  unsigned int bufSize_;
  unsigned int blockSize_;
  unsigned int maxIterations;
  size_t numCUs;
};

#endif  // _OCL_AES256_H_
