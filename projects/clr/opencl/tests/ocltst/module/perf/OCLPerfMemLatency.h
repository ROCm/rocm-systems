/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_MEMLATENCY_H_
#define _OCL_MEMLATENCY_H_

#include "OCLTestImp.h"

class OCLPerfMemLatency : public OCLTestImp {
 public:
  OCLPerfMemLatency();
  virtual ~OCLPerfMemLatency();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void genShader(void);
  void setData(cl_mem buffer, unsigned int data);
  void checkData(cl_mem buffer);

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_kernel kernel2_;
  cl_mem inBuffer_;
  cl_mem outBuffer_;
  cl_int error_;

  unsigned int width_;
  unsigned int bufSizeDW_;
  unsigned int repeats_;
  unsigned int maxSize_;
  bool isAMD_;
  bool moreThreads;
  bool makeVolatile;
  bool makeRW;
};

#endif  // _OCL_MEMLATENCY_H_
