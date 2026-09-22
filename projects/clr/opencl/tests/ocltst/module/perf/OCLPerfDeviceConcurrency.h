/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_Perf_DeviceConcurrency_H_
#define _OCL_Perf_DeviceConcurrency_H_

#include "OCLTestImp.h"

class OCLPerfDeviceConcurrency : public OCLTestImp {
 public:
  OCLPerfDeviceConcurrency();
  virtual ~OCLPerfDeviceConcurrency();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void setData(cl_mem buffer, unsigned int idx, unsigned int data);
  void checkData(cl_mem buffer, unsigned int idx);

#define MAX_DEVICES 16

  cl_context context_;
  cl_command_queue cmd_queue_[MAX_DEVICES];
  cl_program program_[MAX_DEVICES];
  cl_kernel kernel_[MAX_DEVICES];
  cl_mem outBuffer_[MAX_DEVICES];
  cl_int error_;

  cl_uint num_devices;
  cl_uint cur_devices;

  unsigned int width_;
  unsigned int bufSize_;
  unsigned int maxIter;
  unsigned int coordIdx;
  unsigned long long totalIters;
};

#endif  // _OCL_Perf_DeviceConcurrency_H_
