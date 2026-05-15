/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_Perf_Concurrency_H_
#define _OCL_Perf_Concurrency_H_

#include "OCLTestImp.h"

class OCLPerfConcurrency : public OCLTestImp {
 public:
  OCLPerfConcurrency();
  virtual ~OCLPerfConcurrency();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void setData(cl_mem buffer, unsigned int data);
  void checkData(cl_mem buffer);

#define MAX_ASYNC_QUEUES 8

  cl_context context_;
  cl_command_queue cmd_queue_[MAX_ASYNC_QUEUES];
  cl_program program_[MAX_ASYNC_QUEUES];
  cl_kernel kernel_[MAX_ASYNC_QUEUES];
  cl_mem outBuffer_[MAX_ASYNC_QUEUES];
  cl_int error_;

  unsigned int num_cmd_queues;
  unsigned int num_programs;
  unsigned int num_kernels;
  unsigned int num_outbuffers;

  unsigned int width_;
  unsigned int bufSize_;
  unsigned int maxIter;
  unsigned int coordIdx;
  unsigned long long totalIters;
  size_t numCUs;
};

#endif  // _OCL_Perf_Concurrency_H_
