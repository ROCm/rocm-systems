/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_LDS32K_H_
#define _OCL_LDS32K_H_
#include "OCLTestImp.h"

class OCLLDS32K : public OCLTestImp {
 public:
  OCLLDS32K();
  virtual ~OCLLDS32K();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
  void setup_run(const char* cmplr_opt);
  void cleanup_run();
  void exec_kernel(void* a_mem, void* b_mem, void* c_mem, void* d_mem, void* e_mem);
  static const char* kernel_src;
  cl_kernel kernel2_;

 private:
  unsigned int testID_;
  cl_mem a_buf_;
  cl_mem b_buf_;
  cl_mem c_buf_;
  cl_mem d_buf_;
  cl_mem e_buf_;
};

#endif  // _OCL_LDS32K_H_
