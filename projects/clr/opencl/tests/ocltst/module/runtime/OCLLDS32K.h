/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
