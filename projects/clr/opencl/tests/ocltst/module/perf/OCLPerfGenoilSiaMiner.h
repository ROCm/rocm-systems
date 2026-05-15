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

#ifndef _OCL_GenoilSiaMiner_H_
#define _OCL_GenoilSiaMiner_H_

#include "OCLTestImp.h"

class OCLPerfGenoilSiaMiner : public OCLTestImp {
 public:
  OCLPerfGenoilSiaMiner();
  virtual ~OCLPerfGenoilSiaMiner();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 1000;
  // 2^intensity hashes are calculated each time the kernel is called
  // Minimum of 2^8 (256) because our default local_item_size is 256
  // global_item_size (2^intensity) must be a multiple of local_item_size
  // Max of 2^32 so that people can't send an hour of work to the GPU at one
  // time
#define MIN_INTENSITY 8
#define MAX_INTENSITY 32
#define DEFAULT_INTENSITY 16

  // Number of times the GPU kernel is called between updating the command line
  // text
#define MIN_CPI 1      // Must do one call per update
#define MAX_CPI 65536  // 2^16 is a slightly arbitrary max
#define DEFAULT_CPI 30

  // The maximum size of the .cl file we read in and compile
#define MAX_SOURCE_SIZE (0x200000)

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_int error_;
  cl_program program_;
  cl_kernel kernel_;

  // mem objects for storing our kernel parameters
  cl_mem blockHeadermobj_ = NULL;
  cl_mem nonceOutmobj_ = NULL;

  // More gobal variables the grindNonce needs to access
  size_t local_item_size = 256;  // Size of local work groups. 256 is usually optimal
  unsigned int blocks_mined = 0;
  unsigned int intensity = DEFAULT_INTENSITY;
  unsigned cycles_per_iter = DEFAULT_CPI;

  bool isAMD;
  char platformVersion[32];
  void setHeader(uint32_t* ptr);
};

#endif  // _OCL_GenoilSiaMiner_H_
