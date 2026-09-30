/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
