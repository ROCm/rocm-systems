/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_UncoalescedRead_H_
#define _OCL_UncoalescedRead_H_

#include "OCLTestImp.h"
#define NUM_READS 32
class OCLPerfUncoalescedRead : public OCLTestImp {
 public:
  OCLPerfUncoalescedRead();
  virtual ~OCLPerfUncoalescedRead();
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  static const unsigned int NUM_ITER = 1000;
  static const unsigned int SIZE = 250000;
  static const char* kernel_str;
  bool silentFailure;
  float* input_buff;
  void validate(void);
};

#endif  // _OCL_UncoalescedRead_H_
