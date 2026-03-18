/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include "OCLTestImp.h"

class OCLPerfVerticalFetch : public OCLTestImp {
 public:
  OCLPerfVerticalFetch();
  virtual ~OCLPerfVerticalFetch();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  cl_mem srcBuffer_;
  cl_mem dstBuffer_;
  unsigned int nWorkItems;  // number of GPU work items
  unsigned int wgs;         // work group size
  unsigned int nBytes;      // input and output buffer size
  unsigned int nIter;       // overall number of timing loops
  cl_uint inputData;        // input data to fill the input buffer
  bool skip_;
  void* ptr_;
  const char* mem_type_;
  cl_uint dim;
  size_t gws[3];
  size_t lws[3];
  cl_uint numCachedPixels_;
};
