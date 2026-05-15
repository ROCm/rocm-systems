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
