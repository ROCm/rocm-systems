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

#ifndef _OCL_CREATE_BUFFER_H_
#define _OCL_CREATE_BUFFER_H_

#include "OCLTestImp.h"
#define PATTERN_20_08BIT 0x20
#define PATTERN_20_64BIT 0x2020202020202020
#define PATTERN_2A_08BIT 0x2a
#define PATTERN_2A_64BIT 0x2a2a2a2a2a2a2a2a

class OCLCreateBuffer : public OCLTestImp {
 public:
  OCLCreateBuffer();
  virtual ~OCLCreateBuffer();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual void writeBuffer(size_t tmpMaxSize, void* dataBuf);
  virtual void checkResult(size_t tmpMaxSize, void* resultBuf, cl_ulong pattern);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int testID_;
  cl_ulong maxSize_;
};

#endif  // _OCL_CREATE_BUFFER_H_
