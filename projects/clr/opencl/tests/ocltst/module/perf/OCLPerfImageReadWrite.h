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

#ifndef _OCL_ImageReadWrite
#define _OCL_ImageReadWrite

#include "OCLTestImp.h"

class OCLPerfImageReadWrite : public OCLTestImp {
 public:
  OCLPerfImageReadWrite();
  virtual ~OCLPerfImageReadWrite();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
  virtual void setData(void* ptr, unsigned int size, unsigned int value);

  cl_command_queue cmd_queue_;
  cl_mem imageBuffer_;

  unsigned int bufSize_;
  unsigned int bufnum_;
  unsigned int numIter;
  char* memptr;
  unsigned int memSize;
  unsigned int testId_;

  bool skip_;
};

#endif  // _OCL_ImageReadWrite
