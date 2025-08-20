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

#ifndef _OCLTestImp_H_
#define _OCLTestImp_H_

#include <string>
#include <vector>

#include "BaseTestImp.h"
#include "CL/cl.h"
#include "OCL/Thread.h"
#include "OCLTest.h"
#include "OCLWrapper.h"

class OCLTestImp : public BaseTestImp {
 public:
  OCLTestImp();
  virtual ~OCLTestImp();

 public:
  //! Abstract functions being defined here
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId,
                    unsigned int platformIndex);
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void) = 0;
  virtual unsigned int close(void);
  //! Functions to set class members

 public:
  void useCPU();
  int genIntRand(int a, int b);
  int genBitRand(int n);
  void accumulateCRC(const void* buffer, int len);
  void setOCLWrapper(OCLWrapper* wrapper);
  OCLTestImp* toOCLTestImp() { return this; }

  static OCLutil::Lock openDeviceLock;
  static OCLutil::Lock compileLock;

 protected:
  const std::vector<cl_mem>& buffers() const { return buffers_; }

  OCLWrapper* _wrapper;

  int _seed;

  // Common data of any CL program
  cl_int error_;
  cl_uint type_;
  cl_uint deviceCount_;
  cl_device_id* devices_;
  cl_platform_id platform_;
  std::vector<cl_command_queue> cmdQueues_;
  cl_context context_;

  cl_program program_;
  cl_kernel kernel_;
  std::vector<cl_mem> buffers_;
};

// useful for initialization of an array of data types for a test
#define DTYPE(x, y) DataType(x, #x, (unsigned int)y)

#endif
