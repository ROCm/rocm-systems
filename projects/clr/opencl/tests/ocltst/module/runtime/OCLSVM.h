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

#ifndef _OCL_SVM_H_
#define _OCL_SVM_H_

#include <CL/cl.h>

#include "OCLTestImp.h"
#include "stdint.h"

class OCLSVM : public OCLTestImp {
 public:
  OCLSVM();

  virtual ~OCLSVM();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);

  virtual void run(void);

  virtual unsigned int close(void);

 private:
  void runFineGrainedBuffer();
  void runFineGrainedSystem();
  void runFineGrainedSystemLargeAllocations();
  void runLinkedListSearchUsingFineGrainedSystem();
  void runPlatformAtomics();
  void runEnqueueOperations();
  void runSvmArgumentsAreRecognized();
  void runSvmCommandsExecutedInOrder();
  void runIdentifySvmBuffers();
  cl_bool isOpenClSvmAvailable(cl_device_id device_id);

  uint64_t svmCaps_;
};

struct Node {
  Node(uint64_t value, Node* next) : value_(value), next_((uint64_t)next) {}

  uint64_t value_;
  uint64_t next_;
};

#endif  // _OCL_SVM_H_
