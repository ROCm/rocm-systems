/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
