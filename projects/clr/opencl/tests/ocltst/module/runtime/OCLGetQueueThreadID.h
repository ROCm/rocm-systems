/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GET_QUEUE_THREAD_ID_H_
#define _OCL_GET_QUEUE_THREAD_ID_H_

#include "OCLTestImp.h"

class OCLGetQueueThreadID : public OCLTestImp {
 public:
  OCLGetQueueThreadID();
  virtual ~OCLGetQueueThreadID();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
};

#endif  // _OCL_GET_QUEUE_THREAD_ID_H_
