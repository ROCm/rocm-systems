/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_USER_EVENT_FAILURE_H_
#define _OCL_USER_EVENT_FAILURE_H_

#include "OCLTestImp.h"

class OCLUserEventFailure : public OCLTestImp {
 public:
  OCLUserEventFailure();
  virtual ~OCLUserEventFailure();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  void failSingleEvent();
  void failOneOfTwoEvents();

  unsigned int testID_;
};

#endif  // _OCL_USER_EVENT_FAILURE_H_
