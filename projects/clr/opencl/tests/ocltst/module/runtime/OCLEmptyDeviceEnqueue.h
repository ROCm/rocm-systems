/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_EMPTY_DEVICE_ENQUEUE_H_
#define _OCL_EMPTY_DEVICE_ENQUEUE_H_

#include "OCLTestImp.h"

class OCLEmptyDeviceEnqueue : public OCLTestImp {
 public:
  OCLEmptyDeviceEnqueue();
  virtual ~OCLEmptyDeviceEnqueue();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  // Returns true if the built program's code-object metadata for devices_[deviceId]
  // contains the hidden device-enqueue argument markers.
  bool hasDeviceEnqueueMetadata(unsigned int deviceId);

  cl_command_queue deviceQueue_;
  bool skip_;
  unsigned int testID_;
};

#endif  // _OCL_EMPTY_DEVICE_ENQUEUE_H_
