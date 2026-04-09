/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_SdiP2PCopy_H_
#define _OCL_SdiP2PCopy_H_

#include "OCLTestImp.h"

class OCLPerfSdiP2PCopy : public OCLTestImp {
 public:
  OCLPerfSdiP2PCopy();
  virtual ~OCLPerfSdiP2PCopy();
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  static const unsigned int NUM_ITER = 1024;
  bool silentFailure;
  cl_context contexts_[2];
  cl_device_id devices_[2];
  cl_command_queue cmd_queues_[2];
  cl_mem srcBuff_;
  cl_mem extPhysicalBuff_;
  cl_mem busAddressableBuff_;
  cl_int error_;
  cl_bus_address_amd busAddr_;
  cl_uint* inputArr_;
  cl_uint* outputArr_;
  unsigned int bufSize_;
  std::string deviceNames_;
};

#endif  // _OCL_SdiP2PCopy_H_
