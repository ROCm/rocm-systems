/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_OCLSDI_H_
#define _OCL_OCLSDI_H_
#include <string>

#include "OCLTestImp.h"

class OCLSDI : public OCLTestImp {
 public:
  OCLSDI();
  virtual ~OCLSDI();
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
  void threadEntry(int threadID);

 private:
  void testEnqueueWriteBuffer(int threadID);
  void testEnqueueCopyBuffer(int threadID);
  void testEnqueueNDRangeKernel(int threadID);
  void testEnqueueMapBuffer(int threadID);
  void testEnqueueWriteBufferRect(int threadID);
  void testEnqueueCopyImageToBuffer(int threadID);
  void readAndVerifyResult();

  bool silentFailure;
  cl_context contexts_[2];
  cl_device_id devices_[2];
  cl_command_queue cmd_queues_[2];
  cl_mem extPhysicalBuff_;
  cl_mem busAddressableBuff_;
  cl_int error_;
  cl_bus_address_amd busAddr_;
  cl_uint* inputArr_;
  cl_uint* outputArr_;
  unsigned int bufSize_;
  bool success_;
  cl_uint markerValue_;
  cl_mem srcBuff_;
  cl_program program_;
  cl_kernel kernel_;
  cl_mem image_;
  std::string deviceNames_;
};
#endif  // _OCL_OCLSDI_H_
