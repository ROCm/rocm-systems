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
