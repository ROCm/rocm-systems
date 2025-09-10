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

#ifndef _OCL_P2P_BUFFER_H_
#define _OCL_P2P_BUFFER_H_

#include "OCLTestImp.h"

class OCLP2PBuffer : public OCLTestImp {
 public:
  OCLP2PBuffer();
  virtual ~OCLP2PBuffer();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int testID_;
  cl_ulong maxSize_;
  size_t BufferSize;
  int NumChunks;
  int NumIter;
  int NumStages;
  cl_context context0_;
  cl_context context1_;
  cl_command_queue cmdQueue0_;
  cl_command_queue cmdQueue1_;
  cl_uint num_p2p_0_;
  cl_uint num_p2p_1_;
#ifdef CL_VERSION_2_0
  clEnqueueCopyBufferP2PAMD_fn p2p_copy_;
#endif
};

#endif  // _OCL_P2P_BUFFER_H_
