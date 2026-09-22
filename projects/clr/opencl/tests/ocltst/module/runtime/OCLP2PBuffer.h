/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
