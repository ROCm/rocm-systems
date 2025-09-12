/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates. 
 * 
 * SPDX-License-Identifier: MIT
 */
 
#ifndef GET_INFO_H
#define GET_INFO_H

#include "samples/common/hsa_test.h"

class GetInfo : public HsaTest {
 public:
  GetInfo();
  ~GetInfo();

  void Run() override;
};

#endif // GET_INFO_H
