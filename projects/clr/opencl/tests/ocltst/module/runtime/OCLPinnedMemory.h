/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PINNED_MEMORY_H_
#define _OCL_PINNED_MEMORY_H_

#include <cstdint>

#include "OCLTestImp.h"

class OCLPinnedMemory : public OCLTestImp {
 public:
  OCLPinnedMemory();
  ~OCLPinnedMemory();

  void open(unsigned int test, char* units, double& conversion, unsigned int deviceId) override;
  void run() override;
  unsigned int close() override;

 private:
  void runNoPrepinnedMemory();
  void runPrepinnedMemory();

  static constexpr const float ratio_ = 0.4f;
  using row_data_t = uint64_t;

  row_data_t* host_memory_;
  size_t row_data_size_ = sizeof(row_data_t);
  size_t row_size_;
  size_t pin_size_;
};

#endif  // _OCL_PINNED_MEMORY_H_
