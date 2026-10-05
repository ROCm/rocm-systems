/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_LOADER_RESERVED_ADDRESS_H_
#define ROCRTST_SUITES_FUNCTIONAL_LOADER_RESERVED_ADDRESS_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ven_amd_loader.h"
#include "suites/test_common/test_base.h"

class LoaderReservedAddressTest : public TestBase {
 public:
  LoaderReservedAddressTest();

  virtual ~LoaderReservedAddressTest();

  virtual void SetUp();

  virtual void Run();

  virtual void Close();

  virtual void DisplayResults() const;

  virtual void DisplayTestInfo(void);

  // A normal allocation is rejected. An executable vmem mapping accepts the
  // image, and its bytes match a normal load.
  void LoadAtAddressTest(void);

 private:
  hsa_ven_amd_loader_1_04_pfn_t loader_ = {};
  hsa_code_object_reader_t reader_ = {0};
  int kernel_fd_ = -1;
  size_t page_size_ = 0;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_LOADER_RESERVED_ADDRESS_H_
