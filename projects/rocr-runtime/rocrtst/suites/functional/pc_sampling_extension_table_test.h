/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_PC_SAMPLING_EXTENSION_TABLE_TEST_H_
#define ROCRTST_SUITES_FUNCTIONAL_PC_SAMPLING_EXTENSION_TABLE_TEST_H_

#include "suites/test_common/test_base.h"

class PcSamplingExtensionTableTest : public TestBase {
 public:
  PcSamplingExtensionTableTest();
  virtual void Run();
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_PC_SAMPLING_EXTENSION_TABLE_TEST_H_
