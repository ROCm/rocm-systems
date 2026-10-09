/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SIGNAL_CREATE_V2_H_
#define ROCRTST_SUITES_FUNCTIONAL_SIGNAL_CREATE_V2_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/test_common/test_base.h"

class SignalCreateV2Test : public TestBase {
 public:
  SignalCreateV2Test();
  virtual ~SignalCreateV2Test();
  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  // Each documented rejection returns its status and leaves a zero handle.
  void TestRejections(void);
  // A batch with one bad descriptor returns the first error; the rest are created.
  void TestPartialBatch(void);
  // A device resident signal supports loads and plain stores, and refuses a value pointer.
  void TestDeviceResident(void);

 private:
  bool DeviceResidentSupported();
  hsa_amd_signal_create_desc_t DeviceDesc();
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SIGNAL_CREATE_V2_H_
