// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <pthread.h>

#include <cerrno>
#include <thread>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"

TEST(SystemFunctionalReadOnly, ShutdownPreservesDeviceMutex) {
  const auto status = rsmi_init(RSMI_INIT_FLAG_THRAD_ONLY_MUTEX);
  if (status == RSMI_STATUS_NOT_SUPPORTED) {
    GTEST_SKIP() << "ROCm driver is unavailable";
  }
  ASSERT_EQ(RSMI_STATUS_SUCCESS, status);
  MAKE_NAMED_SCOPE_GUARD(shutdown, []() { rsmi_shut_down(); });

  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  if (devices.empty()) {
    GTEST_SKIP() << "No ROCm device mutex to exercise";
  }
  auto device = devices.front();
  auto* mutex = device->mutex();
  ASSERT_EQ(0, pthread_mutex_trylock(mutex));

  EXPECT_EQ(RSMI_STATUS_SUCCESS, rsmi_shut_down());
  shutdown.Dismiss();

  int lock_status = 0;
  std::thread contender([&]() {
    lock_status = pthread_mutex_trylock(mutex);
    if (lock_status == 0) {
      EXPECT_EQ(0, pthread_mutex_unlock(mutex));
    }
  });
  contender.join();

  EXPECT_EQ(EBUSY, lock_status);
  if (lock_status == EBUSY) {
    EXPECT_EQ(0, pthread_mutex_unlock(mutex));
  }
}
