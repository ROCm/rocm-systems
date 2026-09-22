/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef ROCSHMEM_IPC_TEST_CONFIG_HPP
#define ROCSHMEM_IPC_TEST_CONFIG_HPP

#include "../src/ipc_policy.hpp"
#include "../src/envvar.hpp"

namespace rocshmem {

namespace {
// Test-only helper to override cached envvar values before ipcHostInit.
template <typename T>
void override_envvar(const envvar::_detail::var<T>& var, T v) {
  auto& m = const_cast<envvar::_detail::var<T>&>(var);
  m.value = v;
  m.value_set = true;
}
}  // namespace

// Test configuration traits — bundle IPC implementation type + init hooks.
// preInit() overrides cached envvar values before ipcHostInit reads them.

struct IpcOnTestConfig {
  using impl_type = IpcOnImpl;
  static void preInit() {}
};

#if defined(USE_SDMA)
struct IpcSdmaTestConfig {
  using impl_type = IpcSdmaImpl;
  static void preInit() {
    override_envvar(envvar::sdma::enabled, true);
    override_envvar<size_t>(envvar::sdma::threshold, 1);
  }
};
#endif

}  // namespace rocshmem

#endif  // ROCSHMEM_IPC_TEST_CONFIG_HPP
