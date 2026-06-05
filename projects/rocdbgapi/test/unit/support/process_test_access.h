// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

/* Friend-key trampoline that allows unit tests to construct process_t via
   the test-only constructor declared in src/process.h.

   process_t::test_access_key_t has a private default constructor and
   friends exactly one class — process_test_access — so only this header
   can mint a key.  Production code paths cannot reach the injecting
   constructor.  */

#ifndef AMD_DBGAPI_TEST_UNIT_SUPPORT_PROCESS_TEST_ACCESS_H
#define AMD_DBGAPI_TEST_UNIT_SUPPORT_PROCESS_TEST_ACCESS_H 1

#include "amd-dbgapi.h"
#include "os_driver.h"
#include "process.h"

#include <memory>
#include <optional>
#include <utility>

class process_test_access
{
public:
  /* Build a process_t bound to the supplied os_driver.  Skips the
     client_process_get_info() callback path and the client notifier
     pipe open() — see process_t's test ctor for the contract.  */
  static std::unique_ptr<amd::dbgapi::process_t>
  make (amd_dbgapi_process_id_t process_id,
        amd_dbgapi_client_process_id_t client_process_id,
        std::optional<amd_dbgapi_os_process_id_t> os_process_id,
        std::unique_ptr<amd::dbgapi::os_driver_t> os_driver)
  {
    return std::make_unique<amd::dbgapi::process_t> (
      amd::dbgapi::process_t::test_access_key_t{}, process_id,
      client_process_id, os_process_id, std::move (os_driver));
  }
};

#endif /* AMD_DBGAPI_TEST_UNIT_SUPPORT_PROCESS_TEST_ACCESS_H */
