/* Copyright (c) 2026 Advanced Micro Devices, Inc.

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

/* GMock-generated mock for amd::dbgapi::os_driver_t.

   Use together with process_test_access (see process_test_access.h) to
   inject the mock into a process_t via the test-only friend-key ctor.

   Typical usage:

     auto driver = std::make_unique<NiceMock<MockOsDriver>>();
     EXPECT_CALL (*driver, agent_snapshot (...))
       .WillOnce (Return (AMD_DBGAPI_STATUS_SUCCESS));
     auto proc = process_test_access::make (
       amd_dbgapi_process_id_t{ 1 }, fake_client_id, std::nullopt,
       std::move (driver));

   NiceMock<...> is recommended over a bare MockOsDriver because os_driver_t
   exposes 22 virtual methods and most tests only care about a few; without
   NiceMock, every uncalled method produces an "uninteresting call" warning
   that drowns out real failures.  */

#ifndef AMD_DBGAPI_TEST_UNIT_SUPPORT_MOCK_OS_DRIVER_H
#define AMD_DBGAPI_TEST_UNIT_SUPPORT_MOCK_OS_DRIVER_H 1

#include "amd-dbgapi.h"
#include "memory.h"
#include "os_driver.h"

#include <gmock/gmock.h>

#include <cstddef>
#include <optional>

namespace amd::dbgapi::test
{

class MockOsDriver : public os_driver_t
{
public:
  /* Default to no os_pid (matches "no real process attached").  Tests
     that need a specific pid pass it explicitly.  */
  explicit MockOsDriver (
    std::optional<amd_dbgapi_os_process_id_t> os_pid = std::nullopt)
    : os_driver_t (os_pid)
  {
  }

  MOCK_METHOD (bool, is_valid, (), (const, override));
  MOCK_METHOD (amd_dbgapi_status_t, check_version, (), (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, create_core_state_note,
               (const os_runtime_info_t &runtime_info,
                amd_dbgapi_core_state_data_t *data),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, agent_snapshot,
               (os_agent_info_t * snapshots, size_t snapshot_count,
                size_t *agent_count, os_exception_mask_t exceptions_cleared),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, enable_debug,
               (os_exception_mask_t exceptions_reported,
                amd_dbgapi_notifier_t notifier,
                os_runtime_info_t *runtime_info),
               (override));
  MOCK_METHOD (amd_dbgapi_status_t, disable_debug, (), (override));
  MOCK_METHOD (bool, is_debug_enabled, (), (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, set_exceptions_reported,
               (os_exception_mask_t exceptions_reported), (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, send_exceptions,
               (os_exception_mask_t exceptions,
                std::optional<os_agent_id_t> agent_id,
                std::optional<os_queue_id_t> queue_id),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, query_debug_event,
               (os_exception_mask_t * exceptions_present,
                os_queue_id_t *os_queue_id, os_agent_id_t *os_agent_id,
                os_exception_mask_t exceptions_cleared),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, query_exception_info,
               (os_exception_code_t exception, os_source_id_t os_source_id,
                os_exception_info_t *os_exception_info, bool clear_exception),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, suspend_queues,
               (const os_queue_id_t *queues, size_t queue_count,
                os_exception_mask_t exceptions_cleared, size_t *suspended_count,
                os_queue_state_t *queue_states),
               (const, override));
  MOCK_METHOD (amd_dbgapi_status_t, resume_queues,
               (const os_queue_id_t *queues, size_t queue_count,
                size_t *resumed_count, os_queue_state_t *queue_states),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, queue_snapshot,
               (os_queue_snapshot_entry_t * snapshots, size_t snapshot_count,
                size_t *queue_count, os_exception_mask_t exceptions_cleared),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, set_address_watch,
               (os_agent_id_t os_agent_id, agent_address_t address,
                agent_address_t mask, os_watch_mode_t os_watch_mode,
                os_watch_id_t *os_watch_id),
               (const, override));
  MOCK_METHOD (amd_dbgapi_status_t, clear_address_watch,
               (os_agent_id_t os_agent_id, os_watch_id_t os_watch_id),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, set_wave_launch_mode,
               (os_wave_launch_mode_t mode), (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, set_wave_launch_trap_override,
               (os_wave_launch_trap_override_t override_,
                os_wave_launch_trap_mask_t value,
                os_wave_launch_trap_mask_t mask,
                os_wave_launch_trap_mask_t *previous_value,
                os_wave_launch_trap_mask_t *supported_mask),
               (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, set_process_flags,
               (os_process_flags_t flags), (const, override));

  MOCK_METHOD (amd_dbgapi_status_t, xfer_global_memory_partial,
               (global_address_t address, void *read, const void *write,
                size_t *size),
               (const, override));
  MOCK_METHOD (amd_dbgapi_status_t, xfer_host_memory_partial,
               (host_address_t address, void *read, const void *write,
                size_t *size),
               (const, override));
  MOCK_METHOD (amd_dbgapi_status_t, xfer_agent_memory_partial,
               (os_agent_id_t agent, agent_address_t address, void *read,
                const void *write, size_t *size),
               (const, override));
};

} /* namespace amd::dbgapi::test */

#endif /* AMD_DBGAPI_TEST_UNIT_SUPPORT_MOCK_OS_DRIVER_H */
