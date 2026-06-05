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

/* Minimal compute_queue_t / dispatch_t seams for unit tests that need a
   queue or a dispatch object but do not exercise queue-state machinery,
   wave save/restore, scratch refresh, or AQL packet decoding.

   compute_queue_t's production virtual queue_state_changed() walks the
   wave-save area, the cache, and the scratch ring; none of that exists
   in a unit-test TU.  test_compute_queue_t overrides it to a no-op so
   set_state(suspended) can be used to pre-suspend the queue without
   touching real hardware/driver state.

   dispatch_t has two pure virtuals (kernel_descriptor, get_info); the
   tests that need a dispatch only use it as an opaque parent of a
   workgroup_t / displaced_stepping_t, so both are stubbed with
   dbgapi_assert_not_reached.  */

#ifndef AMD_DBGAPI_TEST_UNIT_SUPPORT_TEST_QUEUE_H
#define AMD_DBGAPI_TEST_UNIT_SUPPORT_TEST_QUEUE_H 1

#include "agent.h"
#include "amd-dbgapi.h"
#include "architecture.h"
#include "debug.h"
#include "dispatch.h"
#include "os_driver.h"
#include "queue.h"

namespace amd::dbgapi::test
{

class test_compute_queue_t : public compute_queue_t
{
public:
  using compute_queue_t::compute_queue_t;

private:
  /* Production compute_queue_t::queue_state_changed reads the wave save
     area, walks the cache, and refreshes scratch.  None of that infra
     exists in a unit-test TU, so the test seam keeps the queue state
     mutation but does no work.  */
  void queue_state_changed () override {}
};

class test_dispatch_t : public dispatch_t
{
public:
  test_dispatch_t (amd_dbgapi_dispatch_id_t dispatch_id, compute_queue_t &queue,
                   amd_dbgapi_os_queue_packet_id_t os_queue_packet_id = 0)
    : dispatch_t (dispatch_id, queue, os_queue_packet_id)
  {
  }

  const architecture_t::kernel_descriptor_t &kernel_descriptor () const override
  {
    dbgapi_assert_not_reached (
      "test_dispatch_t::kernel_descriptor must not be called");
  }

  void get_info (amd_dbgapi_dispatch_info_t /* query  */,
                 size_t /* value_size  */,
                 void * /* value  */) const override
  {
    dbgapi_assert_not_reached ("test_dispatch_t::get_info must not be called");
  }
};

/* Build an os_queue_snapshot_entry_t that is "valid enough" for a
   test_compute_queue_t to exist and be set_state(suspended).  Reads/writes
   through the queue (ring_base_address, ctx_save_restore_address) are NOT
   exercised by tests using this helper; the values are placeholders.  */
inline os_queue_snapshot_entry_t
make_test_os_queue_info (os_queue_id_t queue_id = 1,
                         os_agent_id_t gpu_id = 42)
{
  os_queue_snapshot_entry_t info{};
  info.queue_id = queue_id;
  info.gpu_id = gpu_id;
  info.queue_type = os_queue_type_t::compute_aql;
  info.ring_base_address = host_address_t{ 0x1000ull };
  info.ring_size = 0x1000;
  info.ctx_save_restore_address = agent_address_t{ 0x2000ull };
  info.ctx_save_restore_area_size = 0x1000;
  info.compute_tmpring_size = 0;
  return info;
}

} /* namespace amd::dbgapi::test */

#endif /* AMD_DBGAPI_TEST_UNIT_SUPPORT_TEST_QUEUE_H */
