// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

/* Friend trampoline to reach workgroup_t::xfer_local_memory from unit
   tests.  The production path is xfer_segment_memory -> lower() ->
   xfer_local_memory, and lower() on a local address space zero-extends
   the offset to 32 bits.  That truncation makes the offset+size
   overflow regression test (offset near UINT64_MAX) unreachable from
   xfer_segment_memory, hence this seam.  */

#ifndef AMD_DBGAPI_TEST_UNIT_SUPPORT_WORKGROUP_TEST_ACCESS_H
#define AMD_DBGAPI_TEST_UNIT_SUPPORT_WORKGROUP_TEST_ACCESS_H 1

#include "amd-dbgapi.h"
#include "memory.h"
#include "workgroup.h"

#include <cstddef>

namespace amd::dbgapi::test
{

struct workgroup_test_access
{
  static size_t xfer_local_memory (workgroup_t &wg,
                                   const address_space_t &address_space,
                                   amd_dbgapi_segment_address_t segment_address,
                                   void *read, const void *write, size_t size)
  {
    return wg.xfer_local_memory (address_space, segment_address, read, write,
                                 size);
  }
};

} /* namespace amd::dbgapi::test */

#endif /* AMD_DBGAPI_TEST_UNIT_SUPPORT_WORKGROUP_TEST_ACCESS_H */
