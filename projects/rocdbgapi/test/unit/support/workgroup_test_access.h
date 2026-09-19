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
