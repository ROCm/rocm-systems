/* Copyright (c) 2021-2024 Advanced Micro Devices, Inc.

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

#ifndef AMD_DBGAPI_WATCHPOINT_H
#define AMD_DBGAPI_WATCHPOINT_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"
#include "os_driver.h"

#include <cstddef>
#include <optional>

namespace amd::dbgapi
{

class process_t;

/* AMD Debugger API Watchpoint.  */

class watchpoint_t : public detail::handle_object<amd_dbgapi_watchpoint_id_t>
{
private:
  const amd_dbgapi_global_address_t m_requested_address;
  const amd_dbgapi_size_t m_requested_size;
  const amd_dbgapi_watchpoint_kind_t m_kind;

  amd_dbgapi_global_address_t m_address;
  amd_dbgapi_size_t m_size;

  process_t &m_process;

public:
  watchpoint_t (amd_dbgapi_watchpoint_id_t watchpoint_id, process_t &process,
                amd_dbgapi_global_address_t requested_address,
                amd_dbgapi_size_t requested_size,
                amd_dbgapi_watchpoint_kind_t kind);
  ~watchpoint_t () {}

  amd_dbgapi_global_address_t address () const { return m_address; }
  amd_dbgapi_global_address_t size () const { return m_size; }

  amd_dbgapi_global_address_t requested_address () const
  {
    return m_requested_address;
  }
  amd_dbgapi_size_t requested_size () const { return m_requested_size; }

  amd_dbgapi_watchpoint_kind_t kind () const { return m_kind; }

  void get_info (amd_dbgapi_watchpoint_info_t query, size_t value_size,
                 void *value) const;

  process_t &process () const { return m_process; }
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_WATCHPOINT_H */
