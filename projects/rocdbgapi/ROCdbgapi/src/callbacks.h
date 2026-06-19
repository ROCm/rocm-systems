/* Copyright (c) 2019-2024 Advanced Micro Devices, Inc.

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

#ifndef AMD_DBGAPI_CALLBACKS_H
#define AMD_DBGAPI_CALLBACKS_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"

#include <cstddef>
#include <functional>

namespace amd::dbgapi
{

class process_t;

class breakpoint_t : public detail::handle_object<amd_dbgapi_breakpoint_id_t>
{
private:
  using action_callback_t
    = std::function<void (breakpoint_t &, amd_dbgapi_client_thread_id_t,
                          amd_dbgapi_breakpoint_action_t *)>;

  bool m_inserted{ false };

  amd_dbgapi_global_address_t const m_address;
  action_callback_t const m_action;

  process_t &m_process;

public:
  breakpoint_t (amd_dbgapi_breakpoint_id_t breakpoint_id, process_t &process,
                amd_dbgapi_global_address_t address, action_callback_t action);

  ~breakpoint_t ();

  bool is_inserted () const { return m_inserted; }

  amd_dbgapi_global_address_t address () const { return m_address; }
  action_callback_t action () const { return m_action; }

  void get_info (amd_dbgapi_breakpoint_info_t query, size_t value_size,
                 void *value) const;

  process_t &process () const { return m_process; }
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_CALLBACKS_H */
