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

#ifndef AMD_DBGAPI_CODE_OBJECT_H
#define AMD_DBGAPI_CODE_OBJECT_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"
#include "utils.h"

#include <cstddef>
#include <string>
#include <utility>

namespace amd::dbgapi
{

class process_t;

/* ROCm Code Object.  */

class code_object_t : public detail::handle_object<amd_dbgapi_code_object_id_t>
{
private:
  std::string const m_uri;
  amd_dbgapi_global_address_t const m_load_address;

  epoch_t m_mark{ 0 };

  process_t &m_process;

public:
  code_object_t (amd_dbgapi_code_object_id_t code_object_id,
                 process_t &process, std::string uri,
                 amd_dbgapi_global_address_t load_address)
    : handle_object (code_object_id), m_uri (std::move (uri)),
      m_load_address (load_address), m_process (process)
  {
  }

  amd_dbgapi_global_address_t load_address () const { return m_load_address; }
  const std::string &uri () const { return m_uri; }

  static epoch_t next_mark ()
  {
    static monotonic_counter_t<epoch_t, 1> next_code_object_mark{};
    return next_code_object_mark ();
  }
  epoch_t mark () const { return m_mark; }
  void set_mark (epoch_t mark) { m_mark = mark; }

  void get_info (amd_dbgapi_code_object_info_t query, size_t value_size,
                 void *value) const;

  process_t &process () const { return m_process; }
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_CODE_OBJECT_H */
