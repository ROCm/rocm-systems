/* Copyright (c) 2022-2024 Advanced Micro Devices, Inc.

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

#ifndef AMD_DBGAPI_WORKGROUP_H
#define AMD_DBGAPI_WORKGROUP_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"

#include <optional>

namespace amd::dbgapi
{

class address_space_t;
class agent_t;
class architecture_t;
class compute_queue_t;
class dispatch_t;
class process_t;

/* AMD Debugger API Workgroup.  */

class workgroup_t : public detail::handle_object<amd_dbgapi_workgroup_id_t>
{
private:
  std::optional<const std::array<uint32_t, 3>> m_group_ids;
  epoch_t m_mark{ 0 };

  std::optional<amd_dbgapi_global_address_t> m_local_memory_base_address;
  amd_dbgapi_size_t const m_local_memory_size;

  const dispatch_t &m_dispatch;

  [[nodiscard]] size_t
  xfer_local_memory (const address_space_t &address_space,
                     amd_dbgapi_segment_address_t segment_address, void *read,
                     const void *write, size_t size);

public:
  workgroup_t (amd_dbgapi_workgroup_id_t workgroup_id,
               const dispatch_t &dispatch,
               std::optional<const std::array<uint32_t, 3>> group_ids,
               amd_dbgapi_size_t local_memory_size)
    : handle_object (workgroup_id), m_group_ids (group_ids),
      m_local_memory_size (local_memory_size), m_dispatch (dispatch)
  {
  }

  const auto &group_ids () const { return m_group_ids; }

  epoch_t mark () const { return m_mark; }
  void set_mark (epoch_t mark) { m_mark = mark; }

  void update (amd_dbgapi_global_address_t local_memory_base_address);

  [[nodiscard]] size_t
  xfer_segment_memory (const address_space_t &address_space,
                       amd_dbgapi_segment_address_t segment_address,
                       void *read, const void *write, size_t size);

  void get_info (amd_dbgapi_workgroup_info_t query, size_t value_size,
                 void *value) const;

  const dispatch_t &dispatch () const { return m_dispatch; }
  compute_queue_t &queue () const;
  const agent_t &agent () const;
  process_t &process () const;
  const architecture_t &architecture () const;
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_WORKGROUP_H */
