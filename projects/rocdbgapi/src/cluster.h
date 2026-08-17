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

#ifndef AMD_DBGAPI_CLUSTER_H
#define AMD_DBGAPI_CLUSTER_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"

#include <array>
#include <optional>

namespace amd::dbgapi
{

class dispatch_t;
class queue_t;
class agent_t;
class architecture_t;
class process_t;

/* AMD Debugger API Cluster.  */

class cluster_t : public detail::handle_object<amd_dbgapi_cluster_id_t>
{
private:
  /* The cluster ids are the coordinates within the grid.  */
  std::optional<const std::array<uint32_t, 3>> m_cluster_ids;

  /* Number of workgroups this cluster contains in the X, Y, Z dimensions.  */
  const std::array<uint32_t, 3> m_num_wgs;

  epoch_t m_mark{ 0 };

  const dispatch_t &m_dispatch;

public:
  cluster_t (amd_dbgapi_cluster_id_t cluster_id,
             const dispatch_t &dispatch,
             std::optional<const std::array<uint32_t, 3>> cluster_ids,
             const std::array<uint32_t, 3> num_wgs)
    : handle_object (cluster_id), m_cluster_ids (cluster_ids),
      m_num_wgs (num_wgs), m_dispatch (dispatch)
  {
  }

  const auto &cluster_ids () const { return m_cluster_ids; }
  const auto &num_wgs () const { return m_num_wgs; }

  epoch_t mark () const { return m_mark; }
  void set_mark (epoch_t mark) { m_mark = mark; }

  void get_info (amd_dbgapi_cluster_info_t query, size_t value_size,
                 void *value) const;

  const dispatch_t &dispatch () const { return m_dispatch; }
  queue_t &queue () const;
  const agent_t &agent () const;
  process_t &process () const;
  const architecture_t &architecture () const;
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_CLUSTER_H */
