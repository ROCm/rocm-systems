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

#include "cluster.h"
#include "agent.h"
#include "dispatch.h"
#include "process.h"
#include "queue.h"

namespace amd::dbgapi
{

queue_t &
cluster_t::queue () const
{
  return dispatch ().queue ();
}

const agent_t &
cluster_t::agent () const
{
  return queue ().agent ();
}

process_t &
cluster_t::process () const
{
  return agent ().process ();
}

const architecture_t &
cluster_t::architecture () const
{
  return queue ().architecture ();
}

void
cluster_t::get_info (amd_dbgapi_cluster_info_t query, size_t value_size,
                     void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_CLUSTER_INFO_DISPATCH:
      utils::get_info (value_size, value, dispatch ().id ());
      return;

    case AMD_DBGAPI_CLUSTER_INFO_QUEUE:
      utils::get_info (value_size, value, queue ().id ());
      return;

    case AMD_DBGAPI_CLUSTER_INFO_AGENT:
      utils::get_info (value_size, value, agent ().id ());
      return;

    case AMD_DBGAPI_CLUSTER_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;

    case AMD_DBGAPI_CLUSTER_INFO_ARCHITECTURE:
      utils::get_info (value_size, value, architecture ().id ());
      return;

    case AMD_DBGAPI_CLUSTER_INFO_CLUSTER_COORD:
      {
        auto ids = cluster_ids ();
        if (!ids.has_value ())
          THROW (AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
        utils::get_info (value_size, value, *ids);
        return;
      }
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_cluster_get_info (amd_dbgapi_cluster_id_t cluster_id,
                             amd_dbgapi_cluster_info_t query,
                             size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (cluster_id), param_in (query),
               param_in (value_size), param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    cluster_t *cluster = find (cluster_id);

    if (cluster == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_CLUSTER_ID);

    cluster->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_CLUSTER_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
  TRACE_END (make_query_ref (query, param_out (value)));
}
