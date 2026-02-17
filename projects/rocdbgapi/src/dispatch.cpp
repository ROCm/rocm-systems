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

#include "dispatch.h"
#include "agent.h"
#include "architecture.h"
#include "debug.h"
#include "exception.h"
#include "initialization.h"
#include "logging.h"
#include "process.h"
#include "queue.h"
#include "utils.h"

#include <utility>
#include <vector>

namespace amd::dbgapi
{

const agent_t &
dispatch_t::agent () const
{
  return queue ().agent ();
}

process_t &
dispatch_t::process () const
{
  return agent ().process ();
}

const architecture_t &
dispatch_t::architecture () const
{
  return queue ().architecture ();
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_dispatch_get_info (amd_dbgapi_dispatch_id_t dispatch_id,
                              amd_dbgapi_dispatch_info_t query,
                              size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (dispatch_id), param_in (query), param_in (value_size),
               param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    dispatch_t *dispatch = find (dispatch_id);

    if (dispatch == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_DISPATCH_ID);

    dispatch->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_DISPATCH_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_process_dispatch_list (amd_dbgapi_process_id_t process_id,
                                  size_t *dispatch_count,
                                  amd_dbgapi_dispatch_id_t **dispatches,
                                  amd_dbgapi_changed_t *changed)
{
  TRACE_BEGIN (param_in (process_id), param_in (dispatch_count),
               param_in (dispatches), param_in (changed));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    std::vector<process_t *> processes = process_t::match (process_id);

    if (dispatches == nullptr || dispatch_count == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    std::vector<std::pair<process_t *, std::vector<queue_t *>>>
      queues_needing_resume;

    for (auto &&process : processes)
      {
        process->update_queues ();

        std::vector<queue_t *> queues;
        for (auto &&queue : process->range<queue_t> ())
          if (!queue.is_suspended ())
            queues.emplace_back (&queue);

        process->suspend_queues (queues, "refresh dispatch list");

        if (process->forward_progress_needed ())
          queues_needing_resume.emplace_back (process, std::move (queues));
      }

    amd_dbgapi_changed_t dispatch_list_changed;
    auto dispatch_list = utils::get_handle_list<dispatch_t> (
      processes, changed != nullptr ? &dispatch_list_changed : nullptr);

    auto deallocate_dispatch_list = utils::make_scope_fail (
      [&] () { amd::dbgapi::deallocate_memory (dispatches); });

    for (auto &&[process, queues] : queues_needing_resume)
      process->resume_queues (queues, "refresh dispatch list");

    std::tie (*dispatches, *dispatch_count) = dispatch_list;
    if (changed != nullptr)
      *changed = dispatch_list_changed;
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_ref (param_out (dispatch_count)),
             make_ref (make_ref (param_out (dispatches)), *dispatch_count),
             make_ref (param_out (changed)));
}
