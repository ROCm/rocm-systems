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

#include "callbacks.h"
#include "debug.h"
#include "exception.h"
#include "initialization.h"
#include "logging.h"
#include "process.h"
#include "utils.h"

#include <cinttypes>
#include <functional>

namespace amd::dbgapi
{

breakpoint_t::breakpoint_t (amd_dbgapi_breakpoint_id_t breakpoint_id,
                            process_t &process,
                            amd_dbgapi_global_address_t address,
                            action_callback_t action)
  : handle_object (breakpoint_id), m_address (address), m_action (action),
    m_process (process)
{
  amd_dbgapi_status_t status
    = m_process.insert_breakpoint (address, breakpoint_id);

  m_inserted = (status == AMD_DBGAPI_STATUS_SUCCESS);

  if (!m_inserted)
    warning ("Could not insert breakpoint at %#" PRIx64 " (%s)", address,
             to_cstring (status));
}

breakpoint_t::~breakpoint_t ()
{
  if (!m_inserted)
    return;

  amd_dbgapi_status_t status = process ().remove_breakpoint (id ());
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    warning ("remove_breakpoint failed (%s)", to_cstring (status));
}

void
breakpoint_t::get_info (amd_dbgapi_breakpoint_info_t query, size_t value_size,
                        void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_BREAKPOINT_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_report_breakpoint_hit (
  amd_dbgapi_breakpoint_id_t breakpoint_id,
  amd_dbgapi_client_thread_id_t client_thread_id,
  amd_dbgapi_breakpoint_action_t *breakpoint_action)
{
  TRACE_BEGIN (param_in (breakpoint_id), param_in (client_thread_id),
               param_in (breakpoint_action));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    if (breakpoint_action == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    breakpoint_t *breakpoint = find (breakpoint_id);

    if (breakpoint == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID);

    breakpoint->action () (*breakpoint, client_thread_id, breakpoint_action);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
  TRACE_END (make_ref (param_out (breakpoint_action)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_breakpoint_get_info (amd_dbgapi_breakpoint_id_t breakpoint_id,
                                amd_dbgapi_breakpoint_info_t query,
                                size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (breakpoint_id), param_in (query),
               param_in (value_size), param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    breakpoint_t *breakpoint = find (breakpoint_id);

    if (breakpoint == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID);

    breakpoint->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}
