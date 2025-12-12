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

#include "code_object.h"
#include "debug.h"
#include "exception.h"
#include "initialization.h"
#include "logging.h"
#include "process.h"
#include "utils.h"

#include <vector>

namespace amd::dbgapi
{

void
code_object_t::get_info (amd_dbgapi_code_object_info_t query,
                         size_t value_size, void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_CODE_OBJECT_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;

    case AMD_DBGAPI_CODE_OBJECT_INFO_URI_NAME:
      utils::get_info (value_size, value, m_uri);
      return;

    case AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS:
      utils::get_info (value_size, value, m_load_address);
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_code_object_get_info (amd_dbgapi_code_object_id_t code_object_id,
                                 amd_dbgapi_code_object_info_t query,
                                 size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (code_object_id), param_in (query),
               param_in (value_size), param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    code_object_t *code_object = find (code_object_id);

    if (code_object == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_CODE_OBJECT_ID);

    code_object->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_CODE_OBJECT_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_process_code_object_list (
  amd_dbgapi_process_id_t process_id, size_t *code_object_count,
  amd_dbgapi_code_object_id_t **code_objects, amd_dbgapi_changed_t *changed)
{
  TRACE_BEGIN (param_in (process_id), param_in (code_object_count),
               param_in (code_objects), param_in (changed));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    std::vector<process_t *> processes = process_t::match (process_id);

    if (code_objects == nullptr || code_object_count == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    std::tie (*code_objects, *code_object_count)
      = utils::get_handle_list<code_object_t> (processes, changed);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (
    make_ref (param_out (code_object_count)),
    make_ref (make_ref (param_out (code_objects)), *code_object_count),
    make_ref (param_out (changed)));
}
