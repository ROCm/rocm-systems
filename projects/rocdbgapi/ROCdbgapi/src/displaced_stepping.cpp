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

#include "displaced_stepping.h"
#include "debug.h"
#include "exception.h"
#include "initialization.h"
#include "logging.h"
#include "process.h"
#include "queue.h"
#include "utils.h"
#include "wave.h"

#include <cinttypes>
#include <utility>

namespace amd::dbgapi
{

displaced_stepping_t::displaced_stepping_t (
  amd_dbgapi_displaced_stepping_id_t displaced_stepping_id, queue_t &queue,
  instruction_t original_instruction, amd_dbgapi_global_address_t from,
  std::optional<compute_queue_t::displaced_instruction_ptr_t> to)
  : handle_object (displaced_stepping_id),
    m_original_instruction (std::move (original_instruction)), m_from (from),
    m_to (std::move (to)), m_queue (queue)
{
  dbgapi_assert (m_original_instruction.is_valid ());

  log_info ("created new %s (from=%#" PRIx64 "%s)", to_cstring (id ()), m_from,
            m_to ? string_printf (", to=%#" PRIx64, m_to->get ()).c_str ()
                 : "");
}

displaced_stepping_t::~displaced_stepping_t ()
{
  dbgapi_assert (m_reference_count == 0
                 && "all displaced stepping operations should have completed");

  log_info ("destructed %s", to_cstring (id ()));
}

void
displaced_stepping_t::get_info (amd_dbgapi_displaced_stepping_info_t query,
                                size_t value_size, void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

void
displaced_stepping_t::retain (displaced_stepping_t *displaced_stepping)
{
  dbgapi_assert (displaced_stepping);

  [[maybe_unused]] auto prev_count = displaced_stepping->m_reference_count++;
  dbgapi_assert (displaced_stepping->m_reference_count > prev_count);
}

void
displaced_stepping_t::release (displaced_stepping_t *displaced_stepping)
{
  dbgapi_assert (displaced_stepping
                 && displaced_stepping->m_reference_count > 0);

  if (--displaced_stepping->m_reference_count == 0)
    displaced_stepping->process ().destroy (displaced_stepping);
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_displaced_stepping_start (
  amd_dbgapi_wave_id_t wave_id, const void *saved_instruction_bytes,
  amd_dbgapi_displaced_stepping_id_t *displaced_stepping_id)
{
  TRACE_BEGIN (param_in (wave_id),
               make_hex (make_ref (param_in (saved_instruction_bytes), 4)),
               param_in (displaced_stepping_id));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    if (saved_instruction_bytes == nullptr || displaced_stepping_id == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    if (wave->state () != AMD_DBGAPI_WAVE_STATE_STOP)
      THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);

    /* Already displaced stepping?  */
    if (wave->displaced_stepping () != nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_ACTIVE);

    wave->displaced_stepping_start (saved_instruction_bytes);

    *displaced_stepping_id = wave->displaced_stepping ()->id ();
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED,
         AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_ACTIVE,
         AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_BUFFER_NOT_AVAILABLE,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS,
         AMD_DBGAPI_STATUS_ERROR_ILLEGAL_INSTRUCTION);
  TRACE_END (make_ref (param_out (displaced_stepping_id)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_displaced_stepping_complete (
  amd_dbgapi_wave_id_t wave_id,
  amd_dbgapi_displaced_stepping_id_t displaced_stepping_id)
{
  TRACE_BEGIN (param_in (wave_id), param_in (displaced_stepping_id));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    if (wave->process ().is_frozen ())
      THROW (AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);

    if (wave->state () != AMD_DBGAPI_WAVE_STATE_STOP)
      THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);

    displaced_stepping_t *displaced_stepping = find (displaced_stepping_id);

    if (displaced_stepping == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID);

    /* Not displaced stepping or stepping with a different displaced stepping
       buffer?  */
    if (wave->displaced_stepping () != displaced_stepping)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

    wave->displaced_stepping_complete ();
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID,
         AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);
  TRACE_END ();
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_displaced_stepping_get_info (
  amd_dbgapi_displaced_stepping_id_t displaced_stepping_id,
  amd_dbgapi_displaced_stepping_info_t query, size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (displaced_stepping_id), param_in (query),
               param_in (value_size), param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    displaced_stepping_t *displaced_stepping = find (displaced_stepping_id);

    if (displaced_stepping == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID);

    displaced_stepping->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}
