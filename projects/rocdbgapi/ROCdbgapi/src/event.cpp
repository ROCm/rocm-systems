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

#include "event.h"
#include "debug.h"
#include "exception.h"
#include "handle_object.h"
#include "initialization.h"
#include "logging.h"
#include "process.h"
#include "queue.h"
#include "utils.h"
#include "wave.h"

#include <cinttypes>
#include <string>
#include <vector>

namespace amd::dbgapi
{

/* Breakpoint resume event.  */
event_t::event_t (amd_dbgapi_event_id_t event_id, process_t &process,
                  amd_dbgapi_event_kind_t event_kind,
                  amd_dbgapi_breakpoint_id_t breakpoint_id,
                  amd_dbgapi_client_thread_id_t client_thread_id)
  : event_t (event_id, process, event_kind,
             breakpoint_resume_event_t{ breakpoint_id, client_thread_id })
{
  dbgapi_assert (event_kind == AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME
                 && "check event kind");
}

/* Code object list updated event.  */
event_t::event_t (amd_dbgapi_event_id_t event_id, process_t &process,
                  amd_dbgapi_event_kind_t event_kind,
                  amd_dbgapi_event_id_t breakpoint_resume_event_id)
  : event_t (event_id, process, event_kind,
             code_object_list_updated_event_t{ breakpoint_resume_event_id })
{
  dbgapi_assert (event_kind == AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED
                 && "check event kind");
}

/* Runtime event. */
event_t::event_t (amd_dbgapi_event_id_t event_id, process_t &process,
                  amd_dbgapi_event_kind_t event_kind,
                  amd_dbgapi_runtime_state_t runtime_state)
  : event_t (
    event_id, process, event_kind,
    runtime_event_t{
      runtime_state,
      process.is_flag_set (process_t::flag_t::runtime_enable_during_attach) })
{
  dbgapi_assert (event_kind == AMD_DBGAPI_EVENT_KIND_RUNTIME
                 && "check event kind");
}

/* Wave stop / command terminated event.  */
event_t::event_t (amd_dbgapi_event_id_t event_id, process_t &process,
                  amd_dbgapi_event_kind_t event_kind,
                  amd_dbgapi_wave_id_t wave_id)
  : event_t (event_id, process, event_kind, wave_event_t{ wave_id })
{
  dbgapi_assert (
    (event_kind == AMD_DBGAPI_EVENT_KIND_WAVE_STOP
     || event_kind == AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED)
    && "check event kind");
}

std::string
event_t::pretty_printer_string () const
{
  switch (kind ())
    {
    case AMD_DBGAPI_EVENT_KIND_NONE:
      return "null event";

    case AMD_DBGAPI_EVENT_KIND_WAVE_STOP:
    case AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED:
      {
        wave_t *wave
          = process ().find (std::get<wave_event_t> (m_data).wave_id);
        if (wave == nullptr)
          return string_printf (
            "%s for terminated %s", to_cstring (kind ()),
            to_cstring (std::get<wave_event_t> (m_data).wave_id));

        std::string stop_reason_str;
        if (kind () == AMD_DBGAPI_EVENT_KIND_WAVE_STOP)
          stop_reason_str = string_printf (", stop_reason=%s",
                                           to_cstring (wave->stop_reason ()));

        return string_printf ("%s for %s on %s (pc=%#" PRIx64,
                              to_cstring (kind ()), to_cstring (wave->id ()),
                              to_cstring (wave->queue ().id ()), wave->pc ())
               + stop_reason_str + ")";
      }

    case AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED:
      return string_printf (
        "EVENT_KIND_CODE_OBJECT_LIST_UPDATED, resumed with %s",
        to_cstring (std::get<code_object_list_updated_event_t> (m_data)
                      .breakpoint_resume_event_id));

    case AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME:
      return string_printf (
        "EVENT_KIND_BREAKPOINT_RESUME for %s",
        to_cstring (
          std::get<breakpoint_resume_event_t> (m_data).breakpoint_id));

    case AMD_DBGAPI_EVENT_KIND_RUNTIME:
      return string_printf (
        "EVENT_KIND_RUNTIME state=%s",
        to_cstring (std::get<runtime_event_t> (m_data).runtime_state));

    case AMD_DBGAPI_EVENT_KIND_QUEUE_ERROR:
    default:
      return "unhandled event";
    }
}

void
event_t::set_state (state_t state)
{
  /* An event's state can only progress, the last state being 'processed', at
     which point the event may be destroyed.  */
  dbgapi_assert (state > m_state);

  if (state == state_t::processed && kind () == AMD_DBGAPI_EVENT_KIND_RUNTIME)
    {
      if (!std::get<runtime_event_t> (m_data).runtime_enable_during_attach)
        process ().send_exceptions (os_exception_mask_t::process_runtime,
                                    &process ());
    }
  else if (state == state_t::processed
           && kind () == AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED)
    {
      amd_dbgapi_event_id_t event_id
        = std::get<code_object_list_updated_event_t> (m_data)
            .breakpoint_resume_event_id;

      /* Code object updates that are not initiated by the r_brk breakpoint
         callback do not have a breakpoint resume event.  */
      if (event_id != AMD_DBGAPI_EVENT_NONE)
        {
          process ().update_queues ();

          std::vector<queue_t *> queues;
          queues.reserve (process ().count<queue_t> ());

          for (auto &&queue : process ().range<queue_t> ())
            if (!queue.is_suspended ())
              queues.emplace_back (&queue);

          /* FIXME: A breakpoint may have been inserted by the client prior to
             reporting this event as processed.

             Currently, KFD only flushes the L2$ and I$ during CWSRs, so simply
             writting the breakpoint instruction to memory does not guarantee
             visibility from the device side, stale data could still be in I$
             or L2$.

             As a workaround, we can force a CWSR, before allowing the host
             thread to resume execution, by suspending then resuming the
             queues.

             The ROCr runtime creates an internal queue to run the blit kernels
             so, after loading a device code object, we should always have at
             least one queue on each device.  Suspending that queue to update
             the wave list causes the caches to be flushed.  */

          process ().suspend_queues (queues, "code object list updated");
          if (process ().forward_progress_needed ())
            process ().resume_queues (queues, "code object list updated");

          event_t *breakpoint_resume_event = process ().find (event_id);
          dbgapi_assert (breakpoint_resume_event);
          process ().enqueue_event (*breakpoint_resume_event);
        }
    }

  m_state = state;
}

void
event_t::get_info (amd_dbgapi_event_info_t query, size_t value_size,
                   void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_EVENT_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;

    case AMD_DBGAPI_EVENT_INFO_KIND:
      utils::get_info (value_size, value, m_event_kind);
      return;

    case AMD_DBGAPI_EVENT_INFO_WAVE:
      if (kind () != AMD_DBGAPI_EVENT_KIND_WAVE_STOP
          && kind () != AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

      utils::get_info (value_size, value,
                       std::get<wave_event_t> (m_data).wave_id);
      return;

    case AMD_DBGAPI_EVENT_INFO_BREAKPOINT:
      if (kind () != AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

      utils::get_info (
        value_size, value,
        std::get<breakpoint_resume_event_t> (m_data).breakpoint_id);
      return;

    case AMD_DBGAPI_EVENT_INFO_CLIENT_THREAD:
      if (kind () != AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

      utils::get_info (
        value_size, value,
        std::get<breakpoint_resume_event_t> (m_data).client_thread_id);
      return;

    case AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE:
      if (kind () != AMD_DBGAPI_EVENT_KIND_RUNTIME)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

      utils::get_info (value_size, value,
                       std::get<runtime_event_t> (m_data).runtime_state);
      return;

    case AMD_DBGAPI_EVENT_INFO_QUEUE:
      if (kind () != AMD_DBGAPI_EVENT_KIND_QUEUE_ERROR)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

      throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED);
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_process_next_pending_event (amd_dbgapi_process_id_t process_id,
                                       amd_dbgapi_event_id_t *event_id,
                                       amd_dbgapi_event_kind_t *kind)
{
  TRACE_BEGIN (param_in (process_id), param_in (event_id), param_in (kind));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    if (event_id == nullptr || kind == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    event_t *event = nullptr;

    if (process_id != AMD_DBGAPI_PROCESS_NONE)
      {
        process_t *process = process_t::find (process_id);

        if (process == nullptr)
          THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);

        event = process->next_pending_event ();
      }
    else
      {
        for (auto &&process : process_t::all ())
          if ((event = process.next_pending_event ()) != nullptr)
            break;
      }

    if (event == nullptr)
      {
        *event_id = AMD_DBGAPI_EVENT_NONE;
        *kind = AMD_DBGAPI_EVENT_KIND_NONE;
      }
    else
      {
        *event_id = event->id ();
        *kind = event->kind ();
        event->set_state (event_t::state_t::reported);
      }
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
  TRACE_END (make_ref (param_out (event_id)), make_ref (param_out (kind)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_event_get_info (amd_dbgapi_event_id_t event_id,
                           amd_dbgapi_event_info_t query, size_t value_size,
                           void *value)
{
  TRACE_BEGIN (param_in (event_id), param_in (query), param_in (value_size),
               param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    event_t *event = find (event_id);

    if (event == nullptr || event->state () < event_t::state_t::reported)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_EVENT_ID);

    event->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_EVENT_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_event_processed (amd_dbgapi_event_id_t event_id)
{
  TRACE_BEGIN (param_in (event_id));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    event_t *event = find (event_id);

    if (event == nullptr || event->state () < event_t::state_t::reported)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_EVENT_ID);

    event->set_state (event_t::state_t::processed);

    /* We are done with this event, remove it from the map.  */
    event->process ().destroy (event);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_EVENT_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
  TRACE_END ();
}
