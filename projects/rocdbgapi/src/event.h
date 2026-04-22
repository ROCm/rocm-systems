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

#ifndef AMD_DBGAPI_EVENT_H
#define AMD_DBGAPI_EVENT_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"
#include "logging.h"

#include <cstddef>
#include <string>
#include <variant>

namespace amd::dbgapi
{

class process_t;

class event_t : public detail::handle_object<amd_dbgapi_event_id_t>
{
public:
  enum class state_t
  {
    created,
    queued,
    reported,
    processed
  };

private:
  amd_dbgapi_event_kind_t const m_event_kind;
  state_t m_state{ state_t::created };

  /* Breakpoint resume event.  */
  struct breakpoint_resume_event_t
  {
    amd_dbgapi_breakpoint_id_t breakpoint_id;
    amd_dbgapi_client_thread_id_t client_thread_id;
  };

  /* Code object list updated event.  */
  struct code_object_list_updated_event_t
  {
    amd_dbgapi_event_id_t breakpoint_resume_event_id;
  };

  /* Runtime event.  */
  struct runtime_event_t
  {
    amd_dbgapi_runtime_state_t runtime_state;
    bool runtime_enable_during_attach;
  };

  /* Wave stop / command terminated event .  */
  struct wave_event_t
  {
    amd_dbgapi_wave_id_t wave_id;
  };

  std::variant<breakpoint_resume_event_t, code_object_list_updated_event_t,
               runtime_event_t, wave_event_t> const m_data;

  process_t &m_process;

  template <typename T>
  event_t (amd_dbgapi_event_id_t event_id, process_t &process,
           amd_dbgapi_event_kind_t event_kind, T &&data)
    : handle_object (event_id), m_event_kind (event_kind),
      m_data (std::forward<T> (data)), m_process (process)
  {
    log_info ("created %s, %s", to_cstring (id ()),
              pretty_printer_string ().c_str ());
  }

public:
  /* Breakpoint resume event.  */
  event_t (amd_dbgapi_event_id_t event_id, process_t &process,
           amd_dbgapi_event_kind_t event_kind,
           amd_dbgapi_breakpoint_id_t breakpoint_id,
           amd_dbgapi_client_thread_id_t client_thread_id);

  /* Code object list updated event.  */
  event_t (amd_dbgapi_event_id_t event_id, process_t &process,
           amd_dbgapi_event_kind_t event_kind,
           amd_dbgapi_event_id_t breakpoint_resume_event_id);

  /* Runtime event. */
  event_t (amd_dbgapi_event_id_t event_id, process_t &process,
           amd_dbgapi_event_kind_t event_kind,
           amd_dbgapi_runtime_state_t runtime_state);

  /* Wave stop / command terminated event.  */
  event_t (amd_dbgapi_event_id_t event_id, process_t &process,
           amd_dbgapi_event_kind_t event_kind, amd_dbgapi_wave_id_t wave_id);

  amd_dbgapi_event_kind_t kind () const { return m_event_kind; }

  state_t state () const { return m_state; }
  void set_state (state_t state);

  void get_info (amd_dbgapi_event_info_t query, size_t value_size,
                 void *value) const;

  std::string pretty_printer_string () const;

  process_t &process () const { return m_process; }
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_EVENT_H */
