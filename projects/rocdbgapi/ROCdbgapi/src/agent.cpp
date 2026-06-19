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

#include "agent.h"
#include "architecture.h"
#include "debug.h"
#include "exception.h"
#include "handle_object.h"
#include "initialization.h"
#include "logging.h"
#include "os_driver.h"
#include "process.h"
#include "utils.h"
#include "watchpoint.h"

#include <cinttypes>
#include <cstdint>
#include <vector>

namespace amd::dbgapi
{

agent_t::agent_t (amd_dbgapi_agent_id_t agent_id, process_t &process,
                  const architecture_t *architecture,
                  const os_agent_info_t &os_agent_info)
  : handle_object (agent_id), m_os_agent_info (os_agent_info),
    m_architecture (architecture), m_process (process)
{
  if (agent_id != AMD_DBGAPI_AGENT_NONE && !m_os_agent_info.firmware_supported)
    warning ("AMD GPU gpu_id %d's firmware version %d not supported",
             m_os_agent_info.os_agent_id, m_os_agent_info.fw_version);

  if (architecture == nullptr)
    return;

  for (auto &&address_space : architecture->range<address_space_t> ())
    {
      if (address_space.kind () != address_space_t::kind_t::generic)
        continue;

      dbgapi_assert (
        /* Lower the generic apertures addresses the os_driver reported for
           this agent.  The resulting address spaces should be of the kind
           local and private_swizzled for the local_address_aperture_base/limit
           and private_address_aperture_base/limit respectively. If this check
           fails, this agent's generic address spaces should be examined.  */
        address_space.lower (os_info ().local_address_aperture_base)
            .first.kind ()
          == address_space_t::kind_t::local
        && address_space.lower (os_info ().local_address_aperture_limit)
               .first.kind ()
             == address_space_t::kind_t::local
        && address_space.lower (os_info ().private_address_aperture_base)
               .first.kind ()
             == address_space_t::kind_t::private_swizzled
        && address_space.lower (os_info ().private_address_aperture_limit)
               .first.kind ()
             == address_space_t::kind_t::private_swizzled
        && "check the generic address space apertures");
    }

  m_watchpoints.resize (os_info ().address_watch_register_count);
}

bool
agent_t::spi_ttmps_setup_enabled () const
{
  return process ().is_flag_set (process_t::flag_t::spi_ttmps_setup_enabled)
         || os_info ().ttmps_always_initialized;
}

amd_dbgapi_watchpoint_share_kind_t
agent_t::watchpoint_share_kind () const
{
  if (!m_os_agent_info.address_watch_supported)
    return AMD_DBGAPI_WATCHPOINT_SHARE_KIND_UNSUPPORTED;

  return m_os_agent_info.watchpoint_exclusive
           ? AMD_DBGAPI_WATCHPOINT_SHARE_KIND_UNSHARED
           : AMD_DBGAPI_WATCHPOINT_SHARE_KIND_SHARED;
}

void
agent_t::set_exceptions (os_exception_mask_t exceptions)
{
  m_exceptions |= exceptions;
}

void
agent_t::clear_exceptions (os_exception_mask_t exceptions)
{
  m_exceptions &= ~exceptions;
}

void
agent_t::get_info (amd_dbgapi_agent_info_t query, size_t value_size,
                   void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_AGENT_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;

    case AMD_DBGAPI_AGENT_INFO_NAME:
      utils::get_info (value_size, value, m_os_agent_info.name);
      return;

    case AMD_DBGAPI_AGENT_INFO_ARCHITECTURE:
      if (architecture () == nullptr)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
      utils::get_info (value_size, value, architecture ()->id ());
      return;

    case AMD_DBGAPI_AGENT_INFO_STATE:
      utils::get_info (value_size, value,
                       supports_debugging ()
                         ? AMD_DBGAPI_AGENT_STATE_SUPPORTED
                         : AMD_DBGAPI_AGENT_STATE_NOT_SUPPORTED);
      return;

    case AMD_DBGAPI_AGENT_INFO_PCI_DOMAIN:
      utils::get_info (value_size, value, m_os_agent_info.domain);
      return;

    case AMD_DBGAPI_AGENT_INFO_PCI_SLOT:
      utils::get_info (value_size, value, m_os_agent_info.location_id);
      return;

    case AMD_DBGAPI_AGENT_INFO_PCI_VENDOR_ID:
      utils::get_info (value_size, value, m_os_agent_info.vendor_id);
      return;

    case AMD_DBGAPI_AGENT_INFO_PCI_DEVICE_ID:
      utils::get_info (value_size, value, m_os_agent_info.device_id);
      return;

    case AMD_DBGAPI_AGENT_INFO_EXECUTION_UNIT_COUNT:
      utils::get_info (value_size, value, m_os_agent_info.simd_count);
      return;

    case AMD_DBGAPI_AGENT_INFO_MAX_WAVES_PER_EXECUTION_UNIT:
      utils::get_info (value_size, value, m_os_agent_info.max_waves_per_simd);
      return;

    case AMD_DBGAPI_AGENT_INFO_OS_ID:
      utils::get_info (
        value_size, value,
        static_cast<amd_dbgapi_os_agent_id_t> (m_os_agent_info.os_agent_id));
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

void
agent_t::insert_watchpoint (const watchpoint_t &watchpoint)
{
  os_watch_mode_t watch_mode;
  switch (watchpoint.kind ())
    {
    case AMD_DBGAPI_WATCHPOINT_KIND_LOAD:
      watch_mode = os_watch_mode_t::read;
      break;
    case AMD_DBGAPI_WATCHPOINT_KIND_STORE_AND_RMW:
      watch_mode = os_watch_mode_t::nonread;
      break;
    case AMD_DBGAPI_WATCHPOINT_KIND_RMW:
      watch_mode = os_watch_mode_t::atomic;
      break;
    case AMD_DBGAPI_WATCHPOINT_KIND_ALL:
      watch_mode = os_watch_mode_t::all;
      break;
    default:
      dbgapi_assert_not_reached ("not a valid watchpoint kind");
    }

  os_watch_id_t os_watch_id;
  amd_dbgapi_status_t status = process ().os_driver ().set_address_watch (
    os_agent_id (), watchpoint.address (), -watchpoint.size (), watch_mode,
    &os_watch_id);

  if (status == AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE)
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE);
  else if (status != AMD_DBGAPI_STATUS_SUCCESS
           && status != AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED)
    fatal_error ("os_driver_t::set_address_watch () failed (%s)",
                 to_cstring (status));

  if (os_watch_id > os_info ().address_watch_register_count)
    fatal_error (
      "invalid os_watch_id returned by os_driver_t::set_address_watch ()");

  log_info ("%s: set address_watch%d [%#" PRIx64 "-%#" PRIx64 "] (%s)",
            to_cstring (id ()), os_watch_id, watchpoint.address (),
            watchpoint.address () + watchpoint.size (),
            to_cstring (watchpoint.kind ()));

  m_watchpoints[os_watch_id] = &watchpoint;
}

void
agent_t::remove_watchpoint (const watchpoint_t &watchpoint)
{
  auto it = std::find_if (std::begin (m_watchpoints), std::end (m_watchpoints),
                          [&watchpoint] (const watchpoint_t *w)
                          { return w == &watchpoint; });

  /* The watchpoint is not inserted for this agent.  */
  if (it == std::end (m_watchpoints))
    return;

  os_watch_id_t os_watch_id = std::distance (begin (m_watchpoints), it);

  amd_dbgapi_status_t status = process ().os_driver ().clear_address_watch (
    os_agent_id (), os_watch_id);

  if (status != AMD_DBGAPI_STATUS_SUCCESS
      && status != AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED)
    fatal_error ("failed to remove watchpoint (%s)", to_cstring (status));

  log_info ("%s: clear address_watch%d", to_cstring (id ()), os_watch_id);
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_agent_get_info (amd_dbgapi_agent_id_t agent_id,
                           amd_dbgapi_agent_info_t query, size_t value_size,
                           void *value)
{
  TRACE_BEGIN (param_in (agent_id), param_in (query), param_in (value_size),
               param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    agent_t *agent = find (agent_id);

    if (agent == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID);

    agent->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_process_agent_list (amd_dbgapi_process_id_t process_id,
                               size_t *agent_count,
                               amd_dbgapi_agent_id_t **agents,
                               amd_dbgapi_changed_t *changed)
{
  TRACE_BEGIN (param_in (process_id), param_in (agent_count),
               param_in (agents), param_in (changed));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    std::vector<process_t *> processes = process_t::match (process_id);

    if (agents == nullptr || agent_count == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    for (auto &&process : processes)
      process->update_agents ();

    std::tie (*agents, *agent_count)
      = utils::get_handle_list<agent_t> (processes, changed);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_ref (param_out (agent_count)),
             make_ref (make_ref (param_out (agents)), *agent_count),
             make_ref (param_out (changed)));
}
