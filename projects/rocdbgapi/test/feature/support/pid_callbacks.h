// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

/* Callback set where the OS_PID returned to the library is read out
   of the client_process_id pointer the caller hands to
   amd_dbgapi_process_attach.  This is what lets the same callbacks
   serve both self-attach (pid = getpid()) and child-attach (pid =
   spawned child's pid) without per-test global state.

   The contract: the caller allocates an amd_dbgapi_os_process_id_t,
   stores the target PID in it, and passes &that to attach().  The
   library hands the same pointer back as client_process_id when
   calling client_process_get_info, and pid_client_process_get_info
   reads the PID out.  */

#ifndef DBGAPI_FEATURE_SUPPORT_PID_CALLBACKS_H
#define DBGAPI_FEATURE_SUPPORT_PID_CALLBACKS_H

#include "amd-dbgapi.h"

#include "minimal_callbacks.h"

#include <cstring>

namespace amd::dbgapi::test
{

inline amd_dbgapi_status_t
pid_client_process_get_info (
  amd_dbgapi_client_process_id_t client_process_id,
  amd_dbgapi_client_process_info_t query, size_t value_size, void *value)
{
  if (value == nullptr || client_process_id == nullptr)
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

  switch (query)
    {
    case AMD_DBGAPI_CLIENT_PROCESS_INFO_OS_PID:
      {
        if (value_size != sizeof (amd_dbgapi_os_process_id_t))
          return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;
        amd_dbgapi_os_process_id_t pid
          = *reinterpret_cast<amd_dbgapi_os_process_id_t *> (client_process_id);
        std::memcpy (value, &pid, sizeof (pid));
        return AMD_DBGAPI_STATUS_SUCCESS;
      }
    case AMD_DBGAPI_CLIENT_PROCESS_INFO_CORE_STATE:
      return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;
    }
  return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;
}

inline const amd_dbgapi_callbacks_s &
pid_callbacks ()
{
  static const amd_dbgapi_callbacks_s cb = {
    minimal_allocate_memory,
    minimal_deallocate_memory,
    pid_client_process_get_info,
    minimal_insert_breakpoint,
    minimal_remove_breakpoint,
    minimal_xfer_global_memory,
    minimal_log_message,
  };
  return cb;
}

} /* namespace amd::dbgapi::test */

#endif /* DBGAPI_FEATURE_SUPPORT_PID_CALLBACKS_H */
