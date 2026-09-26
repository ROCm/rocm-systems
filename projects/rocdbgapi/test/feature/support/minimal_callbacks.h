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

/* Minimal amd_dbgapi_callbacks_s implementation for feature tests.

   The callbacks here are deliberately spartan:
     - allocate_memory / deallocate_memory wrap malloc / free
     - log_message writes to stderr only when AMD_DBGAPI_TEST_LOG=1 is
       set, so a passing test stays quiet
     - client_process_get_info answers OS_PID with getpid() (Linux)
       and rejects CORE_STATE as NOT_AVAILABLE (no core dumps in tests)
     - insert_breakpoint / remove_breakpoint succeed without actually
       inserting anything: feature tests that need real breakpoints
       wire up their own callbacks via a derived fixture
     - xfer_global_memory routes to ::memcpy when both addresses are
       in this process's address space, which is enough for self-attach
       smoke tests; tests against other targets override it

   Tests that need richer behavior (breakpoint queues, fault injection)
   subclass dbgapi_fixture_t and supply their own callbacks struct.  */

#ifndef DBGAPI_FEATURE_SUPPORT_MINIMAL_CALLBACKS_H
#define DBGAPI_FEATURE_SUPPORT_MINIMAL_CALLBACKS_H

#include "amd-dbgapi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#  include <unistd.h>
#endif

namespace amd::dbgapi::test
{

inline void *
minimal_allocate_memory (size_t byte_size)
{
  if (byte_size == 0)
    return nullptr;
  return std::malloc (byte_size);
}

inline void
minimal_deallocate_memory (void *data)
{
  std::free (data);
}

inline amd_dbgapi_status_t
minimal_client_process_get_info (
  amd_dbgapi_client_process_id_t /* client_process_id */,
  amd_dbgapi_client_process_info_t query, size_t value_size, void *value)
{
  if (value == nullptr)
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

  switch (query)
    {
    case AMD_DBGAPI_CLIENT_PROCESS_INFO_OS_PID:
      {
        if (value_size != sizeof (amd_dbgapi_os_process_id_t))
          return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;
#if defined(__linux__)
        amd_dbgapi_os_process_id_t pid = ::getpid ();
        std::memcpy (value, &pid, sizeof (pid));
        return AMD_DBGAPI_STATUS_SUCCESS;
#else
        return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;
#endif
      }
    case AMD_DBGAPI_CLIENT_PROCESS_INFO_CORE_STATE:
      return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;
    }
  return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;
}

inline amd_dbgapi_status_t
minimal_insert_breakpoint (
  amd_dbgapi_client_process_id_t /* client_process_id */,
  amd_dbgapi_global_address_t /* address */,
  amd_dbgapi_breakpoint_id_t /* breakpoint_id */)
{
  /* Pretend we inserted it.  Tests that exercise the breakpoint path
     supply their own callback that records the request.  */
  return AMD_DBGAPI_STATUS_SUCCESS;
}

inline amd_dbgapi_status_t
minimal_remove_breakpoint (
  amd_dbgapi_client_process_id_t /* client_process_id */,
  amd_dbgapi_breakpoint_id_t /* breakpoint_id */)
{
  return AMD_DBGAPI_STATUS_SUCCESS;
}

inline amd_dbgapi_status_t
minimal_xfer_global_memory (
  amd_dbgapi_client_process_id_t /* client_process_id */,
  amd_dbgapi_global_address_t global_address, amd_dbgapi_size_t *value_size,
  void *read_buffer, const void *write_buffer)
{
  /* This default is for self-attach tests where global_address is just
     a host pointer in the current process.  It is intentionally NOT
     safe against bad pointers — tests using it own that risk.  */
  if (value_size == nullptr
      || (read_buffer == nullptr) == (write_buffer == nullptr))
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;

  if (read_buffer != nullptr)
    std::memcpy (read_buffer,
                 reinterpret_cast<const void *> (
                   static_cast<uintptr_t> (global_address)),
                 *value_size);
  else
    std::memcpy (reinterpret_cast<void *> (
                   static_cast<uintptr_t> (global_address)),
                 write_buffer, *value_size);
  return AMD_DBGAPI_STATUS_SUCCESS;
}

inline void
minimal_log_message (amd_dbgapi_log_level_t /* level */, const char *message)
{
  if (const char *env = std::getenv ("AMD_DBGAPI_TEST_LOG");
      env != nullptr && env[0] == '1')
    std::fprintf (stderr, "[dbgapi] %s\n", message);
}

/* The complete callbacks_s, populated with the minimal callbacks above.
   Pass &minimal_callbacks() to amd_dbgapi_initialize().  */
inline const amd_dbgapi_callbacks_s &
minimal_callbacks ()
{
  static const amd_dbgapi_callbacks_s cb = {
    minimal_allocate_memory,
    minimal_deallocate_memory,
    minimal_client_process_get_info,
    minimal_insert_breakpoint,
    minimal_remove_breakpoint,
    minimal_xfer_global_memory,
    minimal_log_message,
  };
  return cb;
}

} /* namespace amd::dbgapi::test */

#endif /* DBGAPI_FEATURE_SUPPORT_MINIMAL_CALLBACKS_H */
