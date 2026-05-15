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

#include "utils.h"
#include "architecture.h"
#include "handle_object.h"
#include "process.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

namespace amd::dbgapi
{

namespace utils
{
namespace detail
{
/* Make sure the storage size of enums passed by reference matches the
   specification.
 */

/* Verify that all enum types used in the public interface as fields of a
   struct or pointee of a pointer type are the size specified by the public
   interface.
 */
static_assert (
  sizeof (amd_dbgapi_address_class_state_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_address_space_access_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_agent_state_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_breakpoint_action_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_changed_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_dispatch_barrier_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_dispatch_fence_scope_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_event_kind_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_exceptions_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_instruction_properties_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_memory_precision_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_os_queue_type_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_queue_state_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_register_class_state_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_register_properties_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_runtime_state_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_watchpoint_share_kind_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_wave_state_t) == sizeof (uint32_t)
    && sizeof (amd_dbgapi_wave_stop_reasons_t) == sizeof (uint32_t),
  "an enum type is not compatible with the dbgapi enum underlying type");

} /* namespace detail */

template <>
void
get_info (size_t value_size, void *ret, const std::string &value)
{
  if (ret == nullptr)
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  if (value_size != sizeof (char *))
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

  const size_t size = value.size ();
  auto retval = amd::dbgapi::allocate_memory<char[]> (size + 1);

  value.copy (retval.get (), size);
  retval[size] = '\0';

  *static_cast<char **> (ret) = retval.release ();
}

template <>
void
get_info (size_t value_size, void *ret, const instruction_t &value)
{
  if (ret == nullptr)
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  if (value_size != sizeof (uint8_t *))
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

  const size_t size = value.size ();
  auto retval = amd::dbgapi::allocate_memory<uint8_t[]> (size);

  memcpy (retval.get (), value.data (), size);

  *static_cast<uint8_t **> (ret) = retval.release ();
}

template <typename T>
void
get_info (size_t value_size, void *ret, const std::vector<T> &value)
{
  if (!ret)
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  if (value_size != sizeof (T *))
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

  const size_t size = sizeof (T) * value.size ();
  auto retval = amd::dbgapi::allocate_memory<T[]> (size);

  memcpy (retval.get (), value.data (), size);

  *static_cast<T **> (ret) = retval.release ();
}

template void get_info (size_t value_size, void *ret,
                        const std::vector<amd_dbgapi_watchpoint_id_t> &value);

template void get_info (size_t value_size, void *ret,
                        const std::vector<uint8_t> &value);

template <typename Object>
std::pair<typename Object::handle_type * /* objects */, size_t /* count */>
get_handle_list (const std::vector<process_t *> &processes,
                 amd_dbgapi_changed_t *changed)
{
  using Handle = typename Object::handle_type;

  /* Check whether the Objects have changed since the last time get_handle_list
     was called. The flag is set whenever objects are created, or destroyed, or
     invalidated, and cleared when get_handle_list is called.  */
  if (changed)
    {
      bool one_changed{ false };

      for (auto &&process : processes)
        one_changed |= process->set_changed<Object> (false);

      if (!one_changed)
        {
          *changed = AMD_DBGAPI_CHANGED_NO;
          return { nullptr, 0 };
        }
    }

  size_t count{ 0 };
  for (auto &&process : processes)
    count += process->count<Object> ();

  /* The size allocated includes space for all objects, so it is larger than
     necessary if there are invalid objects, but that is conservatively safe.
   */
  auto retval = allocate_memory<Handle[]> (count * sizeof (Handle));

  size_t pos{ 0 };
  for (auto &&process : processes)
    for (auto &&object : process->range<Object> ())
      {
        if (!is_valid (object))
          continue;

        dbgapi_assert (pos < count);
        retval[pos++] = object.id ();
      }

  if (changed)
    *changed = AMD_DBGAPI_CHANGED_YES;

  return { retval.release (), pos };
}

template std::pair<amd_dbgapi_code_object_id_t * /* objects */,
                   size_t /* count */>
get_handle_list<code_object_t> (const std::vector<process_t *> &processes,
                                amd_dbgapi_changed_t *changed);

template std::pair<amd_dbgapi_agent_id_t * /* objects */, size_t /* count */>
get_handle_list<agent_t> (const std::vector<process_t *> &processes,
                          amd_dbgapi_changed_t *changed);

template std::pair<amd_dbgapi_queue_id_t * /* objects */, size_t /* count */>
get_handle_list<queue_t> (const std::vector<process_t *> &processes,
                          amd_dbgapi_changed_t *changed);

template std::pair<amd_dbgapi_dispatch_id_t * /* objects */,
                   size_t /* count */>
get_handle_list<dispatch_t> (const std::vector<process_t *> &processes,
                             amd_dbgapi_changed_t *changed);

template std::pair<amd_dbgapi_workgroup_id_t * /* objects */,
                   size_t /* count */>
get_handle_list<workgroup_t> (const std::vector<process_t *> &processes,
                              amd_dbgapi_changed_t *changed);

template std::pair<amd_dbgapi_wave_id_t * /* objects */, size_t /* count */>
get_handle_list<wave_t> (const std::vector<process_t *> &processes,
                         amd_dbgapi_changed_t *changed);

std::string
human_readable_size (size_t size)
{
  if (size < KiB)
    return string_printf ("%" PRId64, size);
  if (size < MiB)
    return string_printf ("%.1fK", (double)size / KiB);
  if (size < GiB)
    return string_printf ("%.1fM", (double)size / MiB);

  return string_printf ("%.1fG", (double)size / GiB);
}

} /* namespace utils */

std::string
string_vprintf (const char *format, va_list va)
{
  va_list copy;

  va_copy (copy, va);
  size_t size = vsnprintf (NULL, 0, format, copy);
  va_end (copy);

  std::string str (size, '\0');
  vsprintf (&str[0], format, va);

  return str;
}

std::string
string_printf (const char *format, ...)
{
  va_list va;
  va_start (va, format);
  std::string str (string_vprintf (format, va));
  va_end (va);

  return str;
}

} /* namespace amd::dbgapi */
