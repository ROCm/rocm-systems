/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

#include "code_object.h"
#include "debug.h"
#include "logging.h"

#include <amd-dbgapi/amd-dbgapi.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <link.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define DBGAPI_CHECK(expr)                                                    \
  do                                                                          \
    {                                                                         \
      if (amd_dbgapi_status_t status = (expr);                                \
          status != AMD_DBGAPI_STATUS_SUCCESS)                                \
        agent_error ("%s:%d: %s failed (rc=%d)", __FILE__, __LINE__, #expr,   \
                     status);                                                 \
  } while (false)

extern r_debug _amdgpu_r_debug;

using namespace amd::debug_agent;
using namespace std::string_literals;

/* Specialization of std::hash for amd_dbgapi_code_object_id_t, so it can be
   used in std::unordered_map.  */
template <> struct std::hash<amd_dbgapi_code_object_id_t>
{
  std::size_t operator() (const amd_dbgapi_code_object_id_t &id) const
  {
    return std::hash<decltype (id.handle)>{}(id.handle);
  }
};

/* Specialization of std::equal_to for amd_dbgapi_code_object_id_t, so it can
   be used in std::unordered_map.  */
template <> struct std::equal_to<amd_dbgapi_code_object_id_t>
{
  constexpr bool operator() (const amd_dbgapi_code_object_id_t &lhs,
                             const amd_dbgapi_code_object_id_t &rhs) const
  {
    return std::equal_to<decltype (lhs.handle)>{}(lhs.handle, rhs.handle);
  }
};

namespace
{
using code_object_map_t
    = std::unordered_map<amd_dbgapi_code_object_id_t, code_object_t>;

std::optional<std::string> g_code_objects_dir;
bool g_all_wavefronts{ false };
bool g_precise_emmory{ false };
bool g_precise_alu_exceptions{ false };

/* Global state accessed by the dbgapi callbacks.  */
std::optional<amd_dbgapi_breakpoint_id_t> g_rbrk_breakpoint_id;
struct
{
  std::atomic<bool> guard;
  std::optional<std::promise<void>> promise;
} g_rbrk_sync;

amd_dbgapi_status_t
amd_dbgapi_client_process_get_info (
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_client_process_info_t query, size_t value_size, void *value)
{
  if (value == nullptr)
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

  switch (query)
    {
    case AMD_DBGAPI_CLIENT_PROCESS_INFO_OS_PID:
      if (value_size != sizeof (amd_dbgapi_os_process_id_t))
        return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;
      *static_cast<amd_dbgapi_os_process_id_t *> (value) = getpid ();
      return AMD_DBGAPI_STATUS_SUCCESS;

    case AMD_DBGAPI_CLIENT_PROCESS_INFO_CORE_STATE:
      return AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE;
    }

  return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_dbgapi_status_t
amd_dbgapi_xfer_global_memory (
    amd_dbgapi_client_process_id_t client_process_id,
    amd_dbgapi_global_address_t global_address, amd_dbgapi_size_t *value_size,
    void *read_buffer, const void *write_buffer)
{
  if ((read_buffer == nullptr) == (write_buffer == nullptr))
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;

  if (client_process_id == 0)
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

  int *self_mem_fd = reinterpret_cast<int *> (client_process_id);
  if (*self_mem_fd == 0)
    return AMD_DBGAPI_STATUS_ERROR;

  ssize_t nbytes;
  if (write_buffer == nullptr)
    nbytes = pread (*self_mem_fd, read_buffer, *value_size, global_address);
  else
    nbytes = pwrite (*self_mem_fd, write_buffer, *value_size, global_address);

  if (nbytes == -1)
    {
      perror ((write_buffer == nullptr) ? "pread failed" : "pwrite failed");
      return AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS;
    }

  *value_size = nbytes;
  return AMD_DBGAPI_STATUS_SUCCESS;
}

static amd_dbgapi_callbacks_t dbgapi_callbacks = {
  /* allocate_memory.  */
  .allocate_memory = malloc,

  /* deallocate_memory.  */
  .deallocate_memory = free,

  /* get_os_pid.  */
  .client_process_get_info = amd_dbgapi_client_process_get_info,

  /* set_breakpoint callback.  */
  .insert_breakpoint =
      [] (amd_dbgapi_client_process_id_t client_process_id,
          amd_dbgapi_global_address_t address,
          amd_dbgapi_breakpoint_id_t breakpoint_id) {
        if (address == _amdgpu_r_debug.r_brk)
          {
            g_rbrk_breakpoint_id.emplace (breakpoint_id);
            return AMD_DBGAPI_STATUS_SUCCESS;
          }
        return AMD_DBGAPI_STATUS_ERROR;
      },

  /* remove_breakpoint callback.  */
  .remove_breakpoint =
      [] (amd_dbgapi_client_process_id_t client_process_id,
          amd_dbgapi_breakpoint_id_t breakpoint_id) {
        if (g_rbrk_breakpoint_id.has_value ()
            && breakpoint_id.handle == g_rbrk_breakpoint_id.value ().handle)
          {
            g_rbrk_breakpoint_id.reset ();
            return AMD_DBGAPI_STATUS_SUCCESS;
          }
        return AMD_DBGAPI_STATUS_ERROR;
      },

  /* xfer_global_memory callback.  */
  .xfer_global_memory = amd_dbgapi_xfer_global_memory,

  /* log_message callback.  */
  .log_message =
      [] (amd_dbgapi_log_level_t level, const char *message) {
        agent_out << "rocm-dbgapi: " << message << std::endl;
      }
};

std::string
hex_string (const std::vector<uint8_t> &value)
{
  std::string value_string;
  value_string.reserve (2 * value.size ());

  for (size_t pos = value.size (); pos > 0; --pos)
    {
      static constexpr char hex_digits[] = "0123456789abcdef";
      value_string.push_back (hex_digits[value[pos - 1] >> 4]);
      value_string.push_back (hex_digits[value[pos - 1] & 0xF]);
    }

  return value_string;
}

std::string
register_value_string (const std::string &register_type,
                       const std::vector<uint8_t> &register_value)
{
  /* handle vector types..  */
  if (size_t pos = register_type.find_last_of ('['); pos != std::string::npos)
    {
      const std::string element_type = register_type.substr (0, pos);
      const size_t element_count = std::stoi (register_type.substr (pos + 1));
      const size_t element_size = register_value.size () / element_count;

      agent_assert ((register_value.size () % element_size) == 0);

      std::stringstream ss;
      for (size_t i = 0; i < element_count; ++i)
        {
          if (i != 0)
            ss << " ";
          ss << "[" << i << "] ";

          std::vector<uint8_t> element_value (
              &register_value[element_size * i],
              &register_value[element_size * (i + 1)]);

          ss << register_value_string (element_type, element_value);
        }
      return ss.str ();
    }

  return hex_string (register_value);
}

void
print_registers (amd_dbgapi_wave_id_t wave_id)
{
  amd_dbgapi_architecture_id_t architecture_id;
  DBGAPI_CHECK (
      amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
                                sizeof (architecture_id), &architecture_id));

  size_t class_count;
  amd_dbgapi_register_class_id_t *register_class_ids;
  DBGAPI_CHECK (amd_dbgapi_architecture_register_class_list (
      architecture_id, &class_count, &register_class_ids));

  size_t register_count;
  amd_dbgapi_register_id_t *register_ids;
  DBGAPI_CHECK (
      amd_dbgapi_wave_register_list (wave_id, &register_count, &register_ids));

  auto hash = [] (const amd_dbgapi_register_id_t &id) {
    return std::hash<decltype (id.handle)>{}(id.handle);
  };
  auto equal_to = [] (const amd_dbgapi_register_id_t &lhs,
                      const amd_dbgapi_register_id_t &rhs) {
    return std::equal_to<decltype (lhs.handle)>{}(lhs.handle, rhs.handle);
  };
  std::unordered_set<amd_dbgapi_register_id_t, decltype (hash),
                     decltype (equal_to)>
      printed_registers (0, hash, equal_to);

  for (size_t i = 0; i < class_count; ++i)
    {
      amd_dbgapi_register_class_id_t register_class_id = register_class_ids[i];

      char *class_name_;
      DBGAPI_CHECK (amd_dbgapi_architecture_register_class_get_info (
          register_class_id, AMD_DBGAPI_REGISTER_CLASS_INFO_NAME,
          sizeof (class_name_), &class_name_));
      std::string class_name (class_name_);
      free (class_name_);

      /* Always print the "general" register class last.  */
      if (class_name == "general" && i < (class_count - 1))
        {
          register_class_ids[i--] = register_class_ids[class_count - 1];
          register_class_ids[class_count - 1] = register_class_id;
          continue;
        }

      agent_out << std::endl << class_name << " registers:";

      size_t last_register_size = 0;
      for (size_t j = 0, column = 0; j < register_count; ++j)
        {
          amd_dbgapi_register_id_t register_id = register_ids[j];

          /* Skip this register if is has already been printed as part of
             another register class.  */
          if (printed_registers.find (register_id) != printed_registers.end ())
            continue;

          amd_dbgapi_register_class_state_t state;
          DBGAPI_CHECK (amd_dbgapi_register_is_in_register_class (
              register_class_id, register_id, &state));

          if (state != AMD_DBGAPI_REGISTER_CLASS_STATE_MEMBER)
            continue;

          char *register_name_;
          DBGAPI_CHECK (amd_dbgapi_register_get_info (
              register_id, AMD_DBGAPI_REGISTER_INFO_NAME,
              sizeof (register_name_), &register_name_));
          std::string register_name (register_name_);
          free (register_name_);

          char *register_type_;
          DBGAPI_CHECK (amd_dbgapi_register_get_info (
              register_id, AMD_DBGAPI_REGISTER_INFO_TYPE,
              sizeof (register_type_), &register_type_));
          std::string register_type (register_type_);
          free (register_type_);

          size_t register_size;
          DBGAPI_CHECK (amd_dbgapi_register_get_info (
              register_id, AMD_DBGAPI_REGISTER_INFO_SIZE,
              sizeof (register_size), &register_size));

          std::vector<uint8_t> buffer (register_size);
          DBGAPI_CHECK (amd_dbgapi_read_register (
              wave_id, register_id, 0, register_size, buffer.data ()));

          const size_t num_register_per_line = 16 / register_size;

          if (register_size > sizeof (uint64_t) /* Registers larger than a
                                                   uint64_t are printed each
                                                   on a separate line.  */
              || register_size != last_register_size
              || (column++ % num_register_per_line) == 0)
            {
              agent_out << std::endl;
              column = 1;
            }

          last_register_size = register_size;

          agent_out << std::right << std::setfill (' ') << std::setw (16)
                    << (register_name + ": ")
                    << register_value_string (register_type, buffer);

          printed_registers.emplace (register_id);
        }

      agent_out << std::endl;
    }

  free (register_ids);
  free (register_class_ids);
}

void
print_local_memory (amd_dbgapi_wave_id_t wave_id)
{
  amd_dbgapi_process_id_t process_id;
  DBGAPI_CHECK (amd_dbgapi_wave_get_info (wave_id,
                                          AMD_DBGAPI_WAVE_INFO_PROCESS,
                                          sizeof (process_id), &process_id));

  amd_dbgapi_architecture_id_t architecture_id;
  DBGAPI_CHECK (
      amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
                                sizeof (architecture_id), &architecture_id));

  amd_dbgapi_address_space_id_t local_address_space_id;
  DBGAPI_CHECK (amd_dbgapi_dwarf_address_space_to_address_space (
      architecture_id, 0x3 /* DW_ASPACE_AMDGPU_local */,
      &local_address_space_id));

  std::vector<uint32_t> buffer (1024);
  amd_dbgapi_segment_address_t base_address{ 0 };

  while (true)
    {
      size_t requested_size = buffer.size () * sizeof (buffer[0]);
      size_t size = requested_size;
      if (amd_dbgapi_read_memory (process_id, wave_id, 0,
                                  local_address_space_id, base_address, &size,
                                  buffer.data ())
          != AMD_DBGAPI_STATUS_SUCCESS)
        break;

      agent_assert ((size % sizeof (buffer[0])) == 0);
      buffer.resize (size / sizeof (buffer[0]));

      if (!base_address)
        agent_out << std::endl << "Local memory content:";

      for (size_t i = 0, column = 0; i < buffer.size (); ++i)
        {
          if ((column++ % 8) == 0)
            {
              agent_out << std::endl
                        << "    0x" << std::setfill ('0') << std::setw (4)
                        << (base_address + i * sizeof (buffer[0])) << ":";
              column = 1;
            }

          agent_out << " " << std::hex << std::setfill ('0') << std::setw (8)
                    << buffer[i];
        }

      base_address += size;

      if (size != requested_size)
        break;
    }

  if (base_address)
    agent_out << std::endl;
}

void
stop_all_wavefronts (amd_dbgapi_process_id_t process_id)
{
  using wave_handle_type_t = decltype (amd_dbgapi_wave_id_t::handle);
  std::unordered_set<wave_handle_type_t> already_stopped;
  std::unordered_set<wave_handle_type_t> waiting_to_stop;

  agent_log (log_level_t::info, "stopping all wavefronts");
  for (size_t iter = 0;; ++iter)
    {
      agent_log (log_level_t::info, "iteration %zu:", iter);

      while (true)
        {
          amd_dbgapi_event_id_t event_id;
          amd_dbgapi_event_kind_t kind;

          DBGAPI_CHECK (amd_dbgapi_process_next_pending_event (
              process_id, &event_id, &kind));

          if (event_id.handle == AMD_DBGAPI_EVENT_NONE.handle)
            break;

          if (kind == AMD_DBGAPI_EVENT_KIND_WAVE_STOP
              || kind == AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED)
            {
              amd_dbgapi_wave_id_t wave_id;
              DBGAPI_CHECK (amd_dbgapi_event_get_info (
                  event_id, AMD_DBGAPI_EVENT_INFO_WAVE, sizeof (wave_id),
                  &wave_id));

              agent_assert (waiting_to_stop.find (wave_id.handle)
                            != waiting_to_stop.end ());

              waiting_to_stop.erase (wave_id.handle);

              if (kind == AMD_DBGAPI_EVENT_KIND_WAVE_STOP)
                {
                  already_stopped.emplace (wave_id.handle);

                  agent_log (log_level_t::info, "wave_%ld is stopped",
                             wave_id.handle);
                }
              else /* kind == AMD_DBGAPI_EVENT_KIND_COMMAND_TERMINATED */
                {
                  agent_log (log_level_t::info,
                             "wave_%ld terminated while stopping",
                             wave_id.handle);
                }
            }

          DBGAPI_CHECK (amd_dbgapi_event_processed (event_id));
        }

      amd_dbgapi_wave_id_t *wave_ids;
      size_t wave_count;
      DBGAPI_CHECK (amd_dbgapi_process_wave_list (process_id, &wave_count,
                                                  &wave_ids, nullptr));

      /* Stop all waves that are still running.  */
      for (size_t i = 0; i < wave_count; ++i)
        {
          amd_dbgapi_wave_id_t wave_id = wave_ids[i];

          if (already_stopped.find (wave_id.handle) != already_stopped.end ())
            continue;

          /* Already requested to stop.  */
          if (waiting_to_stop.find (wave_id.handle) != waiting_to_stop.end ())
            {
              agent_log (log_level_t::info, "wave_%ld is still stopping",
                         wave_id.handle);
              continue;
            }

          amd_dbgapi_wave_state_t state;
          if (amd_dbgapi_status_t status = amd_dbgapi_wave_get_info (
                  wave_id, AMD_DBGAPI_WAVE_INFO_STATE, sizeof (state), &state);
              status == AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID)
            {
              /* The wave could have terminated since it was reported in the
                 last wave list.  Skip it.  */
              continue;
            }
          else if (status != AMD_DBGAPI_STATUS_SUCCESS)
            agent_error ("amd_dbgapi_wave_get_info failed (rc=%d)", status);

          if (state == AMD_DBGAPI_WAVE_STATE_STOP)
            {
              already_stopped.emplace (wave_ids[i].handle);

              agent_log (log_level_t::info, "wave_%ld is already stopped",
                         wave_id.handle);
              continue;
            }
          if (state == AMD_DBGAPI_WAVE_STATE_SINGLE_STEP)
            {
              /* The wave is single-stepping, it will stop and report an event
                 once the instruction execution is complete.  */
              agent_log (log_level_t::info, "wave_%ld is single-stepping",
                         wave_id.handle);
              continue;
            }

          if (amd_dbgapi_status_t status = amd_dbgapi_wave_stop (wave_id);
              status == AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID)
            {
              /* The wave could have terminated since it was reported in the
                 last wave list.  Skip it.  */
              continue;
            }
          else if (status != AMD_DBGAPI_STATUS_SUCCESS)
            agent_error ("amd_dbgapi_wave_stop failed (rc=%d)", status);

          agent_log (log_level_t::info,
                     "wave_%ld is running, sent stop request", wave_id.handle);

          waiting_to_stop.emplace (wave_id.handle);
        }

      free (wave_ids);

      if (!waiting_to_stop.size ())
        break;
    }

  agent_log (log_level_t::info, "all wavefronts are stopped");
}

void
print_wavefronts (amd_dbgapi_process_id_t process_id, bool all_wavefronts,
                  code_object_map_t &code_object_map)
{
  /* This function is not thread-safe and not re-entrant.  */
  static std::mutex lock;
  if (!lock.try_lock ())
    return;
  /* Make sure the lock is released when this function returns.  */
  std::scoped_lock sl (std::adopt_lock, lock);

  if (all_wavefronts)
    stop_all_wavefronts (process_id);

  amd_dbgapi_wave_id_t *wave_ids;
  size_t wave_count;
  DBGAPI_CHECK (amd_dbgapi_process_wave_list (process_id, &wave_count,
                                              &wave_ids, nullptr));

  for (size_t i = 0; i < wave_count; ++i)
    {
      amd_dbgapi_wave_id_t wave_id = wave_ids[i];

      amd_dbgapi_wave_state_t state;
      DBGAPI_CHECK (amd_dbgapi_wave_get_info (
          wave_id, AMD_DBGAPI_WAVE_INFO_STATE, sizeof (state), &state));

      if (state != AMD_DBGAPI_WAVE_STATE_STOP)
        continue;

      std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason;
      DBGAPI_CHECK (
          amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_STOP_REASON,
                                    sizeof (stop_reason), &stop_reason));

      amd_dbgapi_global_address_t pc;
      DBGAPI_CHECK (amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_PC,
                                              sizeof (pc), &pc));

      std::optional<amd_dbgapi_global_address_t> kernel_entry;
      amd_dbgapi_dispatch_id_t dispatch_id;
      if (auto status
          = amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_DISPATCH,
                                      sizeof (dispatch_id), &dispatch_id);
          status == AMD_DBGAPI_STATUS_SUCCESS)
        {
          DBGAPI_CHECK (amd_dbgapi_dispatch_get_info (
              dispatch_id, AMD_DBGAPI_DISPATCH_INFO_KERNEL_CODE_ENTRY_ADDRESS,
              sizeof (decltype (kernel_entry)::value_type),
              &kernel_entry.emplace ()));
        }
      /* The only possible error is NOT_AVAILABLE if the ttmp registers weren't
         initialized when the wave was created.  */
      else if (status != AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE)
        {
          agent_error ("amd_dbgapi_wave_get_info failed (rc=%d)", status);
        }

      if (i)
        agent_out << std::endl;

      agent_out << "--------------------------------------------------------"
                << std::endl;

      agent_out << "wave_" << std::dec << wave_id.handle << ": pc=0x"
                << std::hex << pc << " (kernel_code_entry=";

      if (kernel_entry)
        {
          agent_out << "0x" << std::hex << *kernel_entry;

          /* Find the code object that contains this kernel.  */
          for ([[maybe_unused]] auto &&[id, code_object] : code_object_map)
            if (*kernel_entry >= code_object.load_address ()
                && *kernel_entry
                       < code_object.load_address () + code_object.mem_size ())
              {
                if (auto symbol = code_object.find_symbol (*kernel_entry))
                  agent_out << " <" << symbol->m_name << ">";
                break;
              }
        }
      else
        agent_out << "not available";

      agent_out << ")";

      std::string stop_reason_str;
      auto stop_reason_bits{ stop_reason };
      do
        {
          /* Consume one bit from the stop reason.  */
          auto one_bit
              = stop_reason_bits ^ (stop_reason_bits & (stop_reason_bits - 1));
          stop_reason_bits ^= one_bit;

          if (!stop_reason_str.empty ())
            stop_reason_str += "|";

          stop_reason_str += [] (amd_dbgapi_wave_stop_reasons_t reason) {
            switch (reason)
              {
              case AMD_DBGAPI_WAVE_STOP_REASON_NONE:
                return "NONE";
              case AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT:
                return "BREAKPOINT";
              case AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT:
                return "WATCHPOINT";
              case AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP:
                return "SINGLE_STEP";
              case AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL:
                return "FP_INPUT_DENORMAL";
              case AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0:
                return "FP_DIVIDE_BY_0";
              case AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW:
                return "FP_OVERFLOW";
              case AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW:
                return "FP_UNDERFLOW";
              case AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT:
                return "FP_INEXACT";
              case AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION:
                return "FP_INVALID_OPERATION";
              case AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0:
                return "INT_DIVIDE_BY_0";
              case AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP:
                return "DEBUG_TRAP";
              case AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP:
                return "ASSERT_TRAP";
              case AMD_DBGAPI_WAVE_STOP_REASON_TRAP:
                return "TRAP";
              case AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION:
                return "MEMORY_VIOLATION";
              case AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR:
                return "ADDRESS_ERROR";
              case AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION:
                return "ILLEGAL_INSTRUCTION";
              case AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR:
                return "ECC_ERROR";
              case AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT:
                return "FATAL_HALT";
#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
              case AMD_DBGAPI_WAVE_STOP_REASON_RESERVED:
                return "RESERVED";
#endif
              }
            return "";
          }(static_cast<amd_dbgapi_wave_stop_reasons_t> (one_bit));
      } while (stop_reason_bits);

      agent_out << " (";
      if (stop_reason != AMD_DBGAPI_WAVE_STOP_REASON_NONE)
        agent_out << "stopped, reason: " << stop_reason_str;
      else
        agent_out << "running";
      agent_out << ")" << std::endl;

      print_registers (wave_id);
      print_local_memory (wave_id);

      /* Find the code object that contains this pc.  */
      code_object_t *code_object_found{ nullptr };
      for ([[maybe_unused]] auto &&[id, code_object] : code_object_map)
        if (pc >= code_object.load_address ()
            && pc < code_object.load_address () + code_object.mem_size ())
          {
            code_object_found = &code_object;
            break;
          }

      if (code_object_found)
        {
          amd_dbgapi_architecture_id_t architecture_id;
          DBGAPI_CHECK (amd_dbgapi_wave_get_info (
              wave_id, AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
              sizeof (architecture_id), &architecture_id));

          /* Disassemble instructions around `pc`  */
          code_object_found->disassemble (architecture_id, pc);
        }
      else
        {
          /* TODO: Add disassembly even if we did not find a code object  */
        }
    }

  free (wave_ids);
}

void
print_usage ()
{
  std::cerr << "ROCdebug-agent usage:" << std::endl;
  std::cerr << "  -a, --all                   "
               "Print all wavefronts."
            << std::endl;
  std::cerr << "  -s, --save-code-objects[=DIR]   "
               "Save all loaded code objects. If the directory"
            << std::endl
            << "                              "
               "is not specified, the code objects are saved in"
            << std::endl
            << "                              "
               "the current directory."
            << std::endl;
  std::cerr << "  -p, --precise-memory        "
            << "Enable precise memory mode which ensures that " << std::endl
            << "                              "
               "when an exception is reported, the PC points to"
            << std::endl
            << "                              "
               "the instruction immediately after the one that"
            << std::endl
            << "                              "
               "caused the exception."
            << std::endl;
  std::cerr << "  -o, --output=FILE           "
               "Save the output in FILE. By default, the output"
            << std::endl
            << "                              "
               "is redirected to stderr."
            << std::endl;
  std::cerr << "  -d, --disable-linux-signals "
               "Disable installing a SIGQUIT signal handler, so"
            << std::endl
            << "                              "
               "that the default Linux handler may dump a core"
            << std::endl
            << "                              "
               "file."
            << std::endl;
  std::cerr << "  -l, --log-level={none|error|warning|info|verbose}"
            << std::endl
            << "                              "
               "Change the Debug Agent and Debugger API log"
            << std::endl
            << "                              "
               "level. The default log level is 'none'."
            << std ::endl;
  std::cerr << "  -h, --help                  "
               "Display a usage message and abort the process."
            << std::endl;

  abort ();
}

/* Called when we expect dbgapi events to be present.  Fetch all events from
   dbgapi and act on the required events.  */

void
process_dbgapi_events (amd_dbgapi_process_id_t process_id, bool all_wavefronts,
                       code_object_map_t &code_object_map)
{
  /* Consume all events available in the queue.  */
  bool need_print_waves = false;
  bool wave_need_resume = false;
  while (true)
    {
      amd_dbgapi_event_id_t event_id;
      amd_dbgapi_event_kind_t event_kind;
      DBGAPI_CHECK (amd_dbgapi_process_next_pending_event (
          process_id, &event_id, &event_kind));

      if (event_kind == AMD_DBGAPI_EVENT_KIND_NONE)
        break;

      switch (event_kind)
        {
        case AMD_DBGAPI_EVENT_KIND_WAVE_STOP:
          {
            /* Fetch the stop reason.  For a debug trap, we just resume
               execution.  */
            amd_dbgapi_wave_stop_reasons_t stop_reason;
            amd_dbgapi_wave_id_t wave_id;
            DBGAPI_CHECK (amd_dbgapi_event_get_info (
                event_id, AMD_DBGAPI_EVENT_INFO_WAVE, sizeof (wave_id),
                &wave_id));
            DBGAPI_CHECK (amd_dbgapi_wave_get_info (
                wave_id, AMD_DBGAPI_WAVE_INFO_STOP_REASON,
                sizeof (stop_reason), &stop_reason));

            if (stop_reason == AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP)
              {
                /* This wave will be silently resumed at the end of this
                   procedure.  */
                wave_need_resume = true;
              }
            else
              need_print_waves = true;
            break;
          }

        case AMD_DBGAPI_EVENT_KIND_QUEUE_ERROR:
          {
            need_print_waves = true;
            break;
          }

        case AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED:
          {
            amd_dbgapi_code_object_id_t *code_objects_ids;
            size_t code_object_count;
            DBGAPI_CHECK (amd_dbgapi_process_code_object_list (
                process_id, &code_object_count, &code_objects_ids, nullptr));

            code_object_map_t fresh_map;
            for (size_t i = 0; i < code_object_count; ++i)
              {
                if (auto it = code_object_map.find (code_objects_ids[i]);
                    it != code_object_map.end ())
                  {
                    fresh_map.emplace (code_objects_ids[i],
                                       std::move (it->second));
                  }
                else
                  {
                    code_object_t code_object (code_objects_ids[i]);

                    code_object.open ();
                    if (!code_object.is_open ())
                      {
                        agent_warning ("could not open code_object_%ld",
                                       code_objects_ids[i].handle);
                        continue;
                      }

                    if (g_code_objects_dir
                        && !code_object.save (*g_code_objects_dir))
                      agent_warning ("could not save code object to %s",
                                     g_code_objects_dir->c_str ());

                    fresh_map.emplace (code_objects_ids[i],
                                       std::move (code_object));
                  }
              }
            free (code_objects_ids);
            std::swap (code_object_map, fresh_map);
            break;
          }

        case AMD_DBGAPI_EVENT_KIND_RUNTIME:
        case AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME:
          /* Ignore.  */
          break;

        default:
          agent_log (log_level_t::warning, "Unexpected event kind %d",
                     event_kind);
          break;
        }

      /* We cannot resume the wave until this is done.  We should drain all
         and resume waves once all events are drained.  */
      DBGAPI_CHECK (amd_dbgapi_event_processed (event_id));
    }

  /* Some events do not require us to do anythig more.  If so, just return
     early.  */
  if (!need_print_waves && !wave_need_resume)
    return;

  /* TODO, we  should have a RAII object to handle forward progress wave
     creation mode override.  */
  DBGAPI_CHECK (amd_dbgapi_process_set_progress (
      process_id, AMD_DBGAPI_PROGRESS_NO_FORWARD));

  DBGAPI_CHECK (amd_dbgapi_process_set_wave_creation (
      process_id, AMD_DBGAPI_WAVE_CREATION_STOP));

  if (need_print_waves)
    print_wavefronts (process_id, all_wavefronts, code_object_map);

  /* We now need to resume execution of the waves present.  This will allow any
     exception to be delivered to the runtime who will be able to act on it if
     required.  */
  amd_dbgapi_wave_id_t *wave_ids;
  size_t wave_count;
  DBGAPI_CHECK (amd_dbgapi_process_wave_list (process_id, &wave_count,
                                              &wave_ids, nullptr));
  std::unique_ptr<amd_dbgapi_wave_id_t, decltype (free) *> wave_ids_cleaner (
      wave_ids, free);

  for (size_t i = 0; i < wave_count; ++i)
    {
      amd_dbgapi_wave_id_t wave_id = wave_ids[i];

      amd_dbgapi_wave_state_t state;
      DBGAPI_CHECK (amd_dbgapi_wave_get_info (
          wave_id, AMD_DBGAPI_WAVE_INFO_STATE, sizeof (state), &state));

      if (state != AMD_DBGAPI_WAVE_STATE_STOP)
        continue;

      std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason;
      DBGAPI_CHECK (
          amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_STOP_REASON,
                                    sizeof (stop_reason), &stop_reason));
      auto stop_reason_bits{ stop_reason };

      std::underlying_type_t<amd_dbgapi_exceptions_t> resume_exceptions = 0;
      do
        {
          auto one_bit
              = stop_reason_bits ^ (stop_reason_bits & (stop_reason_bits - 1));
          stop_reason_bits ^= one_bit;

          switch (stop_reason)
            {
            case AMD_DBGAPI_WAVE_STOP_REASON_NONE:
            case AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP:
              resume_exceptions |= AMD_DBGAPI_EXCEPTION_NONE;
              break;

            case AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT:
            case AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT:
            case AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP:
            case AMD_DBGAPI_WAVE_STOP_REASON_TRAP:
              resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_TRAP;
              break;

            case AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP:
              /* Is this even possible?  */
              resume_exceptions |= AMD_DBGAPI_EXCEPTION_NONE;
              break;

            case AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL:
            case AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0:
            case AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW:
            case AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW:
            case AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT:
            case AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION:
            case AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0:
              resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR;
              break;

            case AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION:
              resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION;
              break;

            case AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR:
              resume_exceptions
                  |= AMD_DBGAPI_EXCEPTION_WAVE_ADDRESS_ERROR;
              break;

            case AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION:
              resume_exceptions
                  |= AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION;
              break;

            case AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR:
            case AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT:
              resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ABORT;
              break;

#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
            case AMD_DBGAPI_WAVE_STOP_REASON_RESERVED:
              break;
#endif
            }
      } while (stop_reason_bits != 0);

      DBGAPI_CHECK (amd_dbgapi_wave_resume (
          wave_id, AMD_DBGAPI_RESUME_MODE_NORMAL,
          static_cast<amd_dbgapi_exceptions_t> (resume_exceptions)));
    }

  DBGAPI_CHECK (amd_dbgapi_process_set_wave_creation (
      process_id, AMD_DBGAPI_WAVE_CREATION_NORMAL));

  DBGAPI_CHECK (amd_dbgapi_process_set_progress (process_id,
                                                 AMD_DBGAPI_PROGRESS_NORMAL));
}

/* Main function of the accessory thread used to handle dbgapi.  The LISTEN_FD
   parameter is the read end of a pipe where the main application can write
   to instruct the worker thread to stop.  */
void
dbgapi_worker (int listen_fd, bool all_wavefronts, bool precise_memory,
               bool precise_alu_exceptions)
{
  amd_dbgapi_process_id_t process_id;
  amd_dbgapi_event_id_t event_id;
  amd_dbgapi_event_kind_t event_kind;
  amd_dbgapi_notifier_t notifier;
  int epoll_fd;
  epoll_event ev{};

  /* Enable and attach dbgapi.  */
  DBGAPI_CHECK (amd_dbgapi_initialize (&dbgapi_callbacks));

  int self_mem_fd = open ("/proc/self/mem", O_RDWR);
  if (self_mem_fd == -1)
    agent_error ("Failed to open /proc/self/mem: %s\n", strerror (errno));
  std::unique_ptr<int, std::function<void (int *)>> self_mem_fd_closer (
      &self_mem_fd, [] (int *fd) { close (*fd); });

  DBGAPI_CHECK (amd_dbgapi_process_attach (
      reinterpret_cast<amd_dbgapi_client_process_id_t> (&self_mem_fd),
      &process_id));

  /* Runtime has been activated just before tools are loaded.  We do expect
     a runtime loaded event to be ready to be consumed.  */
  DBGAPI_CHECK (amd_dbgapi_process_next_pending_event (process_id, &event_id,
                                                       &event_kind));
  if (event_kind != AMD_DBGAPI_EVENT_KIND_RUNTIME)
    agent_error ("Unexpected event kind %d", event_kind);

  amd_dbgapi_runtime_state_t runtime_state;

  DBGAPI_CHECK (
      amd_dbgapi_event_get_info (event_id, AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE,
                                 sizeof (runtime_state), &runtime_state));

  switch (runtime_state)
    {
    case AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS:
      break;

    case AMD_DBGAPI_RUNTIME_STATE_UNLOADED:
      agent_error ("invalid runtime state %d", runtime_state);

    case AMD_DBGAPI_RUNTIME_STATE_LOADED_ERROR_RESTRICTION:
      agent_error ("unable to enable GPU debugging due to a "
                   "restriction error");
      break;
    }

  DBGAPI_CHECK (amd_dbgapi_event_processed (event_id));

  DBGAPI_CHECK (amd_dbgapi_process_get_info (process_id,
                                             AMD_DBGAPI_PROCESS_INFO_NOTIFIER,
                                             sizeof (notifier), &notifier));

  epoll_fd = epoll_create1 (0);
  if (epoll_fd == -1)
    agent_error ("unable to create epoll instance: %s", strerror (errno));

  ev.data.fd = listen_fd;
  ev.events = EPOLLIN;
  if (epoll_ctl (epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev) == -1)
    agent_error ("Unable to add rocr notifier to the epoll instance: %s",
                 strerror (errno));

  ev.data.fd = notifier;
  ev.events = EPOLLIN;
  if (epoll_ctl (epoll_fd, EPOLL_CTL_ADD, notifier, &ev) == -1)
    agent_error ("Unable to add dbgapi notifier to the epoll instance: %s",
                 strerror (errno));

  if (precise_memory)
    {
      amd_dbgapi_status_t r = amd_dbgapi_set_memory_precision (
          process_id, AMD_DBGAPI_MEMORY_PRECISION_PRECISE);
      if (r != AMD_DBGAPI_STATUS_SUCCESS)
        {
          if (r == AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED)
            agent_warning ("Precise memory not supported for all the agents "
                           "of this process.");
          else
            agent_error ("amd_dbgapi_set_memory_precision failed");
        }
    }

  if (precise_alu_exceptions)
    {
      amd_dbgapi_status_t r = amd_dbgapi_set_alu_exceptions_precision (
          process_id, AMD_DBGAPI_ALU_EXCEPTIONS_PRECISION_PRECISE);
      if (r != AMD_DBGAPI_STATUS_SUCCESS)
        {
          if (r == AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED)
            agent_warning ("Precise ALU exceptions not supported for all the "
                           "agents of this process.");
          else
            agent_error ("amd_dbgapi_set_alu_exceptions_precision failed");
        }
    }

  /* The initial setup is finished, notify the main thread it can go on.  */
  [[maybe_unused]] bool promise_available
    = g_rbrk_sync.guard.load (std::memory_order::memory_order_acquire);
  agent_assert (promise_available);
  g_rbrk_sync.promise->set_value ();

  /* Map containing an image of all code objects loaded by the process.  */
  code_object_map_t code_object_map;

  for (bool continue_event_loop = true; continue_event_loop;)
    {
      /* We can wait for events on at most 2 file descriptors.  */
      constexpr size_t max_events = 2;
      epoll_event evs[max_events];

      int nfd = epoll_wait (epoll_fd, evs, max_events, -1);
      if (nfd == -1 && errno == EINTR)
        continue;

      if (nfd == -1)
        agent_error ("epoll_wait failed: %s", strerror (errno));

      for (int i = 0; i < nfd; i++)
        {
          /* Make sure we purge all data from the pipe.  */
          if (evs[i].data.fd == listen_fd)
            {
              char buf = '\0';
              while (read (evs[i].data.fd, &buf, 1) == -1 && errno == EINTR)
                ;

              switch (buf)
                {
                case 'p':
                  print_wavefronts (process_id, true, code_object_map);
                  break;
                case 'q':
                  /* It is time to exit the main event loop and detach dbgapi.
                   */
                  continue_event_loop = false;
                  break;
                case 'b':
                  {
                    /* We need the atomic load to ensure that the promise
                       object we try to access is consistent with the
                       requesting thread.  */
                    [[maybe_unused]] bool promise_available
                        = g_rbrk_sync.guard.load (
                            std::memory_order::memory_order_acquire);

                    agent_assert (promise_available);
                    agent_assert (g_rbrk_sync.promise.has_value ());

                    agent_assert (g_rbrk_breakpoint_id.has_value ());
                    amd_dbgapi_breakpoint_action_t bpaction;
                    DBGAPI_CHECK (amd_dbgapi_report_breakpoint_hit (
                        g_rbrk_breakpoint_id.value (), 0, &bpaction));

                    if (bpaction == AMD_DBGAPI_BREAKPOINT_ACTION_HALT)
                      {
                        /* At this point, dbgapi has created a
                           AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED
                           event, we need to process it before notifying the
                           promise, which will cause the thread that modified
                           the code object list to resume.  */
                        process_dbgapi_events (process_id, all_wavefronts,
                                               code_object_map);
                      }

                    g_rbrk_sync.promise->set_value ();
                    break;
                  }
                }
            }
          else if (evs[i].data.fd == notifier)
            {
              /* Drain the pipe.  */
              int r;
              do
                {
                  char buf;
                  r = read (evs[i].data.fd, &buf, 1);
              } while (r >= 0 || (r == -1 && errno == EINTR));
              process_dbgapi_events (process_id, all_wavefronts,
                                     code_object_map);
            }
          else
            agent_error ("Unknown file descriptor %d", evs[i].data.fd);
        }
    }

  DBGAPI_CHECK (amd_dbgapi_process_detach (process_id));
  DBGAPI_CHECK (amd_dbgapi_finalize ());
}

class DebugAgentWorker
{
public:
  DebugAgentWorker ();
  ~DebugAgentWorker ();

  DebugAgentWorker (const DebugAgentWorker &) = delete;
  DebugAgentWorker (DebugAgentWorker &&) = delete;
  DebugAgentWorker &operator= (const DebugAgentWorker &) = delete;
  DebugAgentWorker &operator= (DebugAgentWorker &&) = delete;

  void query_print_waves () const;
  void update_code_object_list () const;

private:
  std::thread m_worker_thread;
  int m_write_pipe = -1;
};

DebugAgentWorker::DebugAgentWorker ()
{
  /* Create the promise/future pair and make sure the promise is visible by
     the worker thread.  */
  g_rbrk_sync.promise.emplace ();
  auto init_future = g_rbrk_sync.promise->get_future ();
  g_rbrk_sync.guard.store (true, std::memory_order::memory_order_release);

  int pipefd[2];
  if (pipe2 (pipefd, O_CLOEXEC) == -1)
    agent_error ("failed to create pipe: %s", strerror (errno));

  if (fcntl (pipefd[0], F_SETFL, O_NONBLOCK)
      || fcntl (pipefd[1], F_SETFL, O_NONBLOCK))
    agent_error ("failed to set pipe non-blocking: %s", strerror (errno));

  m_write_pipe = pipefd[1];

  m_worker_thread = std::thread (dbgapi_worker, pipefd[0], g_all_wavefronts,
                                 g_precise_emmory, g_precise_alu_exceptions);

  /* Wait for the worker thread to have setup dbgapi.  */
  init_future.wait ();
  g_rbrk_sync.promise.reset ();
  g_rbrk_sync.guard.store (false, std::memory_order::memory_order_release);

  auto pthread_thread = m_worker_thread.native_handle ();
  if (pthread_setname_np (pthread_thread, "RocrDebugAgent") == -1)
    agent_error ("Failed to set thread name: %s", strerror (errno));
}

void
DebugAgentWorker::query_print_waves () const
{
  agent_assert (m_write_pipe != -1);
  char msg = 'p';
  ssize_t written = write (m_write_pipe, &msg, 1);
  if (written == -1)
    agent_error ("Failed to notify RocrDebugAgent thread (%s)",
                 strerror (errno));
  agent_assert (written == 1);
}

void
DebugAgentWorker::update_code_object_list () const
{
  agent_assert (m_write_pipe != -1);
  /* It is OK to have this load removed if the assertion is not compiled in.
     It does not have a role in synchronization.  */
  agent_assert (
      !g_rbrk_sync.guard.load (std::memory_order::memory_order_acquire));
  agent_assert (!g_rbrk_sync.promise.has_value ());

  /* Create the promise/future pair and make sure the promise is visible by
     the worker thread.  */
  g_rbrk_sync.promise.emplace ();
  auto update_brk_future = g_rbrk_sync.promise->get_future ();
  g_rbrk_sync.guard.store (true, std::memory_order::memory_order_release);

  /* Use the pipe to notify the thread a code object list is requested.  */
  char msg = 'b';
  ssize_t written = write (m_write_pipe, &msg, 1);
  if (written == -1)
    agent_error ("Failed to notify RocrDebugAgent thread (%s)",
                 strerror (errno));
  agent_assert (written == 1);

  /* Wait for the worker thread to acknoledge code object update has proceded
     and reset the synch structure so it can be reused in a later call.  */
  update_brk_future.wait ();
  g_rbrk_sync.promise.reset ();
  g_rbrk_sync.guard.store (false, std::memory_order::memory_order_release);
}

DebugAgentWorker::~DebugAgentWorker ()
{
  if (m_write_pipe != -1)
    {
      char msg = 'q';
      ssize_t written = write (m_write_pipe, &msg, 1);
      if (written == -1)
        agent_error ("Failed to notify RocrDebugAgent thread (%s)",
                     strerror (errno));
      agent_assert (written == 1);
      m_worker_thread.join ();
      close (m_write_pipe);
      m_write_pipe = -1;
    }
}

/* Forward declarations of the WorkerThreadAccess factory.  */
class WorkerThreadAccess;
static WorkerThreadAccess get_worker_thread ();

/* Wraps an exclusive access (protected by a mutex) to the debug agent
   worker thread.  */
class WorkerThreadAccess
{
private:
  WorkerThreadAccess (std::mutex &m, std::optional<DebugAgentWorker> &w)
      : m_lock (m), m_worker (w)
  {
  }

  friend WorkerThreadAccess get_worker_thread ();

public:
  WorkerThreadAccess () = delete;
  WorkerThreadAccess (const WorkerThreadAccess &) = delete;
  WorkerThreadAccess (WorkerThreadAccess &&) = delete;
  ~WorkerThreadAccess () = default;

  WorkerThreadAccess &operator= (const WorkerThreadAccess &) = delete;
  WorkerThreadAccess &operator= (WorkerThreadAccess &&) = delete;

  /* Start the worker thread, if not already started.  */
  void start ()
  {
    if (!m_worker.has_value ())
      m_worker.emplace ();
  }

  /* Terminate the worker thread.  */
  void stop ()
  {
    if (m_worker.has_value ())
      m_worker.reset ();
  }

  void update_code_object_list ()
  {
    if (m_worker.has_value ())
      m_worker->update_code_object_list ();
  }

  void query_print_waves ()
  {
    if (m_worker.has_value ())
      m_worker->query_print_waves ();
  }

private:
  std::lock_guard<std::mutex> m_lock;
  std::optional<DebugAgentWorker> &m_worker;
};

static WorkerThreadAccess
get_worker_thread ()
{
  struct WorkerWrapper
  {
    std::optional<DebugAgentWorker> thread;
    std::mutex mutex;
  };
  static WorkerWrapper *wp = [] () {
    alignas (
        WorkerWrapper) static char wrapper_storage[sizeof (WorkerWrapper)];
    WorkerWrapper *wrapper = new (wrapper_storage) WorkerWrapper;

    /* At exit we do not want to destruct the WorkerThreadAccess instance,
       because it is possible that callbacks will try to use it after its dtor
       would have been called.  Instead, we make sure to reset it, which will
       stop the worker thread.  */
    std::atexit ([] () { get_worker_thread ().stop (); });
    return wrapper;
  }();
  return WorkerThreadAccess (wp->mutex, wp->thread);
}

decltype (CoreApiTable::hsa_executable_freeze_fn)
    original_hsa_executable_freeze
    = {};

decltype (CoreApiTable::hsa_executable_destroy_fn)
    original_hsa_executable_destroy
    = {};

hsa_status_t
debug_agent_hsa_executable_freeze (hsa_executable_t executable,
                                   const char *options)
{
  auto v = original_hsa_executable_freeze (executable, options);

  get_worker_thread ().update_code_object_list ();
  return v;
}

hsa_status_t
debug_agent_hsa_executable_destroy (hsa_executable_t executable)
{
  auto v = original_hsa_executable_destroy (executable);

  get_worker_thread ().update_code_object_list ();
  return v;
}

} /* namespace.  */

extern "C" void __attribute__ ((visibility ("default"))) OnUnload ();
extern "C" bool __attribute__ ((visibility ("default")))
OnLoad (void *table, uint64_t runtime_version, uint64_t failed_tool_count,
        const char *const *failed_tool_names)
{
  bool disable_sigquit{ false };

  set_log_level (log_level_t::warning);

  std::istringstream args_stream;
  if (const char *env = ::getenv ("ROCM_DEBUG_AGENT_OPTIONS"))
    args_stream.str (env);

  std::vector<char *> args = { strdup ("rocm-debug-agent") };
  std::transform (
      std::istream_iterator<std::string> (args_stream),
      std::istream_iterator<std::string> (), std::back_inserter (args),
      [] (const std::string &str) { return strdup (str.c_str ()); });

  char *const *argv = const_cast<char *const *> (args.data ());
  int argc = args.size ();

  static struct option options[]
      = { { "all", no_argument, nullptr, 'a' },
          { "disable-linux-signals", no_argument, nullptr, 'd' },
          { "log-level", required_argument, nullptr, 'l' },
          { "output", required_argument, nullptr, 'o' },
          { "save-code-objects", optional_argument, nullptr, 's' },
          { "precise-memory", no_argument, nullptr, 'p' },
          { "precise-alu-exceptions", no_argument, nullptr, 'e' },
          { "help", no_argument, nullptr, 'h' },
          { 0 } };

  /* We use getopt_long locally, so make sure to preserve and reset the
     global optind.  */
  int saved_optind = optind;
  optind = 1;

  while (int c = getopt_long (argc, argv, ":as::o:dpel:h", options, nullptr))
    {
      if (c == -1)
        break;

      std::optional<std::string> argument;

      if (!optarg && optind < argc && *argv[optind] != '-')
        optarg = argv[optind++];

      if (optarg)
        argument.emplace (optarg);

      switch (c)
        {
        case 'a': /* -a or --all  */
          g_all_wavefronts = true;
          break;

        case 'd': /* -d or --disable-linux-signals  */
          disable_sigquit = true;
          break;

        case 'p': /* -p or --precise-memory  */
          g_precise_emmory = true;
          break;

        case 'e': /* -e or --precise-alu-exceptions  */
          g_precise_alu_exceptions = true;
          break;

        case 'l': /* -l or --log-level  */
          if (!argument)
            print_usage ();

          if (argument == "none")
            set_log_level (log_level_t::none);
          else if (argument == "verbose")
            set_log_level (log_level_t::verbose);
          else if (argument == "info")
            set_log_level (log_level_t::info);
          else if (argument == "warning")
            set_log_level (log_level_t::warning);
          else if (argument == "error")
            set_log_level (log_level_t::error);
          else
            print_usage ();
          break;

        case 's': /* -s or --save-code-objects  */
          if (argument)
            {
              struct stat path_stat;
              if (stat (argument->c_str (), &path_stat) == -1
                  || !S_ISDIR (path_stat.st_mode))
                {
                  std::cerr
                      << "error: Cannot access code object save directory `"
                      << *argument << "'" << std::endl;
                  print_usage ();
                }

              g_code_objects_dir = *argument;
            }
          else
            {
              g_code_objects_dir = ".";
            }
          break;

        case 'o': /* -o or --output  */
          if (!argument)
            print_usage ();

          agent_out.open (*argument);
          if (!agent_out.is_open ())
            {
              std::cerr << "could not open `" << *argument << "'" << std::endl;
              abort ();
            }
          break;

        case '?': /* Unrecognized option  */
        case 'h': /* -h or --help */
        default:
          print_usage ();
        }
    }

  /* Restore the global optind.  */
  optind = saved_optind;

  std::for_each (args.begin (), args.end (), [] (char *str) { free (str); });

  if (!agent_out.is_open ())
    {
      agent_out.copyfmt (std::cerr);
      agent_out.clear (std::cerr.rdstate ());
      agent_out.basic_ios<char>::rdbuf (std::cerr.rdbuf ());
    }

  get_worker_thread ().start ();

  if (!disable_sigquit)
    {
      struct sigaction sig_action;

      memset (&sig_action, '\0', sizeof (sig_action));
      sigemptyset (&sig_action.sa_mask);

      sig_action.sa_sigaction = [] (int signal, siginfo_t *, void *) {
        agent_out << std::endl;
        get_worker_thread ().query_print_waves ();
      };

      /* Install a SIGQUIT (Ctrl-\) handler.  */
      sig_action.sa_flags = SA_RESTART;
      sigaction (SIGQUIT, &sig_action, nullptr);
    }

  CoreApiTable *core_table = reinterpret_cast<HsaApiTable *> (table)->core_;

  original_hsa_executable_freeze = core_table->hsa_executable_freeze_fn;
  original_hsa_executable_destroy = core_table->hsa_executable_destroy_fn;

  core_table->hsa_executable_freeze_fn = debug_agent_hsa_executable_freeze;
  core_table->hsa_executable_destroy_fn = debug_agent_hsa_executable_destroy;

  return true;
}

void
OnUnload ()
{
  get_worker_thread ().stop ();
}
