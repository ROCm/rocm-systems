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

#include "logging.h"
#include "architecture.h"
#include "callbacks.h"
#include "code_object.h"
#include "debug.h"
#include "initialization.h"
#include "memory.h"
#include "process.h"
#include "register.h"
#include "utils.h"

#include <cstdarg>
#include <optional>
#include <string>

namespace amd::dbgapi
{

amd_dbgapi_log_level_t log_level = AMD_DBGAPI_LOG_LEVEL_NONE;
size_t detail::log_indent_depth = 0;

void
vlog (amd_dbgapi_log_level_t level, const char *format, va_list va)
{
  if (level > log_level)
    return;

  std::string message;

  if (level == AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR)
    message.append ("fatal error: ");
  else if (level == AMD_DBGAPI_LOG_LEVEL_WARNING)
    message.append ("warning: ");
  else if (detail::log_indent_depth)
    message.append (std::string (detail::log_indent_depth * 3, ' '));

  message.append (string_vprintf (format, va));

  if (detail::process_callbacks.log_message != nullptr)
    (*detail::process_callbacks.log_message) (level, message.c_str ());
}

namespace detail
{

void
log (amd_dbgapi_log_level_t level, const char *format, ...)
{
  va_list va;
  va_start (va, format);
  vlog (level, format, va);
  va_end (va);
}

} /* namespace detail */

template <>
std::string
to_string (amd_dbgapi_architecture_id_t architecture_id)
{
  if (architecture_id == AMD_DBGAPI_ARCHITECTURE_NONE)
    return "ARCHITECTURE_NONE";

  std::string str
    = string_printf ("architecture_%" PRId64, architecture_id.handle);

  if (const architecture_t *architecture = find (architecture_id);
      architecture)
    str += " <" + architecture->name () + ">";

  return str;
}

template <>
std::string
to_string (amd_dbgapi_process_id_t process_id)
{
  if (process_id == AMD_DBGAPI_PROCESS_NONE)
    return "PROCESS_NONE";

  return string_printf ("process_%" PRId64, process_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_code_object_id_t code_object_id)
{
  if (code_object_id == AMD_DBGAPI_CODE_OBJECT_NONE)
    return "CODE_OBJECT_NONE";

  std::string str
    = string_printf ("code_object_%" PRId64, code_object_id.handle);

  if (code_object_t *code_object = find (code_object_id); code_object)
    str += " <\"" + code_object->uri () + "\">";

  return str;
}

template <>
std::string
to_string (amd_dbgapi_agent_id_t agent_id)
{
  if (agent_id == AMD_DBGAPI_AGENT_NONE)
    return "AGENT_NONE";

  return string_printf ("agent_%" PRId64, agent_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_queue_id_t queue_id)
{
  if (queue_id == AMD_DBGAPI_QUEUE_NONE)
    return "QUEUE_NONE";

  return string_printf ("queue_%" PRId64, queue_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_workgroup_id_t workgroup_id)
{
  if (workgroup_id == AMD_DBGAPI_WORKGROUP_NONE)
    return "WORKGROUP_NONE";

  return string_printf ("workgroup_%" PRId64, workgroup_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_dispatch_id_t dispatch_id)
{
  if (dispatch_id == AMD_DBGAPI_DISPATCH_NONE)
    return "DISPATCH_NONE";

  return string_printf ("dispatch_%" PRId64, dispatch_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_wave_id_t wave_id)
{
  if (wave_id == AMD_DBGAPI_WAVE_NONE)
    return "WAVE_NONE";

  return string_printf ("wave_%" PRId64, wave_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_displaced_stepping_id_t displaced_stepping_id)
{
  if (displaced_stepping_id == AMD_DBGAPI_DISPLACED_STEPPING_NONE)
    return "DISPLACED_STEPPING_NONE";

  return string_printf ("displaced_stepping_%" PRId64,
                        displaced_stepping_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_register_class_id_t register_class_id)
{
  if (register_class_id == AMD_DBGAPI_REGISTER_CLASS_NONE)
    return "REGISTER_CLASS_NONE";

  std::string str
    = string_printf ("register_class_%" PRId64, register_class_id.handle);

  if (const register_class_t *register_class = find (register_class_id);
      register_class)
    str += " <" + register_class->architecture ().name ()
           + "::" + register_class->name () + ">";

  return str;
}

template <>
std::string
to_string (amd_dbgapi_register_id_t register_id)
{
  if (register_id == AMD_DBGAPI_REGISTER_NONE)
    return "REGISTER_NONE";

  auto regnum = architecture_t::register_id_to_regnum (register_id);

  const architecture_t *architecture
    = architecture_t::register_id_to_architecture (register_id);

  std::string str = string_printf ("register_%" PRId64, register_id.handle);

  if (architecture != nullptr && regnum
      && architecture->is_register_available (*regnum))
    str += " <" + architecture->name ()
           + "::" + architecture->register_name (*regnum) + ">";

  return str;
}

template <>
std::string
to_string (amd_dbgapi_address_class_id_t address_class_id)
{
  if (address_class_id == AMD_DBGAPI_ADDRESS_CLASS_NONE)
    return "ADDRESS_CLASS_NONE";

  std::string str
    = string_printf ("address_class_%" PRId64, address_class_id.handle);

  if (const address_class_t *address_class = find (address_class_id);
      address_class)
    str += " <" + address_class->name () + ">";

  return str;
}

template <>
std::string
to_string (amd_dbgapi_address_space_id_t address_space_id)
{
  if (address_space_id == AMD_DBGAPI_ADDRESS_SPACE_NONE)
    return "ADDRESS_SPACE_NONE";

  std::string str
    = string_printf ("address_space_%" PRId64, address_space_id.handle);

  if (const address_space_t *address_space = find (address_space_id);
      address_space)
    str += " <" + address_space->name () + ">";

  return str;
}

template <>
std::string
to_string (amd_dbgapi_event_id_t event_id)
{
  if (event_id == AMD_DBGAPI_EVENT_NONE)
    return "EVENT_NONE";

  return string_printf ("event_%" PRId64, event_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_breakpoint_id_t breakpoint_id)
{
  if (breakpoint_id == AMD_DBGAPI_BREAKPOINT_NONE)
    return "BREAKPOINT_NONE";

  return string_printf ("breakpoint_%" PRId64, breakpoint_id.handle);
}

#define CASE(x)                                                               \
  case AMD_DBGAPI_##x:                                                        \
    return #x

template <>
std::string
to_string (amd_dbgapi_agent_state_t agent_state)
{
  switch (agent_state)
    {
      CASE (AGENT_STATE_SUPPORTED);
      CASE (AGENT_STATE_NOT_SUPPORTED);
    }
  return to_string (make_hex (agent_state));
}

template <>
std::string
to_string (amd_dbgapi_changed_t changed)
{
  switch (changed)
    {
      CASE (CHANGED_NO);
      CASE (CHANGED_YES);
    }
  return to_string (make_hex (changed));
}

template <>
std::string
to_string (amd_dbgapi_status_t status)
{
  switch (status)
    {
      CASE (STATUS_SUCCESS);
      CASE (STATUS_ERROR);
      CASE (STATUS_FATAL);
      CASE (STATUS_ERROR_NOT_IMPLEMENTED);
      CASE (STATUS_ERROR_NOT_AVAILABLE);
      CASE (STATUS_ERROR_NOT_SUPPORTED);
      CASE (STATUS_ERROR_INVALID_ARGUMENT);
      CASE (STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
      CASE (STATUS_ERROR_ALREADY_INITIALIZED);
      CASE (STATUS_ERROR_NOT_INITIALIZED);
      CASE (STATUS_ERROR_RESTRICTION);
      CASE (STATUS_ERROR_ALREADY_ATTACHED);
      CASE (STATUS_ERROR_INVALID_ARCHITECTURE_ID);
      CASE (STATUS_ERROR_ILLEGAL_INSTRUCTION);
      CASE (STATUS_ERROR_INVALID_CODE_OBJECT_ID);
      CASE (STATUS_ERROR_INVALID_ELF_AMDGPU_MACHINE);
      CASE (STATUS_ERROR_INVALID_PROCESS_ID);
      CASE (STATUS_ERROR_PROCESS_EXITED);
      CASE (STATUS_ERROR_INVALID_AGENT_ID);
      CASE (STATUS_ERROR_INVALID_QUEUE_ID);
      CASE (STATUS_ERROR_INVALID_DISPATCH_ID);
      CASE (STATUS_ERROR_INVALID_WAVE_ID);
      CASE (STATUS_ERROR_WAVE_NOT_STOPPED);
      CASE (STATUS_ERROR_WAVE_STOPPED);
      CASE (STATUS_ERROR_WAVE_OUTSTANDING_STOP);
      CASE (STATUS_ERROR_WAVE_NOT_RESUMABLE);
      CASE (STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID);
      CASE (STATUS_ERROR_DISPLACED_STEPPING_BUFFER_NOT_AVAILABLE);
      CASE (STATUS_ERROR_DISPLACED_STEPPING_ACTIVE);
      CASE (STATUS_ERROR_RESUME_DISPLACED_STEPPING);
      CASE (STATUS_ERROR_INVALID_WATCHPOINT_ID);
      CASE (STATUS_ERROR_NO_WATCHPOINT_AVAILABLE);
      CASE (STATUS_ERROR_INVALID_REGISTER_CLASS_ID);
      CASE (STATUS_ERROR_INVALID_REGISTER_ID);
      CASE (STATUS_ERROR_INVALID_LANE_ID);
      CASE (STATUS_ERROR_INVALID_ADDRESS_CLASS_ID);
      CASE (STATUS_ERROR_INVALID_ADDRESS_SPACE_ID);
      CASE (STATUS_ERROR_MEMORY_ACCESS);
      CASE (STATUS_ERROR_INVALID_ADDRESS_SPACE_CONVERSION);
      CASE (STATUS_ERROR_INVALID_EVENT_ID);
      CASE (STATUS_ERROR_INVALID_BREAKPOINT_ID);
      CASE (STATUS_ERROR_CLIENT_CALLBACK);
      CASE (STATUS_ERROR_INVALID_CLIENT_PROCESS_ID);
      CASE (STATUS_ERROR_SYMBOL_NOT_FOUND);
      CASE (STATUS_ERROR_REGISTER_NOT_AVAILABLE);
      CASE (STATUS_ERROR_INVALID_WORKGROUP_ID);
      CASE (STATUS_ERROR_INCOMPATIBLE_PROCESS_STATE);
      CASE (STATUS_ERROR_PROCESS_FROZEN);
      CASE (STATUS_ERROR_PROCESS_ALREADY_FROZEN);
      CASE (STATUS_ERROR_PROCESS_NOT_FROZEN);
      CASE (STATUS_ERROR_MEMORY_UNAVAILABLE);
    }
  return to_string (make_hex (status));
}

template <>
std::string
to_string (amd_dbgapi_architecture_info_t architecture_info)
{
  switch (architecture_info)
    {
      CASE (ARCHITECTURE_INFO_NAME);
      CASE (ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE);
      CASE (ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE);
      CASE (ARCHITECTURE_INFO_MINIMUM_INSTRUCTION_ALIGNMENT);
      CASE (ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE);
      CASE (ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION);
      CASE (ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_PC_ADJUST);
      CASE (ARCHITECTURE_INFO_PC_REGISTER);
    }
  return to_string (make_hex (architecture_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_architecture_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_ARCHITECTURE_INFO_NAME:
      return to_string (make_ref (static_cast<char *const *> (value)));
    case AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE:
      return to_string (make_ref (static_cast<const uint32_t *> (value)));
    case AMD_DBGAPI_ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE:
    case AMD_DBGAPI_ARCHITECTURE_INFO_MINIMUM_INSTRUCTION_ALIGNMENT:
    case AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE:
    case AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_PC_ADJUST:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_size_t *> (value)));
    case AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION:
      return to_string (make_hex (
        make_ref (make_ref (static_cast<uint8_t *const *> (value)), 4)));
    case AMD_DBGAPI_ARCHITECTURE_INFO_PC_REGISTER:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_register_id_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_architecture_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_instruction_kind_t instruction_kind)
{
  switch (instruction_kind)
    {
      CASE (INSTRUCTION_KIND_UNKNOWN);
      CASE (INSTRUCTION_KIND_SEQUENTIAL);
      CASE (INSTRUCTION_KIND_DIRECT_BRANCH);
      CASE (INSTRUCTION_KIND_DIRECT_BRANCH_CONDITIONAL);
      CASE (INSTRUCTION_KIND_INDIRECT_BRANCH_REGISTER_PAIR);
      CASE (INSTRUCTION_KIND_INDIRECT_BRANCH_CONDITIONAL_REGISTER_PAIR);
      CASE (INSTRUCTION_KIND_DIRECT_CALL_REGISTER_PAIR);
      CASE (INSTRUCTION_KIND_INDIRECT_CALL_REGISTER_PAIRS);
      CASE (INSTRUCTION_KIND_TERMINATE);
      CASE (INSTRUCTION_KIND_TRAP);
      CASE (INSTRUCTION_KIND_HALT);
      CASE (INSTRUCTION_KIND_BARRIER);
      CASE (INSTRUCTION_KIND_SLEEP);
      CASE (INSTRUCTION_KIND_SPECIAL);
    }
  return to_string (make_hex (instruction_kind));
}

namespace
{

inline std::string
one_instruction_property_to_string (
  amd_dbgapi_instruction_properties_t instruction_property)
{
  switch (instruction_property)
    {
      CASE (INSTRUCTION_PROPERTY_NONE);
    }
  return to_string (make_hex (instruction_property));
}

} /* namespace */

template <>
std::string
to_string (amd_dbgapi_instruction_properties_t instruction_properties)
{
  std::string str;

  if (!instruction_properties)
    return one_instruction_property_to_string (instruction_properties);

  while (instruction_properties)
    {
      amd_dbgapi_instruction_properties_t one_bit
        = instruction_properties
          ^ (instruction_properties & (instruction_properties - 1));

      if (!str.empty ())
        str += " | ";
      str += one_instruction_property_to_string (one_bit);

      instruction_properties ^= one_bit;
    }

  return str;
}

namespace
{

inline std::string
one_register_property_to_string (
  amd_dbgapi_register_properties_t register_property)
{
  switch (register_property)
    {
      CASE (REGISTER_PROPERTY_NONE);
      CASE (REGISTER_PROPERTY_READONLY_BITS);
      CASE (REGISTER_PROPERTY_VOLATILE);
      CASE (REGISTER_PROPERTY_INVALIDATE_VOLATILE);
    }
  return to_string (make_hex (register_property));
}

} /* namespace */

template <>
std::string
to_string (amd_dbgapi_register_properties_t register_properties)
{
  std::string str;

  if (!register_properties)
    return one_register_property_to_string (register_properties);

  while (register_properties)
    {
      amd_dbgapi_register_properties_t one_bit
        = register_properties
          ^ (register_properties & (register_properties - 1));

      if (!str.empty ())
        str += " | ";
      str += one_register_property_to_string (one_bit);

      register_properties ^= one_bit;
    }

  return str;
}

template <>
std::string
to_string (amd_dbgapi_direct_call_register_pair_information_t information)
{
  return to_string (make_hex (information.target_address)) + ",["
         + to_string (information.saved_return_address_register[0]) + ","
         + to_string (information.saved_return_address_register[1]) + "]";
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_instruction_kind_t> ref)
{
  auto [kind, information] = ref;
  switch (kind)
    {
    case AMD_DBGAPI_INSTRUCTION_KIND_UNKNOWN:
    case AMD_DBGAPI_INSTRUCTION_KIND_SEQUENTIAL:
    case AMD_DBGAPI_INSTRUCTION_KIND_TERMINATE:
    case AMD_DBGAPI_INSTRUCTION_KIND_HALT:
    case AMD_DBGAPI_INSTRUCTION_KIND_BARRIER:
    case AMD_DBGAPI_INSTRUCTION_KIND_SLEEP:
    case AMD_DBGAPI_INSTRUCTION_KIND_SPECIAL:
      return {};
    case AMD_DBGAPI_INSTRUCTION_KIND_DIRECT_BRANCH:
    case AMD_DBGAPI_INSTRUCTION_KIND_DIRECT_BRANCH_CONDITIONAL:
      return to_string (make_hex (make_ref (
        make_ref (static_cast<const amd_dbgapi_global_address_t *const *> (
          information)))));
    case AMD_DBGAPI_INSTRUCTION_KIND_INDIRECT_BRANCH_REGISTER_PAIR:
    case AMD_DBGAPI_INSTRUCTION_KIND_INDIRECT_BRANCH_CONDITIONAL_REGISTER_PAIR:
      return to_string (make_ref (
        make_ref (
          static_cast<const amd_dbgapi_register_id_t *const *> (information)),
        2));
    case AMD_DBGAPI_INSTRUCTION_KIND_DIRECT_CALL_REGISTER_PAIR:
      return to_string (make_ref (make_ref (
        static_cast<const amd_dbgapi_direct_call_register_pair_information_t
                      *const *> (information))));
    case AMD_DBGAPI_INSTRUCTION_KIND_INDIRECT_CALL_REGISTER_PAIRS:
      return to_string (make_ref (
        make_ref (
          static_cast<const amd_dbgapi_register_id_t *const *> (information)),
        4));
    case AMD_DBGAPI_INSTRUCTION_KIND_TRAP:
      return to_string (make_ref (
        make_ref (static_cast<const uint64_t *const *> (information))));
    }
  fatal_error ("unhandled amd_dbgapi_instruction_kind_t kind (%s)",
               to_cstring (kind));
}

template <>
std::string
to_string (amd_dbgapi_client_process_info_t client_process_info)
{
  switch (client_process_info)
    {
      CASE (CLIENT_PROCESS_INFO_OS_PID);
      CASE (CLIENT_PROCESS_INFO_CORE_STATE);
    }
  return to_string (make_hex (client_process_info));
}

template <>
std::string
to_string (amd_dbgapi_core_state_data_t core_state)
{
  return string_printf (
    "{endian %s, data %s}",
    core_state.endianness == AMD_DBGAPI_ENDIAN_BIG ? "BIG" : "LITTLE",
    to_cstring (make_hex (make_ref (core_state.data, core_state.size))));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_client_process_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_CLIENT_PROCESS_INFO_OS_PID:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_os_process_id_t *> (value)));
    case AMD_DBGAPI_CLIENT_PROCESS_INFO_CORE_STATE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_core_state_data_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_client_process_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_process_info_t process_info)
{
  switch (process_info)
    {
      CASE (PROCESS_INFO_NOTIFIER);
      CASE (PROCESS_INFO_WATCHPOINT_COUNT);
      CASE (PROCESS_INFO_WATCHPOINT_SHARE);
      CASE (PROCESS_INFO_PRECISE_MEMORY_SUPPORTED);
      CASE (PROCESS_INFO_OS_ID);
      CASE (PROCESS_INFO_CORE_STATE);
      CASE (PROCESS_INFO_PRECISE_ALU_EXCEPTIONS_SUPPORTED);
    }
  return to_string (make_hex (process_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_process_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_PROCESS_INFO_NOTIFIER:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_notifier_t *> (value)));
    case AMD_DBGAPI_PROCESS_INFO_WATCHPOINT_COUNT:
      return to_string (make_ref (static_cast<const size_t *> (value)));
    case AMD_DBGAPI_PROCESS_INFO_WATCHPOINT_SHARE:
      return to_string (make_ref (
        static_cast<const amd_dbgapi_watchpoint_share_kind_t *> (value)));
    case AMD_DBGAPI_PROCESS_INFO_PRECISE_MEMORY_SUPPORTED:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_memory_precision_t *> (value)));
    case AMD_DBGAPI_PROCESS_INFO_OS_ID:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_os_process_id_t *> (value)));
    case AMD_DBGAPI_PROCESS_INFO_CORE_STATE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_core_state_data_t *> (value)));
    case AMD_DBGAPI_PROCESS_INFO_PRECISE_ALU_EXCEPTIONS_SUPPORTED:
      return to_string (make_ref (
        static_cast<const amd_dbgapi_alu_exceptions_precision_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_process_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_progress_t progress)
{
  switch (progress)
    {
      CASE (PROGRESS_NORMAL);
      CASE (PROGRESS_NO_FORWARD);
    }
  return to_string (make_hex (progress));
}

template <>
std::string
to_string (amd_dbgapi_wave_creation_t wave_creation)
{
  switch (wave_creation)
    {
      CASE (WAVE_CREATION_NORMAL);
      CASE (WAVE_CREATION_STOP);
    }
  return to_string (make_hex (wave_creation));
}

template <>
std::string
to_string (amd_dbgapi_watchpoint_info_t watchpoint_info)
{
  switch (watchpoint_info)
    {
      CASE (WATCHPOINT_INFO_PROCESS);
      CASE (WATCHPOINT_INFO_ADDRESS);
      CASE (WATCHPOINT_INFO_SIZE);
    }
  return to_string (make_hex (watchpoint_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_watchpoint_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_WATCHPOINT_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    case AMD_DBGAPI_WATCHPOINT_INFO_ADDRESS:
      return to_string (make_hex (
        make_ref (static_cast<const amd_dbgapi_global_address_t *> (value))));
    case AMD_DBGAPI_WATCHPOINT_INFO_SIZE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_size_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_watchpoint_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_code_object_info_t code_object_info)
{
  switch (code_object_info)
    {
      CASE (CODE_OBJECT_INFO_PROCESS);
      CASE (CODE_OBJECT_INFO_URI_NAME);
      CASE (CODE_OBJECT_INFO_LOAD_ADDRESS);
    }
  return to_string (make_hex (code_object_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_code_object_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_CODE_OBJECT_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    case AMD_DBGAPI_CODE_OBJECT_INFO_URI_NAME:
      return to_string (make_ref (static_cast<char *const *> (value)));
    case AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS:
      return to_string (
        make_hex (make_ref (static_cast<const ptrdiff_t *> (value))));
    }
  fatal_error ("unhandled amd_dbgapi_code_object_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_breakpoint_info_t breakpoint_info)
{
  switch (breakpoint_info)
    {
      CASE (BREAKPOINT_INFO_PROCESS);
    }
  return to_string (make_hex (breakpoint_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_breakpoint_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_BREAKPOINT_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_breakpoint_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_displaced_stepping_info_t displaced_stepping_info)
{
  switch (displaced_stepping_info)
    {
      CASE (DISPLACED_STEPPING_INFO_PROCESS);
    }
  return to_string (make_hex (displaced_stepping_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_displaced_stepping_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_displaced_stepping_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_agent_info_t agent_info)
{
  switch (agent_info)
    {
      CASE (AGENT_INFO_PROCESS);
      CASE (AGENT_INFO_NAME);
      CASE (AGENT_INFO_ARCHITECTURE);
      CASE (AGENT_INFO_STATE);
      CASE (AGENT_INFO_PCI_DOMAIN);
      CASE (AGENT_INFO_PCI_SLOT);
      CASE (AGENT_INFO_PCI_VENDOR_ID);
      CASE (AGENT_INFO_PCI_DEVICE_ID);
      CASE (AGENT_INFO_EXECUTION_UNIT_COUNT);
      CASE (AGENT_INFO_MAX_WAVES_PER_EXECUTION_UNIT);
      CASE (AGENT_INFO_OS_ID);
    }
  return to_string (make_hex (agent_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_agent_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_AGENT_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    case AMD_DBGAPI_AGENT_INFO_NAME:
      return to_string (make_ref (static_cast<char *const *> (value)));
    case AMD_DBGAPI_AGENT_INFO_ARCHITECTURE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_architecture_id_t *> (value)));
    case AMD_DBGAPI_AGENT_INFO_STATE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_agent_state_t *> (value)));
    case AMD_DBGAPI_AGENT_INFO_PCI_DOMAIN:
    case AMD_DBGAPI_AGENT_INFO_PCI_SLOT:
      return to_string (
        make_hex (make_ref (static_cast<const uint16_t *> (value))));
    case AMD_DBGAPI_AGENT_INFO_PCI_VENDOR_ID:
    case AMD_DBGAPI_AGENT_INFO_PCI_DEVICE_ID:
      return to_string (
        make_hex (make_ref (static_cast<const uint32_t *> (value))));
    case AMD_DBGAPI_AGENT_INFO_EXECUTION_UNIT_COUNT:
    case AMD_DBGAPI_AGENT_INFO_MAX_WAVES_PER_EXECUTION_UNIT:
      return to_string (make_ref (static_cast<const size_t *> (value)));
    case AMD_DBGAPI_AGENT_INFO_OS_ID:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_os_agent_id_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_agent_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_queue_info_t queue_info)
{
  switch (queue_info)
    {
      CASE (QUEUE_INFO_AGENT);
      CASE (QUEUE_INFO_PROCESS);
      CASE (QUEUE_INFO_ARCHITECTURE);
      CASE (QUEUE_INFO_TYPE);
      CASE (QUEUE_INFO_STATE);
      CASE (QUEUE_INFO_ERROR_REASON);
      CASE (QUEUE_INFO_ADDRESS);
      CASE (QUEUE_INFO_SIZE);
      CASE (QUEUE_INFO_OS_ID);
    }
  return to_string (make_hex (queue_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_queue_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_QUEUE_INFO_AGENT:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_agent_id_t *> (value)));
    case AMD_DBGAPI_QUEUE_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    case AMD_DBGAPI_QUEUE_INFO_ARCHITECTURE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_architecture_id_t *> (value)));
    case AMD_DBGAPI_QUEUE_INFO_TYPE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_os_queue_type_t *> (value)));
    case AMD_DBGAPI_QUEUE_INFO_STATE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_queue_state_t *> (value)));
    case AMD_DBGAPI_QUEUE_INFO_ERROR_REASON:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_exceptions_t *> (value)));
    case AMD_DBGAPI_QUEUE_INFO_ADDRESS:
      return to_string (make_hex (
        make_ref (static_cast<const amd_dbgapi_global_address_t *> (value))));
    case AMD_DBGAPI_QUEUE_INFO_SIZE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_size_t *> (value)));
    case AMD_DBGAPI_QUEUE_INFO_OS_ID:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_os_queue_id_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_queue_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_os_queue_type_t queue_type)
{
  switch (queue_type)
    {
      CASE (OS_QUEUE_TYPE_UNKNOWN);
      CASE (OS_QUEUE_TYPE_HSA_AQL);
      CASE (OS_QUEUE_TYPE_AMD_PM4);
      CASE (OS_QUEUE_TYPE_AMD_SDMA);
      CASE (OS_QUEUE_TYPE_AMD_SDMA_XGMI);
    }
  return to_string (make_hex (queue_type));
}

template <>
std::string
to_string (amd_dbgapi_queue_state_t queue_state)
{
  switch (queue_state)
    {
      CASE (QUEUE_STATE_VALID);
      CASE (QUEUE_STATE_ERROR);
    }
  return to_string (make_hex (queue_state));
}

namespace
{

inline std::string
one_queue_error_reason_to_string (amd_dbgapi_exceptions_t queue_error_reason)
{
  dbgapi_assert (!(queue_error_reason & (queue_error_reason - 1))
                 && "only 1 bit");

  switch (queue_error_reason)
    {
      CASE (EXCEPTION_NONE);
      CASE (EXCEPTION_WAVE_ABORT);
      CASE (EXCEPTION_WAVE_TRAP);
      CASE (EXCEPTION_WAVE_MATH_ERROR);
      CASE (EXCEPTION_WAVE_ILLEGAL_INSTRUCTION);
      CASE (EXCEPTION_WAVE_MEMORY_VIOLATION);
      CASE (EXCEPTION_WAVE_ADDRESS_ERROR);
      CASE (EXCEPTION_PACKET_DISPATCH_DIM_INVALID);
      CASE (EXCEPTION_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID);
      CASE (EXCEPTION_PACKET_DISPATCH_CODE_INVALID);
      CASE (EXCEPTION_PACKET_UNSUPPORTED);
      CASE (EXCEPTION_PACKET_DISPATCH_WORKGROUP_SIZE_INVALID);
      CASE (EXCEPTION_PACKET_DISPATCH_REGISTER_COUNT_TOO_LARGE);
      CASE (EXCEPTION_PACKET_VENDOR_UNSUPPORTED);
      CASE (EXCEPTION_QUEUE_PREEMPTION_ERROR);
    }
  return to_string (make_hex (queue_error_reason));
}

} /* namespace */

template <>
std::string
to_string (amd_dbgapi_exceptions_t queue_error_reason)
{
  std::string str;

  if (!queue_error_reason)
    return one_queue_error_reason_to_string (queue_error_reason);

  while (queue_error_reason)
    {
      amd_dbgapi_exceptions_t one_bit
        = queue_error_reason ^ (queue_error_reason & (queue_error_reason - 1));

      if (!str.empty ())
        str += " | ";
      str += one_queue_error_reason_to_string (one_bit);

      queue_error_reason ^= one_bit;
    }

  return str;
}

template <>
std::string
to_string (amd_dbgapi_dispatch_info_t dispatch_info)
{
  switch (dispatch_info)
    {
      CASE (DISPATCH_INFO_QUEUE);
      CASE (DISPATCH_INFO_AGENT);
      CASE (DISPATCH_INFO_PROCESS);
      CASE (DISPATCH_INFO_ARCHITECTURE);
      CASE (DISPATCH_INFO_OS_QUEUE_PACKET_ID);
      CASE (DISPATCH_INFO_BARRIER);
      CASE (DISPATCH_INFO_ACQUIRE_FENCE);
      CASE (DISPATCH_INFO_RELEASE_FENCE);
      CASE (DISPATCH_INFO_GRID_DIMENSIONS);
      CASE (DISPATCH_INFO_WORKGROUP_SIZES);
      CASE (DISPATCH_INFO_GRID_SIZES);
      CASE (DISPATCH_INFO_PRIVATE_SEGMENT_SIZE);
      CASE (DISPATCH_INFO_GROUP_SEGMENT_SIZE);
      CASE (DISPATCH_INFO_KERNEL_ARGUMENT_SEGMENT_ADDRESS);
      CASE (DISPATCH_INFO_KERNEL_DESCRIPTOR_ADDRESS);
      CASE (DISPATCH_INFO_KERNEL_CODE_ENTRY_ADDRESS);
      CASE (DISPATCH_INFO_KERNEL_COMPLETION_ADDRESS);
    }
  return to_string (make_hex (dispatch_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_dispatch_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_DISPATCH_INFO_QUEUE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_queue_id_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_AGENT:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_agent_id_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_ARCHITECTURE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_architecture_id_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_OS_QUEUE_PACKET_ID:
      return to_string (make_ref (
        static_cast<const amd_dbgapi_os_queue_packet_id_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_BARRIER:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_dispatch_barrier_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_ACQUIRE_FENCE:
    case AMD_DBGAPI_DISPATCH_INFO_RELEASE_FENCE:
      return to_string (make_ref (
        static_cast<const amd_dbgapi_dispatch_fence_scope_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_GRID_DIMENSIONS:
      return to_string (make_ref (static_cast<const uint32_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES:
      return to_string (make_ref (static_cast<const uint16_t *> (value), 3));
    case AMD_DBGAPI_DISPATCH_INFO_GRID_SIZES:
      return to_string (make_ref (static_cast<const uint32_t *> (value), 3));
    case AMD_DBGAPI_DISPATCH_INFO_PRIVATE_SEGMENT_SIZE:
    case AMD_DBGAPI_DISPATCH_INFO_GROUP_SEGMENT_SIZE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_size_t *> (value)));
    case AMD_DBGAPI_DISPATCH_INFO_KERNEL_ARGUMENT_SEGMENT_ADDRESS:
    case AMD_DBGAPI_DISPATCH_INFO_KERNEL_DESCRIPTOR_ADDRESS:
    case AMD_DBGAPI_DISPATCH_INFO_KERNEL_CODE_ENTRY_ADDRESS:
    case AMD_DBGAPI_DISPATCH_INFO_KERNEL_COMPLETION_ADDRESS:
      return to_string (make_hex (
        make_ref (static_cast<const amd_dbgapi_global_address_t *> (value))));
    }
  fatal_error ("unhandled amd_dbgapi_dispatch_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_dispatch_barrier_t dispatch_barrier)
{
  switch (dispatch_barrier)
    {
      CASE (DISPATCH_BARRIER_NONE);
      CASE (DISPATCH_BARRIER_PRESENT);
    }
  return to_string (make_hex (dispatch_barrier));
}

template <>
std::string
to_string (amd_dbgapi_dispatch_fence_scope_t dispatch_fence_scope)
{
  switch (dispatch_fence_scope)
    {
      CASE (DISPATCH_FENCE_SCOPE_NONE);
      CASE (DISPATCH_FENCE_SCOPE_AGENT);
      CASE (DISPATCH_FENCE_SCOPE_SYSTEM);
    }
  return to_string (make_hex (dispatch_fence_scope));
}

template <>
std::string
to_string (amd_dbgapi_wave_info_t wave_info)
{
  switch (wave_info)
    {
      CASE (WAVE_INFO_STATE);
      CASE (WAVE_INFO_STOP_REASON);
      CASE (WAVE_INFO_WATCHPOINTS);
      CASE (WAVE_INFO_WORKGROUP);
      CASE (WAVE_INFO_DISPATCH);
      CASE (WAVE_INFO_QUEUE);
      CASE (WAVE_INFO_AGENT);
      CASE (WAVE_INFO_PROCESS);
      CASE (WAVE_INFO_ARCHITECTURE);
      CASE (WAVE_INFO_PC);
      CASE (WAVE_INFO_EXEC_MASK);
      CASE (WAVE_INFO_WORKGROUP_COORD);
      CASE (WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP);
      CASE (WAVE_INFO_LANE_COUNT);
    }
  return to_string (make_hex (wave_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_wave_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_WAVE_INFO_STATE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_wave_state_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_STOP_REASON:
      return to_string (make_ref (
        static_cast<const amd_dbgapi_wave_stop_reasons_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_WATCHPOINTS:
      {
        const auto *list
          = static_cast<const amd_dbgapi_watchpoint_list_t *> (value);
        return string_printf (
          "[%zu,%s]@%p", list->count,
          to_cstring (make_ref (list->watchpoint_ids, list->count)),
          static_cast<const void *> (list));
      }
    case AMD_DBGAPI_WAVE_INFO_WORKGROUP:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_workgroup_id_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_DISPATCH:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_dispatch_id_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_QUEUE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_queue_id_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_AGENT:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_agent_id_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_ARCHITECTURE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_architecture_id_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_PC:
      return to_string (make_hex (
        make_ref (static_cast<const amd_dbgapi_global_address_t *> (value))));
    case AMD_DBGAPI_WAVE_INFO_EXEC_MASK:
      return to_string (
        make_hex (make_ref (static_cast<const uint64_t *> (value))));
    case AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD:
      return to_string (make_ref (static_cast<const uint32_t *> (value), 3));
    case AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP:
      return to_string (make_ref (static_cast<const uint32_t *> (value)));
    case AMD_DBGAPI_WAVE_INFO_LANE_COUNT:
      return to_string (make_ref (static_cast<const size_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_wave_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_wave_state_t wave_state)
{
  switch (wave_state)
    {
      CASE (WAVE_STATE_RUN);
      CASE (WAVE_STATE_SINGLE_STEP);
      CASE (WAVE_STATE_STOP);
    }
  return to_string (make_hex (wave_state));
}

namespace
{

inline std::string
one_stop_reason_to_string (amd_dbgapi_wave_stop_reasons_t stop_reason)
{
  dbgapi_assert (!(stop_reason & (stop_reason - 1)) && "only 1 bit");

  switch (stop_reason)
    {
      CASE (WAVE_STOP_REASON_NONE);
      CASE (WAVE_STOP_REASON_BREAKPOINT);
      CASE (WAVE_STOP_REASON_WATCHPOINT);
      CASE (WAVE_STOP_REASON_SINGLE_STEP);
      CASE (WAVE_STOP_REASON_FP_INPUT_DENORMAL);
      CASE (WAVE_STOP_REASON_FP_DIVIDE_BY_0);
      CASE (WAVE_STOP_REASON_FP_OVERFLOW);
      CASE (WAVE_STOP_REASON_FP_UNDERFLOW);
      CASE (WAVE_STOP_REASON_FP_INEXACT);
      CASE (WAVE_STOP_REASON_FP_INVALID_OPERATION);
      CASE (WAVE_STOP_REASON_INT_DIVIDE_BY_0);
      CASE (WAVE_STOP_REASON_DEBUG_TRAP);
      CASE (WAVE_STOP_REASON_ASSERT_TRAP);
      CASE (WAVE_STOP_REASON_TRAP);
      CASE (WAVE_STOP_REASON_MEMORY_VIOLATION);
      CASE (WAVE_STOP_REASON_ADDRESS_ERROR);
      CASE (WAVE_STOP_REASON_ILLEGAL_INSTRUCTION);
      CASE (WAVE_STOP_REASON_ECC_ERROR);
      CASE (WAVE_STOP_REASON_FATAL_HALT);
    }
  return to_string (make_hex (stop_reason));
}

} /* namespace */

template <>
std::string
to_string (amd_dbgapi_wave_stop_reasons_t stop_reason)
{
  std::string str;

  if (!stop_reason)
    return one_stop_reason_to_string (stop_reason);

  while (stop_reason)
    {
      amd_dbgapi_wave_stop_reasons_t one_bit
        = stop_reason ^ (stop_reason & (stop_reason - 1));

      if (!str.empty ())
        str += " | ";
      str += one_stop_reason_to_string (one_bit);

      stop_reason ^= one_bit;
    }

  return str;
}

template <>
std::string
to_string (amd_dbgapi_resume_mode_t resume_mode)
{
  switch (resume_mode)
    {
      CASE (RESUME_MODE_NORMAL);
      CASE (RESUME_MODE_SINGLE_STEP);
    }
  return to_string (make_hex (resume_mode));
}

template <>
std::string
to_string (amd_dbgapi_watchpoint_id_t watchpoint_id)
{
  return string_printf ("watchpoint_%" PRId64, watchpoint_id.handle);
}

template <>
std::string
to_string (amd_dbgapi_watchpoint_share_kind_t watchpoint_share_kind)
{
  switch (watchpoint_share_kind)
    {
      CASE (WATCHPOINT_SHARE_KIND_UNSUPPORTED);
      CASE (WATCHPOINT_SHARE_KIND_UNSHARED);
      CASE (WATCHPOINT_SHARE_KIND_SHARED);
    }
  return to_string (make_hex (watchpoint_share_kind));
}

template <>
std::string
to_string (amd_dbgapi_watchpoint_kind_t watchpoint_kind)
{
  switch (watchpoint_kind)
    {
      CASE (WATCHPOINT_KIND_LOAD);
      CASE (WATCHPOINT_KIND_STORE_AND_RMW);
      CASE (WATCHPOINT_KIND_RMW);
      CASE (WATCHPOINT_KIND_ALL);
    }
  return to_string (make_hex (watchpoint_kind));
}

template <>
std::string
to_string (amd_dbgapi_register_class_info_t register_class_info)
{
  switch (register_class_info)
    {
      CASE (REGISTER_CLASS_INFO_ARCHITECTURE);
      CASE (REGISTER_CLASS_INFO_NAME);
    }
  return to_string (make_hex (register_class_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_register_class_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_REGISTER_CLASS_INFO_ARCHITECTURE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_architecture_id_t *> (value)));
    case AMD_DBGAPI_REGISTER_CLASS_INFO_NAME:
      return to_string (make_ref (static_cast<char *const *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_register_class_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_register_info_t register_info)
{
  switch (register_info)
    {
      CASE (REGISTER_INFO_ARCHITECTURE);
      CASE (REGISTER_INFO_NAME);
      CASE (REGISTER_INFO_SIZE);
      CASE (REGISTER_INFO_TYPE);
      CASE (REGISTER_INFO_DWARF);
      CASE (REGISTER_INFO_PROPERTIES);
    }
  return to_string (make_hex (register_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_register_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_REGISTER_INFO_ARCHITECTURE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_architecture_id_t *> (value)));
    case AMD_DBGAPI_REGISTER_INFO_NAME:
    case AMD_DBGAPI_REGISTER_INFO_TYPE:
      return to_string (make_ref (static_cast<char *const *> (value)));
    case AMD_DBGAPI_REGISTER_INFO_SIZE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_size_t *> (value)));
    case AMD_DBGAPI_REGISTER_INFO_DWARF:
      return to_string (make_ref (static_cast<const uint64_t *> (value)));
    case AMD_DBGAPI_REGISTER_INFO_PROPERTIES:
      return to_string (make_ref (
        static_cast<const amd_dbgapi_register_properties_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_register_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_register_exists_t register_exists)
{
  switch (register_exists)
    {
      CASE (REGISTER_ABSENT);
      CASE (REGISTER_PRESENT);
    }
  return to_string (make_hex (register_exists));
}

template <>
std::string
to_string (amd_dbgapi_register_class_state_t register_class_state)
{
  switch (register_class_state)
    {
      CASE (REGISTER_CLASS_STATE_NOT_MEMBER);
      CASE (REGISTER_CLASS_STATE_MEMBER);
    }
  return to_string (make_hex (register_class_state));
}

template <>
std::string
to_string (amd_dbgapi_address_class_info_t address_class_info)
{
  switch (address_class_info)
    {
      CASE (ADDRESS_CLASS_INFO_NAME);
      CASE (ADDRESS_CLASS_INFO_ADDRESS_SPACE);
      CASE (ADDRESS_CLASS_INFO_DWARF);
    }
  return to_string (make_hex (address_class_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_address_class_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_ADDRESS_CLASS_INFO_NAME:
      return to_string (make_ref (static_cast<char *const *> (value)));
    case AMD_DBGAPI_ADDRESS_CLASS_INFO_ADDRESS_SPACE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_address_space_id_t *> (value)));
    case AMD_DBGAPI_ADDRESS_CLASS_INFO_DWARF:
      return to_string (make_ref (static_cast<const uint64_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_address_class_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_address_space_access_t address_space_access)
{
  switch (address_space_access)
    {
      CASE (ADDRESS_SPACE_ACCESS_ALL);
      CASE (ADDRESS_SPACE_ACCESS_PROGRAM_CONSTANT);
      CASE (ADDRESS_SPACE_ACCESS_DISPATCH_CONSTANT);
    }
  return to_string (make_hex (address_space_access));
}

template <>
std::string
to_string (amd_dbgapi_address_space_info_t address_space_info)
{
  switch (address_space_info)
    {
      CASE (ADDRESS_SPACE_INFO_NAME);
      CASE (ADDRESS_SPACE_INFO_ADDRESS_SIZE);
      CASE (ADDRESS_SPACE_INFO_NULL_ADDRESS);
      CASE (ADDRESS_SPACE_INFO_ACCESS);
      CASE (ADDRESS_SPACE_INFO_DWARF);
    }
  return to_string (make_hex (address_space_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_address_space_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_ADDRESS_SPACE_INFO_NAME:
      return to_string (make_ref (static_cast<char *const *> (value)));
    case AMD_DBGAPI_ADDRESS_SPACE_INFO_ADDRESS_SIZE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_size_t *> (value)));
    case AMD_DBGAPI_ADDRESS_SPACE_INFO_NULL_ADDRESS:
      return to_string (make_hex (
        make_ref (static_cast<const amd_dbgapi_segment_address_t *> (value))));
    case AMD_DBGAPI_ADDRESS_SPACE_INFO_ACCESS:
      return to_string (make_ref (
        static_cast<const amd_dbgapi_address_space_access_t *> (value)));
    case AMD_DBGAPI_ADDRESS_SPACE_INFO_DWARF:
      return to_string (make_ref (static_cast<const uint64_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_address_space_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_segment_address_dependency_t address_dependency)
{
  switch (address_dependency)
    {
      CASE (SEGMENT_ADDRESS_DEPENDENCE_NONE);
      CASE (SEGMENT_ADDRESS_DEPENDENCE_LANE);
      CASE (SEGMENT_ADDRESS_DEPENDENCE_WAVE);
      CASE (SEGMENT_ADDRESS_DEPENDENCE_WORKGROUP);
      CASE (SEGMENT_ADDRESS_DEPENDENCE_AGENT);
      CASE (SEGMENT_ADDRESS_DEPENDENCE_PROCESS);
    }
  return to_string (make_hex (address_dependency));
}

template <>
std::string
to_string (amd_dbgapi_address_class_state_t address_class_state)
{
  switch (address_class_state)
    {
      CASE (ADDRESS_CLASS_STATE_NOT_MEMBER);
      CASE (ADDRESS_CLASS_STATE_MEMBER);
    }
  return to_string (make_hex (address_class_state));
}

template <>
std::string
to_string (amd_dbgapi_memory_precision_t memory_precision)
{
  switch (memory_precision)
    {
      CASE (MEMORY_PRECISION_NONE);
      CASE (MEMORY_PRECISION_PRECISE);
    }
  return to_string (make_hex (memory_precision));
}

template <>
std::string
to_string (amd_dbgapi_alu_exceptions_precision_t alu_exceptions_precision)
{
  switch (alu_exceptions_precision)
    {
      CASE (ALU_EXCEPTIONS_PRECISION_NONE);
      CASE (ALU_EXCEPTIONS_PRECISION_PRECISE);
    }
  return to_string (make_hex (alu_exceptions_precision));
}

template <>
std::string
to_string (amd_dbgapi_event_kind_t event_kind)
{
  switch (event_kind)
    {
      CASE (EVENT_KIND_NONE);
      CASE (EVENT_KIND_WAVE_STOP);
      CASE (EVENT_KIND_WAVE_COMMAND_TERMINATED);
      CASE (EVENT_KIND_CODE_OBJECT_LIST_UPDATED);
      CASE (EVENT_KIND_BREAKPOINT_RESUME);
      CASE (EVENT_KIND_RUNTIME);
      CASE (EVENT_KIND_QUEUE_ERROR);
    }
  return to_string (make_hex (event_kind));
}

template <>
std::string
to_string (amd_dbgapi_runtime_state_t runtime_state)
{
  switch (runtime_state)
    {
      CASE (RUNTIME_STATE_LOADED_SUCCESS);
      CASE (RUNTIME_STATE_UNLOADED);
      CASE (RUNTIME_STATE_LOADED_ERROR_RESTRICTION);
    }
  return to_string (make_hex (runtime_state));
}

template <>
std::string
to_string (amd_dbgapi_event_info_t event_info)
{
  switch (event_info)
    {
      CASE (EVENT_INFO_PROCESS);
      CASE (EVENT_INFO_KIND);
      CASE (EVENT_INFO_WAVE);
      CASE (EVENT_INFO_BREAKPOINT);
      CASE (EVENT_INFO_CLIENT_THREAD);
      CASE (EVENT_INFO_RUNTIME_STATE);
      CASE (EVENT_INFO_QUEUE);
    }
  return to_string (make_hex (event_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_event_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_EVENT_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    case AMD_DBGAPI_EVENT_INFO_KIND:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_event_kind_t *> (value)));
    case AMD_DBGAPI_EVENT_INFO_WAVE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_wave_id_t *> (value)));
    case AMD_DBGAPI_EVENT_INFO_BREAKPOINT:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_breakpoint_id_t *> (value)));
    case AMD_DBGAPI_EVENT_INFO_CLIENT_THREAD:
      return to_string (
        make_hex (make_ref (static_cast<const uintptr_t *> (value))));
    case AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_runtime_state_t *> (value)));
    case AMD_DBGAPI_EVENT_INFO_QUEUE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_queue_id_t *> (value)));
    }
  fatal_error ("unhandled amd_dbgapi_event_info_t query (%s)",
               to_cstring (query));
}

template <>
std::string
to_string (amd_dbgapi_log_level_t level)
{
  switch (level)
    {
      CASE (LOG_LEVEL_NONE);
      CASE (LOG_LEVEL_FATAL_ERROR);
      CASE (LOG_LEVEL_WARNING);
      CASE (LOG_LEVEL_INFO);
      CASE (LOG_LEVEL_TRACE);
      CASE (LOG_LEVEL_VERBOSE);
    }
  return to_string (make_hex (level));
}

template <>
std::string
to_string (amd_dbgapi_breakpoint_action_t breakpoint_action)
{
  switch (breakpoint_action)
    {
      CASE (BREAKPOINT_ACTION_RESUME);
      CASE (BREAKPOINT_ACTION_HALT);
    }
  return to_string (make_hex (breakpoint_action));
}

namespace
{

inline std::string
one_launch_trap_mask_to_string (os_wave_launch_trap_mask_t value)
{
  dbgapi_assert (!(value & (value - 1)) && "only 1 bit");

  switch (value)
    {
    case os_wave_launch_trap_mask_t::none:
      return "none";
    case os_wave_launch_trap_mask_t::fp_invalid:
      return "fp_invalid";
    case os_wave_launch_trap_mask_t::fp_input_denormal:
      return "fp_input_denormal";
    case os_wave_launch_trap_mask_t::fp_divide_by_zero:
      return "fp_divide_by_zero";
    case os_wave_launch_trap_mask_t::fp_overflow:
      return "fp_overflow";
    case os_wave_launch_trap_mask_t::fp_underflow:
      return "fp_underflow";
    case os_wave_launch_trap_mask_t::fp_inexact:
      return "fp_inexact";
    case os_wave_launch_trap_mask_t::int_divide_by_zero:
      return "int_divide_by_zero";
    case os_wave_launch_trap_mask_t::address_watch:
      return "address_watch";
    case os_wave_launch_trap_mask_t::wave_start:
      return "wave_start";
    case os_wave_launch_trap_mask_t::wave_end:
      return "wave_end";
    }
  return to_string (
    make_hex (static_cast<std::underlying_type_t<decltype (value)>> (value)));
}

} /* namespace */

template <>
std::string
to_string (os_wave_launch_trap_mask_t value)
{
  std::string str;

  if (!value)
    return one_launch_trap_mask_to_string (value);

  while (value != os_wave_launch_trap_mask_t::none)
    {
      os_wave_launch_trap_mask_t one_bit = value ^ (value & (value - 1));

      if (!str.empty ())
        str += " | ";
      str += one_launch_trap_mask_to_string (one_bit);

      value ^= one_bit;
    }

  return str;
}

template <>
std::string
to_string (amd_dbgapi_workgroup_info_t workgroup_info)
{
  switch (workgroup_info)
    {
      CASE (WORKGROUP_INFO_DISPATCH);
      CASE (WORKGROUP_INFO_QUEUE);
      CASE (WORKGROUP_INFO_AGENT);
      CASE (WORKGROUP_INFO_PROCESS);
      CASE (WORKGROUP_INFO_ARCHITECTURE);
      CASE (WORKGROUP_INFO_WORKGROUP_COORD);
    }
  return to_string (make_hex (workgroup_info));
}

template <>
std::string
to_string (detail::query_ref<amd_dbgapi_workgroup_info_t> ref)
{
  auto [query, value] = ref;
  switch (query)
    {
    case AMD_DBGAPI_WORKGROUP_INFO_DISPATCH:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_dispatch_id_t *> (value)));
    case AMD_DBGAPI_WORKGROUP_INFO_QUEUE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_queue_id_t *> (value)));
    case AMD_DBGAPI_WORKGROUP_INFO_AGENT:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_agent_id_t *> (value)));
    case AMD_DBGAPI_WORKGROUP_INFO_PROCESS:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_process_id_t *> (value)));
    case AMD_DBGAPI_WORKGROUP_INFO_ARCHITECTURE:
      return to_string (
        make_ref (static_cast<const amd_dbgapi_architecture_id_t *> (value)));
    case AMD_DBGAPI_WORKGROUP_INFO_WORKGROUP_COORD:
      return to_string (make_ref (static_cast<const uint32_t *> (value), 3));
    }
  fatal_error ("unhandled amd_dbgapi_workgroup_info_t query (%s)",
               to_cstring (query));
}

#undef CASE

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

void AMD_DBGAPI
amd_dbgapi_set_log_level (amd_dbgapi_log_level_t level)
{
  amd::dbgapi::log_level = level;

  /* Only log if the client callbacks are initialized.  */
  if (detail::is_initialized)
    {
      TRACE_BEGIN (level);
      TRACE_END ();
    }
}
