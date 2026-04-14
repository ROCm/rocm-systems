/* Copyright (c) 2021-2024 Advanced Micro Devices, Inc.

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

#include "os_driver.h"
#include "debug.h"
#include "logging.h"
#include "process.h"
#include "utils.h"

#include <cinttypes>
#include <type_traits>

namespace amd::dbgapi
{

template <>
std::string
to_string (os_wave_launch_mode_t mode)
{
  switch (mode)
    {
    case os_wave_launch_mode_t::normal:
      return "WAVE_LAUNCH_MODE_NORMAL";
    case os_wave_launch_mode_t::halt:
      return "WAVE_LAUNCH_MODE_HALT";
    case os_wave_launch_mode_t::kill:
      return "WAVE_LAUNCH_MODE_KILL";
    case os_wave_launch_mode_t::single_step:
      return "WAVE_LAUNCH_MODE_SINGLE_STEP";
    case os_wave_launch_mode_t::disable:
      return "WAVE_LAUNCH_MODE_DISABLE";
    }
  return to_string (
    make_hex (static_cast<std::underlying_type_t<decltype (mode)>> (mode)));
}

namespace
{

inline std::string
one_os_process_flag_to_string (os_process_flags_t flag)
{
  dbgapi_assert (!(flag & (flag - 1)) && "only 1 bit");

  switch (flag)
    {
    case os_process_flags_t::precise_memory:
      return "PRECISE_MEMORY";
    case os_process_flags_t::precise_alu_exceptions:
      return "PRECISE_ALU_EXCEPTIONS";
    }

  return to_string (
    make_hex (static_cast<std::underlying_type_t<decltype (flag)>> (flag)));
}

inline std::string
one_os_exception_to_string (os_exception_mask_t exception_mask)
{
  dbgapi_assert (!(exception_mask & (exception_mask - 1)) && "only 1 bit");

  switch (exception_mask)
    {
    case os_exception_mask_t::none:
      return "NONE";
    case os_exception_mask_t::queue_wave_abort:
      return "QUEUE_WAVE_ABORT";
    case os_exception_mask_t::queue_wave_trap:
      return "QUEUE_WAVE_TRAP";
    case os_exception_mask_t::queue_wave_math_error:
      return "QUEUE_WAVE_MATH_ERROR";
    case os_exception_mask_t::queue_wave_illegal_instruction:
      return "QUEUE_WAVE_ILLEGAL_INSTRUCTION";
    case os_exception_mask_t::queue_wave_memory_violation:
      return "QUEUE_WAVE_MEMORY_VIOLATION";
    case os_exception_mask_t::queue_wave_address_error:
      return "QUEUE_WAVE_ADDRESS_ERROR";
    case os_exception_mask_t::queue_packet_dispatch_dim_invalid:
      return "QUEUE_PACKET_DISPATCH_DIM_INVALID";
    case os_exception_mask_t::queue_packet_dispatch_group_segment_size_invalid:
      return "QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID";
    case os_exception_mask_t::queue_packet_dispatch_code_invalid:
      return "QUEUE_PACKET_DISPATCH_CODE_INVALID";
    case os_exception_mask_t::queue_packet_unsupported:
      return "QUEUE_PACKET_UNSUPPORTED";
    case os_exception_mask_t::queue_packet_dispatch_work_group_size_invalid:
      return "QUEUE_PACKET_DISPATCH_WORKGROUP_SIZE_INVALID";
    case os_exception_mask_t::queue_packet_dispatch_register_invalid:
      return "QUEUE_PACKET_DISPATCH_REGISTER_INVALID";
    case os_exception_mask_t::queue_packet_vendor_unsupported:
      return "QUEUE_PACKET_VENDOR_UNSUPPORTED";
    case os_exception_mask_t::queue_preemption_error:
      return "QUEUE_PREEMPTION_ERROR";
    case os_exception_mask_t::queue_new:
      return "QUEUE_NEW";
    case os_exception_mask_t::device_queue_delete:
      return "DEVICE_QUEUE_DELETE";
    case os_exception_mask_t::device_memory_violation:
      return "DEVICE_MEMORY_VIOLATION";
    case os_exception_mask_t::device_ras_error:
      return "DEVICE_RAS_ERROR";
    case os_exception_mask_t::device_fatal_halt:
      return "DEVICE_FATAL_HALT";
    case os_exception_mask_t::device_new:
      return "DEVICE_NEW";
    case os_exception_mask_t::process_runtime:
      return "PROCESS_RUNTIME";
    case os_exception_mask_t::process_device_remove:
      return "PROCESS_REMOVE";
    }

  return to_string (
    make_hex (static_cast<std::underlying_type_t<decltype (exception_mask)>> (
      exception_mask)));
}

} /* namespace */

template <>
std::string
to_string (os_exception_mask_t exception_mask)
{
  std::string str;

  if (exception_mask == os_exception_mask_t::none)
    return one_os_exception_to_string (exception_mask);

  while (exception_mask != os_exception_mask_t::none)
    {
      os_exception_mask_t one_bit
        = exception_mask ^ (exception_mask & (exception_mask - 1));

      if (!str.empty ())
        str += " | ";
      str += one_os_exception_to_string (one_bit);

      exception_mask ^= one_bit;
    }

  return str;
}

template <>
std::string
to_string (os_exception_code_t exception_code)
{
  return one_os_exception_to_string (os_exception_mask (exception_code));
}

template <>
std::string
to_string (os_agent_info_t os_agent_info)
{
  return string_printf (
    "{ .os_agent_id=%d, .name=%s, .domain=%#x, .location_id=%#x, "
    ".gfxip=[%d,%d,%d], .simd_count=%zd, .max_waves_per_simd=%zd, "
    ".shader_engine_count=%zd, .vendor_id=%#x, .device_id=%#x, "
    ".revision_id=%#x, .subsystem_vendor_id=%#x, .subsystem_device_id=%#x, "
    ".fw_version=%d, .local_address_aperture_base=%s, "
    ".local_address_aperture_limit=%s, .private_address_aperture_base=%s, "
    ".private_address_aperture_limit=%s, .debugging_supported=%d, "
    ".address_watch_supported=%d, .address_watch_register_count=%zd, "
    ".address_watch_mask_bits=%#" PRIx64 ", .watchpoint_exclusive=%d, "
    ".precise_memory_supported=%d, .precise_alu_exceptions_supported=%d,"
    ".firmware_supported=%d, ttmps_always_initialized=%d }",
    os_agent_info.os_agent_id, os_agent_info.name.c_str (),
    os_agent_info.domain, os_agent_info.location_id, os_agent_info.gfxip[0],
    os_agent_info.gfxip[1], os_agent_info.gfxip[2], os_agent_info.simd_count,
    os_agent_info.max_waves_per_simd, os_agent_info.shader_engine_count,
    os_agent_info.vendor_id, os_agent_info.device_id,
    os_agent_info.revision_id, os_agent_info.subsystem_vendor_id,
    os_agent_info.subsystem_device_id, os_agent_info.fw_version,
    to_cstring (os_agent_info.local_address_aperture_base),
    to_cstring (os_agent_info.local_address_aperture_limit),
    to_cstring (os_agent_info.private_address_aperture_base),
    to_cstring (os_agent_info.private_address_aperture_limit),
    os_agent_info.debugging_supported, os_agent_info.address_watch_supported,
    os_agent_info.address_watch_register_count,
    os_agent_info.address_watch_mask_bits, os_agent_info.watchpoint_exclusive,
    os_agent_info.precise_memory_supported,
    os_agent_info.precise_alu_exceptions_supported,
    os_agent_info.firmware_supported, os_agent_info.ttmps_always_initialized);
}

template <>
std::string
to_string (os_runtime_state_t runtime_state)
{
  switch (runtime_state)
    {
    case os_runtime_state_t::disabled:
      return "DISABLED";
    case os_runtime_state_t::enabled:
      return "ENABLED";
    case os_runtime_state_t::enabled_busy:
      return "ENABLED_BUSY";
    case os_runtime_state_t::enabled_error:
      return "ENABLED_ERROR";
    }
  return to_string (
    make_hex (static_cast<std::underlying_type_t<decltype (runtime_state)>> (
      runtime_state)));
}

template <>
std::string
to_string (os_runtime_info_t runtime_info)
{
  return string_printf ("{ .r_debug=%s, .runtime_state=%s, .ttmp_setup=%d }",
                        to_cstring (runtime_info.r_debug),
                        to_cstring (runtime_info.runtime_state),
                        runtime_info.ttmp_setup);
}

template <>
std::string
to_string (os_wave_launch_trap_override_t override)
{
  switch (override)
    {
    case os_wave_launch_trap_override_t::apply:
      return "APPLY";
    case os_wave_launch_trap_override_t::replace:
      return "REPLACED";
    }
  return to_string (make_hex (
    static_cast<std::underlying_type_t<decltype (override)>> (override)));
}

template <>
std::string
to_string (os_source_id_t source_id)
{
  return to_string (source_id.raw);
}

namespace
{

inline std::string
one_queue_state_t_to_string (os_queue_state_t state)
{
  dbgapi_assert (!(state & (state - 1)) && "only 1 bit");

  switch (state)
    {
    case os_queue_state_t::error:
      return "error";
    case os_queue_state_t::invalid:
      return "invalid";
    }
  return to_string (
    make_hex (static_cast<std::underlying_type_t<decltype (state)>> (state)));
}

}

template <>
std::string
to_string (os_queue_state_t queue_state)
{
  std::string str;

  if (!queue_state)
    return one_queue_state_t_to_string (queue_state);

  while (!!queue_state)
    {
      os_queue_state_t one_flag
        = queue_state ^ (queue_state & (queue_state - 1));

      if (!str.empty ())
        str += " | ";
      str += one_queue_state_t_to_string (one_flag);

      queue_state ^= one_flag;
    }

  return str;
}

template <>
std::string
to_string (os_queue_snapshot_entry_t snapshot)
{
  return string_printf (
    "{ .queue_id=%d, .state=%s, .gpu_id=%d, .queue_type=%s, "
    ".exception_status=%s, .ring_base_address=%s, .ring_size=%" PRId64 ", "
    ".write_pointer_address=%s, .read_pointer_address=%s, "
    ".ctx_save_restore_address=%s, .ctx_save_restore_area_size=%" PRId64 " }",
    snapshot.queue_id, to_cstring (snapshot.state), snapshot.gpu_id,
    to_cstring (snapshot.queue_type), to_cstring (snapshot.exception_status),
    to_cstring (snapshot.ring_base_address), snapshot.ring_size,
    to_cstring (snapshot.write_pointer_address),
    to_cstring (snapshot.read_pointer_address),
    to_cstring (snapshot.ctx_save_restore_address),
    snapshot.ctx_save_restore_area_size);
}

template <>
std::string
to_string (os_watch_mode_t watch_mode)
{
  switch (watch_mode)
    {
    case os_watch_mode_t::all:
      return "ALL";
    case os_watch_mode_t::atomic:
      return "ATOMIC";
    case os_watch_mode_t::nonread:
      return "NONREAD";
    case os_watch_mode_t::read:
      return "READ";
    }
  return to_string (make_hex (
    static_cast<std::underlying_type_t<decltype (watch_mode)>> (watch_mode)));
}

template <>
std::string
to_string (detail::query_ref<os_exception_code_t> ref)
{
  auto [query, value] = ref;

  if (query == os_exception_code_t::process_runtime)
    return to_string (
      make_ref (static_cast<const os_runtime_info_t *> (value)));

  return {};
}

template <>
std::string
to_string (os_process_flags_t flags)
{
  std::string str;

  if (!flags)
    return one_os_process_flag_to_string (flags);

  while (!!flags)
    {
      os_process_flags_t one_flag = flags ^ (flags & (flags - 1));

      if (!str.empty ())
        str += " | ";
      str += one_os_process_flag_to_string (one_flag);

      flags ^= one_flag;
    }

  return str;
}

template <>
std::string
to_string (os_queue_type_t queue_type)
{
  switch (queue_type)
    {
    case os_queue_type_t::compute:
      return "COMPUTE";
    case os_queue_type_t::sdma:
      return "SDMA";
    case os_queue_type_t::compute_aql:
      return "AQL";
    case os_queue_type_t::sdma_xgmi:
      return "XGMI";
    case os_queue_type_t::unknown:
      return "UNKNOWN";
    }
  return to_string (make_hex (
    static_cast<std::underlying_type_t<decltype (queue_type)>> (queue_type)));
}

} /* namespace amd::dbgapi */
