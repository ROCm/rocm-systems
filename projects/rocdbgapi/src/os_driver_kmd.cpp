/* Copyright (c) 2021-2025 Advanced Micro Devices, Inc.

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

#include "debug.h"
#include "logging.h"
#include "os_driver.h"

#include <ntstatus.h>
#include <windows.h>

#include <d3dkmthk.h>

#include "windows/kmddbg.h"

#include <hsa/amd_hsa_queue.h>

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <vector>

#if defined(WITH_API_TRACING)

#define TRACE_DRIVER_BEGIN(...)                                               \
  TRACE_BEGIN_HELPER (AMD_DBGAPI_LOG_LEVEL_VERBOSE, "driver: ", __VA_ARGS__)

#define TRACE_DRIVER_END(...) TRACE_END_HELPER (__VA_ARGS__)

#else /* !defined (WITH_API_TRACING) */

#define TRACE_DRIVER_BEGIN(...)
#define TRACE_DRIVER_END(...)

#endif /* !defined (WITH_API_TRACING) */

using namespace std::string_literals;

namespace amd::dbgapi
{

namespace
{

static constexpr os_queue_type_t
os_queue_type (enum KMD_DBGR_USER_QUEUE_TYPE kmd_type)
{
  switch (kmd_type)
    {
    case KMD_DBGR_USER_QUEUE_TYPE_GFX:
      return os_queue_type_t::unknown;
    case KMD_DBGR_USER_QUEUE_TYPE_COMPUTE:
      return os_queue_type_t::compute;
    case KMD_DBGR_USER_QUEUE_TYPE_DMA:
      return os_queue_type_t::sdma;
    }

  return os_queue_type_t::unknown;
}

/* Convert kfd_exception_code_t to KMD_DBGR_EXCEPTIONS.  */

static constexpr enum KMD_DBGR_EXCEPTIONS
kmd_exception (os_exception_code_t ec_code)
{
  switch (ec_code)
    {
    case os_exception_code_t::none:
      return KMD_DBGR_EXCP_QUEUE_WAVE_NONE;
    case os_exception_code_t::queue_wave_abort:
      return KMD_DBGR_EXCP_QUEUE_WAVE_ABORT;
    case os_exception_code_t::queue_wave_trap:
      return KMD_DBGR_EXCP_QUEUE_WAVE_TRAP;
    case os_exception_code_t::queue_wave_math_error:
      return KMD_DBGR_EXCP_QUEUE_WAVE_MATH_ERROR;
    case os_exception_code_t::queue_wave_illegal_instruction:
      return KMD_DBGR_EXCP_QUEUE_WAVE_ILLEGAL_INSTRUCTION;
    case os_exception_code_t::queue_wave_memory_violation:
      return KMD_DBGR_EXCP_QUEUE_WAVE_MEMORY_VIOLATION;
    case os_exception_code_t::queue_wave_address_error:
      return KMD_DBGR_EXCP_QUEUE_WAVE_APERTURE_VIOLATION;
    case os_exception_code_t::queue_packet_dispatch_dim_invalid:
      return KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_DIM_INVALID;
    case os_exception_code_t::queue_packet_dispatch_group_segment_size_invalid:
      return KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID;
    case os_exception_code_t::queue_packet_dispatch_code_invalid:
      return KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_CODE_INVALID;
    case os_exception_code_t::queue_packet_unsupported:
      return KMD_DBGR_EXCP_QUEUE_PACKET_UNSUPPORTED;
    case os_exception_code_t::queue_packet_dispatch_work_group_size_invalid:
      return KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID;
    case os_exception_code_t::queue_packet_dispatch_register_invalid:
      return KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_REGISTER_INVALID;
    case os_exception_code_t::queue_packet_vendor_unsupported:
      return KMD_DBGR_EXCP_QUEUE_PACKET_VENDOR_UNSUPPORTED;
    case os_exception_code_t::queue_preemption_error:
      return KMD_DBGR_EXCP_QUEUE_PREEMPTION_ERROR;
    case os_exception_code_t::queue_new:
      return KMD_DBGR_EXCP_QUEUE_NEW;

    case os_exception_code_t::device_queue_delete:
      return KMD_DBGR_EXCP_DEVICE_QUEUE_DELETE;
    case os_exception_code_t::device_memory_violation:
      return KMD_DBGR_EXCP_DEVICE_MEMORY_VIOLATION;
    case os_exception_code_t::device_ras_error:
      return KMD_DBGR_EXCP_DEVICE_RAS_ERROR;
    case os_exception_code_t::device_fatal_halt:
      return KMD_DBGR_EXCP_DEVICE_FATAL_HALT;
    case os_exception_code_t::device_new:
      return KMD_DBGR_EXCP_DEVICE_NEW;

    case os_exception_code_t::process_runtime:
      return KMD_DBGR_EXCP_PROCESS_RUNTIME;
    case os_exception_code_t::process_device_remove:
      return KMD_DBGR_EXCP_PROCESS_DEVICE_REMOVE;
    }

  dbgapi_assert_not_reached ("Unknown exception code");
}

static constexpr std::optional<os_exception_code_t>
os_exception_code (KMD_DBGR_EXCEPTIONS code)
{
  switch (code)
    {
    case KMD_DBGR_EXCP_QUEUE_WAVE_NONE:
      return os_exception_code_t::none;
    case KMD_DBGR_EXCP_QUEUE_WAVE_ABORT:
      return os_exception_code_t::queue_wave_abort;
    case KMD_DBGR_EXCP_QUEUE_WAVE_TRAP:
      return os_exception_code_t::queue_wave_trap;
    case KMD_DBGR_EXCP_QUEUE_WAVE_MATH_ERROR:
      return os_exception_code_t::queue_wave_math_error;
    case KMD_DBGR_EXCP_QUEUE_WAVE_ILLEGAL_INSTRUCTION:
      return os_exception_code_t::queue_wave_illegal_instruction;
    case KMD_DBGR_EXCP_QUEUE_WAVE_MEMORY_VIOLATION:
      return os_exception_code_t::queue_wave_memory_violation;
    case KMD_DBGR_EXCP_QUEUE_WAVE_APERTURE_VIOLATION:
      return os_exception_code_t::queue_wave_address_error;
    case KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_DIM_INVALID:
      return os_exception_code_t::queue_packet_dispatch_dim_invalid;
    case KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID:
      return os_exception_code_t::
        queue_packet_dispatch_group_segment_size_invalid;
    case KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_CODE_INVALID:
      return os_exception_code_t::queue_packet_dispatch_code_invalid;
    case KMD_DBGR_EXCP_QUEUE_PACKET_RESERVED:
      return {};
    case KMD_DBGR_EXCP_QUEUE_PACKET_UNSUPPORTED:
      return os_exception_code_t::queue_packet_unsupported;
    case KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID:
      return os_exception_code_t::
        queue_packet_dispatch_work_group_size_invalid;
    case KMD_DBGR_EXCP_QUEUE_PACKET_DISPATCH_REGISTER_INVALID:
      return os_exception_code_t::queue_packet_dispatch_register_invalid;
    case KMD_DBGR_EXCP_QUEUE_PACKET_VENDOR_UNSUPPORTED:
      return os_exception_code_t::queue_packet_vendor_unsupported;
    case KMD_DBGR_EXCP_QUEUE_PREEMPTION_ERROR:
      return os_exception_code_t::queue_preemption_error;
    case KMD_DBGR_EXCP_QUEUE_NEW:
      return os_exception_code_t::queue_new;
    case KMD_DBGR_EXCP_DEVICE_QUEUE_DELETE:
      return os_exception_code_t::device_queue_delete;
    case KMD_DBGR_EXCP_DEVICE_MEMORY_VIOLATION:
      return os_exception_code_t::device_memory_violation;
    case KMD_DBGR_EXCP_DEVICE_RAS_ERROR:
      return os_exception_code_t::device_ras_error;
    case KMD_DBGR_EXCP_DEVICE_FATAL_HALT:
      return os_exception_code_t::device_fatal_halt;
    case KMD_DBGR_EXCP_DEVICE_NEW:
      return os_exception_code_t::device_new;
    case KMD_DBGR_EXCP_PROCESS_RUNTIME:
      return os_exception_code_t::process_runtime;
    case KMD_DBGR_EXCP_PROCESS_DEVICE_REMOVE:
      return os_exception_code_t::process_device_remove;
    }

  return {};
}

static constexpr KMD_DBGR_WAVE_LAUNCH_MODE
kmd_wave_launch_mode (os_wave_launch_mode_t mode)
{
  switch (mode)
    {
    case os_wave_launch_mode_t::normal:
      return KMD_DBGR_WAVE_LAUNCH_MODE_NORMAL;
    case os_wave_launch_mode_t::halt:
      return KMD_DBGR_WAVE_LAUNCH_MODE_HALT;
    case os_wave_launch_mode_t::kill:
      return KMD_DBGR_WAVE_LAUNCH_MODE_KILL;
    case os_wave_launch_mode_t::single_step:
      return KMD_DBGR_WAVE_LAUNCH_MODE_TRAP_AFTER_INST;
    case os_wave_launch_mode_t::disable:
      return KMD_DBGR_WAVE_LAUNCH_MODE_DISABLE;
    }

  dbgapi_assert_not_reached ("Invalid wave launch mode");
}

static constexpr KMDDBGRIF_ADDR_WATCH_MODE
kmd_addr_watch_mode (os_watch_mode_t mode)
{
  switch (mode)
    {
    case os_watch_mode_t::read:
      return KMDDBGRIF_ADDR_WATCH_MODE_READ;
    case os_watch_mode_t::nonread:
      return KMDDBGRIF_ADDR_WATCH_MODE_NONREAD;
    case os_watch_mode_t::atomic:
      return KMDDBGRIF_ADDR_WATCH_MODE_ATOMIC;
    case os_watch_mode_t::all:
      return KMDDBGRIF_ADDR_WATCH_MODE_ALL;
    }

  dbgapi_assert_not_reached ("Invalid watch mode");
}

static constexpr KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE
kmd_enable_traps_for_exceptions_mode (os_wave_launch_trap_override_t o)
{
  switch (o)
    {
    case os_wave_launch_trap_override_t::apply:
      return KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE_OR;
    case os_wave_launch_trap_override_t::replace:
      return KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE_REPLACE;
    }

  dbgapi_assert_not_reached ("Invalid wave launch trap override");
}

static constexpr ULONG
kmd_wave_launch_traps_mask (os_wave_launch_trap_mask_t mask)
{
  ULONG kmd_mask = 0;

  if (!!(mask & os_wave_launch_trap_mask_t::fp_invalid))
    kmd_mask |= 1 << 0;
  if (!!(mask & os_wave_launch_trap_mask_t::fp_input_denormal))
    kmd_mask |= 1 << 1;
  if (!!(mask & os_wave_launch_trap_mask_t::fp_divide_by_zero))
    kmd_mask |= 1 << 2;
  if (!!(mask & os_wave_launch_trap_mask_t::fp_overflow))
    kmd_mask |= 1 << 3;
  if (!!(mask & os_wave_launch_trap_mask_t::fp_underflow))
    kmd_mask |= 1 << 4;
  if (!!(mask & os_wave_launch_trap_mask_t::fp_inexact))
    kmd_mask |= 1 << 5;
  if (!!(mask & os_wave_launch_trap_mask_t::int_divide_by_zero))
    kmd_mask |= 1 << 6;
  if (!!(mask & os_wave_launch_trap_mask_t::address_watch))
    kmd_mask |= 1 << 7;
  if (!!(mask & os_wave_launch_trap_mask_t::wave_start))
    kmd_mask |= 1 << 30;
  if (!!(mask & os_wave_launch_trap_mask_t::wave_end))
    kmd_mask |= 1 << 31;

  return kmd_mask;
}

constexpr ULONG64
kmd_code_to_mask (enum KMD_DBGR_EXCEPTIONS e)
{
  return 1ull << (e - 1);
};

/* Convert a KMD exception mask to os_exception_mask_t.  */

static constexpr os_exception_mask_t
os_exception_mask (ULONG64 kmd_mask)
{
  os_exception_mask_t mask{};

  if (kmd_mask == 0)
    return mask;

  while (kmd_mask != 0)
    {
      ULONG64 one_bit = kmd_mask ^ (kmd_mask & (kmd_mask - 1));

      auto code = os_exception_code (
        excp_mask_to_excp_code<KMD_DBGR_EXCEPTIONS> (one_bit));

      if (code.has_value ())
        mask |= os_exception_mask (code.value ());
      else
        warning ("Unknown KMD exception code %" PRIx64,
                 static_cast<uint64_t> (one_bit));

      kmd_mask ^= one_bit;
    }

  return mask;
}

/* Convert a os_exception_mask_t to a KMD excepton mask.  */

static constexpr ULONG64
kmd_exception_mask (os_exception_mask_t mask)
{
  ULONG64 kmd_mask = 0;

  if (mask == os_exception_mask_t::none)
    return kmd_mask;

  while (mask != os_exception_mask_t::none)
    {
      os_exception_mask_t one_bit = mask ^ (mask & (mask - 1));

      kmd_mask |= kmd_code_to_mask (
        kmd_exception (excp_mask_to_excp_code<os_exception_code_t> (one_bit)));

      mask ^= one_bit;
    }

  dbgapi_assert (mask == 0);

  return kmd_mask;
}

/* Convert a runtime state returned by KFD to os_runtime_state_t.  */

static constexpr os_runtime_state_t
os_runtime_state (ULONG state)
{
  switch (state)
    {
    case KMD_DBGR_RUNTIME_STATE_DISABLED:
      return os_runtime_state_t::disabled;
    case KMD_DBGR_RUNTIME_STATE_ENABLED:
      return os_runtime_state_t::enabled;
    case KMD_DBGR_RUNTIME_STATE_ENABLED_ERROR:
      return os_runtime_state_t::enabled_error;
    }

  fatal_error ("Unsupported runtime state %#lx", state);
}

} // namespace (anonymous)

template <>
std::string
to_string (KMD_DBGR_CMDS_OP cmd)
{
  switch (cmd)
    {
    case KMD_DBGR_CMD_OP_INVALID:
      return "INVALID";
    case KMD_DBGR_CMD_OP_SET_RUNTIME_INFO:
      return "SET_RUNTIME_INFO";
    case KMD_DBGR_CMD_OP_GET_RUNTIME_INFO:
      return "GET_RUNTIME_INFO";
    case KMD_DBGR_CMD_OP_ENABLE_GLOBAL_TRAP:
      return "ENABLE_GLOBAL_TRAP";
    case KMD_DBGR_CMD_OP_ENABLE_EXCEPTIONS:
      return "ENABLE_EXCEPTIONS";
    case KMD_DBGR_CMD_OP_ENABLE_TRAPS_FOR_EXCEPTIONS:
      return "ENABLE_TRAPS_FOR_EXCEPTIONS";
    case KMD_DBGR_CMD_OP_SET_WAVE_LAUNCH_MODE:
      return "SET_WAVE_LAUNCH_MODE";
    case KMD_DBGR_CMD_OP_SUSPEND_QUEUE:
      return "SUSPEND_QUEUE";
    case KMD_DBGR_CMD_OP_RESUME_QUEUE:
      return "RESUME_QUEUE";
    case KMD_DBGR_CMD_OP_GET_EXCEPTIONS:
      return "GET_EXCEPTIONS";
    case KMD_DBGR_CMD_OP_GET_EXCEPTION_INFO:
      return "GET_EXCEPTION_INFO";
    case KMD_DBGR_CMD_OP_GET_QUEUE_INFO:
      return "GET_QUEUE_INFO";
    case KMD_DBGR_CMD_OP_GET_DEVICE_INFO:
      return "GET_DEVICE_INFO";
    case KMD_DBGR_CMD_OP_GET_TRAP_VERSION:
      return "GET_TRAP_VERSION";
    case KMD_DBGR_CMD_OP_SET_ADDR_WATCH:
      return "SET_ADDR_WATCH";
    case KMD_DBGR_CMD_OP_CLEAR_ADDR_WATCH:
      return "CLEAR_ADDR_WATCH";
    case KMD_DBGR_CMD_OP_SET_PRECISE_MEMOPS:
      return "SET_PRECISE_MEMOPS";
    case KMD_DBGR_CMD_OP_SEND_EVENT_TO_INFERIOR:
      return "SEND_EVENT_TO_INFERIOR";
    case KMD_DBGR_CMD_OP_READ_BUFFER:
      return "READ_BUFFER";
    case KMD_DBGR_CMD_OP_WRITE_BUFFER:
      return "WRITE_BUFFER";
    case KMD_DBGR_CMD_OP_SET_USER_TRAP:
      return "SET_USER_TRAP";
    case KMD_DBGR_CMD_OP_MAX:
      break;
    }
  return "<unkmown>";
}

template <>
std::string
to_string (KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE mode)
{
  switch (mode)
    {
    case KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE_OR:
      return "KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE_OR";
    case KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE_REPLACE:
      return "KMD_DBGR_ENABLE_TRAPS_FOR_EXCEPTIONS_MODE_REPLACE";
    }
  return "UNKNOWN";
}

template <>
std::string
to_string (KMD_DBGR_WAVE_LAUNCH_MODE mode)
{
  switch (mode)
    {
    case KMD_DBGR_WAVE_LAUNCH_MODE_NORMAL:
      return "KMD_DBGR_WAVE_LAUNCH_MODE_NORMAL";
    case KMD_DBGR_WAVE_LAUNCH_MODE_HALT:
      return "KMD_DBGR_WAVE_LAUNCH_MODE_HALT";
    case KMD_DBGR_WAVE_LAUNCH_MODE_KILL:
      return "KMD_DBGR_WAVE_LAUNCH_MODE_KILL";
    case KMD_DBGR_WAVE_LAUNCH_MODE_TRAP_AFTER_INST:
      return "KMD_DBGR_WAVE_LAUNCH_MODE_TRAP_AFTER_INST";
    case KMD_DBGR_WAVE_LAUNCH_MODE_DISABLE:
      return "KMD_DBGR_WAVE_LAUNCH_MODE_DISABLE";
    case KMD_DBGR_WAVE_LAUNCH_MODE_MAX:
      break;
    }
  return "UNKNOWN";
}

template <>
std::string
to_string (KMDDBGRIF_ADDR_WATCH_ID wid)
{
  switch (wid)
    {
    case KMDDBGRIF_ADDR_WATCH_ID_0:
      return "Watch0";
    case KMDDBGRIF_ADDR_WATCH_ID_1:
      return "Watch1";
    case KMDDBGRIF_ADDR_WATCH_ID_2:
      return "Watch2";
    case KMDDBGRIF_ADDR_WATCH_ID_3:
      return "Watch3";
    case KMDDBGRIF_ADDR_WATCH_ID_MAX:
      break;
    }
  return "INVALID";
}

template <>
std::string
to_string (KMDDBGRIF_ADDR_WATCH_MODE m)
{
  switch (m)
    {
    case KMDDBGRIF_ADDR_WATCH_MODE_READ:
      return "READ";
    case KMDDBGRIF_ADDR_WATCH_MODE_NONREAD:
      return "NONREAD";
    case KMDDBGRIF_ADDR_WATCH_MODE_ATOMIC:
      return "ATOMIC";
    case KMDDBGRIF_ADDR_WATCH_MODE_ALL:
      return "ALL";
    case KMDDBGRIF_ADDR_WATCH_MODE_MAX:
      break;
    }
  return "INVALID";
}

static constexpr os_queue_state_t
queue_state (ULONG status)
{
  switch (status)
    {
    case KMD_DBGR_USER_QUEUE_STATUS_ERROR:
      return os_queue_state_t::error;
    case KMD_DBGR_USER_QUEUE_STATUS_INVALID:
      return os_queue_state_t::invalid;
    case KMD_DBGR_USER_QUEUE_STATUS_SUCCESS:
      return {};
    }

  /* Unknown value, treat it as error.  */
  return os_queue_state_t::error;
}

template <>
std::string
to_string (KMDDBGRIF_DBGR_CMDS_INPUT inp)
{
  auto print_payload = [&inp] () -> std::string
  {
    switch (inp.cmd)
      {
      case KMD_DBGR_CMD_OP_ENABLE_GLOBAL_TRAP:
        return string_printf ("{.enabled=%#x, .exception_mask=%#" PRIx64 ", "
                              ".eventHandle=%#" PRIx64 "}",
                              inp.enableGlobalTrapIn.enable,
                              inp.enableGlobalTrapIn.exceptionMask,
                              inp.enableGlobalTrapIn.umdEventHandle64);
      case KMD_DBGR_CMD_OP_ENABLE_EXCEPTIONS:
        return string_printf ("{.exceptionMask=%#" PRIx64 "}",
                              inp.enableExceptionsIn.exceptionMask);
      case KMD_DBGR_CMD_OP_ENABLE_TRAPS_FOR_EXCEPTIONS:
        return string_printf (
          "{.mode=%s, .trapEnableMask=%#lx, .trapRequestedMask=%#lx}",
          to_string (inp.enableTrapsForExceptionsIn.mode).c_str (),
          inp.enableTrapsForExceptionsIn.trapEnableMask,
          inp.enableTrapsForExceptionsIn.trapRequestedMask);

      case KMD_DBGR_CMD_OP_SET_WAVE_LAUNCH_MODE:
        return string_printf (
          "{.mode=%s}", to_string (inp.setWaveLaunchModeIn.mode).c_str ());

      case KMD_DBGR_CMD_OP_SUSPEND_QUEUE:
        return string_printf (
          "{.exceptionMaskToClear=%#" PRIx64
          ", .gracePeriodIn100us=%ld, .queues=%s}",
          inp.suspendQueueIn.exceptionMaskToClear,
          inp.suspendQueueIn.gracePeriodIn100us,
          to_string (make_ref (inp.suspendQueueIn.queueIds,
                               inp.suspendQueueIn.numQueues))
            .c_str ());
      case KMD_DBGR_CMD_OP_RESUME_QUEUE:
        return string_printf (
          "{.queues=%s}", to_string (make_ref (inp.resumeQueueIn.queueIds,
                                               inp.resumeQueueIn.numQueues))
                            .c_str ());
      case KMD_DBGR_CMD_OP_GET_EXCEPTIONS:
        return string_printf ("{.exceptionMaskToClear=%#" PRIx64 "}",
                              inp.getExceptionsIn.exceptionMaskToClear);

      case KMD_DBGR_CMD_OP_GET_EXCEPTION_INFO:
        return string_printf (
          "{.exceptionId=%#" PRIx64 ", .queueId=%ld, .clearExceptions=%d}",
          inp.getExceptionInfoIn.exceptionId, inp.getExceptionInfoIn.queueId,
          inp.getExceptionInfoIn.clearException);

      case KMD_DBGR_CMD_OP_GET_QUEUE_INFO:
        return string_printf ("{.exceptionMaskToClear=%#" PRIx64 "}",
                              inp.getQueueInfoIn.exceptionMaskToClear);

      case KMD_DBGR_CMD_OP_GET_DEVICE_INFO:
        return string_printf ("{.exceptionMaskToClear=%#" PRIx64 "}",
                              inp.getDeviceInfoIn.exceptionMaskToClear);

      case KMD_DBGR_CMD_OP_SET_ADDR_WATCH:
        return string_printf ("{.watchId=%s, .mode=%s, .watchAddr=%#" PRIx64
                              ", .watchAddrMask=%lx}",
                              to_string (inp.setAddrWatchIn.watchId).c_str (),
                              to_string (inp.setAddrWatchIn.mode).c_str (),
                              inp.setAddrWatchIn.watchAddr,
                              inp.setAddrWatchIn.watchAddrMask);

      case KMD_DBGR_CMD_OP_CLEAR_ADDR_WATCH:
        return string_printf (
          "{.watchId=%s}", to_string (inp.clearAddrWatchIn.watchId).c_str ());

      case KMD_DBGR_CMD_OP_SET_PRECISE_MEMOPS:
        return string_printf ("{.enable=%d}", inp.setPreciseMemOpsIn.enable);

      case KMD_DBGR_CMD_OP_SEND_EVENT_TO_INFERIOR:
        return string_printf ("{.exceptionMask=%#" PRIx64 ", .queueId=%ld}",
                              inp.sendEventToInferiorIn.exceptionMask,
                              inp.sendEventToInferiorIn.queueId);

      case KMD_DBGR_CMD_OP_READ_BUFFER:
        return string_printf (
          "{.srcGpuVa=%#" PRIx64 ", .dstCpuVa=%#" PRIx64 ", .size=%ld}",
          inp.readBufferIn.srcGpuVA, inp.readBufferIn.dstCpuVA,
          inp.readBufferIn.size);

      case KMD_DBGR_CMD_OP_WRITE_BUFFER:
        return string_printf (
          "{.srcCpuVa=%#" PRIx64 ", .dstGpuVa=%#" PRIx64 ".size=%ld}",
          inp.writeBufferIn.srcCpuVA, inp.writeBufferIn.dstGpuVA,
          inp.writeBufferIn.size);

      case KMD_DBGR_CMD_OP_GET_RUNTIME_INFO:
      case KMD_DBGR_CMD_OP_GET_TRAP_VERSION:
        return "{}";

      case KMD_DBGR_CMD_OP_INVALID:
      case KMD_DBGR_CMD_OP_SET_USER_TRAP:
      case KMD_DBGR_CMD_OP_SET_RUNTIME_INFO:
      case KMD_DBGR_CMD_OP_MAX:
        /* Those commands are not used by the debugger.  */
        break;
      }
    return "UNKNOWN";
  };
  return string_printf ("{cmd=%s, args=%s}", to_string (inp.cmd).c_str (),
                        print_payload ().c_str ());
}

template <>
std::string
to_string (KMD_DBGR_USER_QUEUE_TYPE type)
{
  switch (type)
    {
    case KMD_DBGR_USER_QUEUE_TYPE_GFX:
      return "GFX";
    case KMD_DBGR_USER_QUEUE_TYPE_COMPUTE:
      return "COMPUTE";
    case KMD_DBGR_USER_QUEUE_TYPE_DMA:
      return "DMA";
    }
  return "UNKNOWN";
}

template <>
std::string
to_string (KMDDBGR_QUEUE_INFO q)
{
  return string_printf (
    "{.queueExceptionMask=%#" PRIx64 ", .ringBufferAddress=%#" PRIx64
    ", .ringSize=%#lx, .writePtrAddress=%#" PRIx64
    ", .readPtrAddress=%#" PRIx64 ", .ctxSaveRestoreAddress=%#" PRIx64
    ", .ctxSaveRestoreSize=%#lx, .queueId=%ld"
    ", .queueType=%s, .aqlPacketList=%#" PRIx64 ", computeTmpRingSize=%#lx}",
    q.queueExceptionMask, q.ringBufferAddress, q.ringSize, q.writePtrAddress,
    q.readPtrAddress, q.ctxSaveRestoreAddress, q.ctxSaveRestoreSize, q.queueId,
    to_string (q.queueType).c_str (), q.aqlPacketList, q.computeTmpRingSize);
}

template <>
std::string
to_string (KMDDBGRIF_DBGR_CMDS_OUTPUT o)
{
  switch (o.cmd)
    {
    case KMD_DBGR_CMD_OP_GET_RUNTIME_INFO:
      return string_printf ("{.rDebug=%#" PRIx64
                            ", .runtimeState=%ld, .ttmpSetup=%ld}",
                            o.getRuntimeInfoOut.rocRuntimeInfo.rDebug,
                            o.getRuntimeInfoOut.rocRuntimeInfo.runtimeState,
                            o.getRuntimeInfoOut.rocRuntimeInfo.ttmpSetup);

    case KMD_DBGR_CMD_OP_ENABLE_TRAPS_FOR_EXCEPTIONS:
      return string_printf ("{.trapSupportMask=%ld}",
                            o.enableTrapsForExceptionsOut.trapSupportMask);

    case KMD_DBGR_CMD_OP_GET_EXCEPTIONS:
      return string_printf (
        "{.exceptionMask=%#" PRIx64 ", .queueId=%ld, .noExceptions=%d}",
        o.getExceptionsOut.exceptionMask, o.getExceptionsOut.queueId,
        o.getExceptionsOut.noExceptions);

    case KMD_DBGR_CMD_OP_GET_EXCEPTION_INFO:
      /* This works because we only request runtime exception info.  */
      return string_printf ("{.runtimeState=%ld}",
                            o.getExceptionInfoOut.rocRuntimeInfo.runtimeState);

    case KMD_DBGR_CMD_OP_GET_QUEUE_INFO:
      return string_printf ("{.queues=%s}",
                            to_string (make_ref (o.getQueueInfoOut.queueInfo,
                                                 o.getQueueInfoOut.numQueues))
                              .c_str ());

    case KMD_DBGR_CMD_OP_GET_DEVICE_INFO:
      return string_printf (
        "{.deviceExceptionMask=%#" PRIx64 ", .ldsBase=%#" PRIx64
        ", .ldsLimit=%#" PRIx64 ", .scratchBase=%#" PRIx64
        ", .scratchLimit=%#" PRIx64 ", .gpuvmBase=%#" PRIx64
        ", .gpuvmLimit=%#" PRIx64 ", .locationId=%#lx"
        ", .vendorId=%#lx, "
        ".deviceId=%#lx, .fwVersion=%#lx, .gfxTargetVersion=%#lx "
        ", .simdCount=%lu, .maxWavesPerSimd=%lu, .arrayCount=%lu "
        ", .simdArraysPerEngine=%lu, .capability=%#lx, .debugProp=%#lx}",
        o.getDeviceInfoOut.deviceInfo.deviceExceptionMask,
        o.getDeviceInfoOut.deviceInfo.ldsBase,
        o.getDeviceInfoOut.deviceInfo.ldsLimit,
        o.getDeviceInfoOut.deviceInfo.scratchBase,
        o.getDeviceInfoOut.deviceInfo.scratchLimit,
        o.getDeviceInfoOut.deviceInfo.gpuvmBase,
        o.getDeviceInfoOut.deviceInfo.gpuvmLimit,
        o.getDeviceInfoOut.deviceInfo.locationId,
        o.getDeviceInfoOut.deviceInfo.vendorId,
        o.getDeviceInfoOut.deviceInfo.deviceId,
        o.getDeviceInfoOut.deviceInfo.fwVersion,
        o.getDeviceInfoOut.deviceInfo.gfxTargetVersion,
        o.getDeviceInfoOut.deviceInfo.simdCount,
        o.getDeviceInfoOut.deviceInfo.maxWavesPerSimd,
        o.getDeviceInfoOut.deviceInfo.arrayCount,
        o.getDeviceInfoOut.deviceInfo.simdArraysPerEngine,
        o.getDeviceInfoOut.deviceInfo.capability,
        o.getDeviceInfoOut.deviceInfo.debugProp);

    case KMD_DBGR_CMD_OP_GET_TRAP_VERSION:
      return string_printf ("{.majorVersion=%ld, .minorVersion=%ld}",
                            o.getTrapVersionOut.majorVersion,
                            o.getTrapVersionOut.minorVersion);

    case KMD_DBGR_CMD_OP_READ_BUFFER:
      return string_printf ("{.bytesRead=%#lx}", o.readBufferOut.bytesRead);

    case KMD_DBGR_CMD_OP_WRITE_BUFFER:
      return string_printf ("{.bytesWritten=%#lx}",
                            o.writeBufferOut.bytesWritten);

    case KMD_DBGR_CMD_OP_ENABLE_GLOBAL_TRAP:
    case KMD_DBGR_CMD_OP_ENABLE_EXCEPTIONS:
    case KMD_DBGR_CMD_OP_SET_WAVE_LAUNCH_MODE:
    case KMD_DBGR_CMD_OP_SUSPEND_QUEUE:
    case KMD_DBGR_CMD_OP_RESUME_QUEUE:
    case KMD_DBGR_CMD_OP_SET_ADDR_WATCH:
    case KMD_DBGR_CMD_OP_CLEAR_ADDR_WATCH:
    case KMD_DBGR_CMD_OP_SET_PRECISE_MEMOPS:
    case KMD_DBGR_CMD_OP_SEND_EVENT_TO_INFERIOR:
      return "{}";

    case KMD_DBGR_CMD_OP_INVALID:
    case KMD_DBGR_CMD_OP_SET_USER_TRAP:
    case KMD_DBGR_CMD_OP_SET_RUNTIME_INFO:
    case KMD_DBGR_CMD_OP_MAX:
      /* Those commands are not used by the debugger.  */
      break;
    }
  return "UNKNOWN";
}

namespace kmd
{
struct d3d_t
{
  d3d_t ();
  ~d3d_t ();
  static constexpr auto d3d_lib_name = "gdi32.dll";

  bool is_valid () const { return d3d_module != nullptr; }

  HMODULE d3d_module = nullptr;
  struct
  {
    PFND3DKMT_QUERYADAPTERINFO query_adapter_info;
    PFND3DKMT_CLOSEADAPTER close_adapter;
    PFND3DKMT_CREATEDEVICE create_device;
    PFND3DKMT_DESTROYDEVICE destroy_device;
    PFND3DKMT_ESCAPE escape;
    PFND3DKMT_ENUMADAPTERS3 enum_adapters;
  } api = {};
};

d3d_t::d3d_t ()
{
  d3d_module = ::LoadLibrary (d3d_lib_name);
  if (d3d_module == nullptr)
    return;

  /* If something fails, just plate *this in an invalid state.  */
  auto close_lib = utils::make_scope_exit (
    [this] ()
    {
      warning ("Failed ot load %s", d3d_lib_name);
      api = {};
    });

#define GET_FUNC(to, name)                                                    \
  reinterpret_cast<to> (                                                      \
    reinterpret_cast<void (*) ()> (::GetProcAddress (d3d_module, #name)))

  api.query_adapter_info
    = GET_FUNC (PFND3DKMT_QUERYADAPTERINFO, D3DKMTQueryAdapterInfo);
  api.close_adapter = GET_FUNC (PFND3DKMT_CLOSEADAPTER, D3DKMTCloseAdapter);
  api.create_device = GET_FUNC (PFND3DKMT_CREATEDEVICE, D3DKMTCreateDevice);
  api.destroy_device = GET_FUNC (PFND3DKMT_DESTROYDEVICE, D3DKMTDestroyDevice);
  api.escape = GET_FUNC (PFND3DKMT_ESCAPE, D3DKMTEscape);
  api.enum_adapters = GET_FUNC (PFND3DKMT_ENUMADAPTERS3, D3DKMTEnumAdapters3);

#undef GET_FUNC

  /* All was good, we can keep *this in the valid state.  */
  close_lib.release ();
}

d3d_t::~d3d_t ()
{
  if (d3d_module != nullptr)
    {
      api = {};
      ::FreeLibrary (d3d_module);
    }
}

struct agent_handle_t
{
  D3DKMT_HANDLE adapter;
  D3DKMT_HANDLE device;
  size_t chain_ordinal;
};

struct queue_handle_t
{
  os_agent_id_t agent_id;
  ULONG kmd_queue_id;
};

} /* namespace amd::dbgapi::kmd  */

template <>
std::string
to_string (kmd::agent_handle_t agent)
{
  return string_printf ("{adapter=%xp, device=%xp, ordinal=%lld}",
                        agent.adapter, agent.device, agent.chain_ordinal);
}

template <>
std::string
to_string (NTSTATUS status)
{
#define NT_CASE(x)                                                            \
  case STATUS_##x:                                                            \
    return #x

  switch (status)
    {
      NT_CASE (SUCCESS);
      NT_CASE (NOT_SUPPORTED);
      NT_CASE (DRIVER_PROCESS_TERMINATED);
      NT_CASE (RESOURCE_IN_USE);
      NT_CASE (NO_MEMORY);
      NT_CASE (INVALID_PARAMETER);
      NT_CASE (RETRY);
      NT_CASE (UNSUCCESSFUL);
    default:
      return string_printf ("STATUS(%#lx)", status);
    }
#undef NT_CASE
}

static amd_dbgapi_status_t
nt_status_to_dbgapi_status (NTSTATUS status)
{
  switch (status)
    {
    case STATUS_SUCCESS:
      return AMD_DBGAPI_STATUS_SUCCESS;
    case STATUS_NOT_SUPPORTED:
      return AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED;
    case STATUS_DRIVER_PROCESS_TERMINATED:
      return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;

    case STATUS_RESOURCE_IN_USE:
    case STATUS_NO_MEMORY:
    case STATUS_INVALID_PARAMETER:
    case STATUS_RETRY:
    case STATUS_UNSUCCESSFUL:
    default:
      return AMD_DBGAPI_STATUS_ERROR;
    }
}

/** The KMD (windows) driver backend for rocm-dbgapi.

  In this driver model, there might be multiple KMD instances our backend needs
  to talk with: up to one per device (a.k.a apapter).  The GPU_IDs / QUEUE_IDs
  used by each to communicate with each KMD instance are only quaranteed to be
  unique for that KMD instance.  This means that we cannot use the KMD's
  GPU/queue ID values as os_agent_id_t / os_queue_id_t.  The kmd_driver_t
  instance needs to keep a translation layer to map os_agent_id_t/os_queue_id_t
  to an adapter and then to that adapter's KMD GPU/QUEUE id.

  Support for this multi-agent configuration is not fully validated at this
  point.
*/

class kmd_driver_t : public null_driver_t
{
public:
  kmd_driver_t (amd_dbgapi_os_process_id_t os_id);
  ~kmd_driver_t () override;

  bool is_valid () const override;
  amd_dbgapi_status_t check_version () const override;

  bool is_debug_enabled () const override { return m_is_debug_enabled; }
  amd_dbgapi_status_t
  agent_snapshot (os_agent_info_t *snapshots, size_t snapshot_count,
                  size_t *agent_count,
                  os_exception_mask_t exceptions_cleared) const override;

  amd_dbgapi_status_t enable_debug (os_exception_mask_t exceptions_reported,
                                    amd_dbgapi_notifier_t notifier,
                                    os_runtime_info_t *runtime_info) override;
  amd_dbgapi_status_t disable_debug () override;

  amd_dbgapi_status_t set_exceptions_reported (
    os_exception_mask_t exceptions_reported) const override;

  amd_dbgapi_status_t
  send_exceptions (os_exception_mask_t exceptions,
                   std::optional<os_agent_id_t> agent_id,
                   std::optional<os_queue_id_t> queue_id) const override;

  amd_dbgapi_status_t
  query_debug_event (os_exception_mask_t *exceptions_present,
                     os_queue_id_t *os_queue_id, os_agent_id_t *os_agent_id,
                     os_exception_mask_t exceptions_cleared) const override;

  amd_dbgapi_status_t
  query_exception_info (os_exception_code_t exception,
                        os_source_id_t os_source_id,
                        os_exception_info_t *os_exception_info,
                        bool clear_exception) const override;

  amd_dbgapi_status_t
  suspend_queues (const os_queue_id_t *queues, size_t queue_count,
                  os_exception_mask_t exceptions_cleared,
                  size_t *suspended_count,
                  os_queue_state_t *queue_states) const override;

  amd_dbgapi_status_t
  resume_queues (const os_queue_id_t *queues, size_t queue_count,
                 size_t *resumed_count,
                 os_queue_state_t *queue_states) const override;

  amd_dbgapi_status_t
  queue_snapshot (os_queue_snapshot_entry_t *snapshots, size_t snapshot_count,
                  size_t *queue_count,
                  os_exception_mask_t exceptions_cleared) const override;

  amd_dbgapi_status_t
  set_address_watch (os_agent_id_t os_agent_id, agent_address_t address,
                     agent_address_t mask, os_watch_mode_t os_watch_mode,
                     os_watch_id_t *os_watch_id) const override;

  amd_dbgapi_status_t
  clear_address_watch (os_agent_id_t os_agent_id,
                       os_watch_id_t os_watch_id) const override;

  amd_dbgapi_status_t
  set_wave_launch_mode (os_wave_launch_mode_t mode) const override;

  amd_dbgapi_status_t set_wave_launch_trap_override (
    os_wave_launch_trap_override_t override, os_wave_launch_trap_mask_t value,
    os_wave_launch_trap_mask_t mask,
    os_wave_launch_trap_mask_t *previous_value = nullptr,
    os_wave_launch_trap_mask_t *supported_mask = nullptr) const override;

  amd_dbgapi_status_t
  set_process_flags (os_process_flags_t flags) const override;

  amd_dbgapi_status_t xfer_host_memory_partial (host_address_t address,
                                                void *read, const void *write,
                                                size_t *size) const override;

  amd_dbgapi_status_t xfer_global_memory_partial (global_address_t address,
                                                  void *read,
                                                  const void *write,
                                                  size_t *size) const override;

  amd_dbgapi_status_t xfer_agent_memory_partial (os_agent_id_t agent,
                                                 agent_address_t address,
                                                 void *read, const void *write,
                                                 size_t *size) const override;

private:
  /* List adapters available on the system and populate m_agents
     with supported agents.  */
  amd_dbgapi_status_t discover_adapters ();

  /* Helper function for discover_adapters, populates m_agents available
     behind ADAPTER.  */
  amd_dbgapi_status_t init_adapter (D3DKMT_HANDLE adapter);

  /* Build a queue_id we can return to the core from the agent_id and KMD
     queue_id.  */
  os_queue_id_t make_queue_id (os_agent_id_t agent_id,
                               ULONG kmd_queue_id) const
  {
    /* Agent ID is the agent index number in m_agent, we do not plan to support
       more than 255 agents on a host, so no more than 8 bits are supported.
       The queue ID is the queue doorbell ID, which requires 10 bits.  Even if
       the data given by KMD is shifted, 24 bits are more than enough.

       Therefore, we can safely build a 32 bit value where the low 24 bits are
       built from the queue ID, and the high 8 bits are the agent id.  If it
       turns out this is too restrictive, we can always change os_queue_id_t
       to be a wider type.  */
    if (agent_id > 0xff)
      fatal_error ("Unsupported agent_id %x", agent_id);
    if (kmd_queue_id > 0xffffff)
      fatal_error ("Unsupported kmd_queue_id %lx", kmd_queue_id);

    return (static_cast<uint32_t> (agent_id) << 24
            | static_cast<uint32_t> (kmd_queue_id));
  }

  /* Retrieve the agent_id from a queue_id.  */
  os_agent_id_t queue_agent_id (os_queue_id_t id) const
  {
    return (id >> 24) & 0xff;
  }

  /* Retrieve the queue ID understood by the driver from the queue_id).  */
  ULONG queue_kmd_id (os_queue_id_t id) const { return id & 0xffffff; }

  NTSTATUS send_escape (const kmd::agent_handle_t &agent,
                        KMDDBGRIF_DBGR_CMDS &arg) const;

  /* Helper function used to group queues by agent they belong to.  */
  std::pair<std::vector<const os_queue_id_t *>,
            std::map<os_agent_id_t, std::vector<const os_queue_id_t *>>>
  queue_per_agent (const os_queue_id_t *ids, size_t count) const;

  kmd::d3d_t m_d3d;

  /* List of connected agents.  This is initialized at constructior time.
     Agent IDs reported to the core are indexes in this vector.  */
  std::vector<kmd::agent_handle_t> m_agents;

  /* The agent through which we can contact runtime (if runtime is
     enabled).  */
  mutable std::optional<const kmd::agent_handle_t *> m_agent_to_runtime;
  bool m_is_debug_enabled{ false };

  /* Local process handle used for VM read/write.  */
  HANDLE m_inferior_handle{ nullptr };
};

kmd_driver_t::kmd_driver_t (amd_dbgapi_os_process_id_t os_id)
  : null_driver_t (os_id)
{
  discover_adapters ();

  m_inferior_handle = ::OpenProcess (
    PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION, FALSE, os_id);
}

kmd_driver_t::~kmd_driver_t ()
{
  /* Make sure to close all devices and adapters we kept open.  Since the same
     device can be listed multiple times in a LDA chain, make sure to
     de-duplicate the handles first.  */
  std::set<D3DKMT_HANDLE> devices, adapters;
  for (const auto &handle : m_agents)
    {
      devices.insert (handle.device);
      adapters.insert (handle.adapter);
    }

  for (D3DKMT_HANDLE device : devices)
    {
      D3DKMT_DESTROYDEVICE args{};
      args.hDevice = device;
      if (m_d3d.api.destroy_device (&args) != STATUS_SUCCESS)
        warning ("D3DKMTDestroyDevice failed");
    }

  for (D3DKMT_HANDLE adapter : adapters)
    {
      D3DKMT_CLOSEADAPTER arg{};
      arg.hAdapter = adapter;
      if (m_d3d.api.close_adapter (&arg) != STATUS_SUCCESS)
        warning ("D3DKMT_CLOSEADAPTER failed");
    }

  if (m_inferior_handle != nullptr)
    {
      CloseHandle (m_inferior_handle);
      m_inferior_handle = nullptr;
    }
}

bool
kmd_driver_t::is_valid () const
{
  /* If the ctor did not find any agent at this point, we cannot contact kernel
     driver, so we are invalid.  */
  return m_d3d.is_valid () && m_agents.size () > 0;
}

amd_dbgapi_status_t
kmd_driver_t::discover_adapters ()
{
  /* Query the maximum number of adapters.  */
  D3DKMT_ENUMADAPTERS3 enum_adapters{};
  enum_adapters.Filter.IncludeComputeOnly = true;
  if (m_d3d.api.enum_adapters (&enum_adapters) != STATUS_SUCCESS)
    return AMD_DBGAPI_STATUS_ERROR;

  auto adapt_holder
    = std::make_unique<D3DKMT_ADAPTERINFO[]> (enum_adapters.NumAdapters);
  enum_adapters.pAdapters = adapt_holder.get ();
  if (m_d3d.api.enum_adapters (&enum_adapters) != STATUS_SUCCESS)
    return AMD_DBGAPI_STATUS_ERROR;

  /* Go through all the adapters, and only keep track of the AMDGPU ones.
     Anu unsupported adapter can be closed now, but we need to keep all
     the supported adapters open for the lifetime of the kmd_driver_t.  */
  for (size_t i = 0; i < enum_adapters.NumAdapters; ++i)
    {
      if (init_adapter (enum_adapters.pAdapters[i].hAdapter)
          != AMD_DBGAPI_STATUS_SUCCESS)
        {
          /* Adapter cannot be used, which is expected for non-AMDGPU
             devices.  */
          D3DKMT_CLOSEADAPTER close_adapter{};
          close_adapter.hAdapter = enum_adapters.pAdapters[i].hAdapter;
          if (m_d3d.api.close_adapter (&close_adapter) != STATUS_SUCCESS)
            warning ("D3DKMT_CLOSEADAPTER failed for %xp",
                     enum_adapters.pAdapters[i].hAdapter);
        }
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
kmd_driver_t::init_adapter (D3DKMT_HANDLE adapter)
{
  D3DKMT_CREATEDEVICE create_device = {};
  create_device.hAdapter = adapter;
  create_device.Flags.DisableGpuTimeout = true;
  if (m_d3d.api.create_device (&create_device) != STATUS_SUCCESS)
    {
      warning ("D3DKMT_CREATEDEVICE failed");
      return AMD_DBGAPI_STATUS_ERROR;
    }

  auto destroy_device = utils::make_scope_exit (
    [destroy_device = m_d3d.api.destroy_device,
     device_handle = create_device.hDevice] ()
    {
      D3DKMT_DESTROYDEVICE args{};
      args.hDevice = device_handle;
      if (destroy_device (&args) != STATUS_SUCCESS)
        warning ("D3DKMTDestroyDevice failed");
    });

  PROXY_ADAPTER_INFO proxy_info{};
  D3DKMT_QUERYADAPTERINFO query_info{};
  query_info.Type = KMTQAITYPE_UMDRIVERPRIVATE;
  query_info.pPrivateDriverData = &proxy_info;
  query_info.PrivateDriverDataSize = sizeof (proxy_info);
  query_info.hAdapter = adapter;

  if (m_d3d.api.query_adapter_info (&query_info) != STATUS_SUCCESS)
    return AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED;

  bool use_adapter = false;
  for (size_t chain_id = 0; chain_id < proxy_info.numChainedGpus; ++chain_id)
    {
      static constexpr uint16_t PCIE_VENSOR_ID_AMD = 0x1002;
      if (proxy_info.VendorID[chain_id] == PCIE_VENSOR_ID_AMD)
        {
          use_adapter = true;
          destroy_device.release ();
          m_agents.push_back ({ adapter, create_device.hDevice, chain_id });
        }
    }

  if (!use_adapter)
    return AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED;
  return AMD_DBGAPI_STATUS_SUCCESS;
}

NTSTATUS
kmd_driver_t::send_escape (const kmd::agent_handle_t &agent,
                           KMDDBGRIF_DBGR_CMDS &arg) const
{
  TRACE_DRIVER_BEGIN (param_in (agent), param_in (arg.Input));

  struct
  {
    PROXY_ESCAPE_INFO proxy_header{};
    KMDDBGRIF_DBGR_CMDS dbgcmd{};
  } payload{};
  /* Check that there is no unexpected alignment.  */
  static_assert (sizeof (payload)
                 == sizeof (PROXY_ESCAPE_INFO) + sizeof (KMDDBGRIF_DBGR_CMDS));

  payload.proxy_header.gpuOrdinal = agent.chain_ordinal;
  payload.proxy_header.privateDataLengthBytes = sizeof (KMDDBGRIF_DBGR_CMDS);
  payload.proxy_header.version = PXDRV_HEADER_VERSION;
  payload.proxy_header.adapterDriverId = AdapterProxyDriver;

  /* Override some generic debug escape fields.  It is easier to set those
     here rather than at each call-site.  */
  arg.Header.ulSize = sizeof (arg.Header);
  arg.Header.ulHeaderVersion = ATI_LHESCAPE_HEADER_VER;
  arg.Header.ulEscapeCode = LHESCAPE_UMDKMDIF_SHADER_DBGR;
  arg.Header.ulClientID = ATI_ESCAPE_CLIENT_ID_UMD;
  arg.Input.processId = *m_os_pid;
  arg.InputSize = sizeof (KMDDBGRIF_DBGR_CMDS_INPUT);
  arg.OutputSize = sizeof (KMDDBGRIF_DBGR_CMDS_OUTPUT);

  payload.dbgcmd = arg;

  D3DKMT_ESCAPE escape_info = {};
  escape_info.pPrivateDriverData = &payload;
  escape_info.PrivateDriverDataSize = sizeof (payload);
  escape_info.Flags.HardwareAccess = true;
  escape_info.hDevice = agent.device;
  escape_info.hAdapter = agent.adapter;
  escape_info.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
  escape_info.hContext = 0;

  auto status = m_d3d.api.escape (&escape_info);

  arg = payload.dbgcmd;
  return status;

  TRACE_DRIVER_END (param_out (arg.Output));
}

amd_dbgapi_status_t
kmd_driver_t::check_version () const
{
  using version_t = std::pair<unsigned int, unsigned int>;
  constexpr version_t min_version = { 1, 0 };
  constexpr version_t max_version = { 2, 0 };

  for (const auto &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_GET_TRAP_VERSION;

      if (auto status = send_escape (agent, cmd); status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);

      const version_t kmd_version{ cmd.Output.getTrapVersionOut.majorVersion,
                                   cmd.Output.getTrapVersionOut.minorVersion };
      if (kmd_version < min_version || kmd_version >= max_version)
        {
          warning ("AMDGPU driver's version %u.%u not supported "
                   "(mursion must be >= %u.%u and < %u.%u)",
                   kmd_version.first, kmd_version.second, min_version.first,
                   min_version.second, max_version.first, max_version.second);
          return AMD_DBGAPI_STATUS_ERROR_RESTRICTION;
        }
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
kmd_driver_t::agent_snapshot (os_agent_info_t *snapshots,
                              size_t snapshot_count, size_t *agent_count,
                              os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (snapshots), param_in (snapshot_count),
                      param_in (agent_count), param_in (exceptions_cleared));

  if (!is_debug_enabled ())
    {
      *agent_count = 0;
      return AMD_DBGAPI_STATUS_SUCCESS;
    }

  *agent_count = std::min (snapshot_count, m_agents.size ());

  size_t snapshot_idx = 0;
  for (const auto &agent_handle : m_agents)
    {
      os_agent_info_t &agent = snapshots[snapshot_idx++];
      agent = {};

      /* Query the agent name from the OS.  */
      D3DKMT_QUERYADAPTERINFO query_info{};
      D3DKMT_ADAPTERREGISTRYINFO adapter_reg_info{};
      query_info.Type = KMTQAITYPE_ADAPTERREGISTRYINFO;
      query_info.hAdapter = agent_handle.adapter;
      query_info.pPrivateDriverData = &adapter_reg_info;
      query_info.PrivateDriverDataSize = sizeof (D3DKMT_ADAPTERREGISTRYINFO);

      if (m_d3d.api.query_adapter_info (&query_info) == STATUS_SUCCESS)
        {
          /* We need to convert from wchar string to std::string.  */
          char tmp[MAX_PATH * MB_CUR_MAX] = {};
          size_t tmp_i = 0;
          for (size_t i = 0;
               i < MAX_PATH && adapter_reg_info.AdapterString[i] != 0; i++)
            {
              int r
                = ::wctomb (&tmp[tmp_i], adapter_reg_info.AdapterString[i]);
              if (r == -1)
                tmp[tmp_i++] = '?';
              else
                tmp_i += r;
            }
          agent.name = tmp;
        }
      else
        agent.name = "<unknown>";

      /* Query the rest of the information to MKD.  */
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_GET_DEVICE_INFO;
      cmd.Input.getDeviceInfoIn.exceptionMaskToClear
        = kmd_exception_mask (exceptions_cleared);

      if (auto status = send_escape (agent_handle, cmd);
          status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);

      KMDDBGR_DEVICE_INFO &kmd_snap = cmd.Output.getDeviceInfoOut.deviceInfo;
      agent.os_agent_id = std::distance (&m_agents.front (), &agent_handle);
      agent.gfxip = { kmd_snap.gfxTargetVersion / 10000,
                      (kmd_snap.gfxTargetVersion / 100) % 100,
                      kmd_snap.gfxTargetVersion % 100 };

      agent.domain = kmd_snap.pciSegment;
      agent.location_id = kmd_snap.locationId;

      agent.vendor_id = kmd_snap.vendorId;
      agent.device_id = kmd_snap.deviceId;
      agent.revision_id = kmd_snap.revisionId;
      agent.subsystem_vendor_id = kmd_snap.subVendorId;
      agent.subsystem_device_id = kmd_snap.subDeviceId;

      agent.simd_count = kmd_snap.simdCount;
      agent.max_waves_per_simd = kmd_snap.maxWavesPerSimd;
      agent.shader_engine_count
        = kmd_snap.arrayCount / kmd_snap.simdArraysPerEngine;
      agent.xcc_count = 1; /* KMD does not support multi-XCC architectures.  */

      agent.fw_version = kmd_snap.fwVersion;

      agent.local_address_aperture_base = kmd_snap.ldsBase;
      agent.local_address_aperture_limit = kmd_snap.ldsLimit;
      agent.private_address_aperture_base = kmd_snap.scratchBase;
      agent.private_address_aperture_limit = kmd_snap.scratchLimit;

      agent.debugging_supported
        = kmd_snap.capability & KMD_DBGR_CAP_TRAP_DEBUG_SUPPORT;
      agent.address_watch_supported
        = kmd_snap.capability & KMD_DBGR_CAP_WATCH_POINTS_SUPPORTED;
      agent.address_watch_register_count
        = 1 << ((kmd_snap.capability
                 & KMD_DBGR_CAP_WATCH_POINTS_TOTALBITS_MASK)
                >> KMD_DBGR_CAP_WATCH_POINTS_TOTALBITS_SHIFT);
      agent.precise_memory_supported
        = kmd_snap.capability
          & KMD_DBGR_CAP_TRAP_DEBUG_PRECISE_MEMORY_OPERATIONS_SUPPORTED;
      agent.firmware_supported
        = kmd_snap.capability & KMD_DBGR_CAP_TRAP_DEBUG_FIRMWARE_SUPPORTED;

      agent.address_watch_mask_bits = utils::bit_mask (
        ((kmd_snap.debugProp & KMD_DBGR_DBG_WATCH_ADDR_MASK_LO_BIT_MASK)
         >> KMD_DBGR_DBG_WATCH_ADDR_MASK_LO_BIT_SHIFT),
        ((kmd_snap.debugProp & KMD_DBGR_DBG_WATCH_ADDR_MASK_HI_BIT_MASK)
         >> KMD_DBGR_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT));
      agent.watchpoint_exclusive = false; /* TODO.  */
      agent.ttmps_always_initialized
        = kmd_snap.debugProp & KMD_DBGR_DBG_DISPATCH_INFO_ALWAYS_VALID;
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END (
    make_ref (param_out (snapshots), std::min (snapshot_count, *agent_count)),
    make_ref (param_out (agent_count)));
}

amd_dbgapi_status_t
kmd_driver_t::enable_debug (os_exception_mask_t exceptions_reported,
                            amd_dbgapi_notifier_t notifier,
                            os_runtime_info_t *runtime_info)
{
  TRACE_DRIVER_BEGIN (param_in (exceptions_reported), param_in (notifier),
                      param_in (runtime_info));

  dbgapi_assert (!is_debug_enabled () && "debug is already enabled");

  std::vector<const kmd::agent_handle_t *> activated (m_agents.size ());
  auto rollback = utils::make_scope_exit (
    [&activated, this] ()
    {
      for (auto agent : activated)
        {
          KMDDBGRIF_DBGR_CMDS cmd{};
          cmd.Input.cmd = KMD_DBGR_CMD_OP_ENABLE_GLOBAL_TRAP;
          cmd.Input.enableGlobalTrapIn.enable = false;
          cmd.Input.enableGlobalTrapIn.exceptionMask = 0;
          cmd.Input.enableGlobalTrapIn.umdEventHandle64 = 0;

          if (send_escape (*agent, cmd) != STATUS_SUCCESS)
            warning ("Failed to disable debug");
        }
    });

  for (auto const &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_ENABLE_GLOBAL_TRAP;
      cmd.Input.enableGlobalTrapIn.enable = true;
      cmd.Input.enableGlobalTrapIn.exceptionMask
        = kmd_exception_mask (exceptions_reported);
      cmd.Input.enableGlobalTrapIn.umdEventHandle64
        = reinterpret_cast<uint64_t> (notifier);

      if (auto status = send_escape (agent, cmd); status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);
      activated.push_back (&agent);
    }

  rollback.release ();

  /* Check if runtime is already active.  This is per-process, so we can only
     ask the first agent (TODO can we? Depends which KMD knows about
     runtime.  Maybe I should iterate through all agents and find which one
     has runtime and remember it).*/
  for (auto const &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_GET_RUNTIME_INFO;

      if (send_escape (agent, cmd) == STATUS_SUCCESS
          && os_runtime_state (
               cmd.Output.getRuntimeInfoOut.rocRuntimeInfo.runtimeState)
               == os_runtime_state_t::enabled)
        {
          runtime_info->runtime_state = os_runtime_state (
            cmd.Output.getRuntimeInfoOut.rocRuntimeInfo.runtimeState);
          runtime_info->r_debug = static_cast<amd_dbgapi_global_address_t> (
            cmd.Output.getRuntimeInfoOut.rocRuntimeInfo.rDebug);
          runtime_info->ttmp_setup
            = cmd.Output.getRuntimeInfoOut.rocRuntimeInfo.ttmpSetup;
          m_agent_to_runtime.emplace (&agent);
          break;
        }
    }

  m_is_debug_enabled = true;
  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (runtime_info)));
}

amd_dbgapi_status_t
kmd_driver_t::set_exceptions_reported (
  os_exception_mask_t exceptions_reported) const
{
  TRACE_DRIVER_BEGIN (exceptions_reported);

  /* TODO rollback.  */
  for (auto const &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_ENABLE_EXCEPTIONS;
      cmd.Input.enableExceptionsIn.exceptionMask
        = kmd_exception_mask (exceptions_reported);

      if (auto status = send_escape (agent, cmd); status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kmd_driver_t::disable_debug ()
{
  TRACE_DRIVER_BEGIN ();
  amd_dbgapi_status_t status = AMD_DBGAPI_STATUS_SUCCESS;

  if (!is_debug_enabled ())
    return status;

  for (auto const &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_ENABLE_GLOBAL_TRAP;
      cmd.Input.enableGlobalTrapIn.enable = false;

      NTSTATUS current_status = send_escape (agent, cmd);

      if (current_status == STATUS_DRIVER_PROCESS_TERMINATED)
        status = AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
      else if (current_status != STATUS_SUCCESS)
        status = AMD_DBGAPI_STATUS_ERROR;
    }

  m_is_debug_enabled = false;
  return status;
  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kmd_driver_t::send_exceptions (os_exception_mask_t exceptions,
                               std::optional<os_agent_id_t> agent_id,
                               std::optional<os_queue_id_t> queue_id) const
{
  TRACE_DRIVER_BEGIN (param_in (exceptions), param_in (agent_id),
                      param_in (queue_id));
  dbgapi_assert (is_debug_enabled () && "debug is not enabled");

  /* If per device or per agent, we need to figure which KMD to talk to.  */
  std::optional<const kmd::agent_handle_t *> agent;
  std::optional<ULONG> queue;

  if (agent_id.has_value ())
    {
      if (*agent_id >= m_agents.size ())
        return AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID;
      agent = &m_agents[*agent_id];
    }
  else if (queue_id.has_value ())
    queue = queue_kmd_id (*queue_id);
  else if ((exceptions & os_process_exception_mask)
           != os_exception_mask_t::none)
    {
      /* We need to route the exception to the first KMD instance.  */
      agent = m_agent_to_runtime;
    }

  if (!agent.has_value ())
    return AMD_DBGAPI_STATUS_ERROR;

  KMDDBGRIF_DBGR_CMDS cmd{};
  cmd.Input.cmd = KMD_DBGR_CMD_OP_SEND_EVENT_TO_INFERIOR;
  cmd.Input.sendEventToInferiorIn.exceptionMask
    = kmd_exception_mask (exceptions);
  cmd.Input.sendEventToInferiorIn.queueId = queue.has_value () ? *queue : 0;

  if (auto status = send_escape (*agent.value (), cmd);
      status != STATUS_SUCCESS)
    return nt_status_to_dbgapi_status (status);

  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kmd_driver_t::query_debug_event (os_exception_mask_t *exceptions_present,
                                 os_queue_id_t *os_queue_id,
                                 os_agent_id_t *os_agent_id,
                                 os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (exceptions_present), param_in (os_queue_id),
                      param_in (os_agent_id), param_in (exceptions_cleared));

  dbgapi_assert (exceptions_present && os_queue_id && os_agent_id
                 && "must not be null");

  /* Clear the results for now.  */
  *exceptions_present = os_exception_mask_t::none;
  *os_queue_id = *os_agent_id = 0;

  if (!is_debug_enabled ())
    return AMD_DBGAPI_STATUS_SUCCESS;

  /* TODO some fairness so we round-robin-ish across all the adapters?  */
  for (const auto &agent_handle : m_agents)
    {
      os_agent_id_t agent_id
        = std::distance (&m_agents.front (), &agent_handle);

      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_GET_EXCEPTIONS;
      cmd.Input.getExceptionsIn.exceptionMaskToClear
        = kmd_exception_mask (exceptions_cleared);

      if (auto status = send_escape (agent_handle, cmd);
          status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);

      if (cmd.Output.getExceptionsOut.noExceptions)
        {
          /* This agent has nothing to report, let's see if another agent
             has exceptions to report.  */
          continue;
        }

      *exceptions_present
        = os_exception_mask (cmd.Output.getExceptionsOut.exceptionMask);

      if ((*exceptions_present & os_agent_exception_mask)
          != os_exception_mask_t::none)
        *os_agent_id = agent_id;

      if ((*exceptions_present & os_queue_exception_mask)
          != os_exception_mask_t::none)
        *os_queue_id
          = make_queue_id (agent_id, cmd.Output.getExceptionsOut.queueId);

      return AMD_DBGAPI_STATUS_SUCCESS;
    }

  /* No agent had any exception to report.  */
  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (exceptions_present)),
                    make_ref (param_out (os_queue_id)),
                    make_ref (param_out (os_agent_id)));
}

amd_dbgapi_status_t
kmd_driver_t::query_exception_info (os_exception_code_t exception,
                                    os_source_id_t os_source_id,
                                    os_exception_info_t *os_exception_info,
                                    bool clear_exception) const
{
  TRACE_DRIVER_BEGIN (param_in (exception), param_in (os_source_id),
                      param_in (os_exception_info),
                      param_in (clear_exception));

  dbgapi_assert (is_debug_enabled () && "debug is not enabled");

  /* Curently, dbgapi's core can only query process_runtime exception info.  */
  dbgapi_assert (exception == os_exception_code_t::process_runtime);

  /* TODO, how to check which agent to contact for this exception?  */
  for (auto const &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_GET_EXCEPTION_INFO;
      cmd.Input.getExceptionInfoIn.exceptionId = kmd_exception (exception);
      cmd.Input.getExceptionInfoIn.clearException = clear_exception;

      /* TODO check what KMD returns if there is no such exception.  */
      if (send_escape (agent, cmd) != STATUS_SUCCESS)
        {
          /* The exception might have come from another agent.  */
          continue;
        }

      /* Translate from KMD's exception info to dbgapi's internal
         os_exception_info.  Runtime info is the only one supported at this
         time. */
      os_exception_info->runtime_info.runtime_state = os_runtime_state (
        cmd.Output.getExceptionInfoOut.rocRuntimeInfo.runtimeState);
      if (os_exception_info->runtime_info.runtime_state
          == os_runtime_state_t::enabled)
        {
          /* The runtime got activated, we need to query additional information
             explicitly.  */
          KMDDBGRIF_DBGR_CMDS querystate_cmd{};
          querystate_cmd.Input.cmd = KMD_DBGR_CMD_OP_GET_RUNTIME_INFO;

          if (auto status = send_escape (agent, querystate_cmd);
              status != STATUS_SUCCESS)
            return nt_status_to_dbgapi_status (status);

          /* The runtime state returned by this call is the "most up to date",
             so that's the one we keep.  */
          os_exception_info->runtime_info.runtime_state
            = os_runtime_state (querystate_cmd.Output.getRuntimeInfoOut
                                  .rocRuntimeInfo.runtimeState);
          os_exception_info->runtime_info.r_debug
            = static_cast<amd_dbgapi_global_address_t> (
              querystate_cmd.Output.getRuntimeInfoOut.rocRuntimeInfo.rDebug);
          os_exception_info->runtime_info.ttmp_setup
            = querystate_cmd.Output.getRuntimeInfoOut.rocRuntimeInfo.ttmpSetup;
          m_agent_to_runtime.emplace (&agent);
        }
      else
        {
          os_exception_info->runtime_info.r_debug = 0;
          os_exception_info->runtime_info.ttmp_setup = false;
          m_agent_to_runtime.reset ();
        }
      return AMD_DBGAPI_STATUS_SUCCESS;
    }

  /* Could not wind which agent/KMD the exception came from.  */
  return AMD_DBGAPI_STATUS_ERROR;
  TRACE_DRIVER_END (make_query_ref (exception, param_out (os_exception_info)));
}

amd_dbgapi_status_t
kmd_driver_t::suspend_queues (const os_queue_id_t *queues, size_t queue_count,
                              os_exception_mask_t exceptions_cleared,
                              size_t *suspended_count,
                              os_queue_state_t *queue_states) const
{
  TRACE_DRIVER_BEGIN (make_ref (param_in (queues), queue_count),
                      param_in (queue_count), param_in (exceptions_cleared),
                      param_in (suspended_count), param_in (queue_states));

  *suspended_count = 0;
  auto [invalid_ids, mappings] = queue_per_agent (queues, queue_count);

  /* Any ID we could not map to a KMD driver is reported as invalid.  */
  for (const os_queue_id_t *invalid_queue_id : invalid_ids)
    queue_states[std::distance (queues, invalid_queue_id)]
      = os_queue_state_t::invalid;

  for (const auto &[agent_id, agent_queue_ids] : mappings)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_SUSPEND_QUEUE;
      cmd.Input.suspendQueueIn.exceptionMaskToClear
        = kmd_exception_mask (exceptions_cleared);
      cmd.Input.suspendQueueIn.gracePeriodIn100us = 500;
      cmd.Input.suspendQueueIn.numQueues = agent_queue_ids.size ();
      std::transform (agent_queue_ids.begin (), agent_queue_ids.end (),
                      cmd.Input.suspendQueueIn.queueIds,
                      [this] (auto &&queue_id)
                      { return queue_kmd_id (*queue_id); });

      dbgapi_assert (agent_id < m_agents.size ());
      if (auto status = send_escape (m_agents[agent_id], cmd);
          status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);

      /* Check if any queue was marked invalid or in error state.  */
      for (size_t i = 0; i < agent_queue_ids.size (); ++i)
        {
          auto idx = std::distance (queues, agent_queue_ids[i]);

          queue_states[idx]
            = queue_state (cmd.Output.suspendQueueOut.queueStatus[i]);

          if (cmd.Output.suspendQueueOut.queueStatus[i]
              == KMD_DBGR_USER_QUEUE_STATUS_SUCCESS)
            (*suspended_count)++;
        }
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END (make_ref (param_out (queue_states), queue_count),
                    make_ref (param_out (suspended_count)));
}

amd_dbgapi_status_t
kmd_driver_t::resume_queues (const os_queue_id_t *queues, size_t queue_count,
                             size_t *resumed_count,
                             os_queue_state_t *queue_states) const
{
  TRACE_DRIVER_BEGIN (make_ref (param_in (queues), queue_count),
                      param_in (queue_count), param_in (resumed_count),
                      param_in (queue_states));

  *resumed_count = 0;
  auto [invalid_ids, mappings] = queue_per_agent (queues, queue_count);

  /* Any ID we could not map to a KMD driver is reported as invalid.  */
  for (const os_queue_id_t *invalid_queue_id : invalid_ids)
    queue_states[std::distance (queues, invalid_queue_id)]
      = os_queue_state_t::invalid;

  for (const auto &[agent_id, agent_queue_ids] : mappings)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_RESUME_QUEUE;
      cmd.Input.resumeQueueIn.numQueues = agent_queue_ids.size ();
      std::transform (agent_queue_ids.begin (), agent_queue_ids.end (),
                      cmd.Input.resumeQueueIn.queueIds,
                      [this] (auto &&queue_id)
                      { return queue_kmd_id (*queue_id); });

      /* TODO, we should probably not early return and try to suspend
         accross all agents first.  */
      dbgapi_assert (agent_id < m_agents.size ());
      if (auto status = send_escape (m_agents[agent_id], cmd);
          status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);

      /* Check if any queue was marked invalid or in error state.  */
      for (size_t i = 0; i < agent_queue_ids.size (); ++i)
        {
          auto idx = std::distance (queues, agent_queue_ids[i]);

          queue_states[idx]
            = queue_state (cmd.Output.suspendQueueOut.queueStatus[i]);

          if (cmd.Output.suspendQueueOut.queueStatus[i]
              == KMD_DBGR_USER_QUEUE_STATUS_SUCCESS)
            (*resumed_count)++;
        }
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END (make_ref (param_out (queue_states), queue_count),
                    make_ref (param_out (resumed_count)));
}

amd_dbgapi_status_t
kmd_driver_t::queue_snapshot (os_queue_snapshot_entry_t *snapshots,
                              size_t snapshot_count, size_t *queue_count,
                              os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (snapshots), param_in (snapshot_count),
                      param_in (queue_count), param_in (exceptions_cleared));

  std::vector<std::pair<os_agent_id_t, KMDDBGR_QUEUE_INFO>> queues;
  for (os_agent_id_t agent_id = 0; agent_id < m_agents.size (); ++agent_id)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_GET_QUEUE_INFO;
      cmd.Input.getQueueInfoIn.exceptionMaskToClear
        = kmd_exception_mask (exceptions_cleared);

      if (auto status = send_escape (m_agents[agent_id], cmd);
          status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);

      for (size_t q = 0; q < cmd.Output.getQueueInfoOut.numQueues; ++q)
        {
          /* Ignore graphics queues.  */
          if (cmd.Output.getQueueInfoOut.queueInfo[q].queueType
              == KMD_DBGR_USER_QUEUE_TYPE_GFX)
            continue;

          queues.emplace_back (agent_id,
                               cmd.Output.getQueueInfoOut.queueInfo[q]);
        }
    }

  *queue_count = std::min (snapshot_count, queues.size ());
  for (size_t queue_idx = 0; queue_idx < *queue_count; ++queue_idx)
    {
      auto &[agent_id, snap] = queues[queue_idx];
      os_queue_snapshot_entry_t &queue = snapshots[queue_idx];
      queue.queue_id = make_queue_id (agent_id, snap.queueId);
      queue.gpu_id = agent_id;
      queue.exception_status = os_exception_mask (snap.queueExceptionMask);
      queue.ctx_save_restore_address
        = static_cast<agent_address_t> (snap.ctxSaveRestoreAddress);
      queue.ctx_save_restore_area_size
        = static_cast<amd_dbgapi_size_t> (snap.ctxSaveRestoreSize);

      if (snap.aqlPacketList != 0)
        {
          auto read_host_memory = [this] (host_address_t address, auto *val)
          {
            size_t requested_size = sizeof (*val);
            amd_dbgapi_status_t status = this->xfer_host_memory_partial (
              address, static_cast<void *> (val), nullptr, &requested_size);

            if (status != AMD_DBGAPI_STATUS_SUCCESS
                || requested_size != sizeof (*val))
              fatal_error ("read_host_memory: failed to read %s",
                           to_string (address).c_str ());
          };

          // Software emulated AQL queue:
          queue.queue_type = os_queue_type_t::compute_aql;

          host_address_t read_pointer_address
            = static_cast<host_address_t> (snap.aqlPacketList);

          queue.read_pointer_address = read_pointer_address;
          queue.write_pointer_address
            = queue.read_pointer_address
              + offsetof (amd_queue_t, write_dispatch_id)
              - offsetof (amd_queue_t, read_dispatch_id);

          uint32_t hsa_queue_base_offset;
          read_host_memory (
            read_pointer_address
              + offsetof (amd_queue_t, read_dispatch_id_field_base_byte_offset)
              - offsetof (amd_queue_t, read_dispatch_id),
            &hsa_queue_base_offset);

          hsa_queue_t hsa_queue;
          read_host_memory (read_pointer_address - hsa_queue_base_offset,
                            &hsa_queue);

          queue.ring_base_address
            = reinterpret_cast<amd_dbgapi_global_address_t> (
              hsa_queue.base_address);
          queue.ring_size
            = static_cast<amd_dbgapi_size_t> (hsa_queue.size << 6);
        }
      else
        {
          queue.queue_type = os_queue_type (snap.queueType);
          queue.ring_base_address = static_cast<host_address_t> (
            snap.ringBufferAddress);
          queue.write_pointer_address
            = static_cast<host_address_t> (snap.writePtrAddress);
          queue.read_pointer_address
            = static_cast<host_address_t> (snap.readPtrAddress);
          queue.ring_size = static_cast<amd_dbgapi_size_t> (snap.ringSize);
        }
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END (
    make_ref (param_out (snapshots), std::min (snapshot_count, *queue_count)),
    make_ref (param_out (queue_count)));
}

amd_dbgapi_status_t
kmd_driver_t::set_address_watch (os_agent_id_t os_agent_id,
                                 agent_address_t address, agent_address_t mask,
                                 os_watch_mode_t os_watch_mode,
                                 os_watch_id_t *os_watch_id) const
{
  TRACE_DRIVER_BEGIN (param_in (os_agent_id), param_in (address),
                      param_in (mask), param_in (os_watch_mode),
                      param_in (os_watch_id));

  if (os_agent_id >= m_agents.size ())
    return AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID;

  /* TODO we should use the number of watchpoints from the agent info snapshot.
   */
  int id = 0;
  for (; id < 4; id += 1)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_SET_ADDR_WATCH;
      cmd.Input.setAddrWatchIn.watchId
        = static_cast<KMDDBGRIF_ADDR_WATCH_ID> (id);
      cmd.Input.setAddrWatchIn.mode = kmd_addr_watch_mode (os_watch_mode);
      cmd.Input.setAddrWatchIn.watchAddr = address;
      cmd.Input.setAddrWatchIn.watchAddrMask = mask;

      NTSTATUS status = send_escape (m_agents[os_agent_id], cmd);
      if (status == STATUS_RESOURCE_IN_USE)
        continue;
      else if (status == STATUS_INVALID_PARAMETER)
        /* We get this when trying to set an invalid watchpoint.  */
        return AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE;
      else if (status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);
      else if (status == STATUS_SUCCESS)
        {
          *os_watch_id = id;
          return AMD_DBGAPI_STATUS_SUCCESS;
        }
    }

  if (id > 3)
    return AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE;

  return AMD_DBGAPI_STATUS_ERROR;

  TRACE_DRIVER_END (make_ref (param_out (os_watch_id)));
}

amd_dbgapi_status_t
kmd_driver_t::clear_address_watch (os_agent_id_t os_agent_id,
                                   os_watch_id_t os_watch_id) const
{
  TRACE_DRIVER_BEGIN (param_in (os_agent_id), param_in (os_watch_id));

  if (os_agent_id >= m_agents.size ())
    return AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID;

  KMDDBGRIF_DBGR_CMDS cmd{};
  cmd.Input.cmd = KMD_DBGR_CMD_OP_CLEAR_ADDR_WATCH;
  cmd.Input.setAddrWatchIn.watchId
    = static_cast<KMDDBGRIF_ADDR_WATCH_ID> (os_watch_id);

  return nt_status_to_dbgapi_status (send_escape (m_agents[os_agent_id], cmd));

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kmd_driver_t::set_wave_launch_mode (os_wave_launch_mode_t mode) const
{
  TRACE_DRIVER_BEGIN (param_in (mode));

  for (const auto &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_SET_WAVE_LAUNCH_MODE;
      cmd.Input.setWaveLaunchModeIn.mode = kmd_wave_launch_mode (mode);

      if (auto status = send_escape (agent, cmd); status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kmd_driver_t::set_wave_launch_trap_override (
  os_wave_launch_trap_override_t override, os_wave_launch_trap_mask_t value,
  os_wave_launch_trap_mask_t mask, os_wave_launch_trap_mask_t *previous_value,
  os_wave_launch_trap_mask_t *supported_mask) const
{
  TRACE_DRIVER_BEGIN (param_in (override), param_in (value), param_in (mask),
                      param_in (previous_value), param_in (supported_mask));
  for (const auto &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_ENABLE_TRAPS_FOR_EXCEPTIONS;
      cmd.Input.enableTrapsForExceptionsIn.mode
        = kmd_enable_traps_for_exceptions_mode (override);
      cmd.Input.enableTrapsForExceptionsIn.trapEnableMask
        = kmd_wave_launch_traps_mask (value);
      cmd.Input.enableTrapsForExceptionsIn.trapRequestedMask
        = kmd_wave_launch_traps_mask (mask);

      if (auto status = send_escape (agent, cmd); status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);

      /* TODO, what to do?.  */
      if (previous_value != nullptr)
        *previous_value = static_cast<os_wave_launch_trap_mask_t> (
          cmd.Output.enableTrapsForExceptionsOut.trapSupportMask);
    }
  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END (make_ref (param_out (previous_value)),
                    make_ref (param_out (supported_mask)));
}

amd_dbgapi_status_t
kmd_driver_t::set_process_flags (os_process_flags_t flags) const
{
  TRACE_DRIVER_BEGIN (param_in (flags));
  /* TODO rollback.  */
  for (const auto &agent : m_agents)
    {
      KMDDBGRIF_DBGR_CMDS cmd{};
      cmd.Input.cmd = KMD_DBGR_CMD_OP_SET_PRECISE_MEMOPS;
      cmd.Input.setPreciseMemOpsIn.enable
        = !!(flags & os_process_flags_t::precise_memory);

      if (auto status = send_escape (agent, cmd); status != STATUS_SUCCESS)
        return nt_status_to_dbgapi_status (status);
    }

  return AMD_DBGAPI_STATUS_SUCCESS;
  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kmd_driver_t::xfer_global_memory_partial (global_address_t address, void *read,
                                          const void *write,
                                          size_t *size) const
{
  TRACE_DRIVER_BEGIN (param_in (address), param_in (read), param_in (write),
                      param_in (size));

  /* Global is either on the host or the device.  */
  if (amd_dbgapi_status_t status
      = xfer_host_memory_partial (address, read, write, size);
      status == AMD_DBGAPI_STATUS_SUCCESS)
    return status;

  /* For HIP on windows we do not have a unified global memory accessible from
     the host.  For almost all allocations, we however have the guarantee that
     the address will only exist on the host or exactly one GPU.  Since the
     address was not valid on the host, try every GPU in turn to find if one of
     them "owns" this address.  */
  for (os_agent_id_t agent_id = 0; agent_id < m_agents.size (); ++agent_id)
    {
      if (amd_dbgapi_status_t status
          = xfer_agent_memory_partial (agent_id, address, read, write, size);
          status == AMD_DBGAPI_STATUS_SUCCESS)
        return status;
      else if (status == AMD_DBGAPI_STATUS_ERROR_MEMORY_UNAVAILABLE)
        return status;
    }

  /* The memory could not be accessed using any of the known agents, this must
     be an invalid address.  */
  return AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS;

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kmd_driver_t::xfer_host_memory_partial (host_address_t address, void *read,
                                        const void *write, size_t *size) const
{
  TRACE_DRIVER_BEGIN (param_in (address), param_in (read), param_in (write),
                      param_in (size));

  SIZE_T processed_size = 0;
  if ((read == nullptr) == (write == nullptr))
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

  if (((read != nullptr)
         ? ::ReadProcessMemory (m_inferior_handle, (LPCVOID)(uintptr_t)address,
                                read, *size, &processed_size)
         : ::WriteProcessMemory (m_inferior_handle, (LPVOID)(uintptr_t)address,
                                 write, *size, &processed_size))
      != 0)
    {
      *size = processed_size;
      return AMD_DBGAPI_STATUS_SUCCESS;
    }

  return AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS;

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kmd_driver_t::xfer_agent_memory_partial (os_agent_id_t agent_id,
                                         agent_address_t address, void *read,
                                         const void *write, size_t *size) const
{
  TRACE_DRIVER_BEGIN (param_in (address), param_in (read), param_in (write),
                      param_in (size));

  if ((read == nullptr) == (write == nullptr))
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

  if (agent_id >= m_agents.size ())
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

  /* KMD requires that the CPU VA is word-aligned (both when source
     and destination), and the READ/WRITE pointers might not be
     aligned.  If so, use an intermediate guaranteed-aligned
     buffer.  */
  size_t misalign = (read != nullptr
		     ? (uintptr_t) read % sizeof (DWORD)
		     : (uintptr_t) write % sizeof (DWORD));
  if (misalign != 0)
    {
      /* Large enough to fit most xfers in one escape call, and small
	 enough to not cause stack overflow problems.  */
      constexpr size_t aligned_buffer_size = 64;
      alignas(DWORD) std::byte aligned_buffer[aligned_buffer_size];

      /* If we need a second xfer, read/write enough bytes such that
	 that xfer will be aligned.  */
      size_t partial_size = (*size > sizeof (aligned_buffer)
			     ? misalign
			     : *size);

      if (write != nullptr)
	std::memcpy (aligned_buffer, write, partial_size);

      /* Recurse to do the aligned partial xfer.  */
      size_t wanted_partial_size = partial_size;
      amd_dbgapi_status_t status
	= xfer_agent_memory_partial (agent_id,
				     address,
				     (read != nullptr
				      ? aligned_buffer
				      : nullptr),
				     (write != nullptr
				      ? aligned_buffer
				      : nullptr),
				     &partial_size);
      if (status != AMD_DBGAPI_STATUS_SUCCESS)
	return status;
      if (read != nullptr)
	std::memcpy (read, aligned_buffer, partial_size);

      /* If we got less than we wanted, then we're already done.  */
      if (wanted_partial_size != partial_size)
	{
	  *size = partial_size;
	  return status;
	}

      /* Now xfer the remainder.  Several callers don't currently
	 handle partial xfers, so if this fails, return failure for
	 the whole xfer.  Once all such callers are fixed, this whole
	 if block can be removed.  */
      size_t remaining_size = *size - partial_size;
      if (remaining_size != 0)
	{
	  if (read != nullptr)
	    read = (char *) read + partial_size;
	  else
	    write = (char *) write + partial_size;

	  /* This access is now guaranteed to be word-aligned.  */
	  size_t new_misalign = (read != nullptr
				 ? (uintptr_t) read % sizeof (DWORD)
				 : (uintptr_t) write % sizeof (DWORD));
	  dbgapi_assert (new_misalign == 0);

	  address += partial_size;

	  /* Recurse.  */
	  status
	    = xfer_agent_memory_partial (agent_id, address,
					 read, write, &remaining_size);

	  /* Don't update SIZE unless we succeeded.  */
	  if (status != AMD_DBGAPI_STATUS_SUCCESS)
	    return status;
	}

      *size = partial_size + remaining_size;
      return status;
    }

  KMDDBGRIF_DBGR_CMDS cmd{};
  if (read == nullptr)
    {
      cmd.Input.cmd = KMD_DBGR_CMD_OP_WRITE_BUFFER;
      cmd.Input.writeBufferIn.srcCpuVA = reinterpret_cast<uint64_t> (write);
      cmd.Input.writeBufferIn.dstGpuVA = static_cast<uint64_t> (address);
      cmd.Input.writeBufferIn.size = *size;
    }
  else
    {
      cmd.Input.cmd = KMD_DBGR_CMD_OP_READ_BUFFER;
      cmd.Input.readBufferIn.srcGpuVA = static_cast<uint64_t> (address);
      cmd.Input.readBufferIn.dstCpuVA = reinterpret_cast<uint64_t> (read);
      cmd.Input.readBufferIn.size = *size;
    }

  auto status = send_escape (m_agents[agent_id], cmd);
  if (status == STATUS_SUCCESS)
    {
      *size = ((read == nullptr) ? cmd.Output.writeBufferOut.bytesWritten
                                 : cmd.Output.readBufferOut.bytesRead);
      return AMD_DBGAPI_STATUS_SUCCESS;
    }
  else if (status == STATUS_RETRY)
    return AMD_DBGAPI_STATUS_ERROR_MEMORY_UNAVAILABLE;

  return AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS;

  TRACE_DRIVER_END ();
}

std::pair<std::vector<const os_queue_id_t *>,
          std::map<os_agent_id_t, std::vector<const os_queue_id_t *>>>
kmd_driver_t::queue_per_agent (const os_queue_id_t *qs, size_t count) const
{
  std::vector<const os_queue_id_t *> invalid_ids;
  std::map<os_agent_id_t, std::vector<const os_queue_id_t *>> map;
  for (size_t i = 0; i < count; ++i)
    {
      const os_agent_id_t agent_id = queue_agent_id (qs[i]);
      if (agent_id >= m_agents.size ())
        invalid_ids.push_back (&qs[i]);
      else
        map[agent_id].push_back (&qs[i]);
    };
  return { invalid_ids, map };
}

std::unique_ptr<os_driver_t>
os_driver_t::create_driver (std::optional<amd_dbgapi_os_process_id_t> os_pid)
{
  if (!os_pid)
    return std::make_unique<null_driver_t> ();

  std::unique_ptr<os_driver_t> os_driver
    = std::make_unique<kmd_driver_t> (os_pid.value ());
  if (os_driver->is_valid ())
    return os_driver;

  return std::make_unique<null_driver_t> ();
}

std::unique_ptr<os_driver_t>
os_driver_t::create_driver (amd_dbgapi_client_process_id_t,
                            const amd_dbgapi_core_state_data_t &)
{
  /* Core dumps are not supported with KMD.  */
  return std::make_unique<null_driver_t> ();
}

} /* namespace amd::dbgapi */
