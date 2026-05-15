/* Copyright (c) 2021-2023 Advanced Micro Devices, Inc.

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
#include "linux/kfd_ioctl.h"
#include "linux/kfd_sysfs.h"
#include "logging.h"
#include "process.h"
#include "utils.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

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

/* Convert a KFD queue type to a os_queue_type_t.  */

static constexpr os_queue_type_t
os_queue_type (decltype (kfd_queue_snapshot_entry::queue_type) kfd_type)
{
  switch (kfd_type)
    {
    case KFD_IOC_QUEUE_TYPE_COMPUTE:
      return os_queue_type_t::compute;
    case KFD_IOC_QUEUE_TYPE_SDMA:
      return os_queue_type_t::sdma;
    case KFD_IOC_QUEUE_TYPE_COMPUTE_AQL:
      return os_queue_type_t::compute_aql;
    case KFD_IOC_QUEUE_TYPE_SDMA_XGMI:
      return os_queue_type_t::sdma_xgmi;
    }

  return os_queue_type_t::unknown;
}

/* Convert a kfd_exception_code_t code to a KFD code.  */

static constexpr decltype (kfd_ioctl_dbg_trap_query_exception_info_args::
                             exception_code)
kfd_exception_code (os_exception_code_t ec_code)
{
  switch (ec_code)
    {
    case os_exception_code_t::none:
      return EC_NONE;

    case os_exception_code_t::queue_wave_abort:
      return EC_QUEUE_WAVE_ABORT;
    case os_exception_code_t::queue_wave_trap:
      return EC_QUEUE_WAVE_TRAP;
    case os_exception_code_t::queue_wave_math_error:
      return EC_QUEUE_WAVE_MATH_ERROR;
    case os_exception_code_t::queue_wave_illegal_instruction:
      return EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION;
    case os_exception_code_t::queue_wave_memory_violation:
      return EC_QUEUE_WAVE_MEMORY_VIOLATION;
    case os_exception_code_t::queue_wave_address_error:
      return EC_QUEUE_WAVE_APERTURE_VIOLATION;
    case os_exception_code_t::queue_packet_dispatch_dim_invalid:
      return EC_QUEUE_PACKET_DISPATCH_DIM_INVALID;
    case os_exception_code_t::queue_packet_dispatch_group_segment_size_invalid:
      return EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID;
    case os_exception_code_t::queue_packet_dispatch_code_invalid:
      return EC_QUEUE_PACKET_DISPATCH_CODE_INVALID;
    case os_exception_code_t::queue_packet_unsupported:
      return EC_QUEUE_PACKET_UNSUPPORTED;
    case os_exception_code_t::queue_packet_dispatch_work_group_size_invalid:
      return EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID;
    case os_exception_code_t::queue_packet_dispatch_register_invalid:
      return EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID;
    case os_exception_code_t::queue_packet_vendor_unsupported:
      return EC_QUEUE_PACKET_VENDOR_UNSUPPORTED;
    case os_exception_code_t::queue_preemption_error:
      return EC_QUEUE_PREEMPTION_ERROR;
    case os_exception_code_t::queue_new:
      return EC_QUEUE_NEW;

    case os_exception_code_t::device_queue_delete:
      return EC_DEVICE_QUEUE_DELETE;
    case os_exception_code_t::device_memory_violation:
      return EC_DEVICE_MEMORY_VIOLATION;
    case os_exception_code_t::device_ras_error:
      return EC_DEVICE_RAS_ERROR;
    case os_exception_code_t::device_fatal_halt:
      return EC_DEVICE_FATAL_HALT;
    case os_exception_code_t::device_new:
      return EC_DEVICE_NEW;

    case os_exception_code_t::process_runtime:
      return EC_PROCESS_RUNTIME;
    case os_exception_code_t::process_device_remove:
      return EC_PROCESS_DEVICE_REMOVE;
    }

  dbgapi_assert_not_reached ("Unknown exception code");
}

static constexpr std::optional<os_exception_code_t>
os_exception_code (kfd_dbg_trap_exception_code kfd_code)
{
  switch (kfd_code)
    {
    case EC_NONE:
      return os_exception_code_t::none;
    case EC_QUEUE_WAVE_ABORT:
      return os_exception_code_t::queue_wave_abort;
    case EC_QUEUE_WAVE_TRAP:
      return os_exception_code_t::queue_wave_trap;
    case EC_QUEUE_WAVE_MATH_ERROR:
      return os_exception_code_t::queue_wave_math_error;
    case EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION:
      return os_exception_code_t::queue_wave_illegal_instruction;
    case EC_QUEUE_WAVE_MEMORY_VIOLATION:
      return os_exception_code_t::queue_wave_memory_violation;
    case EC_QUEUE_WAVE_APERTURE_VIOLATION:
      return os_exception_code_t::queue_wave_address_error;
    case EC_QUEUE_PACKET_DISPATCH_DIM_INVALID:
      return os_exception_code_t::queue_packet_dispatch_dim_invalid;
    case EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID:
      return os_exception_code_t::
        queue_packet_dispatch_group_segment_size_invalid;
    case EC_QUEUE_PACKET_DISPATCH_CODE_INVALID:
      return os_exception_code_t::queue_packet_dispatch_code_invalid;
    case EC_QUEUE_PACKET_UNSUPPORTED:
      return os_exception_code_t::queue_packet_unsupported;
    case EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID:
      return os_exception_code_t::
        queue_packet_dispatch_work_group_size_invalid;
    case EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID:
      return os_exception_code_t::queue_packet_dispatch_register_invalid;
    case EC_QUEUE_PACKET_VENDOR_UNSUPPORTED:
      return os_exception_code_t::queue_packet_vendor_unsupported;
    case EC_QUEUE_PREEMPTION_ERROR:
      return os_exception_code_t::queue_preemption_error;
    case EC_QUEUE_NEW:
      return os_exception_code_t::queue_new;
    case EC_DEVICE_QUEUE_DELETE:
      return os_exception_code_t::device_queue_delete;
    case EC_DEVICE_MEMORY_VIOLATION:
      return os_exception_code_t::device_memory_violation;
    case EC_DEVICE_RAS_ERROR:
      return os_exception_code_t::device_ras_error;
    case EC_DEVICE_FATAL_HALT:
      return os_exception_code_t::device_fatal_halt;
    case EC_DEVICE_NEW:
      return os_exception_code_t::device_new;
    case EC_PROCESS_RUNTIME:
      return os_exception_code_t::process_runtime;
    case EC_PROCESS_DEVICE_REMOVE:
      return os_exception_code_t::process_device_remove;

    case EC_QUEUE_PACKET_RESERVED:
      [[fallthrough]];
    case EC_MAX:
      return {};
    }
  return {};
}

/* Convert a runtime state returned by KFD to os_runtime_state_t.  */

static constexpr os_runtime_state_t
os_runtime_state (decltype (kfd_runtime_info::runtime_state) state)
{
  switch (state)
    {
    case DEBUG_RUNTIME_STATE_DISABLED:
      return os_runtime_state_t::disabled;
    case DEBUG_RUNTIME_STATE_ENABLED:
      return os_runtime_state_t::enabled;
    case DEBUG_RUNTIME_STATE_ENABLED_BUSY:
      return os_runtime_state_t::enabled_busy;
    case DEBUG_RUNTIME_STATE_ENABLED_ERROR:
      return os_runtime_state_t::enabled_error;
    }

  return os_runtime_state_t::disabled;
}

/* Convert a KFD exception mask to os_exception_mask_t.  */

static constexpr os_exception_mask_t
kfd_to_os_exception_mask (__u64 kfd_mask)
{
  os_exception_mask_t mask{};

  if (kfd_mask == 0)
    return mask;

  while (kfd_mask != 0)
    {
      __u64 one_bit = kfd_mask ^ (kfd_mask & (kfd_mask - 1));

      auto code = os_exception_code (
        excp_mask_to_excp_code<kfd_dbg_trap_exception_code> (one_bit));

      if (code.has_value ())
        mask |= os_exception_mask (code.value ());
      else
        warning ("Unknown KFD exception code %" PRIx64,
                 static_cast<uint64_t> (one_bit));

      kfd_mask ^= one_bit;
    }

  return mask;
}

/* Convert a os_exception_mask_t to a KFD excepton mask.  */

static constexpr __u64
kfd_exception_mask (os_exception_mask_t mask)
{
  __u64 kfd_mask = 0;
  if (mask == os_exception_mask_t::none)
    return kfd_mask;

  while (mask != os_exception_mask_t::none)
    {
      os_exception_mask_t one_bit = mask ^ (mask & (mask - 1));

      kfd_mask |= KFD_EC_MASK (kfd_exception_code (
        excp_mask_to_excp_code<os_exception_code_t, os_exception_mask_t> (
          one_bit)));
      mask ^= one_bit;
    }

  return kfd_mask;
}

/* Sanity check: ensure that we agree with kfd_ioctl.h.  */

/* KFD counts EC_QUEUE_PACKET_RESERVED in KFD_EC_MASK_QUEUE, which is not
   a valid exception code ID for dbgapi.  */
static_assert (kfd_exception_mask (os_queue_exception_mask)
               == (KFD_EC_MASK_QUEUE
                   & ~KFD_EC_MASK (EC_QUEUE_PACKET_RESERVED)));

static_assert (kfd_exception_mask (os_agent_exception_mask)
               == KFD_EC_MASK_DEVICE);

static_assert (kfd_exception_mask (os_process_exception_mask)
               == KFD_EC_MASK_PROCESS);

/* Convert a os_wave_launch_trap_override_t to a KFD wave launch override
   mode.  */

static constexpr auto
kfd_wave_launch_trap_override (os_wave_launch_trap_override_t o)
{
  switch (o)
    {
    case os_wave_launch_trap_override_t::apply:
      return KFD_DBG_TRAP_OVERRIDE_OR;
    case os_wave_launch_trap_override_t::replace:
      return KFD_DBG_TRAP_OVERRIDE_REPLACE;
    }

  dbgapi_assert_not_reached ("Unknown wave launch trap override");
}

/* Convert a os_wave_launch_trap_mask_t to a KFD wave_launch_trap_mask.  */

static constexpr __u32
kfd_wave_launch_trap_mask (os_wave_launch_trap_mask_t wave_launch_trap)
{
  __u32 kfd_wave_launch_trap = 0;

  if (wave_launch_trap == os_wave_launch_trap_mask_t::none)
    return kfd_wave_launch_trap;

  auto convert_one = [] (os_wave_launch_trap_mask_t one_bit) -> __u32
  {
    switch (one_bit)
      {
      case os_wave_launch_trap_mask_t::none:
        return 0;
      case os_wave_launch_trap_mask_t::fp_invalid:
        return KFD_DBG_TRAP_MASK_FP_INVALID;
      case os_wave_launch_trap_mask_t::fp_input_denormal:
        return KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL;
      case os_wave_launch_trap_mask_t::fp_divide_by_zero:
        return KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO;
      case os_wave_launch_trap_mask_t::fp_overflow:
        return KFD_DBG_TRAP_MASK_FP_OVERFLOW;
      case os_wave_launch_trap_mask_t::fp_underflow:
        return KFD_DBG_TRAP_MASK_FP_UNDERFLOW;
      case os_wave_launch_trap_mask_t::fp_inexact:
        return KFD_DBG_TRAP_MASK_FP_INEXACT;
      case os_wave_launch_trap_mask_t::int_divide_by_zero:
        return KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO;
      case os_wave_launch_trap_mask_t::address_watch:
        return KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH;
      case os_wave_launch_trap_mask_t::wave_start:
        return KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START;
      case os_wave_launch_trap_mask_t::wave_end:
        return KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END;
      }

    dbgapi_assert_not_reached ();
  };

  while (wave_launch_trap != os_wave_launch_trap_mask_t::none)
    {
      os_wave_launch_trap_mask_t one_bit
        = wave_launch_trap ^ (wave_launch_trap & (wave_launch_trap - 1));

      kfd_wave_launch_trap |= convert_one (one_bit);
      wave_launch_trap ^= one_bit;
    }

  return kfd_wave_launch_trap;
}

/* Convert a KFD wave_launch_trap_mask to os_wave_launch_trap_mask_t.  */

static constexpr os_wave_launch_trap_mask_t
os_wave_launch_trap_mask (__u32 wave_launch_trap)
{
  os_wave_launch_trap_mask_t mask{};

  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_FP_INVALID))
    mask |= os_wave_launch_trap_mask_t::fp_invalid;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL))
    mask |= os_wave_launch_trap_mask_t::fp_input_denormal;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO))
    mask |= os_wave_launch_trap_mask_t::fp_divide_by_zero;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_FP_OVERFLOW))
    mask |= os_wave_launch_trap_mask_t::fp_overflow;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_FP_UNDERFLOW))
    mask |= os_wave_launch_trap_mask_t::fp_underflow;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_FP_INEXACT))
    mask |= os_wave_launch_trap_mask_t::fp_inexact;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO))
    mask |= os_wave_launch_trap_mask_t::int_divide_by_zero;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH))
    mask |= os_wave_launch_trap_mask_t::address_watch;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START))
    mask |= os_wave_launch_trap_mask_t::wave_start;
  if (!!(wave_launch_trap & KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END))
    mask |= os_wave_launch_trap_mask_t::wave_end;

  return mask;
}

/* Convert from os_watch_mode_t to the appropriate KFD type.  */

static constexpr kfd_dbg_trap_address_watch_mode
kfd_watch_mode (os_watch_mode_t mode)
{
  switch (mode)
    {
    case os_watch_mode_t::read:
      return KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ;
    case os_watch_mode_t::nonread:
      return KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD;
    case os_watch_mode_t::atomic:
      return KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC;
    case os_watch_mode_t::all:
      return KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL;
    }

  dbgapi_assert_not_reached ("Invalid watch mode");
}

/* Decode a KFD queue_id.  */
std::tuple<os_queue_id_t, os_queue_state_t>
decode_queue_id (__u32 queue_id)
{
  os_queue_state_t queue_state{};
  if (queue_id & KFD_DBG_QUEUE_ERROR_MASK)
    queue_state |= os_queue_state_t::error;
  if (queue_id & KFD_DBG_QUEUE_INVALID_MASK)
    queue_state |= os_queue_state_t::invalid;

  return { queue_id & ~(KFD_DBG_QUEUE_ERROR_MASK | KFD_DBG_QUEUE_INVALID_MASK),
           queue_state };
}

/* Convert from os_wave_launch_mode_t to the appropriate KFD type.  */

static constexpr kfd_dbg_trap_wave_launch_mode
kfd_wave_launch_mode (os_wave_launch_mode_t mode)
{
  switch (mode)
    {
    case os_wave_launch_mode_t::normal:
      return KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL;
    case os_wave_launch_mode_t::halt:
      return KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT;
    case os_wave_launch_mode_t::single_step:
      return KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG;

    case os_wave_launch_mode_t::kill:
    case os_wave_launch_mode_t::disable:
      dbgapi_assert (false && "Unsupported wave launch mode");
    }

  dbgapi_assert_not_reached ("Unknown wave launch mode");
}

/* OS Driver implementation for the Linux ROCm stack using KFD.  */

class kfd_driver_base_t : public null_driver_t
{
public:
  explicit kfd_driver_base_t (std::optional<amd_dbgapi_os_process_id_t> os_pid)
    : null_driver_t{ os_pid }
  {
  }

  amd_dbgapi_status_t check_version () const override final;

  amd_dbgapi_status_t
  agent_snapshot (os_agent_info_t *snapshots, size_t snapshot_count,
                  size_t *agent_count,
                  os_exception_mask_t exceptions_cleared) const override final;

  amd_dbgapi_status_t
  queue_snapshot (os_queue_snapshot_entry_t *snapshots, size_t snapshot_count,
                  size_t *queue_count,
                  os_exception_mask_t exceptions_cleared) const override final;

protected:
  using version_t = std::pair<uint32_t, uint32_t>;

  /* Query KFD version.  */
  virtual version_t get_kfd_version () const = 0;

  /* Perform the ioctl call, or act as if using core file provided data.  */
  virtual amd_dbgapi_status_t
  kfd_agent_snapshot (kfd_dbg_device_info_entry *agents, size_t snapshot_count,
                      size_t *agent_count,
                      os_exception_mask_t exceptions_cleared) const
    = 0;

  virtual amd_dbgapi_status_t
  kfd_queue_snapshot (kfd_queue_snapshot_entry *queues, size_t snapshot_count,
                      size_t *queue_cout,
                      os_exception_mask_t exceptions_cleared) const
    = 0;

  amd_dbgapi_status_t xfer_host_memory_partial (host_address_t address,
                                                void *read, const void *write,
                                                size_t *size) const override
  {
    /* We have a unified address space, so just use the global memory access
       to handle agent memory.  */
    return xfer_global_memory_partial (global_address_t{ address }, read,
                                       write, size);
  }

  amd_dbgapi_status_t xfer_agent_memory_partial (os_agent_id_t /* agent  */,
                                                 agent_address_t address,
                                                 void *read, const void *write,
                                                 size_t *size) const override
  {
    /* We have a unified address space, so just use the global memory access
       to handle agent memory.  */
    return xfer_global_memory_partial (global_address_t{ address }, read,
                                       write, size);
  }

private:
  static std::string marketing_name (uint32_t vendor_id, uint32_t device_id,
                                     uint32_t revision_id);
};

/* Find the marketing name for the PCI device VENDOR_ID:DEVICE_ID.REV_ID.

   The information is extracted from the amdgpu.ids database shipped with
   libdrm.  */

std::string
kfd_driver_base_t::marketing_name (uint32_t vendor_id, uint32_t device_id,
                                   uint32_t revision_id)
{
  auto fallback_name = [vendor_id, device_id, revision_id] () -> std::string
  {
    std::stringstream name;
    name << "Device " << std::hex << std::setfill ('0') << std::setw (4)
         << vendor_id << ':' << std::setw (4) << device_id << '.'
         << std::setw (2) << revision_id;
    return name.str ();
  };

  auto amdgpu_ids = [] () -> std::optional<std::ifstream>
  {
    static const std::array paths{ "/opt/amdgpu/share/libdrm/amdgpu.ids",
                                   "/usr/share/libdrm/amdgpu.ids" };
    for (auto p : paths)
      {
        std::ifstream st (p);
        if (st.is_open ())
          return { std::move (st) };
      }
    return {};
  }();

  if (!amdgpu_ids.has_value ())
    {
      static bool warned = false;
      if (!warned)
        {
          warning ("Cannot locate the amdgpu.ids file.");
          warned = true;
        }
      return fallback_name ();
    }

  /* Skip over the beginning of the file.  */
  for (std::string line; std::getline (*amdgpu_ids, line);)
    {
      /* Ignore empty lines and comments.  */
      if (line == "" || line[0] == '#')
        continue;

      if (line != "1.0.0")
        {
          warning ("Unsupported amdgpu.ids file version: '%s'", line.c_str ());
          return fallback_name ();
        }

      break;
    }

  /* From now on, each line has the following format:
     DEV_ID,\tREV_ID,\tDEVICE NAME\n  */
  uint32_t did, rid;
  char comma;
  std::string name, delim;
  while (*amdgpu_ids && !amdgpu_ids->eof ())
    {
      if (!(*amdgpu_ids >> std::hex >> did >> comma >> std::ws)
          || !(*amdgpu_ids >> std::hex >> rid >> comma >> std::ws))
        {
          /* Can't parse the did / rid, skip the rest of the line.  */
          amdgpu_ids->clear ();
          amdgpu_ids->ignore (std::numeric_limits<std::streamsize>::max (),
                              '\n');
          continue;
        }
      std::getline (*amdgpu_ids, name);

      if (did == device_id && rid == revision_id)
        return name;
    }

  /* We have not found the device.  */
  return fallback_name ();
}

amd_dbgapi_status_t
kfd_driver_base_t::check_version () const
{
  dbgapi_assert (is_valid ());

  constexpr version_t KFD_IOCTL_VERSION_BEGIN{ 1, 13 };
  constexpr version_t KFD_IOCTL_VERSION_END{ 2, 0 };

  version_t kfd_ioctl_version = get_kfd_version ();

  if (kfd_ioctl_version < KFD_IOCTL_VERSION_BEGIN
      || kfd_ioctl_version >= KFD_IOCTL_VERSION_END)
    {
      warning ("AMD GPU driver's version %u.%u not supported "
               "(version must be >= %u.%u and < %u.%u)",
               kfd_ioctl_version.first, kfd_ioctl_version.second,
               KFD_IOCTL_VERSION_BEGIN.first, KFD_IOCTL_VERSION_BEGIN.second,
               KFD_IOCTL_VERSION_END.first, KFD_IOCTL_VERSION_END.second);
      return AMD_DBGAPI_STATUS_ERROR_RESTRICTION;
    }

  log_info ("using AMD GPU driver version %d.%d", kfd_ioctl_version.first,
            kfd_ioctl_version.second);

  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
kfd_driver_base_t::agent_snapshot (
  os_agent_info_t *snapshots, size_t snapshot_count, size_t *agent_count,
  os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (snapshots), param_in (snapshot_count),
                      param_in (agent_count), param_in (exceptions_cleared));

  dbgapi_assert (snapshots && agent_count && "must not be null");
  dbgapi_assert (snapshot_count <= std::numeric_limits<uint32_t>::max ()
                 && "invalid argument");

  if (!is_debug_enabled ())
    {
      *agent_count = 0;
      return AMD_DBGAPI_STATUS_SUCCESS;
    }

  std::vector<kfd_dbg_device_info_entry> kfd_device_infos (snapshot_count);
  if (amd_dbgapi_status_t status
      = kfd_agent_snapshot (kfd_device_infos.data (), snapshot_count,
                            agent_count, exceptions_cleared);
      status != AMD_DBGAPI_STATUS_SUCCESS)
    return status;

  for (unsigned int i = 0; i < std::min (snapshot_count, *agent_count); i++)
    {
      os_agent_info_t &agent_info = snapshots[i];
      const kfd_dbg_device_info_entry &entry = kfd_device_infos[i];

      agent_info.os_agent_id = entry.gpu_id;

      agent_info.local_address_aperture_base = entry.lds_base;
      agent_info.local_address_aperture_limit = entry.lds_limit;
      agent_info.private_address_aperture_base = entry.scratch_base;
      agent_info.private_address_aperture_limit = entry.scratch_limit;
      agent_info.location_id = entry.location_id;
      agent_info.simd_count = entry.simd_count;
      agent_info.max_waves_per_simd = entry.max_waves_per_simd;
      agent_info.vendor_id = entry.vendor_id;
      agent_info.device_id = entry.device_id;
      agent_info.revision_id = entry.revision_id;
      agent_info.subsystem_vendor_id = entry.subsystem_vendor_id;
      agent_info.subsystem_device_id = entry.subsystem_device_id;
      agent_info.fw_version = entry.fw_version;
      agent_info.gfxip = { entry.gfx_target_version / 10000,
                           (entry.gfx_target_version / 100) % 100,
                           entry.gfx_target_version % 100 };
      agent_info.debugging_supported
        = entry.capability & HSA_CAP_TRAP_DEBUG_SUPPORT;
      agent_info.address_watch_supported
        = entry.capability & HSA_CAP_WATCH_POINTS_SUPPORTED;
      agent_info.address_watch_register_count
        = 1 << ((entry.capability & HSA_CAP_WATCH_POINTS_TOTALBITS_MASK)
                >> HSA_CAP_WATCH_POINTS_TOTALBITS_SHIFT);
      agent_info.precise_memory_supported
        = (entry.capability
           & HSA_CAP_TRAP_DEBUG_PRECISE_MEMORY_OPERATIONS_SUPPORTED);
      agent_info.precise_alu_exceptions_supported
        = (entry.capability
           & HSA_CAP_TRAP_DEBUG_PRECISE_ALU_OPERATIONS_SUPPORTED);
      agent_info.firmware_supported
        = entry.capability & HSA_CAP_TRAP_DEBUG_FIRMWARE_SUPPORTED;
      agent_info.address_watch_mask_bits = utils::bit_mask (
        ((entry.debug_prop & HSA_DBG_WATCH_ADDR_MASK_LO_BIT_MASK)
         >> HSA_DBG_WATCH_ADDR_MASK_LO_BIT_SHIFT),
        ((entry.debug_prop & HSA_DBG_WATCH_ADDR_MASK_HI_BIT_MASK)
         >> HSA_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT));
      agent_info.ttmps_always_initialized
        = entry.debug_prop & HSA_DBG_DISPATCH_INFO_ALWAYS_VALID;
      agent_info.watchpoint_exclusive
        = entry.debug_prop & HSA_DBG_WATCHPOINTS_EXCLUSIVE;
      agent_info.xcc_count = entry.num_xcc;

      if (!agent_info.simd_count || !agent_info.max_waves_per_simd
          || !entry.array_count || !entry.simd_arrays_per_engine)
        fatal_error ("Invalid node properties");

      agent_info.shader_engine_count
        = (entry.array_count * entry.num_xcc) / entry.simd_arrays_per_engine;
      agent_info.name = marketing_name (
        agent_info.vendor_id, agent_info.device_id, agent_info.revision_id);
    }

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (
    make_ref (param_out (snapshots), std::min (snapshot_count, *agent_count)),
    make_ref (param_out (agent_count)));
}

amd_dbgapi_status_t
kfd_driver_base_t::queue_snapshot (
  os_queue_snapshot_entry_t *snapshots, size_t snapshot_count,
  size_t *queue_count, os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (snapshots), param_in (snapshot_count),
                      param_in (queue_count), param_in (exceptions_cleared));

  std::vector<kfd_queue_snapshot_entry> kfd_queues_entries (snapshot_count);
  if (amd_dbgapi_status_t status
      = kfd_queue_snapshot (kfd_queues_entries.data (), snapshot_count,
                            queue_count, exceptions_cleared);
      status != AMD_DBGAPI_STATUS_SUCCESS)
    return status;

  for (size_t i = 0; i < std::min (*queue_count, snapshot_count); ++i)
    {
      auto &queue_info = snapshots[i];
      const kfd_queue_snapshot_entry &entry = kfd_queues_entries[i];

      queue_info = {};

      auto [queue_id, queue_state] = decode_queue_id (entry.queue_id);
      queue_info.queue_id = queue_id;
      queue_info.state = queue_state;
      queue_info.gpu_id = entry.gpu_id;
      queue_info.queue_type = os_queue_type (entry.queue_type);
      queue_info.exception_status
        = kfd_to_os_exception_mask (entry.exception_status);
      queue_info.ring_base_address = entry.ring_base_address;
      queue_info.write_pointer_address = entry.write_pointer_address;
      queue_info.read_pointer_address = entry.read_pointer_address;
      queue_info.ctx_save_restore_address = entry.ctx_save_restore_address;
      queue_info.ctx_save_restore_area_size = entry.ctx_save_restore_area_size;
      queue_info.ring_size = entry.ring_size;
    }

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (
    make_ref (param_out (snapshots), std::min (snapshot_count, *queue_count)),
    make_ref (param_out (queue_count)));
}

/* Data structures used for core dump genertation and loading.  */

enum class amdgpu_core_note_version_t : uint64_t
{
  kfd_note = 1
};

struct kfd_note_header_t
{
  uint32_t kfd_version_major;
  uint32_t kfd_version_minor;
  uint64_t runtime_info_size;
  uint32_t agent_entry_count;
  uint32_t agent_entry_size;
  uint32_t queue_entry_count;
  uint32_t queue_entry_size;
};

class kfd_core_driver_t final : public kfd_driver_base_t
{
public:
  kfd_core_driver_t (amd_dbgapi_client_process_id_t client_process_id,
                     const amd_dbgapi_core_state_data_t &core_state);

  bool is_valid () const override { return m_state != nullptr; }

  kfd_driver_base_t::version_t get_kfd_version () const override final;

  amd_dbgapi_status_t enable_debug (os_exception_mask_t exceptions_reported,
                                    amd_dbgapi_notifier_t notifier,
                                    os_runtime_info_t *runtime_info) override;

  amd_dbgapi_status_t disable_debug () override
  {
    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  bool is_debug_enabled () const override;

  amd_dbgapi_status_t xfer_global_memory_partial (global_address_t address,
                                                  void *read,
                                                  const void *write,
                                                  size_t *size) const override;

protected:
  virtual amd_dbgapi_status_t
  kfd_agent_snapshot (kfd_dbg_device_info_entry *agents, size_t snapshot_count,
                      size_t *agent_count,
                      os_exception_mask_t exceptions_cleared) const override;

  virtual amd_dbgapi_status_t
  kfd_queue_snapshot (kfd_queue_snapshot_entry *queues, size_t snapshot_count,
                      size_t *queue_count,
                      os_exception_mask_t exceptions_cleared) const override;

private:
  /* Keep some internal state so we can simulate what KFD would have done.  */
  struct kfd_simulated_state_t
  {
    kfd_driver_base_t::version_t version;
    kfd_runtime_info runtime_info;
    std::vector<kfd_dbg_device_info_entry> agents;
    std::vector<kfd_queue_snapshot_entry> queues;
  };

  std::unique_ptr<kfd_simulated_state_t> m_state;
  amd_dbgapi_client_process_id_t const m_client_process_id;
};

kfd_driver_base_t::version_t
kfd_core_driver_t::get_kfd_version () const
{
  return m_state->version;
}

namespace
{
class note_reader
{
public:
  explicit note_reader (const amd_dbgapi_core_state_data_t &core_state)
    : head{ static_cast<const std::byte *> (core_state.data) },
      end{ head + core_state.size }
  {
  }

  template <typename T> T read ()
  {
    T val{};
    read (val);
    return val;
  }

  template <typename T> void read (T &val) { read (val, sizeof (T)); }

  /* Read up to SIZE bytes into val from the note.  */
  template <typename T> void read (T &val, size_t size)
  {
    if (!m_error && head + size <= end)
      {
        std::memcpy (&val, head, std::min (size, sizeof (T)));
        head += size;
      }
    else
      {
        /* The user tried to read more data than available.  */
        m_error = true;
      }
  }

  bool eof () const { return !m_error && head == end; }

private:
  const std::byte *head;
  const std::byte *const end;
  bool m_error{ false };
};
};

kfd_core_driver_t::kfd_core_driver_t (
  amd_dbgapi_client_process_id_t client_process_id,
  const amd_dbgapi_core_state_data_t &core_state)
  : kfd_driver_base_t (std::nullopt), m_client_process_id{ client_process_id }
{
  if (core_state.endianness
      != (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ ? AMD_DBGAPI_ENDIAN_LITTLE
                                                    : AMD_DBGAPI_ENDIAN_BIG))
    {
      warning ("Invalid corefile note endianness.");
      return;
    }

  note_reader reader{ core_state };
  [[maybe_unused]] const auto note_version
    = reader.read<amdgpu_core_note_version_t> ();
  dbgapi_assert (note_version == amdgpu_core_note_version_t::kfd_note);

  auto header = reader.read<kfd_note_header_t> ();

  auto state = std::make_unique<kfd_simulated_state_t> ();
  state->version.first = header.kfd_version_major;
  state->version.second = header.kfd_version_minor;

  if (header.runtime_info_size % 8 != 0 || header.agent_entry_size % 8 != 0
      || header.queue_entry_size % 8 != 0)
    {
      warning ("Invalid alignment in corefile note.");
      return;
    }

  state->agents.resize (header.agent_entry_count);
  state->queues.resize (header.queue_entry_count);

  reader.read (state->runtime_info, header.runtime_info_size);
  for (auto &agent : state->agents)
    reader.read (agent, header.agent_entry_size);
  for (auto &queue : state->queues)
    reader.read (queue, header.queue_entry_size);

  if (!reader.eof ())
    {
      warning ("Invalid corefile note.");
      return;
    }

  m_state = std::move (state);
}

bool
kfd_core_driver_t::is_debug_enabled () const
{
  return m_state->runtime_info.runtime_state == os_runtime_state_t::enabled;
}

amd_dbgapi_status_t
kfd_core_driver_t::xfer_global_memory_partial (global_address_t address,
                                               void *read, const void *write,
                                               size_t *size) const
{
  return detail::process_callbacks.xfer_global_memory (
    m_client_process_id, address, size, read, write);
}

amd_dbgapi_status_t
kfd_core_driver_t::kfd_agent_snapshot (
  kfd_dbg_device_info_entry *agents, size_t snapshot_count,
  size_t *agent_count, os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (agents), param_in (snapshot_count),
                      param_in (agent_count), param_in (exceptions_cleared));

  const size_t count = std::min (snapshot_count, m_state->agents.size ());
  for (size_t i = 0; i < count; ++i)
    {
      agents[i] = m_state->agents[i];
      m_state->agents[i].exception_status
        &= ~kfd_exception_mask (exceptions_cleared);
    }
  *agent_count = m_state->agents.size ();

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (
    make_ref (param_out (agents), std::min (snapshot_count, *agent_count)),
    make_ref (param_out (agent_count)));
}

amd_dbgapi_status_t
kfd_core_driver_t::kfd_queue_snapshot (
  kfd_queue_snapshot_entry *queues, size_t snapshot_count, size_t *queue_count,
  os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (queues), param_in (snapshot_count),
                      param_in (queue_count), param_in (exceptions_cleared));

  const size_t count = std::min (snapshot_count, m_state->queues.size ());
  for (size_t i = 0; i < count; i++)
    {
      queues[i] = m_state->queues[i];
      m_state->queues[i].exception_status
        &= ~kfd_exception_mask (exceptions_cleared);
    }
  *queue_count = m_state->queues.size ();

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (
    make_ref (param_out (queues), std::min (snapshot_count, *queue_count)),
    make_ref (param_out (queue_count)));
}

amd_dbgapi_status_t
kfd_core_driver_t::enable_debug ([[maybe_unused]] os_exception_mask_t exceptions_reported,
                                 [[maybe_unused]] amd_dbgapi_notifier_t notifier,
                                 os_runtime_info_t *runtime_info)
{
  TRACE_DRIVER_BEGIN (param_in (exceptions_reported), param_in (notifier),
                      param_in (runtime_info));

  runtime_info->r_debug = m_state->runtime_info.r_debug;
  runtime_info->runtime_state
    = os_runtime_state (m_state->runtime_info.runtime_state);
  runtime_info->ttmp_setup = !!m_state->runtime_info.ttmp_setup;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (runtime_info)));
}

class kfd_driver_t final : public kfd_driver_base_t
{
private:
  static size_t s_kfd_open_count;
  static std::optional<file_desc_t> s_kfd_fd;

  std::optional<file_desc_t> m_proc_mem_fd{};

  mutable size_t m_read_request_count{};
  mutable size_t m_write_request_count{};
  mutable size_t m_bytes_read{};
  mutable size_t m_bytes_written{};

  static void open_kfd ();
  static void close_kfd ();

  bool m_is_debug_enabled{ false };

  int kfd_ioctl (unsigned long request, void *args) const;
  int kfd_dbg_trap_ioctl (uint32_t action,
                          kfd_ioctl_dbg_trap_args *args) const;

public:
  kfd_driver_t (amd_dbgapi_os_process_id_t os_pid) : kfd_driver_base_t (os_pid)
  {
    open_kfd ();
    std::string filename = string_printf ("/proc/%d/mem", os_pid);
    int fd = ::open (filename.c_str (), O_RDWR | O_LARGEFILE | O_CLOEXEC, 0);
    if (fd == -1)
      {
        warning ("Could not open `%s': %s", filename.c_str (),
                 strerror (errno));
        return;
      }

    m_proc_mem_fd.emplace (fd);

    /* See is_valid() for information about how failing to open /proc/pid/mem
       or /dev/kfd is handled.  */
  }

  ~kfd_driver_t () override
  {
    if (is_debug_enabled ())
      disable_debug ();

    close_kfd ();

    log_info ("kfd_driver_t statistics (pid %d): "
              "%ld reads (%s), %ld writes (%s)",
              m_os_pid.value (), m_read_request_count,
              utils::human_readable_size (m_bytes_read).c_str (),
              m_write_request_count,
              utils::human_readable_size (m_bytes_written).c_str ());

    if (m_proc_mem_fd)
      ::close (*m_proc_mem_fd);
  }

  /* Disable copies.  */
  kfd_driver_t (const kfd_driver_t &) = delete;
  kfd_driver_t &operator= (const kfd_driver_t &) = delete;

  bool is_valid () const override
  {
    return s_kfd_fd.has_value () && m_proc_mem_fd.has_value ();
  }

  kfd_driver_base_t::version_t get_kfd_version () const override final;

  amd_dbgapi_status_t
  create_core_state_note (const os_runtime_info_t &runtime_info,
                          amd_dbgapi_core_state_data_t *data) const override;

  amd_dbgapi_status_t
  kfd_agent_snapshot (kfd_dbg_device_info_entry *agents, size_t snapshot_count,
                      size_t *agent_count,
                      os_exception_mask_t exceptions_cleared) const override;

  amd_dbgapi_status_t enable_debug (os_exception_mask_t exceptions_reported,
                                    amd_dbgapi_notifier_t notifier,
                                    os_runtime_info_t *runtime_info) override;
  amd_dbgapi_status_t disable_debug () override;
  bool is_debug_enabled () const override { return m_is_debug_enabled; }

  amd_dbgapi_status_t
  send_exceptions (os_exception_mask_t exceptions,
                   std::optional<os_agent_id_t> agent_id,
                   std::optional<os_queue_id_t> queue_id) const override;

  amd_dbgapi_status_t set_exceptions_reported (
    os_exception_mask_t exceptions_reported) const override;

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
  kfd_queue_snapshot (kfd_queue_snapshot_entry *snapshots,
                      size_t snapshot_count, size_t *queue_count,
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
    os_wave_launch_trap_override_t override,
    os_wave_launch_trap_mask_t trap_mask,
    os_wave_launch_trap_mask_t requested_bits,
    os_wave_launch_trap_mask_t *previous_mask,
    os_wave_launch_trap_mask_t *supported_mask) const override;

  amd_dbgapi_status_t
  set_process_flags (os_process_flags_t flags) const override;

  amd_dbgapi_status_t xfer_global_memory_partial (global_address_t address,
                                                  void *read,
                                                  const void *write,
                                                  size_t *size) const override;
};

size_t kfd_driver_t::s_kfd_open_count{ 0 };
std::optional<file_desc_t> kfd_driver_t::s_kfd_fd;

/* Open the KFD device. The file descriptor is reference counted, multiple
   calls to open_kfd are allowed, as long as the same number of open_kfd and
   close_kfd are called.  The last call to close_kfd closes the device.  */

void
kfd_driver_t::open_kfd ()
{
  if (!s_kfd_open_count++)
    {
      int fd = ::open ("/dev/kfd", O_RDWR | O_CLOEXEC);
      if (fd == -1)
        {
          log_info ("Could not open the KFD device: %s", strerror (errno));
          return;
        }

      dbgapi_assert (!s_kfd_fd && "kfd_fd is already open");
      s_kfd_fd.emplace (fd);
    }
}

void
kfd_driver_t::close_kfd ()
{
  dbgapi_assert (s_kfd_open_count > 0 && "kfd_fd is already closed");

  /* The last call to close_kfd closes the KFD device.  */
  if (!--s_kfd_open_count)
    {
      if (s_kfd_fd && ::close (*s_kfd_fd))
        fatal_error ("failed to close s_kfd_fd");

      s_kfd_fd.reset ();
    }
}

int
kfd_driver_t::kfd_ioctl (unsigned long request, void *args) const
{
  dbgapi_assert (is_valid ());

  int ret;
  do
    {
      /* Retry ioctl call if it's interrupted by a signal.  */
      ret = ::ioctl (*s_kfd_fd, request, args);
    }
  while (ret < 0 && errno == EINTR);

  return ret < 0 ? -errno : ret;
}

int
kfd_driver_t::kfd_dbg_trap_ioctl (uint32_t action,
                                  kfd_ioctl_dbg_trap_args *args) const
{
  dbgapi_assert (m_os_pid);

  args->pid = *m_os_pid;
  args->op = action;

  int ret = kfd_ioctl (AMDKFD_IOC_DBG_TRAP, args);
  if (ret == -ESRCH)
    {
      /* TODO: Should we tear down the process now, so that any operation
         executed after this point returns an error?  */
      return -ESRCH;
    }

  return ret;
}

kfd_driver_base_t::version_t
kfd_driver_t::get_kfd_version () const
{
  kfd_ioctl_get_version_args get_version_args{};

  if (kfd_ioctl (AMDKFD_IOC_GET_VERSION, &get_version_args))
    fatal_error ("AMDKFD_IOC_GET_VERSION failed");

  return { get_version_args.major_version, get_version_args.minor_version };
}

namespace
{

/* Helper structure for kfd_driver_t::create_core_state_note used to capture
   a serializable snapshot of entities from KFD.  */

struct kfd_snapshots
{
  template <typename T, typename F> amd_dbgapi_status_t fetch (F fetcher)
  {
    size_t n_ent = 1;
    std::vector<T> entries (n_ent);

    /* Do a first call to KFD to figure out the total number of entries
       we should capture.  */
    amd_dbgapi_status_t status
      = fetcher (entries.data (), entries.size (), &n_ent);
    if (status != AMD_DBGAPI_STATUS_SUCCESS)
      return status;

    /* Now we can fetch all the entries at once.  */
    entries.resize (n_ent);
    status = fetcher (entries.data (), entries.size (), &n_ent);
    if (status != AMD_DBGAPI_STATUS_SUCCESS)
      return status;

    dbgapi_assert (n_ent == entries.size ());

    n_entries = n_ent;
    entry_size = sizeof (T);

    /* The rest of the code assumes that a T is 64-bit aligned.  */
    static_assert (sizeof (T) % 8 == 0);
    snapshots.resize (n_entries * entry_size);
    std::copy (entries.begin (), entries.end (),
               reinterpret_cast<T *> (snapshots.data ()));

    return AMD_DBGAPI_STATUS_SUCCESS;
  }
  uint32_t n_entries;
  uint32_t entry_size;
  std::vector<std::byte> snapshots;
};

class note_builder
{
public:
  note_builder () {}

  template <typename T, std::enable_if_t<!std::is_pointer_v<T>, int> = 0>
  void write (const T &v)
  {
    const auto pre_size = m_buffer.size ();
    m_buffer.resize (m_buffer.size () + sizeof (v));
    std::memcpy (&m_buffer[pre_size], &v, sizeof (T));
  }

  void write (const std::vector<std::byte> &v)
  {
    m_buffer.reserve (m_buffer.size () + v.size ());
    m_buffer.insert (m_buffer.end (), v.begin (), v.end ());
  }

  size_t size () const { return m_buffer.size (); }

  amd_dbgapi_core_state_data_t note () const
  {
    amd_dbgapi_core_state_data_t note;
    note.endianness
      = (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ ? AMD_DBGAPI_ENDIAN_LITTLE
                                                   : AMD_DBGAPI_ENDIAN_BIG);

    note.size = size ();
    auto buffer = amd::dbgapi::allocate_memory<std::byte> (note.size);
    std::memcpy (buffer.get (), m_buffer.data (), note.size);
    note.data = buffer.release ();

    return note;
  }

private:
  std::vector<std::byte> m_buffer;
};

}; /* anonymous namespace.  */

amd_dbgapi_status_t
kfd_driver_t::create_core_state_note (const os_runtime_info_t &runtime_info,
                                      amd_dbgapi_core_state_data_t *data) const
{
  /* The following assumes that the various structs extracted from KFD
     are multiple of 64 bits in size.  */
  static_assert (sizeof (os_runtime_info_t) % 8 == 0
                   && sizeof (kfd_dbg_device_info_entry) % 8 == 0
                   && sizeof (kfd_queue_snapshot_entry) % 8 == 0,
                 "Incompatible struct size");

  dbgapi_assert (is_debug_enabled () && "debug is not enabled");

  /* In the note, store the lowest between the run-time version (reported by
   * KFD) and the build-time one (from kfd_ioctl.h).  We do not support
   * situations where the major component of the versions differ as they would
   * be incompatible.
   *
   * If the run-time version is above the build-time version, we want to store
   * the build-time version because the snapshot operations only extract the
   * fields known at build-time.
   *
   * If the build-time version is above run-time version, KFD leave fields it
   * does know about uninitialized.  Those uninitialized fields will be
   * recorded in the note but will be ignored when reading the core dump based
   * on the KFD version stored with it.  */

  version_t kfd_version = get_kfd_version ();
  if (kfd_version.first != KFD_IOCTL_MAJOR_VERSION)
    {
      warning ("Unable to encode core state from KFD version %d.x",
               kfd_version.first);
      return AMD_DBGAPI_STATUS_ERROR;
    }

  kfd_version = std::min (
    kfd_version, { KFD_IOCTL_MAJOR_VERSION, KFD_IOCTL_MINOR_VERSION });

  kfd_snapshots agent_snapshots;
  kfd_snapshots queue_snapshots;

  if (amd_dbgapi_status_t status
      = agent_snapshots.fetch<kfd_dbg_device_info_entry> (
        [this] (auto &&...args)
        {
          return this->kfd_agent_snapshot (
            std::forward<decltype (args)> (args)...,
            os_exception_mask_t::none);
        });
      status != AMD_DBGAPI_STATUS_SUCCESS)
    return status;

  if (amd_dbgapi_status_t status
      = queue_snapshots.fetch<kfd_queue_snapshot_entry> (
        [this] (auto &&...args)
        {
          return this->kfd_queue_snapshot (
            std::forward<decltype (args)> (args)...,
            os_exception_mask_t::none);
        });
      status != AMD_DBGAPI_STATUS_SUCCESS)
    return status;

  note_builder builder;

  /*  Note header.  */
  builder.write (amdgpu_core_note_version_t::kfd_note);
  kfd_note_header_t header{};
  header.kfd_version_major = kfd_version.first;
  header.kfd_version_minor = kfd_version.second;
  header.runtime_info_size = sizeof (runtime_info);
  header.agent_entry_count = agent_snapshots.n_entries;
  header.agent_entry_size = agent_snapshots.entry_size;
  header.queue_entry_count = queue_snapshots.n_entries;
  header.queue_entry_size = queue_snapshots.entry_size;
  builder.write (header);

  /* Runtime snapshot.  */
  builder.write (runtime_info);
  dbgapi_assert (builder.size () % 8 == 0 && "invalid alignment");

  /* Agents snapshot.  */
  builder.write (agent_snapshots.snapshots);
  dbgapi_assert (builder.size () % 8 == 0 && "invalid alignment");

  /* Queue snapshots.  */
  builder.write (queue_snapshots.snapshots);
  dbgapi_assert (builder.size () % 8 == 0 && "invalid alignment");

  *data = builder.note ();

  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
kfd_driver_t::kfd_agent_snapshot (kfd_dbg_device_info_entry *agents_infos,
                                  [[maybe_unused]] size_t agent_info_count,
                                  size_t *agent_count,
                                  os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (agents_infos), param_in (agent_info_count),
                      param_in (agent_count), param_in (exceptions_cleared));

  dbgapi_assert (agents_infos != nullptr && agent_count != nullptr
                 && "must not be null");
  dbgapi_assert (agent_info_count <= std::numeric_limits<uint32_t>::max ()
                 && "invalid argument");

  dbgapi_assert (is_debug_enabled () && "debug must be enabled");

  kfd_ioctl_dbg_trap_args args{};
  args.device_snapshot.exception_mask
    = kfd_exception_mask (exceptions_cleared);
  args.device_snapshot.snapshot_buf_ptr
    = reinterpret_cast<uint64_t> (agents_infos);
  args.device_snapshot.num_devices = static_cast<uint32_t> (*agent_count);
  args.device_snapshot.entry_size
    = static_cast<uint32_t> (sizeof (kfd_dbg_device_info_entry));

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (args.device_snapshot.entry_size
             != sizeof (kfd_dbg_device_info_entry)
           || err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  *agent_count = args.device_snapshot.num_devices;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (agents_infos),
                              std::min (agent_info_count, *agent_count)),
                    make_ref (param_out (agent_count)));
}

amd_dbgapi_status_t
kfd_driver_t::enable_debug (os_exception_mask_t exceptions_reported,
                            amd_dbgapi_notifier_t notifier,
                            os_runtime_info_t *runtime_info)
{
  TRACE_DRIVER_BEGIN (param_in (exceptions_reported), param_in (notifier),
                      param_in (runtime_info));

  dbgapi_assert (!is_debug_enabled () && "debug is already enabled");

  kfd_runtime_info os_runtime_info;

  kfd_ioctl_dbg_trap_args args{};
  args.enable.exception_mask = kfd_exception_mask (exceptions_reported);
  args.enable.rinfo_ptr = reinterpret_cast<uintptr_t> (&os_runtime_info);
  args.enable.rinfo_size = sizeof (kfd_runtime_info);
  args.enable.dbg_fd = notifier;

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_ENABLE, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err == -EALREADY)
    return AMD_DBGAPI_STATUS_ERROR_RESTRICTION;
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  if (sizeof (*runtime_info) > args.enable.rinfo_size)
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;

  m_is_debug_enabled = true;

  runtime_info->r_debug = os_runtime_info.r_debug;
  runtime_info->runtime_state
    = os_runtime_state (os_runtime_info.runtime_state);
  runtime_info->ttmp_setup = !!os_runtime_info.ttmp_setup;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (runtime_info)));
}

amd_dbgapi_status_t
kfd_driver_t::disable_debug ()
{
  TRACE_DRIVER_BEGIN ();

  if (!is_debug_enabled ())
    return AMD_DBGAPI_STATUS_SUCCESS;

  kfd_ioctl_dbg_trap_args args{};

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_DISABLE, &args);
  if (err == -ESRCH)
    {
      m_is_debug_enabled = false;
      return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
    }
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  m_is_debug_enabled = false;
  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kfd_driver_t::set_exceptions_reported (
  os_exception_mask_t exceptions_reported) const
{
  TRACE_DRIVER_BEGIN (exceptions_reported);

  kfd_ioctl_dbg_trap_args args{};
  args.set_exceptions_enabled.exception_mask
    = kfd_exception_mask (exceptions_reported);

  int err
    = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kfd_driver_t::send_exceptions (os_exception_mask_t exceptions,
                               std::optional<os_agent_id_t> agent_id,
                               std::optional<os_queue_id_t> queue_id) const
{
  TRACE_DRIVER_BEGIN (param_in (exceptions), param_in (agent_id),
                      param_in (queue_id));

  dbgapi_assert (is_debug_enabled () && "debug is not enabled");

  kfd_ioctl_dbg_trap_args args{};
  args.send_runtime_event.exception_mask = kfd_exception_mask (exceptions);
  args.send_runtime_event.gpu_id = agent_id.has_value () ? *agent_id : -1;
  args.send_runtime_event.queue_id = queue_id.has_value () ? *queue_id : -1;

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kfd_driver_t::query_debug_event (os_exception_mask_t *exceptions_present,
                                 os_queue_id_t *os_queue_id,
                                 os_agent_id_t *os_agent_id,
                                 os_exception_mask_t exceptions_cleared) const
{
  TRACE_DRIVER_BEGIN (param_in (exceptions_present), param_in (os_queue_id),
                      param_in (os_agent_id), param_in (exceptions_cleared));

  dbgapi_assert (exceptions_present && os_queue_id && os_agent_id
                 && "must not be null");

  if (!is_debug_enabled ())
    {
      *exceptions_present = os_exception_mask_t::none;
      *os_queue_id = *os_agent_id = 0;
      return AMD_DBGAPI_STATUS_SUCCESS;
    }

  kfd_ioctl_dbg_trap_args args{};
  args.query_debug_event.exception_mask
    = kfd_exception_mask (exceptions_cleared);

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err == -EAGAIN)
    {
      /* There are no more events.  */
      *exceptions_present = os_exception_mask_t::none;
      *os_queue_id = *os_agent_id = 0;
      return AMD_DBGAPI_STATUS_SUCCESS;
    }
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  *exceptions_present
    = kfd_to_os_exception_mask (args.query_debug_event.exception_mask);
  *os_queue_id = args.query_debug_event.queue_id;
  *os_agent_id = args.query_debug_event.gpu_id;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (exceptions_present)),
                    make_ref (param_out (os_queue_id)),
                    make_ref (param_out (os_agent_id)));
}

amd_dbgapi_status_t
kfd_driver_t::query_exception_info (os_exception_code_t exception,
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
  kfd_runtime_info runtime_info;

  kfd_ioctl_dbg_trap_args args{};
  args.query_exception_info.info_ptr
    = reinterpret_cast<uintptr_t> (&runtime_info);
  args.query_exception_info.info_size = sizeof (kfd_runtime_info);
  args.query_exception_info.source_id = os_source_id.raw;
  args.query_exception_info.exception_code = kfd_exception_code (exception);
  args.query_exception_info.clear_exception = clear_exception ? 1 : 0;

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  if (sizeof (kfd_runtime_info) > args.query_exception_info.info_size)
    return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY;

  os_exception_info->runtime_info.r_debug = runtime_info.r_debug;
  os_exception_info->runtime_info.runtime_state
    = os_runtime_state (runtime_info.runtime_state);
  os_exception_info->runtime_info.ttmp_setup = !!runtime_info.ttmp_setup;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_query_ref (exception, param_out (os_exception_info)));
}

amd_dbgapi_status_t
kfd_driver_t::suspend_queues (const os_queue_id_t *queues, size_t queue_count,
                              os_exception_mask_t exceptions_cleared,
                              size_t *suspended_count,
                              os_queue_state_t *queue_states) const
{
  TRACE_DRIVER_BEGIN (make_ref (param_in (queues), queue_count),
                      param_in (queue_count), param_in (exceptions_cleared),
                      param_in (suspended_count), param_in (queue_states));

  dbgapi_assert (suspended_count != nullptr);
  dbgapi_assert (queue_count <= std::numeric_limits<uint32_t>::max ());

  auto kfd_queue_ids = std::make_unique<__u32[]> (queue_count);
  std::copy (queues, queues + queue_count, kfd_queue_ids.get ());

  kfd_ioctl_dbg_trap_args args{};
  args.suspend_queues.exception_mask = kfd_exception_mask (exceptions_cleared);
  args.suspend_queues.queue_array_ptr
    = reinterpret_cast<uint64_t> (kfd_queue_ids.get ());
  args.suspend_queues.num_queues = static_cast<uint32_t> (queue_count);
  args.suspend_queues.grace_period = 0;

  int ret = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_SUSPEND_QUEUES, &args);
  if (ret == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (ret < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  *suspended_count = ret;
  for (size_t i = 0; i < queue_count; i++)
    {
      auto [queue_id, queue_state] = decode_queue_id (kfd_queue_ids[i]);

      if (queue_id != queues[i])
        fatal_error ("Unexpected queue ID, %s != %d",
                     to_string (queues[i]).c_str (), queue_id);

      queue_states[i] = queue_state;
    }

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (queue_states), queue_count),
                    make_ref (param_out (suspended_count)));
}

amd_dbgapi_status_t
kfd_driver_t::resume_queues (const os_queue_id_t *queues, size_t queue_count,
                             size_t *resumed_count,
                             os_queue_state_t *queue_states) const
{
  TRACE_DRIVER_BEGIN (make_ref (param_in (queues), queue_count),
                      param_in (queue_count), param_in (resumed_count),
                      param_in (queue_states));

  dbgapi_assert (resumed_count != nullptr);
  dbgapi_assert (queue_count <= std::numeric_limits<uint32_t>::max ());

  auto kfd_queue_ids = std::make_unique<__u32[]> (queue_count);
  std::copy (queues, queues + queue_count, kfd_queue_ids.get ());

  kfd_ioctl_dbg_trap_args args{};
  args.resume_queues.queue_array_ptr
    = reinterpret_cast<uint64_t> (kfd_queue_ids.get ());
  args.resume_queues.num_queues = static_cast<uint32_t> (queue_count);

  int ret = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_RESUME_QUEUES, &args);
  if (ret == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (ret < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  *resumed_count = ret;
  for (size_t i = 0; i < queue_count; i++)
    {
      auto [queue_id, queue_state] = decode_queue_id (kfd_queue_ids[i]);

      if (queue_id != queues[i])
        fatal_error ("Unexpected queue ID, %s != %d",
                     to_string (queues[i]).c_str (), queue_id);

      queue_states[i] = queue_state;
    }

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (queue_states), queue_count),
                    make_ref (param_out (resumed_count)));
}

amd_dbgapi_status_t
kfd_driver_t::kfd_queue_snapshot (kfd_queue_snapshot_entry *snapshots,
                                  size_t snapshot_count, size_t *queue_count,
                                  os_exception_mask_t exceptions_cleared) const
{
  dbgapi_assert (snapshots != nullptr && queue_count != nullptr
                 && "must not be null");
  dbgapi_assert (snapshot_count <= std::numeric_limits<uint32_t>::max ()
                 && "invalid argument");

  if (!is_debug_enabled ())
    {
      *queue_count = 0;
      return AMD_DBGAPI_STATUS_SUCCESS;
    }

  kfd_ioctl_dbg_trap_args args{};
  args.queue_snapshot.exception_mask = kfd_exception_mask (exceptions_cleared);
  args.queue_snapshot.snapshot_buf_ptr
    = reinterpret_cast<uint64_t> (snapshots);
  args.queue_snapshot.num_queues = static_cast<uint32_t> (snapshot_count);
  args.queue_snapshot.entry_size
    = static_cast<uint32_t> (sizeof (kfd_queue_snapshot_entry));

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (args.queue_snapshot.entry_size != sizeof (kfd_queue_snapshot_entry)
           || err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  /* KFD writes up to snapshot_count queue snapshots, but returns the number of
     queues in the process so that we can check if we have allocated enough
     memory to hold all the snapshots.  */
  *queue_count = args.queue_snapshot.num_queues;

  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
kfd_driver_t::set_address_watch (os_agent_id_t os_agent_id,
                                 agent_address_t address, agent_address_t mask,
                                 os_watch_mode_t os_watch_mode,
                                 os_watch_id_t *os_watch_id) const
{
  TRACE_DRIVER_BEGIN (param_in (address), param_in (mask),
                      param_in (os_watch_mode), param_in (os_watch_id));

  dbgapi_assert (os_watch_id && "must not be null");

  kfd_ioctl_dbg_trap_args args{};
  args.set_node_address_watch.address = address;
  args.set_node_address_watch.mode = kfd_watch_mode (os_watch_mode);
  args.set_node_address_watch.mask = mask;
  args.set_node_address_watch.gpu_id = os_agent_id;

  int err
    = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH, &args);
  if (err == -ENOMEM)
    return AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE;
  else if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err < 0)
    fatal_error ("failed to set address watch: %s", strerror (err));

  *os_watch_id = args.set_node_address_watch.id;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (os_watch_id)));
}

amd_dbgapi_status_t
kfd_driver_t::clear_address_watch (os_agent_id_t os_agent_id,
                                   os_watch_id_t os_watch_id) const
{
  TRACE_DRIVER_BEGIN (param_in (os_watch_id));

  kfd_ioctl_dbg_trap_args args{};
  args.clear_node_address_watch.gpu_id = os_agent_id;
  args.clear_node_address_watch.id = os_watch_id;

  int err
    = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kfd_driver_t::set_wave_launch_mode (os_wave_launch_mode_t mode) const
{
  TRACE_DRIVER_BEGIN (param_in (mode));

  kfd_ioctl_dbg_trap_args args{};
  args.launch_mode.launch_mode = kfd_wave_launch_mode (mode);

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kfd_driver_t::set_wave_launch_trap_override (
  os_wave_launch_trap_override_t override, os_wave_launch_trap_mask_t value,
  os_wave_launch_trap_mask_t mask, os_wave_launch_trap_mask_t *previous_value,
  os_wave_launch_trap_mask_t *supported_mask) const
{
  TRACE_DRIVER_BEGIN (param_in (override), param_in (value), param_in (mask),
                      param_in (previous_value), param_in (supported_mask));

  kfd_ioctl_dbg_trap_args args{};
  args.launch_override.override_mode
    = kfd_wave_launch_trap_override (override);
  args.launch_override.enable_mask = kfd_wave_launch_trap_mask (value);
  args.launch_override.support_request_mask = kfd_wave_launch_trap_mask (mask);

  int err
    = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err == -EPERM || err == -EACCES)
    return AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED;
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  if (previous_value != nullptr)
    *previous_value
      = os_wave_launch_trap_mask (args.launch_override.enable_mask);
  if (supported_mask != nullptr)
    *supported_mask
      = os_wave_launch_trap_mask (args.launch_override.support_request_mask);

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END (make_ref (param_out (previous_value)),
                    make_ref (param_out (supported_mask)));
}

amd_dbgapi_status_t
kfd_driver_t::set_process_flags (os_process_flags_t flags) const
{
  TRACE_DRIVER_BEGIN (param_in (flags));

  kfd_ioctl_dbg_trap_args args{};

  if (!!(flags & os_process_flags_t::precise_memory))
    args.set_flags.flags |= KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP;
  if (!!(flags & os_process_flags_t::precise_alu_exceptions))
    args.set_flags.flags |= KFD_DBG_TRAP_FLAG_SINGLE_ALU_OP;

  int err = kfd_dbg_trap_ioctl (KFD_IOC_DBG_TRAP_SET_FLAGS, &args);
  if (err == -ESRCH)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (err < 0)
    return AMD_DBGAPI_STATUS_ERROR;

  return AMD_DBGAPI_STATUS_SUCCESS;

  TRACE_DRIVER_END ();
}

amd_dbgapi_status_t
kfd_driver_t::xfer_global_memory_partial (global_address_t address, void *read,
                                          const void *write,
                                          size_t *size) const
{
  dbgapi_assert (!read != !write && "either read or write buffer");
  dbgapi_assert (is_valid ());

  ++(read != nullptr ? m_read_request_count : m_write_request_count);

  ssize_t ret = read != nullptr
                  ? pread (*m_proc_mem_fd, read, *size, address)
                  : pwrite (*m_proc_mem_fd, write, *size, address);

  if (ret < 0 && errno != EIO && errno != EINVAL)
    warning ("kfd_driver_t::xfer_memory failed: %s", strerror (errno));

  if (ret == 0 && *size != 0)
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  else if (ret < 0)
    return AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS;

  (read != nullptr ? m_bytes_read : m_bytes_written) += ret;

  *size = ret;
  return AMD_DBGAPI_STATUS_SUCCESS;
}

std::unique_ptr<os_driver_t>
os_driver_t::create_driver (std::optional<amd_dbgapi_os_process_id_t> os_pid)
{
  if (!os_pid)
    return std::make_unique<null_driver_t> ();

  std::unique_ptr<os_driver_t> os_driver{ new kfd_driver_t (*os_pid) };
  if (os_driver->is_valid ())
    return os_driver;

  /* If we failed to create a kfd_driver_t (kfd is not installed?), then revert
     to a plain null driver.  */
  return std::make_unique<null_driver_t> (*os_pid);
}

std::unique_ptr<os_driver_t>
os_driver_t::create_driver (amd_dbgapi_client_process_id_t client_process_id,
                            const amd_dbgapi_core_state_data_t &core_state)
{
  std::unique_ptr<os_driver_t> os_driver;

  /* We need to read the first 8 bytes of the note to figure out which
     backend should be used.  */
  note_reader reader{ core_state };
  auto note_version = reader.read<amdgpu_core_note_version_t> ();
  switch (note_version)
    {
    case amdgpu_core_note_version_t::kfd_note:
      os_driver
        = std::make_unique<kfd_core_driver_t> (client_process_id, core_state);
      break;
    default:
      warning ("Cannot open core state version %" PRIu64,
               static_cast<std::underlying_type_t<decltype (note_version)>> (
                 note_version));
    }

  if (os_driver != nullptr && os_driver->is_valid ())
    return os_driver;

  /* Fallback to the null_driver if none of the above was appropriate or could
     be initialized successfully.  */
  return std::make_unique<null_driver_t> (std::nullopt);
}

template <>
std::string
to_string (kfd_queue_snapshot_entry entry)
{
  return string_printf (
    "{ .exception_status=%#llx, .ring_base_address=%#llx, "
    ".write_pointer_address=%#llx, .read_pointer_address=%#llx, "
    ".ctx_save_restore_address=%#llx, .queue_id=%d, "
    ".gpu_id=%d, .ring_size=%#x, .queue_type=%d, "
    ".ctx_save_restore_area_size=%#x}",
    entry.exception_status, entry.ring_base_address,
    entry.write_pointer_address, entry.read_pointer_address,
    entry.ctx_save_restore_address, entry.queue_id, entry.gpu_id,
    entry.ring_size, entry.queue_type, entry.ctx_save_restore_area_size);
}

template <>
std::string
to_string (kfd_dbg_device_info_entry entry)
{
  return string_printf (
    "{ .exception_status=%#llx, .lds_base=%#llx, .lds_limit=%#llx, "
    ".scratch_base=%#llx, .scratch_limit=%#llx, .gpuvm_base=%#llx, "
    ".gpuvm_limit=%#llx, .gpu_id=%d, .location_id=%#x, .vendor_id=%#x, "
    ".device_id=%#x, .fw_version=%d, .gfx_target_version=%#x, "
    ".simd_count=%d, .max_waves_per_simd=%d, .array_count=%d, "
    ".simd_arrays_per_engine=%d, .capability=%#x, .debug_prop=%#x }",
    entry.exception_status, entry.lds_base, entry.lds_limit,
    entry.scratch_base, entry.scratch_limit, entry.gpuvm_base,
    entry.gpuvm_limit, entry.gpu_id, entry.location_id, entry.vendor_id,
    entry.device_id, entry.fw_version, entry.gfx_target_version,
    entry.simd_count, entry.max_waves_per_simd, entry.array_count,
    entry.simd_arrays_per_engine, entry.capability, entry.debug_prop);
}

} /* namespace amd::dbgapi */
