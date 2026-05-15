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

#ifndef AMD_DBGAPI_OS_DRIVER_H
#define AMD_DBGAPI_OS_DRIVER_H 1

#include "amd-dbgapi.h"
#include "logging.h"
#include "memory.h"
#include "runtime_rdebug.h"
#include "utils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace amd::dbgapi
{

/* See https://llvm.org/docs/AMDGPUUsage.html#amdgpu-ef-amdgpu-mach-table  */
enum elf_amdgpu_machine_t : uint32_t
{
  EF_AMDGPU_MACH_NONE = 0x000,
  EF_AMDGPU_MACH_AMDGCN_GFX900 = 0x02c,
  EF_AMDGPU_MACH_AMDGCN_GFX902 = 0x02d,
  EF_AMDGPU_MACH_AMDGCN_GFX904 = 0x02e,
  EF_AMDGPU_MACH_AMDGCN_GFX906 = 0x02f,
  EF_AMDGPU_MACH_AMDGCN_GFX908 = 0x030,
  EF_AMDGPU_MACH_AMDGCN_GFX90A = 0x03f,
  EF_AMDGPU_MACH_AMDGCN_GFX942 = 0x04c,
  EF_AMDGPU_MACH_AMDGCN_GFX950 = 0x04f,
  EF_AMDGPU_MACH_AMDGCN_GFX1010 = 0x033,
  EF_AMDGPU_MACH_AMDGCN_GFX1011 = 0x034,
  EF_AMDGPU_MACH_AMDGCN_GFX1012 = 0x035,
  EF_AMDGPU_MACH_AMDGCN_GFX1030 = 0x036,
  EF_AMDGPU_MACH_AMDGCN_GFX1031 = 0x037,
  EF_AMDGPU_MACH_AMDGCN_GFX1032 = 0x038,
  EF_AMDGPU_MACH_AMDGCN_GFX1100 = 0x041,
  EF_AMDGPU_MACH_AMDGCN_GFX1101 = 0x046,
  EF_AMDGPU_MACH_AMDGCN_GFX1102 = 0x047,
  EF_AMDGPU_MACH_AMDGCN_GFX1150 = 0x043,
  EF_AMDGPU_MACH_AMDGCN_GFX1151 = 0x04a,
  EF_AMDGPU_MACH_AMDGCN_GFX1152 = 0x055,
  EF_AMDGPU_MACH_AMDGCN_GFX1153 = 0x058,
  EF_AMDGPU_MACH_AMDGCN_GFX1200 = 0x048,
  EF_AMDGPU_MACH_AMDGCN_GFX1201 = 0x04e,
  EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC = 0x051,
  EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC = 0x052,
  EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC = 0x053,
  EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC = 0x054,
  EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC = 0x059,
  EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC = 0x05f
};

using os_agent_id_t = uint32_t;

struct os_agent_info_t
{
  /* Agent Id assigned by the driver.  */
  os_agent_id_t os_agent_id{};
  /* Public name of the "device".  */
  std::string name{};
  /* The graphics IP version {major, minor, stepping}.  */
  std::array<unsigned, 3> gfxip{};
  /* PCI domain the agent is in.  */
  uint16_t domain{ 0 };
  /* BDF - Identifies the device location in the PCI domain.  */
  uint16_t location_id{ 0 };
  /* Number of FCompute cores.  */
  size_t simd_count{ 0 };
  /* Maximum number of launched waves per SIMD.  */
  size_t max_waves_per_simd{ 0 };
  /* Number of shader engines.  */
  size_t shader_engine_count{ 0 };
  /* Number of Compute Accelerator Chips.  */
  uint32_t xcc_count{ 0 };
  /* PCI vendor id.  */
  uint32_t vendor_id{ 0 };
  /* PCI device id.  */
  uint32_t device_id{ 0 };
  /* PCI revision id.  */
  uint32_t revision_id{ 0 };
  /* PCI subsystem vendor id.  */
  uint32_t subsystem_vendor_id{ 0 };
  /* PCI subsystem device id.  */
  uint32_t subsystem_device_id{ 0 };
  /* ucode version.  */
  uint32_t fw_version{ 0 };
  /* local/shared address aperture base.  */
  agent_address_t local_address_aperture_base{ 0 };
  /* local/shared address aperture limit.  */
  agent_address_t local_address_aperture_limit{ 0 };
  /* private/scratch address aperture base.  */
  agent_address_t private_address_aperture_base{ 0 };
  /* private/scratch address aperture limit.  */
  agent_address_t private_address_aperture_limit{ 0 };
  /* indicates if this agent's debugging capabilities are sufficient.  */
  bool debugging_supported{ false };
  /* indicates if address watch is supported.  */
  bool address_watch_supported{ false };
  /* number of address watch registers.  */
  size_t address_watch_register_count{ 0 };
  /* bits that can be programmed in the address watch mask.  */
  uint64_t address_watch_mask_bits{ 0 };
  /* whether the address watch registers are shared between processes.  */
  bool watchpoint_exclusive{ false };
  /* indicates if precise memory operations reporting is supported.  */
  bool precise_memory_supported{ false };
  /* indicates if precise ALU operations reporting is supported.  */
  bool precise_alu_exceptions_supported{ false };
  /* indicates that the command process firmware is supported.  */
  bool firmware_supported{ false };
  /* indicates that the trap temporaries are always set up.  */
  bool ttmps_always_initialized{ false };
};

using os_queue_id_t = uint32_t;
using os_watch_id_t = uint32_t;

enum class os_queue_state_t : uint8_t
{
  error = 1 << 0,
  invalid = 1 << 1
};
template <> struct is_flag<os_queue_state_t> : std::true_type
{
};

enum class os_queue_type_t : uint32_t
{
  unknown,
  compute,
  sdma,
  compute_aql,
  sdma_xgmi,
};

union os_source_id_t
{
  os_agent_id_t agent;
  os_queue_id_t queue;
  uint32_t raw;
};
static_assert (sizeof (os_source_id_t) == sizeof (uint32_t));

enum class os_exception_code_t : uint32_t
{
  none = 0,

  /* per queue exceptions  */
  queue_wave_abort = 1,
  queue_wave_trap = 2,
  queue_wave_math_error = 3,
  queue_wave_illegal_instruction = 4,
  queue_wave_memory_violation = 5,
  queue_wave_address_error = 6,
  queue_packet_dispatch_dim_invalid = 16,
  queue_packet_dispatch_group_segment_size_invalid = 17,
  queue_packet_dispatch_code_invalid = 18,
  queue_packet_unsupported = 20,
  queue_packet_dispatch_work_group_size_invalid = 21,
  queue_packet_dispatch_register_invalid = 22,
  queue_packet_vendor_unsupported = 23,
  queue_preemption_error = 30,
  queue_new = 31,

  /* per device exceptions  */
  device_queue_delete = 32,
  device_memory_violation = 33,
  device_ras_error = 34,
  device_fatal_halt = 35,
  device_new = 36,

  /* per process exceptions  */
  process_runtime = 48,
  process_device_remove = 49,
};

enum class os_runtime_state_t : uint32_t
{
  disabled = 0,
  enabled = 1,
  enabled_busy = 2,
  enabled_error = 3,
};

struct os_runtime_info_t
{
  uintptr_t r_debug;
  os_runtime_state_t runtime_state;
  bool ttmp_setup;
};

constexpr bool
operator== (std::underlying_type_t<os_runtime_state_t> lhs,
            os_runtime_state_t rhs)
{
  return lhs == static_cast<std::underlying_type_t<decltype (rhs)>> (rhs);
}
constexpr bool
operator== (os_runtime_state_t lhs,
            std::underlying_type_t<os_runtime_state_t> rhs)
{
  return rhs == lhs;
}

constexpr bool
operator!= (std::underlying_type_t<os_runtime_state_t> lhs,
            os_runtime_state_t rhs)
{
  return !(lhs == rhs);
}
constexpr bool
operator!= (os_runtime_state_t lhs,
            std::underlying_type_t<os_runtime_state_t> rhs)
{
  return rhs != lhs;
}

constexpr auto
ec_mask (os_exception_code_t c)
{
  using underlying_t = typename std::underlying_type_t<os_exception_code_t>;
  return 1ull << (static_cast<underlying_t> (c) - 1);
}

enum class os_exception_mask_t : uint64_t
{
  none = 0,

  /* per queue exceptions  */
  queue_wave_abort = ec_mask (os_exception_code_t::queue_wave_abort),
  queue_wave_trap = ec_mask (os_exception_code_t::queue_wave_trap),
  queue_wave_math_error = ec_mask (os_exception_code_t::queue_wave_math_error),
  queue_wave_illegal_instruction
  = ec_mask (os_exception_code_t::queue_wave_illegal_instruction),
  queue_wave_memory_violation
  = ec_mask (os_exception_code_t::queue_wave_memory_violation),
  queue_wave_address_error
  = ec_mask (os_exception_code_t::queue_wave_address_error),
  queue_packet_dispatch_dim_invalid
  = ec_mask (os_exception_code_t::queue_packet_dispatch_dim_invalid),
  queue_packet_dispatch_group_segment_size_invalid = ec_mask (
    os_exception_code_t::queue_packet_dispatch_group_segment_size_invalid),
  queue_packet_dispatch_code_invalid
  = ec_mask (os_exception_code_t::queue_packet_dispatch_code_invalid),
  queue_packet_unsupported
  = ec_mask (os_exception_code_t::queue_packet_unsupported),
  queue_packet_dispatch_work_group_size_invalid = ec_mask (
    os_exception_code_t::queue_packet_dispatch_work_group_size_invalid),
  queue_packet_dispatch_register_invalid
  = ec_mask (os_exception_code_t::queue_packet_dispatch_register_invalid),
  queue_packet_vendor_unsupported
  = ec_mask (os_exception_code_t::queue_packet_vendor_unsupported),
  queue_preemption_error
  = ec_mask (os_exception_code_t::queue_preemption_error),
  queue_new = ec_mask (os_exception_code_t::queue_new),

  /* per device exceptions  */
  device_queue_delete = ec_mask (os_exception_code_t::device_queue_delete),
  device_memory_violation
  = ec_mask (os_exception_code_t::device_memory_violation),
  device_ras_error = ec_mask (os_exception_code_t::device_ras_error),
  device_fatal_halt = ec_mask (os_exception_code_t::device_fatal_halt),
  device_new = ec_mask (os_exception_code_t::device_new),

  /* per process exceptions  */
  process_runtime = ec_mask (os_exception_code_t::process_runtime),
  process_device_remove = ec_mask (os_exception_code_t::process_device_remove),
};

/* Helper function to convert an exception mask with only 1 bit set to an
   exception code.  This is written as templated code as driver backends
   might use this helper as well.  */

template <typename Code, typename Mask>
static inline constexpr Code
excp_mask_to_excp_code (Mask m)
{
  using scalar_mask_t =
    typename std::conditional_t<std::is_enum_v<Mask>,
                                std::underlying_type<Mask>,
                                utils::type_identity<Mask>>::type;

  const auto imask{ static_cast<scalar_mask_t> (m) };
  dbgapi_assert (utils::is_power_of_two (imask));

  return static_cast<Code> (utils::trailing_zeroes_count (imask) + 1);
}

template <> struct is_flag<os_exception_mask_t> : std::true_type
{
};

static constexpr os_exception_mask_t os_queue_exception_mask
  = os_exception_mask_t::queue_wave_abort
    | os_exception_mask_t::queue_wave_trap
    | os_exception_mask_t::queue_wave_math_error
    | os_exception_mask_t::queue_wave_illegal_instruction
    | os_exception_mask_t::queue_wave_memory_violation
    | os_exception_mask_t::queue_wave_address_error
    | os_exception_mask_t::queue_packet_dispatch_dim_invalid
    | os_exception_mask_t::queue_packet_dispatch_group_segment_size_invalid
    | os_exception_mask_t::queue_packet_dispatch_code_invalid
    | os_exception_mask_t::queue_packet_unsupported
    | os_exception_mask_t::queue_packet_dispatch_work_group_size_invalid
    | os_exception_mask_t::queue_packet_dispatch_register_invalid
    | os_exception_mask_t::queue_packet_vendor_unsupported
    | os_exception_mask_t::queue_preemption_error
    | os_exception_mask_t::queue_new;

static constexpr os_exception_mask_t os_agent_exception_mask
  = os_exception_mask_t::device_queue_delete
    | os_exception_mask_t::device_ras_error
    | os_exception_mask_t::device_fatal_halt
    | os_exception_mask_t::device_memory_violation
    | os_exception_mask_t::device_new;

static constexpr os_exception_mask_t os_process_exception_mask
  = os_exception_mask_t::process_runtime
    | os_exception_mask_t::process_device_remove;

constexpr os_exception_mask_t
os_exception_mask (os_exception_code_t os_exception_code)
{
  if (os_exception_code == os_exception_code_t::none)
    return os_exception_mask_t::none;

  return os_exception_mask_t{ ec_mask (os_exception_code) };
}

/* Helper function to build an os_exception_mask_t from a list of
   os_exception_code_t.  */
template <typename... ExceptionCodes>
constexpr std::enable_if_t<std::conjunction_v<std::is_convertible<
                             ExceptionCodes, os_exception_code_t>...>,
                           os_exception_mask_t>
os_exception_mask (ExceptionCodes... exception_codes)
{
  return (os_exception_mask (std::forward<ExceptionCodes> (exception_codes))
          | ...);
}

enum class os_wave_launch_trap_override_t : uint32_t
{
  apply = 0,   /* OR mask with HSA_DBG_TRAP_MASK.  */
  replace = 1, /* Replace mask.  */
};

enum class os_wave_launch_trap_mask_t : uint32_t
{
  none = 0,
  fp_invalid = 1 << 0,
  fp_input_denormal = 1 << 1,
  fp_divide_by_zero = 1 << 2,
  fp_overflow = 1 << 3,
  fp_underflow = 1 << 4,
  fp_inexact = 1 << 5,
  int_divide_by_zero = 1 << 6,
  address_watch = 1 << 7,
  wave_start = 1 << 30,
  wave_end = 1u << 31
};
template <> struct is_flag<os_wave_launch_trap_mask_t> : std::true_type
{
};

template <> std::string to_string (os_wave_launch_trap_mask_t value);

enum class os_watch_mode_t : uint32_t
{
  read = 0,    /* Read operations only.  */
  nonread = 1, /* Write or atomic operations only.  */
  atomic = 2,  /* Atomic operations only.  */
  all = 3,     /* Read, write or atomic operations.  */
};

struct os_queue_snapshot_entry_t
{
  os_queue_id_t queue_id;
  os_queue_state_t state{};
  os_agent_id_t gpu_id;
  os_queue_type_t queue_type{ os_queue_type_t::unknown };
  os_exception_mask_t exception_status;
  host_address_t ring_base_address;
  amd_dbgapi_size_t ring_size;
  host_address_t write_pointer_address;
  host_address_t read_pointer_address;
  agent_address_t ctx_save_restore_address;
  amd_dbgapi_size_t ctx_save_restore_area_size;
};

enum class os_wave_launch_mode_t : uint32_t
{
  normal = 0,      /* Waves launch normally.  */
  halt = 1,        /* Waves launch in halted mode.  */
  kill = 2,        /* Waves terminate before executing any instructions.  */
  single_step = 3, /* Waves launch in single-step mode.  */
  disable = 4,     /* Disable launching any new waves.  */
};

enum class os_process_flags_t : uint32_t
{
  precise_memory = 1 << 0,
  precise_alu_exceptions = 1 << 1
};
template <> struct is_flag<os_process_flags_t> : std::true_type
{
};

union os_exception_info_t
{
  os_runtime_info_t runtime_info;
};

class os_driver_t
{
protected:
  std::optional<amd_dbgapi_os_process_id_t> const m_os_pid;

  os_driver_t (std::optional<amd_dbgapi_os_process_id_t> os_pid)
    : m_os_pid (os_pid)
  {
  }

public:
  virtual ~os_driver_t () = default;

  /* Disable copies.  */
  os_driver_t (const os_driver_t &) = delete;
  os_driver_t &operator= (const os_driver_t &) = delete;

  static std::unique_ptr<os_driver_t>
  create_driver (std::optional<amd_dbgapi_os_process_id_t> os_pid);

  static std::unique_ptr<os_driver_t>
  create_driver (amd_dbgapi_client_process_id_t client_process_id,
                 const amd_dbgapi_core_state_data_t &core_state);

  virtual bool is_valid () const = 0;

  virtual amd_dbgapi_status_t check_version () const = 0;

  virtual amd_dbgapi_status_t
  create_core_state_note (const os_runtime_info_t &runtime_info,
                          amd_dbgapi_core_state_data_t *data) const
    = 0;

  virtual amd_dbgapi_status_t
  agent_snapshot (os_agent_info_t *snapshots, size_t snapshot_count,
                  size_t *agent_count,
                  os_exception_mask_t exceptions_cleared) const
    = 0;

  virtual amd_dbgapi_status_t
  enable_debug (os_exception_mask_t exceptions_reported,
                amd_dbgapi_notifier_t notifier,
                os_runtime_info_t *runtime_info)
    = 0;
  virtual amd_dbgapi_status_t disable_debug () = 0;
  virtual bool is_debug_enabled () const = 0;

  virtual amd_dbgapi_status_t
  set_exceptions_reported (os_exception_mask_t exceptions_reported) const
    = 0;

  virtual amd_dbgapi_status_t
  send_exceptions (os_exception_mask_t exceptions,
                   std::optional<os_agent_id_t> agent_id,
                   std::optional<os_queue_id_t> queue_id) const
    = 0;

  virtual amd_dbgapi_status_t
  query_debug_event (os_exception_mask_t *exceptions_present,
                     os_queue_id_t *os_queue_id, os_agent_id_t *os_agent_id,
                     os_exception_mask_t exceptions_cleared) const
    = 0;

  virtual amd_dbgapi_status_t query_exception_info (
    os_exception_code_t exception, os_source_id_t os_source_id,
    os_exception_info_t *os_exception_info, bool clear_exception) const
    = 0;

  virtual amd_dbgapi_status_t
  suspend_queues (const os_queue_id_t *queues, size_t queue_count,
                  os_exception_mask_t exceptions_cleared,
                  size_t *suspended_count,
                  os_queue_state_t *queue_states) const
    = 0;
  virtual amd_dbgapi_status_t
  resume_queues (const os_queue_id_t *queues, size_t queue_count,
                 size_t *resumed_count, os_queue_state_t *queue_states) const
    = 0;

  virtual amd_dbgapi_status_t
  queue_snapshot (os_queue_snapshot_entry_t *snapshots, size_t snapshot_count,
                  size_t *queue_count,
                  os_exception_mask_t exceptions_cleared) const
    = 0;

  virtual amd_dbgapi_status_t
  set_address_watch (os_agent_id_t os_agent_id, agent_address_t address,
                     agent_address_t mask, os_watch_mode_t os_watch_mode,
                     os_watch_id_t *os_watch_id) const
    = 0;

  virtual amd_dbgapi_status_t
  clear_address_watch (os_agent_id_t os_agent_id,
                       os_watch_id_t os_watch_id) const
    = 0;

  virtual amd_dbgapi_status_t
  set_wave_launch_mode (os_wave_launch_mode_t mode) const
    = 0;

  virtual amd_dbgapi_status_t set_wave_launch_trap_override (
    os_wave_launch_trap_override_t override, os_wave_launch_trap_mask_t value,
    os_wave_launch_trap_mask_t mask,
    os_wave_launch_trap_mask_t *previous_value = nullptr,
    os_wave_launch_trap_mask_t *supported_mask = nullptr) const
    = 0;

  virtual amd_dbgapi_status_t
  set_process_flags (os_process_flags_t flags) const
    = 0;

  virtual amd_dbgapi_status_t
  xfer_global_memory_partial (global_address_t address, void *read,
                              const void *write, size_t *size) const
    = 0;

  virtual amd_dbgapi_status_t
  xfer_host_memory_partial (host_address_t address, void *read,
                            const void *write, size_t *size) const
    = 0;

  virtual amd_dbgapi_status_t
  xfer_agent_memory_partial (os_agent_id_t agent, agent_address_t address,
                             void *read, const void *write, size_t *size) const
    = 0;
};

/* OS driver class that implements no access that can be used if there is no
   process.  */

class null_driver_t : public os_driver_t
{
public:
  null_driver_t (std::optional<amd_dbgapi_os_process_id_t> os_pid = {})
    : os_driver_t (os_pid)
  {
  }
  ~null_driver_t () override = default;

  /* Disable copies.  */
  null_driver_t (const null_driver_t &) = delete;
  null_driver_t &operator= (const null_driver_t &) = delete;

  bool is_valid () const override { return true; }

  amd_dbgapi_status_t check_version () const override
  {
    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  amd_dbgapi_status_t
  create_core_state_note (const os_runtime_info_t & /* runtime_info  */,
                          amd_dbgapi_core_state_data_t *data) const override
  {
    memset (data, 0, sizeof (amd_dbgapi_core_state_data_t));
    return AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED;
  }

  amd_dbgapi_status_t
  agent_snapshot (os_agent_info_t * /* snapshots  */,
                  size_t /* snapshot_count  */, size_t *agent_count,
                  os_exception_mask_t /* exceptions_cleared  */) const override
  {
    *agent_count = 0;
    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  amd_dbgapi_status_t
  enable_debug (os_exception_mask_t /* exceptions_reported  */,
                amd_dbgapi_notifier_t /* notifier  */,
                os_runtime_info_t * /* runtime_info  */) override
  {
    return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;
  }

  amd_dbgapi_status_t disable_debug () override
  {
    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  bool is_debug_enabled () const override { return false; }

  amd_dbgapi_status_t set_exceptions_reported (
    os_exception_mask_t /* exceptions_reported  */) const override
  {
    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  amd_dbgapi_status_t
  send_exceptions (os_exception_mask_t /* exceptions  */,
                   std::optional<os_agent_id_t> /* agent_id  */,
                   std::optional<os_queue_id_t> /* queue_id  */) const override
  {
    return AMD_DBGAPI_STATUS_ERROR;
  }

  amd_dbgapi_status_t query_debug_event (
    os_exception_mask_t *exceptions_present, os_queue_id_t *os_queue_id,
    os_agent_id_t *os_agent_id,
    os_exception_mask_t /* exceptions_cleared  */) const override
  {
    *exceptions_present = os_exception_mask_t::none;
    *os_queue_id = *os_agent_id = 0;
    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  amd_dbgapi_status_t
  query_exception_info (os_exception_code_t /* exception */,
                        os_source_id_t /* os_source_id  */,
                        os_exception_info_t * /* os_exception_info */,
                        bool /* clear_exception  */) const override
  {
    return AMD_DBGAPI_STATUS_ERROR;
  }

  amd_dbgapi_status_t
  suspend_queues (const os_queue_id_t * /* queues  */, size_t queue_count,
                  os_exception_mask_t /* exceptions_cleared  */,
                  size_t *suspended_count,
                  os_queue_state_t * /* queue_states  */) const override
  {
    dbgapi_assert (suspended_count != nullptr);

    if (queue_count > 0)
      fatal_error (
        "should not call this, null_driver does not have any queues");

    *suspended_count = 0;
    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  amd_dbgapi_status_t
  resume_queues (const os_queue_id_t * /* queues  */, size_t queue_count,
                 size_t *resumed_count,
                 os_queue_state_t * /* queue_states  */) const override
  {
    dbgapi_assert (resumed_count != nullptr);

    if (queue_count > 0)
      fatal_error (
        "should not call this, null_driver does not have any queues");

    *resumed_count = 0;

    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  amd_dbgapi_status_t
  queue_snapshot (os_queue_snapshot_entry_t * /* snapshots  */,
                  size_t /* snapshot_count  */, size_t *queue_count,
                  os_exception_mask_t /* exceptions_cleared  */) const override
  {
    *queue_count = 0;
    return AMD_DBGAPI_STATUS_SUCCESS;
  }

  amd_dbgapi_status_t set_address_watch (
    os_agent_id_t /* os_agent_id  */, agent_address_t /* address  */,
    agent_address_t /* mask  */, os_watch_mode_t /* os_watch_mode  */,
    os_watch_id_t * /* os_watch_id  */) const override
  {
    return AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED;
  }

  amd_dbgapi_status_t
  clear_address_watch (os_agent_id_t /* os_agent_id  */,
                       os_watch_id_t /* os_watch_id  */) const override
  {
    fatal_error (
      "should not call this, null_driver does not support watchpoints");
  }

  amd_dbgapi_status_t
  set_wave_launch_mode (os_wave_launch_mode_t /* mode  */) const override
  {
    return AMD_DBGAPI_STATUS_ERROR;
  }

  amd_dbgapi_status_t set_wave_launch_trap_override (
    os_wave_launch_trap_override_t /* override  */,
    os_wave_launch_trap_mask_t /* trap_mask  */,
    os_wave_launch_trap_mask_t /* requested_bits  */,
    os_wave_launch_trap_mask_t * /* previous_mask  */,
    os_wave_launch_trap_mask_t * /* supported_mask  */) const override
  {
    return AMD_DBGAPI_STATUS_ERROR;
  }

  amd_dbgapi_status_t
  set_process_flags (os_process_flags_t /* flags  */) const override
  {
    return AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED;
  }

  amd_dbgapi_status_t
  xfer_global_memory_partial (global_address_t /* address  */, void *read,
                              const void *write,
                              size_t * /* size  */) const override
  {
    dbgapi_assert (!read != !write && "either read or write buffer");
    /* Suppress warnings in release builds.  */
    [] (auto &&...) {}(read, write);
    return AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS;
  }

  amd_dbgapi_status_t
  xfer_host_memory_partial (host_address_t /* address  */, void *read,
                            const void *write,
                            size_t * /* size  */) const override
  {
    dbgapi_assert (!read != !write && "either read or write buffer");
    /* Suppress warnings in release builds.  */
    [] (auto &&...) {}(read, write);
    return AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS;
  }

  amd_dbgapi_status_t xfer_agent_memory_partial (
    os_agent_id_t /* agent  */, agent_address_t /* address */, void *read,
    const void *write, size_t * /* size  */) const override
  {
    dbgapi_assert (!read != !write && "either read or write buffer");
    /* Suppress warnings in release builds.  */
    [] (auto &&...) {}(read, write);
    return AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS;
  }
};

template <> std::string to_string (os_agent_info_t os_agent_info);
template <> std::string to_string (os_exception_code_t exception_code);
template <> std::string to_string (os_exception_mask_t exception_mask);
template <> std::string to_string (os_process_flags_t flags);
template <> std::string to_string (os_queue_snapshot_entry_t snapshot);
template <> std::string to_string (os_queue_type_t queue_type);
template <> std::string to_string (os_runtime_info_t runtime_info);
template <> std::string to_string (os_runtime_state_t runtime_state);
template <> std::string to_string (os_source_id_t source_id);
template <> std::string to_string (os_queue_state_t queue_state);
template <> std::string to_string (os_watch_mode_t watch_mode);
template <> std::string to_string (os_wave_launch_mode_t mode);
template <> std::string to_string (os_wave_launch_trap_override_t override);
template <> std::string to_string (detail::query_ref<os_exception_code_t> ref);

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_OS_DRIVER_H */
