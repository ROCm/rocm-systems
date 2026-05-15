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

#include "wave.h"
#include "architecture.h"
#include "dispatch.h"
#include "displaced_stepping.h"
#include "event.h"
#include "exception.h"
#include "initialization.h"
#include "logging.h"
#include "memory.h"
#include "os_driver.h"
#include "process.h"
#include "queue.h"
#include "register.h"
#include "utils.h"
#include "watchpoint.h"
#include "workgroup.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

namespace amd::dbgapi
{

wave_t::wave_t (amd_dbgapi_wave_id_t wave_id, workgroup_t &workgroup,
                std::optional<uint32_t> wave_in_group)
  : handle_object (wave_id), m_wave_in_group (wave_in_group),
    m_workgroup (workgroup)
{
}

wave_t::~wave_t ()
{
  if (displaced_stepping () != nullptr)
    {
      /* displaced step operations are cancelled by the process on detach,
         unless the process has exited and the queue is invalid, in which case,
         we simply release the displaced stepping buffer.  */
      dbgapi_assert (!queue ().is_valid ());
      displaced_stepping_t::release (m_displaced_stepping);
    }

  /* If the wave was single-stepping, the client is expecting either a stop
     event, or a command terminated event.  */
  if (state () == AMD_DBGAPI_WAVE_STATE_SINGLE_STEP)
    raise_event (AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED);
}

const dispatch_t &
wave_t::dispatch () const
{
  return m_workgroup.dispatch ();
}

compute_queue_t &
wave_t::queue () const
{
  return dispatch ().queue ();
}

const agent_t &
wave_t::agent () const
{
  return queue ().agent ();
}

process_t &
wave_t::process () const
{
  return agent ().process ();
}

const architecture_t &
wave_t::architecture () const
{
  return queue ().architecture ();
}

bool
wave_t::is_halted () const
{
  return architecture ().wave_get_halt (*this);
}

void
wave_t::set_halted (bool halted)
{
  architecture ().wave_set_halt (*this, halted);
}

void
wave_t::set_visibility (visibility_t visibility)
{
  if (m_visibility == visibility)
    return;

  log_info ("changing %s's visibility to %s", to_cstring (id ()),
            visibility == visibility_t::visible ? "visible" : "hidden");

  m_visibility = visibility;

  /* Since the visibility of this wave has changed, the list of waves returned
     by the process has also changed.  */
  process ().set_changed<wave_t> (true);
}

uint64_t
wave_t::exec_mask () const
{

  if (lane_count () == 32)
    {
      uint32_t exec;
      read_register (amdgpu_regnum_t::exec_32, &exec);
      return exec;
    }
  else if (lane_count () == 64)
    {
      uint64_t exec;
      read_register (amdgpu_regnum_t::exec_64, &exec);
      return exec;
    }
  fatal_error ("Not a valid lane_count for EXEC mask: %zu", lane_count ());
}

agent_address_t
wave_t::pc () const
{
  uint64_t pc;
  read_register (amdgpu_regnum_t::pc, &pc);
  return pc;
}

std::optional<instruction_t>
wave_t::instruction_at_pc (size_t pc_adjust) const
{
  size_t instruction_size = architecture ().largest_instruction_size ();
  std::vector<std::byte> instruction_bytes (instruction_size);

  agent_address_t instruction_pc = pc () + pc_adjust;
  dbgapi_assert (utils::is_aligned (
    instruction_pc, architecture ().minimum_instruction_alignment ()));

  try
    {
      instruction_size = agent ().read_agent_memory_partial (
        instruction_pc, instruction_bytes.data (), instruction_size);
    }
  catch (...)
    {
      return std::nullopt;
    }

  /* Trim partial and unread bytes.  */
  instruction_bytes.resize (instruction_size);

  return instruction_t (architecture (), std::move (instruction_bytes));
}

void
wave_t::park ()
{
  dbgapi_assert (state () == AMD_DBGAPI_WAVE_STATE_STOP
                 && "Cannot park a running wave");

  dbgapi_assert (!m_is_parked && "already parked");

  /* On architectures that do not support halting at certain instructions when
     a wave is stopped, for example a terminating instruction, we change its pc
     to point to an immutable trap instruction.  This guarantees that the
     wave will never be halted at such instructions.  */
  architecture ().save_pc_for_park (*this, pc ());

  write_register (amdgpu_regnum_t::pc, queue ().park_instruction_address ());

  m_is_parked = true;
  /* From now on, every read/write to the pc register will be done via
     architecture_t::saved_parked_pc /architecture_t::.save_pc_for_park.  The
     real pc in the context save area will be untouched.  */

  log_verbose ("parked %s (pc=%s)", to_cstring (id ()),
               to_cstring (architecture ().saved_parked_pc (*this)));
}

void
wave_t::unpark ()
{
  dbgapi_assert (state () != AMD_DBGAPI_WAVE_STATE_STOP
                 && "Cannot unpark a stopped wave");

  dbgapi_assert (m_is_parked && "not parked");

  agent_address_t saved_pc = pc ();

  m_is_parked = false;
  /* From now on, every read/write to the pc register will be from/to the
     context save area.  */

  write_register (amdgpu_regnum_t::pc, saved_pc);

  log_verbose ("unparked %s (pc=%s)", to_cstring (id ()), to_cstring (pc ()));
}

void
wave_t::terminate ()
{
  if (m_displaced_stepping != nullptr)
    {
      displaced_stepping_t::release (m_displaced_stepping);
      m_displaced_stepping = nullptr;
    }

  /* Mark the wave as invalid and un-halt it at a terminating instruction. This
     allows the hardware to terminate the wave, while ensuring that the wave is
     never reported to the client as existing.  */

  /* Make the PC point to an immutable terminating instruction.  */
  write_register (amdgpu_regnum_t::pc,
                  queue ().terminating_instruction_address ());

  /* Hide this wave so that it isn't reported to the client.  */
  set_visibility (wave_t::visibility_t::hidden_at_terminating_instruction);

  set_state (AMD_DBGAPI_WAVE_STATE_RUN);
}

void
wave_t::displaced_stepping_start (const void *saved_instruction_bytes)
{
  dbgapi_assert (!m_displaced_stepping && "already displaced stepping");
  dbgapi_assert (state () == AMD_DBGAPI_WAVE_STATE_STOP && "not stopped");

  /* Check if we already have a displaced stepping buffer for this pc
     that can be shared between waves associated with the same queue.
   */
  displaced_stepping_t *displaced_stepping = process ().find_if (
    [&] (const displaced_stepping_t &other)
    { return other.queue () == queue () && other.from () == pc (); });

  /* If we can't share a displaced stepping operation with another wave, create
     a new one.  */
  if (displaced_stepping == nullptr)
    {
      /* Reconstitute the original instruction bytes.  */
      std::vector<std::byte> original_instruction_bytes (
        architecture ().largest_instruction_size ());

      memcpy (original_instruction_bytes.data (), saved_instruction_bytes,
              architecture ().breakpoint_instruction ().size ());

      size_t offset = architecture ().breakpoint_instruction ().size ();
      size_t remaining = original_instruction_bytes.size () - offset;

      remaining = agent ().read_agent_memory_partial (
        pc () + offset, &original_instruction_bytes[offset], remaining);

      /* Trim partial/unread bytes.  */
      original_instruction_bytes.resize (offset + remaining);

      instruction_t original_instruction (
        architecture (), std::move (original_instruction_bytes));

      std::optional<compute_queue_t::displaced_instruction_ptr_t>
        displaced_instruction_ptr;

      if (architecture ().can_simulate (*this, original_instruction))
        {
          /* Since this instruction will be simulated when the wave is resumed,
             there is no need to allocate a displaced stepping buffer.  */
        }
      else if (architecture ().can_execute_displaced (*this,
                                                      original_instruction))
        {
          displaced_instruction_ptr.emplace (
            queue ().allocate_displaced_instruction (original_instruction));
        }
      else
        {
          /* If this instruction cannot be simulated nor displaced-stepped,
             then it must be inline-stepped.  */
          throw api_error_t (AMD_DBGAPI_STATUS_ERROR_ILLEGAL_INSTRUCTION);
        }

      displaced_stepping = &process ().create<displaced_stepping_t> (
        queue (), std::move (original_instruction), pc (),
        std::move (displaced_instruction_ptr));
    }

  if (displaced_stepping->to ())
    {
      write_register (amdgpu_regnum_t::pc, *displaced_stepping->to ());

      log_info ("changing %s's pc from %s to %s (started %s)",
                to_cstring (id ()), to_cstring (displaced_stepping->from ()),
                to_cstring (displaced_stepping->to ()),
                to_cstring (displaced_stepping->id ()));
    }

  displaced_stepping_t::retain (displaced_stepping);
  m_displaced_stepping = displaced_stepping;
}

void
wave_t::displaced_stepping_complete ()
{
  dbgapi_assert (m_displaced_stepping && "not displaced stepping");
  dbgapi_assert (state () == AMD_DBGAPI_WAVE_STATE_STOP && "not stopped");

  if (m_displaced_stepping->to ())
    {
      agent_address_t displaced_pc = pc ();
      agent_address_t restored_pc
        = displaced_pc
          + (m_displaced_stepping->from () - *m_displaced_stepping->to ());
      write_register (amdgpu_regnum_t::pc, restored_pc);

      log_info ("changing %s's pc from %s to %s (%s %s)", to_cstring (id ()),
                to_cstring (displaced_pc), to_cstring (pc ()),
                displaced_pc == *m_displaced_stepping->to () ? "aborted"
                                                             : "completed",
                to_cstring (m_displaced_stepping->id ()));
    }

  displaced_stepping_t::release (m_displaced_stepping);
  m_displaced_stepping = nullptr;
}

void
wave_t::update (
  std::unique_ptr<const architecture_t::cwsr_record_t> cwsr_record)
{
  dbgapi_assert (queue ().is_suspended ());
  const architecture_t &architecture = this->architecture ();

  dbgapi_assert (cwsr_record != nullptr);
  m_cwsr_record = std::move (cwsr_record);

  /* Check that the PC in the wave state save area is correctly aligned.  */
  if (!utils::is_aligned (pc (),
                          architecture.minimum_instruction_alignment ()))
    fatal_error ("corrupted state for %s: misaligned pc: %s",
                 to_cstring (id ()), to_cstring (pc ()));

  if (!m_ttmps_initialized)
    {
      /* Initialize the ttmp registers normally set up by SPI if this wave was
         created before the SPI ttmps setup was enabled.  */
      if (!agent ().spi_ttmps_setup_enabled ())
        architecture.initialize_spi_ttmps (*this);

      /* If the wave has not yet entered the trap handler, the ttmps used to
         communicate with the debugger API library may be undefined on some
         architectures.  */
      if (!architecture.are_trap_handler_ttmps_initialized (*this))
        architecture.initialize_trap_handler_ttmps (*this);

      architecture.record_spi_ttmps_setup (
        *this, agent ().spi_ttmps_setup_enabled ());

      write_register (amdgpu_regnum_t::wave_id, id ());
      m_ttmps_initialized = true;
    }

  /* Update the wave's state if this is a new wave, or if the wave was running
     the last time the queue it belongs to was resumed.  */
  amd_dbgapi_wave_state_t prev_state = m_state;
  if (prev_state != AMD_DBGAPI_WAVE_STATE_STOP)
    std::tie (m_state, m_stop_reason) = architecture.wave_get_state (*this);

  auto wave_state_to_string = [] (amd_dbgapi_wave_state_t state,
                                  amd_dbgapi_wave_stop_reasons_t stop_reason)
  {
    std::string string = to_string (state);
    if (state == AMD_DBGAPI_WAVE_STATE_STOP)
      string += string_printf (", stop_reason=%s", to_cstring (stop_reason));
    return string;
  };

  log_verbose (
    "%s%s in %s (pc=%s, state=%s) context_save:[%s..%s[",
    visibility () != visibility_t::visible ? "invisible " : "",
    to_cstring (id ()), to_cstring (workgroup ().id ()), to_cstring (pc ()),
    wave_state_to_string (m_state, m_stop_reason).c_str (),
    to_cstring (m_cwsr_record->begin ()), to_cstring (m_cwsr_record->end ()));

  /* The wave was running, and it is now stopped.  */
  if (prev_state != AMD_DBGAPI_WAVE_STATE_STOP
      && m_state == AMD_DBGAPI_WAVE_STATE_STOP)
    {
      /* Park the wave if the architecture does not support halting at certain
         instructions.  If the wave is loaded from a core dump, we do not park
         the wave as there is no trap instruction to park the wave to, and
         there is no need to park the wave in the first place either.  */
      if (architecture.park_stopped_waves (
            process ().runtime_rdebug_version ()))
        park ();

      if (visibility () == visibility_t::visible
          && m_stop_reason != AMD_DBGAPI_WAVE_STOP_REASON_NONE)
        raise_event (AMD_DBGAPI_EVENT_KIND_WAVE_STOP);
    }
}

void
wave_t::set_state (amd_dbgapi_wave_state_t state,
                   amd_dbgapi_exceptions_t exceptions)
{
  dbgapi_assert ((exceptions == AMD_DBGAPI_EXCEPTION_NONE
                  || state != AMD_DBGAPI_WAVE_STATE_STOP)
                 && "raising an exception requires the wave to be resumed");

  const architecture_t &architecture = this->architecture ();
  amd_dbgapi_wave_state_t prev_state = m_state;

  if (state == prev_state)
    return;

  dbgapi_assert (
    (!m_displaced_stepping || state != AMD_DBGAPI_WAVE_STATE_RUN)
    && "displaced-stepping waves can only be stopped or single-stepped");

  m_stop_requested = state == AMD_DBGAPI_WAVE_STATE_STOP;

  std::optional<instruction_t> instruction;
  if (m_displaced_stepping == nullptr
      && state == AMD_DBGAPI_WAVE_STATE_SINGLE_STEP)
    instruction = instruction_at_pc ();

  /* A wave single-stepping a terminating instruction does not generate a trap
     exception upon executing the instruction, so we need to immediately
     terminate the wave and enqueue an aborted command event.  */
  if (state == AMD_DBGAPI_WAVE_STATE_SINGLE_STEP
      && exceptions == AMD_DBGAPI_EXCEPTION_NONE &&
      [&] ()
      {
        if (m_displaced_stepping != nullptr)
          /* The displaced instruction is a terminating instruction.  */
          return architecture.is_terminating_instruction (
            m_displaced_stepping->original_instruction ());

        /* The current instruction at pc is a terminating instruction.  */
        return instruction
               && architecture.is_terminating_instruction (*instruction);
      }())
    {
      terminate ();
      raise_event (AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED);
      return;
    }

  log_info ("changing %s%s's state from %s to %s %s(pc=%s)",
            visibility () != visibility_t::visible ? "invisible " : "",
            to_cstring (id ()), to_cstring (prev_state), to_cstring (state),
            exceptions != AMD_DBGAPI_EXCEPTION_NONE
              ? ("with " + to_string (exceptions) + " ").c_str ()
              : "",
            to_cstring (pc ()));

  architecture.wave_set_state (*this, state);
  m_state = state;
  queue ().wave_state_changed (*this);

  if (architecture.park_stopped_waves (process ().runtime_rdebug_version ()))
    {
      if (state == AMD_DBGAPI_WAVE_STATE_STOP)
        park ();
      else
        unpark ();
    }

  if (state != AMD_DBGAPI_WAVE_STATE_STOP)
    {
      dbgapi_assert (prev_state == AMD_DBGAPI_WAVE_STATE_STOP
                     && "cannot resume an already running wave");

      /* m_last_stopped_pc is used to detect spurious single-step events
         (entered the trap handler with mode.debug_en=1 but pc ==
         m_last_stopped_pc).  Save the pc here as this is the last known
         pc before the wave is unhalted.  */
      m_last_stopped_pc = pc ();

      /* If there is a stop event, it must have already been reported otherwise
         the wave could not be resumed, and we can now clear it.  */
      [[maybe_unused]] auto has_unreported_stop_event = [this] ()
      {
        const event_t *event = process ().find (m_last_stop_event_id);
        return event != nullptr
               && event->state () < event_t::state_t::reported;
      };
      dbgapi_assert (!has_unreported_stop_event ()
                     && "a stop event for this wave is still pending");
      m_last_stop_event_id = AMD_DBGAPI_EVENT_NONE;

      /* Clear the stop reason.  */
      m_stop_reason = AMD_DBGAPI_WAVE_STOP_REASON_NONE;
    }
  else if (prev_state != AMD_DBGAPI_WAVE_STATE_STOP)
    {
      /* We requested the wave be stopped, and the wave wasn't already stopped,
         report an event to acknowledge that the wave has stopped.  */

      m_stop_reason = AMD_DBGAPI_WAVE_STOP_REASON_NONE;

      if (visibility () == visibility_t::visible)
        raise_event (prev_state == AMD_DBGAPI_WAVE_STATE_SINGLE_STEP
                       ? AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED
                       : AMD_DBGAPI_EVENT_KIND_WAVE_STOP);
    }

  if (state == AMD_DBGAPI_WAVE_STATE_SINGLE_STEP
        && exceptions == AMD_DBGAPI_EXCEPTION_NONE &&
        [&] () -> bool /* Return true if the instruction was simulated.  */
      {
        if (m_displaced_stepping != nullptr)
          {
            /* Simulate displaced instructions that are position sensitive (for
               example, instructions that manipulate the program counter).
               Displaced stepping buffers for such instructions do not have a
               to () address.  */

            return !m_displaced_stepping->to ()
                   && architecture.simulate (
                     *this, m_displaced_stepping->from (),
                     m_displaced_stepping->original_instruction ());
          }

        /* Simulate all instructions that can be simulated.  */
        return instruction && architecture.can_simulate (*this, *instruction)
               && architecture.simulate (*this, pc (), *instruction);
      }())
    {
      /* The instruction was successfully executed, update wave state and
         raise a stop event.  */
      update (std::move (m_cwsr_record));
      queue ().wave_state_changed (*this);
    }

  if (exceptions != AMD_DBGAPI_EXCEPTION_NONE)
    {
      auto convert_one_exception = [&] (amd_dbgapi_exceptions_t one_exception)
      {
        if (one_exception == AMD_DBGAPI_EXCEPTION_WAVE_ABORT)
          return os_exception_mask_t::queue_wave_abort;

        if (one_exception == AMD_DBGAPI_EXCEPTION_WAVE_TRAP)
          return os_exception_mask_t::queue_wave_trap;

        if (one_exception == AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR)
          return os_exception_mask_t::queue_wave_math_error;

        if (one_exception == AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION)
          return os_exception_mask_t::queue_wave_illegal_instruction;

        if (one_exception == AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION)
          return os_exception_mask_t::queue_wave_memory_violation
                 | (agent ().exceptions ()
                    & os_exception_mask_t::device_memory_violation);

        if (one_exception == AMD_DBGAPI_EXCEPTION_WAVE_ADDRESS_ERROR)
          return os_exception_mask_t::queue_wave_address_error;

        dbgapi_assert_not_reached ("not a valid exception");
      };

      /* Convert an amd_dbgapi_exception_t into an os_exception_mask_t.  */
      os_exception_mask_t os_exceptions = os_exception_mask_t::none;

      while (exceptions)
        {
          auto one_exception = exceptions ^ (exceptions & (exceptions - 1));
          os_exceptions |= convert_one_exception (one_exception);
          exceptions ^= one_exception;
        }

      /* A wave should only send queue exceptions, sometimes combined with a
         device_memory_exception.  */
      dbgapi_assert ((os_exceptions & os_queue_exception_mask) != 0);

      /* Halt the wave if resuming with exceptions.  */
      architecture.wave_set_halt (*this, true);

      process ().send_exceptions (os_exceptions, &queue ());
    }
}

bool
wave_t::is_register_available (amdgpu_regnum_t regnum) const
{
  if (is_pseudo_register (regnum))
    return architecture ().is_pseudo_register_available (*this, regnum);

  return register_address (regnum).has_value ();
}

void
wave_t::read_register (amdgpu_regnum_t regnum, size_t offset,
                       size_t value_size, void *value) const
{
  if (is_pseudo_register (regnum))
    return architecture ().read_pseudo_register (*this, regnum, offset,
                                                 value_size, value);

  dbgapi_assert (value_size
                 && (offset + value_size)
                      <= architecture ().register_size (regnum)
                 && "read_register is out of bounds");

  auto reg_addr = register_address (regnum);

  /* Out of range sgpr, read s0.  */
  if (!reg_addr
      && (regnum >= amdgpu_regnum_t::first_sgpr
          && regnum <= amdgpu_regnum_t::last_sgpr))
    reg_addr = register_address (amdgpu_regnum_t::s0);

  /* Out of range vgpr, read v0.  */
  if (!reg_addr
      && (regnum >= amdgpu_regnum_t::first_vgpr
          && regnum <= amdgpu_regnum_t::last_vgpr))
    reg_addr = register_address (lane_count () == 32 ? amdgpu_regnum_t::v0_32
                                                     : amdgpu_regnum_t::v0_64);

  dbgapi_assert (reg_addr);

  if (m_is_parked && regnum == amdgpu_regnum_t::pc)
    {
      agent_address_t parked_pc = architecture ().saved_parked_pc (*this);
      memcpy (value, reinterpret_cast<const std::byte *> (&parked_pc) + offset,
              value_size);
      return;
    }

  std::optional<scoped_queue_suspend_t> suspend;
  if (!queue ().is_suspended ()
      && !agent ().memory_cache ().contains_all (*reg_addr + offset,
                                                 value_size))
    {
      /* Get the wave_id before suspending the queue, as this wave could have
         exited, and queue_t::update_waves may destroy this wave_t.  */
      amd_dbgapi_wave_id_t wave_id = id ();

      suspend.emplace (queue (), "read register");

      /* Look for the wave_id again, the wave may have exited.  */
      wave_t *wave = find (wave_id);
      if (wave == nullptr)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

      dbgapi_assert (wave == this);

      /* The wave's saved state may have changed location in memory.  */
      reg_addr = register_address (regnum);
    }

  agent ().read_agent_memory (*reg_addr + offset, value, value_size);
}

void
wave_t::write_register (amdgpu_regnum_t regnum, size_t offset,
                        size_t value_size, const void *value) const
{
  if (is_pseudo_register (regnum))
    return architecture ().write_pseudo_register (*this, regnum, offset,
                                                  value_size, value);

  dbgapi_assert (value_size
                 && (offset + value_size)
                      <= architecture ().register_size (regnum)
                 && "write_register is out of bounds");

  auto reg_addr = register_address (regnum);

  if (!reg_addr
      && ((regnum >= amdgpu_regnum_t::first_sgpr
           && regnum <= amdgpu_regnum_t::last_sgpr)
          || (regnum >= amdgpu_regnum_t::first_vgpr
              && regnum <= amdgpu_regnum_t::last_vgpr)))
    /* Out of range sgpr or vgpr, the register write is dropped.  */
    return;

  dbgapi_assert (reg_addr);

  if (m_is_parked && regnum == amdgpu_regnum_t::pc)
    {
      agent_address_t parked_pc = architecture ().saved_parked_pc (*this);
      if (auto *read_only
          = architecture ().register_read_only_mask (amdgpu_regnum_t::pc);
          read_only != nullptr)
        {
          agent_address_t pc = parked_pc;
          memcpy (reinterpret_cast<std::byte *> (&pc) + offset, value,
                  value_size);

          uint64_t mask = *static_cast<const uint64_t *> (read_only);

          architecture ().save_pc_for_park (*this,
                                            (pc & ~mask) | (parked_pc & mask));
        }
      else
        {
          memcpy (reinterpret_cast<std::byte *> (&parked_pc) + offset, value,
                  value_size);
        }

      return;
    }

  std::optional<scoped_queue_suspend_t> suspend;
  if (!queue ().is_suspended ())
    {
      /* Get the wave_id before suspending the queue, as this wave could have
         exited, and queue_t::update_waves may destroy this wave_t.  */
      amd_dbgapi_wave_id_t wave_id = id ();

      suspend.emplace (queue (), "write register");

      /* Look for the wave_id again, the wave may have exited.  */
      wave_t *wave = find (wave_id);
      if (wave == nullptr)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

      dbgapi_assert (wave == this);

      /* The wave's saved state may have changed location in memory.  */
      reg_addr = register_address (regnum);
    }

  /* This register might not be entirely writable.  Read-only bits in the new
     value must be replaced by the current content of the register.  */
  if (auto *read_only = architecture ().register_read_only_mask (regnum);
      read_only != nullptr)
    {
      void *masked_value = alloca (value_size);
      agent ().read_agent_memory (*reg_addr + offset, masked_value,
                                  value_size);

      if (offset == 0 && value_size == 8)
        {
          /* Optimize the case when writing an entire 64-bit register.  */
          uint64_t mask = *static_cast<const uint64_t *> (read_only);
          uint64_t &x8 = *static_cast<uint64_t *> (masked_value);
          x8 = (*static_cast<const uint64_t *> (value) & ~mask) | (x8 & mask);
        }
      else if (offset == 0 && value_size == 4)
        {
          /* Optimize the case when writing an entire 32-bit register.  */
          uint32_t mask = *static_cast<const uint32_t *> (read_only);
          uint32_t &x4 = *static_cast<uint32_t *> (masked_value);
          x4 = (*static_cast<const uint32_t *> (value) & ~mask) | (x4 & mask);
        }
      else /* General case, mask one byte at a time.  */
        for (size_t i = 0; i < value_size; ++i)
          {
            std::byte mask
              = static_cast<const std::byte *> (read_only)[offset + i];
            std::byte &x1 = static_cast<std::byte *> (masked_value)[i];
            x1 = (*static_cast<const std::byte *> (value) & ~mask)
                 | (x1 & mask);
          }

      value = masked_value;
    }

  agent ().write_agent_memory (*reg_addr + offset, value, value_size);
}

/* Return the wave's scratch memory region (address and size).  */
std::pair<agent_address_t /* address */, amd_dbgapi_size_t /* size */>
wave_t::scratch_memory_region () const
{
  auto [address, size]
    = queue ().scratch_memory_region (*m_cwsr_record.get ());

  /* On architectures with an architected flat_scratch register, check that the
     computed slot scratch address matches the content of the register.  */
  if (architecture ().has_architected_flat_scratch ())
    {
      agent_address_t flat_scratch;
      read_register (amdgpu_regnum_t::flat_scratch, &flat_scratch);

      if (address != flat_scratch && size != 0)
        {
          warning ("flat_scratch may be corrupted, "
                   "private memory access is disabled");

          /* If the computed scratch address differs from the content of the
             flat_scratch register, either the calculation is incorrect or the
             register is corrupted.  Disable access to the scratch memory
             region by returning a 0 size.  */
          size = 0;
        }
    }

  return { address, size };
}

size_t
wave_t::xfer_private_memory (const address_space_t &address_space,
                             amd_dbgapi_segment_address_t segment_address,
                             amd_dbgapi_lane_id_t lane_id, void *read,
                             const void *write, size_t size)
{
  /* private_swizzled and private_unswizzled memory is backed by global memory,
     so we can convert the private segment addresses and read/write from
     global memory.  */

  size_t xfer_bytes = 0;
  while (size > 0)
    {
      global_address_t global_address;
      size_t contiguous_bytes;

      try
        {
          std::tie (global_address, contiguous_bytes)
            = address_space_t::global ().convert (
              *this, lane_id, address_space, segment_address + xfer_bytes);
        }
      catch (...)
        {
          /* A conversion exception means that the segment address is out of
             bounds for the given address space.  Return the number of bytes
             transferred so far, or throw a memory access error exception if
             none were transferred.  */

          if (!xfer_bytes)
            throw memory_access_error_t (address_space, segment_address,
                                         "address is out of bounds");
          break;
        }

      /* The transfer size is limited by the amount of contiguous bytes for
         this conversion.  Multiple iterations may be needed to completely
         transfer the host buffer using swizzled address spaces.  */
      size_t request_size = std::min (size, contiguous_bytes);

      size_t xfer_size = process ().xfer_global_memory (
        global_address,
        read != nullptr ? static_cast<std::byte *> (read) + xfer_bytes : read,
        write != nullptr ? static_cast<const std::byte *> (write) + xfer_bytes
                         : write,
        request_size);

      size -= xfer_size;
      xfer_bytes += xfer_size;

      if (xfer_size != request_size)
        {
          /* global_address + xfer_size may have reached the end of a mapped
             global memory region.  */
          break;
        }
    }

  return xfer_bytes;
}

size_t
wave_t::xfer_segment_memory (const address_space_t &address_space,
                             amd_dbgapi_segment_address_t segment_address,
                             amd_dbgapi_lane_id_t lane_id, void *read,
                             const void *write, size_t size)
{
  dbgapi_assert (state () == AMD_DBGAPI_WAVE_STATE_STOP
                 && "the wave must be stopped to read/write memory");
  dbgapi_assert (!read != !write && "either read or write buffer");

  auto [lowered_address_space, lowered_address]
    = address_space.lower (segment_address);

  switch (lowered_address_space.kind ())
    {
    case address_space_t::kind_t::private_swizzled:
    case address_space_t::kind_t::private_unswizzled:
      return xfer_private_memory (lowered_address_space, lowered_address,
                                  lane_id, read, write, size);

    default:
      throw memory_access_error_t (lowered_address_space, lowered_address,
                                   "address is not supported");
    }
}

void
wave_t::raise_event (amd_dbgapi_event_kind_t event_kind)
{
  process_t &process = this->process ();
  event_t &event = process.create<event_t> (process, event_kind, id ());

  if (event_kind == AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED
      || event_kind == AMD_DBGAPI_EVENT_KIND_WAVE_STOP)
    m_last_stop_event_id = event.id ();

  process.enqueue_event (event);
}

const event_t *
wave_t::last_stop_event () const
{
  dbgapi_assert (state () == AMD_DBGAPI_WAVE_STATE_STOP);
  return process ().find (m_last_stop_event_id);
}

amd_dbgapi_wave_state_t
wave_t::client_visible_state () const
{
  amd_dbgapi_wave_state_t state = this->state ();

  if (state != AMD_DBGAPI_WAVE_STATE_STOP)
    return state;

  if (const event_t *event = last_stop_event ();
      event == nullptr || event->state () >= event_t::state_t::reported)
    return AMD_DBGAPI_WAVE_STATE_STOP;

  /* If the wave is stopped, but the wave stop event has not yet been
     reported to the client, return the last resumed state.  */
  return (stop_reason () & AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP)
           ? AMD_DBGAPI_WAVE_STATE_SINGLE_STEP
           : AMD_DBGAPI_WAVE_STATE_RUN;
}

void
wave_t::get_info (amd_dbgapi_wave_info_t query, size_t value_size,
                  void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_WAVE_INFO_STATE:
      utils::get_info (value_size, value, client_visible_state ());
      return;

    case AMD_DBGAPI_WAVE_INFO_STOP_REASON:
      utils::get_info (value_size, value, stop_reason ());
      return;

    case AMD_DBGAPI_WAVE_INFO_WORKGROUP:
      if (workgroup ().id () == AMD_DBGAPI_WORKGROUP_NONE)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
      utils::get_info (value_size, value, workgroup ().id ());
      return;

    case AMD_DBGAPI_WAVE_INFO_DISPATCH:
      if (dispatch ().id () == AMD_DBGAPI_DISPATCH_NONE)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
      utils::get_info (value_size, value, dispatch ().id ());
      return;

    case AMD_DBGAPI_WAVE_INFO_QUEUE:
      utils::get_info (value_size, value, queue ().id ());
      return;

    case AMD_DBGAPI_WAVE_INFO_AGENT:
      utils::get_info (value_size, value, agent ().id ());
      return;

    case AMD_DBGAPI_WAVE_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;

    case AMD_DBGAPI_WAVE_INFO_ARCHITECTURE:
      utils::get_info (value_size, value, architecture ().id ());
      return;

    case AMD_DBGAPI_WAVE_INFO_PC:
      utils::get_info (value_size, value, pc ());
      return;

    case AMD_DBGAPI_WAVE_INFO_EXEC_MASK:
      utils::get_info (value_size, value, exec_mask ());
      return;

    case AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD:
      if (!workgroup ().group_ids ())
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
      utils::get_info (value_size, value, *workgroup ().group_ids ());
      return;

    case AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP:
      if (!m_wave_in_group)
        throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
      utils::get_info (value_size, value, *m_wave_in_group);
      return;

    case AMD_DBGAPI_WAVE_INFO_WATCHPOINTS:
      {
        auto os_watch_ids = architecture ().triggered_watchpoints (*this);

        auto watchpoint_ids = allocate_memory<amd_dbgapi_watchpoint_id_t[]> (
          os_watch_ids.size () * sizeof (amd_dbgapi_watchpoint_id_t));

        auto watchpoint_id = [this] (os_watch_id_t os_watch_id)
        {
          const watchpoint_t *watchpoint
            = agent ().get_watchpoint (os_watch_id);
          if (watchpoint == nullptr)
            fatal_error ("kfd_watch_%d not set on %s", os_watch_id,
                         to_cstring (agent ().id ()));
          return watchpoint->id ();
        };

        std::transform (os_watch_ids.begin (), os_watch_ids.end (),
                        watchpoint_ids.get (), watchpoint_id);

        utils::get_info (value_size, value,
                         amd_dbgapi_watchpoint_list_t{
                           os_watch_ids.size (), watchpoint_ids.get () });

        watchpoint_ids.release ();
        return;
      }

    case AMD_DBGAPI_WAVE_INFO_LANE_COUNT:
      utils::get_info (value_size, value, lane_count ());
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_wave_stop (amd_dbgapi_wave_id_t wave_id)
{
  TRACE_BEGIN (param_in (wave_id));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    if (wave->client_visible_state () == AMD_DBGAPI_WAVE_STATE_STOP)
      THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED);

    if (wave->stop_requested ())
      THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_OUTSTANDING_STOP);

    scoped_queue_suspend_t suspend (wave->queue (), "stop wave");

    /* Look for the wave_id again, the wave may have exited.  */
    if ((wave = find (wave_id)) == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    if (wave->process ().is_frozen ())
      THROW (AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);

    wave->set_state (AMD_DBGAPI_WAVE_STATE_STOP);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED,
         AMD_DBGAPI_STATUS_ERROR_WAVE_OUTSTANDING_STOP,
         AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);
  TRACE_END ();
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_wave_resume (amd_dbgapi_wave_id_t wave_id,
                        amd_dbgapi_resume_mode_t resume_mode,
                        amd_dbgapi_exceptions_t exceptions)
{
  TRACE_BEGIN (param_in (wave_id), param_in (resume_mode),
               param_in (exceptions));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    if (wave->process ().is_frozen ())
      THROW (AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);

    if (resume_mode != AMD_DBGAPI_RESUME_MODE_NORMAL
        && resume_mode != AMD_DBGAPI_RESUME_MODE_SINGLE_STEP)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    if ((exceptions
         & ~(AMD_DBGAPI_EXCEPTION_WAVE_ABORT | AMD_DBGAPI_EXCEPTION_WAVE_TRAP
             | AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR
             | AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION
             | AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION
             | AMD_DBGAPI_EXCEPTION_WAVE_ADDRESS_ERROR))
        != AMD_DBGAPI_EXCEPTION_NONE)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    if (wave->client_visible_state () != AMD_DBGAPI_WAVE_STATE_STOP)
      THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);

    /* The wave is not resumable if the stop event is not yet processed.  */
    if (const event_t *event = wave->last_stop_event ();
        event != nullptr && event->state () < event_t::state_t::processed)
      {
        log_verbose ("%s is not resumable because its last stop event (%s) "
                     "has not been processed",
                     to_cstring (wave->id ()), to_cstring (event->id ()));
        THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_RESUMABLE);
      }

    if (wave->displaced_stepping () != nullptr
        && resume_mode != AMD_DBGAPI_RESUME_MODE_SINGLE_STEP)
      THROW (AMD_DBGAPI_STATUS_ERROR_RESUME_DISPLACED_STEPPING);

    scoped_queue_suspend_t suspend (wave->queue (), "resume wave");

    /* Look for the wave_id again, the wave may have exited.  */
    if ((wave = find (wave_id)) == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    wave->set_state (resume_mode == AMD_DBGAPI_RESUME_MODE_SINGLE_STEP
                       ? AMD_DBGAPI_WAVE_STATE_SINGLE_STEP
                       : AMD_DBGAPI_WAVE_STATE_RUN,
                     exceptions);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED,
         AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_RESUMABLE,
         AMD_DBGAPI_STATUS_ERROR_RESUME_DISPLACED_STEPPING,
         AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);
  TRACE_END ();
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_wave_get_info (amd_dbgapi_wave_id_t wave_id,
                          amd_dbgapi_wave_info_t query, size_t value_size,
                          void *value)
{
  TRACE_BEGIN (param_in (wave_id), param_in (query), param_in (value_size),
               param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    switch (query)
      {
      case AMD_DBGAPI_WAVE_INFO_STOP_REASON:
      case AMD_DBGAPI_WAVE_INFO_PC:
      case AMD_DBGAPI_WAVE_INFO_EXEC_MASK:
      case AMD_DBGAPI_WAVE_INFO_WATCHPOINTS:
        if (wave->client_visible_state () != AMD_DBGAPI_WAVE_STATE_STOP)
          THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);
      default:
        break;
      };

    wave->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE,
         AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_process_wave_list (amd_dbgapi_process_id_t process_id,
                              size_t *wave_count, amd_dbgapi_wave_id_t **waves,
                              amd_dbgapi_changed_t *changed)
{
  TRACE_BEGIN (param_in (process_id), param_in (wave_count), param_in (waves),
               param_in (changed));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    std::vector<process_t *> processes = process_t::match (process_id);

    if (waves == nullptr || wave_count == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    std::vector<std::pair<process_t *, std::vector<queue_t *>>>
      queues_needing_resume;

    for (auto &&process : processes)
      {
        process->update_queues ();

        std::vector<queue_t *> queues;
        for (auto &&queue : process->range<queue_t> ())
          if (!queue.is_suspended ())
            queues.emplace_back (&queue);

        process->suspend_queues (queues, "refresh wave list");

        if (process->forward_progress_needed ())
          queues_needing_resume.emplace_back (process, std::move (queues));
      }

    amd_dbgapi_changed_t wave_list_changed;
    auto wave_list = utils::get_handle_list<wave_t> (
      processes, changed != nullptr ? &wave_list_changed : nullptr);

    auto deallocate_wave_list = utils::make_scope_fail (
      [&] () { amd::dbgapi::deallocate_memory (wave_list.first); });

    for (auto &&[process, queues] : queues_needing_resume)
      process->resume_queues (queues, "refresh wave list");

    std::tie (*waves, *wave_count) = wave_list;
    if (changed != nullptr)
      *changed = wave_list_changed;
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_ref (param_out (wave_count)),
             make_ref (make_ref (param_out (waves)), *wave_count),
             make_ref (param_out (changed)));
}
