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

#include "queue.h"
#include "architecture.h"
#include "debug.h"
#include "dispatch.h"
#include "exception.h"
#include "handle_object.h"
#include "initialization.h"
#include "logging.h"
#include "memory.h"
#include "process.h"
#include "register.h"
#include "utils.h"
#include "wave.h"

#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>

#include <array>
#include <cinttypes>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace amd::dbgapi
{

process_t &
queue_t::process () const
{
  return agent ().process ();
}

const architecture_t &
queue_t::architecture () const
{
  dbgapi_assert (agent ().architecture ());
  return *agent ().architecture ();
}

namespace detail
{

class aql_queue_t : public compute_queue_t
{
private:
  static constexpr uint64_t aql_packet_size = 64;
  static constexpr amd_dbgapi_size_t debugger_memory_chunk_size = 32;

  struct context_save_area_header_s
  {
    uint32_t control_stack_offset;
    uint32_t control_stack_size;
    uint32_t wave_state_offset;
    uint32_t wave_state_size;
    uint32_t debugger_memory_offset;
    uint32_t debugger_memory_size;
  };

  class aql_dispatch_t : public dispatch_t
  {
  private:
    hsa_kernel_dispatch_packet_t m_packet{};
    std::unique_ptr<const architecture_t::kernel_descriptor_t>
      m_kernel_descriptor{};

  public:
    aql_dispatch_t (amd_dbgapi_dispatch_id_t dispatch_id,
                    compute_queue_t &queue,
                    amd_dbgapi_os_queue_packet_id_t os_queue_packet_id);

    const architecture_t::kernel_descriptor_t &
    kernel_descriptor () const override
    {
      dbgapi_assert (m_kernel_descriptor);
      return *m_kernel_descriptor;
    }

    uint32_t grid_dimensions () const;
    std::array<uint32_t, 3> grid_sizes () const;
    std::array<uint16_t, 3> workgroup_sizes () const;

    void get_info (amd_dbgapi_dispatch_info_t query, size_t value_size,
                   void *value) const override;
  };

  std::optional<amd_dbgapi_os_queue_packet_id_t> m_read_packet_id{};
  std::optional<amd_dbgapi_os_queue_packet_id_t> m_write_packet_id{};
  amd_dbgapi_global_address_t m_scratch_backing_memory_address{ 0 };
  uint32_t m_compute_tmpring_size{ 0 };

  /* The memory reserved by the thunk library for the debugger is used to store
     instruction buffers.  Instruction buffers are lazily allocated from the
     reserved memory, and when freed, their index is returned to a free list.
     Each wave is guaranteed its own unique instruction buffer.  */
  std::optional<agent_address_t> m_debugger_memory_base{};

  uint16_t m_debugger_memory_chunk_count{ 0 };
  uint16_t m_debugger_memory_next_chunk{ 0 };
  std::vector<uint16_t> m_debugger_memory_free_chunks{};

  std::optional<displaced_instruction_ptr_t> m_park_instruction_ptr{};
  std::optional<displaced_instruction_ptr_t> m_terminating_instruction_ptr{};

  /* Value used to mark waves that are found in the context save area. When
     sweeping, any wave found with a mark less than the current mark will be
     deleted, as these waves are no longer active.  */
  monotonic_counter_t<epoch_t, 1> m_next_wave_mark{};

  std::optional<amd_dbgapi_os_queue_packet_id_t> get_os_queue_packet_id (
    const architecture_t::cwsr_record_t &cwsr_record) const;

  displaced_instruction_ptr_t
  allocate_displaced_instruction (const instruction_t &instruction) override;

  void queue_state_changed () override;

  void update_waves ();

public:
  aql_queue_t (amd_dbgapi_queue_id_t queue_id, const agent_t &agent,
               const os_queue_snapshot_entry_t &os_queue_info);
  ~aql_queue_t () override;

  /* Return the queue type.  */
  amd_dbgapi_os_queue_type_t type () const override
  {
    return AMD_DBGAPI_OS_QUEUE_TYPE_HSA_AQL;
  }

  /* Return the address of a park instruction.  */
  agent_address_t park_instruction_address () override
  {
    /* When the queue is loaded from a core dump, there is no running wave, and
       we cannot allocate nor initialize displaced instruction buffers. Setting
       the PC to 0 is harmless for a stopped wave.  */
    if (process ().from_core ())
      return {};

    if (!m_park_instruction_ptr)
      m_park_instruction_ptr.emplace (allocate_displaced_instruction (
        architecture ().assert_instruction ()));

    return m_park_instruction_ptr->get ();
  }

  /* Return the address of a terminating instruction.  */
  agent_address_t terminating_instruction_address () override
  {
    /* See park_instruction_address.  */
    if (process ().from_core ())
      return {};

    if (!m_terminating_instruction_ptr)
      m_terminating_instruction_ptr.emplace (allocate_displaced_instruction (
        architecture ().terminating_instruction ()));

    return m_terminating_instruction_ptr->get ();
  }

  std::pair<agent_address_t /* address */, amd_dbgapi_size_t /* size */>
  scratch_memory_region (
    const architecture_t::cwsr_record_t &cwsr_record) const override;

  size_t packet_size () const override { return aql_packet_size; };

  void active_packets_info (amd_dbgapi_os_queue_packet_id_t *read_packet_id_p,
                            amd_dbgapi_os_queue_packet_id_t *write_packet_id_p,
                            size_t *packets_byte_size_p) const override;

  void active_packets_bytes (amd_dbgapi_os_queue_packet_id_t read_packet_id,
                             amd_dbgapi_os_queue_packet_id_t write_packet_id,
                             void *memory, size_t memory_size) const override;
};

aql_queue_t::aql_dispatch_t::aql_dispatch_t (
  amd_dbgapi_dispatch_id_t dispatch_id, compute_queue_t &queue,
  amd_dbgapi_os_queue_packet_id_t os_queue_packet_id)
  : dispatch_t (dispatch_id, queue, os_queue_packet_id)
{
  amd_dbgapi_global_address_t packet_address
    = queue.address ()
      + (os_queue_packet_id * aql_packet_size) % queue.size ();

  /* Read the dispatch packet and kernel descriptor.  */
  process ().read_host_memory (packet_address, &m_packet);

  m_kernel_descriptor = architecture ().make_kernel_descriptor (
    process (), m_packet.kernel_object);
}

uint32_t
aql_queue_t::aql_dispatch_t::grid_dimensions () const
{
  return static_cast<uint32_t> (utils::bit_extract (
    m_packet.setup, HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS,
    HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
      + HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS - 1));
}

std::array<uint32_t, 3>
aql_queue_t::aql_dispatch_t::grid_sizes () const
{
  return { m_packet.grid_size_x, m_packet.grid_size_y, m_packet.grid_size_z };
}

std::array<uint16_t, 3>
aql_queue_t::aql_dispatch_t::workgroup_sizes () const
{
  return { m_packet.workgroup_size_x, m_packet.workgroup_size_y,
           m_packet.workgroup_size_z };
}

void
aql_queue_t::aql_dispatch_t::get_info (amd_dbgapi_dispatch_info_t query,
                                       size_t value_size, void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_DISPATCH_INFO_QUEUE:
      utils::get_info (value_size, value, queue ().id ());
      return;

    case AMD_DBGAPI_DISPATCH_INFO_AGENT:
      utils::get_info (value_size, value, agent ().id ());
      return;

    case AMD_DBGAPI_DISPATCH_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;

    case AMD_DBGAPI_DISPATCH_INFO_ARCHITECTURE:
      utils::get_info (value_size, value, architecture ().id ());
      return;

    case AMD_DBGAPI_DISPATCH_INFO_OS_QUEUE_PACKET_ID:
      utils::get_info (value_size, value, os_queue_packet_id ());
      return;

    case AMD_DBGAPI_DISPATCH_INFO_BARRIER:
      utils::get_info (value_size, value,
                       utils::bit_extract (m_packet.header,
                                           HSA_PACKET_HEADER_BARRIER,
                                           HSA_PACKET_HEADER_BARRIER)
                         ? AMD_DBGAPI_DISPATCH_BARRIER_PRESENT
                         : AMD_DBGAPI_DISPATCH_BARRIER_NONE);
      return;

    case AMD_DBGAPI_DISPATCH_INFO_ACQUIRE_FENCE:
      static_assert ((int)AMD_DBGAPI_DISPATCH_FENCE_SCOPE_NONE
                       == (int)HSA_FENCE_SCOPE_NONE,
                     "amd_dbgapi_dispatch_fence_scope_t != hsa_fence_scope_t");
      static_assert ((int)AMD_DBGAPI_DISPATCH_FENCE_SCOPE_AGENT
                       == (int)HSA_FENCE_SCOPE_AGENT,
                     "amd_dbgapi_dispatch_fence_scope_t != hsa_fence_scope_t");
      static_assert ((int)AMD_DBGAPI_DISPATCH_FENCE_SCOPE_SYSTEM
                       == (int)HSA_FENCE_SCOPE_SYSTEM,
                     "amd_dbgapi_dispatch_fence_scope_t != hsa_fence_scope_t");

      utils::get_info (
        value_size, value,
        static_cast<amd_dbgapi_dispatch_fence_scope_t> (utils::bit_extract (
          m_packet.header, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
          HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
            + HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE - 1)));
      return;

    case AMD_DBGAPI_DISPATCH_INFO_RELEASE_FENCE:
      utils::get_info (
        value_size, value,
        static_cast<amd_dbgapi_dispatch_fence_scope_t> (utils::bit_extract (
          m_packet.header, HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
          HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
            + HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE - 1)));
      return;

    case AMD_DBGAPI_DISPATCH_INFO_GRID_DIMENSIONS:
      utils::get_info (value_size, value, grid_dimensions ());
      return;

    case AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES:
      {
        utils::get_info (value_size, value, workgroup_sizes ());
        return;
      }
    case AMD_DBGAPI_DISPATCH_INFO_GRID_SIZES:
      {
        utils::get_info (value_size, value, grid_sizes ());
        return;
      }
    case AMD_DBGAPI_DISPATCH_INFO_PRIVATE_SEGMENT_SIZE:
      utils::get_info (
        value_size, value,
        static_cast<amd_dbgapi_size_t> (m_packet.private_segment_size));
      return;

    case AMD_DBGAPI_DISPATCH_INFO_GROUP_SEGMENT_SIZE:
      utils::get_info (
        value_size, value,
        static_cast<amd_dbgapi_size_t> (m_packet.group_segment_size));
      return;

    case AMD_DBGAPI_DISPATCH_INFO_KERNEL_ARGUMENT_SEGMENT_ADDRESS:
      utils::get_info (value_size, value, m_packet.kernarg_address);
      return;

    case AMD_DBGAPI_DISPATCH_INFO_KERNEL_DESCRIPTOR_ADDRESS:
      utils::get_info (value_size, value, kernel_descriptor ().address ());
      return;

    case AMD_DBGAPI_DISPATCH_INFO_KERNEL_CODE_ENTRY_ADDRESS:
      utils::get_info (value_size, value,
                       kernel_descriptor ().entry_address ());
      return;

    case AMD_DBGAPI_DISPATCH_INFO_KERNEL_COMPLETION_ADDRESS:
      utils::get_info (value_size, value, m_packet.completion_signal);
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

aql_queue_t::aql_queue_t (amd_dbgapi_queue_id_t queue_id, const agent_t &agent,
                          const os_queue_snapshot_entry_t &os_queue_info)
  : compute_queue_t (queue_id, agent, os_queue_info)
{
  /* The dispatch packet index is stored in a trap temporary register when a
     wave associated with that packet is initialized.  Make sure the field
     containing the packet index is large enough.  The ROCr should not create
     queues with more packets than the field can hold.  */
  if ((size () / aql_packet_size)
      > architecture ().maximum_queue_packet_count ())
    fatal_error ("queue ring size = %#zx is not supported", size ());
}

aql_queue_t::~aql_queue_t ()
{
  process_t &process = this->process ();

  /* Destruct all waves, workgroups, and dispatches associated with this queue.
     Waves that are executing a displaced stepped instruction will release the
     displaced stepping buffer when destructed, and raise a command terminated
     event if single-stepping (see wave_t::~wave_t).  */

  auto &&wave_range = process.range<wave_t> ();
  for (auto it = wave_range.begin (); it != wave_range.end ();)
    it = (it->queue () == *this) ? process.destroy (it) : ++it;

  auto &&workgroup_range = process.range<workgroup_t> ();
  for (auto it = workgroup_range.begin (); it != workgroup_range.end ();)
    it = (it->queue () == *this) ? process.destroy (it) : ++it;

  auto &&dispatch_range = process.range<dispatch_t> ();
  for (auto it = dispatch_range.begin (); it != dispatch_range.end ();)
    it = (it->queue () == *this) ? process.destroy (it) : ++it;

  /* Discard any cached data associated with the queue since the queue may have
     been deleted.  In that case, the memory being used for the context save
     area could be re-allocated for other purposes.  If that happens, we do not
     want to see stale data, or for stale dirty data to corrupt the future use.
     If the process is being detached, then there's really no need to discard
     since the process will be destructed and the cache destructed.  */

  const auto xcc_count = agent ().os_info ().xcc_count;

  agent ().memory_cache ().discard (
    m_os_queue_info.ctx_save_restore_address,
    xcc_count * m_os_queue_info.ctx_save_restore_area_size, true);
}

compute_queue_t::displaced_instruction_ptr_t
aql_queue_t::allocate_displaced_instruction (const instruction_t &instruction)
{
  dbgapi_assert (!process ().from_core ());

  if (!m_debugger_memory_base)
    {
      auto ctx_save_base = m_os_queue_info.ctx_save_restore_address;

      struct context_save_area_header_s header;
      agent ().read_agent_memory (ctx_save_base, &header);

      if (!header.debugger_memory_offset || !header.debugger_memory_size)
        fatal_error ("Per-queue memory reserved for the debugger is missing");

      /* Make sure the debugger memory is aligned so that it can be used to
         store displaced instructions, and that each chunk out of that memory
         register is also aligned.  */
      dbgapi_assert (
        utils::is_aligned (debugger_memory_chunk_size,
                           architecture ().minimum_instruction_alignment ()));

      m_debugger_memory_base.emplace (agent_address_t{
        utils::align_up (ctx_save_base + header.debugger_memory_offset,
                         debugger_memory_chunk_size) });

      auto chunk_count = ((ctx_save_base + header.debugger_memory_size
                           + header.debugger_memory_offset)
                          - *m_debugger_memory_base)
                         / debugger_memory_chunk_size;

      /* Ensure that the number of chunks does not overflow the 16 bit index.
       */
      if (chunk_count > std::numeric_limits<
            decltype (m_debugger_memory_chunk_count)>::max ())
        fatal_error ("Increase the width of m_debugger_memory_chunk_count");

      m_debugger_memory_chunk_count = chunk_count;

      m_debugger_memory_free_chunks.reserve (m_debugger_memory_chunk_count);
    }

  auto assert_instruction = architecture ().assert_instruction ();
  agent_address_t instruction_buffer_address;

  if (!m_debugger_memory_free_chunks.empty ())
    {
      auto index = m_debugger_memory_free_chunks.back ();
      m_debugger_memory_free_chunks.pop_back ();

      instruction_buffer_address
        = *m_debugger_memory_base + index * debugger_memory_chunk_size;
    }
  else if (m_debugger_memory_next_chunk < m_debugger_memory_chunk_count)
    {
      instruction_buffer_address
        = *m_debugger_memory_base
          + m_debugger_memory_next_chunk++ * debugger_memory_chunk_size;

      /* An instruction buffer is always terminated by a 'guard' instruction
         (assert_trap) so that the pc never points to an invalid instruction or
         unmapped memory if the instruction is single-stepped.  We use a trap
         instruction to prevent runaway waves from executing from unmapped
         memory.  */

      agent ().write_agent_memory (
        instruction_buffer_address + debugger_memory_chunk_size
          - assert_instruction.size (),
        assert_instruction.data (), assert_instruction.size ());
    }
  else
    fatal_error ("could not allocate debugger memory");

  auto deleter = [this] (agent_address_t ptr)
  {
    size_t index
      = (ptr - *m_debugger_memory_base) / debugger_memory_chunk_size;

    dbgapi_assert (index < m_debugger_memory_chunk_count);
    m_debugger_memory_free_chunks.emplace_back (static_cast<uint16_t> (index));
  };

  dbgapi_assert (instruction.size () + assert_instruction.size ()
                 <= debugger_memory_chunk_size);

  auto displaced_instruction_address
    = instruction_buffer_address + debugger_memory_chunk_size
      - assert_instruction.size () - instruction.size ();

  /* Make sure the new displaced instruction is properly aligned for this
     architecture.  */
  dbgapi_assert (
    utils::is_aligned (displaced_instruction_address,
                       architecture ().minimum_instruction_alignment ()));

  agent ().write_agent_memory (displaced_instruction_address,
                               instruction.data (), instruction.size ());

  return { displaced_instruction_address, deleter };
}

void
aql_queue_t::queue_state_changed ()
{
  const auto xcc_count = agent ().os_info ().xcc_count;

  switch (state ())
    {
    case state_t::running:
      /* Reset member variables that are undefined while the queue is in the
         running state.  */
      m_read_packet_id.reset ();
      m_write_packet_id.reset ();
      m_waves_running.reset ();

      /* The queue just changed state and is about to be placed back onto the
         hardware.  Write back dirty cache lines in the wave saved state
         region, but leave the cache lines valid so that accessing stopped
         waves' cached registers does not require a queue suspend/resume.  The
         saved state cache lines will be discarded when this queue is next
         suspended again (see the 'case state_t::suspended:' below).  */
      agent ().memory_cache ().write_back (
        m_os_queue_info.ctx_save_restore_address,
        xcc_count * m_os_queue_info.ctx_save_restore_area_size);
      break;

    case state_t::suspended:
      /* Discard the previously cached wave saved state lines.  The saved state
         areas may be mapped to a different address in this new context wave
         save.  */
      agent ().memory_cache ().discard (
        m_os_queue_info.ctx_save_restore_address,
        xcc_count * m_os_queue_info.ctx_save_restore_area_size);

      /* Refresh the scratch_backing_memory_location and
         scratch_backing_memory_size everytime the queue is suspended.

         The scratch backing memory address is stored in the ABI-stable part of
         the amd_queue_t. Since we know the address of the read_dispatch_id
         (obtained from the KFD through the queue snapshot info), which is also
         stored in the ABI-stable part of the amd_queue_t, we can calculate the
         address of the pointer to the scratch_backing_memory_location and read
         it.  We cannot cache this value as the runtime may change the
         allocation dynamically.  */

      process ().read_host_memory (
        m_os_queue_info.read_pointer_address
          + offsetof (amd_queue_t, scratch_backing_memory_location)
          - offsetof (amd_queue_t, read_dispatch_id),
        &m_scratch_backing_memory_address);

      process ().read_host_memory (
        m_os_queue_info.read_pointer_address
          + offsetof (amd_queue_t, compute_tmpring_size)
          - offsetof (amd_queue_t, read_dispatch_id),
        &m_compute_tmpring_size);

      /* Read the queue's write_packet_id and read_packet_id.  */

      process ().read_host_memory (m_os_queue_info.write_pointer_address,
                                   &m_write_packet_id.emplace ());

      process ().read_host_memory (m_os_queue_info.read_pointer_address,
                                   &m_read_packet_id.emplace ());

      /* Iterate the control stack and update/create waves that were saved in
         the last context wave save.  Waves that are no longer present will be
         destroyed.  */
      update_waves ();
      break;

    case state_t::invalid:
      break;
    }
}

std::optional<amd_dbgapi_os_queue_packet_id_t>
aql_queue_t::get_os_queue_packet_id (
  const architecture_t::cwsr_record_t &cwsr_record) const
{
  auto dispatch_packet_id = architecture ().dispatch_packet_id (cwsr_record);

  if (!dispatch_packet_id.has_value ())
    return std::nullopt;

  if ((dispatch_packet_id.value () * packet_size ()) >= size ())
    fatal_error ("dispatch_packet_index %#" PRIx64 " is out of bounds in %s",
                 dispatch_packet_id.value (), to_cstring (id ()));

  amd_dbgapi_global_address_t packet_address
    = address () + (dispatch_packet_id.value () * packet_size ());

  /* Calculate the monotonic dispatch id for this packet.  It is
       between read_packet_id and write_packet_id.  */

  const size_t ring_size = size () / aql_packet_size;

  /* The read_packet_id and write_packet_id should have been read when the
     queue was suspended and hold a value.  */
  dbgapi_assert (m_read_packet_id && m_write_packet_id);

  amd_dbgapi_os_queue_packet_id_t os_queue_packet_id
    = (packet_address - address ()) / aql_packet_size
      + (*m_read_packet_id / ring_size) * ring_size;

  if (os_queue_packet_id < *m_read_packet_id
      && (*m_read_packet_id % ring_size) >= (*m_write_packet_id % ring_size))
    os_queue_packet_id += ring_size;

  /* Check that the dispatch_id is between the command
     processor's read_id and write_id.  */
  if (os_queue_packet_id < *m_read_packet_id
      || os_queue_packet_id >= *m_write_packet_id)
    fatal_error ("os_queue_packet_id %#" PRIx64 " is not within "
                 "[%#" PRIx64 "..%#" PRIx64 "[ in %s",
                 os_queue_packet_id, *m_read_packet_id, *m_write_packet_id,
                 to_cstring (id ()));

  return os_queue_packet_id;
}

void
aql_queue_t::update_waves ()
{
  /* Value used to mark waves that are found in the context save area. When
     sweeping, any wave found with a mark less than the current mark will be
     deleted, as these waves are no longer active.  */
  const epoch_t wave_mark = wave_t::next_mark ();
  wave_t *group_leader = nullptr;

  auto process_cwsr_record
    = [this, wave_mark, &group_leader] (auto cwsr_record)
  {
    dbgapi_assert (*this == cwsr_record->queue ());
    process_t &process = cwsr_record->process ();

    auto prefetch_begin
      = cwsr_record->register_address (amdgpu_regnum_t::first_hwreg).value ();
    auto prefetch_end
      = cwsr_record->register_address (amdgpu_regnum_t::last_ttmp).value ()
        + architecture ().register_size (amdgpu_regnum_t::last_ttmp);

    dbgapi_assert (prefetch_end > prefetch_begin);
    cwsr_record->agent ().memory_cache ().prefetch (
      prefetch_begin, prefetch_end - prefetch_begin);

    wave_t *wave = nullptr;

    if (process.is_flag_set (process_t::flag_t::runtime_enable_during_attach))
      {
        /* Assign new ids to all waves regardless of the content of their
           wave_id register.  This is needed during attach as waves created
           before the debugger attached to the process may have stale
           wave_ids.  */
      }
    else if (amd_dbgapi_wave_id_t wave_id = cwsr_record->id ();
             wave_id != wave_t::undefined)
      {
        /* The wave already has a wave_id, so we should find it in this queue.
           Search all visible and invisible waves.  */
        wave = process.find (wave_id, true /* include invisible waves  */);

        if (!wave)
          {
            /* The wave_id saved in the ttmp registers may be corrupted.  */
            fatal_error ("%s not found in %s", to_cstring (wave_id),
                         to_cstring (id ()));
          }
      }

    if (!wave)
      {
        workgroup_t *workgroup = nullptr;

        if (group_leader)
          {
            /* We already have identified the workgroup_t this wave belongs to,
               it is the same workgroup_t as the thread group leader's.  */
            workgroup = &group_leader->workgroup ();

            if (workgroup->group_ids () != cwsr_record->group_ids ())
              fatal_error ("not in the same workgroup as the group_leader");
          }
        else
          {
            const auto packet_id = get_os_queue_packet_id (*cwsr_record);
            const auto group_ids = cwsr_record->group_ids ();

            /* Find the dispatch this wave belongs to using the packet_id.  The
               packet_id is only unique for a given queue.  */
            dispatch_t *dispatch = process.find_if (
              [this, packet_id] (const dispatch_t &d) {
                return d.queue () == *this
                       && d.os_queue_packet_id () == packet_id;
              });

            if (dispatch)
              {
                workgroup = process.find_if (
                  [dispatch, group_ids] (const workgroup_t &w) {
                    return w.dispatch () == *dispatch
                           && w.group_ids () == group_ids;
                  });
              }
            else if (packet_id)
              {
                dispatch = &process.create<aql_dispatch_t> (*this, *packet_id);
              }
            else
              {
                /* If this wave does not have a packet_id (ttmps are not setup
                   or may be corrupted), then create a new workgroup associated
                   with the dummy_dispatch.  All waves belonging to this
                   workgroup will be associated with this instance.  */
                dispatch = &m_dummy_dispatch;
              }

            if (!workgroup)
              workgroup = &process.create<workgroup_t> (
                *dispatch, group_ids, cwsr_record->lds_size ());
          }

        dbgapi_assert (workgroup != nullptr);
        wave = &process.create<wave_t> (*workgroup,
                                        cwsr_record->position_in_group ());
      }

    bool is_first_wave = cwsr_record->is_first_wave ();
    bool is_last_wave = cwsr_record->is_last_wave ();

    /* The first wave in the group is the group leader.  The group leader owns
       the backing store for the group memory (LDS).  */
    if (is_first_wave)
      {
        group_leader = wave;

        auto shared_memory_base_address
          = cwsr_record->register_address (amdgpu_regnum_t::lds_0);

        dbgapi_assert (shared_memory_base_address);
        group_leader->workgroup ().update (*shared_memory_base_address);
      }

    if (!group_leader)
      fatal_error ("No group_leader, the control stack may be corrupted");

    /* Update this wave's state using the context save area as it may have
       changed since the queue was last suspended (or the wave is new).  */
    wave->update (std::move (cwsr_record));

    if (wave->state () != AMD_DBGAPI_WAVE_STATE_STOP)
      ++*m_waves_running;

    /* Hide new waves halted at launch until the process' wave creation mode is
       changed to not stopped.  A wave is halted at launch if it is halted
       (status.halt=1) without having entered the trap handler, and its pc
       points to the kernel entry point.  */
    if (!wave->mark () /* A wave without a mark is a new wave that has not been
                          seen by queue_t::update_waves before.  */
        && wave->state () == AMD_DBGAPI_WAVE_STATE_RUN
        && wave->dispatch ().kernel_descriptor ().is_at_kernel_entry (
          global_address_t{wave->pc ()})
        && wave->is_halted ())
      {
        log_verbose ("%s is halted at launch", to_cstring (wave->id ()));

        wave->set_visibility (wave_t::visibility_t::hidden_halted_at_launch);
        wave->set_halted (false);
        wave->set_state (AMD_DBGAPI_WAVE_STATE_STOP);
      }

    /* This was the last wave in the group. Make sure we have a new group
       leader for the remaining waves.  */
    if (is_last_wave)
      group_leader = nullptr;

    wave->set_mark (wave_mark);
    if (is_first_wave)
      wave->workgroup ().set_mark (wave_mark);
  };

  process_t &process = this->process ();
  size_t wave_count = 0; /* Number of waves seen in the context save area.  */

  /* Start with 0 running waves.  When iterating the control stack (below)
     each discovered wave in the running state will increment this count.  */
  m_waves_running.emplace (0);

  for (uint32_t xcc_id = 0; xcc_id < agent ().os_info ().xcc_count; ++xcc_id)
    {
      auto ctx_save_address
        = m_os_queue_info.ctx_save_restore_address
          + xcc_id * m_os_queue_info.ctx_save_restore_area_size;

      /* Retrieve the control stack and wave save area memory locations.  */
      context_save_area_header_s header;
      agent ().read_agent_memory (ctx_save_address, &header);

      auto control_stack_begin
        = ctx_save_address + header.control_stack_offset;
      auto control_stack_end = control_stack_begin + header.control_stack_size;
      auto wave_area_end = ctx_save_address + header.wave_state_offset;
      auto wave_area_begin = wave_area_end - header.wave_state_size;

      /* The control stack and the wave save area should be contiguous.  */
      if (control_stack_end != wave_area_begin)
        fatal_error ("corrupted context save area header");

      if (control_stack_begin != control_stack_end)
        {
          log_info ("decoding %s's context save area #%u: "
                    "ctrl_stk:[%s..%s[, wave_area:[%s..%s[",
                    to_cstring (id ()), xcc_id,
                    to_cstring (control_stack_begin),
                    to_cstring (control_stack_end),
                    to_cstring (wave_area_begin), to_cstring (wave_area_end));

          /* Read the entire control stack from the inferior in one go.  */
          amd_dbgapi_size_t size = control_stack_end - control_stack_begin;
          if (!utils::is_aligned (size, sizeof (uint32_t)))
            fatal_error ("corrupted control stack");

          auto memory
            = std::make_unique<uint32_t[]> (size / sizeof (uint32_t));
          agent ().read_agent_memory (control_stack_begin, &memory[0], size);

          /* Decode the control stack.  For each entry in the control stack,
             the provided callback function is called with a CWSR record.  */
          wave_count += architecture ().control_stack_iterate (
            *this, xcc_id, &memory[0], size / sizeof (uint32_t), wave_area_end,
            wave_area_end - wave_area_begin, process_cwsr_record);
        }
    }

  if (wave_count)
    log_info ("%zu out of %zu wave%s running on %s", *m_waves_running,
              wave_count, wave_count > 1 ? "s" : "", to_cstring (id ()));

  /* Iterate all waves, workgroups and dispatches belonging to this queue, and
     prune waves and workgroups with a mark older than the current mark, and
     dispatches with ids older (smaller) than the queue current read dispatch
     id.

     Note that the waves must be pruned before the workgroups and the
     workgroups must be pruned before the dispatches to ensure there are no
     dangling pointers to pruned objects.  */

  auto &&wave_range = process.range<wave_t> ();
  for (auto it = wave_range.begin (); it != wave_range.end ();)
    if (it->queue () == *this && it->mark () < wave_mark)
      {
        dbgapi_assert (it->state () != AMD_DBGAPI_WAVE_STATE_STOP
                       && "a stopped wave cannot terminate");
        it = process.destroy (it);
      }
    else
      ++it;

  auto &&workgroup_range = process.range<workgroup_t> ();
  for (auto it = workgroup_range.begin (); it != workgroup_range.end ();)
    if (it->queue () == *this && it->mark () < wave_mark)
      it = process.destroy (it);
    else
      ++it;

  auto &&dispatch_range = process.range<dispatch_t> ();
  for (auto it = dispatch_range.begin (); it != dispatch_range.end ();)
    if (it->queue () == *this && it->os_queue_packet_id () < m_read_packet_id)
      it = process.destroy (it);
    else
      ++it;
}

std::pair<agent_address_t /* address */, amd_dbgapi_size_t /* size */>
aql_queue_t::scratch_memory_region (
  const architecture_t::cwsr_record_t &cwsr_record) const
{
  auto [offset, size] = architecture ().scratch_memory_region (
    agent (), m_compute_tmpring_size, cwsr_record);

  return { m_scratch_backing_memory_address + offset, size };
}

void
aql_queue_t::active_packets_info (
  amd_dbgapi_os_queue_packet_id_t *read_packet_id_p,
  amd_dbgapi_os_queue_packet_id_t *write_packet_id_p,
  size_t *packets_byte_size_p) const
{
  dbgapi_assert (is_suspended ());

  amd_dbgapi_os_queue_packet_id_t read_packet_id;
  process ().read_host_memory (m_os_queue_info.read_pointer_address,
                               &read_packet_id);

  amd_dbgapi_os_queue_packet_id_t write_packet_id;
  process ().read_host_memory (m_os_queue_info.write_pointer_address,
                               &write_packet_id);

  if (read_packet_id > write_packet_id)
    fatal_error ("corrupted read/write packet ids");

  *read_packet_id_p = read_packet_id;
  *write_packet_id_p = write_packet_id;
  *packets_byte_size_p = (write_packet_id - read_packet_id) * aql_packet_size;
}

void
aql_queue_t::active_packets_bytes (
  amd_dbgapi_os_queue_packet_id_t read_packet_id,
  amd_dbgapi_os_queue_packet_id_t write_packet_id, void *memory,
  size_t memory_size) const
{
  dbgapi_assert (is_suspended ());

  if (read_packet_id > write_packet_id)
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  amd_dbgapi_size_t packets_byte_size
    = (write_packet_id - read_packet_id) * aql_packet_size;

  if (memory_size != packets_byte_size)
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  /* size must be a power of 2.  */
  if (!utils::is_power_of_two (size ()))
    fatal_error ("size is not a power of 2");

  const uint64_t id_mask = size () / aql_packet_size - 1;

  host_address_t read_packet_ptr
    = address () + (read_packet_id & id_mask) * aql_packet_size;
  host_address_t write_packet_ptr
    = address () + (write_packet_id & id_mask) * aql_packet_size;

  if (read_packet_ptr < write_packet_ptr)
    process ().read_host_memory (read_packet_ptr, memory, packets_byte_size);

  else if (read_packet_ptr > write_packet_ptr)
    {
      size_t first_part_size = address () + size () - read_packet_ptr;

      process ().read_host_memory (read_packet_ptr, memory, first_part_size);

      size_t second_part_size = write_packet_ptr - address ();

      process ().read_host_memory (
        address (), static_cast<char *> (memory) + first_part_size,
        second_part_size);
    }
}

class unsupported_queue_t : public queue_t
{
public:
  unsupported_queue_t (amd_dbgapi_queue_id_t queue_id, const agent_t &agent,
                       const os_queue_snapshot_entry_t &os_queue_info)
    : queue_t (queue_id, agent, os_queue_info)
  {
  }

  amd_dbgapi_os_queue_type_t type () const override;

  size_t packet_size () const override { return 1; };

  void active_packets_info (amd_dbgapi_os_queue_packet_id_t *read_packet_id_p,
                            amd_dbgapi_os_queue_packet_id_t *write_packet_id_p,
                            size_t *packets_byte_size_p) const override;

  void active_packets_bytes (amd_dbgapi_os_queue_packet_id_t read_packet_id,
                             amd_dbgapi_os_queue_packet_id_t write_packet_id,
                             void *memory, size_t memory_size) const override;
};

amd_dbgapi_os_queue_type_t
unsupported_queue_t::type () const
{
  switch (m_os_queue_info.queue_type)
    {
    case os_queue_type_t::compute:
      return AMD_DBGAPI_OS_QUEUE_TYPE_AMD_PM4;

    case os_queue_type_t::sdma:
      return AMD_DBGAPI_OS_QUEUE_TYPE_AMD_SDMA;

    case os_queue_type_t::sdma_xgmi:
      return AMD_DBGAPI_OS_QUEUE_TYPE_AMD_SDMA_XGMI;

    case os_queue_type_t::compute_aql:
      fatal_error ("should not reach here");

    default:
      return AMD_DBGAPI_OS_QUEUE_TYPE_UNKNOWN;
    }
}

void
unsupported_queue_t::active_packets_info (
  amd_dbgapi_os_queue_packet_id_t * /* read_packet_id_p  */,
  amd_dbgapi_os_queue_packet_id_t * /* write_packet_id_p  */,
  size_t * /* packets_byte_size_p  */) const
{
  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED);
}

void
unsupported_queue_t::active_packets_bytes (
  amd_dbgapi_os_queue_packet_id_t /* read_packet_id  */,
  amd_dbgapi_os_queue_packet_id_t /* write_packet_id  */, void * /* memory  */,
  size_t /* memory_size  */) const
{
  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED);
}

} /* namespace detail */

bool
compute_queue_t::is_all_stopped () const
{
  return process ().wave_launch_mode () == os_wave_launch_mode_t::halt
         && m_waves_running && *m_waves_running == 0;
}

void
compute_queue_t::wave_state_changed (const wave_t &wave)
{
  dbgapi_assert (m_waves_running);

  if (wave.state () == AMD_DBGAPI_WAVE_STATE_STOP)
    {
      dbgapi_assert (*m_waves_running > 0);
      --*m_waves_running;
    }
  else
    ++*m_waves_running;
}

queue_t &
queue_t::create (std::optional<amd_dbgapi_queue_id_t> queue_id,
                 const agent_t &agent,
                 const os_queue_snapshot_entry_t &os_queue_info)
{
  switch (os_queue_info.queue_type)
    {
    case os_queue_type_t::compute_aql:
      return agent.process ().create<detail::aql_queue_t> (queue_id, agent,
                                                           os_queue_info);

    default:
      return agent.process ().create<detail::unsupported_queue_t> (
        queue_id, agent, os_queue_info);
    }
}

std::optional<os_queue_id_t>
queue_t::os_queue_id () const
{
  if (is_valid ())
    return m_os_queue_info.queue_id;
  else
    return {};
}

void
queue_t::set_state (state_t state)
{
  if (m_state == state)
    return; /* State is unchanged.  */

  dbgapi_assert (m_state != state_t::invalid
                 && "an invalid queue cannot change state");

  m_state = state;

  /* queue_t::set_state should not throw exceptions, if the process has exited,
     mark the queue as invalid.  */
  try
    {
      if (!is_all_stopped ())
        queue_state_changed ();
    }
  catch (const process_exited_exception_t &)
    {
      m_state = state_t::invalid;
    }

  if (m_state == state_t::invalid)
    log_info ("invalidated %s", to_cstring (id ()));
}

host_address_t
queue_t::address () const
{
  return m_os_queue_info.ring_base_address;
}

amd_dbgapi_size_t
queue_t::size () const
{
  return m_os_queue_info.ring_size;
}

void
queue_t::get_info (amd_dbgapi_queue_info_t query, size_t value_size,
                   void *value) const
{
  /* The caller of this method (amd_dbgapi_queue_get_info) must ensure
     that the queue is valid.  Since the queue is retreived using
     amd::dbgapi::handle_object_set_t<amd::dbgapi::queue_t>::find, any invalid
     queue would have been filtered at this stage.  */
  dbgapi_assert (is_valid ());

  switch (query)
    {
    case AMD_DBGAPI_QUEUE_INFO_AGENT:
      utils::get_info (value_size, value, agent ().id ());
      return;

    case AMD_DBGAPI_QUEUE_INFO_PROCESS:
      utils::get_info (value_size, value, process ().id ());
      return;

    case AMD_DBGAPI_QUEUE_INFO_ARCHITECTURE:
      utils::get_info (value_size, value, architecture ().id ());
      return;

    case AMD_DBGAPI_QUEUE_INFO_TYPE:
      utils::get_info (value_size, value, type ());
      return;

    case AMD_DBGAPI_QUEUE_INFO_OS_ID:
      {
        std::optional<os_queue_id_t> os_id = os_queue_id ();
        dbgapi_assert (os_id.has_value ());
        utils::get_info (
          value_size, value,
          static_cast<amd_dbgapi_os_queue_id_t> (os_id.value ()));
        return;
      }

    case AMD_DBGAPI_QUEUE_INFO_ADDRESS:
      utils::get_info (value_size, value, address ());
      return;

    case AMD_DBGAPI_QUEUE_INFO_SIZE:
      utils::get_info (value_size, value, size ());
      return;

    case AMD_DBGAPI_QUEUE_INFO_STATE:
    case AMD_DBGAPI_QUEUE_INFO_ERROR_REASON:
      throw api_error_t (AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED);
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

scoped_queue_suspend_t::scoped_queue_suspend_t (queue_t &queue,
                                                const char *reason)
  : m_reason (reason), m_queue (!queue.is_suspended () ? &queue : nullptr)
{
  if (m_queue == nullptr)
    return;

  if (m_queue->process ().suspend_queues ({ m_queue }, m_reason) != 1)
    {
      if (m_queue->is_valid ())
        fatal_error ("process::suspend_queues failed");

      /* The queue became invalid, we should not try to resume it.  */
      m_queue = nullptr;
    }
}

scoped_queue_suspend_t::~scoped_queue_suspend_t ()
{
  if (m_queue == nullptr /* scoped_queue_suspend instance did not suspend the
                            queue. */
      || !m_queue->process ().forward_progress_needed ())
    return;

  if (m_queue->process ().resume_queues ({ m_queue }, m_reason) != 1
      && m_queue->is_valid ())
    fatal_error ("process::resume_queues failed");
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_queue_get_info (amd_dbgapi_queue_id_t queue_id,
                           amd_dbgapi_queue_info_t query, size_t value_size,
                           void *value)
{
  TRACE_BEGIN (param_in (queue_id), param_in (query), param_in (value_size),
               param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    queue_t *queue = find (queue_id);

    if (queue == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_QUEUE_ID);

    queue->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_QUEUE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_process_queue_list (amd_dbgapi_process_id_t process_id,
                               size_t *queue_count,
                               amd_dbgapi_queue_id_t **queues,
                               amd_dbgapi_changed_t *changed)
{
  TRACE_BEGIN (param_in (process_id), param_in (queue_count),
               param_in (queues), param_in (changed));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    std::vector<process_t *> processes = process_t::match (process_id);

    if (queues == nullptr || queue_count == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    for (auto &&process : processes)
      process->update_queues ();

    std::tie (*queues, *queue_count)
      = utils::get_handle_list<queue_t> (processes, changed);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_ref (param_out (queue_count)),
             make_ref (make_ref (param_out (queues)), *queue_count),
             make_ref (param_out (changed)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_queue_packet_list (
  amd_dbgapi_queue_id_t queue_id,
  amd_dbgapi_os_queue_packet_id_t *read_packet_id_p,
  amd_dbgapi_os_queue_packet_id_t *write_packet_id_p,
  amd_dbgapi_size_t *packets_byte_size_p, void **packets_bytes_p)
{
  TRACE_BEGIN (param_in (queue_id), param_in (read_packet_id_p),
               param_in (write_packet_id_p), param_in (packets_byte_size_p),
               param_in (packets_bytes_p));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    if (read_packet_id_p == nullptr || write_packet_id_p == nullptr
        || packets_byte_size_p == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    queue_t *queue = find (queue_id);

    if (queue == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_QUEUE_ID);

    scoped_queue_suspend_t suspend (*queue, "refresh packet list");

    amd_dbgapi_os_queue_packet_id_t read_packet_id, write_packet_id;
    size_t memory_size;

    queue->active_packets_info (&read_packet_id, &write_packet_id,
                                &memory_size);

    if (packets_bytes_p != nullptr)
      {
        auto memory = allocate_memory (memory_size);

        queue->active_packets_bytes (read_packet_id, write_packet_id,
                                     memory.get (), memory_size);

        *packets_bytes_p = memory.release ();
      }

    *read_packet_id_p = read_packet_id;
    *write_packet_id_p = write_packet_id;
    *packets_byte_size_p = memory_size;
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED, AMD_DBGAPI_STATUS_ERROR,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_hex (make_ref (param_out (read_packet_id_p))),
             make_hex (make_ref (param_out (write_packet_id_p))),
             make_ref (param_out (packets_byte_size_p)),
             make_hex (make_ref (make_ref (param_out (packets_bytes_p)),
                                 *packets_byte_size_p)));
}
