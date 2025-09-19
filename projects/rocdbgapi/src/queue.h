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

#ifndef AMD_DBGAPI_QUEUE_H
#define AMD_DBGAPI_QUEUE_H 1

#include "amd-dbgapi.h"
#include "architecture.h"
#include "debug.h"
#include "dispatch.h"
#include "handle_object.h"
#include "memory.h"
#include "os_driver.h"
#include "utils.h"
#include "wave.h"
#include "workgroup.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>

namespace amd::dbgapi
{

class agent_t;
class process_t;
class wave_t;

/* AMD Debugger API Queue.  */

class queue_t : public detail::handle_object<amd_dbgapi_queue_id_t>
{
public:
  enum class state_t
  {
    invalid,   /* The queue is invalid. Calls to os_queue_id () will return an
                  empty optional.  Calls to process_t::find and
                  process_t::find_if will not return this queue.  Once a queue
                  becomes invalid, its state can no longer be changed.  */
    suspended, /* The queue is suspended, its state can be inspected.  */
    running    /* The queue is running.  */
  };

protected:
  os_queue_snapshot_entry_t m_os_queue_info;

private:
  state_t m_state{ state_t::running };
  epoch_t m_mark{ 0 };

  const agent_t &m_agent;

  /* Called whenever the queue changes state.  */
  virtual void queue_state_changed () {}

public:
  static queue_t &create (std::optional<amd_dbgapi_queue_id_t> queue_id,
                          const agent_t &agent,
                          const os_queue_snapshot_entry_t &os_queue_info);

  queue_t (amd_dbgapi_queue_id_t queue_id, const agent_t &agent,
           const os_queue_snapshot_entry_t &os_queue_info)
    : handle_object (queue_id), m_os_queue_info (os_queue_info),
      m_agent (agent)
  {
  }

  virtual ~queue_t () = default;

  void update (const os_queue_snapshot_entry_t &queue_info)
  {
    m_os_queue_info = queue_info;
  }

  /* Return the queue's type.  */
  virtual amd_dbgapi_os_queue_type_t type () const = 0;

  state_t state () const { return m_state; }
  void set_state (state_t state);

  bool is_valid () const { return m_state != state_t::invalid; }
  bool is_suspended () const { return m_state == state_t::suspended; }
  bool is_running () const { return m_state == state_t::running; }

  std::optional<os_queue_id_t> os_queue_id () const;

  static epoch_t next_mark ()
  {
    static monotonic_counter_t<epoch_t, 1> next_queue_mark{};
    return next_queue_mark ();
  }
  epoch_t mark () const { return m_mark; }
  void set_mark (epoch_t mark) { m_mark = mark; }

  /* Return the address of the memory holding the queue packets.  */
  std::variant<host_address_t, agent_address_t> address () const;

  /* Return the size of the memory holding the queue packets.  */
  amd_dbgapi_size_t size () const;

  /* Return the byte size of a packet in this queue.  */
  virtual size_t packet_size () const = 0;

  /* Return true if the queue does not have any visible activity.  */
  virtual bool is_all_stopped () const { return false; }

  virtual void
  active_packets_info (amd_dbgapi_os_queue_packet_id_t *read_packet_id_p,
                       amd_dbgapi_os_queue_packet_id_t *write_packet_id_p,
                       size_t *packets_byte_size_p) const
    = 0;

  virtual void
  active_packets_bytes (amd_dbgapi_os_queue_packet_id_t read_packet_id,
                        amd_dbgapi_os_queue_packet_id_t write_packet_id,
                        void *memory, size_t memory_size) const
    = 0;

  void get_info (amd_dbgapi_queue_info_t query, size_t value_size,
                 void *value) const;

  const agent_t &agent () const { return m_agent; }
  process_t &process () const;
  const architecture_t &architecture () const;
};

/* Interface implemented by all compute queues.  */

class compute_queue_t : public queue_t
{
public:
  /* A displaced instruction ptr holds the address of an instruction in
     device accessible memory.  The address is returned to the queue when the
     displaced instruction ptr is destructed.  */
  using displaced_instruction_ptr_t
    = utils::unique_resource_t<agent_address_t,
                               std::function<void (agent_address_t)>>;

protected:
  struct context_save_area_header_s
  {
    uint32_t control_stack_offset;
    uint32_t control_stack_size;
    uint32_t wave_state_offset;
    uint32_t wave_state_size;
    uint32_t debugger_memory_offset;
    uint32_t debugger_memory_size;
  };

  class dummy_dispatch_t : public dispatch_t
  {
  private:
    class dummy_descriptor_t : public architecture_t::kernel_descriptor_t
    {
    public:
      dummy_descriptor_t (process_t &process)
        : architecture_t::kernel_descriptor_t (process, 0)
      {
      }
      global_address_t entry_address () const override { return 0; }
      bool is_at_kernel_entry (global_address_t /* pc  */) const override
      {
        return false;
      }
    } m_dummy_descriptor;

  public:
    dummy_dispatch_t (compute_queue_t &queue)
      : dispatch_t (AMD_DBGAPI_DISPATCH_NONE, queue, 0),
        m_dummy_descriptor (queue.process ())
    {
    }

    const architecture_t::kernel_descriptor_t &
    kernel_descriptor () const override
    {
      return m_dummy_descriptor;
    }

    void get_info (amd_dbgapi_dispatch_info_t /* query  */,
                   size_t /* value_size  */,
                   void * /* value  */) const override
    {
      dbgapi_assert_not_reached ("should not call this");
    }
  } m_dummy_dispatch;

  /* Number of waves in the running state.  Only holds a value when the queue
     is suspended.  */
  std::optional<size_t> m_waves_running{};

  static constexpr amd_dbgapi_size_t debugger_memory_chunk_size = 32;

  uint16_t m_debugger_memory_chunk_count{ 0 };
  uint16_t m_debugger_memory_next_chunk{ 0 };
  std::vector<uint16_t> m_debugger_memory_free_chunks{};

  std::optional<displaced_instruction_ptr_t> m_park_instruction_ptr{};
  std::optional<displaced_instruction_ptr_t> m_terminating_instruction_ptr{};

  /* The memory reserved by the thunk library for the debugger is used to store
     instruction buffers.  Instruction buffers are lazily allocated from the
     reserved memory, and when freed, their index is returned to a free list.
     Each wave is guaranteed its own unique instruction buffer.  */
  std::optional<agent_address_t> m_debugger_memory_base{};

  std::optional<amd_dbgapi_os_queue_packet_id_t> m_read_packet_id{};
  std::optional<amd_dbgapi_os_queue_packet_id_t> m_write_packet_id{};
  uint32_t m_compute_tmpring_size{ 0 };

  /* Called whenever the queue changes state.  */
  virtual void queue_state_changed () override;

  void update_waves ();

  virtual dispatch_t *
  create_dispatch (amd_dbgapi_os_queue_packet_id_t packet_id);
  virtual std::optional<amd_dbgapi_os_queue_packet_id_t>
  get_os_queue_packet_id (
    const architecture_t::cwsr_record_t & /* cwsr_record  */) const
  {
    /* Raw compute queue do not have have link to the dispatch packets.  */
    return {};
  }

  /* TODO lsix: make that better.  */
  virtual void refresh_scratch_on_suspend ();

public:
  compute_queue_t (amd_dbgapi_queue_id_t queue_id, const agent_t &agent,
                   const os_queue_snapshot_entry_t &os_queue_info)
    : queue_t (queue_id, agent, os_queue_info), m_dummy_dispatch (*this)
  {
  }

  void wave_state_changed (const wave_t &wave);

  bool is_all_stopped () const override;

  /* Return the address of a park instruction.  */
  virtual agent_address_t park_instruction_address ();
  /* Return the address of a terminating instruction.  */
  virtual agent_address_t terminating_instruction_address ();

  /* Return the wave's scratch memory region (address and size).  */
  virtual std::pair<agent_address_t /* address */,
                    amd_dbgapi_size_t /* size */>
  scratch_memory_region (
    const architecture_t::cwsr_record_t &cwsr_record) const;

  /* Return a pointer to device accessible memory containing the given
     instruction bytes.  */
  virtual displaced_instruction_ptr_t
  allocate_displaced_instruction (const instruction_t &instruction);

  amd_dbgapi_os_queue_type_t type () const override
  {
    return AMD_DBGAPI_OS_QUEUE_TYPE_AMD_PM4;
  }

  size_t packet_size () const override { return 0; }

  void active_packets_info (amd_dbgapi_os_queue_packet_id_t *,
                            amd_dbgapi_os_queue_packet_id_t *,
                            size_t *) const override
  {
  }

  void active_packets_bytes (amd_dbgapi_os_queue_packet_id_t,
                             amd_dbgapi_os_queue_packet_id_t, void *,
                             size_t) const override
  {
  }
};

/* Wraps a queue and provides a RAII mechanism to suspend it if it wasn't
   already suspended. The queue is suspended when the object is constructed,
   if the queue is not invalid or not already suspended.  When control leaves
   the scope in which the object was created, the queue is resumed if it was
   suspended by this instance of scoped_queue_suspend_t, the queue is still
   valid, and forward progress is required".
 */
class scoped_queue_suspend_t
{
public:
  scoped_queue_suspend_t (queue_t &queue, const char *reason);
  ~scoped_queue_suspend_t ();

  /* Disable copies.  */
  scoped_queue_suspend_t (const scoped_queue_suspend_t &) = delete;
  scoped_queue_suspend_t &operator= (const scoped_queue_suspend_t &) = delete;

public:
  const char *const m_reason;
  queue_t *m_queue;
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_QUEUE_H */
