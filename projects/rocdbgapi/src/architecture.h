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

#ifndef AMD_DBGAPI_ARCHITECTURE_H
#define AMD_DBGAPI_ARCHITECTURE_H 1

#include "amd-dbgapi.h"

#include "agent.h"
#include "handle_object.h"
#include "memory.h"
#include "os_driver.h"
#include "register.h"
#include "runtime_rdebug.h"
#include "utils.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <amd_comgr/amd_comgr.h>

namespace amd::dbgapi
{

class architecture_t;
class process_t;
class compute_queue_t;
class wave_t;

namespace detail
{

/* The architecture that contained the last successful find result.  The next
   global find will start searching here.

   FIXME: If multi-process support is added this mechanism will need
   re-implementing to be thread safe.  */
extern const architecture_t *last_found_architecture;

} /* namespace detail */

class architecture_t;

struct legal_instruction_t
{
  explicit legal_instruction_t () = default;
};
inline constexpr legal_instruction_t legal_instruction{};

/* Holds one instruction's words. A word is the smallest type on which the
   instruction is aligned.  The ::instruction_t is associated with an
   architecture, and can be validated for that architecture.  Once validated,
   the exact instruction size is known if it is a valid encoding for that
   architecture.  */
class instruction_t
{
private:
  std::vector<std::byte> m_bytes;
  mutable std::optional<size_t> m_size{};
  std::reference_wrapper<const architecture_t> m_architecture;

public:
  instruction_t (const architecture_t &architecture,
                 std::vector<std::byte> bytes)
    : m_bytes (std::move (bytes)), m_architecture (architecture)
  {
  }
  instruction_t (legal_instruction_t, const architecture_t &architecture,
                 std::vector<std::byte> bytes)
    : m_bytes (std::move (bytes)), m_architecture (architecture)
  {
    /* The instruction is guaranteed to be valid, and its byte size is exactly
       that of the bytes vector passed in.  */
    m_size.emplace (m_bytes.size ());
  }

  /* The number of bytes reserved in the instruction bytes storage.  Not all
     bytes in the storage belong to the instruction, the instruction could have
     been created from memory for the largest instruction byte size the
     architecture supports.  */
  size_t capacity () const { return m_bytes.size (); }

  /* The instruction size in bytes, or 0 if the instruction is invalid.  An
     instruction is invalid if the architecture's disassembler does not
     recognize the instruction, or if the instruction's encoding is known to be
     invalid (for example, misaligned register pair index).  */
  size_t size () const;

  /* Return a pointer to the instruction bytes.  */
  const void *data () const { return m_bytes.data (); }

  /* A valid instruction has a non-zero size.  */
  bool is_valid () const { return size () != 0; }

  /* Return the Nth instruction word.  */
  template <size_t pos> uint32_t word () const
  {
    dbgapi_assert (capacity () >= sizeof (uint32_t[pos + 1]));
    return *std::launder (
      reinterpret_cast<const uint32_t *> (&m_bytes[pos * sizeof (uint32_t)]));
  }
};

/* Architecture.  */

class architecture_t : private utils::not_copyable_t
{
  static_assert (is_handle_type_v<amd_dbgapi_architecture_id_t>,
                 "amd_dbgapi_architecture_id_t is not a handle type");

private:
  static monotonic_counter_t<
    uint32_t, monotonic_counter_start_v<amd_dbgapi_architecture_id_t>>
    s_next_architecture_id;

  static std::unordered_map<amd_dbgapi_architecture_id_t,
                            std::unique_ptr<const architecture_t>,
                            hash<amd_dbgapi_architecture_id_t>>
    s_architecture_map;

  amd_dbgapi_architecture_id_t const m_architecture_id;

  elf_amdgpu_machine_t const m_e_machine;
  std::string const m_target_triple;

  std::tuple<handle_object_set_t<address_space_t>,
             handle_object_set_t<address_class_t>,
             handle_object_set_t<register_class_t>>
    m_handle_object_sets{};

protected:
  architecture_t (elf_amdgpu_machine_t e_machine, std::string target_triple);

public:
  /* Return the map of all instantiated architectures  */
  static const auto &all () { return s_architecture_map; }

  class kernel_descriptor_t
  {
  private:
    global_address_t const m_address;
    process_t &m_process;

  public:
    kernel_descriptor_t (process_t &process, global_address_t address)
      : m_address (address), m_process (process)
    {
    }
    virtual ~kernel_descriptor_t () = default;

    virtual global_address_t entry_address () const = 0;
    virtual bool is_at_kernel_entry (global_address_t pc) const = 0;

    global_address_t address () const { return m_address; }
    process_t &process () const { return m_process; }
  };

  class cwsr_record_t
  {
  private:
    const compute_queue_t &m_queue;
    uint32_t m_xcc_id;

  public:
    cwsr_record_t (const compute_queue_t &queue, uint32_t xcc_id)
      : m_queue (queue), m_xcc_id (xcc_id)
    {
    }
    /* cwsr_record_t is a polymorphic base class.  */
    virtual ~cwsr_record_t () = default;

    /* Indicates if content of TTMP registers which should be initialized by
       SPI are meaningful or cleared by dbgapi and should be ignored.  */
    virtual bool spi_ttmps_setup_enabled () const = 0;
    /* Return the globally unique wave identifier.  */
    virtual amd_dbgapi_wave_id_t id () const = 0;
    /* The 3-dimensional workgroup coordinates.  */
    virtual std::optional<std::array<uint32_t, 3>> group_ids () const = 0;
    /* Return the record's position in the workgroup.  */
    virtual std::optional<uint32_t> position_in_group () const = 0;

    /* Number of work-items in one wave.  */
    virtual size_t lane_count () const = 0;
    /* Last wave of threadgroup.  */
    virtual bool is_last_wave () const = 0;
    /* First wave of threadgroup.  */
    virtual bool is_first_wave () const = 0;

    /* Size of the local data share.  */
    virtual size_t lds_size () const = 0;

    virtual std::optional<agent_address_t>
    register_address (amdgpu_regnum_t regnum) const = 0;

    /* Return true if a scratch slot is allocated for this record.  */
    virtual bool is_scratch_enabled () const = 0;

    /* The shader engine ID this wave was created on.  */
    virtual uint32_t shader_engine_id () const = 0;
    /* Scratch region slot ID.  */
    virtual uint32_t scratch_scoreboard_id () const = 0;

    /* The address of the first byte in the wave's context save.  */
    virtual agent_address_t begin () const = 0;
    /* The address of the byte following the last byte in the context save. */
    virtual agent_address_t end () const = 0;

    uint32_t xcc_id () const { return m_xcc_id; }

    const compute_queue_t &queue () const { return m_queue; }
    const agent_t &agent () const;
    process_t &process () const;
    const architecture_t &architecture () const;
  };

  virtual ~architecture_t ();

  std::string name () const;

  /* Since architecture objects disallow copying & moving, two architecture
     objects are identical if they have the same address.  */
  bool operator== (const architecture_t &other) const
  {
    return this == &other;
  }
  bool operator!= (const architecture_t &other) const
  {
    return this != &other;
  }

  virtual size_t control_stack_iterate (
    compute_queue_t &queue, uint32_t xcc_id, const uint32_t *control_stack,
    size_t control_stack_words, agent_address_t wave_area_address,
    amd_dbgapi_size_t wave_area_size,
    const std::function<void (std::unique_ptr<const cwsr_record_t>)>
      &wave_callback) const
    = 0;

  virtual std::optional<uint64_t>
  dispatch_packet_id (const architecture_t::cwsr_record_t &cwsr_record) const
    = 0;

  /* Default initialize the trap temporary registers normally set up by SPI.
   */
  virtual void initialize_spi_ttmps (const wave_t &wave) const = 0;

  virtual void record_spi_ttmps_setup (const wave_t &wave, bool enabled) const
    = 0;

  /* Return true if the trap temporary registers used by the trap handler to
     communicate with the debugger API have been initialized.  */
  virtual bool are_trap_handler_ttmps_initialized (const wave_t &wave) const
    = 0;
  /* Default initialize the trap temporary registers normally set up by the
     trap handler.  */
  virtual void initialize_trap_handler_ttmps (const wave_t &wave) const = 0;

  virtual size_t maximum_queue_packet_count () const = 0;

  virtual std::unique_ptr<const kernel_descriptor_t>
  make_kernel_descriptor (process_t &process,
                          global_address_t kernel_descriptor_address) const
    = 0;

  virtual std::pair<amd_dbgapi_size_t /* offset  */,
                    amd_dbgapi_size_t /* size  */>
  scratch_memory_region (const agent_t &,
                         uint32_t compute_tmpring_size_register,
                         const cwsr_record_t &cwsr_record) const
    = 0;

  virtual bool
  is_address_space_supported (const address_space_t &address_space) const
    = 0;

  virtual bool
  is_address_class_supported (const address_class_t &address_class) const
    = 0;

  /* Return the watchpoints for which an exception was generated in the given
     stopped wave.  */
  virtual std::vector<os_watch_id_t>
  triggered_watchpoints (const wave_t &wave) const = 0;

  virtual size_t largest_instruction_size () const = 0;
  virtual size_t minimum_instruction_alignment () const = 0;
  virtual size_t breakpoint_instruction_pc_adjust () const = 0;
  virtual instruction_t breakpoint_instruction () const = 0;
  virtual instruction_t assert_instruction () const = 0;
  virtual instruction_t debug_trap_instruction () const = 0;
  virtual instruction_t terminating_instruction () const = 0;

  virtual bool
  is_terminating_instruction (const instruction_t &instruction) const
    = 0;

  virtual bool can_execute_displaced (wave_t &wave,
                                      const instruction_t &instruction) const
    = 0;
  virtual bool can_simulate (wave_t &wave,
                             const instruction_t &instruction) const
    = 0;

  virtual bool simulate (wave_t &wave, agent_address_t pc,
                         const instruction_t &instruction) const
    = 0;

  /* Report that a given ABI version is unreliable for the current
     architecture.  */
  virtual bool check_runtime_abi_version (runtime_rdebug_version_t) const = 0;

  virtual bool park_stopped_waves (runtime_rdebug_version_t) const = 0;
  virtual void save_pc_for_park (const wave_t &wave, agent_address_t pc) const
    = 0;
  virtual agent_address_t saved_parked_pc (const wave_t &wave) const = 0;

  virtual bool has_architected_flat_scratch () const = 0;

  virtual amd_dbgapi_size_t
  instruction_size (const instruction_t &instruction) const
    = 0;

  virtual std::tuple<amd_dbgapi_instruction_kind_t /* instruction_kind  */,
                     amd_dbgapi_instruction_properties_t /* properties  */,
                     size_t /* instruction_size  */,
                     std::vector<uint64_t> /* information  */>
  classify_instruction (agent_address_t address,
                        const instruction_t &instruction) const
    = 0;

  virtual std::tuple<amd_dbgapi_size_t /* instruction_size  */,
                     std::string /* instruction_text  */,
                     std::vector<uint64_t> /* address_operands  */>
  disassemble_instruction (agent_address_t address,
                           const instruction_t &instruction) const
    = 0;

  virtual std::pair<amd_dbgapi_wave_state_t, amd_dbgapi_wave_stop_reasons_t>
  wave_get_state (wave_t &wave) const = 0;
  virtual void wave_set_state (wave_t &wave,
                               amd_dbgapi_wave_state_t state) const
    = 0;

  virtual bool wave_get_halt (const wave_t &wave) const = 0;
  virtual void wave_set_halt (wave_t &wave, bool halt) const = 0;

  virtual void wave_enable_traps (wave_t &wave,
                                  os_wave_launch_trap_mask_t mask) const
    = 0;
  virtual void wave_disable_traps (wave_t &wave,
                                   os_wave_launch_trap_mask_t mask) const
    = 0;

  amd_dbgapi_architecture_id_t id () const { return m_architecture_id; }
  elf_amdgpu_machine_t elf_amdgpu_machine () const { return m_e_machine; }
  const std::string &target_triple () const { return m_target_triple; }

  static const architecture_t *
  find (amd_dbgapi_architecture_id_t architecture_id, int ignore = 0);
  static const architecture_t *find (elf_amdgpu_machine_t elf_amdgpu_machine);
  static const architecture_t *find (const std::string &name);

  amd_dbgapi_register_id_t regnum_to_register_id (amdgpu_regnum_t regnum) const
  {
    using integral_type = decltype (amd_dbgapi_register_id_t::handle);

    static_assert (
      sizeof (typename decltype (s_next_architecture_id)::value_type) <= 4
      && sizeof (amdgpu_regnum_t) <= 4 && sizeof (integral_type) >= 8);

    auto register_id = (static_cast<integral_type> (id ().handle) << 32)
                       | static_cast<integral_type> (regnum);

    return amd_dbgapi_register_id_t{ register_id };
  }
  static std::optional<amdgpu_regnum_t>
  register_id_to_regnum (amd_dbgapi_register_id_t register_id)
  {
    amdgpu_regnum_t regnum = static_cast<amdgpu_regnum_t> (
      utils::bit_extract (register_id.handle, 0, 31));

    return (regnum <= amdgpu_regnum_t::last_regnum)
             ? std::make_optional (regnum)
             : std::nullopt;
  }
  static const architecture_t *
  register_id_to_architecture (amd_dbgapi_register_id_t register_id)
  {
    return find (amd_dbgapi_architecture_id_t{
      utils::bit_extract (register_id.handle, 32, 63) });
  }

  /* Return the pointer to a statically allocated mask of read-only bits for
     the given register, or nullptr if all bits are writable.  */
  virtual const void *register_read_only_mask (amdgpu_regnum_t regnum) const
    = 0;

  virtual std::string register_name (amdgpu_regnum_t regnum) const = 0;
  virtual std::string register_type (amdgpu_regnum_t regnum) const = 0;
  virtual amd_dbgapi_size_t register_size (amdgpu_regnum_t regnum) const = 0;
  virtual amd_dbgapi_register_properties_t
  register_properties (amdgpu_regnum_t regnum) const
    = 0;

  std::set<amdgpu_regnum_t> register_set () const;
  bool is_register_available (amdgpu_regnum_t regnum) const;

  virtual bool is_pseudo_register_available (const wave_t &wave,
                                             amdgpu_regnum_t regnum) const
    = 0;

  virtual void read_pseudo_register (const wave_t &wave,
                                     amdgpu_regnum_t regnum, size_t offset,
                                     size_t value_size, void *value) const
    = 0;

  virtual void write_pseudo_register (const wave_t &wave,
                                      amdgpu_regnum_t regnum, size_t offset,
                                      size_t value_size,
                                      const void *value) const
    = 0;

  void get_info (amd_dbgapi_architecture_info_t query, size_t value_size,
                 void *value) const;

  template <typename Object, typename... Args> auto &create (Args &&...args)
  {
    return get_base_type_element<Object> (m_handle_object_sets)
      .template create_object<Object> (std::forward<Args> (args)...);
  }

  /* Return an Object range. A range implements begin () and end (), and
     can be used to iterate the Objects.  */
  template <typename Object> auto range () const
  {
    return std::get<handle_object_set_t<Object>> (m_handle_object_sets)
      .range ();
  }

  /* Return the element count for the sub-Object.  */
  template <typename Object> size_t count () const
  {
    return std::get<handle_object_set_t<Object>> (m_handle_object_sets)
      .size ();
  }

  /* Find an object with the given handle.  */
  template <typename Handle,
            std::enable_if_t<!std::is_void_v<object_type_from_handle_t<
                               Handle, decltype (m_handle_object_sets)>>,
                             int>
            = 0>
  const auto *find (Handle id) const
  {
    using object_type
      = object_type_from_handle_t<Handle, decltype (m_handle_object_sets)>;

    if constexpr (std::is_same_v<Handle, amd_dbgapi_address_space_id_t>)
      if (id == AMD_DBGAPI_ADDRESS_SPACE_GLOBAL)
        return &address_space_t::global ();

    return std::get<handle_object_set_t<object_type>> (m_handle_object_sets)
      .find (id);
  }
  template <typename Handle,
            std::enable_if_t<!std::is_void_v<object_type_from_handle_t<
                               Handle, decltype (m_handle_object_sets)>>,
                             int>
            = 0>
  auto *find (Handle id)
  {
    using object_type
      = object_type_from_handle_t<Handle, decltype (m_handle_object_sets)>;

    /* Cannot return a non-const global address space pointer.  */
    static_assert (!std::is_same_v<Handle, amd_dbgapi_address_space_id_t>);

    return std::get<handle_object_set_t<object_type>> (m_handle_object_sets)
      .find (id);
  }

  /* Find an object for which the unary predicate f returns true.  */
  template <typename Functor>
  const auto *find_if (Functor predicate, bool all = false) const
  {
    using object_type = std::decay_t<utils::first_argument_of_t<Functor>>;

    if constexpr (std::is_same_v<object_type, address_space_t>)
      if (predicate (address_space_t::global ()))
        return &address_space_t::global ();

    return std::get<handle_object_set_t<object_type>> (m_handle_object_sets)
      .find_if (predicate, all);
  }
  /* Find an object for which the unary predicate f returns true.  */
  template <typename Functor>
  auto *find_if (Functor predicate, bool all = false)
  {
    using object_type = std::decay_t<utils::first_argument_of_t<Functor>>;

    /* Cannot return a non-const global address space pointer.  */
    static_assert (!std::is_same_v<object_type, address_space_t>);

    return std::get<handle_object_set_t<object_type>> (m_handle_object_sets)
      .find_if (predicate, all);
  }
};

namespace detail
{
template <typename Handle>
using architecture_find_t
  = decltype (std::declval<const architecture_t> ().find (
    std::declval<Handle> ()));
} /* namespace detail */

/* Find an object with the given handle.  */
template <typename Handle,
          std::enable_if_t<
            utils::is_detected_v<detail::architecture_find_t, Handle>, int>
          = 0>
auto
find (Handle id) -> decltype (std::declval<const architecture_t> ().find (id))
{
  if constexpr (std::is_same_v<Handle, amd_dbgapi_address_space_id_t>)
    if (id == AMD_DBGAPI_ADDRESS_SPACE_GLOBAL)
      return &address_space_t::global ();

  if (detail::last_found_architecture)
    if (const auto *value = detail::last_found_architecture->find (id); value)
      return value;

  for (auto &&architecture : architecture_t::all ())
    {
      if (architecture.second.get () == detail::last_found_architecture)
        continue;
      if (const auto *value = architecture.second->find (id); value)
        {
          detail::last_found_architecture = architecture.second.get ();
          return value;
        }
    }

  return nullptr;
}

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_ARCHITECTURE_H */
