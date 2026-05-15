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

#ifndef AMD_DBGAPI_MEMORY_H
#define AMD_DBGAPI_MEMORY_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"
#include "logging.h"
#include "utils.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace amd::dbgapi
{

class architecture_t;
class process_t;
class wave_t;

/* AMDGPU DWARF Address Class Mapping
   See https://llvm.org/docs/AMDGPUUsage.html#address-class-identifier
 */
constexpr uint64_t DW_ADDR_none = 0x0000;
constexpr uint64_t DW_ADDR_LLVM_global = 0x0001;
constexpr uint64_t DW_ADDR_LLVM_constant = 0x0002;
constexpr uint64_t DW_ADDR_LLVM_group = 0x0003;
constexpr uint64_t DW_ADDR_LLVM_private = 0x0004;
constexpr uint64_t DW_ADDR_AMDGPU_region = 0x8000;

/* AMDGPU DWARF Address Space Mapping
   See https://llvm.org/docs/AMDGPUUsage.html#address-space-identifier
 */
constexpr uint64_t DW_ASPACE_none = 0x00;
constexpr uint64_t DW_ASPACE_AMDGPU_generic = 0x01;
constexpr uint64_t DW_ASPACE_AMDGPU_region = 0x02;
constexpr uint64_t DW_ASPACE_AMDGPU_local = 0x03;
constexpr uint64_t DW_ASPACE_AMDGPU_private_lane = 0x05;
constexpr uint64_t DW_ASPACE_AMDGPU_private_wave = 0x06;

namespace detail
{

template <typename T> class base_address_t
{
protected:
  uint64_t m_address;

public:
  constexpr base_address_t () = default;
  constexpr base_address_t (uint64_t address) : m_address (address) {}
  constexpr operator uint64_t () const { return m_address; }

  template <typename U> T operator+ (U increment) const
  {
    return T{ m_address + increment };
  }
  template <typename U> T operator- (U decrement) const
  {
    return T{ m_address - decrement };
  }
  template <typename U> T &operator+= (U increment)
  {
    m_address += increment;
    return static_cast<T &> (*this);
  }
  template <typename U> T &operator-= (U decrement)
  {
    m_address -= decrement;
    return static_cast<T &> (*this);
  }
};

} /* namespace detail  */

class agent_address_t : public detail::base_address_t<agent_address_t>
{
public:
  constexpr agent_address_t () : base_address_t (){};
  constexpr agent_address_t (uint64_t address) : base_address_t (address) {}
};

class host_address_t : public detail::base_address_t<host_address_t>
{
public:
  constexpr host_address_t () : base_address_t (){};
  constexpr host_address_t (uint64_t address) : base_address_t (address) {}
};

class global_address_t : public detail::base_address_t<global_address_t>
{
public:
  constexpr global_address_t () : base_address_t (){};
  constexpr global_address_t (uint64_t address) : base_address_t (address) {}
  operator agent_address_t () { return agent_address_t{ m_address }; }
  operator host_address_t () { return host_address_t{ m_address }; }
};

template <> std::string to_string (agent_address_t address);
template <> std::string to_string (host_address_t address);
template <> std::string to_string (global_address_t address);

class address_class_t;

class address_space_t
  : public detail::handle_object<amd_dbgapi_address_space_id_t>
{
public:
  enum class kind_t
  {
    generic = 1,
    local,
    global,
    private_swizzled,
    private_unswizzled,
    agent,
    host
  };

  enum class reserved_ids_t
  {
    global = AMD_DBGAPI_ADDRESS_SPACE_GLOBAL.handle,
    host,
    next_non_reserved_id
  };

  static const address_space_t &global ();
  static const address_space_t &host ();

private:
  kind_t const m_kind;
  std::string const m_name;
  std::optional<uint64_t> const m_dwarf_value;
  amd_dbgapi_size_t const m_address_size;
  amd_dbgapi_segment_address_t const m_null_address;
  amd_dbgapi_address_space_access_t const m_access;

protected:
  address_space_t (amd_dbgapi_address_space_id_t address_space_id, kind_t kind,
                   std::string name, std::optional<uint64_t> dwarf_value,
                   amd_dbgapi_size_t address_size,
                   amd_dbgapi_segment_address_t null_address,
                   amd_dbgapi_address_space_access_t access)
    : handle_object (address_space_id), m_kind (kind),
      m_name (std::move (name)), m_dwarf_value (dwarf_value),
      m_address_size (address_size), m_null_address (null_address),
      m_access (access)
  {
    dbgapi_assert (m_address_size <= sizeof (amd_dbgapi_segment_address_t) * 8
                   && "address_size is too big");
  }

public:
  virtual ~address_space_t () = default;

  std::optional<uint64_t> dwarf_value () const { return m_dwarf_value; }
  const std::string &name () const { return m_name; }
  kind_t kind () const { return m_kind; }
  bool is_internal () const { return !dwarf_value ().has_value (); }
  bool is_valid () const { return !is_internal (); }

  amd_dbgapi_size_t address_size () const { return m_address_size; }
  amd_dbgapi_segment_address_t null_address () const { return m_null_address; }
  amd_dbgapi_segment_address_t last_address () const
  {
    return utils::bit_mask<amd_dbgapi_segment_address_t> (0,
                                                          address_size () - 1);
  }

  bool
  address_is_in_address_class (const wave_t &wave,
                               amd_dbgapi_lane_id_t lane_id,
                               amd_dbgapi_segment_address_t segment_address,
                               const address_class_t &address_class) const;

  virtual amd_dbgapi_segment_address_dependency_t
  address_dependency (amd_dbgapi_segment_address_t address) const
    = 0;

  /* Lower an address in this address space to an address in a base address
     space in the same architecture.  The base address spaces kinds are global,
     local, private_swizzled, and private_unswizzled.  */
  virtual std::pair<const address_space_t & /* lowered_address_space  */,
                    amd_dbgapi_segment_address_t /* lowered_address  */>
  lower (amd_dbgapi_segment_address_t address) const = 0;

  /* Convert an address in the given address space to an address in this
     address space.  Return both the converted address and the number of
     bytes that are contiguous in both address spaces.  Throws
     AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_CONVERSION and
     AMD_DBGAPI_STATUS_ERROR_INVALID_LANE_ID errors.  */
  virtual std::pair<amd_dbgapi_segment_address_t /* to_address  */,
                    amd_dbgapi_size_t /* to_contiguous_bytes  */>
  convert (const wave_t &wave, amd_dbgapi_lane_id_t lane_id,
           const address_space_t &from_address_space,
           amd_dbgapi_segment_address_t from_address) const
    = 0;

  void get_info (amd_dbgapi_address_space_info_t query, size_t value_size,
                 void *value) const;
};

class global_address_space_t : public address_space_t
{
public:
  global_address_space_t (amd_dbgapi_address_space_id_t address_space_id,
                          std::string name)
    : address_space_t (address_space_id, kind_t::global, std::move (name),
                       { DW_ASPACE_none }, 64, 0x0000000000000000,
                       AMD_DBGAPI_ADDRESS_SPACE_ACCESS_ALL)
  {
  }

  amd_dbgapi_segment_address_dependency_t address_dependency (
    amd_dbgapi_segment_address_t /* address  */) const override
  {
    return AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_PROCESS;
  }

  std::pair<const address_space_t &, amd_dbgapi_segment_address_t>
  lower (amd_dbgapi_segment_address_t global_address) const override;

  std::pair<amd_dbgapi_segment_address_t, amd_dbgapi_size_t>
  convert (const wave_t &wave, amd_dbgapi_lane_id_t lane_id,
           const address_space_t &from_address_space,
           amd_dbgapi_segment_address_t from_address) const override;
};

class local_address_space_t : public address_space_t
{
public:
  local_address_space_t (amd_dbgapi_address_space_id_t address_space_id,
                         std::string name)
    : address_space_t (address_space_id, kind_t::local, std::move (name),
                       { DW_ASPACE_AMDGPU_local }, 32, 0xFFFFFFFF,
                       AMD_DBGAPI_ADDRESS_SPACE_ACCESS_ALL)
  {
  }

  amd_dbgapi_segment_address_dependency_t address_dependency (
    amd_dbgapi_segment_address_t /* address  */) const override
  {
    return AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_WORKGROUP;
  }

  std::pair<const address_space_t &, amd_dbgapi_segment_address_t>
  lower (amd_dbgapi_segment_address_t local_address) const override;

  std::pair<amd_dbgapi_segment_address_t, amd_dbgapi_size_t>
  convert (const wave_t &wave, amd_dbgapi_lane_id_t lane_id,
           const address_space_t &from_address_space,
           amd_dbgapi_segment_address_t from_address) const override;
};

class private_swizzled_address_space_t : public address_space_t
{
private:
  amd_dbgapi_size_t const m_interleave_size;

public:
  private_swizzled_address_space_t (
    amd_dbgapi_address_space_id_t address_space_id, std::string name,
    amd_dbgapi_size_t interleave_size)
    : address_space_t (address_space_id, kind_t::private_swizzled,
                       std::move (name), { DW_ASPACE_AMDGPU_private_lane }, 32,
                       0xFFFFFFFF, AMD_DBGAPI_ADDRESS_SPACE_ACCESS_ALL),
      m_interleave_size (interleave_size)
  {
  }

  /* Return the number of bytes (N) used to interleave private swizzled memory
     accesses.  Private swizzled memory has the following layout in global
     memory (X is the number of lanes in a wavefront):

     global     lane0 private      lane1 private           laneX private
     addresses  addresses          addresses               addresses
     0*X*N:     [0*N, ..., 1*N-1], [0*N, ..., 1*N-1], ..., [0*N, ..., 1*N-1]
     1*X*N:     [1*N, ..., 2*N-1], [1*N, ..., 2*N-1], ..., [1*N, ..., 2*N-1]
     2*X*N:     [2*N, ..., 3*N-1], [2*N, ..., 3*N-1], ..., [2*N, ..., 3*N-1]
     ...  */
  amd_dbgapi_size_t interleave_size () const { return m_interleave_size; }

  amd_dbgapi_segment_address_dependency_t address_dependency (
    amd_dbgapi_segment_address_t /* address  */) const override
  {
    return AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_LANE;
  }

  std::pair<const address_space_t &, amd_dbgapi_segment_address_t>
  lower (amd_dbgapi_segment_address_t private_address) const override;

  std::pair<amd_dbgapi_segment_address_t, amd_dbgapi_size_t>
  convert (const wave_t &wave, amd_dbgapi_lane_id_t lane_id,
           const address_space_t &from_address_space,
           amd_dbgapi_segment_address_t from_address) const override;
};

class private_unswizzled_address_space_t : public address_space_t
{
public:
  private_unswizzled_address_space_t (
    amd_dbgapi_address_space_id_t address_space_id, std::string name)
    : address_space_t (address_space_id, kind_t::private_unswizzled,
                       std::move (name), { DW_ASPACE_AMDGPU_private_wave }, 32,
                       0xFFFFFFFF, AMD_DBGAPI_ADDRESS_SPACE_ACCESS_ALL)
  {
  }

  amd_dbgapi_segment_address_dependency_t address_dependency (
    amd_dbgapi_segment_address_t /* address  */) const override
  {
    return AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_WAVE;
  }

  std::pair<const address_space_t &, amd_dbgapi_segment_address_t>
  lower (amd_dbgapi_segment_address_t private_address) const override;

  std::pair<amd_dbgapi_segment_address_t, amd_dbgapi_size_t>
  convert (const wave_t &wave, amd_dbgapi_lane_id_t lane_id,
           const address_space_t &from_address_space,
           amd_dbgapi_segment_address_t from_address) const override;
};

class generic_address_space_t : public address_space_t
{
public:
  struct aperture_t
  {
    agent_address_t base;
    agent_address_t mask;
    const address_space_t &address_space;
  };

private:
  std::vector<aperture_t> const m_apertures;

  /* Return the generic address for a given segment address space, segment
     address pair.  Converting an address from an address space other than
     one in the apertures is invalid.  */
  std::optional<amd_dbgapi_segment_address_t>
  generic_address_for_address_space (
    const address_space_t &segment_address_space,
    amd_dbgapi_segment_address_t segment_address) const;

public:
  generic_address_space_t (amd_dbgapi_address_space_id_t address_space_id,
                           std::string name,
                           std::vector<aperture_t> apertures);

  amd_dbgapi_segment_address_dependency_t
  address_dependency (amd_dbgapi_segment_address_t address) const override;

  std::pair<const address_space_t &, amd_dbgapi_segment_address_t>
  lower (amd_dbgapi_segment_address_t generic_address) const override;

  std::pair<amd_dbgapi_segment_address_t, amd_dbgapi_size_t>
  convert (const wave_t &wave, amd_dbgapi_lane_id_t lane_id,
           const address_space_t &from_address_space,
           amd_dbgapi_segment_address_t from_address) const override;
};

class agent_address_space_t : public address_space_t
{
public:
  agent_address_space_t (amd_dbgapi_address_space_id_t address_space_id,
                         std::string name)
    : address_space_t (address_space_id, kind_t::agent, std::move (name),
                       std::nullopt, 64, 0x0000000000000000,
                       AMD_DBGAPI_ADDRESS_SPACE_ACCESS_ALL)
  {
  }

  amd_dbgapi_segment_address_dependency_t address_dependency (
    amd_dbgapi_segment_address_t /* address  */) const override
  {
    return AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_AGENT;
  }

  std::pair<const address_space_t &, amd_dbgapi_segment_address_t>
  lower (amd_dbgapi_segment_address_t local_address) const override;

  std::pair<amd_dbgapi_segment_address_t, amd_dbgapi_size_t>
  convert (const wave_t &wave, amd_dbgapi_lane_id_t lane_id,
           const address_space_t &from_address_space,
           amd_dbgapi_segment_address_t from_address) const override;
};

class host_address_space_t : public address_space_t
{
public:
  host_address_space_t (amd_dbgapi_address_space_id_t address_space_id,
                        std::string name)
    : address_space_t (address_space_id, kind_t::host, std::move (name),
                       std::nullopt, 64, 0x0000000000000000,
                       AMD_DBGAPI_ADDRESS_SPACE_ACCESS_ALL)
  {
  }

  amd_dbgapi_segment_address_dependency_t address_dependency (
    amd_dbgapi_segment_address_t /* address  */) const override
  {
    return AMD_DBGAPI_SEGMENT_ADDRESS_DEPENDENCE_AGENT;
  }

  std::pair<const address_space_t &, amd_dbgapi_segment_address_t>
  lower (amd_dbgapi_segment_address_t host_address) const override
  {
    return { *this, host_address };
  }

  std::pair<amd_dbgapi_segment_address_t, amd_dbgapi_size_t>
  convert (const wave_t & /* wave  */, amd_dbgapi_lane_id_t /* lane_id  */,
           const address_space_t &from_address_space,
           amd_dbgapi_segment_address_t from_address) const override
  {
    auto [lowered_address_space, lowered_address]
      = from_address_space.lower (from_address);

    if (lowered_address_space.kind () == kind_t::host)
      return { lowered_address, last_address () - lowered_address + 1 };

    throw api_error_t (
      AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_CONVERSION);
  }
};

/* Some IDs are reserved for static address spaces (global, host).  Make sure
   to start the numbering at the first non-reserved address space ID.  */
template <>
struct monotonic_counter_start_t<amd_dbgapi_address_space_id_t>
  : public std::integral_constant<
      decltype (amd_dbgapi_address_space_id_t::handle),
      static_cast<std::underlying_type_t<address_space_t::reserved_ids_t>> (
        address_space_t::reserved_ids_t::next_non_reserved_id)>
{
};

class address_class_t
  : public detail::handle_object<amd_dbgapi_address_class_id_t>
{
private:
  std::string const m_name;
  uint64_t const m_dwarf_value;
  const address_space_t &m_address_space;

public:
  address_class_t (amd_dbgapi_address_class_id_t address_class_id,
                   std::string name, uint64_t dwarf_value,
                   const address_space_t &address_space)
    : handle_object (address_class_id), m_name (std::move (name)),
      m_dwarf_value (dwarf_value), m_address_space (address_space)
  {
  }

public:
  uint64_t dwarf_value () const { return m_dwarf_value; }
  const std::string &name () const { return m_name; }
  const address_space_t &address_space () const { return m_address_space; }

  void get_info (amd_dbgapi_address_class_info_t query, size_t value_size,
                 void *value) const;
};

template <typename AddressType> class memory_cache_t
{
public:
  enum class policy_t
  {
    /* If uncached is used, data is immediately written to global memory, and
       is not written to the cache.  */
    uncached = 0,
    /* If write-through is used, data is written both to global memory and to
       the cache.  */
    write_through,
    /* If write-back is used, data is immediately updated in the cache, and
       later updated in memory when the cache is flushed.  */
    write_back
  };

  static constexpr size_t cache_line_size = 64;
  static constexpr policy_t policy = policy_t::write_back;

private:
  using delegate_fn_type
    = std::function<size_t (AddressType /* address */, void * /* read */,
                            const void * /* write */, size_t /* size */)>;

  struct cache_line_t
  {
    std::array<std::byte, cache_line_size> m_data{};
    bool m_dirty{ false };
  };

  std::map<AddressType, cache_line_t> m_cache_line_map;
  delegate_fn_type const m_xfer_global_memory;

  void fetch_cache_line (cache_line_t &cache_line, AddressType address) const;
  void commit_cache_line (cache_line_t &cache_line, AddressType address) const;
  void allocate_0_cache_line (cache_line_t &cache_line) const;

  size_t xfer_global_memory (AddressType address, void *read,
                             const void *write, size_t size);

public:
  memory_cache_t (delegate_fn_type xfer_global_memory)
    : m_xfer_global_memory (std::move (xfer_global_memory))
  {
  }
  ~memory_cache_t () { dbgapi_assert (m_cache_line_map.empty ()); }

  bool contains_all (AddressType address, amd_dbgapi_size_t size) const;

  /* Create cache lines if not already valid, and immediately fill them in.  */
  void prefetch (AddressType address, amd_dbgapi_size_t size);

  /* Discard all cache lines in the specified range.  If FORCE_DISCARD
     is true, dirty lines are silently dropped.  Otherwise it is an error to
     discarded dirty cache lines.  */
  void discard (AddressType address = 0, amd_dbgapi_size_t size = -1,
                bool force_discard = false);

  /* Write dirty lines back to memory.  */
  void write_back (AddressType address = 0, amd_dbgapi_size_t size = -1);

  [[nodiscard]] size_t read_global_memory (AddressType address, void *buffer,
                                           size_t size)
  {
    return xfer_global_memory (address, buffer, nullptr, size);
  }

  [[nodiscard]] size_t write_global_memory (AddressType address,
                                            const void *buffer, size_t size)
  {
    return xfer_global_memory (address, nullptr, buffer, size);
  }
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_MEMORY_H */
