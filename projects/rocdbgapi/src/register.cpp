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

#include "register.h"
#include "architecture.h"
#include "debug.h"
#include "exception.h"
#include "initialization.h"
#include "logging.h"
#include "memory.h"
#include "process.h"
#include "queue.h"
#include "utils.h"
#include "wave.h"

#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <utility>

namespace amd::dbgapi
{

/* Register class.  */

bool
register_class_t::add_registers (amdgpu_regnum_t first, amdgpu_regnum_t last)
{
  dbgapi_assert (last >= first);
  return m_register_map.emplace (first, last).second;
}

bool
register_class_t::remove_registers (amdgpu_regnum_t first,
                                    amdgpu_regnum_t last)
{
  dbgapi_assert (last >= first);
  while (first <= last)
    {
      auto it = m_register_map.upper_bound (first);
      if (it == m_register_map.begin ())
        return false;

      /* Get the range that contains 'first'.  */
      std::advance (it, -1);
      if (first < it->first || first > it->second)
        return false;

      auto [range_first, range_last] = *it;
      it = m_register_map.erase (it);

      if (range_first < first)
        m_register_map.emplace_hint (it, range_first, first - 1);

      if (last < range_last)
        m_register_map.emplace_hint (it, last + 1, range_last);

      first = range_last + 1;
    }

  return true;
}

bool
register_class_t::contains (amdgpu_regnum_t regnum) const
{
  auto it = m_register_map.upper_bound (regnum);
  if (it == m_register_map.begin ())
    return false;

  std::advance (it, -1);
  return regnum >= it->first && regnum <= it->second;
}

std::set<amdgpu_regnum_t>
register_class_t::register_set () const
{
  std::set<amdgpu_regnum_t> all_registers;

  for (auto interval : m_register_map)
    for (amdgpu_regnum_t regnum = interval.first; regnum <= interval.second;
         ++regnum)
      all_registers.insert (regnum);

  return all_registers;
}

void
register_class_t::get_info (amd_dbgapi_register_class_info_t query,
                            size_t value_size, void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_REGISTER_CLASS_INFO_ARCHITECTURE:
      utils::get_info (value_size, value, architecture ().id ());
      return;

    case AMD_DBGAPI_REGISTER_CLASS_INFO_NAME:
      utils::get_info (value_size, value, name ());
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

namespace
{

std::optional<uint64_t>
amdgpu_regnum_to_dwarf_register (amdgpu_regnum_t regnum)
{
  /* See
     https://llvm.org/docs/AMDGPUUsage.html#amdgpu-dwarf-register-mapping-table
   */
  if (regnum == amdgpu_regnum_t::pseudo_exec_32)
    {
      return 1;
    }
  else if (regnum == amdgpu_regnum_t::pseudo_exec_64)
    {
      return 17;
    }
  else if (regnum == amdgpu_regnum_t::pc)
    {
      return 16;
    }
  else if (regnum >= amdgpu_regnum_t::first_sgpr
           && regnum < (amdgpu_regnum_t::first_sgpr + 64))
    {
      /* Scalar registers 0-63.  */
      return 32 + (regnum - amdgpu_regnum_t::first_sgpr);
    }
  else if (regnum == amdgpu_regnum_t::pseudo_status)
    {
      return 128;
    }
  else if (regnum == amdgpu_regnum_t::pseudo_vcc_32)
    {
      return 512;
    }
  else if (regnum == amdgpu_regnum_t::pseudo_vcc_64)
    {
      return 768;
    }
  else if (regnum >= amdgpu_regnum_t::first_sgpr + 64
           && regnum <= (amdgpu_regnum_t::first_sgpr + 105))
    {
      /* Scalar registers 64-105.  */
      return 1024 + (regnum - amdgpu_regnum_t::first_sgpr);
    }
  else if (regnum >= amdgpu_regnum_t::first_vgpr_32
           && regnum < (amdgpu_regnum_t::first_vgpr_32 + 256))
    {
      /* Vector registers 0-255 (wave32).  */
      return 1536 + (regnum - amdgpu_regnum_t::first_vgpr_32);
    }
  else if (regnum >= amdgpu_regnum_t::first_accvgpr_32
           && regnum < (amdgpu_regnum_t::first_accvgpr_32 + 256))
    {
      /* Accumulation Vector registers 0-255 (wave32).  */
      return 2048 + (regnum - amdgpu_regnum_t::first_accvgpr_32);
    }
  else if (regnum >= amdgpu_regnum_t::first_vgpr_64
           && regnum < (amdgpu_regnum_t::first_vgpr_64 + 256))
    {
      /* Vector registers 0-255 (wave64).  */
      return 2560 + (regnum - amdgpu_regnum_t::first_vgpr_64);
    }
  else if (regnum >= amdgpu_regnum_t::first_accvgpr_64
           && regnum < (amdgpu_regnum_t::first_accvgpr_64 + 256))
    {
      /* Accumulation Vector registers 0-255 (wave64).  */
      return 3072 + (regnum - amdgpu_regnum_t::first_accvgpr_64);
    }

  return std::nullopt;
}

std::optional<amdgpu_regnum_t>
dwarf_register_to_amdgpu_regnum (uint64_t dwarf_register)
{
  /* See https://llvm.org/docs/AMDGPUUsage.html#register-mapping.  */
  if (dwarf_register == 1)
    {
      return amdgpu_regnum_t::pseudo_exec_32;
    }
  else if (dwarf_register == 17)
    {
      return amdgpu_regnum_t::pseudo_exec_64;
    }
  else if (dwarf_register == 16)
    {
      return amdgpu_regnum_t::pc;
    }
  else if (dwarf_register >= 32 && dwarf_register <= 95)
    {
      /* Scalar registers 0-63.  */
      return amdgpu_regnum_t::first_sgpr + (dwarf_register - 32);
    }
  else if (dwarf_register == 128)
    {
      return amdgpu_regnum_t::pseudo_status;
    }
  else if (dwarf_register == 512)
    {
      return amdgpu_regnum_t::pseudo_vcc_32;
    }
  else if (dwarf_register == 768)
    {
      return amdgpu_regnum_t::pseudo_vcc_64;
    }
  else if (dwarf_register >= 1088 && dwarf_register <= 1129)
    {
      /* Scalar registers 64-105.  */
      return amdgpu_regnum_t::first_sgpr + (dwarf_register - 1024);
    }
  else if (dwarf_register >= 1536 && dwarf_register <= 1791)
    {
      /* Vector registers 0-255 (wave32).  */
      return amdgpu_regnum_t::first_vgpr_32 + (dwarf_register - 1536);
    }
  else if (dwarf_register >= 2048 && dwarf_register <= 2303)
    {
      /* Accumulation Vector registers 0-255 (wave32).  */
      return amdgpu_regnum_t::first_accvgpr_32 + (dwarf_register - 2048);
    }
  else if (dwarf_register >= 2560 && dwarf_register <= 2815)
    {
      /* Vector registers 0-255 (wave64).  */
      return amdgpu_regnum_t::first_vgpr_64 + (dwarf_register - 2560);
    }
  else if (dwarf_register >= 3072 && dwarf_register <= 3327)
    {
      /* Accumulation Vector registers 0-255 (wave64).  */
      return amdgpu_regnum_t::first_accvgpr_64 + (dwarf_register - 3072);
    }

  return std::nullopt;
}

} /* anonymous namespace */

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_architecture_register_class_get_info (
  amd_dbgapi_register_class_id_t register_class_id,
  amd_dbgapi_register_class_info_t query, size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (register_class_id), param_in (query),
               param_in (value_size), param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    const register_class_t *register_class = find (register_class_id);

    if (register_class == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID);

    register_class->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_architecture_register_class_list (
  amd_dbgapi_architecture_id_t architecture_id, size_t *register_class_count,
  amd_dbgapi_register_class_id_t **register_classes)
{
  TRACE_BEGIN (param_in (architecture_id), param_in (register_class_count),
               param_in (register_classes));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    const architecture_t *architecture
      = architecture_t::find (architecture_id);

    if (architecture == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID);

    if (register_class_count == nullptr || register_classes == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    size_t count = architecture->count<register_class_t> ();

    auto class_ids = allocate_memory<amd_dbgapi_register_class_id_t[]> (
      count * sizeof (amd_dbgapi_register_class_id_t));

    size_t pos = 0;
    for (auto &&register_class : architecture->range<register_class_t> ())
      class_ids[pos++] = register_class.id ();

    *register_class_count = count;
    *register_classes = class_ids.release ();
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (
    make_ref (param_out (register_class_count)),
    make_ref (make_ref (param_out (register_classes)), *register_class_count));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_register_get_info (amd_dbgapi_register_id_t register_id,
                              amd_dbgapi_register_info_t query,
                              size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (register_id), param_in (query), param_in (value_size),
               param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    auto regnum = architecture_t::register_id_to_regnum (register_id);

    const architecture_t *architecture
      = architecture_t::register_id_to_architecture (register_id);

    if (architecture == nullptr || !regnum
        || !architecture->is_register_available (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);

    if (value == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    [&] ()
    {
      switch (query)
        {
        case AMD_DBGAPI_REGISTER_INFO_ARCHITECTURE:
          utils::get_info (value_size, value, architecture->id ());
          return;

        case AMD_DBGAPI_REGISTER_INFO_NAME:
          utils::get_info (value_size, value,
                           architecture->register_name (*regnum));
          return;

        case AMD_DBGAPI_REGISTER_INFO_TYPE:
          utils::get_info (value_size, value,
                           architecture->register_type (*regnum));
          return;

        case AMD_DBGAPI_REGISTER_INFO_SIZE:
          utils::get_info (value_size, value,
                           architecture->register_size (*regnum));
          return;

        case AMD_DBGAPI_REGISTER_INFO_DWARF:
          {
            auto dwarf_register = amdgpu_regnum_to_dwarf_register (*regnum);
            if (!dwarf_register)
              THROW (AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);

            utils::get_info (value_size, value, *dwarf_register);
            return;
          }

        case AMD_DBGAPI_REGISTER_INFO_PROPERTIES:
          utils::get_info (value_size, value,
                           architecture->register_properties (*regnum));
          return;
        }

      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }();
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_architecture_register_list (
  amd_dbgapi_architecture_id_t architecture_id, size_t *register_count,
  amd_dbgapi_register_id_t **registers)
{
  TRACE_BEGIN (param_in (architecture_id), param_in (register_count),
               param_in (registers));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    const architecture_t *architecture
      = architecture_t::find (architecture_id);

    if (architecture == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID);

    if (register_count == nullptr || registers == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    auto arch_registers = architecture->register_set ();

    auto retval = allocate_memory<amd_dbgapi_register_id_t[]> (
      arch_registers.size () * sizeof (amd_dbgapi_register_id_t));

    size_t count = 0;
    for (auto it = arch_registers.begin (); it != arch_registers.end (); ++it)
      retval[count++] = architecture->regnum_to_register_id (*it);

    *register_count = count;
    *registers = retval.release ();
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_ref (param_out (register_count)),
             make_ref (make_ref (param_out (registers)), *register_count));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_register_is_in_register_class (
  amd_dbgapi_register_class_id_t register_class_id,
  amd_dbgapi_register_id_t register_id,
  amd_dbgapi_register_class_state_t *register_class_state)
{
  TRACE_BEGIN (param_in (register_class_id), param_in (register_id),
               param_in (register_class_state));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    const register_class_t *register_class = find (register_class_id);

    if (register_class == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID);

    auto regnum = architecture_t::register_id_to_regnum (register_id);

    const architecture_t *architecture
      = architecture_t::register_id_to_architecture (register_id);

    if (!regnum || architecture == nullptr
        || !architecture->is_register_available (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);

    if (register_class_state == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    if (*architecture != register_class->architecture ())
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

    *register_class_state = register_class->contains (*regnum)
                              ? AMD_DBGAPI_REGISTER_CLASS_STATE_MEMBER
                              : AMD_DBGAPI_REGISTER_CLASS_STATE_NOT_MEMBER;
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
  TRACE_END (make_ref (param_out (register_class_state)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_dwarf_register_to_register (
  amd_dbgapi_architecture_id_t architecture_id, uint64_t dwarf_register,
  amd_dbgapi_register_id_t *register_id)
{
  TRACE_BEGIN (param_in (architecture_id), param_in (dwarf_register),
               param_in (register_id));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    const architecture_t *architecture
      = architecture_t::find (architecture_id);

    if (architecture == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID);

    if (register_id == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    auto regnum = dwarf_register_to_amdgpu_regnum (dwarf_register);
    if (!regnum)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

    if (!architecture->is_register_available (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

    *register_id = architecture->regnum_to_register_id (*regnum);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
  TRACE_END (make_ref (param_out (register_id)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_read_register (amd_dbgapi_wave_id_t wave_id,
                          amd_dbgapi_register_id_t register_id,
                          amd_dbgapi_size_t offset,
                          amd_dbgapi_size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (wave_id), param_in (register_id), param_in (offset),
               param_in (value_size), param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    auto regnum = architecture_t::register_id_to_regnum (register_id);

    const architecture_t *architecture
      = architecture_t::register_id_to_architecture (register_id);

    if (!regnum || architecture == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);

    if (wave->state () != AMD_DBGAPI_WAVE_STATE_STOP)
      THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);

    if (value == nullptr || !value_size)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    if (*architecture != wave->architecture ()
        || (offset + value_size) > architecture->register_size (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

    if (!wave->is_register_available (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE);

    wave->read_register (*regnum, offset, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID,
         AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE);
  TRACE_END (make_hex (make_ref (param_out (value), value_size)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_write_register (amd_dbgapi_wave_id_t wave_id,
                           amd_dbgapi_register_id_t register_id,
                           amd_dbgapi_size_t offset,
                           amd_dbgapi_size_t value_size, const void *value)
{
  TRACE_BEGIN (param_in (wave_id), param_in (register_id), param_in (offset),
               param_in (value_size),
               make_hex (make_ref (param_in (value), value_size)));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    if (wave->process ().is_frozen ())
      THROW (AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);

    auto regnum = architecture_t::register_id_to_regnum (register_id);

    const architecture_t *architecture
      = architecture_t::register_id_to_architecture (register_id);

    if (!regnum || architecture == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);

    if (wave->state () != AMD_DBGAPI_WAVE_STATE_STOP)
      THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);

    /* Is displaced stepping active?  */
    if (wave->displaced_stepping () != nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_ACTIVE);

    if (value == nullptr || !value_size)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    if (*architecture != wave->architecture ()
        || (offset + value_size) > architecture->register_size (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

    if (!wave->is_register_available (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE);

    wave->write_register (*regnum, offset, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID,
         AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED,
         AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_ACTIVE,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE,
         AMD_DBGAPI_STATUS_ERROR_PROCESS_FROZEN);
  TRACE_END ();
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_wave_register_exists (amd_dbgapi_wave_id_t wave_id,
                                 amd_dbgapi_register_id_t register_id,
                                 amd_dbgapi_register_exists_t *exists)
{
  TRACE_BEGIN (param_in (wave_id), param_in (register_id), param_in (exists));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    auto regnum = architecture_t::register_id_to_regnum (register_id);

    const architecture_t *architecture
      = architecture_t::register_id_to_architecture (register_id);

    if (!regnum || architecture == nullptr
        || !architecture->is_register_available (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);

    if (exists == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    if (*architecture != wave->architecture ())
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

    *exists = wave->is_register_available (*regnum)
                ? AMD_DBGAPI_REGISTER_PRESENT
                : AMD_DBGAPI_REGISTER_ABSENT;
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
  TRACE_END (make_ref (param_out (exists)));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_wave_register_list (amd_dbgapi_wave_id_t wave_id,
                               size_t *register_count,
                               amd_dbgapi_register_id_t **registers)
{
  TRACE_BEGIN (param_in (wave_id), param_in (register_count),
               param_in (registers));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    if (registers == nullptr || register_count == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    auto architecture_registers = wave->architecture ().register_set ();
    auto retval = allocate_memory<amd_dbgapi_register_id_t[]> (
      architecture_registers.size () * sizeof (amd_dbgapi_register_id_t));

    size_t count = 0;
    for (auto &&regnum : architecture_registers)
      if (wave->is_register_available (regnum))
        retval[count++] = wave->architecture ().regnum_to_register_id (regnum);

    *register_count = count;
    *registers = retval.release ();
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_ref (param_out (register_count)),
             make_ref (make_ref (param_out (registers)), *register_count));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_prefetch_register (
  amd_dbgapi_wave_id_t wave_id, amd_dbgapi_register_id_t register_id,
  [[maybe_unused]] amd_dbgapi_size_t register_count)
{
  TRACE_BEGIN (param_in (wave_id), param_in (register_id),
               param_in (register_count));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    wave_t *wave = find (wave_id);

    if (wave == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);

    auto regnum = architecture_t::register_id_to_regnum (register_id);

    const architecture_t *architecture
      = architecture_t::register_id_to_architecture (register_id);

    if (!regnum || architecture == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);

    if (wave->state () != AMD_DBGAPI_WAVE_STATE_STOP)
      THROW (AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);

    if (*architecture != wave->architecture ())
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

    if (!wave->is_register_available (*regnum))
      THROW (AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID,
         AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE);
  TRACE_END ();
}
