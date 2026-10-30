/* Copyright (c) 2026 Advanced Micro Devices, Inc.

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

#include "property.h"
#include "amd-dbgapi.h"
#include "logging.h"
#include "process.h"

#include <algorithm>
#include <optional>

namespace amd::dbgapi
{

std::pair<amd_dbgapi_property_id_t, std::unique_ptr<property_t>>
make_property (property_id_t property_id, std::string name, std::string descr)
{
  amd_dbgapi_property_id_t id = { static_cast<uint64_t> (property_id) };
  std::unique_ptr<property_t> property
    = std::make_unique<property_t> (id, name, descr);
  return std::make_pair (id, std::move (property));
}

property_t::properties_map_t property_t::s_property_map{
  [] ()
  {
    properties_map_t map;

    map.emplace (make_property (property_id_t::precise_memory_exception,
                                "precise-memory",
                                "precise memory access reporting"));

    map.emplace (make_property (property_id_t::precise_alu_exception,
                                "precise-alu-exceptions",
                                "precise ALU exception reporting"));

    map.emplace (
      make_property (property_id_t::lds_exception, "lds-exception",
                     "local memory out-of-range address exception reporting"));

    return map;
  } ()
};

/* property_t::property_type_t --> amd_dbgapi_property_type_t  */

static amd_dbgapi_property_type_t
type_to_intf (property_t::property_type_t type)
{
  switch (type)
    {
    case property_t::property_type_t::boolean:
      return AMD_DBGAPI_PROPERTY_TYPE_BOOL;
    }
  dbgapi_assert_not_reached ("property type is unknown");
}

/* property_t::property_scope_t --> amd_dbgapi_property_scope_t  */

static amd_dbgapi_property_scope_t
scope_to_intf (property_t::property_scope_t scope_mask)
{
  using T = property_t::property_scope_t;
  using U = std::underlying_type_t<T>;
  amd_dbgapi_property_scope_t value = AMD_DBGAPI_PROPERTY_SCOPE_NONE;

  utils::for_each_flag<U> (scope_mask, [&] (U one_bit)
    {
      switch (static_cast<T> (one_bit))
        {
        case property_t::property_scope_t::process:
          value |= AMD_DBGAPI_PROPERTY_SCOPE_PROCESS;
          return;
        }
        dbgapi_assert_not_reached ("property scope is unknown");
    });

  return value;
}

void
property_t::get_info (amd_dbgapi_property_info_t query,
                      size_t value_size, void *value) const
{
  switch (query)
    {
    case AMD_DBGAPI_PROPERTY_INFO_NAME:
      utils::get_info (value_size, value, name ());
      return;

    case AMD_DBGAPI_PROPERTY_INFO_DESCRIPTION:
      utils::get_info (value_size, value, desc ());
      return;

    case AMD_DBGAPI_PROPERTY_INFO_TYPE:
      utils::get_info (value_size, value, type_to_intf (type ()));
      return;

    case AMD_DBGAPI_PROPERTY_INFO_SCOPE:
      utils::get_info (value_size, value, scope_to_intf (scope ()));
      return;
    }

  throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

void
property_t::set (process_t *proc, size_t value_size, const void *value) const
{
  std::optional<bool> value_bool;

  /* Verify the type and cast "value" based on it.  */
  if (type () == property_type_t::boolean)
    {
      if (value_size != sizeof (amd_dbgapi_property_bool_t))
        THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
      auto ptr = static_cast<const amd_dbgapi_property_bool_t *> (value);
      value_bool = !!(*ptr);
    }

  /* Right now, there's only bool support.  So assert here once,
     instead of before each ...->set_*().  */
  dbgapi_assert (value_bool.has_value ());

  switch (static_cast<property_id_t> (id ().handle))
    {
    case property_id_t::none:
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    case property_id_t::precise_memory_exception:
      proc->set_precise_memory (*value_bool);
      break;
    case property_id_t::precise_alu_exception:
      proc->set_precise_alu_exceptions (*value_bool);
      break;
    case property_id_t::lds_exception:
      proc->set_reporting_lds_exceptions (*value_bool);
      break;
    }
}

} /* namespace amd::dbgapi */

using namespace amd::dbgapi;

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_property_list (size_t *property_count,
                          amd_dbgapi_property_id_t **properties)
{
  TRACE_BEGIN (param_in (property_count), param_in (properties));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    if (property_count == nullptr || properties == nullptr)
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

    const property_t::properties_map_t &all_properties_map
      = property_t::all ();
    size_t byte_size
      = all_properties_map.size () * sizeof (amd_dbgapi_property_id_t);
    auto list
      = amd::dbgapi::allocate_memory<amd_dbgapi_property_id_t[]> (byte_size);
    size_t cnt = 0;
    for (const property_t::properties_map_element_t &property :
         all_properties_map)
      list[cnt++] = property.first;

    /* The underlying map is unordered, sort by handle to report the
       properties in a deterministic way.  */
    std::sort (list.get (), list.get () + cnt,
               [] (const amd_dbgapi_property_id_t &a,
                   const amd_dbgapi_property_id_t &b)
               { return a.handle < b.handle; });

    *property_count = cnt;
    *properties = list.release ();
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_ref (param_out (property_count)),
             make_ref (make_ref (param_out (properties)), *property_count));
}

amd_dbgapi_status_t AMD_DBGAPI
amd_dbgapi_property_get_info (amd_dbgapi_property_id_t property_id,
                              amd_dbgapi_property_info_t query,
                              size_t value_size, void *value)
{
  TRACE_BEGIN (param_in (property_id), param_in (query), param_in (value_size),
               param_in (value));
  TRY
  {
    if (!detail::is_initialized)
      THROW (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);

    const property_t::properties_map_t &all_properties_map
      = property_t::all ();

    if (all_properties_map.find (property_id) == all_properties_map.end ())
      THROW (AMD_DBGAPI_STATUS_ERROR_INVALID_PROPERTY_ID);

    property_t::property_map_value_t &property
      = all_properties_map.at (property_id);

    property->get_info (query, value_size, value);
  }
  CATCH (AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED,
         AMD_DBGAPI_STATUS_ERROR_INVALID_PROPERTY_ID,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT,
         AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY,
         AMD_DBGAPI_STATUS_ERROR_CLIENT_CALLBACK);
  TRACE_END (make_query_ref (query, param_out (value)));
}
