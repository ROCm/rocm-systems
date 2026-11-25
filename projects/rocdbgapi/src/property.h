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

#ifndef AMD_DBGAPI_PROPERTY_H
#define AMD_DBGAPI_PROPERTY_H 1

#include "amd-dbgapi.h"
#include "handle_object.h"

#include <unordered_map>

namespace amd::dbgapi
{

/* Property IDs for every properties.  */

enum class property_id_t : uint32_t
{
  none,

  /* Report memory exceptions with the exact incurring PC address.  */
  precise_memory_exception,

  /* Report ALU exceptions with the exact incurring PC address.  */
  precise_alu_exception,

  /* Enable LDS out-of-range address exceptions.  */
  lds_exception,
};

class property_t : public detail::handle_object<amd_dbgapi_property_id_t>
{
public:
  /* All the types that the properties may have.  */
  enum property_type_t
  {
    boolean, /* Of an enable/disable nature.  */
  };

  /* Properties scope

     The context(s) that the property is valid for.  */
  enum property_scope_t
  {
    process = 1 << 0,
  };

  property_t (amd_dbgapi_property_id_t id, std::string name, std::string desc,
              property_type_t type = property_type_t::boolean,
              property_scope_t scope = property_scope_t::process)
    : handle_object (id), m_name (std::move (name)), m_desc (std::move (desc)),
      m_type (type), m_scope (scope)
  {
  }

  const std::string &name () const { return m_name; }
  const std::string &desc () const { return m_desc; }
  property_type_t type () const { return m_type; }
  property_scope_t scope () const { return m_scope; }
  /* The VALUE must comply with property_type_t.  */
  void get_info (amd_dbgapi_property_info_t query,
                 size_t value_size, void *value) const;
  void set (process_t *proc, size_t value_size, const void *value) const;

  using property_map_value_t = const std::unique_ptr<property_t>;
  using properties_map_t
    = std::unordered_map<amd_dbgapi_property_id_t, property_map_value_t,
                         hash<amd_dbgapi_property_id_t>>;
  using properties_map_element_t
    = std::pair<const amd_dbgapi_property_id_t, property_map_value_t>;

  /* Return the map of all instantiated properties.  */
  static const properties_map_t &all () { return s_property_map; }

private:
  std::string m_name;
  std::string m_desc;
  property_type_t m_type;
  property_scope_t m_scope;

  /* A global map from ID (in the form of amd_dbgapi_property_t.id)
     to property itself.  */
  static properties_map_t s_property_map;
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_PROPERTY_H */
