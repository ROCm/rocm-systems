/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

#ifndef _ROCM_DEBUG_AGENT_CODE_OBJECT_H
#define _ROCM_DEBUG_AGENT_CODE_OBJECT_H 1

#include <amd-dbgapi/amd-dbgapi.h>

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace amd::debug_agent
{

class code_object_t
{
private:
  struct symbol_info_t
  {
    const std::string m_name;
    amd_dbgapi_global_address_t m_value;
    amd_dbgapi_size_t m_size;
  };

  void load_symbol_map ();
  void load_debug_info ();

public:
  code_object_t (amd_dbgapi_code_object_id_t code_object_id);
  code_object_t (code_object_t &&rhs);

  ~code_object_t ();

  void open ();
  bool is_open () const { return m_fd.has_value (); }

  amd_dbgapi_global_address_t load_address () const { return m_load_address; }
  amd_dbgapi_size_t mem_size () const { return m_mem_size; }

  std::optional<symbol_info_t>
  find_symbol (amd_dbgapi_global_address_t address);

  void disassemble (amd_dbgapi_architecture_id_t architecture_id,
                    amd_dbgapi_global_address_t pc);

  bool save (const std::string &directory) const;

private:
  amd_dbgapi_global_address_t m_load_address{ 0 };
  amd_dbgapi_size_t m_mem_size{ 0 };
  std::optional<int> m_fd;

  std::optional<
      std::map<amd_dbgapi_global_address_t, std::pair<std::string, size_t>>>
      m_line_number_map;

  std::optional<
      std::map<amd_dbgapi_global_address_t, amd_dbgapi_global_address_t>>
      m_pc_ranges_map;

  std::optional<std::map<amd_dbgapi_global_address_t,
                         std::pair<std::string, amd_dbgapi_size_t>>>
      m_symbol_map;

  std::string m_uri;
  amd_dbgapi_code_object_id_t const m_code_object_id;
};

} /* namespace amd::debug_agent */

#endif /* _ROCM_DEBUG_AGENT_CODE_OBJECT_H */
