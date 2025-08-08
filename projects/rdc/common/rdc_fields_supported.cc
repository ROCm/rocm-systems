/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "common/rdc_fields_supported.h"

#include <assert.h>

#include <algorithm>

#include "rdc/rdc.h"
namespace amd {
namespace rdc {

#define FLD_DESC_ENT(ID, DESC, LABEL, DISPLAY) \
  {static_cast<uint32_t>(ID), {#ID, (DESC), (LABEL), (DISPLAY)}},
static const fld_id2name_map_t field_id_to_descript = {
#include "common/rdc_field.data"
};
#undef FLD_DESC_ENT

#define FLD_DESC_ENT(ID, DESC, LABEL, DISPLAY) {#ID, (ID)},
static fld_name2id_map_t field_name_to_id = {
#include "common/rdc_field.data"
};
#undef FLD_DESC_ENT

amd::rdc::fld_id2name_map_t& get_field_id_description_from_id(void) { return field_id_to_descript; }

bool get_field_id_from_name(const std::string name, rdc_field_t* value) {
  assert(value != nullptr);
  auto id = field_name_to_id.find(name);
  if (id == field_name_to_id.end()) {
    return false;
  }

  *value = static_cast<rdc_field_t>(id->second);
  return true;
}

bool is_field_valid(rdc_field_t field_id) {
  if (field_id == RDC_FI_INVALID) {
    return false;
  }
  return field_id_to_descript.find(static_cast<uint32_t>(field_id)) != field_id_to_descript.end();
}

}  // namespace rdc
}  // namespace amd
