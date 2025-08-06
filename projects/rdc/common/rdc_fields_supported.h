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
#ifndef COMMON_RDC_FIELDS_SUPPORTED_H_
#define COMMON_RDC_FIELDS_SUPPORTED_H_

#include <map>
#include <string>
#include <unordered_map>

#include "rdc/rdc.h"

namespace amd {
namespace rdc {

typedef struct {
  std::string enum_name;
  std::string description;
  std::string label;
  bool do_display;
} field_id_descript;

typedef const std::map<uint32_t, const field_id_descript> fld_id2name_map_t;
typedef std::unordered_map<std::string, uint32_t> fld_name2id_map_t;

bool get_field_id_from_name(const std::string name, rdc_field_t* value);
fld_id2name_map_t& get_field_id_description_from_id(void);  // NOLINT
bool is_field_valid(rdc_field_t field_id);

}  // namespace rdc
}  // namespace amd

#endif  // COMMON_RDC_FIELDS_SUPPORTED_H_
