/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "amd_smi/impl/amd_smi_addc.h"

#include <cctype>

#ifdef AMDSMI_ENABLE_ADDC
#include "addc/lib_c.h"
#endif

namespace amdsmi {
namespace addc {

// Dependency-free scan of ADDC's error-summary JSON. Assumes ADDC's fixed
// schema where the token "afid" only ever appears as an object key.
bool parse_afids_from_summary(const std::string& summary_json, std::vector<uint64_t>& out) {
  out.clear();

  size_t i = 0;
  while (i < summary_json.size() && std::isspace(static_cast<unsigned char>(summary_json[i]))) {
    ++i;
  }
  if (i >= summary_json.size() || summary_json[i] != '[') {
    return false;
  }

  const std::string key = "\"afid\"";
  for (size_t pos = summary_json.find(key); pos != std::string::npos;
       pos = summary_json.find(key, pos + key.size())) {
    size_t j = pos + key.size();
    while (j < summary_json.size() && std::isspace(static_cast<unsigned char>(summary_json[j]))) {
      ++j;
    }
    if (j >= summary_json.size() || summary_json[j] != ':') {
      continue;
    }
    ++j;
    while (j < summary_json.size() && std::isspace(static_cast<unsigned char>(summary_json[j]))) {
      ++j;
    }
    const size_t start = j;
    while (j < summary_json.size() && std::isdigit(static_cast<unsigned char>(summary_json[j]))) {
      ++j;
    }
    if (j == start) {
      continue;  // no digits: negative, string, or other non-unsigned value
    }
    const char after = (j < summary_json.size()) ? summary_json[j] : '\0';
    if (after == '.' || after == 'e' || after == 'E') {
      continue;  // float, not an unsigned integer
    }
    uint64_t value = 0;
    for (size_t k = start; k < j; ++k) {
      value = value * 10 + static_cast<uint64_t>(summary_json[k] - '0');
    }
    out.push_back(value);
  }
  return true;
}

#ifdef AMDSMI_ENABLE_ADDC
amdsmi_status_t get_afids_via_addc(const char* cper_buffer, uint32_t buf_size,
                                   std::vector<uint64_t>& out) {
  out.clear();

  char* json_out = nullptr;
  char* error_out = nullptr;
  int rc = addc_get_error_summary(reinterpret_cast<const uint8_t*>(cper_buffer), buf_size,
                                  /*filename=*/nullptr, &json_out, &error_out);
  if (rc != 0 || json_out == nullptr) {
    addc_free_string(json_out);
    addc_free_string(error_out);
    return AMDSMI_STATUS_UNEXPECTED_DATA;
  }

  const bool ok = parse_afids_from_summary(json_out, out);
  addc_free_string(json_out);
  addc_free_string(error_out);
  return ok ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_UNEXPECTED_DATA;
}
#endif  // AMDSMI_ENABLE_ADDC

}  // namespace addc
}  // namespace amdsmi
