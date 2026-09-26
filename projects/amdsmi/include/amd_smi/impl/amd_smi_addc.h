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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_ADDC_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_ADDC_H_

#include <cstdint>
#include <string>
#include <vector>

#ifdef AMDSMI_ENABLE_ADDC
#include "amd_smi/amdsmi.h"
#endif

namespace amdsmi {
namespace addc {

// Parses an ADDC error-summary JSON document of the form
// [{"fru":"..","afid":N}, ..] into the AFIDs it carries. Entries without an
// "afid" member are skipped. Trusting ADDC's fixed schema, it validates only
// that the root is a JSON array (returns false with out cleared if not) and
// does not reject otherwise malformed or truncated input. An empty array
// yields true with an empty result.
bool parse_afids_from_summary(const std::string& summary_json, std::vector<uint64_t>& out);

#ifdef AMDSMI_ENABLE_ADDC
// Decodes the AFIDs in a single raw CPER record by routing the record through
// the ADDC library's error-summary C API. On success returns
// AMDSMI_STATUS_SUCCESS and fills out; out is cleared on failure. Note this
// differs from the default ras-decode path: a record ADDC cannot summarize
// yields AMDSMI_STATUS_UNEXPECTED_DATA rather than success with zero AFIDs.
amdsmi_status_t get_afids_via_addc(const char* cper_buffer, uint32_t buf_size,
                                   std::vector<uint64_t>& out);
#endif  // AMDSMI_ENABLE_ADDC

}  // namespace addc
}  // namespace amdsmi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_ADDC_H_
