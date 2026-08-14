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

// Container-ID parser used by fdinfo.cc to extract container identifiers
// from /proc/<pid>/cgroup lines.
//
// Design (see tests/amd_smi_test/unit/container_id_parser/ for the full
// threat model):
//
//   1. Anchored type_name match — the container-type string ("docker",
//      "lxc") must be preceded by '/' (or start-of-line) AND followed by
//      '/' or '-'. Prevents substring false positives (CWE-20) such as
//      "/not-docker-evil/..." matching "docker".
//
//   2. Charset whitelist [a-zA-Z0-9_-] — the union of Docker SHA-256 hex
//      and LXC container-name charsets. Rejects at the byte level:
//        - control bytes and CR/LF (CWE-117 log injection)
//        - non-ASCII UTF-8 (CWE-1007 Trojan Source / homoglyphs)
//        - embedded NUL (CWE-626 smuggling)
//        - shell metacharacters (CWE-78 OS command injection)
//        - path separators (CWE-22 traversal)
//
//   3. Bounded length — capped at AMDSMI_MAX_CONTAINER_ID_LENGTH (64),
//      matching Docker's fullLen = 64 from moby/moby client/pkg/stringid.
//      Constant-time worst case in the consume loop (CWE-400 resistant).
//
// Header-only / inline so that both the production code (fdinfo.cc) and
// the unit tests link to a single canonical implementation — avoiding
// drift between the two.

#ifndef AMD_SMI_INCLUDE_AMD_SMI_CONTAINER_ID_PARSER_H_
#define AMD_SMI_INCLUDE_AMD_SMI_CONTAINER_ID_PARSER_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

#include "amd_smi/amdsmi.h"

namespace amd::smi {

// Character class accepted in extracted container IDs.
inline bool IsContainerIdChar(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
         (c >= 'A' && c <= 'Z') || c == '_' || c == '-';
}

// Locate `type_name` inside `line` where both surrounding boundaries are
// cgroup-path separators: the byte before must be '/' (or start-of-line),
// and the byte after must be '/' or '-'. Returns std::string::npos if no
// anchored match exists.
inline size_t FindAnchoredContainerType(const std::string& line,
                                        const char* type_name) {
  const size_t type_len = std::strlen(type_name);
  if (type_len == 0) return std::string::npos;
  size_t pos = 0;
  while ((pos = line.find(type_name, pos)) != std::string::npos) {
    const bool prefix_ok = (pos == 0) || (line[pos - 1] == '/');
    const size_t after = pos + type_len;
    const bool suffix_ok =
        (after < line.size()) && (line[after] == '/' || line[after] == '-');
    if (prefix_ok && suffix_ok) return pos;
    ++pos;
  }
  return std::string::npos;
}

// Extract a container ID for `type_name` from a cgroup `line` into the
// caller-supplied `out` buffer (capacity `out_cap`, always NUL-terminated
// on non-zero capacity). Returns the number of bytes written, not counting
// the NUL, or 0 if no valid ID is found.
inline size_t ExtractContainerId(const std::string& line,
                                 const char* type_name, char* out,
                                 size_t out_cap) {
  if (out_cap == 0) return 0;
  out[0] = '\0';

  const size_t type_pos = FindAnchoredContainerType(line, type_name);
  if (type_pos == std::string::npos) return 0;

  const size_t id_start = type_pos + std::strlen(type_name) + 1;
  if (id_start >= line.size()) return 0;

  const size_t max_len = std::min<size_t>(
      static_cast<size_t>(AMDSMI_MAX_CONTAINER_ID_LENGTH), out_cap - 1);

  size_t id_len = 0;
  while (id_len < max_len && id_start + id_len < line.size() &&
         IsContainerIdChar(
             static_cast<unsigned char>(line[id_start + id_len]))) {
    ++id_len;
  }
  if (id_len == 0) return 0;

  std::memcpy(out, line.data() + id_start, id_len);
  out[id_len] = '\0';
  return id_len;
}

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_CONTAINER_ID_PARSER_H_
