// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bounded string helpers used by fdinfo.cc to populate the fixed-size char
// arrays in amdsmi_proc_info_t from /proc content.
//
// Two extractors, because cgroup paths name containers two different ways.
// Every OCI runtime (Docker, containerd, CRI-O, Podman) names the cgroup
// after the full 64-char SHA-256, varying only in the runtime prefix and an
// optional ".scope" suffix, so ExtractOciContainerId matches the identifier
// itself and covers all of them -- and therefore Kubernetes -- without
// enumerating runtimes. ExtractContainerId handles what carries no SHA-256:
// LXC names and Docker short IDs.
//
// Both anchor on cgroup path separators so a substring cannot produce a false
// positive ("/not-docker-evil/..." must not match "docker"), and both refuse
// rather than truncate, because a caller cannot tell a clipped prefix from a
// complete ID.
//
// Header-only so fdinfo.cc and the unit tests share one implementation.

#ifndef AMD_SMI_INCLUDE_AMD_SMI_CONTAINER_ID_PARSER_H_
#define AMD_SMI_INCLUDE_AMD_SMI_CONTAINER_ID_PARSER_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

namespace amd::smi {

// Copy `src` into `out` (capacity `out_cap`), always NUL-terminating.
// Returns the number of bytes written, not counting the NUL.
inline size_t CopyBounded(char* out, size_t out_cap, const std::string& src) {
  if (out_cap == 0) return 0;
  const size_t len = std::min<size_t>(out_cap - 1, src.length());
  std::memcpy(out, src.data(), len);
  out[len] = '\0';
  return len;
}

// Character class accepted in extracted container IDs.
inline bool IsContainerIdChar(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         c == '-';
}

// Length of the SHA-256 identifier OCI runtimes name container cgroups after
// (Docker's `fullLen`, moby client/pkg/stringid/stringid.go).
inline constexpr size_t kOciContainerIdLength = 64;

inline bool IsLowerHexChar(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
}

// Extract the OCI container ID from a cgroup `line` into `out` (capacity
// `out_cap`, always NUL-terminated on non-zero capacity). Returns the number
// of bytes written, not counting the NUL, or 0 if the line holds no ID or the
// ID does not fit.
//
// Matches the first maximal run of exactly kOciContainerIdLength lowercase hex
// digits that is delimited like a cgroup path component: preceded by '/' or
// '-' (or start-of-line) and followed by '/', '.' or end-of-line. Any runtime
// prefix and the ".scope" suffix are therefore excluded from the result, which
// is what `docker inspect` and the CRI APIs expect to be given.
inline size_t ExtractOciContainerId(const std::string& line, char* out, size_t out_cap) {
  if (out_cap == 0) return 0;
  out[0] = '\0';
  // An ID that does not fit is reported as absent rather than truncated.
  if (out_cap <= kOciContainerIdLength) return 0;

  const size_t line_len = line.size();
  size_t pos = 0;
  while (pos < line_len) {
    if (!IsLowerHexChar(static_cast<unsigned char>(line[pos]))) {
      ++pos;
      continue;
    }
    const size_t id_start = pos;
    while (pos < line_len && IsLowerHexChar(static_cast<unsigned char>(line[pos]))) ++pos;
    if (pos - id_start != kOciContainerIdLength) continue;

    const bool prefix_ok =
        (id_start == 0) || (line[id_start - 1] == '/') || (line[id_start - 1] == '-');
    const bool suffix_ok = (pos == line_len) || (line[pos] == '/') || (line[pos] == '.');
    if (prefix_ok && suffix_ok) {
      std::memcpy(out, line.data() + id_start, kOciContainerIdLength);
      out[kOciContainerIdLength] = '\0';
      return kOciContainerIdLength;
    }
  }
  return 0;
}

// Locate `type_name` inside `line` where both surrounding boundaries are
// cgroup-path separators: the byte before must be '/' (or start-of-line),
// and the byte after must be '/' or '-'. Returns std::string::npos if no
// anchored match exists.
inline size_t FindAnchoredContainerType(const std::string& line, const char* type_name) {
  const size_t type_len = std::strlen(type_name);
  if (type_len == 0) return std::string::npos;
  size_t pos = 0;
  while ((pos = line.find(type_name, pos)) != std::string::npos) {
    const bool prefix_ok = (pos == 0) || (line[pos - 1] == '/');
    const size_t after = pos + type_len;
    const bool suffix_ok = (after < line.size()) && (line[after] == '/' || line[after] == '-');
    if (prefix_ok && suffix_ok) return pos;
    ++pos;
  }
  return std::string::npos;
}

// Extract a container ID for `type_name` from a cgroup `line` into the
// caller-supplied `out` buffer (capacity `out_cap`, always NUL-terminated
// on non-zero capacity). Returns the number of bytes written, not counting
// the NUL, or 0 if no valid ID is found or the ID does not fit in `out_cap`.
inline size_t ExtractContainerId(const std::string& line, const char* type_name, char* out,
                                 size_t out_cap) {
  if (out_cap == 0) return 0;
  out[0] = '\0';

  const size_t type_pos = FindAnchoredContainerType(line, type_name);
  if (type_pos == std::string::npos) return 0;

  const size_t id_start = type_pos + std::strlen(type_name) + 1;
  if (id_start >= line.size()) return 0;

  size_t id_len = 0;
  while (id_len < out_cap && id_start + id_len < line.size() &&
         IsContainerIdChar(static_cast<unsigned char>(line[id_start + id_len]))) {
    ++id_len;
  }
  // id_len == out_cap means the ID needs at least out_cap bytes plus a NUL,
  // so it cannot be represented; report no ID rather than a silent prefix.
  if (id_len == 0 || id_len == out_cap) return 0;

  std::memcpy(out, line.data() + id_start, id_len);
  out[id_len] = '\0';
  return id_len;
}

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_CONTAINER_ID_PARSER_H_
