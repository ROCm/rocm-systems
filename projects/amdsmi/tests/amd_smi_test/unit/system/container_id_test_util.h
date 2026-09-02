// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared inputs and value-returning wrappers over the production container-ID
// parser, for the equality-style assertions that do not care about the output
// buffer. Tests that exercise the buffer contract call the parser directly.

#ifndef AMDSMI_TESTS_UNIT_SYSTEM_CONTAINER_ID_TEST_UTIL_H_
#define AMDSMI_TESTS_UNIT_SYSTEM_CONTAINER_ID_TEST_UTIL_H_

#include <cstddef>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_container_id_parser.h"

namespace amdsmi_test {

// Valid 64-char lowercase hex SHA-256, as produced by Docker/containerd.
// 64 is Docker's `fullLen` in client/pkg/stringid/stringid.go:
// https://github.com/moby/moby/blob/2200f277f9f576886e90ca75929a2bb892b9ef23/client/pkg/stringid/stringid.go#L14-L15
inline constexpr char kDocker64[] =
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

inline std::string ExtractIdString(const std::string& line, const char* prefix) {
  char buf[AMDSMI_MAX_STRING_LENGTH] = {0};
  amd::smi::ExtractContainerId(line, prefix, buf, sizeof(buf));
  return std::string(buf);
}

inline std::string ExtractOciIdString(const std::string& line) {
  char buf[AMDSMI_MAX_STRING_LENGTH] = {0};
  amd::smi::ExtractOciContainerId(line, buf, sizeof(buf));
  return std::string(buf);
}

}  // namespace amdsmi_test

#endif  // AMDSMI_TESTS_UNIT_SYSTEM_CONTAINER_ID_TEST_UTIL_H_
