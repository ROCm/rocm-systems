// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tool_result.h
/// @brief Small result type shared by internal rocjitsu tool entry points.

#ifndef ROCJITSU_TOOLS_TOOL_RESULT_H_
#define ROCJITSU_TOOLS_TOOL_RESULT_H_

#include <string>
#include <vector>

namespace rocjitsu::tools {

/// @brief Structured error returned by a tool entry point.
///
/// The exit code is intentionally stored with the error so CLI wrappers can
/// return the same code that in-process tests would assert on. This keeps tests
/// from scraping stderr just to decide what failed.
struct ToolError {
  int exit_code = 1;
  std::string message;
};

/// @brief Value-or-errors result for tool functions.
template <typename T> struct ToolResult {
  T value;
  std::vector<ToolError> errors;

  [[nodiscard]] bool ok() const { return errors.empty(); }
};

} // namespace rocjitsu::tools

#endif // ROCJITSU_TOOLS_TOOL_RESULT_H_
