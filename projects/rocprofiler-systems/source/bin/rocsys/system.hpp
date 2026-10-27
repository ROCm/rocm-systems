// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/tool_runner.hpp"

#include <concepts>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::rocsys
{

/// @brief Process boundary used by @ref launcher. Tests substitute their own model.
///        `run_tool` and `exec` take the C argv that @ref launcher builds. The real
///        model forwards those to `run_tool` and `execve`.
template <typename T>
concept system_ops = requires(T& system, const char* text, int argc, char** argv,
                              common_utils::tool_mode mode, std::string_view message) {
    { system.set_env(text, text) } -> std::same_as<int>;
    { system.unset_env(text) } -> std::same_as<int>;
    { system.exists(text) } -> std::same_as<bool>;
    { system.executable_directory() } -> std::convertible_to<std::string>;
    { system.environ() } -> std::convertible_to<std::vector<std::string>>;
    { system.run_tool(argc, argv, mode) } -> std::same_as<int>;
    { system.exec(text, argv, argv) } -> std::same_as<int>;
    { system.write_error(message) } -> std::same_as<void>;
};

/// @brief The real process: setenv, execve, and the sibling directory of this binary.
class default_system
{
public:
    [[nodiscard]] int set_env(const char* key, const char* value) const;

    [[nodiscard]] int unset_env(const char* key) const;

    [[nodiscard]] bool exists(const char* path) const;

    /// @brief Directory of this executable, with symlinks resolved. Empty on failure.
    [[nodiscard]] std::string executable_directory() const;

    [[nodiscard]] std::vector<std::string> environ() const;

    [[nodiscard]] int run_tool(int argc, char** argv, common_utils::tool_mode mode) const;

    /// @brief Replace this process. Returns an errno value when execve fails.
    [[nodiscard]] int exec(const char* path, char** argv, char** envp) const;

    void write_error(std::string_view message) const;
};

static_assert(system_ops<default_system>);

}  // namespace rocprofsys::rocsys
