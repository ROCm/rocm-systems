// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "argparse.hpp"

#include <cstdint>
#include <deque>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace argparse
{

/**
 * Engine-agnostic accessor for parsed argument values.
 *
 * Custom actions receive a `parsed_values` instead of the underlying
 * `parser_t`, so descriptor-driven code never names the parsing engine.
 * The implementation lives in parsed_values.cpp (the only TU outside of
 * detail/ that knows about the concrete engine).
 *
 * Lifetime: non-owning. The engine pointer must outlive the accessor.
 * In practice `parsed_values` is constructed by the interpreter for the
 * duration of a single action callback, so the contract holds naturally.
 *
 * Unsupported value types deliberately fail at compile time via the
 * deleted primary template — callers that try `get<unknown_t>(...)` get
 * a "deleted function" error at the call site rather than an undefined
 * symbol at link time.
 */
class parsed_values
{
public:
    explicit parsed_values(parser_t& engine) noexcept
    : m_engine{ &engine }
    {}

    parsed_values(const parsed_values&)            = delete;
    parsed_values& operator=(const parsed_values&) = delete;
    parsed_values(parsed_values&&)                 = delete;
    parsed_values& operator=(parsed_values&&)      = delete;
    ~parsed_values()                               = default;

    template <typename Tp>
    [[nodiscard]] Tp get(std::string_view key) const = delete;

    [[nodiscard]] bool exists(std::string_view key) const;

    void set_use_color(bool enabled);

private:
    parser_t* m_engine;  // non-owning; parser must outlive *this
};

// Supported value types. Adding a new type requires:
//   1. a declaration here, and
//   2. a matching definition in parsed_values.cpp.
template <>
[[nodiscard]] bool parsed_values::get<bool>(std::string_view) const;

template <>
[[nodiscard]] int parsed_values::get<int>(std::string_view) const;

template <>
[[nodiscard]] std::int64_t parsed_values::get<std::int64_t>(std::string_view) const;

template <>
[[nodiscard]] double parsed_values::get<double>(std::string_view) const;

template <>
[[nodiscard]] std::string parsed_values::get<std::string>(std::string_view) const;

template <>
[[nodiscard]] std::set<std::string> parsed_values::get<std::set<std::string>>(
    std::string_view) const;

template <>
[[nodiscard]] std::vector<std::string> parsed_values::get<std::vector<std::string>>(
    std::string_view) const;

template <>
[[nodiscard]] std::deque<std::string> parsed_values::get<std::deque<std::string>>(
    std::string_view) const;

template <>
[[nodiscard]] std::vector<std::int64_t> parsed_values::get<std::vector<std::int64_t>>(
    std::string_view) const;

}  // namespace argparse
}  // namespace rocprofsys
