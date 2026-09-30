// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace addc
{

/// Stable common-schema key for non-fatal decode and recovery diagnostics.
/// The field is omitted when empty. Frontends may also emit these entries to
/// their logging system, but must not treat them as decode failures.
inline constexpr const char* kDiagnosticsKey = "diagnostics";

enum class DiagLevel
{
    Debug,
    Info,
    Warning,
    Error
};

[[nodiscard]] inline std::string_view to_string(DiagLevel level)
{
    switch (level)
    {
        case DiagLevel::Debug:
            return "debug";
        case DiagLevel::Info:
            return "info";
        case DiagLevel::Warning:
            return "warning";
        case DiagLevel::Error:
            return "error";
    }
    return "info";
}

struct Diagnostic
{
    DiagLevel level;
    std::string code;
    std::string message;
    nlohmann::ordered_json context = nullptr;
};

class Diagnostics
{
  public:
    void debug(std::string code, std::string message,
               nlohmann::ordered_json context = nullptr)
    {
        add(DiagLevel::Debug, std::move(code), std::move(message),
            std::move(context));
    }
    void info(std::string code, std::string message,
              nlohmann::ordered_json context = nullptr)
    {
        add(DiagLevel::Info, std::move(code), std::move(message),
            std::move(context));
    }
    void warn(std::string code, std::string message,
              nlohmann::ordered_json context = nullptr)
    {
        add(DiagLevel::Warning, std::move(code), std::move(message),
            std::move(context));
    }
    void error(std::string code, std::string message,
               nlohmann::ordered_json context = nullptr)
    {
        add(DiagLevel::Error, std::move(code), std::move(message),
            std::move(context));
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return items_.empty();
    }

    [[nodiscard]] std::size_t count(DiagLevel level) const noexcept
    {
        std::size_t n = 0;
        for (const auto& d : items_)
        {
            if (d.level == level)
            {
                ++n;
            }
        }
        return n;
    }

    [[nodiscard]] const std::vector<Diagnostic>& items() const noexcept
    {
        return items_;
    }

    [[nodiscard]] nlohmann::ordered_json to_json() const
    {
        nlohmann::ordered_json arr = nlohmann::ordered_json::array();
        for (const auto& d : items_)
        {
            nlohmann::ordered_json entry;
            entry["level"] = to_string(d.level);
            entry["code"] = d.code;
            entry["message"] = d.message;
            if (!d.context.is_null())
            {
                entry["context"] = d.context;
            }
            arr.push_back(std::move(entry));
        }
        return arr;
    }

  private:
    void add(DiagLevel level, std::string code, std::string message,
             nlohmann::ordered_json context)
    {
        items_.push_back(Diagnostic{level, std::move(code), std::move(message),
                                    std::move(context)});
    }

    std::vector<Diagnostic> items_;
};

} // namespace addc
