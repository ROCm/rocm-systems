// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output_file_registry.hpp"

#include "logger/debug.hpp"

#include <iostream>
#include <string_view>
#include <utility>

namespace rocprofsys
{

output_file
output_file_registry::make_entry(std::string path, output_format format,
                                 const std::string& component_name)
{
    switch(format)
    {
        case output_format::perfetto:
            return { "Perfetto trace", std::move(path),
                     "Open in https://ui.perfetto.dev" };
        case output_format::rocpd:
            return { "RocPD database", std::move(path),
                     "sqlite3, AMD Visualizer (OPTIQ), or rocprofiler-sdk provided rocpd "
                     "Python module for conversion to other formats" };
        case output_format::json:
            return { component_name.empty() ? "JSON output"
                                            : fmt::format("JSON ({})", component_name),
                     path, fmt::format("jq . {}", path) };
        case output_format::text:
            return { component_name.empty() ? "Text profile"
                                            : fmt::format("Profile ({})", component_name),
                     path, fmt::format("cat {}", path) };
        case output_format::causal_json:
            return { "Causal profile (JSON)", path, fmt::format("jq . {}", path) };
        case output_format::causal_text:
            return { "Causal profile (text)", path, fmt::format("cat {}", path) };
    }
    return { "Unknown", std::move(path), "" };
}

void
output_file_registry::register_file(std::string path, output_format format)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back(make_entry(std::move(path), format));
}

void
output_file_registry::register_file(std::string path, output_format format,
                                    std::string component_name)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.push_back(make_entry(std::move(path), format, component_name));
}

void
output_file_registry::print_summary() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if(m_files.empty()) return;

    constexpr std::string_view header =
        "\n"
        "  ┌──────────────────────────────────────────────┐\n"
        "  │            Output Summary                    │\n"
        "  └──────────────────────────────────────────────┘\n";

    auto _msg = std::string{ header };

    auto it = m_files.begin();
    while(it != m_files.end())
    {
        auto        next    = std::next(it);
        bool        is_last = (next == m_files.end());
        const auto* branch  = is_last ? "└─" : "├─";
        const auto* cont    = is_last ? "  " : "│ ";

        _msg += fmt::format("  {} {}\n"
                            "  {}   File: {}\n"
                            "  {}   View with: {}\n",
                            branch, it->label, cont, it->path, cont, it->viewer);
        if(!is_last) _msg += "  │\n";
        it = next;
    }

    std::cout << _msg;
}

void
output_file_registry::clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_files.clear();
}

}  // namespace rocprofsys
