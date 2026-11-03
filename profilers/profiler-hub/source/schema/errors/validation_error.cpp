// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "validation_error.hpp"

#include <ostream>
#include <sstream>
#include <string>

namespace profiler_hub::schema
{

std::string
documentation_link(profiler_hub::version_t version)
{
    return "docs/schema/" + version.to_string() + "_usage.md";
}

namespace
{

void
write_indented(std::ostringstream& out, std::string_view text)
{
    std::size_t start = 0;
    while(start < text.size())
    {
        const auto end  = text.find('\n', start);
        const auto line = text.substr(
            start, end == std::string_view::npos ? text.size() - start : end - start);
        out << "    " << line << '\n';
        if(end == std::string_view::npos) break;
        start = end + 1;
    }
}

}  // namespace

std::string
validation_result::format() const
{
    if(ok)
    {
        return "Validation Succeeded\n";
    }

    std::ostringstream out;
    out << "Validation Failed\n\n";
    out << "Schema:\n    " << schema_version << "\n\n";
    out << "Object:\n    " << object << "\n\n";

    for(const auto& error : errors)
    {
        if(!error.field.empty())
        {
            out << "Field:\n    " << error.field << "\n\n";
        }
        out << "Reason:\n";
        write_indented(out, error.reason);
        out << '\n';
    }

    const std::string link =
        errors.empty() ? std::string{} : errors.front().documentation;
    out << "Documentation:\n    " << link << '\n';
    return out.str();
}

void
validation_result::print(std::ostream& out) const
{
    out << format();
}

}  // namespace profiler_hub::schema
