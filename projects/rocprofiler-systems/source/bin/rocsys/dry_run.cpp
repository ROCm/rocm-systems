// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dry_run.hpp"

#include <algorithm>
#include <ostream>
#include <string>

namespace rocprofsys::rocsys
{
namespace
{
constexpr std::string_view k_overrides_label =
    "rocsys overrides; legacy flags are applied later by the tool";
constexpr std::string_view k_embedded_label = "default in binary";

void
write_argv(std::ostream& out, std::span<const std::string> argv)
{
    std::string line;
    for(const auto& token : argv)
    {
        if(!line.empty())
        {
            line.push_back(' ');
        }
        line.append(token);
    }
    out << "argv: " << line << '\n';
}

void
write_op(std::ostream& out, const env_op& op)
{
    out << "  " << op.key << '=' << op.value << " origin=" << op.origin
        << " embed=" << (op.embed ? "true" : "false") << '\n';
}

void
write_ops(std::ostream& out, std::string_view label, const env_plan& ops, bool embed)
{
    out << label << '\n';
    for(const auto& op : ops)
    {
        if(op.embed == embed)
        {
            write_op(out, op);
        }
    }
}

[[nodiscard]] bool
has_embedded(const env_plan& ops)
{
    return std::ranges::any_of(ops, &env_op::embed);
}

}  // namespace

void
write_dry_run(std::ostream& out, std::string_view verb, std::span<const std::string> argv,
              const env_plan& ops)
{
    out << "verb: " << verb << '\n';
    write_argv(out, argv);
    write_ops(out, k_overrides_label, ops, false);
    if(has_embedded(ops))
    {
        write_ops(out, k_embedded_label, ops, true);
    }
}

}  // namespace rocprofsys::rocsys
