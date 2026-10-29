// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/parse_result.hpp"
#include "parser/schema.hpp"

#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofsys::parser
{
/**
 * Reads the environment variable named by each parameter's env spec.
 * Unset variables are skipped. The lookup is injected so tests do not touch
 * the process environment; from_process() reads it with std::getenv.
 */
class env_parser
{
public:
    using lookup_t = std::function<std::optional<std::string>(std::string_view)>;

    explicit env_parser(lookup_t lookup);

    [[nodiscard]] static env_parser from_process();

    [[nodiscard]] parse_result parse(const schema& config_schema) const;

private:
    lookup_t m_lookup;
};

}  // namespace rocprofsys::parser
