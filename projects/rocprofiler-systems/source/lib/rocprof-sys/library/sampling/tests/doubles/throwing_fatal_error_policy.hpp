// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace rocprofsys::sampling
{

class sampling_fatal_error : public std::runtime_error
{
public:
    sampling_fatal_error(char const* file, int line, std::string msg)
    : std::runtime_error(msg)
    , m_file(file)
    , m_line(line)
    {}

    [[nodiscard]] char const* file() const noexcept { return m_file; }
    [[nodiscard]] int         line() const noexcept { return m_line; }

private:
    char const* m_file;
    int         m_line;
};

}  // namespace rocprofsys::sampling

namespace rocprofsys::sampling::test
{

struct throwing_fatal_error_policy
{
    template <class... Args>
    [[noreturn]] void fatal(char const* file, int line, std::string_view msg,
                            Args const&...)
    {
        throw sampling_fatal_error(file, line, std::string(msg));
    }
};

}  // namespace rocprofsys::sampling::test
