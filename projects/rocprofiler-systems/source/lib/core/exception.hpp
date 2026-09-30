// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <stdexcept>
#include <timemory/unwind/backtrace.hpp>
#include <timemory/utility/backtrace.hpp>
#include <type_traits>

namespace rocprofsys
{
template <typename Tp>
class exception : public Tp
{
public:
    explicit exception(const std::string& _msg);
    explicit exception(const char* _msg);
    explicit exception(const std::string& _msg, bool with_backtrace);

    ~exception() override;

    exception(exception&&) noexcept            = default;
    exception& operator=(exception&&) noexcept = default;

    exception(const exception&);
    exception& operator=(const exception&);

    const char* what() const noexcept override;

private:
    char* m_what = nullptr;
};
}  // namespace rocprofsys
