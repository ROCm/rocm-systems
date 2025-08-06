// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
