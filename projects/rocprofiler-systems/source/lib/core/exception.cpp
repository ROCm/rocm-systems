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

#include "exception.hpp"

#include <any>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <new>
#include <regex>
#include <stdexcept>
#include <system_error>
#include <timemory/unwind/backtrace.hpp>
#include <timemory/utility/backtrace.hpp>
#include <typeinfo>
#include <variant>

namespace rocprofsys
{
namespace
{
template <typename... Args>
void
consume_args(Args&&...)
{}

template <typename... Args>
auto
get_backtrace(Args... _arg)
{
    consume_args(_arg...);
    auto _bt = std::stringstream{};
    if constexpr(sizeof...(Args) > 0)
    {
        ((_bt << _arg), ...) << "\n";
    }
    tim::unwind::detailed_backtrace<2>(_bt, true);
    return strdup(_bt.str().c_str());
}
}  // namespace

template <typename Tp>
exception<Tp>::exception(const std::string& _msg)
: Tp{ _msg }
, m_what{ get_backtrace(_msg) }
{}

template <typename Tp>
exception<Tp>::exception(const char* _msg)
: Tp{ _msg }
, m_what{ get_backtrace(_msg) }
{}

template <typename Tp>
exception<Tp>::exception(const std::string& _msg, bool with_backtrace)
: Tp{ _msg }
, m_what{ with_backtrace ? get_backtrace(_msg) : strdup(_msg.c_str()) }
{}

template <typename Tp>
exception<Tp>::~exception()
{
    free(m_what);
}

template <typename Tp>
exception<Tp>::exception(const exception& _rhs)
: Tp{ _rhs }
, m_what{ strdup(_rhs.m_what) }
{}

template <typename Tp>
exception<Tp>&
exception<Tp>::operator=(const exception& _rhs)
{
    if(this != &_rhs)
    {
        Tp::operator=(_rhs);
        m_what = strdup(_rhs.m_what);
    }
    return *this;
}

template <typename Tp>
const char*
exception<Tp>::what() const noexcept
{
    return (m_what) ? m_what : Tp::what();
}

template class exception<std::runtime_error>;
template class exception<std::logic_error>;
template class exception<std::length_error>;
template class exception<std::out_of_range>;
template class exception<std::invalid_argument>;
template class exception<std::domain_error>;
template class exception<std::range_error>;
template class exception<std::overflow_error>;
template class exception<std::underflow_error>;
// template class exception<std::future_error>;
// template class exception<std::regex_error>;
// template class exception<std::system_error>;
// template class exception<std::bad_exception>;
// template class exception<std::bad_function_call>;
// template class exception<std::bad_alloc>;
// template class exception<std::bad_array_new_length>;
// template class exception<std::bad_cast>;
// template class exception<std::bad_typeid>;
// template class exception<std::bad_weak_ptr>;
// template class exception<std::bad_any_cast>;
// template class exception<std::bad_variant_access>;
}  // namespace rocprofsys
