// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "timemory/general/types.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/mpl/apply.hpp"

#include <array>
#include <cstring>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tim
{
namespace impl
{
template <typename... Tp>
struct source_location_constant_char;

template <>
struct source_location_constant_char<>
{
    static constexpr bool value = true;
};

template <>
struct source_location_constant_char<const char*>
{
    static constexpr bool value = true;
};

template <typename Tp>
struct source_location_constant_char<Tp>
{
    static constexpr bool value = std::is_same<Tp, const char*>::value;
};

template <typename Tp, typename Up, typename... Tail>
struct source_location_constant_char<Tp, Up, Tail...>
{
    static constexpr bool value = source_location_constant_char<Tp>::value &&
                                  source_location_constant_char<Up>::value &&
                                  source_location_constant_char<Tail...>::value;
};

}  // namespace impl

/// \class tim::source_location
/// \brief Provides source location information and variadic joining of source location
/// tags
class source_location
{
public:
    using join_type   = mpl::apply<std::string>;
    using result_type = std::tuple<std::string, size_t>;

private:
    template <typename... Tp>
    static constexpr bool is_constant_char()
    {
        return impl::source_location_constant_char<decay_t<Tp>...>::value;
    }

public:
    //
    //  Public types
    //

    //==================================================================================//
    //
    enum class mode : short
    {
        blank    = 0,
        basic    = 1,
        full     = 2,
        complete = 3
    };

    //==================================================================================//
    //
    struct captured
    {
    public:
        const result_type&  get() const { return m_result; }
        const std::string&  get_id() const { return std::get<0>(m_result); }
        const hash_value_t& get_hash() const { return std::get<1>(m_result); }

        explicit captured(result_type _result)
        : m_result(std::move(_result))
        {}

        TIMEMORY_DEFAULT_OBJECT(captured)

    private:
        friend class source_location;
        result_type m_result = result_type("", 0);

        template <typename... ArgsT, enable_if_t<(sizeof...(ArgsT) > 0), int> = 0>
        captured& set(const source_location& obj, ArgsT&&... _args)
        {
            switch(obj.m_mode)
            {
                case mode::blank:
                {
                    auto&& _tmp = join_type::join("", std::forward<ArgsT>(_args)...);
                    m_result    = result_type(_tmp, add_hash_id(_tmp));
                    break;
                }
                case mode::basic:
                case mode::full:
                case mode::complete:
                {
                    auto&& _suffix = join_type::join("", std::forward<ArgsT>(_args)...);
                    if(_suffix.empty())
                    {
                        m_result = result_type(obj.m_prefix, add_hash_id(obj.m_prefix));
                    }
                    else
                    {
                        auto _tmp = join_type::join('/', obj.m_prefix.c_str(), _suffix);
                        m_result  = result_type(_tmp, add_hash_id(_tmp));
                    }
                    break;
                }
            }
            return *this;
        }

        template <typename... ArgsT, enable_if_t<sizeof...(ArgsT) == 0, int> = 0>
        captured& set(const source_location& obj, ArgsT&&... _args)
        {
            tim::consume_parameters(std::forward<ArgsT>(_args)...);
            m_result = result_type(obj.m_prefix, add_hash_id(obj.m_prefix));
            return *this;
        }
    };

public:
    //
    //  Static functions
    //

    //==================================================================================//
    //
    template <typename... ArgsT>
    static captured get_captured_inline(const mode& _mode, const char* _func, int _line,
                                        const char* _fname, ArgsT&&... _args)
    {
        source_location _loc{ _mode, _func, _line, _fname,
                              std::forward<ArgsT>(_args)... };
        return _loc.get_captured(std::forward<ArgsT>(_args)...);
    }

public:
    //
    //  Constructors, destructors, etc.
    //

    //----------------------------------------------------------------------------------//
    //
    template <typename... ArgsT, enable_if_t<!is_constant_char<ArgsT...>(), int> = 0>
    source_location(const mode& _mode, const char* _func, int _line, const char* _fname,
                    ArgsT&&...)
    : m_mode(_mode)
    {
        switch(m_mode)
        {
            case mode::blank: break;
            case mode::basic: compute_data(_func); break;
            case mode::complete:
            case mode::full:
                compute_data(_func, _line, _fname, m_mode == mode::full);
                break;
        }
    }

    //----------------------------------------------------------------------------------//
    //  if a const char* is passed, assume it is constant
    //
    template <typename... ArgsT, enable_if_t<is_constant_char<ArgsT...>(), int> = 0>
    source_location(const mode& _mode, const char* _func, int _line, const char* _fname,
                    ArgsT&&... _args)
    : m_mode(_mode)
    {
        auto _arg = join_type::join("", _args...);
        switch(m_mode)
        {
            case mode::blank:
            {
                // label and hash
                auto&& _label = _arg;
                auto&& _hash  = add_hash_id(_label);
                m_captured    = captured(result_type{ _label, _hash });
                break;
            }
            case mode::basic:
            {
                compute_data(_func);
                auto&& _label = _join(_arg.c_str());
                auto&& _hash  = add_hash_id(_label);
                m_captured    = captured(result_type{ _label, _hash });
                break;
            }
            case mode::full:
            case mode::complete:
            {
                compute_data(_func, _line, _fname, m_mode == mode::full);
                // label and hash
                auto&& _label = _join(_arg.c_str());
                auto&& _hash  = add_hash_id(_label);
                m_captured    = captured(result_type{ _label, _hash });
                break;
            }
        }
    }

    //----------------------------------------------------------------------------------//
    //
    source_location()                       = delete;
    ~source_location()                      = default;
    source_location(const source_location&) = delete;
    source_location(source_location&&)      = default;
    source_location& operator=(const source_location&) = delete;
    source_location& operator=(source_location&&) = default;

protected:
    //----------------------------------------------------------------------------------//
    //
    void compute_data(const char* _func) { m_prefix = _func; }

    //----------------------------------------------------------------------------------//
    //
    void compute_data(const char* _func, int _line, const char* _fname, bool shorten)
    {
#if defined(TIMEMORY_WINDOWS)
        static const char delim = '\\';
#else
        static const char delim = '/';
#endif
        std::string _filename(_fname);
        if(shorten)
        {
            if(_filename.find(delim) != std::string::npos)
                _filename = _filename.substr(_filename.find_last_of(delim) + 1);
        }

        if(_line < 0)
        {
            if(_filename.length() > 0)
            {
                m_prefix = join_type::join("", _func, '@', _filename);
            }
            else
            {
                m_prefix = _func;
            }
        }
        else
        {
            if(_filename.length() > 0)
            {
                m_prefix = join_type::join("", _func, '@', _filename, ':', _line);
            }
            else
            {
                m_prefix = join_type::join("", _func, ':', _line);
            }
        }
    }

public:
    //----------------------------------------------------------------------------------//
    //
    template <
        typename... ArgsT,
        enable_if_t<(sizeof...(ArgsT) > 0) && !is_constant_char<ArgsT...>(), int> = 0>
    const captured& get_captured(ArgsT&&... _args)
    {
        return m_captured.set(*this, std::forward<ArgsT>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    const captured& get_captured() const { return m_captured; }

    //----------------------------------------------------------------------------------//
    //
    template <
        typename... ArgsT,
        enable_if_t<(sizeof...(ArgsT) > 0) && is_constant_char<ArgsT...>(), int> = 0>
    const captured& get_captured(ArgsT&&...)
    {
        return m_captured;
    }

private:
    mode        m_mode;
    std::string m_prefix = {};
    captured    m_captured;

private:
    std::string _join(const char* _arg)
    {
        return (strcmp(_arg, "") == 0) ? m_prefix
                                       : join_type::join('/', m_prefix.c_str(), _arg);
    }
};
//
namespace internal
{
namespace
{
// anonymous namespace to make sure the static instance is local to each .cpp
// making it static also helps ensure this
template <size_t, size_t, typename... Args>
static inline auto&
get_static_source_location(Args&&... args)
{
    static source_location _instance{ std::forward<Args>(args)... };
    consume_parameters(std::forward<Args>(args)...);
    return _instance;
}
}  // namespace
}  // namespace internal
//
// static function should be local to each .cpp
//
// LineN should be '__LINE__': this ensures that each use in a source file is unique
// CountN should be '__COUNTER__': this ensures that each source file has unique integer
template <size_t LineN, size_t CountN, typename... Args>
static inline auto&
get_static_source_location(Args&&... args)
{
    return internal::get_static_source_location<LineN, CountN>(
        std::forward<Args>(args)...);
    consume_parameters(std::forward<Args>(args)...);
}
//
}  // namespace tim
