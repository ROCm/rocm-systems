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

/**
 * \file timemory/environment/definition.hpp
 * \brief The definitions for the types in environment
 */

#pragma once

#include "timemory/environment/declaration.hpp"
#include "timemory/environment/macros.hpp"
#include "timemory/environment/types.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/utility/utility.hpp"

#include <atomic>
#include <iosfwd>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace regex_const = std::regex_constants;
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_ENVIRONMENT_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
//                              environment
//
//--------------------------------------------------------------------------------------//
//
// if passed a global, insert m_env into it's array
//
TIMEMORY_ENVIRONMENT_LINKAGE(env_settings::env_settings)
(env_settings* _global, int _id)
: m_id(_id)
, m_env(new env_uomap_t)
{
    if(_global && _id != 0)
    {
        auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
        lock_flag().store(true);
        if(!lk.owns_lock())
            lk.lock();
        _global->m_env_other.emplace_back(m_env);
        lock_flag().store(false);
    }
}
//
//--------------------------------------------------------------------------------------//
//
// only the zero id deletes
//
TIMEMORY_ENVIRONMENT_LINKAGE(env_settings::~env_settings())
{
    if(m_id == 0)
    {
        delete m_env;
        for(auto& itr : m_env_other)
            delete itr;
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ENVIRONMENT_LINKAGE(env_settings::env_map_t)
env_settings::get() const
{
    auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
    lock_flag().store(true);
    if(!lk.owns_lock())
        lk.lock();

    auto      _tmp = *m_env;
    env_map_t _ret;
    for(const auto& itr : _tmp)
        _ret[itr.first] = itr.second;

    lock_flag().store(false);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ENVIRONMENT_LINKAGE(void)
env_settings::print(std::ostream& os) const
{
    if(m_id == 0)
        const_cast<env_settings&>(*this).collapse();

    auto _data = get();

    size_t _w = 35;
    for(const auto& itr : _data)
        _w = std::max<size_t>(itr.first.length(), _w);

    std::stringstream filler;
    filler.fill('-');
    filler << '#';
    filler << std::setw(88) << "";
    filler << '#';

    std::stringstream ss;
    ss << filler.str() << '\n';
    ss << "# Environment settings:\n";
    for(const auto& itr : _data)
    {
        ss << "# " << std::setw(_w) << std::right << itr.first << "\t = \t" << std::left
           << itr.second << '\n';
    }
    ss << filler.str();
    os << ss.str() << '\n';
}
//
//--------------------------------------------------------------------------------------//
//
// NOLINTNEXTLINE
TIMEMORY_ENVIRONMENT_LINKAGE(void)
env_settings::collapse()
{
    if(m_id != 0)
        return;

    auto_lock_t lk(type_mutex<env_settings>(), std::defer_lock);
    if(!lk.owns_lock())
        lk.lock();

    lock_flag().store(true);

    // insert all the other maps into this map
    for(size_t i = 0; i < m_env_other.size(); ++i)
    {
        // get the map instance
        auto* itr = m_env_other[i];
        if(itr)
        {
            // loop over entries
            for(const auto& mitr : *m_env_other[i])
            {
                auto key = mitr.first;
                if(m_env->find(key) != m_env->end())
                {
                    // if the key already exists and if the entries differ,
                    // create a new unique entry for the index
                    if(m_env->find(key)->second != mitr.second)
                    {
                        key += string_t("[@") + std::to_string(i + 1) + string_t("]");
                        (*m_env)[key] = mitr.second;
                    }
                }
                else
                {
                    // if a new entry, insert without unique tag to avoid duplication
                    // when thread count is very high
                    (*m_env)[key] = mitr.second;
                }
            }
        }
    }
    lock_flag().store(false);
}
//
//--------------------------------------------------------------------------------------//
//
// specialization for string since the above will have issues if string includes spaces
//
template <>
TIMEMORY_ENVIRONMENT_LINKAGE(std::string)
get_env(const std::string& env_id, std::string _default, bool _store)
{
    if(env_id.empty())
        return _default;

    auto* _env_settings = env_settings::instance();
    char* env_var       = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::stringstream ss;
        ss << env_var;
        if(_env_settings && _store)
            _env_settings->insert(env_id, ss.str());
        return ss.str();
    }
    // record default value
    if(_env_settings && _store)
        _env_settings->insert(env_id, _default);

    // return default if not specified in environment
    return _default;
}
//
//--------------------------------------------------------------------------------------//
//
//  overload for boolean
//
template <>
TIMEMORY_ENVIRONMENT_LINKAGE(bool)
get_env(const std::string& env_id, bool _default, bool _store)
{
    if(env_id.empty())
        return _default;

    auto* _env_settings = env_settings::instance();
    char* env_var       = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string var = std::string(env_var);
        bool        val = true;
        if(var.find_first_not_of("0123456789") == std::string::npos)
        {
            val = (bool) atoi(var.c_str());
        }
        else
        {
            for(auto& itr : var)
                itr = tolower(itr);
            for(const auto& itr : { "off", "false", "no", "n", "f", "0" })
            {
                if(var == itr)
                {
                    if(_env_settings && _store)
                        _env_settings->insert<bool>(env_id, false);
                    return false;
                }
            }
        }
        if(_env_settings && _store)
            _env_settings->insert<bool>(env_id, val);
        return val;
    }
    // record default value
    if(_env_settings && _store)
        _env_settings->insert<bool>(env_id, _default);

    // return default if not specified in environment
    return _default;
}
//
//--------------------------------------------------------------------------------------//
//
// specialization for string since the above will have issues if string includes spaces
//
template <>
TIMEMORY_ENVIRONMENT_LINKAGE(std::string)
load_env(const std::string& env_id, std::string _default)
{
    if(env_id.empty())
        return _default;

    auto* _env_settings = env_settings::instance();
    auto  itr           = _env_settings->get(env_id);
    if(itr != _env_settings->end())
        return itr->second;

    // return default if not specified in environment
    return _default;
}
//
//--------------------------------------------------------------------------------------//
//
//  overload for boolean
//
template <>
TIMEMORY_ENVIRONMENT_LINKAGE(bool)
load_env(const std::string& env_id, bool _default)
{
    if(env_id.empty())
        return _default;

    auto* _env_settings = env_settings::instance();
    auto  itr           = _env_settings->get(env_id);
    if(itr != _env_settings->end())
    {
        auto              val             = itr->second;
        const auto        regex_constants = regex_const::egrep | regex_const::icase;
        const std::string pattern         = "^(off|false|no|n|f|0)$";
        bool              _match          = false;
        try
        {
            _match = std::regex_match(val, std::regex(pattern, regex_constants));
        } catch(std::bad_cast&)
        {
            for(auto& vitr : val)
                vitr = tolower(vitr);
            for(const auto& vitr : { "off", "false", "no", "n", "f", "0" })
            {
                if(val == vitr)
                {
                    _match = true;
                    break;
                }
            }
        }
        return (_match) ? false : true;
    }

    // return default if not specified in environment
    return _default;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ENVIRONMENT_LINKAGE(void)
print_env(std::ostream& os) { os << (*env_settings::instance()); }
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_ENVIRONMENT_LINKAGE(tim::env_settings*)
env_settings::instance()
{
    static std::atomic<int>           _count{ 0 };
    static env_settings               _instance{};
    static thread_local int           _id = _count++;
    static thread_local env_settings* _local =
        (_id == 0) ? (&_instance) : new env_settings{ &_instance, _id };
    static thread_local auto _ldtor = scope::destructor{ []() {
        if(_local == &_instance || _id == 0)
            return;
        delete _local;
        _local                      = nullptr;
    } };
    return _local;
    (void) _ldtor;
}
//
//--------------------------------------------------------------------------------------//
//
#endif  // !defined(TIMEMORY_USE_ENVIRONMENT_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
