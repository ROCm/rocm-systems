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

#include "timemory/environment/macros.hpp"
#include "timemory/utility/types.hpp"

#include <iosfwd>
#include <iostream>
#include <set>
#include <string>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              environment
//
//--------------------------------------------------------------------------------------//
//
class env_settings;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
get_env(const std::string& env_id, Tp _default = Tp{}, bool _store = true);
//
//--------------------------------------------------------------------------------------//
//
template <>
std::string
get_env(const std::string& env_id, std::string _default, bool _store);
//
//--------------------------------------------------------------------------------------//
//
template <>
bool
get_env(const std::string& env_id, bool _default, bool _store);
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
get_env_choice(const std::string& env_id, Tp _default, std::set<Tp> _choices);
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
load_env(const std::string& env_id, Tp _default = Tp{});
//
//--------------------------------------------------------------------------------------//
//
template <>
std::string
load_env(const std::string& env_id, std::string _default);
//
//--------------------------------------------------------------------------------------//
//
template <>
bool
load_env(const std::string& env_id, bool _default);
//
//--------------------------------------------------------------------------------------//
//
void
print_env(std::ostream& os = std::cout);
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
set_env(const std::string& env_var, const Tp& _val, int override = 0);
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

TIMEMORY_SET_CLASS_VERSION(0, tim::env_settings)
