// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/string_entry.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"

#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace common
{
namespace
{
using name_array_t = std::unordered_map<size_t, std::unique_ptr<std::string>>;

auto&
get_sync()
{
    static auto*& _v = static_object<std::shared_mutex>::construct();
    return *CHECK_NOTNULL(_v);
}

name_array_t*
get_string_array()
{
    static auto*& _v = static_object<name_array_t>::construct();
    return _v;
}
}  // namespace

const std::string*
get_string_entry(std::string_view name)
{
    if(!get_string_array()) return nullptr;

    auto _hash_v = std::hash<std::string_view>{}(name);
    {
        auto _lk = std::shared_lock<std::shared_mutex>{get_sync()};
        if(get_string_array()->count(_hash_v) > 0) return get_string_array()->at(_hash_v).get();
    }

    auto _lk = std::unique_lock<std::shared_mutex>{get_sync()};
    return get_string_array()
        ->emplace(_hash_v, std::make_unique<std::string>(name))
        .first->second.get();
}

const std::string*
get_string_entry(size_t _hash_v)
{
    if(!get_string_array()) return nullptr;

    auto _lk = std::shared_lock<std::shared_mutex>{get_sync()};
    if(get_string_array()->count(_hash_v) > 0) return get_string_array()->at(_hash_v).get();

    return nullptr;
}

size_t
add_string_entry(std::string_view name)
{
    if(!get_string_array()) return 0;

    auto _hash_v = std::hash<std::string_view>{}(name);
    {
        auto _lk = std::shared_lock<std::shared_mutex>{get_sync()};
        if(get_string_array()->count(_hash_v) > 0) return _hash_v;
    }

    auto _lk = std::unique_lock<std::shared_mutex>{get_sync()};
    get_string_array()->emplace(_hash_v, std::make_unique<std::string>(name));

    return _hash_v;
}
}  // namespace common
}  // namespace rocprofiler
