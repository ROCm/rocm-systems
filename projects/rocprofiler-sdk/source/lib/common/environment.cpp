// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "lib/common/environment.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace common
{
namespace impl
{
std::string
get_env(std::string_view env_id, std::string_view _default)
{
    if(env_id.empty()) return std::string{_default};
    char* env_var = ::std::getenv(env_id.data());
    if(env_var) return std::string{env_var};
    return std::string{_default};
}

std::string
get_env(std::string_view env_id, const char* _default)
{
    return get_env(env_id, std::string_view{_default});
}

int
get_env(std::string_view env_id, int _default)
{
    if(env_id.empty()) return _default;
    char* env_var = ::std::getenv(env_id.data());
    if(env_var)
    {
        try
        {
            return std::stoi(env_var);
        } catch(std::exception& _e)
        {
            LOG(WARNING) << "[rocprofiler][get_env] Exception thrown converting getenv(\"" << env_id
                         << "\") = " << env_var << " to integer :: " << _e.what()
                         << ". Using default value of " << _default << "\n";
        }
        return _default;
    }
    return _default;
}

bool
get_env(std::string_view env_id, bool _default)
{
    if(env_id.empty()) return _default;
    char* env_var = ::std::getenv(env_id.data());
    if(env_var)
    {
        if(std::string_view{env_var}.empty())
        {
            throw std::runtime_error(std::string{"No boolean value provided for "} +
                                     std::string{env_id});
        }

        if(std::string_view{env_var}.find_first_not_of("0123456789") == std::string_view::npos)
        {
            return static_cast<bool>(std::stoi(env_var));
        }

        for(size_t i = 0; i < std::string_view{env_var}.length(); ++i)
            env_var[i] = tolower(env_var[i]);

        for(const auto& itr : {"off", "false", "no", "n", "f", "0"})
            if(std::string_view{env_var} == itr) return false;

        return true;
    }
    return _default;
}
}  // namespace impl
}  // namespace common
}  // namespace rocprofiler
