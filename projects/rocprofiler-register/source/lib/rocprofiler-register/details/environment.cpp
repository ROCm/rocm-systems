// Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "environment.hpp"

#include <cstdlib>
#include <string>

#if defined(_WIN32)
// WINDOWS-DIVERGENCE: Win32 process-environment access. CRT getenv() is buffered
// against the C runtime's snapshot of the environment block; if anything in the
// process called SetEnvironmentVariableW, getenv() can return stale data. Use
// GetEnvironmentVariableW so we always see the current Win32 environment.
#    include "details/platform/windows/encoding.hpp"  // also pulls in rocprofiler_register_windows.h
#endif

namespace rocprofiler_register
{
namespace common
{
#if defined(_WIN32)
namespace
{
using platform::encoding::utf8_to_wide;
using platform::encoding::wide_to_utf8;
}  // namespace

std::optional<std::string>
read_env_string(std::string_view env_id)
{
    if(env_id.empty()) return std::nullopt;

    auto wide_name = utf8_to_wide(env_id);
    // Probe for required buffer size (including NUL). Returns 0 only on error;
    // an empty value returns 1 (just the NUL terminator).
    ::SetLastError(ERROR_SUCCESS);
    auto required = ::GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
    if(required == 0) return std::nullopt;  // ERROR_ENVVAR_NOT_FOUND or other error

    auto buffer = std::wstring(static_cast<size_t>(required), L'\0');
    ::SetLastError(ERROR_SUCCESS);
    auto written = ::GetEnvironmentVariableW(wide_name.c_str(), buffer.data(), required);
    // written == 0 can mean empty value (success) or failure; use GetLastError to tell
    // apart.
    if(written == 0 && ::GetLastError() != ERROR_SUCCESS) return std::nullopt;
    // 'written' is character count excluding NUL.
    return wide_to_utf8(buffer.data(), static_cast<int>(written));
}

int
write_env_string(std::string_view env_id, std::string_view value, bool overwrite)
{
    if(env_id.empty()) return -1;

    if(!overwrite)
    {
        // POSIX setenv() with overwrite=0 leaves an existing value untouched
        // and reports success.
        auto existing = read_env_string(env_id);
        if(existing) return 0;
    }

    // _putenv_s mirrors POSIX setenv(name, value, /*overwrite=*/1) semantics:
    // it always writes (or removes when value is empty). It also pushes the
    // value through to the Win32 environment block.
    auto name_str  = std::string{ env_id };
    auto value_str = std::string{ value };
    return ::_putenv_s(name_str.c_str(), value_str.c_str());
}
#else   // !defined(_WIN32)
std::optional<std::string>
read_env_string(std::string_view env_id)
{
    if(env_id.empty()) return std::nullopt;
    auto  name_str = std::string{ env_id };
    auto* env_var  = std::getenv(name_str.c_str());
    if(env_var == nullptr) return std::nullopt;
    return std::string{ env_var };
}

int
write_env_string(std::string_view env_id, std::string_view value, bool overwrite)
{
    if(env_id.empty()) return -1;
    auto name_str  = std::string{ env_id };
    auto value_str = std::string{ value };
    return ::setenv(name_str.c_str(), value_str.c_str(), overwrite ? 1 : 0);
}
#endif  // defined(_WIN32)
}  // namespace common
}  // namespace rocprofiler_register
