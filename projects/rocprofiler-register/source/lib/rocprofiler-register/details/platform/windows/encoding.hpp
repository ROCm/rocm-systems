// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

// WINDOWS-DIVERGENCE: shared UTF-8 <-> UTF-16 conversion helpers used by the
// Win32 platform-abstraction sources (windows/loader.cpp, windows/pe_parser.cpp,
// details/environment.cpp). Header-only because the helpers are short, used in
// only a handful of translation units, and depend solely on <windows.h>.

#if !defined(_WIN32)
#    error "details/platform/windows/encoding.hpp is Windows-only"
#endif

#include "details/platform/windows/rocprofiler_register_windows.h"

#include <string>
#include <string_view>

namespace rocprofiler_register
{
namespace platform
{
namespace encoding
{
inline std::wstring
utf8_to_wide(std::string_view utf8)
{
    if(utf8.empty()) return std::wstring{};
    auto required = ::MultiByteToWideChar(
        CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    if(required <= 0) return std::wstring{};
    auto result = std::wstring(static_cast<size_t>(required), L'\0');
    ::MultiByteToWideChar(
        CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), result.data(), required);
    return result;
}

inline std::string
wide_to_utf8(const wchar_t* wide, int wide_len)
{
    if(wide == nullptr || wide_len <= 0) return std::string{};
    auto required =
        ::WideCharToMultiByte(CP_UTF8, 0, wide, wide_len, nullptr, 0, nullptr, nullptr);
    if(required <= 0) return std::string{};
    auto result = std::string(static_cast<size_t>(required), '\0');
    ::WideCharToMultiByte(
        CP_UTF8, 0, wide, wide_len, result.data(), required, nullptr, nullptr);
    return result;
}

inline std::string
wide_to_utf8(const std::wstring& wide)
{
    return wide_to_utf8(wide.data(), static_cast<int>(wide.size()));
}
}  // namespace encoding
}  // namespace platform
}  // namespace rocprofiler_register
