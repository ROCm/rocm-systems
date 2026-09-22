// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "kernel_name.h"

#include <cxxabi.h>

#include <cstdlib>
#include <iostream>

namespace rocprofiler_compute_tool
{
namespace
{
std::string cxa_demangle(const std::string& mangled_name, int* status)
{
    // return the mangled since there is no buffer
    if (mangled_name.empty())
    {
        *status = -2;
        return std::string{};
    }

    auto _demangled_name = std::string{mangled_name};

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be
    //      demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into
    //      which the demangled name is stored. If output_buffer is not long
    //      enough, it is expanded using realloc. output_buffer may instead be
    //      NULL; in that case, the demangled name is placed in a region of memory
    //      allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the
    //      demangled name is placed in *length.
    //  status:
    //      *status is set to one of the following values
    size_t _demang_len = 0;
    char*  _demang = abi::__cxa_demangle(_demangled_name.c_str(), nullptr, &_demang_len, status);
    switch (*status)
    {
    //  0 : The demangling operation succeeded.
    // -1 : A memory allocation failure occurred.
    // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
    // -3 : One of the arguments is invalid.
    case 0:
    {
        if (_demang)
            _demangled_name = std::string{_demang};
        break;
    }
    case -1:
    {
        std::clog << "[rocprofiler-compute] memory allocation failure occurred "
                     "demangling "
                  << _demangled_name << std::endl;
        break;
    }
    case -2:
    {
        break;
    }
    case -3:
    {
        std::clog << "[rocprofiler-compute] Invalid argument in: (\"" << _demangled_name
                  << "\", nullptr, nullptr, " << static_cast<void*>(status) << ")" << std::endl;
        break;
    }
    default:
        break;
    };

    // if it "demangled" but the length is zero, set the status to -2
    if (_demang_len == 0 && *status == 0)
        *status = -2;

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
}
}  // namespace

std::string format_kernel_name(const char* mangled_name)
{
    if (mangled_name == nullptr)
        return std::string{};

    auto name = std::string{mangled_name};
    // Kernel descriptor symbols carry a ".kd" suffix that does not demangle.
    constexpr std::string_view kKernelDescriptorSuffix = ".kd";
    if (name.size() > kKernelDescriptorSuffix.size() &&
        name.compare(name.size() - kKernelDescriptorSuffix.size(),
                     kKernelDescriptorSuffix.size(),
                     kKernelDescriptorSuffix) == 0)
    {
        name.erase(name.size() - kKernelDescriptorSuffix.size());
    }

    int demangle_status = 0;
    return cxa_demangle(name, &demangle_status);
}

std::string truncate_name(std::string_view name)
{
    // The function extracts the kernel name from
    // input string. By using the iterators it finds the
    // window in the string which contains only the kernel name.
    // For example 'Foo<int, float>::foo(a[], int (int))' -> 'foo'
    auto     rit         = name.rbegin();
    auto     rend        = name.rend();
    uint32_t counter     = 0;
    char     open_token  = 0;
    char     close_token = 0;
    while (rit != rend)
    {
        if (counter == 0)
        {
            switch (*rit)
            {
            case ')':
                counter     = 1;
                open_token  = ')';
                close_token = '(';
                break;
            case '>':
                counter     = 1;
                open_token  = '>';
                close_token = '<';
                break;
            case ']':
                counter     = 1;
                open_token  = ']';
                close_token = '[';
                break;
            case ' ':
                ++rit;
                continue;
            }
            if (counter == 0)
                break;
        }
        else
        {
            if (*rit == open_token)
                counter++;
            if (*rit == close_token)
                counter--;
        }
        ++rit;
    }
    auto rbeg = rit;
    while ((rit != rend) && (*rit != ' ') && (*rit != ':'))
        rit++;
    return std::string{name.substr(rend - rit, rit - rbeg)};
}

}  // namespace rocprofiler_compute_tool
