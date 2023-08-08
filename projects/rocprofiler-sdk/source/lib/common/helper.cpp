/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "lib/common/helper.hpp"

#include <amd_comgr/amd_comgr.h>

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cxxabi.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <set>

#define ENABLE_BACKTRACE
#if defined(ENABLE_BACKTRACE)
#    include <backtrace.h>
#endif

#define amd_comgr_(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        if(amd_comgr_status_t status = amd_comgr_##call; status != AMD_COMGR_STATUS_SUCCESS)       \
        {                                                                                          \
            const char* reason = "";                                                               \
            amd_comgr_status_string(status, &reason);                                              \
            fprintf(stderr, #call " failed: %s\n", reason);                                        \
            abort();                                                                               \
        }                                                                                          \
    } while(false)

namespace rocprofiler
{
namespace common
{
std::string
cxa_demangle(std::string_view _mangled_name, int* _status)
{
    constexpr size_t buffer_len = 4096;
    // return the mangled since there is no buffer
    if(_mangled_name.empty())
    {
        *_status = -2;
        return std::string{};
    }

    auto _demangled_name = std::string{_mangled_name};

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into which the
    //      demangled name is stored. If output_buffer is not long enough, it is expanded
    //      using realloc. output_buffer may instead be NULL; in that case, the demangled
    //      name is placed in a region of memory allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the demangled name
    //      is placed in *length.
    //  status:
    //      *status is set to one of the following values
    size_t _demang_len = 0;
    char*  _demang = abi::__cxa_demangle(_demangled_name.c_str(), nullptr, &_demang_len, _status);
    switch(*_status)
    {
        //  0 : The demangling operation succeeded.
        // -1 : A memory allocation failure occurred.
        // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
        // -3 : One of the arguments is invalid.
        case 0:
        {
            if(_demang) _demangled_name = std::string{_demang};
            break;
        }
        case -1:
        {
            char _msg[buffer_len];
            ::memset(_msg, '\0', buffer_len * sizeof(char));
            ::snprintf(_msg,
                       buffer_len,
                       "memory allocation failure occurred demangling %s",
                       _demangled_name.c_str());
            ::perror(_msg);
            break;
        }
        case -2: break;
        case -3:
        {
            char _msg[buffer_len];
            ::memset(_msg, '\0', buffer_len * sizeof(char));
            ::snprintf(_msg,
                       buffer_len,
                       "Invalid argument in: (\"%s\", nullptr, nullptr, %p)",
                       _demangled_name.c_str(),
                       (void*) _status);
            ::perror(_msg);
            break;
        }
        default: break;
    };

    // if it "demangled" but the length is zero, set the status to -2
    if(_demang_len == 0 && *_status == 0) *_status = -2;

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
}

namespace
{
#if defined(ENABLE_BACKTRACE)

// struct BackTraceInfo
// {
//     struct ::backtrace_state* state = nullptr;
//     std::stringstream         sstream{};
//     int                       depth = 0;
//     int                       error = 0;
// };

// void
// errorCallback(void* data, const char* message, int errnum)
// {
//     BackTraceInfo* info = static_cast<BackTraceInfo*>(data);
//     info->sstream << "ROCProfiler: error: " << message << '(' << errnum << ')';
//     info->error = 1;
// }

// void
// syminfoCallback(void* data,
//                 uintptr_t /* pc  */,
//                 const char* symname,
//                 uintptr_t /* symval  */,
//                 uintptr_t /* symsize  */)
// {
//     BackTraceInfo* info = static_cast<BackTraceInfo*>(data);

//     if(symname == nullptr) return;

//     int    status     = 0;
//     auto&& _demangled = cxa_demangle(symname, &status);
//     info->sstream << ' '
//                   << (status == 0 ? std::string_view{_demangled} : std::string_view{symname});
// }

// int
// fullCallback(void* data, uintptr_t pc, const char* filename, int lineno, const char* function)
// {
//     BackTraceInfo* info = static_cast<BackTraceInfo*>(data);

//     info->sstream << std::endl
//                   << "    #" << std::dec << info->depth++ << ' ' << "0x" << std::noshowbase
//                   << std::hex << std::setfill('0') << std::setw(sizeof(pc) * 2) << pc;
//     if(function == nullptr)
//     {
//         backtrace_syminfo(info->state, pc, syminfoCallback, errorCallback, data);
//     }
//     else
//     {
//         int    status     = 0;
//         auto&& _demangled = cxa_demangle(function, &status);
//         info->sstream << ' '
//                       << (status == 0 ? std::string_view{_demangled} :
//                       std::string_view{function});

//         if(filename != nullptr)
//         {
//             info->sstream << " in " << filename;
//             if(lineno != 0) info->sstream << ':' << std::dec << lineno;
//         }
//     }

//     return info->error;
// }
#endif  // defined (ENABLE_BACKTRACE)
}  // namespace

/* The function extracts the kernel name from
input string. By using the iterators it finds the
window in the string which contains only the kernel name.
For example 'Foo<int, float>::foo(a[], int (int))' -> 'foo'*/
std::string
truncate_name(std::string_view name)
{
    auto     rit         = name.rbegin();
    auto     rend        = name.rend();
    uint32_t counter     = 0;
    char     open_token  = 0;
    char     close_token = 0;
    while(rit != rend)
    {
        if(counter == 0)
        {
            switch(*rit)
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
                case ' ': ++rit; continue;
            }
            if(counter == 0) break;
        }
        else
        {
            if(*rit == open_token) counter++;
            if(*rit == close_token) counter--;
        }
        ++rit;
    }
    auto rbeg = rit;
    while((rit != rend) && (*rit != ' ') && (*rit != ':'))
        rit++;
    return std::string{name.substr(rend - rit, rit - rbeg)};
}

// C++ symbol demangle
std::string
cxx_demangle(std::string_view symbol)
{
    int  _status       = 0;
    auto demangled_str = cxa_demangle(symbol, &_status);
    if(_status == 0)
    {
        return demangled_str;
    }

    amd_comgr_data_t mangled_data;
    amd_comgr_(create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled_data));
    amd_comgr_(set_data(mangled_data, symbol.size(), symbol.data()));

    amd_comgr_data_t demangled_data;
    amd_comgr_(demangle_symbol_name(mangled_data, &demangled_data));

    size_t demangled_size = 0;
    amd_comgr_(get_data(demangled_data, &demangled_size, nullptr));

    demangled_str.resize(demangled_size);
    amd_comgr_(get_data(demangled_data, &demangled_size, demangled_str.data()));

    amd_comgr_(release_data(mangled_data));
    amd_comgr_(release_data(demangled_data));
    return demangled_str;
}

// check if string has special char
bool
has_special_char(std::string_view str)
{
    return std::find_if(str.begin(), str.end(), [](unsigned char ch) {
               return !((isalnum(ch) != 0) || ch == '_' || ch == ':' || ch == ' ');
           }) != str.end();
}

// check if string has correct counter format
bool
has_counter_format(std::string_view str)
{
    return std::find_if(str.begin(), str.end(), [](unsigned char ch) {
               return ((isalnum(ch) != 0) || ch == '_');
           }) != str.end();
}

// trims the begining of the line for spaces
std::string
left_trim(std::string_view s)
{
    constexpr std::string_view WHITESPACE = " \n\r\t\f\v";
    size_t                     start      = s.find_first_not_of(WHITESPACE);
    if(start == std::string_view::npos) return std::string{};
    return std::string{s.substr(start)};
}

// trims begining and end of input line in place
void
trim(std::string& str)
{
    // Remove leading spaces.
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {
                  return std::isspace(ch) == 0;
              }));
    // Remove trailing spaces.
    str.erase(std::find_if(
                  str.rbegin(), str.rend(), [](unsigned char ch) { return std::isspace(ch) == 0; })
                  .base(),
              str.end());
}

// replace unsuported specail chars with space
static void
handle_special_chars(std::string& str)
{
    std::set<char> specialChars = {'!', '@', '#', '$', '%', '&', '(', ')', ',',
                                   '*', '+', '-', '.', '/', ';', '<', '=', '>',
                                   '?', '@', '{', '}', '^', '`', '~', '|', ':'};

    // Iterate over the string and replace any special characters with a space.
    for(char& i : str)
    {
        if(specialChars.find(i) != specialChars.end())
        {
            i = ' ';
        }
    }
}

// validate input coutners and correct format if needed
void
validate_counters_format(std::vector<std::string>& counters, std::string line)
{
    // trim line for any white spaces
    trim(line);

    if(!(line[0] == '#' || line.find("pmc") == std::string::npos))
    {
        handle_special_chars(line);

        std::stringstream input_line(line);
        std::string       counter;
        while(getline(input_line, counter, ' '))
        {
            if(counter.substr(0, 3) != "pmc" && has_counter_format(counter))
            {
                counters.push_back(counter);
            }
        }
    }

    // raise exception with correct usage if user still managed to corrupt input
    for(const auto& itr : counters)
    {
        if(!has_counter_format(itr))
        {
            fprintf(stderr,
                    "[rocprofiler] Bad input metric. usage --> pmc: <counter1> <counter2>\n");
        }
    }
}

}  // namespace common
}  // namespace rocprofiler
