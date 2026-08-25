// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#define _GNU_SOURCE 1

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/cxx/utility.hpp>

#include <dlfcn.h>
#include <iostream>

namespace rocprofiler
{
namespace shared_library
{
// Hidden visibility so references to it always resolve to the local instance. Ensure the symbol
// is not inlined via noinline attribute. When the library is stripped, using dladdr on this symbol
// will typically return non-zero (success) and dli_fname will give the correct .so path but
// dli_sname will often be NULL or name the *nearest* exported symbol and dli_saddr may point to a
// different exported symbol.
ROCPROFILER_NOINLINE void
local_shared_library_resolver(void) ROCPROFILER_HIDDEN_API;

void
local_shared_library_resolver(void)
{
    rocprofiler::registration::init_logging();
}

namespace
{
// used in determining whether this rocprofiler-sdk shared library is the global one
struct library_info
{
    library_info() = default;
    library_info(std::string_view sym_name, void* sym);

    std::string as_string() const;
    friend bool operator==(const library_info& lhs, const library_info& rhs);

    std::string fname  = {};
    void*       handle = nullptr;
};

library_info::library_info(std::string_view sym_name, void* sym)
{
    auto _info = Dl_info{};
    auto _ec   = ::dladdr(sym, &_info);
    if(_ec != 0 && _info.dli_fname != nullptr)
    {
        fname  = _info.dli_fname;
        handle = ::dlopen(_info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    }
    else
    {
        ROCP_WARNING << fmt::format(
            "Failed to resolve rocprofiler-sdk shared library path to symbol '{}' ({}) :: "
            "file_name={}, symbol_name={}, load_address={}, nearest_symbol={}",
            sym_name,
            sdk::utility::as_hex(sym),
            _info.dli_fname,
            _info.dli_sname,
            sdk::utility::as_hex(_info.dli_fbase),
            sdk::utility::as_hex(_info.dli_saddr));
    }
}

std::string
library_info::as_string() const
{
    return fmt::format("fname='{}', handle={}", fname, sdk::utility::as_hex(handle));
}

bool
operator==(const library_info& lhs, const library_info& rhs)
{
    // if both handles are resolved, compare the handles
    if(lhs.handle != nullptr && rhs.handle != nullptr) return (lhs.handle == rhs.handle);

    // if both have names, compare the file names
    if(!lhs.fname.empty() && !rhs.fname.empty()) return (lhs.fname == rhs.fname);

    // comparison not possible, return false
    return false;
}

auto
get_this_library_info()
{
    return library_info{
        "local_shared_library_resolver",
        reinterpret_cast<void*>(&::rocprofiler::shared_library::local_shared_library_resolver)};
}

auto
get_global_library_info()
{
    void* _global_addr = dlsym(RTLD_DEFAULT, "rocprofiler_set_api_table");
    if(!_global_addr)
    {
        ROCP_WARNING << "Failed to resolve global symbol 'rocprofiler_set_api_table'";
        return library_info{};
    }

    return library_info{"rocprofiler_set_api_table", _global_addr};
}

struct lifetime
{
    lifetime();
    ~lifetime();
};

lifetime::lifetime()
{
    registration::init_logging();
    local_shared_library_resolver();  // effectively a no-op but references the symbol to prevent
                                      // unused func warnings, etc.

    if(common::get_env("ROCPROFILER_LIBRARY_CTOR", false))
    {
        auto _this = get_this_library_info();
        auto _glob = get_global_library_info();
        if(_this == _glob)
        {
            ROCP_INFO << "Initializing rocprofiler-sdk library...";
            registration::initialize();
            ROCP_INFO << "rocprofiler-sdk library initialized";
        }
        else
        {
            ROCP_INFO << fmt::format(
                "Skipping rocprofiler-sdk shared library initialization within it is already "
                "initialized in the global context. Local: [{}], Global: [{}]",
                _this.as_string(),
                _glob.as_string());
        }
    }
}

lifetime::~lifetime()
{
    if(common::get_env("ROCPROFILER_LIBRARY_DTOR", false))
    {
        ROCP_INFO << "Finalizing rocprofiler-sdk library...";
        registration::finalize();
        ROCP_INFO << "rocprofiler-sdk library finalized";
    }
}

auto*&
get_lifetime()
{
    static auto* _v = common::static_object<lifetime>::construct();
    return _v;
}
}  // namespace
}  // namespace shared_library

auto rocprofiler_sdk_shlib_lifetime = shared_library::get_lifetime();

void
rocprofiler_sdk_shlib_ctor() ROCPROFILER_ATTRIBUTE(constructor(101));

void
rocprofiler_sdk_shlib_ctor()
{
    (void) shared_library::get_lifetime();
}
}  // namespace rocprofiler
