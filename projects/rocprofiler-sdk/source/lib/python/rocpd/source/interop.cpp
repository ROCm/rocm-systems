// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "interop.hpp"
#include "pysqlite_Connection.h"

#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/output/timestamps.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/utility.hpp>

#include <fmt/format.h>
#include <gotcha/gotcha.h>
#include <gotcha/gotcha_types.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <sqlite3.h>
#include <algorithm>

namespace rocpd
{
namespace interop
{
namespace
{
namespace sdk    = ::rocprofiler::sdk;
namespace common = ::rocprofiler::common;
// namespace tool = ::rocprofiler::tool;
// namespace sql  = ::rocprofiler::tool::sql;

using sqlite3_open_func_t    = int (*)(const char*, sqlite3**);
using sqlite3_open_v2_func_t = int (*)(const char*, sqlite3**, int, const char*);
using sqlite3_close_func_t   = int (*)(sqlite3*);

gotcha_wrappee_handle_t orig_sqlite3_open     = {};
gotcha_wrappee_handle_t orig_sqlite3_open_v2  = {};
gotcha_wrappee_handle_t orig_sqlite3_close    = {};
gotcha_wrappee_handle_t orig_sqlite3_close_v2 = {};

using sqlite_object_map_t = std::unordered_map<PyObject*, sqlite3*>;

thread_local sqlite3* last_sqlite3       = nullptr;
auto                  sqlite_obj_mapping = sqlite_object_map_t{};

auto*
get_mapping()
{
    static auto*& _v = common::static_object<sqlite_object_map_t>::construct();
    return _v;
}

void
erase_connection(sqlite3* pDb)
{
    if(!get_mapping()) return;

    if(last_sqlite3 && last_sqlite3 == pDb) last_sqlite3 = nullptr;

    auto itr = std::find_if(get_mapping()->begin(), get_mapping()->end(), [pDb](const auto& val) {
        return (val.second == pDb);
    });
    if(itr != get_mapping()->end()) get_mapping()->erase(itr);
}

namespace impl
{
int
sqlite3_open(const char* filename, /* Database filename (UTF-8) */
             sqlite3**   ppDb      /* OUT: SQLite db handle */
)
{
    ROCP_TRACE << fmt::format("invoking {}... {}", __FUNCTION__, filename);

    auto func = reinterpret_cast<sqlite3_open_func_t>(gotcha_get_wrappee(orig_sqlite3_open));
    auto ret  = func(filename, ppDb);

    if(ppDb) last_sqlite3 = *ppDb;

    return ret;
}

int
sqlite3_open_v2(const char* filename, sqlite3** ppDb, int flags, const char* zVfs)
{
    ROCP_TRACE << fmt::format("invoking {}... {}", __FUNCTION__, filename);

    auto func = reinterpret_cast<sqlite3_open_v2_func_t>(gotcha_get_wrappee(orig_sqlite3_open_v2));
    auto ret  = func(filename, ppDb, flags, zVfs);

    if(ppDb) last_sqlite3 = *ppDb;

    return ret;
}

int
sqlite3_close(sqlite3* pDb)
{
    ROCP_TRACE << fmt::format("invoking {}... ", __FUNCTION__);

    auto func = reinterpret_cast<sqlite3_close_func_t>(gotcha_get_wrappee(orig_sqlite3_close));
    auto ret  = func(pDb);

    erase_connection(pDb);

    return ret;
}

int
sqlite3_close_v2(sqlite3* pDb)
{
    ROCP_TRACE << fmt::format("invoking {}... ", __FUNCTION__);

    auto func = reinterpret_cast<sqlite3_close_func_t>(gotcha_get_wrappee(orig_sqlite3_close_v2));
    auto ret  = func(pDb);

    erase_connection(pDb);

    return ret;
}
}  // namespace impl

auto bindings = std::array<gotcha_binding_t, 4>{
    gotcha_binding_t{"sqlite3_open",
                     reinterpret_cast<void*>(impl::sqlite3_open),
                     &orig_sqlite3_open},
    gotcha_binding_t{"sqlite3_close",
                     reinterpret_cast<void*>(impl::sqlite3_close),
                     &orig_sqlite3_close},
    gotcha_binding_t{"sqlite3_open_v2",
                     reinterpret_cast<void*>(impl::sqlite3_open_v2),
                     &orig_sqlite3_open_v2},
    gotcha_binding_t{"sqlite3_close_v2",
                     reinterpret_cast<void*>(impl::sqlite3_close_v2),
                     &orig_sqlite3_close_v2},
};
}  // namespace

void
activate_gotcha_bindings()
{
    // activate the gotcha wrappers
    auto _err = gotcha_wrap(bindings.data(), bindings.size(), "rocpd.sqlite3");
    ROCP_WARNING_IF(_err != GOTCHA_SUCCESS) << "gotcha error for rocpd.sqlite3";
}

sqlite3*
map_connection(py::object obj)
{
    if(!get_mapping()) return nullptr;

    get_mapping()->emplace(obj.ptr(), last_sqlite3);

    auto* _ret = get_mapping()->at(obj.ptr());
    ROCP_INFO << "[pyrocpd][mapping] " << obj.ptr() << " <-> " << _ret;

    return get_mapping()->at(obj.ptr());
}

sqlite3*
get_connection(py::object&& obj)
{
    if(!obj) return nullptr;

    if(get_mapping())
    {
        if(auto itr = get_mapping()->find(obj.ptr());
           itr != get_mapping()->end() && itr->second != nullptr)
        {
            ROCP_INFO << fmt::format(
                "sqlite3 python connection ({}) mapped to sqlite3* ({}) safely "
                "via gotcha capture of sqlite3_open",
                sdk::utility::as_hex(obj.ptr(), 16),
                sdk::utility::as_hex(itr->second, 16));
            return itr->second;
        }
    }

    ROCP_CI_LOG(WARNING) << fmt::format(
        "sqlite3 python connection ({}) not captured by gotcha... accessing sqlite3* via "
        "reinterpret_cast<pysqlite_Connection>(PyObject*&)->db :: [unsafe]",
        sdk::utility::as_hex(obj.ptr(), 16));

    return reinterpret_cast<pysqlite_Connection*&>(obj.ptr())->db;
}
}  // namespace interop
}  // namespace rocpd
