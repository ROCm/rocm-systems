// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "symbol_lookup.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"

#include <dlfcn.h>
#include <link.h>

#include <optional>
#include <unordered_map>
#include <vector>

namespace fs = rocprofiler::common::filesystem;

namespace rocprofiler
{
namespace rocattach
{
namespace
{
constexpr char ROCATTACH_LIBRARY_NAME[]                       = "librocprofiler-sdk-rocattach.so.1";
std::unordered_map<std::string, void*> m_target_library_addrs = {};
std::unordered_map<std::string, void*> m_target_symbol_addrs  = {};

using open_modes_vec_t = std::vector<int>;

std::optional<std::string>
get_linked_path(std::string_view _name, open_modes_vec_t&& _open_modes)
{
    const open_modes_vec_t default_link_open_modes = {(RTLD_LAZY | RTLD_NOLOAD)};
    if(_name.empty()) return fs::current_path().string();

    if(_open_modes.empty()) _open_modes = default_link_open_modes;

    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name.data(), _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        if(_link_map != nullptr && !std::string_view{_link_map->l_name}.empty())
        {
            return fs::absolute(fs::path{_link_map->l_name}).string();
        }
        if(_noload == false) dlclose(_handle);
    }

    return std::nullopt;
}

auto
get_this_library_path()
{
    auto _this_lib_path = get_linked_path(ROCATTACH_LIBRARY_NAME, {RTLD_NOLOAD | RTLD_LAZY});
    LOG_IF(FATAL, !_this_lib_path) << "[rocprofiler-sdk-rocattach] " << ROCATTACH_LIBRARY_NAME
                                   << " could not locate itself in the list of loaded libraries";
    return fs::path{*_this_lib_path}.parent_path().string();
}

void*
get_library_handle(std::string_view _lib_name)
{
    void* _lib_handle = nullptr;

    if(_lib_name.empty()) return nullptr;

    auto _lib_path       = fs::path{_lib_name};
    auto _lib_path_fname = _lib_path.filename();
    auto _lib_path_abs =
        (_lib_path.is_absolute()) ? _lib_path : (fs::path{get_this_library_path()} / _lib_path);

    // check to see if the rocprofiler library is already loaded
    _lib_handle = dlopen(_lib_path.c_str(), RTLD_NOLOAD | RTLD_LAZY);

    if(_lib_handle)
    {
        LOG(INFO) << "[rocprofiler-sdk-rocattach] loaded " << _lib_name << " library at "
                  << _lib_path.string() << " (handle=" << _lib_handle
                  << ") via RTLD_NOLOAD | RTLD_LAZY";
    }

    // try to load with the given path
    if(!_lib_handle)
    {
        _lib_handle = dlopen(_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);

        if(_lib_handle)
        {
            LOG(INFO) << "[rocprofiler-sdk-rocattach] loaded " << _lib_name << " library at "
                      << _lib_path.string() << " (handle=" << _lib_handle
                      << ") via RTLD_GLOBAL | RTLD_LAZY";
        }
    }

    // try to load with the absoulte path
    if(!_lib_handle)
    {
        _lib_path   = _lib_path_abs;
        _lib_handle = dlopen(_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
    }

    // try to load with the basename path
    if(!_lib_handle)
    {
        _lib_path   = _lib_path_fname;
        _lib_handle = dlopen(_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
    }

    LOG(INFO) << "[rocprofiler-sdk-rocattach] loaded " << _lib_name << " library at "
              << _lib_path.string() << " (handle=" << _lib_handle << ")";

    LOG_IF(WARNING, _lib_handle == nullptr) << _lib_name << " failed to load\n";

    return _lib_handle;
}
}  // namespace

bool
find_library(void*& addr, int inpid, const std::string& library)
{
    std::stringstream searchname;
    searchname << inpid << "::" << library;
    // TODO: add this back
    // if (target_library_addrs.find(searchname.str()) != target_library_addrs.end())
    //{
    //    return target_library_addrs[searchname.str()];
    //}

    // uses "maps" file to find where library has been loaded in target process
    // does not require this process to be attached
    std::stringstream filename;
    filename << "/proc/" << inpid << "/maps";
    std::ifstream maps(filename.str().c_str());

    if(!maps)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Couldn't open " << filename.str();
        return false;
    }

    std::string line;
    while(std::getline(maps, line))
    {
        if(line.find(library) != std::string::npos)
        {
            ROCP_TRACE << "[rocprofiler-sdk-rocattach] Entry in pid " << inpid
                       << " maps file is: " << line;
            break;
        }
    }

    if(!maps)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Couldn't find library " << library << " in "
                   << filename.str();
        return false;
    }

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    addr = reinterpret_cast<void*>(std::stoull(line, nullptr, 16));
    //  target_library_addrs[searchname.str()] = addr;
    return true;
}

bool
find_symbol(int target_pid, void*& addr, const std::string& library, const std::string& symbol)
{
    auto searchname = std::stringstream{};
    searchname << target_pid << "::" << library << "::" << symbol;
    if(auto itr = m_target_symbol_addrs.find(searchname.str()); itr != m_target_symbol_addrs.end())
    {
        ROCP_TRACE << "[rocprofiler-sdk-rocattach] Found symbol for " << searchname.str() << " at "
                   << itr->second;
        return itr->second != nullptr;
    }

    void* libraryaddr = nullptr;
    void* symboladdr  = nullptr;

    // Load the library in our process to determine the offset of the requested symbol from the
    // start address of the library
    addr        = nullptr;
    libraryaddr = get_library_handle(library);

    if(!libraryaddr)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Host couldn't dlopen " << library;
        return false;
    }

    symboladdr = dlsym(libraryaddr, symbol.c_str());
    if(!symboladdr)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Host couldn't dlsym " << symbol;
        return false;
    }

    // Find the start address of the library in our process
    void* hostlibraryaddr;
    if(!find_library(hostlibraryaddr, getpid(), library))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Couldn't determine where " << library
                   << " was loaded for host";
        return false;
    }

    // Caluclate the offset
    size_t offset =
        reinterpret_cast<size_t>(symboladdr) - reinterpret_cast<size_t>(hostlibraryaddr);
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Offset of " << symbol << " into " << library
               << " calculated as " << offset;

    // Find the start address of the library in the target process
    void* targetlibraryaddr;
    if(!find_library(targetlibraryaddr, target_pid, library))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Couldn't determine where " << library
                   << " was loaded for target";
        return false;
    }

    // Calculate address of symbol in the target process using the offset
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    addr = reinterpret_cast<void*>(reinterpret_cast<size_t>(targetlibraryaddr) + offset);
    m_target_symbol_addrs[searchname.str()] = addr;
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Found symbol for " << searchname.str() << " at "
               << addr;
    return true;
}
}  // namespace rocattach
}  // namespace rocprofiler
