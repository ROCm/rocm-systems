// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "internal_libs.hpp"
#include "binary/analysis.hpp"
#include "binary/binary_info.hpp"
#include "binary/link_map.hpp"
#include "binary/scope_filter.hpp"
#include "binary/symbol.hpp"
#include "common/defines.h"
#include "core/utility.hpp"
#include "fwd.hpp"
#include "log.hpp"

#include <timemory/components/rusage/components.hpp>
#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/environment/types.hpp>
#include <timemory/log/macros.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>
#include <timemory/utility/types.hpp>

#include <algorithm>
#include <dlfcn.h>
#include <initializer_list>
#include <set>
#include <string>
#include <vector>

namespace
{
namespace filepath = ::tim::filepath;
using ::tim::delimit;
using ::tim::get_env;
using ::timemory::join::join;
using strview_init_t   = std::initializer_list<std::string_view>;
using strview_set_t    = std::set<std::string_view>;
using open_modes_vec_t = std::vector<int>;

auto
get_exe_realpath()
{
    return filepath::realpath("/proc/self/exe", nullptr, false);
}

auto&
get_symtab_file_cache()
{
    static auto _cache = std::unordered_map<std::string, std::pair<symtab_t*, bool>>{};
    return _cache;
}

symtab_t*
get_symtab_file(const std::string& _name)
{
    auto& _cache = get_symtab_file_cache();
    auto  itr    = _cache.find(_name);
    if(itr == _cache.end())
    {
        symtab_t* _v        = SymTab::Symtab::findOpenSymtab(_name);
        bool      _closable = (_v == nullptr);
        if(!_v) SymTab::Symtab::openFile(_v, _name);

        TIMEMORY_PREFER(_v != nullptr)
            << "Warning! Dyninst could not open a Symtab instance for file '" << _name
            << "'\n";
        _cache.emplace(_name, std::make_pair(_v, _closable));
    }

    return _cache.at(_name).first;
}

bool
close_symtab_file(const std::string& _name)
{
    auto& _cache = get_symtab_file_cache();
    auto  itr    = _cache.find(_name);
    if(itr != _cache.end())
    {
        symtab_t* _symtab   = itr->second.first;
        bool      _closable = itr->second.second;
        if(_symtab && _closable) SymTab::Symtab::closeSymtab(_symtab);
        _cache.erase(itr);
        return true;
    }
    return false;
}

template <template <typename, typename...> class ContainerT, typename... TailT>
bool
check_regex_restrictions(const ContainerT<std::string, TailT...>& _names,
                         const regexvec_t&                        _regexes)
{
    for(const auto& nitr : _names)
        for(const auto& ritr : _regexes)
            if(std::regex_search(nitr, ritr)) return true;
    return false;
}

std::optional<std::string>
get_linked_path(const char*        _name,
                open_modes_vec_t&& _open_modes = { (RTLD_LAZY | RTLD_NOLOAD) })
{
    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name, _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    tim::scope::destructor _dtor{ [&_noload, &_handle]() {
        if(_noload == false) dlclose(_handle);
    } };

    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        if(_link_map != nullptr && !std::string_view{ _link_map->l_name }.empty())
        {
            return filepath::realpath(_link_map->l_name, nullptr, false);
        }
    }

    return std::optional<std::string>{};
}

std::set<std::string>
get_link_map(const std::string& _lib,
             open_modes_vec_t&& _open_modes = { (RTLD_LAZY | RTLD_NOLOAD) })
{
    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_lib.c_str(), _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    auto _chain = std::set<std::string>{};
    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        struct link_map* _next = _link_map;
        while(_next)
        {
            if(!std::string_view{ _next->l_name }.empty() &&
               std::string_view{ _next->l_name } != _lib)
            {
                _chain.emplace(filepath::realpath(_next->l_name, nullptr, false));
            }
            _next = _next->l_next;
        }

        if(_noload == false) dlclose(_handle);
    }
    return _chain;
}

std::vector<std::string>
get_library_search_paths_impl()
{
    auto _paths = std::vector<std::string>{};

    auto _path_exists = [](const std::string& _filename) {
        struct stat dummy;
        return (_filename.empty()) ? false : (stat(_filename.c_str(), &dummy) == 0);
    };

    auto _emplace_if_exists = [&_paths, _path_exists](const std::string& _directory) {
        if(_path_exists(_directory)) _paths.emplace_back(_directory);
    };

    // search paths from environment variables
    for(const auto& itr : delimit(get_env("LD_LIBRARY_PATH", std::string{}, false), ":"))
        _emplace_if_exists(itr);

    for(const auto& itr : { get_env<std::string>("ROCPROFSYS_ROCM_PATH", ""),
                            get_env<std::string>("ROCM_PATH", ""),
                            std::string{ ROCPROFSYS_DEFAULT_ROCM_PATH } })
    {
        if(!itr.empty())
        {
            for(const auto& ditr : delimit(itr, ":"))
            {
                _emplace_if_exists(join('/', ditr, "lib"));
            }
        }
    }

    // search ld.so.cache
    // apparently ubuntu doesn't like pclosing NULL, so a shared pointer custom
    // destructor is out. Ugh.
    FILE* ldconfig = popen("/sbin/ldconfig -p", "r");
    if(ldconfig)
    {
        constexpr size_t buffer_size = 512;
        char             buffer[buffer_size];
        // ignore first line
        if(fgets(buffer, buffer_size, ldconfig))
        {
            // each line constaining relevant info should be in form:
            //      <LIBRARY_BASENAME> (...) => <RESOLVED_ABSOLUTE_PATH>
            // example:
            //      libz.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libz.so
            auto _get_entry = [](const std::string& _inp) {
                auto _paren_pos = _inp.find('(');
                auto _arrow_pos = _inp.find("=>", _paren_pos);
                if(_arrow_pos == std::string::npos || _paren_pos == std::string::npos)
                    return std::string{};
                if(_arrow_pos + 2 < _inp.length())
                {
                    auto _pos = _inp.find_first_not_of(" \t", _arrow_pos + 2);
                    if(_pos < _inp.length()) return _inp.substr(_pos);
                }
                return std::string{};
            };

            auto _data = std::stringstream{};
            while(fgets(buffer, buffer_size, ldconfig) != nullptr)
            {
                _data << buffer;
                auto _len = strnlen(buffer, buffer_size);
                if(_len > 0 && buffer[_len - 1] == '\n')
                {
                    auto _v = _data.str();
                    if(!_v.empty())
                    {
                        _v = _v.substr(_v.find_first_not_of(" \t"));
                        if(_v.length() > 1)
                        {
                            auto _entry = _get_entry(_v.substr(0, _v.length() - 1));
                            if(!_entry.empty()) _emplace_if_exists(_entry);
                        }
                    }
                    _data = std::stringstream{};
                }
            }
        }
        pclose(ldconfig);
    }

    // search hard-coded system paths
    for(const char* itr :
        { "/usr/local/lib", "/usr/share/lib", "/usr/lib", "/usr/lib64",
          "/usr/lib/x86_64-linux-gnu", "/lib", "/lib64", "/lib/x86_64-linux-gnu" })
    {
        _emplace_if_exists(itr);
    }

    return _paths;
}

std::set<std::string>
get_internal_basic_libs_impl()
{
    auto       _libs    = std::set<std::string>{};
    const auto _exclude = strview_set_t{ LIBM_SO, LIBMVEC_SO };

    // GNU libraries likely to be used by instrumentation
    const auto _gnu_libs = strview_init_t{ LD_LINUX_X86_64_SO, LD_SO,
                                           LIBANL_SO,          LIBBROKENLOCALE_SO,
                                           LIBCRYPT_SO,        LIBC_SO,
                                           LIBDL_SO,           LIBGCC_S_SO,
                                           LIBMVEC_SO,         LIBM_SO,
                                           LIBNSL_SO,          LIBNSS_COMPAT_SO,
                                           LIBNSS_DB_SO,       LIBNSS_DNS_SO,
                                           LIBNSS_FILES_SO,    LIBNSS_HESIOD_SO,
                                           LIBNSS_LDAP_SO,     LIBNSS_NISPLUS_SO,
                                           LIBNSS_NIS_SO,      LIBNSS_TEST1_SO,
                                           LIBNSS_TEST2_SO,    LIBPTHREAD_SO,
                                           LIBRESOLV_SO,       LIBRT_SO,
                                           LIBTHREAD_DB_SO,    LIBUTIL_SO };

    // shared libraries used by or provided by dyninst
    const auto _dyn_libs = strview_init_t{ "libdyninstAPI_RT.so",
                                           "libcommon.so",
                                           "libbfd.so",
                                           "libelf.so",
                                           "libdwarf.so",
                                           "libdw.so",
                                           "libtbb.so",
                                           "libtbbmalloc.so",
                                           "libtbbmalloc_proxy.so",
                                           "libz.so",
                                           "libzstd.so",
                                           "libbz2.so",
                                           "liblzma.so" };

    // shared libraries used by rocprof-sys
    const auto _omni_libs = strview_init_t{ "libstdc++.so.6",
                                            "libgotcha.so",
                                            "libunwind-coredump.so",
                                            "libunwind-generic.so",
                                            "libunwind-ptrace.so",
                                            "libunwind-setjmp.so",
                                            "libunwind.so",
                                            "libunwind-x86_64.so",
                                            "libpapi.so",
                                            "libpfm.so",
                                            "librocm_smi64.so",
                                            "libroctx64.so",
                                            "librocmtools.so",
                                            "libroctracer64.so",
                                            "librocprofiler64.so",
                                            "librocprofiler-register.so",
                                            "librocprofiler-sdk.so",
                                            "librocprofiler-sdk-roctx.so",
                                            "libamd_smi.so",
                                            "libamd_comgr.so" };

    // shared libraries potentially used by timemory
    const auto _3rdparty_libs = strview_init_t{ "libcaliper.so",
                                                "liblikwid.so",
                                                "libprofiler.so",
                                                "libtcmalloc.so",
                                                "libtcmalloc_and_profiler.so",
                                                "libtcmalloc_debug.so",
                                                "libtcmalloc_minimal.so",
                                                "libtcmalloc_minimal_debug.so" };

    for(const auto& gitr : { _gnu_libs, _dyn_libs, _omni_libs, _3rdparty_libs })
    {
        for(auto itr : gitr)
        {
            if(!itr.empty() && _exclude.count(itr) == 0) _libs.emplace(itr);
        }
    }

    // auto _link_map = binary::get_link_map(nullptr, "", "", { (RTLD_LAZY | RTLD_NOLOAD)
    // }); for(const auto& itr : _link_map)
    //    _libs.emplace(itr.real());

    return _libs;
}

std::set<std::string>
get_internal_libs_impl()
{
    auto _libs = get_internal_basic_libs();
    for(auto itr : get_internal_basic_libs())
    {
        if(!itr.empty())
        {
            if(parse_all_modules)
            {
                auto _lib_v = find_libraries(itr);
                if(!_lib_v.empty())
                {
                    for(const auto& litr : _lib_v)
                    {
                        verbprintf(2, "Library '%s' found: %s\n", itr.data(),
                                   litr.c_str());
                        _libs.emplace(litr);
                    }
                }
                else
                {
                    verbprintf(2, "Library '%s' not found\n", itr.data());
                }
            }
            else
            {
                auto _lib_v = find_library(itr);
                if(_lib_v)
                {
                    verbprintf(2, "Library '%s' found: '%s'\n", itr.data(),
                               _lib_v->c_str());
                    _libs.emplace(*_lib_v);
                    if(include_internal_linked_libs)
                    {
                        for(auto&& litr : get_link_map(*_lib_v))
                        {
                            verbprintf(2, "Library '%s' found: '%s' (linked by '%s')\n",
                                       itr.data(), litr.c_str(), _lib_v->c_str());
                            _libs.emplace(litr);
                        }
                    }
                }
                else
                {
                    verbprintf(2, "Library '%s' not found\n", itr.data());
                }
            }
        }
    }

    return _libs;
}

library_module_map_t
get_internal_libs_data_impl()
{
    auto _wc = tim::component::wall_clock{};
    auto _pr = tim::component::peak_rss{};
    _wc.start();
    _pr.start();

    auto _libs_v = get_internal_libs();
    auto _libs   = std::vector<std::string>{};
    _libs.assign(_libs_v.begin(), _libs_v.end());

    auto _rocprofsys_base_path = filepath::dirname(
        filepath::dirname(filepath::realpath("/proc/self/exe", nullptr, false)));
    auto _rocprofsys_lib_path = std::string{};

    for(const auto* itr : { "lib", "lib64" })
    {
        for(const auto* litr :
            { "librocprof-sys-dl.so", "librocprof-sys-user.so", "librocprof-sys-rt.so" })
        {
            auto _libpath = join('/', _rocprofsys_base_path, itr, litr);
            if(filepath::exists(_libpath))
            {
                _libs.emplace_back(filepath::realpath(_libpath, nullptr, false));
            }
        }
    }

    rocprofsys::utility::filter_sort_unique(
        _libs, [](const auto& itr) { return itr.empty() || !filepath::exists(itr); });

    auto _data = library_module_map_t{};
    for(const auto& itr : _libs)
    {
        auto _fpath = filepath::realpath(itr, nullptr, false);
        // allow the user to request this library be considered for instrumentation
        if(check_regex_restrictions(strvec_t{ itr, _fpath }, file_internal_include))
            continue;

        _data.emplace(_fpath, module_func_map_t{});
    }

    auto _odata = ordered(_data);
    for(const auto& itr : _odata)
    {
        symtab_t* _symtab = get_symtab_file(itr.first);
        if(!_symtab) continue;

        verbprintf(0, "[internal] parsing library: '%s'...\n", itr.first.c_str());

        auto _wc_v = tim::component::wall_clock{};
        auto _pr_v = tim::component::peak_rss{};
        _wc_v.start();
        _pr_v.start();

        auto _modules = std::vector<symtab_module_t*>{};
        _symtab->getAllModules(_modules);

        for(const auto& mitr : _modules)
        {
            const auto& _mname = mitr->fileName();
            const auto& _mpath = mitr->fullName();
            // allow the user to request this library be considered for instrumentation
            if(check_regex_restrictions(strvec_t{ _mname, _mpath },
                                        file_internal_include))
                continue;

            verbprintf(3, "[internal]     parsing module: '%s' (via '%s')...\n",
                       _mname.c_str(), filepath::basename(itr.first));

            _data[itr.first].emplace(_mpath, func_set_t{});
            _data[itr.first].emplace(_mname, func_set_t{});

            auto _funcs = mitr->getAllFunctions();

            for(const auto& fitr : _funcs)
            {
                auto _fname = fitr->getName();
                auto _dname = tim::demangle(_fname);

                _data[itr.first][_mpath].emplace(_fname);
                _data[itr.first][_mpath].emplace(_dname);
            }
        }

        _pr_v.stop();
        _wc_v.stop();
        verbprintf(1, "[internal] parsing library: '%s'... Done (%.3f %s, %.3f %s)\n",
                   itr.first.c_str(), _wc_v.get(), _wc_v.display_unit().c_str(),
                   _pr_v.get(), _pr_v.display_unit().c_str());

        // close_symtab_file(itr.first);
    }

    _pr.stop();
    _wc.stop();
    verbprintf(0, "[internal] binary info processing required %.3f %s and %.3f %s\n",
               _wc.get(), _wc.display_unit().c_str(), _pr.get(),
               _pr.display_unit().c_str());

    return _data;
}
}  // namespace

template <typename Tp, typename... TailT>
std::set<Tp>
ordered(const std::unordered_set<Tp, TailT...>& _unordered)
{
    auto _ordered = std::set<Tp>{};
    for(const auto& itr : _unordered)
        _ordered.emplace(itr);
    return _ordered;
}

template <typename KeyT, typename MappedT, typename... TailT>
std::map<KeyT, MappedT>
ordered(const std::unordered_map<KeyT, MappedT, TailT...>& _unordered)
{
    auto _ordered = std::map<KeyT, MappedT>{};
    for(const auto& itr : _unordered)
        _ordered.emplace(itr.first, itr.second);
    return _ordered;
}

std::optional<std::string>
find_library(std::string_view _lib_v)
{
    auto _lib = get_linked_path(_lib_v.data(), { (RTLD_LAZY | RTLD_NOLOAD) });
    if(_lib) return _lib;

    for(const auto& itr : get_library_search_paths())
    {
        auto _path = join('/', itr, _lib_v);
        if(filepath::exists(_path)) return std::optional<std::string>{ _path };
    }

    return std::optional<std::string>{};
}

std::vector<std::string>
find_libraries(std::string_view _lib_v)
{
    auto _libs = std::vector<std::string>{};

    auto _lib = get_linked_path(_lib_v.data(), { (RTLD_LAZY | RTLD_NOLOAD) });
    if(_lib) _libs.emplace_back(*_lib);

    for(const auto& itr : get_library_search_paths())
    {
        auto _path = join('/', itr, _lib_v);
        if(filepath::exists(_path)) _libs.emplace_back(_path);
    }

    return _libs;
}

const std::vector<std::string>&
get_library_search_paths()
{
    static auto _v = get_library_search_paths_impl();
    return _v;
}

std::set<std::string>&
get_internal_basic_libs()
{
    static auto _v = get_internal_basic_libs_impl();
    return _v;
}

std::set<std::string>&
get_internal_libs()
{
    static auto _v = get_internal_libs_impl();
    return _v;
}

const library_module_map_t&
get_internal_libs_data()
{
    static auto _v = get_internal_libs_data_impl();
    return _v;
}

void
parse_internal_libs_data()
{
    (void) get_internal_libs_data();
}
