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

#include "function_signature.hpp"
#include "fwd.hpp"
#include "log.hpp"
#include "rocprof-sys-instrument.hpp"

#include <timemory/components/rusage/components.hpp>
#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/utility/join.hpp>

#include <algorithm>
#include <link.h>
#include <linux/limits.h>
#include <string>
#include <vector>

static int expect_error = NO_ERROR;
static int error_print  = 0;

// set of whole function names to exclude
strset_t
get_whole_function_names()
{
    return strset_t{
        "sem_init", "sem_destroy", "sem_open", "sem_close", "sem_post", "sem_wait",
        "sem_getvalue", "sem_clockwait", "sem_timedwait", "sem_trywait", "sem_unlink",
        "fork", "do_futex_wait", "dl_iterate_phdr", "dlinfo", "dlopen", "dlmopen",
        "dlvsym", "dlsym", "dlerror", "dladdr", "_dl_sym", "_dl_vsym", "_dl_addr",
        "_dl_relocate_static_pie", "getenv", "setenv", "unsetenv", "printf", "fprintf",
        "vprintf", "buffered_vfprintf", "vfprintf", "printf_positional", "puts", "fputs",
        "vfputs", "fflush", "fwrite", "malloc", "malloc_stats", "malloc_trim", "mallopt",
        "calloc", "free", "pvalloc", "valloc", "sysmalloc", "posix_memalign", "freehook",
        "mallochook", "memalignhook", "mprobe", "reallochook", "mmap", "munmap", "fopen",
        "fclose", "fmemopen", "fmemclose", "backtrace", "backtrace_symbols",
        "backtrace_symbols_fd", "sigaddset", "sigandset", "sigdelset", "sigemptyset",
        "sigfillset", "sighold", "sigisemptyset", "sigismember", "sigorset", "sigrelse",
        "sigvec", "strtok", "strstr", "sbrk", "strxfrm", "atexit", "ompt_start_tool",
        "nanosleep", "cfree", "tolower", "toupper", "fileno", "fileno_unlocked", "exit",
        "quick_exit", "abort", "mbind", "migrate_pages", "move_pages",
        "numa_migrate_pages", "numa_move_pages", "numa_alloc", "numa_alloc_local",
        "numa_alloc_interleaved", "numa_alloc_onnode", "numa_realloc", "numa_free",
        "round_and_return", "_init", "_fini", "_start", "__do_global_dtors_aux",
        "__libc_csu_init", "__libc_csu_fini", "__hip_module_ctor", "__hip_module_dtor",
        "__hipRegisterManagedVar", "__hipRegisterFunction", "__hipPushCallConfiguration",
        "__hipPopCallConfiguration", "hipApiName", "enlarge_userbuf",
        // below are functions which never terminate
        "rocr::core::Signal::WaitAny", "rocr::core::Runtime::AsyncEventsLoop",
        "rocr::core::BusyWaitSignal::WaitAcquire",
        "rocr::core::BusyWaitSignal::WaitRelaxed", "rocr::HSA::hsa_signal_wait_scacquire",
        "rocr::os::ThreadTrampoline", "rocr::image::ImageRuntime::CreateImageManager",
        "rocr::AMD::GpuAgent::GetInfo", "rocr::HSA::hsa_agent_get_info",
        "event_base_loop", "bootstrapRoot", "bootstrapNetAccept", "ncclCommInitRank",
        "ncclCommInitAll", "ncclCommDestroy", "ncclCommCount", "ncclCommCuDevice",
        "ncclCommUserRank", "ncclReduce", "ncclBcast", "ncclBroadcast", "ncclAllReduce",
        "ncclReduceScatter", "ncclAllGather", "ncclGroupStart", "ncclGroupEnd",
        "ncclSend", "ncclRecv", "ncclGather", "ncclScatter", "ncclAllToAll",
        "ncclAllToAllv", "ncclSocketAccept", "vaBeginPicture", "vaCreateBuffer",
        "vaCreateConfig", "vaCreateContext", "vaCreateSurfaces", "vaDestroySurfaces",
        "vaSyncSurface", "vaDestroyBuffer", "vaDestroyConfig", "vaDestroyContext",
        "vaEndPicture", "vaExportSurfaceHandle", "vaGetConfigAttributes", "vaInitialize",
        "vaQueryConfigEntrypoints", "vaQuerySurfaceAttributes", "vaQuerySurfaceStatus",
        "vaRenderPicture", "vaTerminate", "vaDisplayIsValid"
    };
}

//======================================================================================//
//
//  Helper functions because the syntax for getting a function or module name is unwieldy
//
std::string_view
get_name(procedure_t* _func)
{
    static auto _v = std::unordered_map<procedure_t*, std::string>{};

    auto itr = _v.find(_func);
    if(itr == _v.end())
    {
        _v.emplace(_func, (_func) ? _func->getDemangledName() : std::string{});
    }

    return _v.at(_func);
}

std::string_view
get_name(module_t* _module)
{
    static auto _v = std::unordered_map<module_t*, std::string>{};

    auto itr = _v.find(_module);
    if(itr == _v.end())
    {
        char _name[FUNCNAMELEN + 1];
        memset(_name, '\0', FUNCNAMELEN + 1);

        if(_module)
        {
            _module->getFullName(_name, FUNCNAMELEN);
            _v.emplace(_module, std::string{ _name });
        }
        else
        {
            _v.emplace(nullptr, std::string{});
        }
    }

    return _v.at(_module);
}

symtab_func_t*
get_symtab_function(procedure_t* _func)
{
    static auto _v = std::unordered_map<procedure_t*, symtab_func_t*>{};

    auto itr = _v.find(_func);
    if(itr == _v.end())
    {
        auto _name = _func->getName();
        {
            auto nitr = symtab_data.mangled_symbol_names.find(_name);
            if(nitr != symtab_data.mangled_symbol_names.end())
            {
                _v.emplace(_func, nitr->second->getFunction());
                return _v.at(_func);
            }
        }

        for(auto& fitr : symtab_data.symbols)
        {
            if(_name == fitr.first->getName())
            {
                _v.emplace(_func, fitr.first);
                return _v.at(_func);
            }
        }

        auto _dname = _func->getDemangledName();

        {
            auto nitr = symtab_data.typed_func_names.find(_dname);
            if(nitr != symtab_data.typed_func_names.end())
            {
                _v.emplace(_func, nitr->second);
                return _v.at(_func);
            }
        }

        {
            auto nitr = symtab_data.typed_symbol_names.find(_dname);
            if(nitr != symtab_data.typed_symbol_names.end())
            {
                _v.emplace(_func, nitr->second->getFunction());
                return _v.at(_func);
            }
        }

        if(_v.find(_func) == _v.end()) _v.emplace(_func, nullptr);
    }

    return _v.at(_func);
}

namespace
{
std::string
get_return_type(procedure_t* func)
{
    if(func && func->isInstrumentable() && func->getReturnType())
        return func->getReturnType()->getName();
    return std::string{};
}

auto
get_parameter_types(procedure_t* func)
{
    auto _param_names = std::vector<std::string>{};
    if(func && func->isInstrumentable())
    {
        auto* _params = func->getParams();
        if(_params)
        {
            _param_names.reserve(_params->size());
            for(auto* itr : *_params)
            {
                std::string _name = itr->getType()->getName();
                if(_name.empty()) _name = itr->getName();
                _param_names.emplace_back(_name);
            }
        }
    }
    return _param_names;
}
}  // namespace

//======================================================================================//
//
//  We create a new name that embeds the file and line information in the name
//
function_signature
get_func_file_line_info(module_t* module, procedure_t* func)
{
    using address_t = Dyninst::Address;

    ROCPROFSYS_ADD_LOG_ENTRY("Getting function line info for", get_name(func));

    auto _file_name   = get_name(module);
    auto _func_name   = get_name(func);
    auto _return_type = get_return_type(func);
    auto _param_types = get_parameter_types(func);
    auto _base_addr   = address_t{};
    auto _last_addr   = address_t{};
    auto _src_lines   = std::vector<statement_t>{};

    if(func->getAddressRange(_base_addr, _last_addr) &&
       module->getSourceLines(_base_addr, _src_lines) && !_src_lines.empty())
    {
        auto _row = _src_lines.front().lineNumber();
        return function_signature(_return_type, _func_name, _file_name, _param_types,
                                  { _row, 0 }, { 0, 0 }, false, true, false);
    }
    else
    {
        return function_signature(_return_type, _func_name, _file_name, _param_types,
                                  { 0, 0 }, { 0, 0 }, false, false, false);
    }
}

//======================================================================================//
//
//  Gets information (line number, filename, and column number) about
//  the instrumented loop and formats it properly.
//
function_signature
get_loop_file_line_info(module_t* module, procedure_t* func, flow_graph_t*,
                        basic_loop_t* loopToInstrument)
{
    ROCPROFSYS_ADD_LOG_ENTRY("Getting loop line info for", get_name(func));

    auto basic_blocks = std::vector<BPatch_basicBlock*>{};
    loopToInstrument->getLoopBasicBlocksExclusive(basic_blocks);

    if(basic_blocks.empty()) return function_signature{ "", "", "" };

    auto* _block     = basic_blocks.front();
    auto  _base_addr = _block->getStartAddress();
    auto  _last_addr = _block->getEndAddress();
    for(const auto& itr : basic_blocks)
    {
        if(itr == _block) continue;
        if(itr->dominates(_block))
        {
            _base_addr = itr->getStartAddress();
            _last_addr = itr->getEndAddress();
            _block     = itr;
        }
    }

    auto _file_name   = get_name(module);
    auto _func_name   = get_name(func);
    auto _return_type = get_return_type(func);
    auto _param_types = get_parameter_types(func);
    auto _lines_beg   = std::vector<statement_t>{};
    auto _lines_end   = std::vector<statement_t>{};

    if(module->getSourceLines(_base_addr, _lines_beg))
    {
        // filename = lines[0].fileName();
        int _row1 = 0;
        int _col1 = 0;
        for(auto& itr : _lines_beg)
        {
            if(itr.lineNumber() > 0)
            {
                _row1 = itr.lineNumber();
                _col1 = itr.lineOffset();
                break;
            }
        }

        if(_row1 == 0 && _col1 == 0)
            return function_signature(_return_type, _func_name, _file_name, _param_types);

        int _row2 = 0;
        int _col2 = 0;
        for(auto& itr : _lines_beg)
        {
            _row2 = std::max(_row2, itr.lineNumber());
            _col2 = std::max(_col2, itr.lineOffset());
        }

        if(_col1 < 0) _col1 = 0;

        if(module->getSourceLines(_last_addr, _lines_end))
        {
            for(auto& itr : _lines_end)
            {
                _row2 = std::max(_row2, itr.lineNumber());
                _col2 = std::max(_col2, itr.lineOffset());
            }
            if(_col2 < 0) _col2 = 0;
            if(_row2 < _row1) _row1 = _row2;  // Fix for wrong line numbers

            return function_signature(_return_type, _func_name, _file_name, _param_types,
                                      { _row1, _row2 }, { _col1, _col2 }, true, true,
                                      true);
        }
        else
        {
            return function_signature(_return_type, _func_name, _file_name, _param_types,
                                      { _row1, 0 }, { _col1, 0 }, true, true, false);
        }
    }
    else
    {
        return function_signature(_return_type, _func_name, _file_name, _param_types,
                                  { 0, 0 }, { 0, 0 }, true, false, false);
    }
}

//======================================================================================//
//
//  Gets information (line number, filename, and column number) about
//  the instrumented loop and formats it properly.
//
std::map<basic_block_t*, basic_block_signature>
get_basic_block_file_line_info(module_t* module, procedure_t* func)
{
    std::map<basic_block_t*, basic_block_signature> _data{};
    if(!func) return _data;

    ROCPROFSYS_ADD_LOG_ENTRY("Getting basic block line info for", get_name(func));

    auto* _cfg          = func->getCFG();
    auto  _basic_blocks = std::set<BPatch_basicBlock*>{};
    _cfg->getAllBasicBlocks(_basic_blocks);

    if(_basic_blocks.empty()) return _data;

    auto _file_name   = get_name(module);
    auto _func_name   = get_name(func);
    auto _return_type = get_return_type(func);
    auto _param_types = get_parameter_types(func);

    for(auto&& itr : _basic_blocks)
    {
        auto _base_addr = itr->getStartAddress();
        auto _last_addr = itr->getEndAddress();

        verbprintf(4,
                   "[%s][%s] basic_block: size = %lu: base_addr = %lu, last_addr = %lu\n",
                   _file_name.data(), _func_name.data(),
                   (unsigned long) (_last_addr - _base_addr), _base_addr, _last_addr);

        auto _lines_beg = std::vector<statement_t>{};
        auto _lines_end = std::vector<statement_t>{};

        if(module->getSourceLines(_base_addr, _lines_beg) && !_lines_beg.empty())
        {
            int _row1 = _lines_beg.front().lineNumber();
            int _col1 = _lines_beg.front().lineOffset();

            verbprintf(4, "size of _lines_end = %lu\n",
                       (unsigned long) _lines_end.size());

            if(module->getSourceLines(_last_addr, _lines_end) && !_lines_end.empty())
            {
                int _row2 = _lines_end.back().lineNumber();
                int _col2 = _lines_end.back().lineOffset();

                if(_row2 < _row1) std::swap(_row1, _row2);
                if(_row1 == _row2 && _col2 < _col1) std::swap(_col1, _col2);

                _data.emplace(
                    itr, basic_block_signature{
                             _base_addr, _last_addr,
                             function_signature(_return_type, _func_name, _file_name,
                                                _param_types, { _row1, _row2 },
                                                { _col1, _col2 }, true, true, true) });
            }
            else
            {
                _data.emplace(itr,
                              basic_block_signature{
                                  _base_addr, _last_addr,
                                  function_signature(_return_type, _func_name, _file_name,
                                                     _param_types, { _row1, 0 },
                                                     { _col1, 0 }, true, true, false) });
            }
        }
        else
        {
            _data.emplace(itr, basic_block_signature{
                                   _base_addr, _last_addr,
                                   function_signature(_return_type, _func_name,
                                                      _file_name, _param_types) });
        }
    }

    return _data;
}

//======================================================================================//
//
//  We create a new name that embeds the file and line information in the name
//
std::vector<statement_t>
get_source_code(module_t* module, procedure_t* func)
{
    ROCPROFSYS_ADD_LOG_ENTRY("Getting source code for", get_name(func));

    std::vector<statement_t> _lines{};
    if(!module || !func) return _lines;
    auto*                        _cfg = func->getCFG();
    std::set<BPatch_basicBlock*> _basic_blocks{};
    _cfg->getAllBasicBlocks(_basic_blocks);

    for(auto&& itr : _basic_blocks)
    {
        auto _base_addr = itr->getStartAddress();
        auto _last_addr = itr->getEndAddress();
        for(decltype(_base_addr) _addr = _base_addr; _addr <= _last_addr; ++_addr)
        {
            std::vector<statement_t> _src{};
            if(module->getSourceLines(_addr, _src))
            {
                for(auto&& iitr : _src)
                    _lines.emplace_back(iitr);
            }
        }
    }
    return _lines;
}

//======================================================================================//
//
//  For compatibility purposes
//
procedure_t*
find_function(image_t* app_image, const std::string& _name, const strset_t& _extra)
{
    if(_name.empty()) return nullptr;

    auto _find = [app_image](const std::string& _f) -> procedure_t* {
        // Extract the vector of functions
        std::vector<procedure_t*> _found;
        auto* ret = app_image->findFunction(_f.c_str(), _found, false, true, true);
        if(ret == nullptr || _found.empty()) return nullptr;
        return _found.at(0);
    };

    procedure_t* _func = _find(_name);
    auto         itr   = _extra.begin();
    while(_func == nullptr && itr != _extra.end())
    {
        _func = _find(*itr);
        ++itr;
    }

    if(!_func)
    {
        verbprintf(1, "function: '%s' ... not found\n", _name.c_str());
    }
    else
    {
        verbprintf(1, "function: '%s' ... found\n", _name.c_str());
    }

    return _func;
}

//======================================================================================//
//
//  Get the realpath to this exe
//
bool
is_text_file(const std::string& filename)
{
    std::ifstream _file{ filename, std::ios::in | std::ios::binary };
    if(!_file.is_open())
    {
        errprintf(-1, "Error! '%s' could not be opened...\n", filename.c_str());
        return false;
    }

    constexpr size_t buffer_size = 1024;
    char             buffer[buffer_size];
    while(_file.read(buffer, sizeof(buffer)))
    {
        for(char itr : buffer)
        {
            if(itr == '\0') return false;
        }
    }

    if(_file.gcount() > 0)
    {
        for(std::streamsize i = 0; i < _file.gcount(); ++i)
        {
            if(buffer[i] == '\0') return false;
        }
    }

    return true;
}

//======================================================================================//
//
//  Get the realpath to this exe
//
std::string&
rocprofsys_get_exe_realpath()
{
    static std::string _v = []() {
        auto _cmd_line = tim::read_command_line(tim::process::get_id());
        if(!_cmd_line.empty())
        {
            using array_config_t = timemory::join::array_config;
            ROCPROFSYS_ADD_DETAILED_LOG_ENTRY(array_config_t{ " ", "[ ", " ]" },
                                              "cmdline:: ", _cmd_line);
            return _cmd_line.front();
            // return tim::filepath::realpath(_cmd_line.front(), nullptr, false);
        }
        return std::string{};
    }();
    return _v;
}

//======================================================================================//
//
//  Error callback routine.
//
std::vector<std::string>
rocprofsys_get_link_map(const char* _lib, const std::string& _exclude_linked_by,
                        const std::string& _exclude_re, std::vector<int>&& _open_modes)
{
    if(_open_modes.empty()) _open_modes = { (RTLD_LAZY | RTLD_NOLOAD) };

    auto _get_chain = [&_open_modes](const char* _name) {
        void* _handle = nullptr;
        bool  _noload = false;
        for(auto _mode : _open_modes)
        {
            _handle = dlopen(_name, _mode);
            _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
            if(_handle) break;
        }

        auto _chain = std::vector<std::string>{};
        if(_handle)
        {
            struct link_map* _link_map = nullptr;
            dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
            struct link_map* _next = _link_map;
            while(_next)
            {
                if(_name == nullptr && _next == _link_map &&
                   std::string_view{ _next->l_name }.empty())
                {
                    // only insert exe name if dlopened the exe and
                    // empty name is first entry
                    _chain.emplace_back(rocprofsys_get_exe_realpath());
                }
                else if(!std::string_view{ _next->l_name }.empty())
                {
                    _chain.emplace_back(_next->l_name);
                }
                _next = _next->l_next;
            }

            if(_noload == false) dlclose(_handle);
        }
        return _chain;
    };

    auto _full_chain = _get_chain(_lib);
    auto _excl_chain = (_exclude_linked_by.empty())
                           ? std::vector<std::string>{}
                           : _get_chain(_exclude_linked_by.c_str());
    auto _fini_chain = std::vector<std::string>{};
    _fini_chain.reserve(_full_chain.size());

    for(const auto& itr : _full_chain)
    {
        auto _found = std::any_of(_excl_chain.begin(), _excl_chain.end(),
                                  [itr](const auto& _v) { return (itr == _v); });
        if(!_found)
        {
            if(_exclude_re.empty() || !std::regex_search(itr, std::regex{ _exclude_re }))
                _fini_chain.emplace_back(itr);
            else
                _excl_chain.emplace_back(itr);
        }
    }

    return _fini_chain;
}

//======================================================================================//
//
//  Get the path of a loaded dynamic binary
//
std::optional<std::string>
rocprofsys_get_loaded_path(const char* _name, std::vector<int>&& _open_modes)
{
    if(_open_modes.empty()) _open_modes = { (RTLD_LAZY | RTLD_NOLOAD) };

    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name, _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        if(_link_map != nullptr && !std::string_view{ _link_map->l_name }.empty())
        {
            return tim::filepath::realpath(_link_map->l_name, nullptr, false);
        }
        if(_noload == false) dlclose(_handle);
    }

    return std::optional<std::string>{};
}

//======================================================================================//
//
//  Get the path of a loaded dynamic binary
//
std::optional<std::string>
rocprofsys_get_origin(const char* _name, std::vector<int>&& _open_modes)
{
    if(_open_modes.empty()) _open_modes = { (RTLD_LAZY | RTLD_NOLOAD) };

    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name, _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    if(_handle)
    {
        char _buffer[PATH_MAX + 1];
        memset(_buffer, '\0', PATH_MAX * sizeof(char));
        dlinfo(_handle, RTLD_DI_ORIGIN, _buffer);
        if(strnlen(_buffer, PATH_MAX + 1) <= PATH_MAX)
        {
            return tim::filepath::realpath(_buffer, nullptr, false);
        }
        if(_noload == false) dlclose(_handle);
    }

    return std::optional<std::string>{};
}

//======================================================================================//
//
//  Error callback routine.
//
void
errorFunc(error_level_t level, int num, const char** params)
{
    error_func_real(level, num, params);
}

//======================================================================================//
//
void
error_func_real(error_level_t level, int num, const char* const* params)
{
    char line[4096];

    const char* msg = bpatch->getEnglishErrorString(num);
    bpatch->formatErrorString(line, sizeof(line), msg, params);

    ROCPROFSYS_ADD_LOG_ENTRY("Dyninst error function called with level", level,
                             ":: ID# =", num, "::", line)
        .force(level < BPatchInfo);

    if(num == 0)
    {
        // conditional reporting of warnings and informational messages
        if(error_print > 0)
        {
            if(level == BPatchInfo)
            {
                errprintf(2, "%s :: %i :: %s\n%s", std::to_string(level).c_str(), num,
                          line, tim::log::color::end());
            }
            else
            {
                verbprintf(0, "%s :: %i :: %s\n%s", std::to_string(level).c_str(), num,
                           line, tim::log::color::end());
            }
        }
    }
    else
    {
        // reporting of actual errors
        if(num != expect_error)
        {
            verbprintf(-1, "%s :: %i :: %s\n%s", std::to_string(level).c_str(), num, line,
                       tim::log::color::end());
            // We consider some errors fatal.
            if(num == 101) throw std::runtime_error(msg);
        }
    }
}

//======================================================================================//
//
//  Just log it
//
void
error_func_fake(error_level_t level, int num, const char* const* params)
{
    char line[4096];

    const char* msg = bpatch->getEnglishErrorString(num);
    bpatch->formatErrorString(line, sizeof(line), msg, params);

    // just log it
    ROCPROFSYS_ADD_LOG_ENTRY("Dyninst error function called with level", level,
                             ":: ID# =", num, "::", line)
        .force(level < BPatchInfo);
}

#include "internal_libs.hpp"

#include <timemory/components/timing/wall_clock.hpp>
#include <timemory/utility/join.hpp>

using ::timemory::join::join;

//======================================================================================//
//
//  Read the symtab data from Dyninst
//
void
process_modules(const std::vector<module_t*>& _app_modules)
{
    parse_internal_libs_data();

    auto _erase_nullptrs = [](auto& _vec) {
        _vec.erase(std::remove_if(_vec.begin(), _vec.end(),
                                  [](const auto* itr) { return (itr == nullptr); }),
                   _vec.end());
    };

    auto _wc = tim::component::wall_clock{};
    auto _pr = tim::component::peak_rss{};
    _wc.start();
    _pr.start();

    for(auto* itr : _app_modules)
    {
        auto* _module = SymTab::convert(itr);
        if(_module) symtab_data.modules.emplace_back(_module);
    }

    _erase_nullptrs(symtab_data.modules);

    verbprintf(0, "Processing %zu modules...\n", symtab_data.modules.size());

    if(symtab_data.modules.empty()) return;

    const auto& _data  = get_internal_libs_data();
    auto        _names = std::set<std::string_view>{};
    for(const auto& itr : _data)
    {
        if(!itr.first.empty())
        {
            _names.emplace(itr.first);
            for(const auto& ditr : itr.second)
                _names.emplace(ditr.first);
        }
    }

    for(auto* itr : symtab_data.modules)
    {
        const auto* _base_name = tim::filepath::basename(itr->fullName());
        auto        _real_name = tim::filepath::realpath(itr->fullName(), nullptr, false);

        if(!_base_name) continue;

        if(_names.count(_base_name) == 0 && _names.count(_real_name) == 0)
        {
            verbprintf(2, "Processing symbol table for module '%s'...\n",
                       itr->fullName().c_str());
        }

        symtab_data.functions.emplace(itr, std::vector<symtab_func_t*>{});
        if(itr->getAllFunctions().empty()) continue;
        _erase_nullptrs(symtab_data.functions.at(itr));

        for(auto* fitr : symtab_data.functions.at(itr))
        {
            symtab_data.typed_func_names[tim::demangle(fitr->getName())] = fitr;

            symtab_data.symbols.emplace(fitr, std::vector<symtab_symbol_t*>{});
            if(!fitr->getSymbols(symtab_data.symbols.at(fitr))) continue;
            _erase_nullptrs(symtab_data.symbols.at(fitr));

            for(auto* sitr : symtab_data.symbols.at(fitr))
            {
                symtab_data.mangled_symbol_names[sitr->getMangledName()] = sitr;
                symtab_data.typed_symbol_names[sitr->getTypedName()]     = sitr;
            }
        }
    }

    _pr.stop();
    _wc.stop();
    verbprintf(0, "Processing %zu modules... Done (%.3f %s, %.3f %s)\n",
               _app_modules.size(), _wc.get(), _wc.display_unit().c_str(), _pr.get(),
               _pr.display_unit().c_str());
}

//======================================================================================//
//
//  I/O assistance
//

namespace std
{
std::string
to_string(instruction_category_t _category)
{
    using namespace Dyninst::InstructionAPI;
    switch(_category)
    {
        case c_CallInsn: return "function_call";
        case c_ReturnInsn: return "return";
        case c_BranchInsn: return "branch";
        case c_CompareInsn: return "compare";
        case c_PrefetchInsn: return "prefetch";
        case c_SysEnterInsn: return "sys_enter";
        case c_SyscallInsn: return "sys_call";
        case c_VectorInsn: return "vector";
        case c_GPUKernelExitInsn: return "gpu_kernel_exit";
        case c_NoCategory: return "no_category";
    }
    return std::string{ "unknown_category_id_" } +
           std::to_string(static_cast<int>(_category));
}

std::string
to_string(error_level_t _level)
{
    switch(_level)
    {
        case BPatchFatal:
        {
            return JOIN("", tim::log::color::fatal(), "FatalError");
        }
        case BPatchSerious:
        {
            return JOIN("", tim::log::color::fatal(), "SeriousError");
        }
        case BPatchWarning:
        {
            return JOIN("", tim::log::color::warning(), "Warning");
        }
        case BPatchInfo:
        {
            return JOIN("", tim::log::color::info(), "Info");
        }
        default: break;
    }

    return JOIN("", tim::log::color::warning(), "UnknownErrorLevel",
                static_cast<int>(_level));
}

namespace
{
std::string&&
to_lower(std::string&& _v)
{
    for(auto& itr : std::move(_v))
        itr = tolower(itr);
    return std::move(_v);
}
}  // namespace

std::string
to_string(symbol_visibility_t _v)
{
    return to_lower(SymTab::Symbol::symbolVisibility2Str(_v) + 3);
}

std::string
to_string(symbol_linkage_t _v)
{
    return to_lower(SymTab::Symbol::symbolLinkage2Str(_v) + 3);
}
}  // namespace std

template <typename Tp>
Tp
from_string(std::string_view _v)
{
    if constexpr(std::is_same<Tp, symbol_visibility_t>::value)
    {
        for(const auto& itr :
            { SV_UNKNOWN, SV_DEFAULT, SV_INTERNAL, SV_HIDDEN, SV_PROTECTED })
            if(_v == std::to_string(itr)) return itr;
        return SV_UNKNOWN;
    }
    else if constexpr(std::is_same<Tp, symbol_linkage_t>::value)
    {
        for(const auto& itr : { SL_UNKNOWN, SL_GLOBAL, SL_LOCAL, SL_WEAK, SL_UNIQUE })
            if(_v == std::to_string(itr)) return itr;
        return SL_UNKNOWN;
    }
    else
    {
        static_assert(std::is_empty<Tp>::value, "Error! not defined");
        return Tp{};
    }
}

template symbol_visibility_t
from_string<symbol_visibility_t>(std::string_view _v);

template symbol_linkage_t
from_string<symbol_linkage_t>(std::string_view _v);

std::ostream&
operator<<(std::ostream& _os, symbol_linkage_t _v)
{
    return (_os << std::to_string(_v));
}

std::ostream&
operator<<(std::ostream& _os, symbol_visibility_t _v)
{
    return (_os << std::to_string(_v));
}

std::istream&
operator>>(std::istream& _is, symbol_linkage_t& _v)
{
    auto _v_s = std::string{};
    _is >> _v_s;
    _v = from_string<symbol_linkage_t>(_v_s);
    return _is;
}

std::istream&
operator>>(std::istream& _is, symbol_visibility_t& _v)
{
    auto _v_s = std::string{};
    _is >> _v_s;
    _v = from_string<symbol_visibility_t>(_v_s);
    return _is;
}
