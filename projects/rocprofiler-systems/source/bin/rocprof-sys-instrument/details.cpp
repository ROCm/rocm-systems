// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "function_signature.hpp"
#include "fwd.hpp"
#include "log.hpp"
#include "rocprof-sys-instrument.hpp"

#include <timemory/components/rusage/components.hpp>
#include <timemory/components/timing/wall_clock.hpp>

#include "common/path.hpp"
#include "core/demangler.hpp"

#include <spdlog/fmt/ranges.h>

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

// True if any regex in _res matches _name.
bool
regex_match_any(const std::string& _name, const regexvec_t& _res)
{
    return std::any_of(_res.begin(), _res.end(),
                       [&](const auto& _re) { return std::regex_search(_name, _re); });
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

        // Filter out DWARF line-0 entries ("no source statement") which some
        // compilers (e.g. amdclang++) emit for compiler-generated basic blocks.
        auto _remove_line_zero = [](std::vector<statement_t>& _lines) {
            _lines.erase(
                std::remove_if(_lines.begin(), _lines.end(),
                               [](statement_t& stmt) { return stmt.lineNumber() == 0; }),
                _lines.end());
        };

        if(module->getSourceLines(_base_addr, _lines_beg)) _remove_line_zero(_lines_beg);

        if(!_lines_beg.empty())
        {
            int _row1 = _lines_beg.front().lineNumber();
            int _col1 = _lines_beg.front().lineOffset();

            verbprintf(4, "size of _lines_end = %lu\n",
                       (unsigned long) _lines_end.size());

            if(module->getSourceLines(_last_addr, _lines_end))
                _remove_line_zero(_lines_end);

            if(!_lines_end.empty())
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
//  find_function: the module list overload searches only pre-parsed modules (fast).
//  The object list overload searches objects (which map over their contained modules).
//

template <typename FinderT>
static procedure_t*
find_function_impl(FinderT&& _find, const std::string& _name, const strset_t& _extra)
{
    if(_name.empty()) return nullptr;

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

procedure_t*
find_function(const std::vector<module_t*>& _modules, const std::string& _name,
              const strset_t& _extra)
{
    return find_function_impl(
        [&_modules](const std::string& _f) -> procedure_t* {
            std::vector<procedure_t*> _found;
            for(auto* mod : _modules)
            {
                if(!mod) continue;
                _found.clear();
                auto* ret =
                    mod->findFunction(_f.c_str(), _found, false, true, false, false);
                if(ret && !_found.empty()) return _found.at(0);
            }
            return nullptr;
        },
        _name, _extra);
}

procedure_t*
find_function(const std::vector<object_t*>& _objects, const std::string& _name,
              const strset_t& _extra)
{
    return find_function_impl(
        [&_objects](const std::string& _f) -> procedure_t* {
            std::vector<procedure_t*> _found;
            for(auto* obj : _objects)
            {
                if(!obj) continue;
                _found.clear();
                auto* ret = obj->findFunction(_f, _found, false, true, false, false);
                if(ret && !_found.empty()) return _found.at(0);
            }
            return nullptr;
        },
        _name, _extra);
}

//======================================================================================//
//
//  Find undefined function symbols (external references) across the provided objects
//
symtab_symbol_t*
find_undefined_function_symbol(const std::unordered_set<object_t*>& _objects,
                               const std::string&                   _name)
{
    if(_name.empty() || _objects.empty()) return nullptr;

    // Search helper lambda for code reuse
    auto _find_symbol = [](SymTab::Symtab*    symtab,
                           const std::string& target_name) -> symtab_symbol_t* {
        if(!symtab) return nullptr;

        std::vector<SymTab::Symbol*> all_symbols;
        if(!symtab->getAllSymbols(all_symbols)) return nullptr;

        for(auto* symbol : all_symbols)
        {
            if(!symbol || symbol->getType() != SymTab::Symbol::ST_FUNCTION ||
               symbol->getRegion())
                continue;

            // Try all possible symbol name representations
            std::string symbol_name = symbol->getPrettyName();
            if(symbol_name.empty()) symbol_name = symbol->getMangledName();
            if(symbol_name.empty()) symbol_name = symbol->getTypedName();

            // Check for exact match and undefined function criteria
            if(symbol_name == target_name) return symbol;
        }
        return nullptr;
    };

    for(auto* obj : _objects)
    {
        if(!obj) continue;

        std::string binary_path = obj->pathName();
        // Open Symtab directly for comprehensive symbol access
        SymTab::Symtab* symtab = nullptr;
        if(!SymTab::Symtab::openFile(symtab, binary_path))
        {
            verbprintf(1, "Failed to open Symtab for: %s\n", binary_path.c_str());
            continue;
        }

        // Search for the primary symbol name
        auto* result = _find_symbol(symtab, _name);
        if(result)
        {
            verbprintf(1, "Found undefined function symbol: '%s' in %s\n", _name.c_str(),
                       binary_path.c_str());
            return result;
        }
    }

    verbprintf(1, "Undefined function symbol: '%s' ... not found\n", _name.c_str());
    return nullptr;
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
            ROCPROFSYS_ADD_LOG_ENTRY(
                fmt::format("cmdline:: [ {} ]", fmt::join(_cmd_line, " ")));
            return _cmd_line.front();
        }
        return std::string{};
    }();
    return _v;
}

//======================================================================================//
//
//  Get the estimated number of procedures in an object via Dyninst SymtabAPI.
//  This is orders of magnitude quicker than querying the respective
//  getProcedures()->size() However, we lose some accuracy.
//
//  We do not assume that process_module has been called.
//
//  E.g: With libomptarget.so, SymtabAPI reports ~49750 procedures, whilst
//       BPatch_module::getProcedures()->size() reports ~49990.
//
//  Due to this, the returned value should be considered a lower bound.
//

size_t
get_object_procedure_count_lb(object_t* _object)
{
    if(!_object) return 0;

    SymTab::Symtab* _st = SymTab::convert(_object);
    if(!_st)
    {
        verbprintf(1,
                   "Warning! Failed to convert object %s to SymtabAPI for "
                   "procedure count... assuming 0\n",
                   _object->name().c_str());
        return 0;
    }

    std::vector<symtab_func_t*> _fns;
    _st->getAllFunctions(_fns);  // API does not return object
    return _fns.size();
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
            return rocprofsys::path::realpath(_link_map->l_name);
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
            return rocprofsys::path::realpath(_buffer);
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
#include <timemory/utility/filepath.hpp>

//======================================================================================//
//
//  Filters app_objects (internal constraints are applied in module level filtering)
//

std::vector<object_t*>
filter_objects(std::vector<object_t*>* app_objects)
{
    if(!app_objects || app_objects->empty()) return {};

    auto _wc = tim::component::wall_clock{};
    auto _pr = tim::component::peak_rss{};
    _wc.start();
    _pr.start();

    auto   _result         = std::vector<object_t*>{};
    size_t _excluded_count = 0;

    for(auto* obj : *app_objects)
    {
        if(!obj) continue;

        // Exclude every shared library when --exe-only is requested.
        if(exe_only && obj->isSharedLib())
        {
            verbprintf(0, "[filter] skipping shared lib '%s' (--exe-only)\n",
                       obj->name().c_str());
            continue;
        }

        // If function filtering is active, keep the object so the later
        // function-level filtering can make the decision
        if(!func_include.empty() || !func_restrict.empty())
        {
            _result.emplace_back(obj);
            continue;
        }

        bool _is_excluded     = false;
        bool _included_module = false;

        // -MI/-MR: if any module within this object matches a
        // module-include or module-restrict regex, keep the object so the later
        // module-level filtering can make the final per-module decision.
        if(!file_include.empty() || !file_restrict.empty())
        {
            auto _mods = std::vector<module_t*>{};
            obj->modules(_mods);  // Inexpensive
            for(auto* mod : _mods)
            {
                if(!mod) continue;
                auto _module_name = std::string{ get_name(mod) };
                if(regex_match_any(_module_name, file_include))
                {
                    _included_module = true;
                    verbprintf(2,
                               "[filter] forcing object '%s' "
                               "(module-include-regex matched '%s')\n",
                               obj->name().c_str(), _module_name.c_str());
                    break;
                }

                if(regex_match_any(_module_name, file_restrict))
                {
                    _included_module = true;
                    verbprintf(2,
                               "[filter] forcing object '%s' "
                               "(module-restrict-regex matched '%s')\n",
                               obj->name().c_str(), _module_name.c_str());
                    break;
                }
            }
        }

        // --max-library-functions: shared libs only; main executable is never gated
        if(!_included_module && max_library_functions > 0 && obj->isSharedLib())
        {
            auto _proc_count = get_object_procedure_count_lb(obj);
            if(_proc_count > max_library_functions)
            {
                _is_excluded = true;
                verbprintf(0,
                           "[filter] skipping shared lib '%s' "
                           "(%zu functions > --max-library-functions=%zu)\n",
                           obj->name().c_str(), _proc_count, max_library_functions);
            }
        }

        if(_is_excluded)
        {
            ++_excluded_count;
            continue;
        }

        _result.emplace_back(obj);
    }

    _wc.stop();
    _pr.stop();
    verbprintf(0,
               "Filtered objects: %zu of %zu included (%zu excluded) "
               "(%.3f %s, %.3f %s)\n",
               _result.size(), app_objects->size(), _excluded_count, _wc.get(),
               _wc.display_unit().c_str(), _pr.get(), _pr.display_unit().c_str());

    if(verbose_level >= 2)
    {
        verbprintf(2, "[filter] The following objects will be processed:\n");
        for(auto* obj : _result)
        {
            if(!obj) continue;
            verbprintf(2, "[filter] '%s'\n", obj->name().c_str());
        }
    }

    return _result;
}

//======================================================================================//
//
//  Filters app_modules by removing internal and user-excluded modules.
//

std::vector<module_t*>
filter_modules(std::vector<module_t*>* app_modules)
{
    if(!app_modules || app_modules->empty()) return {};

    auto _wc = tim::component::wall_clock{};
    auto _pr = tim::component::peak_rss{};
    _wc.start();
    _pr.start();

    // This does determine objects/procedures associated with a module, but it
    // internally uses Dyninst's Symtab API (faster)
    const auto& _internal_libs = get_internal_libs_data();

    auto   _result         = std::vector<module_t*>{};
    size_t _excluded_count = 0;

    for(auto* mod : *app_modules)
    {
        if(!mod) continue;

        auto _module_name = std::string{ get_name(mod) };
        auto _module_base = std::string{ tim::filepath::basename(_module_name) };
        auto _module_real = rocprofsys::path::realpath(_module_name);

        bool _is_excluded = false;

        if(_internal_libs.find(_module_name) != _internal_libs.end() ||
           _internal_libs.find(_module_real) != _internal_libs.end() ||
           _internal_libs.find(_module_base) != _internal_libs.end())
        {
            _is_excluded = true;
        }

        if(!_is_excluded)
        {
            for(const auto& [lib_path, sub_map] : _internal_libs)
            {
                auto _lib_base = std::string{ tim::filepath::basename(lib_path) };
                if(_module_base == _lib_base || _module_real == lib_path ||
                   sub_map.find(_module_base) != sub_map.end() ||
                   sub_map.find(_module_real) != sub_map.end() ||
                   sub_map.find(_module_name) != sub_map.end())
                {
                    _is_excluded = true;
                    break;
                }
            }
        }

        if(_is_excluded)
        {
            // Do not filter it out if internal function regex is active
            if(!func_internal_include.empty())
            {
                _result.emplace_back(mod);
                continue;
            }
            verbprintf(3, "[filter] Skipping internal module: '%s'\n",
                       _module_name.c_str());
            ++_excluded_count;
            continue;
        }

        // -ME: skip if module matches an exclude regex
        if(regex_match_any(_module_name, file_exclude))
        {
            _is_excluded = true;
            verbprintf(2, "[filter] skipping module-exclude-regex: '%s'\n",
                       _module_name.c_str());
        }

        // -MR: skip if restrict is specified and module does NOT match
        if(!_is_excluded && !file_restrict.empty() &&
           !regex_match_any(_module_name, file_restrict))
        {
            _is_excluded = true;
            verbprintf(2, "[filter] skipping module-restrict-regex: '%s'\n",
                       _module_name.c_str());
        }

        // -MI: if module matches an include regex, force it through
        if(_is_excluded && regex_match_any(_module_name, file_include))
        {
            _is_excluded = false;
            verbprintf(2, "[filter] forcing module-include-regex: '%s'\n",
                       _module_name.c_str());
        }

        if(_is_excluded)
        {
            ++_excluded_count;
            continue;
        }

        _result.emplace_back(mod);
    }

    _pr.stop();
    _wc.stop();
    verbprintf(0,
               "Filtered modules: %zu of %zu included (%zu excluded) "
               "(%.3f %s, %.3f %s)\n",
               _result.size(), app_modules->size(), _excluded_count, _wc.get(),
               _wc.display_unit().c_str(), _pr.get(), _pr.display_unit().c_str());

    if(verbose_level >= 2)
    {
        verbprintf(2, "[filter] The following modules will be processed:\n");
        for(auto* mod : _result)
        {
            if(!mod) continue;
            verbprintf(2, "[filter] '%s'\n", std::string{ get_name(mod) }.c_str());
        }
    }

    return _result;
}

//======================================================================================//
//
//  Fetches procedures/modules from the given modules/objects. Assumes modules have
//  already been filtered by the respective filter functions. Both functions always
//  return a valid (possibly empty) pointer, never nullptr; callers check ->empty().
//

std::unique_ptr<std::vector<module_t*>>
get_modules(std::vector<object_t*>* app_objects)
{
    auto modlist = std::make_unique<std::vector<module_t*>>();
    if(!app_objects || app_objects->empty()) return modlist;

    auto _wc = tim::component::wall_clock{};
    auto _pr = tim::component::peak_rss{};
    _wc.start();
    _pr.start();

    // BPatch_object exposes modules() as an out-param API, not a getter
    auto _scratch = std::vector<module_t*>{};
    for(auto* obj : *app_objects)
    {
        if(!obj) continue;
        _scratch.clear();
        obj->modules(_scratch);
        if(!_scratch.empty())
            modlist->insert(modlist->end(), _scratch.begin(), _scratch.end());
    }

    _pr.stop();
    _wc.stop();
    verbprintf(1,
               "Fetched modules from %zu objects: "
               "%zu modules found (%.3f %s, %.3f %s)\n",
               app_objects->size(), modlist->size(), _wc.get(),
               _wc.display_unit().c_str(), _pr.get(), _pr.display_unit().c_str());

    return modlist;
}

std::unique_ptr<std::vector<procedure_t*>>
get_procedures(std::vector<module_t*>* app_modules, bool include_uninstrumentable)
{
    auto proclist = std::make_unique<std::vector<procedure_t*>>();
    if(!app_modules || app_modules->empty())
    {
        verbprintf(0, "No modules found\n");
        return proclist;
    }

    auto _wc = tim::component::wall_clock{};
    auto _pr = tim::component::peak_rss{};
    _wc.start();
    _pr.start();

    for(auto* mod : *app_modules)
    {
        if(!mod) continue;
        std::unique_ptr<std::vector<procedure_t*>> procs{ mod->getProcedures(
            include_uninstrumentable) };
        if(procs && !procs->empty())
            proclist->insert(proclist->end(), procs->begin(), procs->end());
    }

    _pr.stop();
    _wc.stop();
    verbprintf(1,
               "Fetched procedures from %zu modules: "
               "%zu procedures found (%.3f %s, %.3f %s)\n",
               app_modules->size(), proclist->size(), _wc.get(),
               _wc.display_unit().c_str(), _pr.get(), _pr.display_unit().c_str());

    return proclist;
}

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
        auto        _real_name = rocprofsys::path::realpath(itr->fullName());

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
            symtab_data.typed_func_names[rocprofsys::utility::demangle(fitr->getName())] =
                fitr;

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
            return fmt::format("{}FatalError", tim::log::color::fatal());
        }
        case BPatchSerious:
        {
            return fmt::format("{}SeriousError", tim::log::color::fatal());
        }
        case BPatchWarning:
        {
            return fmt::format("{}Warning", tim::log::color::warning());
        }
        case BPatchInfo:
        {
            return fmt::format("{}Info", tim::log::color::info());
        }
        default:
        {
            return fmt::format("{}UnknownErrorLevel{}", tim::log::color::warning(),
                               static_cast<int>(_level));
        }
    }
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
