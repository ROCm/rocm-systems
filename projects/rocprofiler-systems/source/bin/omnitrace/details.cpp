// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "omnitrace.hpp"

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
        "dlvsym", "dlsym", "getenv", "setenv", "unsetenv", "printf", "fprintf", "fflush",
        "malloc", "malloc_stats", "malloc_trim", "mallopt", "calloc", "free", "pvalloc",
        "valloc", "mmap", "munmap", "fopen", "fclose", "fmemopen", "fmemclose",
        "backtrace", "backtrace_symbols", "backtrace_symbols_fd", "sigaddset",
        "sigandset", "sigdelset", "sigemptyset", "sigfillset", "sighold", "sigisemptyset",
        "sigismember", "sigorset", "sigrelse", "sigvec", "strtok", "strstr", "sbrk",
        "strxfrm", "atexit", "ompt_start_tool", "nanosleep", "cfree", "tolower",
        "toupper", "fileno", "fileno_unlocked", "exit", "quick_exit", "abort", "mbind",
        "migrate_pages", "move_pages", "numa_migrate_pages", "numa_move_pages",
        "numa_alloc", "numa_alloc_local", "numa_alloc_interleaved", "numa_alloc_onnode",
        "numa_realloc", "numa_free",
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
        "ncclAllToAllv"
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
//  Error callback routine.
//
void
errorFunc(error_level_t level, int num, const char** params)
{
    char line[256];

    const char* msg = bpatch->getEnglishErrorString(num);
    bpatch->formatErrorString(line, sizeof(line), msg, params);

    if(num != expect_error)
    {
        printf("Error #%d (level %d): %s\n", num, level, line);
        // We consider some errors fatal.
        if(num == 101) exit(-1);
    }
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
        bpvector_t<procedure_t*> _found;
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
void
error_func_real(error_level_t level, int num, const char* const* params)
{
    if(num == 0)
    {
        // conditional reporting of warnings and informational messages
        if(error_print > 0)
        {
            if(level == BPatchInfo)
            {
                if(error_print > 1) printf("%s\n", params[0]);
            }
            else
                printf("%s", params[0]);
        }
    }
    else
    {
        // reporting of actual errors
        char        line[256];
        const char* msg = bpatch->getEnglishErrorString(num);
        bpatch->formatErrorString(line, sizeof(line), msg, params);
        if(num != expect_error)
        {
            printf("Error #%d (level %d): %s\n", num, level, line);
            // We consider some errors fatal.
            if(num == 101) exit(-1);
        }
    }
}

//======================================================================================//
//
//  We've a null error function when we don't want to display an error
//
void
error_func_fake(error_level_t level, int num, const char* const* params)
{
    consume_parameters(level, num, params);
    // It does nothing.
}
