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

#pragma once

#include "function_signature.hpp"
#include "fwd.hpp"
#include "info.hpp"
#include "log.hpp"
#include "module_function.hpp"

#include <timemory/utility/filepath.hpp>
#include <timemory/utility/join.hpp>

#include <dlfcn.h>
#include <ios>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

//======================================================================================//

bool
is_text_file(const std::string& filename);

//======================================================================================//

inline string_t
to_lower(string_t s)
{
    for(auto& itr : s)
        itr = tolower(itr);
    return s;
}
//
//======================================================================================//
//
template <typename Tp, std::enable_if_t<!std::is_same<Tp, std::string>::value, int> = 0>
snippet_pointer_t
get_snippet(Tp arg)
{
    return std::make_shared<snippet_t>(const_expr_t{ arg });
}
//
//======================================================================================//
//
template <typename Tp, std::enable_if_t<std::is_same<Tp, std::string>::value, int> = 0>
snippet_pointer_t
get_snippet(const Tp& arg)
{
    return std::make_shared<snippet_t>(const_expr_t{ arg.c_str() });
}
//
//======================================================================================//
//
template <typename... Args>
snippet_pointer_vec_t
get_snippets(Args&&... args)
{
    snippet_pointer_vec_t _tmp{};
    TIMEMORY_FOLD_EXPRESSION(_tmp.push_back(get_snippet(std::forward<Args>(args))));
    return _tmp;
}
//
//======================================================================================//
//
struct rocprofsys_call_expr
{
    using snippet_pointer_t = std::shared_ptr<snippet_t>;

    template <typename... Args>
    rocprofsys_call_expr(Args&&... args)
    : m_params(get_snippets(std::forward<Args>(args)...))
    {}

    snippet_vec_t get_params()
    {
        snippet_vec_t _ret;
        for(auto& itr : m_params)
            _ret.push_back(itr.get());
        return _ret;
    }

    inline call_expr_pointer_t get(procedure_t* func)
    {
        return call_expr_pointer_t((func) ? new call_expr_t(*func, get_params())
                                          : nullptr);
    }

private:
    snippet_pointer_vec_t m_params;
};
//
//======================================================================================//
//
struct rocprofsys_snippet_vec
{
    using entry_type = std::vector<rocprofsys_call_expr>;
    using value_type = std::vector<call_expr_pointer_t>;

    template <typename... Args>
    void generate(procedure_t* func, Args&&... args)
    {
        auto _expr = rocprofsys_call_expr(std::forward<Args>(args)...);
        auto _call = _expr.get(func);
        if(_call)
        {
            m_entries.push_back(_expr);
            m_data.push_back(_call);
            // m_data.push_back(entry_type{ _call, _expr });
        }
    }

    void append(snippet_vec_t& _obj)
    {
        for(auto& itr : m_data)
            _obj.push_back(itr.get());
    }

private:
    entry_type m_entries;
    value_type m_data;
};
//
//======================================================================================//
//
static inline bool
rocprofsys_get_is_executable(std::string_view _cmd, bool _default_v)
{
    bool _is_executable = _default_v;

    if(_cmd.empty())
    {
        if(!tim::filepath::exists(std::string{ _cmd }))
        {
            verbprintf(
                0,
                "Warning! '%s' was not found. Dyninst may fail to open the binary for "
                "instrumentation...\n",
                _cmd.data());
        }

        Dyninst::SymtabAPI::Symtab* _symtab = nullptr;
        if(Dyninst::SymtabAPI::Symtab::openFile(_symtab, _cmd.data()))
        {
            _is_executable = _symtab->isExecutable() && _symtab->isExec();
            Dyninst::SymtabAPI::Symtab::closeSymtab(_symtab);
        }
    }
    return _is_executable;
}
//
//======================================================================================//
//
static inline address_space_t*
rocprofsys_get_address_space(patch_pointer_t& _bpatch, int _cmdc, char** _cmdv,
                             const std::vector<std::string>& _cmdenv, bool _rewrite,
                             int _pid = -1, const std::string& _name = {})
{
    address_space_t* mutatee = nullptr;

    if(_rewrite)
    {
        if(is_text_file(_name))
        {
            errprintf(-127,
                      "'%s' is a text file. rocprof-sys only supports instrumenting "
                      "binary files",
                      _name.c_str());
        }

        verbprintf(1, "Opening '%s' for binary rewrite... ", _name.c_str());
        fflush(stderr);
        if(!_name.empty()) mutatee = _bpatch->openBinary(_name.c_str(), false);
        if(!mutatee)
        {
            verbprintf(-1, "Failed to open binary '%s'\n", _name.c_str());
            throw std::runtime_error("Failed to open binary");
        }
        verbprintf_bare(1, "Done\n");
    }
    else
    {
        bool _attach = (_pid >= 0);

        // override the current environment create/attach to process, revert environment
        using strpair_t    = std::pair<std::string, std::string>;
        auto _imported     = std::vector<strpair_t>{};
        auto _exported     = std::vector<strpair_t>{};
        auto _get_env_pair = [](const std::string& _full) {
            auto _pos = _full.find('=');
            if(_pos < _full.length())
                return std::make_pair(_full.substr(0, _pos), _full.substr(_pos + 1));
            return strpair_t{};
        };

        if(environ)
        {
            size_t _idx = 0;
            while(environ[_idx] != nullptr)
                _imported.emplace_back(_get_env_pair(environ[_idx++]));
        }

        for(const auto& itr : _cmdenv)
        {
            _exported.emplace_back(_get_env_pair(itr));
        }

        for(const auto& itr : _exported)
        {
            setenv(itr.first.c_str(), itr.second.c_str(), 1);
            verbprintf(4, "[env] %s=%s\n", itr.first.c_str(), itr.second.c_str());
        }

        if(_attach)
        {
            verbprintf(1, "Attaching to process %i... ", _pid);
            fflush(stderr);
            char* _cmdv0 = (_cmdc > 0) ? _cmdv[0] : nullptr;
            mutatee      = _bpatch->processAttach(_cmdv0, _pid);
            if(!mutatee)
            {
                verbprintf(-1, "Failed to connect to process %i\n", (int) _pid);
                throw std::runtime_error("Failed to attach to process");
            }
            verbprintf_bare(1, "Done\n");
        }
        else
        {
            if(_cmdc < 1) errprintf(-127, "No command provided");

            if(is_text_file(_cmdv[0]))
            {
                errprintf(-1,
                          "'%s' is a text file. rocprof-sys only supports instrumenting "
                          "binary files",
                          _cmdv[0]);
            }

            std::stringstream ss;
            for(int i = 0; i < _cmdc; ++i)
            {
                if(!_cmdv || !_cmdv[i]) continue;
                ss << " " << _cmdv[i];
            }
            auto _cmd_msg = ss.str();
            if(_cmd_msg.length() > 1) _cmd_msg = _cmd_msg.substr(1);

            verbprintf(1, "Creating process '%s'... ", _cmd_msg.c_str());
            fflush(stderr);
            mutatee = _bpatch->processCreate(_cmdv[0], (const char**) _cmdv, nullptr);
            if(!mutatee)
            {
                verbprintf(-1, "Failed to create process: '%s'\n", _cmd_msg.c_str());
                throw std::runtime_error("Failed to create process");
            }
            verbprintf_bare(1, "Done\n");
        }
    }

    return mutatee;
}
//
//======================================================================================//
//
TIMEMORY_NOINLINE inline void
rocprofsys_thread_exit(thread_t* thread, BPatch_exitType exit_type)
{
    if(!thread) return;

    ROCPROFSYS_ADD_LOG_ENTRY("Executing the thread callback");

    BPatch_process* app = thread->getProcess();

    if(!terminate_expr)
    {
        fprintf(stderr, "[rocprof-sys][exe] continuing execution\n");
        app->continueExecution();
        return;
    }

    switch(exit_type)
    {
        case ExitedNormally:
        {
            fprintf(stderr, "[rocprof-sys][exe] Thread exited normally\n");
            break;
        }
        case ExitedViaSignal:
        {
            fprintf(stderr, "[rocprof-sys][exe] Thread terminated unexpectedly\n");
            break;
        }
        case NoExit:
        default:
        {
            fprintf(stderr, "[rocprof-sys][exe] %s invoked with NoExit\n", __FUNCTION__);
            break;
        }
    }

    // terminate_expr = nullptr;
    thread->oneTimeCode(*terminate_expr);

    fprintf(stderr, "[rocprof-sys][exe] continuing execution\n");
    app->continueExecution();
}
//
//======================================================================================//
//
TIMEMORY_NOINLINE inline void
rocprofsys_fork_callback(thread_t* parent, thread_t* child)
{
    ROCPROFSYS_ADD_LOG_ENTRY("Executing the fork callback");

    if(child)
    {
        auto* app = child->getProcess();
        if(app)
        {
            verbprintf(4, "Stopping execution and detaching child fork...\n");
            app->stopExecution();
            app->detach(true);
            // app->terminateExecution();
            // app->continueExecution();
        }
    }

    if(parent)
    {
        auto* app = parent->getProcess();
        if(app)
        {
            verbprintf(4, "Continuing execution on parent after fork callback...\n");
            app->continueExecution();
        }
    }
}
//
//======================================================================================//
// path resolution helpers
//
std::string&
rocprofsys_get_exe_realpath();
//
std::optional<std::string>
rocprofsys_get_origin(const char*        _name,
                      std::vector<int>&& _open_modes = { (RTLD_LAZY | RTLD_NOLOAD) });
//
std::vector<std::string>
rocprofsys_get_link_map(const char* _lib, const std::string& _exclude_linked_by = {},
                        const std::string& _exclude_re = {},
                        std::vector<int>&& _open_modes = { (RTLD_LAZY | RTLD_NOLOAD) });
//
//======================================================================================//
// insert_instr -- insert instrumentation into a function
//
template <typename Tp>
bool
insert_instr(address_space_t* mutatee, const std::vector<point_t*>& _points, Tp traceFunc,
             procedure_loc_t, bool allow_traps)
{
    if(!traceFunc || _points.empty()) return false;

    auto _names = [&_points]() {
        std::set<std::string> _v{};
        for(const auto& itr : _points)
            if(itr && itr->getFunction()) _v.emplace(get_name(itr->getFunction()));
        return _v;
    }();

    ROCPROFSYS_ADD_LOG_ENTRY("Inserting", _points.size(),
                             "instrumentation points into function(s)", _names);

    auto _trace = traceFunc.get();
    auto _traps = std::set<point_t*>{};
    if(!allow_traps)
    {
        for(const auto& itr : _points)
        {
            if(itr && itr->usesTrap_NP()) _traps.insert(itr);
        }
    }

    ROCPROFSYS_ADD_LOG_ENTRY("Found", _traps.size(),
                             "instrumentation points using traps in function(s)", _names);

    size_t _n = 0;
    for(const auto& itr : _points)
    {
        if(!itr || _traps.count(itr) > 0) continue;
        mutatee->insertSnippet(*_trace, *itr);
        ++_n;
    }

    ROCPROFSYS_ADD_LOG_ENTRY("Inserted", _n, "instrumentation points in function(s)",
                             _names);

    return (_n > 0);
}
//
//======================================================================================//
// insert_instr -- insert instrumentation into loops
//
template <typename Tp>
bool
insert_instr(address_space_t* mutatee, procedure_t* funcToInstr, Tp traceFunc,
             procedure_loc_t traceLoc, flow_graph_t* cfGraph,
             basic_loop_t* loopToInstrument, bool allow_traps)
{
    if(!funcToInstr) return false;
    module_t* module = funcToInstr->getModule();
    if(!module || !traceFunc) return false;

    std::vector<point_t*>* _points = nullptr;
    auto                   _trace  = traceFunc.get();

    ROCPROFSYS_ADD_LOG_ENTRY("Searching for loop instrumentation points in function",
                             get_name(funcToInstr));

    if(!cfGraph) funcToInstr->getCFG();
    if(cfGraph && loopToInstrument)
    {
        if(traceLoc == BPatch_entry)
            _points = cfGraph->findLoopInstPoints(BPatch_locLoopEntry, loopToInstrument);
        else if(traceLoc == BPatch_exit)
            _points = cfGraph->findLoopInstPoints(BPatch_locLoopExit, loopToInstrument);
    }
    else
    {
        _points = funcToInstr->findPoint(traceLoc);
    }

    if(_points == nullptr) return false;
    if(_points->empty()) return false;

    ROCPROFSYS_ADD_LOG_ENTRY("Inserting max of", _points->size(),
                             "loop instrumentation points in function",
                             get_name(funcToInstr));

    std::set<point_t*> _traps{};
    if(!allow_traps)
    {
        for(auto& itr : *_points)
        {
            if(itr && itr->usesTrap_NP()) _traps.insert(itr);
        }
    }

    ROCPROFSYS_ADD_LOG_ENTRY("Found", _traps.size(),
                             "loop instrumentation points using traps in function",
                             get_name(funcToInstr));

    size_t _n = 0;
    for(auto& itr : *_points)
    {
        if(!itr || _traps.count(itr) > 0) continue;
        mutatee->insertSnippet(*_trace, *itr);
        ++_n;
    }

    ROCPROFSYS_ADD_LOG_ENTRY("Inserted", _n, "loop instrumentation points in function",
                             get_name(funcToInstr));

    return (_n > 0);
}
//
//======================================================================================//
// insert_instr -- insert instrumentation into basic blocks
//
template <typename Tp>
bool
insert_instr(address_space_t* mutatee, Tp traceFunc, procedure_loc_t traceLoc,
             basic_block_t* basicBlock, bool allow_traps)
{
    if(!basicBlock) return false;

    point_t* _point = nullptr;
    auto     _trace = traceFunc.get();

    ROCPROFSYS_ADD_LOG_ENTRY(
        "Searching for basic-block entry and exit instrumentation points ::",
        *basicBlock);

    basic_block_t* _bb = basicBlock;
    switch(traceLoc)
    {
        case BPatch_entry: _point = _bb->findEntryPoint(); break;
        case BPatch_exit: _point = _bb->findExitPoint(); break;
        default:
            verbprintf(0, "Warning! trace location type %i not supported\n",
                       (int) traceLoc);
            return false;
    }

    if(_point == nullptr)
    {
        ROCPROFSYS_ADD_LOG_ENTRY("No instrumentation points were found in basic-block ",
                                 *basicBlock);
        return false;
    }

    if(!allow_traps && _point->usesTrap_NP())
    {
        ROCPROFSYS_ADD_LOG_ENTRY("Basic-block", *basicBlock,
                                 "uses traps and traps are disallowed");
        return false;
    }

    switch(traceLoc)
    {
        case BPatch_entry:
        case BPatch_exit: return (mutatee->insertSnippet(*_trace, *_point) != nullptr);
        default:
        {
            verbprintf(0, "Warning! trace location type %i not supported\n",
                       (int) traceLoc);
            return false;
        }
    }

    return false;
}
