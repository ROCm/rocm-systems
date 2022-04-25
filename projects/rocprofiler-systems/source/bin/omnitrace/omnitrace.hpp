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

#pragma once

#include "function_signature.hpp"
#include "fwd.hpp"
#include "info.hpp"
#include "module_function.hpp"

//======================================================================================//

inline string_t
get_absolute_path(const char* fname)
{
    char  path_save[PATH_MAX];
    char  abs_exe_path[PATH_MAX];
    char* p = nullptr;

    if(!(p = strrchr((char*) fname, '/')))
    {
        auto* ret = getcwd(abs_exe_path, sizeof(abs_exe_path));
        consume_parameters(ret);
    }
    else
    {
        auto* rets = getcwd(path_save, sizeof(path_save));
        auto  retf = chdir(fname);
        auto* reta = getcwd(abs_exe_path, sizeof(abs_exe_path));
        auto  retp = chdir(path_save);
        consume_parameters(rets, retf, reta, retp);
    }
    return string_t(abs_exe_path);
}

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
struct omnitrace_call_expr
{
    using snippet_pointer_t = std::shared_ptr<snippet_t>;

    template <typename... Args>
    omnitrace_call_expr(Args&&... args)
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
struct omnitrace_snippet_vec
{
    using entry_type = std::vector<omnitrace_call_expr>;
    using value_type = std::vector<call_expr_pointer_t>;

    template <typename... Args>
    void generate(procedure_t* func, Args&&... args)
    {
        auto _expr = omnitrace_call_expr(std::forward<Args>(args)...);
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
static inline address_space_t*
omnitrace_get_address_space(patch_pointer_t& _bpatch, int _cmdc, char** _cmdv,
                            bool _rewrite, int _pid = -1, const string_t& _name = {})
{
    address_space_t* mutatee = nullptr;

    if(_rewrite)
    {
        verbprintf(1, "Opening '%s' for binary rewrite... ", _name.c_str());
        fflush(stderr);
        if(!_name.empty()) mutatee = _bpatch->openBinary(_name.c_str(), false);
        if(!mutatee)
        {
            fprintf(stderr, "[omnitrace][exe] Failed to open binary '%s'\n",
                    _name.c_str());
            throw std::runtime_error("Failed to open binary");
        }
        verbprintf_bare(1, "Done\n");
    }
    else if(_pid >= 0)
    {
        verbprintf(1, "Attaching to process %i... ", _pid);
        fflush(stderr);
        char* _cmdv0 = (_cmdc > 0) ? _cmdv[0] : nullptr;
        mutatee      = _bpatch->processAttach(_cmdv0, _pid);
        if(!mutatee)
        {
            fprintf(stderr, "[omnitrace][exe] Failed to connect to process %i\n",
                    (int) _pid);
            throw std::runtime_error("Failed to attach to process");
        }
        verbprintf_bare(1, "Done\n");
    }
    else
    {
        verbprintf(1, "Creating process '%s'... ", _cmdv[0]);
        fflush(stderr);
        mutatee = _bpatch->processCreate(_cmdv[0], (const char**) _cmdv, nullptr);
        if(!mutatee)
        {
            std::stringstream ss;
            for(int i = 0; i < _cmdc; ++i)
            {
                if(!_cmdv[i]) continue;
                ss << _cmdv[i] << " ";
            }
            fprintf(stderr, "[omnitrace][exe] Failed to create process: '%s'\n",
                    ss.str().c_str());
            throw std::runtime_error("Failed to create process");
        }
        verbprintf_bare(1, "Done\n");
    }

    return mutatee;
}
//
//======================================================================================//
//
TIMEMORY_NOINLINE inline void
omnitrace_thread_exit(thread_t* thread, BPatch_exitType exit_type)
{
    if(!thread) return;

    BPatch_process* app = thread->getProcess();

    if(!terminate_expr)
    {
        fprintf(stderr, "[omnitrace][exe] continuing execution\n");
        app->continueExecution();
        return;
    }

    switch(exit_type)
    {
        case ExitedNormally:
        {
            fprintf(stderr, "[omnitrace][exe] Thread exited normally\n");
            break;
        }
        case ExitedViaSignal:
        {
            fprintf(stderr, "[omnitrace][exe] Thread terminated unexpectedly\n");
            break;
        }
        case NoExit:
        default:
        {
            fprintf(stderr, "[omnitrace][exe] %s invoked with NoExit\n", __FUNCTION__);
            break;
        }
    }

    // terminate_expr = nullptr;
    thread->oneTimeCode(*terminate_expr);

    fprintf(stderr, "[omnitrace][exe] continuing execution\n");
    app->continueExecution();
}
//
//======================================================================================//
//
TIMEMORY_NOINLINE inline void
omnitrace_fork_callback(thread_t* parent, thread_t* child)
{
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
// insert_instr -- insert instrumentation into a function
//
template <typename Tp>
bool
insert_instr(address_space_t* mutatee, const bpvector_t<point_t*>& _points, Tp traceFunc,
             procedure_loc_t traceLoc, bool allow_traps)
{
    if(!traceFunc || _points.empty()) return false;

    auto _trace = traceFunc.get();
    auto _traps = std::set<point_t*>{};
    if(!allow_traps)
    {
        for(const auto& itr : _points)
        {
            if(itr && itr->usesTrap_NP()) _traps.insert(itr);
        }
    }

    size_t _n = 0;
    for(const auto& itr : _points)
    {
        if(!itr || _traps.count(itr) > 0)
            continue;
        else if(traceLoc == BPatch_entry)
            mutatee->insertSnippet(*_trace, *itr, BPatch_callBefore, BPatch_firstSnippet);
        else
            mutatee->insertSnippet(*_trace, *itr);
        ++_n;
    }

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
    module_t* module = funcToInstr->getModule();
    if(!module || !traceFunc) return false;

    bpvector_t<point_t*>* _points = nullptr;
    auto                  _trace  = traceFunc.get();

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

    std::set<point_t*> _traps{};
    if(!allow_traps)
    {
        for(auto& itr : *_points)
        {
            if(itr && itr->usesTrap_NP()) _traps.insert(itr);
        }
    }

    size_t _n = 0;
    for(auto& itr : *_points)
    {
        if(!itr || _traps.count(itr) > 0)
            continue;
        else if(traceLoc == BPatch_entry)
            mutatee->insertSnippet(*_trace, *itr, BPatch_callBefore, BPatch_firstSnippet);
        else
            mutatee->insertSnippet(*_trace, *itr);
        ++_n;
    }

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
    point_t* _point = nullptr;
    auto     _trace = traceFunc.get();

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

    if(_point == nullptr) return false;

    if(!allow_traps && _point->usesTrap_NP()) return false;

    switch(traceLoc)
    {
        case BPatch_entry:
            return (mutatee->insertSnippet(*_trace, *_point, BPatch_callBefore,
                                           BPatch_firstSnippet) != nullptr);
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
