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

#include "libpyrocprofsys.hpp"
#include "dl/dl.hpp"
#include "library/coverage.hpp"
#include "library/coverage/impl.hpp"
#include "rocprofiler-systems/categories.h"
#include "rocprofiler-systems/user.h"

#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/environment.hpp>
#include <timemory/mpl/apply.hpp>
#include <timemory/mpl/policy.hpp>
#include <timemory/operations/types/file_output_message.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/variadic/macros.hpp>

#include <pybind11/detail/common.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pyerrors.h>

#include <cctype>
#include <cstdint>
#include <exception>
#include <locale>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_map>

#define ROCPROFSYS_PYTHON_VERSION                                                        \
    ((10000 * PY_MAJOR_VERSION) + (100 * PY_MINOR_VERSION) + PY_MICRO_VERSION)

namespace pyrocprofsys
{
namespace pyprofile
{
py::module
generate(py::module& _pymod);
}
namespace pycoverage
{
py::module
generate(py::module& _pymod);
}
namespace pyuser
{
py::module
generate(py::module& _pymod);
}
}  // namespace pyrocprofsys

template <typename... Tp>
using uomap_t = std::unordered_map<Tp...>;

PYBIND11_MODULE(libpyrocprofsys, omni)
{
    using namespace pyrocprofsys;

    py::doc("ROCm Systems Profiler Python bindings for profiling, user API, and "
            "code coverage post-processing");

    static bool _is_initialized = false;
    static bool _is_finalized   = false;
    static auto _get_use_mpi    = []() {
        bool _use_mpi = false;
        try
        {
            py::module::import("mpi4py");
            _use_mpi = true;
        } catch(py::error_already_set& _exc)
        {
            if(!_exc.matches(PyExc_ImportError)) throw;
        }
        return _use_mpi;
    };

    omni.def("is_initialized", []() { return _is_initialized; }, "Initialization state");

    omni.def("is_finalized", []() { return _is_finalized; }, "Finalization state");

    omni.def(
        "initialize",
        [](const std::string& _v) {
            if(_is_initialized)
                throw std::runtime_error("Error! rocprofsys is already initialized");
            _is_initialized = true;
            rocprofsys_set_mpi(_get_use_mpi(), false);
            rocprofsys_init("trace", false, _v.c_str());
        },
        "Initialize rocprofsys");

    omni.def(
        "initialize",
        [](const py::list& _v) {
            if(_is_initialized)
                throw std::runtime_error("Error! rocprofsys is already initialized");
            _is_initialized = true;
            rocprofsys_set_instrumented(
                static_cast<int>(rocprofsys::dl::InstrumentMode::PythonProfile));
            rocprofsys_set_mpi(_get_use_mpi(), false);
            std::string _cmd      = {};
            std::string _cmd_line = {};
            for(auto&& itr : _v)
            {
                if(_cmd.empty()) _cmd = itr.cast<std::string>();
                _cmd_line += " " + itr.cast<std::string>();
            }
            if(!_cmd_line.empty())
            {
                _cmd_line = _cmd_line.substr(_cmd_line.find_first_not_of(' '));
                tim::set_env("ROCPROFSYS_COMMAND_LINE", _cmd_line, 0);
            }
            rocprofsys_init("trace", false, _cmd.c_str());
        },
        "Initialize rocprofsys");

    omni.def(
        "finalize",
        []() {
            if(_is_finalized)
                throw std::runtime_error("Error! rocprofsys is already finalized");
            _is_finalized = true;
            rocprofsys_finalize();
        },
        "Finalize rocprofsys");

    pyprofile::generate(omni);
    pycoverage::generate(omni);
    pyuser::generate(omni);

    auto _python_path = tim::get_env("ROCPROFSYS_PATH", std::string{}, false);
    auto _libpath     = std::string{ "librocprof-sys-dl.so" };
    if(!_python_path.empty()) _libpath = TIMEMORY_JOIN("/", _python_path, _libpath);
    // permit env override if default path fails/is wrong
    _libpath = tim::get_env("ROCPROFSYS_DL_LIBRARY", _libpath);
    // this is necessary when building with -static-libstdc++
    // without it, loading librocprof-sys.so within librocprof-sys-dl.so segfaults
    if(!dlopen(_libpath.c_str(), RTLD_NOW | RTLD_GLOBAL))
    {
        auto _msg =
            TIMEMORY_JOIN("", "dlopen(\"", _libpath, "\", RTLD_NOW | RTLD_GLOBAL)");
        perror(_msg.c_str());
        fprintf(stderr, "[rocprofsys][dl][pid=%i] %s :: %s\n", getpid(), _msg.c_str(),
                dlerror());
    }
}

//======================================================================================//
//
namespace pyrocprofsys
{
namespace pyprofile
{
//
using profiler_t           = std::function<void()>;
using profiler_vec_t       = std::vector<profiler_t>;
using profiler_label_map_t = std::unordered_map<std::string, profiler_vec_t>;
using profiler_index_map_t = std::unordered_map<uint32_t, profiler_label_map_t>;
using strset_t             = std::unordered_set<std::string>;
using note_t               = rocprofsys_annotation_t;
using annotations_t        = std::array<note_t, 6>;
//
namespace
{
strset_t default_exclude_functions = { "^<.*>$" };
strset_t default_exclude_filenames = { "(encoder|decoder|threading).py$", "^<.*>$" };
}  // namespace
//
auto&
get_paused()
{
    static thread_local int64_t _v = 0;
    return _v;
}
//
struct config
{
    bool                    is_running         = false;
    bool                    trace_c            = false;
    bool                    include_internal   = false;
    bool                    include_args       = false;
    bool                    include_line       = false;
    bool                    include_filename   = false;
    bool                    full_filepath      = false;
    bool                    annotate_trace     = false;
    int32_t                 ignore_stack_depth = 0;
    int32_t                 base_stack_depth   = -1;
    int32_t                 verbose            = 0;
    int64_t                 depth_tracker      = 0;
    std::string             base_module_path   = {};
    strset_t                restrict_functions = {};
    strset_t                restrict_filenames = {};
    strset_t                include_functions  = {};
    strset_t                include_filenames  = {};
    strset_t                exclude_functions  = default_exclude_functions;
    strset_t                exclude_filenames  = default_exclude_filenames;
    std::vector<profiler_t> records            = {};
    annotations_t           annotations = { note_t{ "file", ROCPROFSYS_STRING, nullptr },
                                            note_t{ "line", ROCPROFSYS_INT32, nullptr },
                                            note_t{ "lasti", ROCPROFSYS_INT32, nullptr },
                                            note_t{ "argcount", ROCPROFSYS_INT32, nullptr },
                                            note_t{ "nlocals", ROCPROFSYS_INT32, nullptr },
                                            note_t{ "stacksize", ROCPROFSYS_INT32, nullptr } };
};
//
inline config&
get_config()
{
    static auto*              _instance    = new config{};
    static thread_local auto* _tl_instance = []() {
        static std::atomic<uint32_t> _count{ 0 };
        auto                         _cnt = _count++;
        if(_cnt == 0) return _instance;

        auto* _tmp               = new config{};
        _tmp->is_running         = _instance->is_running;
        _tmp->trace_c            = _instance->trace_c;
        _tmp->include_internal   = _instance->include_internal;
        _tmp->include_args       = _instance->include_args;
        _tmp->include_line       = _instance->include_line;
        _tmp->include_filename   = _instance->include_filename;
        _tmp->full_filepath      = _instance->full_filepath;
        _tmp->annotate_trace     = _instance->annotate_trace;
        _tmp->base_module_path   = _instance->base_module_path;
        _tmp->restrict_functions = _instance->restrict_functions;
        _tmp->restrict_filenames = _instance->restrict_filenames;
        _tmp->include_functions  = _instance->include_functions;
        _tmp->include_filenames  = _instance->include_filenames;
        _tmp->exclude_functions  = _instance->exclude_functions;
        _tmp->exclude_filenames  = _instance->exclude_filenames;
        _tmp->verbose            = _instance->verbose;
        _tmp->annotations        = _instance->annotations;
        // if full filepath is specified, include filename is implied
        if(_tmp->full_filepath && !_tmp->include_filename) _tmp->include_filename = true;
        return _tmp;
    }();
    return *_tl_instance;
}
//
int
get_frame_lineno(PyFrameObject* frame)
{
#if ROCPROFSYS_PYTHON_VERSION >= 31100
    return PyFrame_GetLineNumber(frame);
#else
    return frame->f_lineno;
#endif
}
//
int
get_frame_lasti(PyFrameObject* frame)
{
#if ROCPROFSYS_PYTHON_VERSION >= 31100
    return PyFrame_GetLasti(frame);
#else
    return frame->f_lasti;
#endif
}
//
auto
get_frame_code(PyFrameObject* frame)
{
#if ROCPROFSYS_PYTHON_VERSION >= 31100
    return PyFrame_GetCode(frame);
#else
    return frame->f_code;
#endif
}
//
void
profiler_function(py::object pframe, const char* swhat, py::object arg)
{
    if(get_paused() > 0) return;

    static thread_local auto& _config  = get_config();
    static thread_local auto  _disable = false;

    if(_disable) return;

    _disable = true;
    tim::scope::destructor _dtor{ []() { _disable = false; } };
    (void) _dtor;

    if(pframe.is_none() || pframe.ptr() == nullptr) return;

    static auto _rocprofsys_path = _config.base_module_path;

    auto* frame = reinterpret_cast<PyFrameObject*>(pframe.ptr());

    int what = (strcmp(swhat, "call") == 0)       ? PyTrace_CALL
               : (strcmp(swhat, "c_call") == 0)   ? PyTrace_C_CALL
               : (strcmp(swhat, "return") == 0)   ? PyTrace_RETURN
               : (strcmp(swhat, "c_return") == 0) ? PyTrace_C_RETURN
                                                  : -1;
    // only support PyTrace_{CALL,C_CALL,RETURN,C_RETURN}
    if(what < 0)
    {
        if(_config.verbose > 2)
            TIMEMORY_PRINT_HERE("%s :: %s",
                                "Ignoring what != {CALL,C_CALL,RETURN,C_RETURN}", swhat);
        return;
    }

    auto _update_ignore_stack_depth = [what]() {
        switch(what)
        {
            case PyTrace_CALL: ++_config.ignore_stack_depth; break;
            case PyTrace_RETURN: --_config.ignore_stack_depth; break;
            default: break;
        }
    };

    if(_config.ignore_stack_depth > 0)
    {
        if(_config.verbose > 2)
            TIMEMORY_PRINT_HERE("%s :: %s :: %u", "Ignoring call/return", swhat,
                                _config.ignore_stack_depth);
        _update_ignore_stack_depth();
        return;
    }
    else if(_config.ignore_stack_depth < 0)
    {
        TIMEMORY_PRINT_HERE("WARNING! ignore_stack_depth is < 0 :: ",
                            _config.ignore_stack_depth);
    }

    // if PyTrace_C_{CALL,RETURN} is not enabled
    if(!_config.trace_c && (what == PyTrace_C_CALL || what == PyTrace_C_RETURN))
    {
        if(_config.verbose > 2)
            TIMEMORY_PRINT_HERE("%s :: %s", "Ignoring C call/return", swhat);
        return;
    }

    // get the arguments
    auto _get_args = [&]() {
        auto inspect = py::module::import("inspect");
        try
        {
            return py::cast<std::string>(
                inspect.attr("formatargvalues")(*inspect.attr("getargvalues")(pframe)));
        } catch(py::error_already_set& _exc)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(_config.verbose > 1, "Error! %s",
                                            _exc.what());
            if(!_exc.matches(PyExc_AttributeError)) throw;
        }
        return std::string{};
    };

    // get the final label
    auto _get_label = [&](auto& _funcname, auto& _filename, auto& _fullpath) {
        auto _bracket = _config.include_filename;
        if(_bracket) _funcname.insert(0, "[");
        // append the arguments
        if(_config.include_args) _funcname.append(_get_args());
        if(_bracket) _funcname.append("]");
        // append the filename
        if(_config.include_filename)
        {
            if(_config.full_filepath)
                _funcname.append(TIMEMORY_JOIN("", '[', std::move(_fullpath)));
            else
                _funcname.append(TIMEMORY_JOIN("", '[', std::move(_filename)));
        }
        // append the line number
        if(_config.include_line && _config.include_filename)
            _funcname.append(TIMEMORY_JOIN("", ':', get_frame_lineno(frame), ']'));
        else if(_config.include_line)
            _funcname.append(TIMEMORY_JOIN("", ':', get_frame_lineno(frame)));
        else if(_config.include_filename)
            _funcname += "]";
        return _funcname;
    };

    auto _find_matching = [](const strset_t& _expr, const std::string& _name) {
        const auto _rconstants =
            std::regex_constants::egrep | std::regex_constants::optimize;
        for(const auto& itr : _expr)  // NOLINT
        {
            if(std::regex_search(_name, std::regex(itr, _rconstants))) return true;
        }
        return false;
    };

    bool  _force      = false;
    auto& _only_funcs = _config.restrict_functions;
    auto& _incl_funcs = _config.include_functions;
    auto& _skip_funcs = _config.exclude_functions;
    auto  _func       = py::cast<std::string>(get_frame_code(frame)->co_name);

    if(!_only_funcs.empty())
    {
        _force = _find_matching(_only_funcs, _func);
        if(!_force)
        {
            if(_config.verbose > 2)
                TIMEMORY_PRINT_HERE("Skipping non-restricted function: %s",
                                    _func.c_str());
            return;
        }
    }

    if(!_force)
    {
        if(_find_matching(_incl_funcs, _func))
        {
            _force = true;
        }
        else if(_find_matching(_skip_funcs, _func))
        {
            if(_config.verbose > 1)
                TIMEMORY_PRINT_HERE("Skipping designated function: '%s'", _func.c_str());
            if(!_find_matching(default_exclude_functions, _func))
                _update_ignore_stack_depth();
            return;
        }
    }

    auto& _only_files = _config.restrict_filenames;
    auto& _incl_files = _config.include_filenames;
    auto& _skip_files = _config.exclude_filenames;
    auto  _full       = py::cast<std::string>(get_frame_code(frame)->co_filename);
    auto  _file       = (_full.find('/') != std::string::npos)
                            ? _full.substr(_full.find_last_of('/') + 1)
                            : _full;

    if(!_config.include_internal &&
       strncmp(_full.c_str(), _rocprofsys_path.c_str(), _rocprofsys_path.length()) == 0)
    {
        if(_config.verbose > 2)
            TIMEMORY_PRINT_HERE("Skipping internal function: %s", _func.c_str());
        return;
    }

    if(!_force && !_only_files.empty())
    {
        _force = _find_matching(_only_files, _full);
        if(!_force)
        {
            if(_config.verbose > 2)
                TIMEMORY_PRINT_HERE("Skipping non-restricted file: %s", _full.c_str());
            return;
        }
    }

    if(!_force)
    {
        if(_find_matching(_incl_files, _full))
        {
            _force = true;
        }
        else if(_find_matching(_skip_files, _full))
        {
            if(_config.verbose > 2)
                TIMEMORY_PRINT_HERE("Skipping non-included file: %s", _full.c_str());
            return;
        }
    }

    TIMEMORY_CONDITIONAL_PRINT_HERE(_config.verbose > 3, "%8s | %s%s | %s | %s", swhat,
                                    _func.c_str(), _get_args().c_str(), _file.c_str(),
                                    _full.c_str());

    auto _label = _get_label(_func, _file, _full);
    if(_label.empty()) return;

    static thread_local strset_t _labels{};
    const auto&                  _label_ref = *_labels.emplace(_label).first;
    auto                         _annotate  = _config.annotate_trace;

    // start function
    auto _profiler_call = [&]() {
        int _lineno = 0;
        int _lasti  = 0;
        if(_annotate)
        {
            _lineno                         = get_frame_lineno(frame);
            _lasti                          = get_frame_lasti(frame);
            _config.annotations.at(0).value = const_cast<char*>(_full.c_str());
            _config.annotations.at(1).value = &_lineno;
            _config.annotations.at(2).value = &_lasti;
            _config.annotations.at(3).value = &get_frame_code(frame)->co_argcount;
            _config.annotations.at(4).value = &get_frame_code(frame)->co_nlocals;
            _config.annotations.at(5).value = &get_frame_code(frame)->co_stacksize;
        }

        _config.records.emplace_back([&_label_ref, _annotate]() {
            rocprofsys_pop_category_region(ROCPROFSYS_CATEGORY_PYTHON, _label_ref.c_str(),
                                           (_annotate) ? _config.annotations.data()
                                                       : nullptr,
                                           _config.annotations.size());
        });
        rocprofsys_push_category_region(ROCPROFSYS_CATEGORY_PYTHON, _label_ref.c_str(),
                                        (_annotate) ? _config.annotations.data()
                                                    : nullptr,
                                        _config.annotations.size());
    };

    // stop function
    auto _profiler_return = [&]() {
        if(!_config.records.empty())
        {
            _config.records.back()();
            _config.records.pop_back();
        }
    };

    // process what
    switch(what)
    {
        case PyTrace_CALL:
        case PyTrace_C_CALL: _profiler_call(); break;
        case PyTrace_RETURN:
        case PyTrace_C_RETURN: _profiler_return(); break;
        default: break;
    }

    // don't do anything with arg
    tim::consume_parameters(arg);
}
//
py::module
generate(py::module& _pymod)
{
    py::module _prof = _pymod.def_submodule("profiler", "Profiling functions");

    auto _init = []() {
        try
        {
            auto _file =
                py::module::import("rocprofsys").attr("__file__").cast<std::string>();
            if(_file.find('/') != std::string::npos)
                _file = _file.substr(0, _file.find_last_of('/'));
            get_config().base_module_path = _file;
        } catch(py::cast_error& e)
        {
            std::cerr << "[profiler_init]> " << e.what() << std::endl;
        }
        if(get_config().is_running) return;
        get_config().records.clear();
        get_config().base_stack_depth = -1;
        get_config().is_running       = true;
    };

    auto _fini = []() {
        if(!get_config().is_running) return;
        get_config().is_running       = false;
        get_config().base_stack_depth = -1;
        get_config().records.clear();
    };

    auto _sys        = py::module::import("sys");
    auto _setprofile = _sys.attr("setprofile");

    _prof.def("profiler_function", &profiler_function, "Profiling function");
    _prof.def("profiler_init", _init, "Initialize the profiler");
    _prof.def("profiler_finalize", _fini, "Finalize the profiler");
    _prof.def(
        "profiler_pause",
        [_setprofile]() {
            if(++get_paused() == 1) _setprofile(nullptr);
        },
        "Pause the profiler");
    _prof.def(
        "profiler_resume",
        [_setprofile]() {
            if(--get_paused() == 0) _setprofile(py::cpp_function{ profiler_function });
        },
        "Resume the profiler");

    py::class_<config> _pyconfig(_prof, "config", "Profiler configuration");

#define CONFIGURATION_PROPERTY(NAME, TYPE, DOC, ...)                                     \
    _pyconfig.def_property_static(                                                       \
        NAME, [](py::object&&) { return __VA_ARGS__; },                                  \
        [](py::object&&, TYPE val) { __VA_ARGS__ = val; }, DOC);

    CONFIGURATION_PROPERTY("_is_running", bool, "Profiler is currently running",
                           get_config().is_running)
    CONFIGURATION_PROPERTY("trace_c", bool, "Enable tracing C functions",
                           get_config().trace_c)
    CONFIGURATION_PROPERTY("include_internal", bool, "Include functions within timemory",
                           get_config().include_internal)
    CONFIGURATION_PROPERTY("include_args", bool, "Encode the function arguments",
                           get_config().include_args)
    CONFIGURATION_PROPERTY("include_line", bool, "Encode the function line number",
                           get_config().include_line)
    CONFIGURATION_PROPERTY("include_filename", bool,
                           "Encode the function filename (see also: full_filepath)",
                           get_config().include_filename)
    CONFIGURATION_PROPERTY("full_filepath", bool,
                           "Display the full filepath (instead of file basename)",
                           get_config().full_filepath)
    CONFIGURATION_PROPERTY(
        "annotate_trace", bool,
        "Add detailed annotations to the trace about the executing function",
        get_config().annotate_trace)
    CONFIGURATION_PROPERTY("verbosity", int32_t, "Verbosity of the logging",
                           get_config().verbose)

    static auto _get_strset = [](const strset_t& _targ) {
        auto _out = py::list{};
        for(auto itr : _targ)
            _out.append(itr);
        return _out;
    };

    static auto _set_strset = [](const py::list& _inp, strset_t& _targ) {
        for(const auto& itr : _inp)
            _targ.insert(itr.cast<std::string>());
    };

#define CONFIGURATION_PROPERTY_LAMBDA(NAME, DOC, GET, SET)                               \
    _pyconfig.def_property_static(NAME, GET, SET, DOC);
#define CONFIGURATION_STRSET(NAME, DOC, ...)                                             \
    {                                                                                    \
        auto GET = [](py::object&&) { return _get_strset(__VA_ARGS__); };                \
        auto SET = [](py::object&&, const py::list& val) {                               \
            _set_strset(val, __VA_ARGS__);                                               \
        };                                                                               \
        CONFIGURATION_PROPERTY_LAMBDA(NAME, DOC, GET, SET)                               \
    }

    CONFIGURATION_STRSET("restrict_functions", "Function regexes to collect exclusively",
                         get_config().restrict_functions)
    CONFIGURATION_STRSET("restrict_modules", "Filename regexes to collect exclusively",
                         get_config().restrict_filenames)
    CONFIGURATION_STRSET("include_functions",
                         "Function regexes to always include in collection",
                         get_config().include_functions)
    CONFIGURATION_STRSET("include_modules",
                         "Filename regexes to always include in collection",
                         get_config().include_filenames)
    CONFIGURATION_STRSET("exclude_functions",
                         "Function regexes to filter out of collection",
                         get_config().exclude_functions)
    CONFIGURATION_STRSET("exclude_modules",
                         "Filename regexes to filter out of collection",
                         get_config().exclude_filenames)

    return _prof;
}
}  // namespace pyprofile

namespace pycoverage
{
py::module
generate(py::module& _pymod)
{
    namespace coverage           = rocprofsys::coverage;
    using coverage_data_vector_t = std::vector<coverage::coverage_data>;

    py::module _pycov = _pymod.def_submodule("coverage", "Code coverage");

#define DEFINE_PROPERTY(PYCLASS, CLASS, TYPE, NAME, ...)                                 \
    PYCLASS.def_property(                                                                \
        #NAME, [](CLASS* _object) { return _object->NAME; },                             \
        [](CLASS* _object, TYPE v) { return _object->NAME = v; }, __VA_ARGS__)

    py::class_<coverage::code_coverage> _pycov_summary{ _pymod, "summary",
                                                        "Code coverage summary" };

    _pycov_summary.def(py::init([]() { return new coverage::code_coverage{}; }),
                       "Create a default instance");

    _pycov_summary.def(
        "get_code_coverage",
        [](coverage::code_coverage* _v) {
            return _v->get(coverage::code_coverage::STANDARD);
        },
        "Get coverage fraction");
    _pycov_summary.def(
        "get_module_coverage",
        [](coverage::code_coverage* _v) {
            return _v->get(coverage::code_coverage::MODULE);
        },
        "Get coverage fraction");
    _pycov_summary.def(
        "get_function_coverage",
        [](coverage::code_coverage* _v) {
            return _v->get(coverage::code_coverage::FUNCTION);
        },
        "Get coverage fraction");
    _pycov_summary.def("get_uncovered_modules",
                       &coverage::code_coverage::get_uncovered_modules,
                       "List of uncovered modules");
    _pycov_summary.def("get_uncovered_functions",
                       &coverage::code_coverage::get_uncovered_functions,
                       "List of uncovered functions");

    DEFINE_PROPERTY(_pycov_summary, coverage::code_coverage, size_t, count,
                    "Number of times covered");
    DEFINE_PROPERTY(_pycov_summary, coverage::code_coverage, size_t, size,
                    "Total number of coverage entries");
    DEFINE_PROPERTY(_pycov_summary, coverage::code_coverage,
                    coverage::code_coverage::data, covered, "Covered information");
    DEFINE_PROPERTY(_pycov_summary, coverage::code_coverage,
                    coverage::code_coverage::data, possible, "Possible information");

    py::class_<coverage::code_coverage::data> _pycov_summary_data{ _pycov_summary, "data",
                                                                   "Code coverage data" };

    DEFINE_PROPERTY(_pycov_summary_data, coverage::code_coverage::data, std::set<size_t>,
                    addresses, "Addresses");
    DEFINE_PROPERTY(_pycov_summary_data, coverage::code_coverage::data,
                    std::set<std::string>, modules, "Modules");
    DEFINE_PROPERTY(_pycov_summary_data, coverage::code_coverage::data,
                    std::set<std::string>, functions, "Functions");

    py::class_<coverage::coverage_data> _pycov_details{ _pycov, "details",
                                                        "Code coverage data" };

    _pycov_details.def(py::init([]() { return new coverage::coverage_data{}; }),
                       "Create a default instance");

    DEFINE_PROPERTY(_pycov_details, coverage::coverage_data, size_t, count,
                    "Number of times invoked");
    DEFINE_PROPERTY(_pycov_details, coverage::coverage_data, size_t, address,
                    "Address of coverage entity (in binary)");
    DEFINE_PROPERTY(_pycov_details, coverage::coverage_data, size_t, line,
                    "Line number of coverage entity");
    DEFINE_PROPERTY(_pycov_details, coverage::coverage_data, std::string, module,
                    "Name of the containing module");
    DEFINE_PROPERTY(_pycov_details, coverage::coverage_data, std::string, function,
                    "Name of the function (basic)");
    DEFINE_PROPERTY(_pycov_details, coverage::coverage_data, std::string, source,
                    "Full signature of the function");

    _pycov_details.def(py::self + py::self);
    _pycov_details.def(py::self += py::self);
    _pycov_details.def(py::self == py::self);
    _pycov_details.def(py::self != py::self);
    _pycov_details.def(py::self < py::self);
    _pycov_details.def(py::self > py::self);
    _pycov_details.def(py::self <= py::self);
    _pycov_details.def(py::self >= py::self);

    auto _load_coverage = [](const std::string& _inp) {
        coverage::code_coverage* _summary = nullptr;
        coverage_data_vector_t*  _details = nullptr;
        std::ifstream            ifs{ _inp };
        if(ifs)
        {
            namespace cereal = tim::cereal;
            auto ar = tim::policy::input_archive<cereal::JSONInputArchive>::get(ifs);

            try
            {
                ar->setNextName("rocprofsys");
                ar->startNode();
                ar->setNextName("coverage");
                ar->startNode();
                _summary = new coverage::code_coverage{};
                _details = new coverage_data_vector_t{};
                (*ar)(cereal::make_nvp("summary", *_summary));
                (*ar)(cereal::make_nvp("details", *_details));
                ar->finishNode();
                ar->finishNode();
            } catch(std::exception&)
            {}
        }
        return std::make_tuple(_summary, _details);
    };

    auto _save_coverage = [](coverage::code_coverage* _summary,
                             coverage_data_vector_t* _details, std::string _name) {
        std::stringstream oss{};
        {
            namespace cereal = tim::cereal;
            auto ar =
                tim::policy::output_archive<cereal::PrettyJSONOutputArchive>::get(oss);

            ar->setNextName("rocprofsys");
            ar->startNode();
            ar->setNextName("coverage");
            ar->startNode();
            (*ar)(cereal::make_nvp("summary", *_summary));
            (*ar)(cereal::make_nvp("details", *_details));
            ar->finishNode();
            ar->finishNode();
        }
        _name = TIMEMORY_JOIN(
            '.', std::regex_replace(_name, std::regex{ "(.*)(\\.json$)" }, "$1"), "json");
        std::ofstream ofs{};
        if(tim::filepath::open(ofs, _name))
        {
            tim::operation::file_output_message<rocprofsys::coverage::code_coverage>{}(
                _name, std::string{ "coverage" });
            ofs << oss.str() << "\n";
        }
        else
        {
            throw std::runtime_error(
                TIMEMORY_JOIN("", "Error opening coverage output file: ", _name));
        }
    };

    _pycov.def("load", _load_coverage, "Load code coverage data");
    _pycov.def("save", _save_coverage, "Save code coverage data", py::arg("summary"),
               py::arg("details"), py::arg("filename") = "coverage.json");

    auto _concat_coverage = [](coverage_data_vector_t* _lhs,
                               coverage_data_vector_t* _rhs) {
        std::sort(_rhs->begin(), _rhs->end(), std::greater<coverage::coverage_data>{});

        auto _find = [_lhs](const auto& _v) {
            for(auto iitr = _lhs->begin(); iitr != _lhs->end(); ++iitr)
            {
                if(*iitr == _v) return std::make_pair(iitr, true);
            }
            return std::make_pair(_lhs->end(), false);
        };

        std::vector<coverage::coverage_data*> _new_entries{};
        _new_entries.reserve(_rhs->size());
        for(auto& itr : *_rhs)
        {
            auto litr = _find(itr);
            if(!litr.second)
                _new_entries.emplace_back(&itr);
            else
                *litr.first += itr;
        }

        _lhs->reserve(_lhs->size() + _new_entries.size());
        for(auto& itr : _new_entries)
            _lhs->emplace_back(std::move(*itr));
        _rhs->clear();

        std::sort(_lhs->begin(), _lhs->end(), std::greater<coverage::coverage_data>{});
        return _lhs;
    };

    _pycov.def("concat", _concat_coverage, "Combined code coverage details");

    using coverage_data_map =
        uomap_t<std::string_view, uomap_t<std::string_view, std::map<size_t, size_t>>>;

    auto _coverage_summary = [](coverage_data_vector_t* _data) {
        coverage::code_coverage _summary{};
        coverage_data_map       _mdata{};

        for(auto& itr : *_data)
            _mdata[itr.module][itr.function][itr.address] += itr.count;

        for(const auto& file : _mdata)
        {
            for(const auto& func : file.second)
            {
                for(const auto& addr : func.second)
                {
                    if(addr.second > 0)
                    {
                        _summary.count += 1;
                        _summary.covered.modules.emplace(file.first);
                        _summary.covered.functions.emplace(func.first);
                        _summary.covered.addresses.emplace(addr.first);
                    }
                    _summary.size += 1;
                    _summary.possible.modules.emplace(file.first);
                    _summary.possible.functions.emplace(func.first);
                    _summary.possible.addresses.emplace(addr.first);
                }
            }
        }

        return _summary;
    };

    _pycov.def("get_summary", _coverage_summary, "Generate a code coverage summary");

    auto _get_top = [](coverage_data_vector_t* _data, size_t _n) {
        auto _ret = *_data;
        std::sort(_ret.begin(), _ret.end(), std::greater<coverage::coverage_data>{});
        _ret.resize(std::min<size_t>(_n, _ret.size()));
        _ret.shrink_to_fit();
        return _ret;
    };

    _pycov.def("get_top", _get_top, "Get the top covered functions", py::arg("details"),
               py::arg("n") = 10);

    auto _get_bottom = [](coverage_data_vector_t* _data, size_t _n) {
        auto _ret = *_data;
        std::sort(_ret.begin(), _ret.end(), std::less<coverage::coverage_data>{});
        _ret.resize(std::min<size_t>(_n, _ret.size()));
        _ret.shrink_to_fit();
        return _ret;
    };

    _pycov.def("get_bottom", _get_bottom, "Get the bottom covered functions",
               py::arg("details"), py::arg("n") = 10);

    return _pycov;
}
}  // namespace pycoverage

namespace pyuser
{
py::module
generate(py::module& _pymod)
{
    py::module _pyuser = _pymod.def_submodule("user", "User instrumentation");

    _pyuser.def("start_trace", &rocprofsys_user_start_trace,
                "Enable tracing on this thread and all subsequently created threads");
    _pyuser.def("stop_trace", &rocprofsys_user_stop_trace,
                "Disable tracing on this thread and all subsequently created threads");
    _pyuser.def(
        "start_thread_trace", &rocprofsys_user_start_thread_trace,
        "Enable tracing on this thread. Does not apply to subsequently created threads");
    _pyuser.def(
        "stop_thread_trace", &rocprofsys_user_stop_thread_trace,
        "Enable tracing on this thread. Does not apply to subsequently created threads");
    _pyuser.def("push_region", &rocprofsys_user_push_region,
                "Start a user-defined region");
    _pyuser.def("pop_region", &rocprofsys_user_pop_region, "Start a user-defined region");
    _pyuser.def("error_string", &rocprofsys_user_error_string,
                "Return a descriptor for the provided error code");

    return _pyuser;
}
}  // namespace pyuser
}  // namespace pyrocprofsys
//
//======================================================================================//
