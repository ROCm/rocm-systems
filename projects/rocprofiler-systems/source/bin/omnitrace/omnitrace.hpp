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

#include <timemory/backends/process.hpp>
#include <timemory/environment.hpp>
#include <timemory/mpl/apply.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/policy.hpp>
#include <timemory/tpls/cereal/archives.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/popen.hpp>
#include <timemory/variadic/macros.hpp>

#include <BPatch.h>
#include <BPatch_Vector.h>
#include <BPatch_addressSpace.h>
#include <BPatch_basicBlockLoop.h>
#include <BPatch_callbacks.h>
#include <BPatch_function.h>
#include <BPatch_point.h>
#include <BPatch_process.h>
#include <BPatch_snippet.h>
#include <BPatch_statement.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#define MUTNAMELEN       1024
#define FUNCNAMELEN      32 * 1024
#define NO_ERROR         -1
#define TIMEMORY_BIN_DIR "bin"

#if !defined(PATH_MAX)
#    define PATH_MAX std::numeric_limits<int>::max();
#endif

struct function_signature;
struct module_function;

template <typename Tp>
using bpvector_t = BPatch_Vector<Tp>;

using string_t              = std::string;
using stringstream_t        = std::stringstream;
using strvec_t              = std::vector<string_t>;
using strset_t              = std::set<string_t>;
using regexvec_t            = std::vector<std::regex>;
using fmodset_t             = std::set<module_function>;
using fixed_modset_t        = std::map<fmodset_t*, bool>;
using exec_callback_t       = BPatchExecCallback;
using exit_callback_t       = BPatchExitCallback;
using fork_callback_t       = BPatchForkCallback;
using patch_t               = BPatch;
using process_t             = BPatch_process;
using thread_t              = BPatch_thread;
using binary_edit_t         = BPatch_binaryEdit;
using image_t               = BPatch_image;
using module_t              = BPatch_module;
using procedure_t           = BPatch_function;
using snippet_t             = BPatch_snippet;
using call_expr_t           = BPatch_funcCallExpr;
using address_space_t       = BPatch_addressSpace;
using flow_graph_t          = BPatch_flowGraph;
using basic_loop_t          = BPatch_basicBlockLoop;
using procedure_loc_t       = BPatch_procedureLocation;
using point_t               = BPatch_point;
using local_var_t           = BPatch_localVar;
using const_expr_t          = BPatch_constExpr;
using error_level_t         = BPatchErrorLevel;
using patch_pointer_t       = std::shared_ptr<patch_t>;
using snippet_pointer_t     = std::shared_ptr<snippet_t>;
using call_expr_pointer_t   = std::shared_ptr<call_expr_t>;
using snippet_vec_t         = bpvector_t<snippet_t*>;
using procedure_vec_t       = bpvector_t<procedure_t*>;
using basic_loop_vec_t      = bpvector_t<basic_loop_t*>;
using snippet_pointer_vec_t = std::vector<snippet_pointer_t>;

void
omnitrace_prefork_callback(thread_t* parent, thread_t* child);

//======================================================================================//
//
//                                  Global Variables
//
//======================================================================================//
//
//  boolean settings
//
static bool use_return_info = false;
static bool use_args_info   = false;
static bool use_file_info   = false;
static bool use_line_info   = false;
//
//  integral settings
//
extern bool debug_print;
extern int  verbose_level;
//
//  string settings
//
static string_t main_fname         = "main";
static string_t argv0              = {};
static string_t cmdv0              = {};
static string_t default_components = "wall_clock";
static string_t prefer_library     = {};
//
//  global variables
//
static patch_pointer_t bpatch                        = {};
static call_expr_t*    terminate_expr                = nullptr;
static snippet_vec_t   init_names                    = {};
static snippet_vec_t   fini_names                    = {};
static fmodset_t       available_module_functions    = {};
static fmodset_t       instrumented_module_functions = {};
static fmodset_t       overlapping_module_functions  = {};
static fmodset_t       excluded_module_functions     = {};
static fixed_modset_t  fixed_module_functions        = {};
static regexvec_t      func_include                  = {};
static regexvec_t      func_exclude                  = {};
static regexvec_t      file_include                  = {};
static regexvec_t      file_exclude                  = {};
static regexvec_t      file_restrict                 = {};
static regexvec_t      func_restrict                 = {};
//
//======================================================================================//

// control debug printf statements
#define dprintf(...)                                                                     \
    if(debug_print || verbose_level > 0)                                                 \
        fprintf(stderr, "[omnitrace][exe] " __VA_ARGS__);                                \
    fflush(stderr);

// control verbose printf statements
#define verbprintf(LEVEL, ...)                                                           \
    if(verbose_level >= LEVEL) fprintf(stdout, "[omnitrace][exe] " __VA_ARGS__);         \
    fflush(stdout);

#define verbprintf_bare(LEVEL, ...)                                                      \
    if(verbose_level >= LEVEL) fprintf(stdout, __VA_ARGS__);                             \
    fflush(stdout);

//======================================================================================//

template <typename... T>
void
consume_parameters(T&&...)
{}

//======================================================================================//

extern "C"
{
    bool are_file_include_exclude_lists_empty();
    bool instrument_module(const string_t& file_name);
    bool instrument_entity(const string_t& function_name);
    bool module_constraint(char* fname);
    bool routine_constraint(const char* fname);
}

//======================================================================================//

strset_t
get_whole_function_names();

function_signature
get_func_file_line_info(module_t* mutatee_module, procedure_t* f);

function_signature
get_loop_file_line_info(module_t* mutatee_module, procedure_t* f, flow_graph_t* cfGraph,
                        basic_loop_t* loopToInstrument);

bool
query_instr(procedure_t* funcToInstr, procedure_loc_t traceLoc,
            flow_graph_t* cfGraph = nullptr, basic_loop_t* loopToInstrument = nullptr,
            bool allow_traps = true);

template <typename Tp>
bool
insert_instr(address_space_t* mutatee, procedure_t* funcToInstr, Tp traceFunc,
             procedure_loc_t traceLoc, flow_graph_t* cfGraph = nullptr,
             basic_loop_t* loopToInstrument = nullptr, bool allow_traps = true);

void
errorFunc(error_level_t level, int num, const char** params);

procedure_t*
find_function(image_t* appImage, const string_t& functionName, const strset_t& = {});

void
error_func_real(error_level_t level, int num, const char* const* params);

void
error_func_fake(error_level_t level, int num, const char* const* params);

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
struct function_signature
{
    using location_t = std::pair<unsigned long, unsigned long>;

    bool             m_loop      = false;
    bool             m_info_beg  = false;
    bool             m_info_end  = false;
    location_t       m_row       = { 0, 0 };
    location_t       m_col       = { 0, 0 };
    string_t         m_return    = {};
    string_t         m_name      = {};
    string_t         m_params    = "()";
    string_t         m_file      = {};
    mutable string_t m_signature = {};

    TIMEMORY_DEFAULT_OBJECT(function_signature)

    template <typename ArchiveT>
    void serialize(ArchiveT& _ar, const unsigned)
    {
        namespace cereal = tim::cereal;
        (void) get();
        _ar(cereal::make_nvp("loop", m_loop), cereal::make_nvp("info_beg", m_info_beg),
            cereal::make_nvp("info_end", m_info_end), cereal::make_nvp("row", m_row),
            cereal::make_nvp("col", m_col), cereal::make_nvp("return", m_return),
            cereal::make_nvp("name", m_name), cereal::make_nvp("params", m_params),
            cereal::make_nvp("file", m_file), cereal::make_nvp("signature", m_signature));
        (void) get();
    }

    function_signature(string_t _ret, const string_t& _name, string_t _file,
                       location_t _row = { 0, 0 }, location_t _col = { 0, 0 },
                       bool _loop = false, bool _info_beg = false, bool _info_end = false)
    : m_loop(_loop)
    , m_info_beg(_info_beg)
    , m_info_end(_info_end)
    , m_row(std::move(_row))
    , m_col(std::move(_col))
    , m_return(std::move(_ret))
    , m_name(tim::demangle(_name))
    , m_file(std::move(_file))
    {
        if(m_file.find('/') != string_t::npos)
            m_file = m_file.substr(m_file.find_last_of('/') + 1);
    }

    function_signature(const string_t& _ret, const string_t& _name, const string_t& _file,
                       const std::vector<string_t>& _params, location_t _row = { 0, 0 },
                       location_t _col = { 0, 0 }, bool _loop = false,
                       bool _info_beg = false, bool _info_end = false)
    : function_signature(_ret, _name, _file, _row, _col, _loop, _info_beg, _info_end)
    {
        m_params = "(";
        for(const auto& itr : _params)
            m_params.append(itr + ", ");
        if(!_params.empty()) m_params = m_params.substr(0, m_params.length() - 2);
        m_params += ")";
    }

    friend bool operator==(const function_signature& lhs, const function_signature& rhs)
    {
        return lhs.get() == rhs.get();
    }

    static auto get(function_signature& sig) { return sig.get(); }

    string_t get() const
    {
        std::stringstream ss;
        if(use_return_info && !m_return.empty()) ss << m_return << " ";
        ss << m_name;
        if(use_args_info) ss << m_params;
        if(m_loop && m_info_beg)
        {
            if(m_info_end)
            {
                ss << " [{" << m_row.first << "," << m_col.first << "}-{" << m_row.second
                   << "," << m_col.second << "}]";
            }
            else
            {
                ss << "[{" << m_row.first << "," << m_col.first << "}]";
            }
        }
        if(use_file_info && m_file.length() > 0) ss << " [" << m_file;
        if(use_line_info && m_row.first > 0) ss << ":" << m_row.first;
        if(use_file_info && m_file.length() > 0) ss << "]";

        m_signature = ss.str();
        return m_signature;
    }
};
//
//======================================================================================//
//
struct module_function
{
    using width_t   = std::array<size_t, 3>;
    using address_t = Dyninst::Address;

    static constexpr size_t absolute_max_width = 80;

    static auto& get_width()
    {
        static width_t _instance = []() {
            width_t _tmp;
            _tmp.fill(0);
            return _tmp;
        }();
        return _instance;
    }

    TIMEMORY_DEFAULT_OBJECT(module_function)

    static void reset_width() { get_width().fill(0); }

    static void update_width(const module_function& rhs)
    {
        get_width()[0] = std::max<size_t>(get_width()[0], rhs.module.length());
        get_width()[1] = std::max<size_t>(get_width()[1], rhs.function.length());
        get_width()[2] = std::max<size_t>(get_width()[2], rhs.signature.get().length());
    }

    module_function(string_t _module, string_t _func, function_signature _sign,
                    procedure_t* proc)
    : module(std::move(_module))
    , function(std::move(_func))
    , signature(std::move(_sign))
    {
        if(proc)
        {
            std::pair<address_t, address_t> _range{};
            if(proc->getAddressRange(_range.first, _range.second))
                address_range = _range.second - _range.first;
        }
    }

    module_function(module_t* mod, procedure_t* proc)
    {
        char modname[FUNCNAMELEN];
        char fname[FUNCNAMELEN];

        mod->getFullName(modname, FUNCNAMELEN);
        proc->getName(fname, FUNCNAMELEN);

        module    = modname;
        function  = fname;
        signature = get_func_file_line_info(mod, proc);
        if(!proc->isInstrumentable())
        {
            verbprintf(0,
                       "Warning! module function generated for un-instrumentable "
                       "function: %s [%s]\n",
                       function.c_str(), module.c_str());
        }
        std::pair<address_t, address_t> _range{};
        if(proc->getAddressRange(_range.first, _range.second))
            address_range = _range.second - _range.first;
    }

    friend bool operator<(const module_function& lhs, const module_function& rhs)
    {
        return (lhs.module == rhs.module)
                   ? ((lhs.function == rhs.function)
                          ? (lhs.signature.get() < rhs.signature.get())
                          : (lhs.function < rhs.function))
                   : (lhs.module < rhs.module);
    }

    friend bool operator==(const module_function& lhs, const module_function& rhs)
    {
        return std::tie(lhs.module, lhs.function, lhs.signature, lhs.address_range) ==
               std::tie(rhs.module, rhs.function, rhs.signature, rhs.address_range);
    }

    static void write_header(std::ostream& os)
    {
        auto w0 = std::min<size_t>(get_width()[0], absolute_max_width);
        auto w1 = std::min<size_t>(get_width()[1], absolute_max_width);
        auto w2 = std::min<size_t>(get_width()[2], absolute_max_width);

        std::stringstream ss;
        ss << std::setw(14) << "AddressRange"
           << "  " << std::setw(w0 + 8) << std::left << "Module"
           << " " << std::setw(w1 + 8) << std::left << "Function"
           << " " << std::setw(w2 + 8) << std::left << "FunctionSignature"
           << "\n";
        os << ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const module_function& rhs)
    {
        std::stringstream ss;

        auto w0 = std::min<size_t>(get_width()[0], absolute_max_width);
        auto w1 = std::min<size_t>(get_width()[1], absolute_max_width);
        auto w2 = std::min<size_t>(get_width()[2], absolute_max_width);

        auto _get_str = [](const std::string& _inc) {
            if(_inc.length() > absolute_max_width)
                return _inc.substr(0, absolute_max_width - 3) + "...";
            return _inc;
        };

        // clang-format off
        ss << std::setw(14) << rhs.address_range << "  "
           << std::setw(w0 + 8) << std::left << _get_str(rhs.module) << " "
           << std::setw(w1 + 8) << std::left << _get_str(rhs.function) << " "
           << std::setw(w2 + 8) << std::left << _get_str(rhs.signature.get());
        // clang-format on

        os << ss.str();
        return os;
    }

    size_t             address_range = 0;
    string_t           module        = {};
    string_t           function      = {};
    function_signature signature     = {};

    template <typename ArchiveT>
    void serialize(ArchiveT& _ar, const unsigned)
    {
        namespace cereal = tim::cereal;
        _ar(cereal::make_nvp("address_range", address_range),
            cereal::make_nvp("module", module), cereal::make_nvp("function", function),
            cereal::make_nvp("signature", signature));
    }
};
//
//======================================================================================//
//
static inline void
dump_info(std::ostream& _os, const fmodset_t& _data)
{
    module_function::reset_width();
    for(const auto& itr : _data)
        module_function::update_width(itr);

    module_function::write_header(_os);
    for(const auto& itr : _data)
        _os << itr << '\n';

    module_function::reset_width();
}
//
template <typename ArchiveT,
          std::enable_if_t<tim::concepts::is_archive<ArchiveT>::value, int> = 0>
static inline void
dump_info(ArchiveT& _ar, const fmodset_t& _data)
{
    _ar(tim::cereal::make_nvp("module_functions", _data));
}
//
static inline void
dump_info(const string_t& _label, string_t _oname, const string_t& _ext,
          const fmodset_t& _data, int _level, bool _fail)
{
    namespace cereal = tim::cereal;
    namespace policy = tim::policy;

    _oname += "." + _ext;
    auto _handle_error = [&]() {
        std::stringstream _msg{};
        _msg << "[dump_info] Error opening '" << _oname << " for output";
        verbprintf(_level, "%s\n", _msg.str().c_str());
        if(_fail)
            throw std::runtime_error(std::string{ "[omnitrace][exe]" } + _msg.str());
    };

    if(!debug_print && verbose_level < _level) return;

    if(_ext == "txt")
    {
        std::ofstream ofs{};
        if(!tim::filepath::open(ofs, _oname))
            _handle_error();
        else
        {
            verbprintf(_level, "Outputting '%s'... ", _oname.c_str());
            dump_info(ofs, _data);
            verbprintf_bare(_level, "Done\n");
        }
        ofs.close();
    }
    else if(_ext == "xml")
    {
        std::stringstream oss{};
        {
            using output_policy     = policy::output_archive<cereal::XMLOutputArchive>;
            output_policy::indent() = true;
            auto ar                 = output_policy::get(oss);

            ar->setNextName("omnitrace");
            ar->startNode();
            ar->setNextName(_label.c_str());
            ar->startNode();
            (*ar)(cereal::make_nvp("module_functions", _data));
            ar->finishNode();
            ar->finishNode();
        }

        std::ofstream ofs{};
        if(!tim::filepath::open(ofs, _oname))
            _handle_error();
        else
        {
            verbprintf(_level, "Outputting '%s'... ", _oname.c_str());
            ofs << oss.str() << std::endl;
            verbprintf_bare(_level, "Done\n");
        }
        ofs.close();
    }
    else if(_ext == "json")
    {
        std::stringstream oss{};
        {
            using output_policy = policy::output_archive<cereal::PrettyJSONOutputArchive>;
            auto ar             = output_policy::get(oss);

            ar->setNextName("omnitrace");
            ar->startNode();
            ar->setNextName(_label.c_str());
            ar->startNode();
            (*ar)(cereal::make_nvp("module_functions", _data));
            ar->finishNode();
            ar->finishNode();
        }

        std::ofstream ofs{};
        if(!tim::filepath::open(ofs, _oname))
            _handle_error();
        else
        {
            verbprintf(_level, "Outputting '%s'... ", _oname.c_str());
            ofs << oss.str() << std::endl;
            verbprintf_bare(_level, "Done\n");
        }
        ofs.close();
    }
    else
    {
        throw std::runtime_error(TIMEMORY_JOIN(
            "", "[omnitrace][exe] Error in ", __FUNCTION__, " :: filename '", _oname,
            "' does not have one of recognized file extensions: txt, json, xml"));
    }
}
//
static inline void
dump_info(const string_t& _oname, const fmodset_t& _data, int _level, bool _fail,
          const string_t& _type, const strset_t& _ext)
{
    for(const auto& itr : _ext)
        dump_info(_type, _oname, itr, _data, _level, _fail);
}
//
static inline void
load_info(const string_t& _label, const string_t& _iname, fmodset_t& _data, int _level)
{
    namespace cereal = tim::cereal;
    namespace policy = tim::policy;

    auto        _pos = _iname.find_last_of('.');
    std::string _ext = {};
    if(_pos != std::string::npos) _ext = _iname.substr(_pos + 1, _iname.length());

    auto _handle_error = [&]() {
        std::stringstream _msg{};
        _msg << "[load_info] Error opening '" << _iname << " for input";
        verbprintf(_level, "%s\n", _msg.str().c_str());
        throw std::runtime_error(std::string{ "[omnitrace][exe]" } + _msg.str());
    };

    if(_ext == "xml")
    {
        verbprintf(_level, "Reading '%s'... ", _iname.c_str());
        std::ifstream ifs{ _iname };
        if(!ifs)
            _handle_error();
        else
        {
            using input_policy = policy::input_archive<cereal::XMLInputArchive>;
            auto ar            = input_policy::get(ifs);

            ar->setNextName("omnitrace");
            ar->startNode();
            ar->setNextName(_label.c_str());
            ar->startNode();
            (*ar)(cereal::make_nvp("module_functions", _data));
            ar->finishNode();
            ar->finishNode();
        }
        verbprintf_bare(_level, "Done\n");
        ifs.close();
    }
    else if(_ext == "json")
    {
        verbprintf(_level, "Reading '%s'... ", _iname.c_str());
        std::ifstream ifs{ _iname };
        if(!ifs)
            _handle_error();
        else
        {
            using input_policy = policy::input_archive<cereal::JSONInputArchive>;
            auto ar            = input_policy::get(ifs);

            ar->setNextName("omnitrace");
            ar->startNode();
            ar->setNextName(_label.c_str());
            ar->startNode();
            (*ar)(cereal::make_nvp("module_functions", _data));
            ar->finishNode();
            ar->finishNode();
        }
        verbprintf_bare(_level, "Done\n");
        ifs.close();
    }
    else
    {
        throw std::runtime_error(TIMEMORY_JOIN(
            "", "[omnitrace][exe] Error in ", __FUNCTION__, " :: filename '", _iname,
            "' does not have one of recognized extentions: txt, json, xml :: ", _ext));
    }
}
//
static inline void
load_info(const string_t& _inp, std::map<std::string, fmodset_t*>& _data, int _level)
{
    std::vector<std::string> _exceptions{};
    _exceptions.reserve(_data.size());
    for(auto& itr : _data)
    {
        try
        {
            fmodset_t _tmp{};
            load_info(itr.first, _inp, _tmp, _level);
            // add to the existing
            itr.second->insert(_tmp.begin(), _tmp.end());
            // if it did not throw it was successfully loaded
            _exceptions.clear();
            break;
        } catch(std::exception& _e)
        {
            _exceptions.emplace_back(_e.what());
        }
    }
    if(!_exceptions.empty())
    {
        std::stringstream _msg{};
        for(auto& itr : _exceptions)
        {
            _msg << "[omnitrace][exe] " << itr << "\n";
        }
        throw std::runtime_error(_msg.str());
    }
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
//
