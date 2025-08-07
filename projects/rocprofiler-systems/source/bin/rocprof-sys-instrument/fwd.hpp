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

#include "log.hpp"

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
#include <BPatch_basicBlock.h>
#include <BPatch_basicBlockLoop.h>
#include <BPatch_callbacks.h>
#include <BPatch_function.h>
#include <BPatch_instruction.h>
#include <BPatch_object.h>
#include <BPatch_point.h>
#include <BPatch_process.h>
#include <BPatch_snippet.h>
#include <BPatch_statement.h>
#include <Function.h>
#include <Instruction.h>
#include <InstructionCategories.h>
#include <Module.h>
#include <Symtab.h>
#include <SymtabReader.h>
#include <dyntypes.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <istream>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define MUTNAMELEN       1024
#define FUNCNAMELEN      32 * 1024
#define NO_ERROR         -1
#define TIMEMORY_BIN_DIR "bin"

#if !defined(PATH_MAX)
#    define PATH_MAX std::numeric_limits<int>::max();
#endif

struct function_signature;
struct basic_block_signature;
struct module_function;

using string_t               = std::string;
using string_view_t          = std::string_view;
using stringstream_t         = std::stringstream;
using strvec_t               = std::vector<string_t>;
using strset_t               = std::set<string_t>;
using regexvec_t             = std::vector<std::regex>;
using fmodset_t              = std::set<module_function>;
using fixed_modset_t         = std::map<fmodset_t*, bool>;
using exec_callback_t        = BPatchExecCallback;
using exit_callback_t        = BPatchExitCallback;
using fork_callback_t        = BPatchForkCallback;
using patch_t                = BPatch;
using process_t              = BPatch_process;
using thread_t               = BPatch_thread;
using binary_edit_t          = BPatch_binaryEdit;
using image_t                = BPatch_image;
using module_t               = BPatch_module;
using procedure_t            = BPatch_function;
using snippet_t              = BPatch_snippet;
using call_expr_t            = BPatch_funcCallExpr;
using address_space_t        = BPatch_addressSpace;
using flow_graph_t           = BPatch_flowGraph;
using statement_t            = BPatch_statement;
using basic_block_t          = BPatch_basicBlock;
using basic_loop_t           = BPatch_basicBlockLoop;
using procedure_loc_t        = BPatch_procedureLocation;
using point_t                = BPatch_point;
using object_t               = BPatch_object;
using local_var_t            = BPatch_localVar;
using sequence_t             = BPatch_sequence;
using const_expr_t           = BPatch_constExpr;
using error_level_t          = BPatchErrorLevel;
using snippet_handle_t       = BPatchSnippetHandle;
using patch_pointer_t        = std::shared_ptr<patch_t>;
using snippet_pointer_t      = std::shared_ptr<snippet_t>;
using call_expr_pointer_t    = std::shared_ptr<call_expr_t>;
using snippet_vec_t          = std::vector<snippet_t*>;
using procedure_vec_t        = std::vector<procedure_t*>;
using basic_block_set_t      = std::set<basic_block_t*>;
using basic_loop_vec_t       = std::vector<basic_loop_t*>;
using snippet_pointer_vec_t  = std::vector<snippet_pointer_t>;
using instruction_t          = Dyninst::InstructionAPI::Instruction;
using instruction_category_t = Dyninst::InstructionAPI::InsnCategory;

namespace SymTab          = ::Dyninst::SymtabAPI;
using symtab_t            = SymTab::Symtab;
using symtab_module_t     = SymTab::Module;
using symtab_symbol_t     = SymTab::Symbol;
using symtab_func_t       = SymTab::Function;
using symbol_linkage_t    = SymTab::Symbol::SymbolLinkage;
using symbol_visibility_t = SymTab::Symbol::SymbolVisibility;

constexpr auto SL_UNKNOWN = symtab_symbol_t::SL_UNKNOWN;
constexpr auto SL_GLOBAL  = symtab_symbol_t::SL_GLOBAL;
constexpr auto SL_LOCAL   = symtab_symbol_t::SL_LOCAL;
constexpr auto SL_WEAK    = symtab_symbol_t::SL_WEAK;
constexpr auto SL_UNIQUE  = symtab_symbol_t::SL_UNIQUE;

constexpr auto SV_UNKNOWN   = symtab_symbol_t::SV_UNKNOWN;
constexpr auto SV_DEFAULT   = symtab_symbol_t::SV_DEFAULT;
constexpr auto SV_INTERNAL  = symtab_symbol_t::SV_INTERNAL;
constexpr auto SV_HIDDEN    = symtab_symbol_t::SV_HIDDEN;
constexpr auto SV_PROTECTED = symtab_symbol_t::SV_PROTECTED;

constexpr auto SL_END_V =
    std::max({ SL_UNKNOWN, SL_GLOBAL, SL_LOCAL, SL_WEAK, SL_UNIQUE }) + 1;

constexpr auto SV_END_V =
    std::max({ SV_UNKNOWN, SV_DEFAULT, SV_INTERNAL, SV_HIDDEN, SV_PROTECTED }) + 1;

void
rocprofsys_prefork_callback(thread_t* parent, thread_t* child);

enum CodeCoverageMode
{
    CODECOV_NONE = 0,
    CODECOV_FUNCTION,
    CODECOV_BASIC_BLOCK
};

//======================================================================================//
//
//                                  Global Variables
//
//======================================================================================//
//
//  label settings
//
extern bool use_return_info;
extern bool use_args_info;
extern bool use_file_info;
extern bool use_line_info;
//
//  heuristic settings
//
extern bool   allow_overlapping;
extern bool   loop_level_instr;
extern bool   instr_dynamic_callsites;
extern bool   instr_traps;
extern bool   instr_loop_traps;
extern bool   parse_all_modules;
extern size_t min_address_range;
extern size_t min_loop_address_range;
extern size_t min_instructions;
extern size_t min_loop_instructions;
//
//  debug settings
//
extern bool werror;
extern bool debug_print;
extern bool instr_print;
extern int  verbose_level;
extern int  num_log_entries;
//
//  instrumentation settings
//
extern bool simulate;
extern bool include_uninstr;
extern bool include_internal_linked_libs;
//
//  string settings
//
extern string_t main_fname;
extern string_t argv0;
extern string_t cmdv0;
extern string_t default_components;
extern string_t prefer_library;
//
//  global variables
//
extern patch_pointer_t  bpatch;
extern call_expr_t*     terminate_expr;
extern snippet_vec_t    init_names;
extern snippet_vec_t    fini_names;
extern fmodset_t        available_module_functions;
extern fmodset_t        instrumented_module_functions;
extern fmodset_t        overlapping_module_functions;
extern fmodset_t        excluded_module_functions;
extern fixed_modset_t   fixed_module_functions;
extern regexvec_t       func_include;
extern regexvec_t       func_exclude;
extern regexvec_t       file_include;
extern regexvec_t       file_exclude;
extern regexvec_t       file_restrict;
extern regexvec_t       func_restrict;
extern regexvec_t       caller_include;
extern regexvec_t       func_internal_include;
extern regexvec_t       file_internal_include;
extern regexvec_t       instruction_exclude;
extern CodeCoverageMode coverage_mode;
//
// symtab variables
//
struct symtab_data_s
{
    std::vector<symtab_module_t*>                           modules              = {};
    std::map<symtab_module_t*, std::vector<symtab_func_t*>> functions            = {};
    std::map<symtab_func_t*, std::vector<symtab_symbol_t*>> symbols              = {};
    std::unordered_map<std::string, symtab_symbol_t*>       mangled_symbol_names = {};
    std::unordered_map<std::string, symtab_func_t*>         typed_func_names     = {};
    std::unordered_map<std::string, symtab_symbol_t*>       typed_symbol_names   = {};
};

extern symtab_data_s                 symtab_data;
extern std::set<symbol_linkage_t>    enabled_linkage;
extern std::set<symbol_visibility_t> enabled_visibility;

// logging
extern std::unique_ptr<std::ofstream> log_ofs;
//
//======================================================================================//

// control debug printf statements
#define errprintf(LEVEL, ...)                                                            \
    {                                                                                    \
        char _logmsgbuff[FUNCNAMELEN];                                                   \
        snprintf(_logmsgbuff, FUNCNAMELEN, __VA_ARGS__);                                 \
        ROCPROFSYS_ADD_LOG_ENTRY(_logmsgbuff);                                           \
        if(werror || LEVEL < 0)                                                          \
        {                                                                                \
            if(debug_print || verbose_level >= LEVEL)                                    \
                fprintf(stderr, "[rocprof-sys][exe] Error! " __VA_ARGS__);               \
            char _buff[FUNCNAMELEN];                                                     \
            sprintf(_buff, "[rocprof-sys][exe] Error! " __VA_ARGS__);                    \
            throw std::runtime_error(std::string{ _buff });                              \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            if(debug_print || verbose_level >= LEVEL)                                    \
                fprintf(stderr, "[rocprof-sys][exe] Warning! " __VA_ARGS__);             \
        }                                                                                \
        fflush(stderr);                                                                  \
    }

// control verbose printf statements
#define verbprintf(LEVEL, ...)                                                           \
    {                                                                                    \
        char _logmsgbuff[FUNCNAMELEN];                                                   \
        snprintf(_logmsgbuff, FUNCNAMELEN, __VA_ARGS__);                                 \
        ROCPROFSYS_ADD_LOG_ENTRY(_logmsgbuff);                                           \
        if(debug_print || verbose_level >= LEVEL)                                        \
            fprintf(stdout, "[rocprof-sys][exe] " __VA_ARGS__);                          \
        fflush(stdout);                                                                  \
    }

#define verbprintf_bare(LEVEL, ...)                                                      \
    {                                                                                    \
        char _logmsgbuff[FUNCNAMELEN];                                                   \
        snprintf(_logmsgbuff, FUNCNAMELEN, __VA_ARGS__);                                 \
        ROCPROFSYS_ADD_LOG_ENTRY(_logmsgbuff);                                           \
        if(debug_print || verbose_level >= LEVEL) fprintf(stdout, __VA_ARGS__);          \
        fflush(stdout);                                                                  \
    }

//======================================================================================//

template <typename... T>
void
consume_parameters(T&&...)
{}

//======================================================================================//

void
process_modules(const std::vector<module_t*>&);

strset_t
get_whole_function_names();

function_signature
get_func_file_line_info(module_t* mutatee_module, procedure_t* f);

function_signature
get_loop_file_line_info(module_t* mutatee_module, procedure_t* f, flow_graph_t* cfGraph,
                        basic_loop_t* loopToInstrument);

std::map<basic_block_t*, basic_block_signature>
get_basic_block_file_line_info(module_t* module, procedure_t* func);

std::vector<statement_t>
get_source_code(module_t* module, procedure_t* func);

std::tuple<size_t, size_t>
query_instr(procedure_t* funcToInstr, procedure_loc_t traceLoc,
            flow_graph_t* cfGraph = nullptr, basic_loop_t* loopToInstrument = nullptr);

bool
query_instr(procedure_t* funcToInstr, procedure_loc_t traceLoc, flow_graph_t* cfGraph,
            basic_loop_t* loopToInstrument, bool allow_traps);

template <typename Tp>
bool
insert_instr(address_space_t* mutatee, const std::vector<point_t*>& _points, Tp traceFunc,
             procedure_loc_t traceLoc, bool allow_traps = instr_traps);

template <typename Tp>
bool
insert_instr(address_space_t* mutatee, procedure_t* funcToInstr, Tp traceFunc,
             procedure_loc_t traceLoc, flow_graph_t* cfGraph = nullptr,
             basic_loop_t* loopToInstrument = nullptr, bool allow_traps = instr_traps);

template <typename Tp>
bool
insert_instr(address_space_t* mutatee, Tp traceFunc, procedure_loc_t traceLoc,
             basic_block_t* basicBlock, bool allow_traps = instr_traps);

procedure_t*
find_function(image_t* appImage, const string_t& functionName, const strset_t& = {});

symtab_symbol_t*
find_undefined_function_symbol(image_t* app_image, const std::string& _name);

void
error_func_real(error_level_t level, int num, const char* const* params);

void
error_func_fake(error_level_t level, int num, const char* const* params);

std::string_view
get_name(procedure_t*);

std::string_view
get_name(module_t*);

symtab_func_t*
get_symtab_function(procedure_t*);

namespace std
{
std::string to_string(instruction_category_t);
std::string to_string(error_level_t);
std::string to_string(symbol_visibility_t);
std::string to_string(symbol_linkage_t);
}  // namespace std

template <typename Tp>
Tp from_string(std::string_view);

std::ostream&
operator<<(std::ostream&, symbol_linkage_t);

std::ostream&
operator<<(std::ostream&, symbol_visibility_t);

std::istream&
operator>>(std::istream&, symbol_linkage_t&);

std::istream&
operator>>(std::istream&, symbol_visibility_t&);
