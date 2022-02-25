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

#include "omnitrace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>

bool debug_print   = false;
int  verbose_level = tim::get_env<int>("TIMEMORY_RUN_VERBOSE", 0);

namespace
{
bool                                       binary_rewrite          = false;
bool                                       is_attached             = false;
bool                                       loop_level_instr        = false;
bool                                       werror                  = false;
bool                                       use_mpi                 = false;
bool                                       is_static_exe           = false;
bool                                       is_driver               = false;
bool                                       allow_overlapping       = false;
bool                                       instr_dynamic_callsites = false;
bool                                       instr_traps             = false;
bool                                       instr_loop_traps        = false;
bool                                       explicit_dump_and_exit  = false;
size_t                                     batch_size              = 50;
strset_t                                   extra_libs              = {};
size_t                                     min_address_range       = (1 << 8);  // 256
size_t                                     min_loop_address_range  = (1 << 8);  // 256
std::vector<std::pair<uint64_t, string_t>> hash_ids                = {};
std::map<string_t, bool>                   use_stubs               = {};
std::map<string_t, procedure_t*>           beg_stubs               = {};
std::map<string_t, procedure_t*>           end_stubs               = {};
strvec_t                                   init_stub_names         = {};
strvec_t                                   fini_stub_names         = {};
strset_t                                   used_stub_names         = {};
std::vector<call_expr_pointer_t>           env_variables           = {};
std::map<string_t, call_expr_pointer_t>    beg_expr                = {};
std::map<string_t, call_expr_pointer_t>    end_expr                = {};
const auto                                 npos_v                  = string_t::npos;
string_t                                   instr_mode              = "trace";
string_t                                   print_instrumented      = {};
string_t                                   print_excluded          = {};
string_t                                   print_available         = {};
string_t                                   print_overlapping       = {};
strset_t                                   print_formats           = { "txt", "json" };
std::string                                modfunc_dump_dir        = {};
auto regex_opts = std::regex_constants::egrep | std::regex_constants::optimize;

std::string
get_absolute_exe_filepath(std::string exe_name, const std::string& env_path = "PATH");

std::string
get_absolute_lib_filepath(std::string        lib_name,
                          const std::string& env_path = "LD_LIBRARY_PATH");

bool
file_exists(const std::string& name);

std::string
get_realpath(const std::string&);

std::string
get_cwd();
}  // namespace

//======================================================================================//
//
// entry point
//
//======================================================================================//
//
int
main(int argc, char** argv)
{
#if defined(DYNINST_API_RT)
    auto _dyn_api_rt_paths = tim::delimit(DYNINST_API_RT, ":");
#else
    auto _dyn_api_rt_paths = std::vector<std::string>{};
#endif
    auto _dyn_api_rt_abs = get_absolute_lib_filepath("libdyninstAPI_RT.so");
    _dyn_api_rt_paths.insert(_dyn_api_rt_paths.begin(), _dyn_api_rt_abs);
    for(auto&& itr : _dyn_api_rt_paths)
    {
        auto _file_exists = [](const std::string& _fname) {
            struct stat _buffer;
            if(stat(_fname.c_str(), &_buffer) == 0)
                return (S_ISREG(_buffer.st_mode) != 0 || S_ISLNK(_buffer.st_mode) != 0);
            return false;
        };
        if(_file_exists(itr))
            tim::set_env<string_t>("DYNINSTAPI_RT_LIB", itr, 0);
        else if(_file_exists(TIMEMORY_JOIN('/', itr, "libdyninstAPI_RT.so")))
            tim::set_env<string_t>("DYNINSTAPI_RT_LIB",
                                   TIMEMORY_JOIN('/', itr, "libdyninstAPI_RT.so"), 0);
        else if(_file_exists(TIMEMORY_JOIN('/', itr, "libdyninstAPI_RT.a")))
            tim::set_env<string_t>("DYNINSTAPI_RT_LIB",
                                   TIMEMORY_JOIN('/', itr, "libdyninstAPI_RT.a"), 0);
    }
    verbprintf(0, "DYNINST_API_RT: %s\n",
               tim::get_env<string_t>("DYNINSTAPI_RT_LIB", "").c_str());

    argv0 = argv[0];

    bpatch                              = std::make_shared<patch_t>();
    address_space_t*      addr_space    = nullptr;
    string_t              mutname       = {};
    string_t              outfile       = {};
    std::vector<string_t> inputlib      = { "libomnitrace-dl" };
    std::vector<string_t> libname       = {};
    std::vector<string_t> sharedlibname = {};
    std::vector<string_t> staticlibname = {};
    tim::process::id_t    _pid          = -1;

    fixed_module_functions = {
        { &available_module_functions, false },
        { &instrumented_module_functions, false },
        { &excluded_module_functions, false },
        { &overlapping_module_functions, false },
    };

    bpatch->setTypeChecking(true);
    bpatch->setSaveFPR(true);
    bpatch->setDelayedParsing(true);
    bpatch->setDebugParsing(false);
    bpatch->setInstrStackFrames(false);
    bpatch->setLivenessAnalysis(false);
    bpatch->setBaseTrampDeletion(false);
    bpatch->setTrampRecursive(false);
    bpatch->setMergeTramp(true);

    std::set<std::string> dyninst_defs = { "TypeChecking", "SaveFPR", "DelayedParsing",
                                           "MergeTramp" };

    int _argc = argc;
    int _cmdc = 0;

    char** _argv = new char*[_argc];
    char** _cmdv = nullptr;

    for(int i = 0; i < argc; ++i)
        _argv[i] = nullptr;

    auto copy_str = [](char*& _dst, const char* _src) { _dst = strdup(_src); };

    copy_str(_argv[0], argv[0]);

    for(int i = 1; i < argc; ++i)
    {
        string_t _arg = argv[i];
        if(_arg.length() == 2 && _arg == "--")
        {
            _argc        = i;
            _cmdc        = argc - i - 1;
            _cmdv        = new char*[_cmdc + 1];
            _cmdv[_cmdc] = nullptr;
            int k        = 0;
            for(int j = i + 1; j < argc; ++j, ++k)
            {
                copy_str(_cmdv[k], argv[j]);
            }
            mutname = _cmdv[0];
            break;
        }
        else
        {
            copy_str(_argv[i], argv[i]);
        }
    }

    auto cmd_string = [](int _ac, char** _av) -> std::string {
        if(_ac == 0) return std::string{};
        stringstream_t ss;
        for(int i = 0; i < _ac; ++i)
            ss << " " << _av[i];
        return ss.str().substr(1);
    };

    if(_cmdc > 0 && !mutname.empty())
    {
        auto resolved_mutname = get_absolute_exe_filepath(mutname);
        if(resolved_mutname != mutname)
        {
            mutname = resolved_mutname;
            delete _cmdv[0];
            copy_str(_cmdv[0], resolved_mutname.c_str());
        }
    }

    if(verbose_level > 1)
    {
        std::cout << "[omnitrace][exe][original]: " << cmd_string(argc, argv)
                  << std::endl;
        std::cout << "[omnitrace][exe][cfg-args]: " << cmd_string(_argc, _argv)
                  << std::endl;
    }

    if(_cmdc > 0)
        std::cout << "\n[omnitrace][exe][command]: " << cmd_string(_cmdc, _cmdv)
                  << "\n\n";

    if(_cmdc > 0) cmdv0 = _cmdv[0];

    // now can loop through the options.  If the first character is '-', then we know
    // we have an option.  Check to see if it is one of our options and process it. If
    // it is unrecognized, then set the errflag to report an error.  When we come to a
    // non '-' charcter, then we must be at the application name.
    using parser_t = tim::argparse::argument_parser;
    parser_t parser("omnitrace");
    string_t extra_help = "-- <CMD> <ARGS>";

    parser.enable_help();
    parser.add_argument({ "" }, "");
    parser.add_argument({ "[DEBUG OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser.add_argument({ "--debug" }, "Debug output")
        .max_count(1)
        .action([](parser_t& p) {
            debug_print = p.get<bool>("debug");
            if(debug_print && !p.exists("verbose")) verbose_level = 256;
        });
    parser.add_argument({ "-v", "--verbose" }, "Verbose output")
        .max_count(1)
        .action([](parser_t& p) {
            if(p.get_count("v") == 0)
                verbose_level = 1;
            else
                verbose_level = p.get<int>("v");
        });
    parser.add_argument({ "-e", "--error" }, "All warnings produce runtime errors")
        .dtype("boolean")
        .max_count(1)
        .action([](parser_t& p) { werror = p.get<bool>("error"); });
    parser
        .add_argument({ "--simulate" },
                      "Exit after outputting diagnostic "
                      "{available,instrumented,excluded,overlapping} module "
                      "function lists, e.g. available-instr.txt")
        .max_count(1)
        .dtype("bool")
        .action([](parser_t& p) { explicit_dump_and_exit = p.get<bool>("simulate"); });
    parser
        .add_argument({ "--print-format" },
                      "Output format for diagnostic "
                      "{available,instrumented,excluded,overlapping} module "
                      "function lists, e.g. {print-dir}/available-instr.txt")
        .min_count(1)
        .max_count(3)
        .dtype("string")
        .choices({ "xml", "json", "txt" })
        .action([](parser_t& p) { print_formats = p.get<strset_t>("print-format"); });
    parser
        .add_argument({ "--print-dir" },
                      "Output directory for diagnostic "
                      "{available,instrumented,excluded,overlapping} module "
                      "function lists, e.g. {print-dir}/available-instr.txt")
        .count(1)
        .dtype("string")
        .action([](parser_t& p) { modfunc_dump_dir = p.get<std::string>("print-dir"); });
    parser
        .add_argument(
            { "--print-available" },
            "Print the available entities for instrumentation (functions, modules, or "
            "module-function pair) to stdout applying regular expressions and exit")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action(
            [](parser_t& p) { print_available = p.get<std::string>("print-available"); });
    parser
        .add_argument(
            { "--print-instrumented" },
            "Print the instrumented entities (functions, modules, or module-function "
            "pair) to stdout after applying regular expressions and exit")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action([](parser_t& p) {
            print_instrumented = p.get<std::string>("print-instrumented");
        });
    parser
        .add_argument({ "--print-excluded" },
                      "Print the entities for instrumentation (functions, modules, or "
                      "module-function "
                      "pair) which are excluded from the instrumentation to stdout after "
                      "applying regular expressions and exit")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action(
            [](parser_t& p) { print_excluded = p.get<std::string>("print-excluded"); });
    parser
        .add_argument(
            { "--print-overlapping" },
            "Print the entities for instrumentation (functions, modules, or "
            "module-function pair) which overlap other function calls or have multiple "
            "entry points to stdout applying regular expressions and exit")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action([](parser_t& p) {
            print_overlapping = p.get<std::string>("print-overlapping");
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[MODE OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser
        .add_argument({ "-o", "--output" },
                      "Enable generation of a new executable (binary-rewrite). If a "
                      "filename is not provided, omnitrace will use the basename and "
                      "output to the cwd, unless the target binary is in the cwd. In the "
                      "latter case, omnitrace will either use ${PWD}/<basename>.inst "
                      "(non-libraries) or ${PWD}/instrumented/<basename> (libraries)")
        .min_count(0)
        .max_count(1)
        .dtype("string")
        .action([&outfile](parser_t& p) {
            binary_rewrite = true;
            outfile        = p.get<string_t>("output");
        });
    parser.add_argument({ "-p", "--pid" }, "Connect to running process")
        .dtype("int")
        .count(1)
        .action([&_pid](parser_t& p) { _pid = p.get<int>("pid"); });
    parser
        .add_argument({ "-M", "--mode" },
                      "Instrumentation mode. 'trace' mode instruments the selected "
                      "functions, 'sampling' mode only instruments the main function to "
                      "start and stop the sampler.")
        .choices({ "trace", "sampling" })
        .count(1)
        .action([](parser_t& p) { instr_mode = p.get<string_t>("mode"); });
    if(_cmdc == 0)
    {
        parser
            .add_argument({ "-c", "--command" },
                          "Input executable and arguments (if '-- <CMD>' not provided)")
            .count(1)
            .action([&](parser_t& p) {
                auto keys = p.get<strvec_t>("c");
                if(keys.empty())
                {
                    p.print_help(extra_help);
                    std::exit(EXIT_FAILURE);
                }
                keys.at(0) = get_absolute_exe_filepath(keys.at(0));
                mutname    = keys.at(0);
                _cmdc      = keys.size();
                _cmdv      = new char*[_cmdc];
                for(int i = 0; i < _cmdc; ++i)
                    copy_str(_cmdv[i], keys.at(i).c_str());
            });
    }

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[LIBRARY OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser.add_argument({ "--prefer" }, "Prefer this library types when available")
        .choices({ "shared", "static" })
        .count(1)
        .action([](parser_t& p) { prefer_library = p.get<string_t>("prefer"); });
    parser
        .add_argument(
            { "-L", "--library" },
            "Libraries with instrumentation routines (default: \"libomnitrace\")")
        .action([&inputlib](parser_t& p) { inputlib = p.get<strvec_t>("library"); });
    parser
        .add_argument({ "-m", "--main-function" },
                      "The primary function to instrument around, e.g. 'main'")
        .count(1)
        .action([](parser_t& p) { main_fname = p.get<string_t>("main-function"); });
    /*
    parser
        .add_argument({ "-s", "--stubs" }, "Instrument with library stubs for LD_PRELOAD")
        .dtype("boolean")
        .max_count(1)
        .action([&inputlib](parser_t& p) {
            if(p.get<bool>("stubs"))
            {
                for(auto& itr : inputlib)
                    itr += "-stubs";
            }
        });
    */
    parser.add_argument({ "--driver" }, "Force main or _init/_fini instrumentation")
        .dtype("boolean")
        .max_count(1)
        .action([](parser_t& p) { is_driver = p.get<bool>("driver"); });
    parser
        .add_argument({ "--load" },
                      "Supplemental instrumentation library names w/o extension (e.g. "
                      "'libinstr' for 'libinstr.so' or 'libinstr.a')")
        .dtype("string")
        .action([](parser_t& p) {
            auto _load = p.get<strvec_t>("load");
            for(const auto& itr : _load)
                extra_libs.insert(itr);
        });
    parser
        .add_argument({ "--load-instr" },
                      "Load {available,instrumented,excluded,overlapping}-instr JSON or "
                      "XML file(s) and override what is read from the binary")
        .dtype("filepath")
        .max_count(-1)
        .action([](parser_t& p) {
            auto                              _load = p.get<strvec_t>("load-instr");
            std::map<std::string, fmodset_t*> module_function_map = {
                { "available_module_functions", &available_module_functions },
                { "instrumented_module_functions", &instrumented_module_functions },
                { "excluded_module_functions", &excluded_module_functions },
                { "overlapping_module_functions", &overlapping_module_functions },
            };
            for(const auto& itr : _load)
                load_info(itr, module_function_map, 0);
            for(const auto& itr : module_function_map)
            {
                auto _empty = itr.second->empty();
                if(!_empty)
                    verbprintf(0, "Loaded %zu module functions for %s\n",
                               itr.second->size(), itr.first.c_str());
                fixed_module_functions.at(itr.second) = !_empty;
            }
        });
    parser
        .add_argument({ "--init-functions" },
                      "Initialization function(s) for supplemental instrumentation "
                      "libraries (see '--load' option)")
        .dtype("string")
        .action([](parser_t& p) { init_stub_names = p.get<strvec_t>("init-functions"); });
    parser
        .add_argument({ "--fini-functions" },
                      "Finalization function(s) for supplemental instrumentation "
                      "libraries (see '--load' option)")
        .dtype("string")
        .action([](parser_t& p) { init_stub_names = p.get<strvec_t>("fini-functions"); });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[SYMBOL SELECTION OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser.add_argument({ "-I", "--function-include" },
                        "Regex(es) for including functions (despite heuristics)");
    parser.add_argument({ "-E", "--function-exclude" },
                        "Regex(es) for excluding functions (always applied)");
    parser.add_argument({ "-R", "--function-restrict" },
                        "Regex(es) for restricting functions only to those "
                        "that match the provided regular-expressions");
    parser.add_argument({ "-MI", "--module-include" },
                        "Regex(es) for selecting modules/files/libraries "
                        "(despite heuristics)");
    parser.add_argument({ "-ME", "--module-exclude" },
                        "Regex(es) for excluding modules/files/libraries "
                        "(always applied)");
    parser.add_argument({ "-MR", "--module-restrict" },
                        "Regex(es) for restricting modules/files/libraries only to those "
                        "that match the provided regular-expressions");

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[RUNTIME OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser
        .add_argument({ "--label" },
                      "Labeling info for functions. By default, just the function name "
                      "is recorded. Use these options to gain more information about the "
                      "function signature or location of the functions")
        .choices({ "file", "line", "return", "args" })
        .dtype("string")
        .action([](parser_t& p) {
            auto _labels = p.get<strvec_t>("label");
            for(const auto& itr : _labels)
            {
                if(std::regex_match(itr, std::regex("file", std::regex_constants::icase)))
                    use_file_info = true;
                else if(std::regex_match(
                            itr, std::regex("return", std::regex_constants::icase)))
                    use_return_info = true;
                else if(std::regex_match(itr,
                                         std::regex("args", std::regex_constants::icase)))
                    use_args_info = true;
                else if(std::regex_match(itr,
                                         std::regex("line", std::regex_constants::icase)))
                    use_line_info = true;
            }
        });
    parser.add_argument()
        .names({ "-d", "--default-components" })
        .dtype("string")
        .description("Default components to instrument (only useful when timemory is "
                     "enabled in omnitrace library)")
        .action([](parser_t& p) {
            auto _components   = p.get<strvec_t>("default-components");
            default_components = {};
            for(size_t i = 0; i < _components.size(); ++i)
            {
                if(_components.at(i) == "none")
                {
                    default_components = "none";
                    break;
                }
                default_components += _components.at(i);
                if(i + 1 < _components.size()) default_components += ",";
            }
            if(default_components == "none")
                default_components = {};
            else
            {
                auto _strcomp = p.get<std::string>("default-components");
                if(!_strcomp.empty() && default_components.empty())
                    default_components = _strcomp;
            }
        });
    parser.add_argument({ "--env" },
                        "Environment variables to add to the runtime in form "
                        "VARIABLE=VALUE. E.g. use '--env OMNITRACE_USE_TIMEMORY=ON' to "
                        "default to using timemory instead of perfetto");
    parser
        .add_argument({ "--mpi" },
                      "Enable MPI support (requires omnitrace built w/ MPI and GOTCHA "
                      "support). NOTE: this will automatically be activated if "
                      "MPI_Init/MPI_Init_thread and MPI_Finalize are found in the symbol "
                      "table of target")
        .max_count(1)
        .action([](parser_t& p) { use_mpi = p.get<bool>("mpi"); });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[GRANULARITY OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser.add_argument({ "-l", "--instrument-loops" }, "Instrument at the loop level")
        .dtype("boolean")
        .max_count(1)
        .action([](parser_t& p) { loop_level_instr = p.get<bool>("instrument-loops"); });
    parser
        .add_argument({ "-r", "--min-address-range" },
                      "If the address range of a function is less than this value, "
                      "exclude it from instrumentation")
        .count(1)
        .dtype("int")
        .set_default(min_address_range)
        .action(
            [](parser_t& p) { min_address_range = p.get<size_t>("min-address-range"); });
    parser
        .add_argument({ "--min-address-range-loop" },
                      "If the address range of a function containing a loop is less than "
                      "this value, exclude it from instrumentation")
        .count(1)
        .dtype("int")
        .set_default(min_loop_address_range)
        .action([](parser_t& p) {
            min_loop_address_range = p.get<size_t>("min-address-range-loop");
        });
    parser
        .add_argument({ "--dynamic-callsites" },
                      "Force instrumentation if a function has dynamic callsites (e.g. "
                      "function pointers)")
        .max_count(1)
        .dtype("boolean")
        .action([](parser_t& p) {
            instr_dynamic_callsites = p.get<bool>("dynamic-callsites");
        });
    parser
        .add_argument(
            { "--traps" },
            "Instrument points which require using a trap. On the x86 architecture, "
            "because instructions are of variable size, the instruction at a point may "
            "be too small for Dyninst to replace it with the normal code sequence used "
            "to call instrumentation. Also, when instrumentation is placed at points "
            "other than subroutine entry, exit, or call points, traps may be used to "
            "ensure the instrumentation fits. In this case, Dyninst replaces the "
            "instruction with a single-byte instruction that generates a trap.")
        .max_count(1)
        .dtype("bool")
        .set_default(instr_traps)
        .action([](parser_t& p) { instr_traps = p.get<bool>("traps"); });
    parser
        .add_argument({ "--loop-traps" },
                      "Instrument points within a loop which require using a trap (only "
                      "relevant when --instrument-loops is enabled).")
        .max_count(1)
        .dtype("bool")
        .set_default(instr_loop_traps)
        .action([](parser_t& p) { instr_loop_traps = p.get<bool>("loop-traps"); });
    parser
        .add_argument(
            { "--allow-overlapping" },
            "Allow dyninst to instrument either multiple functions which overlap (share "
            "part of same function body) or single functions with multiple entry points. "
            "For more info, see Section 2 of the DyninstAPI documentation.")
        .count(0)
        .action([](parser_t&) { allow_overlapping = true; });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[DYNINST OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser
        .add_argument(
            { "-b", "--batch-size" },
            "Dyninst supports batch insertion of multiple points during runtime "
            "instrumentation. If one large batch "
            "insertion fails, this value will be used to create smaller batches. Larger "
            "batches generally decrease the instrumentation time")
        .count(1)
        .dtype("int")
        .action([](parser_t& p) { batch_size = p.get<size_t>("batch-size"); });

    parser
        .add_argument({ "--dyninst-options" },
                      "Advanced dyninst options: BPatch::set<OPTION>(bool), e.g. "
                      "bpatch->setTrampRecursive(true)")
        .choices({ "TypeChecking", "SaveFPR", "DebugParsing", "DelayedParsing",
                   "InstrStackFrames", "TrampRecursive", "MergeTramp",
                   "BaseTrampDeletion" });

    auto err = parser.parse(_argc, _argv);

    if(parser.exists("h") || parser.exists("help"))
    {
        parser.print_help(extra_help);
        return 0;
    }

    if(err)
    {
        std::cerr << err << std::endl;
        parser.print_help(extra_help);
        return -1;
    }

    if(binary_rewrite && outfile.empty())
    {
        auto _is_local = (get_realpath(cmdv0) ==
                          TIMEMORY_JOIN('/', get_cwd(), ::basename(cmdv0.c_str())));
        auto _cmd      = std::string{ ::basename(cmdv0.c_str()) };
        if(_cmd.find('.') == std::string::npos)
        {
            // there is no extension, assume it is an exe
            outfile = (_is_local) ? TIMEMORY_JOIN('.', _cmd, "inst") : _cmd;
        }
        else if(_cmd.find("lib") == 0 || _cmd.find(".so") != std::string::npos ||
                _cmd.find(".a") == _cmd.length() - 2)
        {
            // if it starts with lib, ends with .a, or contains .so (e.g. libfoo.so,
            // libfoo.so.2), assume it is a library and retain the name but put it in a
            // different directory
            outfile = (_is_local) ? TIMEMORY_JOIN('/', "instrumented", _cmd) : _cmd;
        }
        else
        {
            outfile = (_is_local) ? TIMEMORY_JOIN('.', _cmd, "inst") : _cmd;
        }
        verbprintf(0,
                   "Binary rewrite was activated via '-o' but no filename was provided. "
                   "Using: '%s'\n",
                   outfile.c_str());
    }

    if(modfunc_dump_dir.empty())
    {
        modfunc_dump_dir = tim::get_env<std::string>("OMNITRACE_OUTPUT_PATH", "");
        if(modfunc_dump_dir.empty())
        {
            auto _exe_base = (binary_rewrite) ? outfile : std::string{ cmdv0 };
            auto _pos      = _exe_base.find_last_of('/');
            if(_pos != std::string::npos && _pos + 1 < _exe_base.length())
                _exe_base = _exe_base.substr(_pos + 1);
            modfunc_dump_dir = TIMEMORY_JOIN("-", "omnitrace", _exe_base, "output");
        }
    }

    if(verbose_level >= 0) tim::makedir(modfunc_dump_dir);

    //----------------------------------------------------------------------------------//
    //
    //                              REGEX OPTIONS
    //
    //----------------------------------------------------------------------------------//
    //
    {
        //  Helper function for adding regex expressions
        auto add_regex = [](auto& regex_array, const string_t& regex_expr) {
            if(!regex_expr.empty())
                regex_array.emplace_back(std::regex(regex_expr, regex_opts));
        };

        add_regex(func_include, tim::get_env<string_t>("OMNITRACE_REGEX_INCLUDE", ""));
        add_regex(func_exclude, tim::get_env<string_t>("OMNITRACE_REGEX_EXCLUDE", ""));
        add_regex(func_restrict, tim::get_env<string_t>("OMNITRACE_REGEX_RESTRICT", ""));

        add_regex(file_include,
                  tim::get_env<string_t>("OMNITRACE_REGEX_MODULE_INCLUDE", ""));
        add_regex(file_exclude,
                  tim::get_env<string_t>("OMNITRACE_REGEX_MODULE_EXCLUDE", ""));
        add_regex(file_restrict,
                  tim::get_env<string_t>("OMNITRACE_REGEX_MODULE_RESTRICT", ""));

        //  Helper function for parsing the regex options
        auto _parse_regex_option = [&parser, &add_regex](const string_t& _option,
                                                         regexvec_t&     _regex_vec) {
            if(parser.exists(_option))
            {
                auto keys = parser.get<strvec_t>(_option);
                for(const auto& itr : keys)
                    add_regex(_regex_vec, itr);
            }
        };

        _parse_regex_option("function-include", func_include);
        _parse_regex_option("function-exclude", func_exclude);
        _parse_regex_option("function-restrict", func_restrict);
        _parse_regex_option("module-include", file_include);
        _parse_regex_option("module-exclude", file_exclude);
        _parse_regex_option("module-restrict", file_restrict);
    }

    //----------------------------------------------------------------------------------//
    //
    //                              DYNINST OPTIONS
    //
    //----------------------------------------------------------------------------------//

    int dyninst_verb = 2;
    if(parser.exists("dyninst-options"))
    {
        dyninst_defs = parser.get<std::set<std::string>>("dyninst-options");
        dyninst_verb = 0;
    }

    auto get_dyninst_option = [&](const std::string& _opt) {
        bool _ret = dyninst_defs.find(_opt) != dyninst_defs.end();
        verbprintf(dyninst_verb, "[dyninst-option]> %-20s = %4s\n", _opt.c_str(),
                   (_ret) ? "on" : "off");
        return _ret;
    };

    bpatch->setTypeChecking(get_dyninst_option("TypeChecking"));
    bpatch->setSaveFPR(get_dyninst_option("SaveFPR"));
    bpatch->setDelayedParsing(get_dyninst_option("DelayedParsing"));

    bpatch->setDebugParsing(get_dyninst_option("DebugParsing"));
    bpatch->setInstrStackFrames(get_dyninst_option("InstrStackFrames"));
    bpatch->setTrampRecursive(get_dyninst_option("TrampRecursive"));
    bpatch->setMergeTramp(get_dyninst_option("MergeTramp"));
    bpatch->setBaseTrampDeletion(get_dyninst_option("BaseTrampDeletion"));

    //----------------------------------------------------------------------------------//
    //
    //                              MAIN
    //
    //----------------------------------------------------------------------------------//

    if(_cmdc == 0)
    {
        parser.print_help(extra_help);
        fprintf(stderr, "\nError! No command for dynamic instrumentation. Use "
                        "\n\tomnitrace <OPTIONS> -- <COMMAND> <ARGS>\nE.g. "
                        "\n\tomnitrace -o foo.inst -- ./foo\nwill output an "
                        "instrumented version of 'foo' executable to 'foo.inst'\n");
        return EXIT_FAILURE;
    }

    verbprintf(0, "instrumentation target: %s\n", mutname.c_str());

    // did we load a library?  if not, load the default
    auto generate_libnames = [](auto& _targ, const auto& _base,
                                const std::set<string_t>& _ext) {
        for(const auto& bitr : _base)
            for(const auto& eitr : _ext)
            {
                _targ.emplace_back(bitr + eitr);
            }
    };

    generate_libnames(libname, inputlib, { "" });
    generate_libnames(sharedlibname, inputlib, { ".so" });
    generate_libnames(staticlibname, inputlib, { ".a" });

    // Register a callback function that prints any error messages
    bpatch->registerErrorCallback(error_func_real);

    //----------------------------------------------------------------------------------//
    //
    //  Start the instrumentation procedure by opening a file for binary editing,
    //  attaching to a running process, or starting a process
    //
    //----------------------------------------------------------------------------------//

    // for runtime instrumentation, we need to set this before the process gets created
    if(!binary_rewrite)
    {
        tim::set_env("HSA_ENABLE_INTERRUPT", "0", 0);
        if(_pid >= 0)
        {
            verbprintf(-10, "#-------------------------------------------------------"
                            "-------------------------------------------#\n");
            verbprintf(-10, "\n");
            verbprintf(-10, "WARNING! Sampling may result in ioctl() deadlock within "
                            "the ROCR runtime.\n");
            verbprintf(-10,
                       "To avoid this, set HSA_ENABLE_INTERRUPT=0 in the environment "
                       "before starting your ROCm/HIP application\n");
            verbprintf(-10, "\n");
            verbprintf(-10, "#-------------------------------------------------------"
                            "-------------------------------------------#\n");
        }
    }

    addr_space =
        omnitrace_get_address_space(bpatch, _cmdc, _cmdv, binary_rewrite, _pid, mutname);

    if(!addr_space)
    {
        errprintf(-1, "address space for dynamic instrumentation was not created\n");
    }

    process_t*     app_thread = nullptr;
    binary_edit_t* app_binary = nullptr;

    // get image
    verbprintf(1, "Getting the address space image, modules, and procedures...\n");
    image_t*                  app_image     = addr_space->getImage();
    bpvector_t<module_t*>*    app_modules   = app_image->getModules();
    bpvector_t<procedure_t*>* app_functions = app_image->getProcedures();
    bpvector_t<module_t*>     modules;
    bpvector_t<procedure_t*>  functions;

    //----------------------------------------------------------------------------------//
    //
    //  Generate a log of all the available procedures and modules
    //
    //----------------------------------------------------------------------------------//
    std::set<std::string> module_names;

    static auto _insert_module_function = [](fmodset_t& _module_funcs, auto _v) {
        if(!fixed_module_functions.at(&_module_funcs)) _module_funcs.emplace(_v);
    };

    auto _add_overlapping = [](module_t* mitr, procedure_t* pitr) {
        if(!pitr->isInstrumentable()) return;
        std::vector<procedure_t*> _overlapping{};
        if(pitr->findOverlapping(_overlapping))
        {
            _insert_module_function(overlapping_module_functions,
                                    module_function{ mitr, pitr });
            for(auto* oitr : _overlapping)
            {
                if(!oitr->isInstrumentable()) continue;
                _insert_module_function(overlapping_module_functions,
                                        module_function{ oitr->getModule(), oitr });
            }
        }
    };

    if(app_modules && !app_modules->empty())
    {
        modules = *app_modules;
        for(auto* itr : modules)
        {
            auto* procedures = itr->getProcedures();
            if(procedures)
            {
                for(auto* pitr : *procedures)
                {
                    if(!pitr->isInstrumentable()) continue;
                    auto _modfn = module_function{ itr, pitr };
                    module_names.insert(_modfn.module);
                    _insert_module_function(available_module_functions, _modfn);
                    _add_overlapping(itr, pitr);
                }
            }
        }
    }
    else
    {
        verbprintf(0, "Warning! No modules in application...\n");
    }

    if(app_functions && !app_functions->empty())
    {
        functions = *app_functions;
        for(auto* itr : functions)
        {
            module_t* mod = itr->getModule();
            if(mod && itr->isInstrumentable())
            {
                auto _modfn = module_function{ mod, itr };
                module_names.insert(_modfn.module);
                _insert_module_function(available_module_functions, _modfn);
                _add_overlapping(mod, itr);
            }
        }
    }
    else
    {
        verbprintf(0, "Warning! No functions in application...\n");
    }

    verbprintf(0, "Module size before loading instrumentation library: %lu\n",
               (long unsigned) modules.size());

    if(debug_print || verbose_level > 2)
    {
        module_function::reset_width();
        for(const auto& itr : available_module_functions)
            module_function::update_width(itr);

        auto mwid = module_function::get_width().at(0);
        mwid      = std::max<size_t>(mwid, 15);
        mwid      = std::min<size_t>(mwid, 90);
        auto ncol = 180 / std::min<size_t>(mwid, 180);
        std::cout << "### MODULES ###\n| ";
        for(size_t i = 0; i < module_names.size(); ++i)
        {
            auto itr = module_names.begin();
            std::advance(itr, i);
            std::string _v = *itr;
            if(_v.length() >= mwid)
            {
                auto _resume = _v.length() - mwid + 15;
                _v           = _v.substr(0, 12) + "..." + _v.substr(_resume);
            }
            std::cout << std::setw(mwid) << _v << " | ";
            if(i % ncol == ncol - 1) std::cout << "\n| ";
        }
        std::cout << '\n' << std::endl;
    }

    auto _output_prefix = tim::get_env<std::string>("OMNITRACE_OUTPUT_PREFIX", "");

    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "available-instr")),
              available_module_functions, 1, werror, "available-instr", print_formats);
    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "overlapping-instr")),
              overlapping_module_functions, 1, werror, "overlapping_module_functions",
              print_formats);

    //----------------------------------------------------------------------------------//
    //
    //  Get the derived type of the address space
    //
    //----------------------------------------------------------------------------------//

    is_static_exe = addr_space->isStaticExecutable();

    if(binary_rewrite)
        app_binary = static_cast<BPatch_binaryEdit*>(addr_space);
    else
        app_thread = static_cast<BPatch_process*>(addr_space);

    is_attached = (_pid >= 0 && app_thread != nullptr);

    if(!app_binary && !app_thread)
    {
        errprintf(-1, "No application thread or binary! nullptr to BPatch_binaryEdit* "
                      "and BPatch_process*\n")
    }

    //----------------------------------------------------------------------------------//
    //
    //  Helper functions for library stuff
    //
    //----------------------------------------------------------------------------------//

    auto load_library = [addr_space](const std::vector<string_t>& _libnames) {
        bool result = false;
        // track the tried library names
        string_t _tried_libs;
        for(auto _libname : _libnames)
        {
            _libname = get_absolute_lib_filepath(_libname);
            _tried_libs += string_t("|") + _libname;
            verbprintf(0, "loading library: '%s'...\n", _libname.c_str());
            result = (addr_space->loadLibrary(_libname.c_str()) != nullptr);
            verbprintf(2, "loadLibrary(%s) result = %s\n", _libname.c_str(),
                       (result) ? "success" : "failure");
            if(result) break;
        }
        if(!result)
        {
            fprintf(stderr,
                    "Error: 'loadLibrary(%s)' failed.\nPlease ensure that the "
                    "library directory is in LD_LIBRARY_PATH environment variable "
                    "or absolute path is provided\n",
                    _tried_libs.substr(1).c_str());
            exit(EXIT_FAILURE);
        }
    };

    auto get_library_ext = [=](const std::vector<string_t>& linput) {
        auto lnames           = linput;
        auto _get_library_ext = [](string_t lname) {
            if(lname.find(".so") != string_t::npos ||
               lname.find(".a") == lname.length() - 2)
                return lname;
            if(!prefer_library.empty())
                return (lname +
                        ((prefer_library == "static" || is_static_exe) ? ".a" : ".so"));
            else
                return (lname + ((is_static_exe) ? ".a" : ".so"));
        };
        for(auto& lname : lnames)
            lname = _get_library_ext(lname);
        return lnames;
    };

    //----------------------------------------------------------------------------------//
    //
    //  find _init and _fini before loading instrumentation library!
    //  These will be used for initialization and finalization if main is not found
    //
    //----------------------------------------------------------------------------------//

    auto* _mutatee_init = find_function(app_image, "_init");
    auto* _mutatee_fini = find_function(app_image, "_fini");
    auto* main_func     = find_function(app_image, main_fname.c_str());
    auto* mpi_init_func = find_function(app_image, "MPI_Init", { "MPI_Init_thread" });
    auto* mpi_fini_func = find_function(app_image, "MPI_Finalize");

    //----------------------------------------------------------------------------------//
    //
    //  Load the instrumentation libraries
    //
    //----------------------------------------------------------------------------------//

    load_library(get_library_ext(libname));

    for(const auto& itr : extra_libs)
        load_library(get_library_ext({ itr }));

    //----------------------------------------------------------------------------------//
    //
    //  Find the primary functions that will be used for instrumentation
    //
    //----------------------------------------------------------------------------------//

    verbprintf(0, "Finding functions in image...\n");

    auto* entr_trace = find_function(app_image, "omnitrace_push_trace");
    auto* exit_trace = find_function(app_image, "omnitrace_pop_trace");
    auto* entr_hash  = find_function(app_image, "omnitrace_push_trace_hash");
    auto* exit_hash  = find_function(app_image, "omnitrace_pop_trace_hash");
    auto* init_func  = find_function(app_image, "omnitrace_init");
    auto* fini_func  = find_function(app_image, "omnitrace_finalize");
    auto* env_func   = find_function(app_image, "omnitrace_set_env");
    auto* mpi_func   = find_function(app_image, "omnitrace_set_mpi");
    auto* hash_func  = find_function(app_image, "omnitrace_add_hash_id");

    if(!main_func && main_fname == "main") main_func = find_function(app_image, "_main");

    if(mpi_init_func && mpi_fini_func) use_mpi = true;

    bool use_mpip = false;
    if(use_mpi) use_mpip = true;

    //----------------------------------------------------------------------------------//
    //
    //  Handle supplemental instrumentation library functions
    //
    //----------------------------------------------------------------------------------//

    auto add_instr_library = [&](const string_t& _name, const string_t& _beg,
                                 const string_t& _end) {
        verbprintf(3,
                   "Attempting to find instrumentation for '%s' via '%s' and '%s'...\n",
                   _name.c_str(), _beg.c_str(), _end.c_str());
        if(_beg.empty() || _end.empty()) return false;
        auto* _beg_func = find_function(app_image, _beg);
        auto* _end_func = find_function(app_image, _end);
        if(_beg_func && _end_func)
        {
            use_stubs[_name] = true;
            beg_stubs[_name] = _beg_func;
            end_stubs[_name] = _end_func;
            used_stub_names.insert(_beg);
            used_stub_names.insert(_end);
            verbprintf(0, "Instrumenting '%s' via '%s' and '%s'...\n", _name.c_str(),
                       _beg.c_str(), _end.c_str());
            return true;
        }
        return false;
    };

    if(!extra_libs.empty())
    {
        verbprintf(2, "Adding extra libraries...\n");
    }

    for(const auto& itr : extra_libs)
    {
        string_t _name = itr;
        size_t   _pos  = _name.find_last_of('/');
        if(_pos != npos_v) _name = _name.substr(_pos + 1);
        _pos = _name.find('.');
        if(_pos != npos_v) _name = _name.substr(0, _pos);
        _pos = _name.find("libomnitrace-");
        if(_pos != npos_v)
            _name = _name.erase(_pos, std::string("libomnitrace-").length());
        _pos = _name.find("lib");
        if(_pos == 0) _name = _name.substr(_pos + std::string("lib").length());
        while((_pos = _name.find('-')) != npos_v)
            _name.replace(_pos, 1, "_");

        verbprintf(2,
                   "Supplemental instrumentation library '%s' is named '%s' after "
                   "removing everything before last '/', everything after first '.', and "
                   "'libomnitrace-'...\n",
                   itr.c_str(), _name.c_str());

        use_stubs[_name] = false;

        string_t best_init_name = {};
        for(const auto& sitr : init_stub_names)
        {
            if(sitr.find(_name) != npos_v && used_stub_names.count(sitr) == 0)
            {
                verbprintf(
                    3, "Found possible match for '%s' instrumentation init: '%s'...\n",
                    _name.c_str(), sitr.c_str());
                best_init_name = sitr;
                break;
            }
        }

        string_t base_fini_name = {};
        for(const auto& sitr : fini_stub_names)
        {
            if(sitr.find(_name) != npos_v && used_stub_names.count(sitr) == 0)
            {
                verbprintf(
                    3, "Found possible match for '%s' instrumentation fini: '%s'...\n",
                    _name.c_str(), sitr.c_str());
                base_fini_name = sitr;
                break;
            }
        }

        if(add_instr_library(_name, best_init_name, base_fini_name)) continue;

        // check user-specified signatures first
        for(const auto& bitr : init_stub_names)
        {
            if(used_stub_names.find(bitr) != used_stub_names.end()) continue;
            for(const auto& fitr : fini_stub_names)
            {
                if(used_stub_names.find(fitr) != used_stub_names.end()) continue;
                if(add_instr_library(_name, bitr, fitr))
                    goto found_instr_functions;  // exit loop after match
            }
        }

        // check standard function signature if no user-specified matches
        if(add_instr_library(_name, TIMEMORY_JOIN("", "omnitrace_register_" + _name),
                             TIMEMORY_JOIN("", "omnitrace_deregister_" + _name)))
            continue;

    found_instr_functions:
        continue;
    }

    //----------------------------------------------------------------------------------//
    //
    //  Check for any issues finding the required functions
    //
    //----------------------------------------------------------------------------------//

    if(!main_func && is_driver)
    {
        errprintf(0, "could not find '%s'\n", main_fname.c_str());
        if(!_mutatee_init || !_mutatee_fini)
        {
            errprintf(-1, "could not find '%s' or '%s', aborting\n", "_init", "_fini");
        }
        else
        {
            errprintf(0, "using '%s' and '%s' in lieu of '%s'...", "_init", "_fini",
                      main_fname.c_str());
        }
    }
    else if(!main_func && !is_driver)
    {
        verbprintf(0, "Warning! No main function and is not driver!\n");
    }

    using pair_t = std::pair<procedure_t*, string_t>;

    for(const auto& itr :
        { pair_t(main_func, main_fname), pair_t(entr_trace, "omnitrace_push_trace"),
          pair_t(exit_trace, "omnitrace_pop_trace"), pair_t(init_func, "omnitrace_init"),
          pair_t(fini_func, "omnitrace_finalize"),
          pair_t(env_func, "omnitrace_set_env") })
    {
        if(itr.first == main_func && !is_driver) continue;
        if(!itr.first)
        {
            errprintf(-1, "could not find required function :: '%s;\n",
                      itr.second.c_str());
        }
    }

    if(use_mpi && !(mpi_func || (mpi_init_func && mpi_fini_func)))
    {
        errprintf(-1, "MPI support was requested but omnitrace was not built with MPI "
                      "and GOTCHA support");
    }

    auto check_for_debug_info = [](bool& _has_debug_info, auto* _func) {
        // This heuristic guesses that debugging info is available if function
        // is not defined in the DEFAULT_MODULE
        if(_func && !_has_debug_info)
        {
            module_t* _module = _func->getModule();
            if(_module)
            {
                char moduleName[FUNCNAMELEN];
                _module->getFullName(moduleName, FUNCNAMELEN);
                if(strcmp(moduleName, "DEFAULT_MODULE") != 0) _has_debug_info = true;
            }
        }
    };

    bool has_debug_info = false;
    check_for_debug_info(has_debug_info, main_func);
    check_for_debug_info(has_debug_info, _mutatee_init);
    check_for_debug_info(has_debug_info, _mutatee_fini);

    //----------------------------------------------------------------------------------//
    //
    //  Find the entry/exit point of either the main (if executable) or the _init
    //  and _fini functions of the library
    //
    //----------------------------------------------------------------------------------//

    bpvector_t<point_t*>* main_entr_points = nullptr;
    bpvector_t<point_t*>* main_exit_points = nullptr;

    if(main_func)
    {
        verbprintf(2, "Finding main function entry/exit... ");
        main_entr_points = main_func->findPoint(BPatch_entry);
        main_exit_points = main_func->findPoint(BPatch_exit);
        verbprintf(2, "Done\n");
    }
    else if(is_driver)
    {
        if(_mutatee_init)
        {
            verbprintf(2, "Finding init entry...\n");
            main_entr_points = _mutatee_init->findPoint(BPatch_entry);
        }
        if(_mutatee_fini)
        {
            verbprintf(2, "Finding fini exit...\n");
            main_exit_points = _mutatee_fini->findPoint(BPatch_exit);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Create the call arguments for the initialization and finalization routines
    //  and the snippets which are inserted using those arguments
    //
    //----------------------------------------------------------------------------------//

    // begin insertion
    if(binary_rewrite)
    {
        verbprintf(2, "Beginning insertion set...\n");
        addr_space->beginInsertionSet();
    }

    function_signature main_sign("int", "main", "", { "int", "char**" });
    if(main_func)
    {
        verbprintf(2, "Getting main function signature...\n");
        main_sign = get_func_file_line_info(main_func->getModule(), main_func);
        if(main_sign.m_params == "()") main_sign.m_params = "(int argc, char** argv)";
    }

    verbprintf(2, "Getting call expressions... ");

    auto _init_arg0 = main_fname;
    if(main_func) main_sign.get();

    auto main_call_args = omnitrace_call_expr(main_sign.get());
    auto init_call_args = omnitrace_call_expr(instr_mode, binary_rewrite, _init_arg0);
    auto fini_call_args = omnitrace_call_expr();
    auto umpi_call_args = omnitrace_call_expr(use_mpi, is_attached);
    auto none_call_args = omnitrace_call_expr();

    verbprintf(2, "Done\n");
    verbprintf(2, "Getting call snippets... ");

    auto init_call = init_call_args.get(init_func);
    auto fini_call = fini_call_args.get(fini_func);
    auto umpi_call = umpi_call_args.get(mpi_func);

    auto main_beg_call = main_call_args.get(entr_trace);

    verbprintf(2, "Done\n");

    for(const auto& itr : use_stubs)
    {
        if(beg_stubs[itr.first] && end_stubs[itr.first])
        {
            beg_expr[itr.first] = none_call_args.get(beg_stubs[itr.first]);
            end_expr[itr.first] = none_call_args.get(end_stubs[itr.first]);
        }
    }

    std::string _libname = {};
    for(auto&& itr : sharedlibname)
    {
        if(_libname.empty()) _libname = get_absolute_lib_filepath(itr, "LD_LIBRARY_PATH");
        if(_libname.empty()) _libname = get_absolute_lib_filepath(itr, "LIBRARY_PATH");
    }
    for(auto&& itr : staticlibname)
    {
        if(_libname.empty()) _libname = get_absolute_lib_filepath(itr, "LIBRARY_PATH");
        if(_libname.empty()) _libname = get_absolute_lib_filepath(itr, "LD_LIBRARY_PATH");
    }
    if(_libname.empty()) _libname = "libomnitrace.so";

    // prioritize the user environment arguments
    auto env_vars = parser.get<strvec_t>("env");
    env_vars.emplace_back(TIMEMORY_JOIN('=', "OMNITRACE_MODE", instr_mode));
    env_vars.emplace_back(TIMEMORY_JOIN('=', "HSA_ENABLE_INTERRUPT", "0"));
    env_vars.emplace_back(TIMEMORY_JOIN('=', "HSA_TOOLS_LIB", _libname));
    env_vars.emplace_back(TIMEMORY_JOIN('=', "OMNITRACE_MPI_INIT", "OFF"));
    env_vars.emplace_back(TIMEMORY_JOIN('=', "OMNITRACE_MPI_FINALIZE", "OFF"));
    env_vars.emplace_back(
        TIMEMORY_JOIN('=', "OMNITRACE_TIMEMORY_COMPONENTS", default_components));
    env_vars.emplace_back(
        TIMEMORY_JOIN('=', "OMNITRACE_USE_MPIP",
                      (binary_rewrite && use_mpi && use_mpip) ? "ON" : "OFF"));
    if(use_mpi) env_vars.emplace_back(TIMEMORY_JOIN('=', "OMNITRACE_USE_PID", "ON"));

    for(auto& itr : env_vars)
    {
        auto p = tim::delimit(itr, "=");
        if(p.size() != 2)
        {
            errprintf(0, "environment variable %s not in form VARIABLE=VALUE\n",
                      itr.c_str());
        }
        tim::set_env(p.at(0), p.at(1));
        auto _expr = omnitrace_call_expr(p.at(0), p.at(1));
        env_variables.emplace_back(_expr.get(env_func));
    }

    //----------------------------------------------------------------------------------//
    //
    //  Configure the initialization and finalization routines
    //
    //----------------------------------------------------------------------------------//

    for(const auto& itr : env_variables)
    {
        if(itr) init_names.emplace_back(itr.get());
    }

    for(const auto& itr : beg_expr)
    {
        if(itr.second)
        {
            verbprintf(1, "+ Adding %s instrumentation...\n", itr.first.c_str());
            init_names.emplace_back(itr.second.get());
        }
        else
        {
            verbprintf(1, "- Skipping %s instrumentation...\n", itr.first.c_str());
        }
    }

    if(umpi_call) init_names.emplace_back(umpi_call.get());
    if(init_call) init_names.emplace_back(init_call.get());
    if(main_beg_call) init_names.emplace_back(main_beg_call.get());

    for(const auto& itr : end_expr)
        if(itr.second) fini_names.emplace_back(itr.second.get());
    if(fini_call) fini_names.emplace_back(fini_call.get());

    //----------------------------------------------------------------------------------//
    //
    //  Lambda for instrumenting procedures. The first pass (usage_pass = true) will
    //  generate the hash_ids for each string so that these can be inserted in bulk
    //  with one operation and do not have to be calculated during runtime.
    //
    //----------------------------------------------------------------------------------//
    std::vector<std::function<void()>> instr_procedure_functions;
    auto instr_procedures = [&](const procedure_vec_t& procedures) {
        //
        auto _report = [](int _lvl, const string_t& _action, const string_t& _type,
                          const string_t& _reason, const string_t& _name,
                          const std::string& _extra = {}) {
            static std::map<std::string, strset_t> already_reported{};
            if(already_reported[_type].count(_name) == 0)
            {
                verbprintf(_lvl, "[%s][%s] %s :: '%s'", _type.c_str(), _action.c_str(),
                           _reason.c_str(), _name.c_str());
                if(!_extra.empty()) verbprintf_bare(_lvl, " (%s)", _extra.c_str());
                verbprintf_bare(_lvl, "...\n");
                already_reported[_type].insert(_name);
            }
        };

        auto check_regex_restrictions = [](const std::string& _name,
                                           const regexvec_t&  _regexes) {
            // NOLINTNEXTLINE
            for(auto& itr : _regexes)
                if(std::regex_search(_name, itr)) return true;
            return false;
        };

        std::pair<size_t, size_t> _count = { 0, 0 };
        for(auto* itr : procedures)
        {
            if(!itr) continue;

            char modname[FUNCNAMELEN];
            char fname[FUNCNAMELEN];

            itr->getName(fname, FUNCNAMELEN);
            module_t* mod = itr->getModule();
            if(mod)
                mod->getFullName(modname, FUNCNAMELEN);
            else
                itr->getModuleName(modname, FUNCNAMELEN);

            if(!itr->isInstrumentable())
            {
                _report(3, "Skipping", "function", "uninstrumentable", fname);
                continue;
            }

            auto name = get_func_file_line_info(mod, itr);

            if(itr == main_func || name.m_name == "main" || name.get() == main_sign.get())
            {
                hash_ids.emplace_back(std::hash<string_t>()(main_sign.get()),
                                      main_sign.get());
                _insert_module_function(
                    available_module_functions,
                    module_function{ modname, fname, main_sign, itr });
                _insert_module_function(
                    instrumented_module_functions,
                    module_function{ modname, fname, main_sign, itr });
                continue;
            }

            if(strlen(modname) == 0)
            {
                _report(3, "Skipping", "module", "empty name", modname);
                continue;
            }

            if(name.get().empty())
            {
                _report(3, "Skipping", "function", "empty name", fname);
                continue;
            }

            // apply module and function restrictions
            auto _force_inc = false;

            //--------------------------------------------------------------------------//
            //
            //              RESTRICT REGEXES
            //
            //--------------------------------------------------------------------------//
            if(!file_restrict.empty())
            {
                if(check_regex_restrictions(modname, file_restrict))
                {
                    _report(2, "Forcing", "module", "module-restrict-regex", modname);
                    _force_inc = true;
                }
                else
                {
                    _report(3, "Skipping", "module", "module-restrict-regex", modname);
                    continue;
                }
            }

            if(!func_restrict.empty())
            {
                if(check_regex_restrictions(name.m_name, func_restrict))
                {
                    _report(2, "Forcing", "function", "function-restrict-regex",
                            name.m_name);
                    _force_inc = true;
                }
                else if(check_regex_restrictions(name.get(), func_restrict))
                {
                    _report(2, "Forcing", "function", "function-restrict-regex",
                            name.get());
                    _force_inc = true;
                }
                else
                {
                    _report(3, "Skipping", "function", "function-restrict-regex",
                            name.get());
                    continue;
                }
            }

            //--------------------------------------------------------------------------//
            //
            //              INCLUDE REGEXES
            //
            //--------------------------------------------------------------------------//
            if(!file_include.empty())
            {
                if(check_regex_restrictions(modname, file_include))
                {
                    _report(2, "Forcing", "module", "module-include-regex", modname);
                    _force_inc = true;
                }
            }

            if(!func_include.empty())
            {
                if(check_regex_restrictions(name.m_name, func_include))
                {
                    _report(2, "Forcing", "function", "function-include-regex",
                            name.m_name);
                    _force_inc = true;
                }
                else if(check_regex_restrictions(name.get(), func_include))
                {
                    _report(2, "Forcing", "function", "function-include-regex",
                            name.get());
                    _force_inc = true;
                }
            }

            //--------------------------------------------------------------------------//
            //
            //              EXCLUDE REGEXES
            //
            //--------------------------------------------------------------------------//
            if(!file_exclude.empty())
            {
                if(check_regex_restrictions(modname, file_exclude))
                {
                    _report(2, "Skipping", "module", "module-exclude-regex", modname);
                    continue;
                }
            }

            if(!func_exclude.empty())
            {
                if(check_regex_restrictions(name.m_name, func_exclude))
                {
                    _report(2, "Skipping", "function", "function-exclude-regex",
                            name.m_name);
                    continue;
                }
                else if(check_regex_restrictions(name.get(), func_exclude))
                {
                    _report(2, "Skipping", "function", "function-exclude-regex",
                            name.get());
                    continue;
                }
            }

            // try to get loops via the control flow graph
            flow_graph_t*    cfg = itr->getCFG();
            basic_loop_vec_t basic_loop{};
            if(cfg) cfg->getOuterLoops(basic_loop);

            if(!_force_inc)
            {
                if(module_constraint(modname)) continue;
                if(!instrument_module(modname)) continue;

                if(routine_constraint(name.m_name.c_str())) continue;
                if(!instrument_entity(name.m_name)) continue;

                if(is_static_exe && has_debug_info && string_t{ fname } == "_fini" &&
                   string_t{ modname } == "DEFAULT_MODULE")
                {
                    _report(3, "Skipping", "function", "DEFAULT_MODULE", fname);
                    continue;
                }

                _add_overlapping(mod, itr);

                if(!allow_overlapping &&
                   overlapping_module_functions.find(module_function{ mod, itr }) !=
                       overlapping_module_functions.end())
                {
                    _report(3, "Skipping", "function", "overlapping", fname);
                    continue;
                }

                // directly try to get loop entry points
                const std::vector<point_t*>* _loop_entries =
                    itr->findPoint(BPatch_locLoopEntry);

                // if the function has dynamic callsites and user specified instrumenting
                // dynamic callsites, force the instrumentation
                bool _force_instr = false;
                if(cfg && instr_dynamic_callsites)
                    _force_instr = cfg->containsDynamicCallsites();

                auto _address_range = module_function{ mod, itr }.address_range;
                auto _num_loop_entries =
                    (_loop_entries)
                        ? std::max<size_t>(_loop_entries->size(), basic_loop.size())
                        : basic_loop.size();
                auto _has_loop_entries = (_num_loop_entries > 0);
                auto _skip_range =
                    (_has_loop_entries) ? false : (_address_range < min_address_range);
                auto _skip_loop_range = (_has_loop_entries)
                                            ? (_address_range < min_loop_address_range)
                                            : false;

                if(_force_instr && (_skip_range || _skip_loop_range))
                {
                    _report(2, "Forcing", "function", "dynamic-callsite", fname);
                }
                else if(_skip_range)
                {
                    _report(2, "Skipping", "function", "min-address-range", fname,
                            TIMEMORY_JOIN('=', "range", _address_range));
                    continue;
                }
                else if(_skip_loop_range)
                {
                    _report(2, "Skipping", "function", "min-address-range-loop", fname,
                            TIMEMORY_JOIN('=', "range", _address_range));
                    continue;
                }
            }

            bool _entr_success =
                query_instr(itr, BPatch_entry, nullptr, nullptr, instr_traps);
            bool _exit_success =
                query_instr(itr, BPatch_exit, nullptr, nullptr, instr_traps);
            if(!_entr_success && !_exit_success)
            {
                _report(3, "Skipping", "function",
                        "Either no entry "
                        "instrumentation points were found or instrumentation "
                        "required traps and instrumenting via traps were disabled.",
                        fname);
                continue;
            }
            else if(_entr_success && !_exit_success)
            {
                std::stringstream _ss{};
                _ss << "Function can be only partially instrument (entry = "
                    << std::boolalpha << _entr_success << ", exit = " << _exit_success
                    << ")";
                _report(3, "Skipping", "function", _ss.str(), fname);
                continue;
            }

            hash_ids.emplace_back(std::hash<string_t>()(name.get()), name.get());
            _insert_module_function(available_module_functions,
                                    module_function{ mod, itr });
            _insert_module_function(instrumented_module_functions,
                                    module_function{ mod, itr });

            auto _f = [=]() {
                static std::set<size_t> _reported{};
                auto                    _hashv =
                    std::hash<std::string>{}(TIMEMORY_JOIN('|', modname, name.m_name));
                if(!_reported.emplace(_hashv).second)
                {
                    verbprintf(1, "Instrumenting |> [ %s ] -> [ %s ]\n", modname,
                               name.m_name.c_str());
                }

                auto _name       = name.get();
                auto _hash       = std::hash<string_t>()(_name);
                auto _trace_entr = (entr_hash) ? omnitrace_call_expr(_hash)
                                               : omnitrace_call_expr(_name.c_str());
                auto _trace_exit = (exit_hash) ? omnitrace_call_expr(_hash)
                                               : omnitrace_call_expr(_name.c_str());
                auto _entr       = _trace_entr.get((entr_hash) ? entr_hash : entr_trace);
                auto _exit       = _trace_exit.get((exit_hash) ? exit_hash : exit_trace);

                insert_instr(addr_space, itr, _entr, BPatch_entry, nullptr, nullptr,
                             instr_traps);
                insert_instr(addr_space, itr, _exit, BPatch_exit, nullptr, nullptr,
                             instr_traps);
            };

            instr_procedure_functions.emplace_back(_f);
            ++_count.first;

            if(loop_level_instr)
            {
                verbprintf(3, "Instrumenting at the loop level: %s\n",
                           name.m_name.c_str());

                for(auto* litr : basic_loop)
                {
                    bool _lentr_success =
                        query_instr(itr, BPatch_entry, cfg, litr, instr_loop_traps);
                    bool _lexit_success =
                        query_instr(itr, BPatch_exit, cfg, litr, instr_loop_traps);
                    if(!_lentr_success && !_lexit_success)
                    {
                        _report(
                            3, "Skipping", "function-loop",
                            "Either no entry instrumentation points were found or "
                            "instrumentation "
                            "required traps and instrumenting via traps were disabled.",
                            fname);
                        continue;
                    }
                    else if(_lentr_success && !_lexit_success)
                    {
                        std::stringstream _ss{};
                        _ss << "Function can be only partially instrument (entry = "
                            << std::boolalpha << _lentr_success
                            << ", exit = " << _lexit_success << ")";
                        _report(3, "Skipping", "function-loop", _ss.str(), fname);
                        continue;
                    }

                    auto lname  = get_loop_file_line_info(mod, itr, cfg, litr);
                    auto _lname = lname.get();
                    auto _lhash = std::hash<string_t>()(_lname);
                    hash_ids.emplace_back(_lhash, _lname);
                    auto _lf = [=]() {
                        static std::set<size_t> _reported{};
                        auto                    _hashv = std::hash<std::string>{}(
                            TIMEMORY_JOIN('|', modname, name.m_name));
                        if(!_reported.emplace(_hashv).second)
                        {
                            verbprintf(1, "Loop Instrumenting |> [ %s ] -> [ %s ]\n",
                                       modname, name.m_name.c_str());
                        }
                        auto _ltrace_entr = (entr_hash)
                                                ? omnitrace_call_expr(_lhash)
                                                : omnitrace_call_expr(_lname.c_str());
                        auto _ltrace_exit = (exit_hash)
                                                ? omnitrace_call_expr(_lhash)
                                                : omnitrace_call_expr(_lname.c_str());
                        auto _lentr =
                            _ltrace_entr.get((entr_hash) ? entr_hash : entr_trace);
                        auto _lexit =
                            _ltrace_exit.get((exit_hash) ? exit_hash : exit_trace);

                        insert_instr(addr_space, itr, _lentr, BPatch_entry, cfg, litr,
                                     instr_loop_traps);
                        insert_instr(addr_space, itr, _lexit, BPatch_exit, cfg, litr,
                                     instr_loop_traps);
                    };
                    instr_procedure_functions.emplace_back(_lf);
                    ++_count.second;
                }
            }
        }
        return _count;
    };

    //----------------------------------------------------------------------------------//
    //
    //  Do a first pass through all procedures to generate the hash ids
    //
    //----------------------------------------------------------------------------------//

    if(instr_mode == "trace")
    {
        const int _verbose_lvl = 2;
        verbprintf(2, "Beginning loop over modules [hash id generation pass]\n");
        std::vector<std::pair<std::string, std::pair<size_t, size_t>>> _pass_info{};
        for(auto& m : modules)
        {
            char modname[1024];
            m->getName(modname, 1024);
            if(strstr(modname, "libdyninst") != nullptr) continue;

            if(!m->getProcedures())
            {
                verbprintf(_verbose_lvl, "Skipping module w/ no procedures: '%s'\n",
                           modname);
                continue;
            }

            verbprintf(_verbose_lvl + 1, "Parsing module: %s\n", modname);
            bpvector_t<procedure_t*>* p = m->getProcedures();
            if(!p) continue;

            verbprintf(_verbose_lvl, "%4zu procedures are begin processed in %s\n",
                       p->size(), modname);

            auto _count = instr_procedures(*p);

            _pass_info.emplace_back(modname, _count);
        }
        // report the instrumented
        for(auto& itr : _pass_info)
        {
            auto _valid = (verbose_level > _verbose_lvl ||
                           (itr.second.first + itr.second.second) > 0);
            if(_valid)
            {
                verbprintf(_verbose_lvl, "%4zu instrumented procedures in %s\n",
                           itr.second.first, itr.first.c_str());
                _valid = (loop_level_instr &&
                          (verbose_level > _verbose_lvl || itr.second.second > 0));
                if(_valid)
                {
                    verbprintf(_verbose_lvl, "%4zu instrumented loop procedures in %s\n",
                               itr.second.second, itr.first.c_str());
                }
            }
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Add the snippet that assign the hash ids
    //
    //----------------------------------------------------------------------------------//

    omnitrace_snippet_vec hash_snippet_vec{};
    // generate a call expression for each hash + key
    for(auto& itr : hash_ids)
        hash_snippet_vec.generate(hash_func, itr.first, itr.second.c_str());
    // append all the call expressions to init names
    hash_snippet_vec.append(init_names);

    //----------------------------------------------------------------------------------//
    //
    //  Insert the initialization and finalization routines into the main entry and
    //  exit points
    //
    //----------------------------------------------------------------------------------//

    if(app_thread && is_attached)
    {
        assert(app_thread != nullptr);
        verbprintf(1, "Executing initial snippets...\n");
        for(auto* itr : init_names)
            app_thread->oneTimeCode(*itr);
    }
    else
    {
        if(main_entr_points)
        {
            verbprintf(1, "Adding main entry snippets...\n");
            addr_space->insertSnippet(BPatch_sequence(init_names), *main_entr_points,
                                      BPatch_callBefore, BPatch_firstSnippet);
        }
    }

    if(main_exit_points)
    {
        verbprintf(1, "Adding main exit snippets...\n");
        addr_space->insertSnippet(BPatch_sequence(fini_names), *main_exit_points,
                                  BPatch_callAfter, BPatch_firstSnippet);
    }

    //----------------------------------------------------------------------------------//
    //
    //  Actually insert the instrumentation into the procedures
    //
    //----------------------------------------------------------------------------------//
    if(app_thread)
    {
        verbprintf(1, "Beginning insertion set...\n");
        addr_space->beginInsertionSet();
    }

    verbprintf(2, "Beginning loop over modules [instrumentation pass]\n");
    verbprintf(1, "\n");
    for(auto& instr_procedure : instr_procedure_functions)
        instr_procedure();
    verbprintf(1, "\n");

    if(app_thread)
    {
        verbprintf(1, "Finalizing insertion set...\n");
        bool modified = true;
        bool success  = addr_space->finalizeInsertionSet(true, &modified);
        if(!success)
        {
            verbprintf(
                1,
                "Using insertion set failed. Restarting with individual insertion...\n");
            auto _execute_batch = [&instr_procedure_functions, &addr_space](size_t _beg,
                                                                            size_t _end) {
                verbprintf(1, "Instrumenting batch of functions [%lu, %lu)\n",
                           (unsigned long) _beg, (unsigned long) _end);
                addr_space->beginInsertionSet();
                for(size_t i = _beg; i < _end; ++i)
                {
                    if(i < instr_procedure_functions.size())
                        instr_procedure_functions.at(i)();
                }
                bool _modified = true;
                bool _success  = addr_space->finalizeInsertionSet(true, &_modified);
                return _success;
            };

            auto execute_batch = [&_execute_batch,
                                  &instr_procedure_functions](size_t _beg) {
                if(!_execute_batch(_beg, _beg + batch_size))
                {
                    verbprintf(1,
                               "Batch instrumentation of functions [%lu, %lu) failed. "
                               "Beginning non-batched instrumentation for this set\n",
                               (unsigned long) _beg, (unsigned long) _beg + batch_size);
                    for(size_t i = _beg; i < _beg + batch_size; ++i)
                    {
                        if(i < instr_procedure_functions.size())
                            instr_procedure_functions.at(i)();
                    }
                }
                return _beg + batch_size;
            };

            size_t nidx = 0;
            while(nidx < instr_procedure_functions.size())
            {
                nidx = execute_batch(nidx);
            }
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Dump the available instrumented modules/functions (re-dump available)
    //
    //----------------------------------------------------------------------------------//

    for(const auto& itr : available_module_functions)
    {
        _insert_module_function(excluded_module_functions, itr);
    }

    bool _dump_and_exit = ((print_available.length() + print_instrumented.length() +
                            print_overlapping.length() + print_excluded.length()) > 0) ||
                          explicit_dump_and_exit;

    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "available-instr")),
              available_module_functions, 0, werror, "available_module_functions",
              print_formats);
    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "instrumented-instr")),
              instrumented_module_functions, 0, werror, "instrumented_module_functions",
              print_formats);
    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "excluded-instr")),
              excluded_module_functions, 0, werror, "excluded_module_functions",
              print_formats);
    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "overlapping-instr")),
              overlapping_module_functions, 0, werror, "overlapping_module_functions",
              print_formats);

    auto _dump_info = [](const std::string& _label, const string_t& _mode,
                         const fmodset_t& _modset) {
        std::map<std::string, std::vector<std::string>>                  _data{};
        std::unordered_map<std::string, std::unordered_set<std::string>> _dups{};
        auto _insert = [&](const std::string& _m, const std::string& _v) {
            if(_dups[_m].find(_v) == _dups[_m].end())
            {
                _dups[_m].emplace(_v);
                _data[_m].emplace_back(_v);
            }
        };
        if(_mode == "modules")
        {
            for(const auto& itr : _modset)
                _insert(itr.module, TIMEMORY_JOIN("", "[", itr.module, "]"));
        }
        else if(_mode == "functions")
        {
            for(const auto& itr : _modset)
                _insert(itr.module, TIMEMORY_JOIN("", "[", itr.function, "][",
                                                  itr.address_range, "]"));
        }
        else if(_mode == "functions+")
        {
            for(const auto& itr : _modset)
                _insert(itr.module, TIMEMORY_JOIN("", "[", itr.signature.get(), "][",
                                                  itr.address_range, "]"));
        }
        else if(_mode == "pair")
        {
            for(const auto& itr : _modset)
            {
                std::stringstream _ss{};
                _ss << std::boolalpha;
                _ss << "" << itr.module << "] --> [" << itr.function << "]["
                    << itr.address_range << "]";
                _insert(itr.module, _ss.str());
            }
        }
        else if(_mode == "pair+")
        {
            for(const auto& itr : _modset)
            {
                std::stringstream _ss{};
                _ss << std::boolalpha;
                _ss << "[" << itr.module << "] --> [" << itr.signature.get() << "]["
                    << itr.address_range << "]";
                _insert(itr.module, _ss.str());
            }
        }
        else
        {
            errprintf(0, "Unknown mode :: %s\n", _mode.c_str());
        }
        for(auto& mitr : _data)
        {
            if(_mode != "modules" && _mode != "pair" && _mode != "pair+")
                std::cout << "\n[" << _label << "] " << mitr.first << ":\n";
            std::sort(mitr.second.begin(), mitr.second.end());
            for(auto& itr : mitr.second)
            {
                std::cout << "[" << _label << "]    " << itr << "\n";
            }
        }
    };

    if(!print_available.empty())
        _dump_info("available", print_available, available_module_functions);
    if(!print_instrumented.empty())
        _dump_info("instrumented", print_instrumented, instrumented_module_functions);
    if(!print_excluded.empty())
        _dump_info("excluded", print_excluded, excluded_module_functions);
    if(!print_overlapping.empty())
        _dump_info("overlapping", print_overlapping, overlapping_module_functions);

    if(_dump_and_exit) exit(EXIT_SUCCESS);

    //----------------------------------------------------------------------------------//
    //
    //  Either write the instrumented binary or execute the application
    //
    //----------------------------------------------------------------------------------//
    if(binary_rewrite) addr_space->finalizeInsertionSet(false, nullptr);

    int code = -1;
    if(binary_rewrite)
    {
        const auto& outf = outfile;
        if(outf.find('/') != string_t::npos)
        {
            auto outdir = outf.substr(0, outf.find_last_of('/'));
            tim::makedir(outdir);
        }

        bool success = app_binary->writeFile(outfile.c_str());
        code         = (success) ? EXIT_SUCCESS : EXIT_FAILURE;
        if(success)
        {
            verbprintf(0, "\n");
            if(outfile.find('/') != 0)
            {
                verbprintf(0, "The instrumented executable image is stored in '%s/%s'\n",
                           get_cwd().c_str(), outfile.c_str());
            }
            else
            {
                verbprintf(0, "The instrumented executable image is stored in '%s'\n",
                           outfile.c_str());
            }
        }

        if(main_func)
        {
            verbprintf(0, "Getting linked libraries for %s...\n", cmdv0.c_str());
            verbprintf(0, "Consider instrumenting the relevant libraries...\n\n");

            using TIMEMORY_PIPE = tim::popen::TIMEMORY_PIPE;

            tim::set_env("LD_TRACE_LOADED_OBJECTS", "1", 1);
            TIMEMORY_PIPE* ldd = tim::popen::popen(cmdv0.c_str());
            tim::set_env("LD_TRACE_LOADED_OBJECTS", "0", 1);

            strvec_t linked_libraries = tim::popen::read_fork(ldd);

            auto perr = tim::popen::pclose(ldd);
            if(perr != 0) perror("Error in omnitrace_fork");

            for(const auto& itr : linked_libraries)
                printf("\t%s\n", itr.c_str());

            printf("\n");
        }
    }
    else
    {
        verbprintf(0, "Executing...\n");

        if(!app_thread->isTerminated())
        {
            pid_t cpid   = app_thread->getPid();
            int   status = 0;
            app_thread->detach(true);
            do
            {
                pid_t w = waitpid(cpid, &status, WUNTRACED);
                if(w == -1)
                {
                    perror("waitpid");
                    exit(EXIT_FAILURE);
                }

                if(WIFEXITED(status))
                {
                    code = WEXITSTATUS(status);
                }
                else if(WIFSIGNALED(status))
                {
                    code = WTERMSIG(status);
                }
                else if(WIFSTOPPED(status))
                {
                    code = WSTOPSIG(status);
                }
                else if(WIFCONTINUED(status))
                {
                    code = WIFCONTINUED(status);
                }
            } while(!WIFEXITED(status) && !WIFSIGNALED(status));
        }
        else
        {
            if(app_thread->terminationStatus() == ExitedNormally)
            {
                if(app_thread->isTerminated()) verbprintf(0, "End of omnitrace\n");
            }
            else if(app_thread->terminationStatus() == ExitedViaSignal)
            {
                auto sign = app_thread->getExitSignal();
                fprintf(stderr, "\nApplication exited with signal: %i\n", int(sign));
            }
            code = app_thread->getExitCode();
        }
    }

    // cleanup
    for(int i = 0; i < argc; ++i)
        delete[] _argv[i];
    delete[] _argv;
    for(int i = 0; i < _cmdc; ++i)
        delete[] _cmdv[i];
    delete[] _cmdv;

    verbprintf(0, "End of omnitrace\n");
    verbprintf(1, "Exit code: %i\n", code);
    return code;
}

//======================================================================================//

bool
instrument_module(const string_t& file_name)
{
    auto _report = [&file_name](const string_t& _action, const string_t& _reason,
                                int _lvl) {
        static strset_t already_reported{};
        if(already_reported.count(file_name) == 0)
        {
            verbprintf(_lvl, "%s module [%s] : '%s'...\n", _action.c_str(),
                       _reason.c_str(), file_name.c_str());
            already_reported.insert(file_name);
        }
    };

    static std::regex ext_regex{ "\\.(s|S)$", regex_opts };
    static std::regex sys_regex{ "^(s|k|e|w)_[A-Za-z_0-9\\-]+\\.(c|C)$", regex_opts };
    static std::regex sys_build_regex{ "^(\\.\\./sysdeps/|/build/)", regex_opts };
    static std::regex dyninst_regex{ "(dyninst|DYNINST|(^|/)RT[[:graph:]]+\\.c$)",
                                     regex_opts };
    static std::regex dependlib_regex{ "^(lib|)(omnitrace|pthread|caliper|gotcha|papi|"
                                       "cupti|TAU|likwid|pfm|nvperf|unwind)",
                                       regex_opts };
    static std::regex core_cmod_regex{
        "^(malloc|(f|)lock|sig|sem)[a-z_]+(|64|_r|_l)\\.c$"
    };
    static std::regex core_lib_regex{
        "^(lib|)(c|z|rt|dl|dw|util|zstd|elf|pthread|open[\\-]rte|open[\\-]pal|"
        "gcc_s|tcmalloc|profiler|tbbmalloc|tbbmalloc_proxy|event_pthreads|ltdl|"
        "stdc\\+\\+|malloc|selinux|pcre[0-9]+)(-|\\.)",
        regex_opts
    };
    static std::regex prefix_regex{ "^(_|\\.[a-zA-Z0-9])", regex_opts };

    // file extensions that should not be instrumented
    if(std::regex_search(file_name, ext_regex))
    {
        return (_report("Excluding", "file extension", 3), false);
    }

    // system modules that should not be instrumented (wastes time)
    if(std::regex_search(file_name, sys_regex) ||
       std::regex_search(file_name, sys_build_regex))
    {
        return (_report("Excluding", "system module", 3), false);
    }

    // dyninst modules that must not be instrumented
    if(std::regex_search(file_name, dyninst_regex))
    {
        return (_report("Excluding", "dyninst module", 3), false);
    }

    // modules used by omnitrace and dependent libraries
    if(std::regex_search(file_name, core_lib_regex) ||
       std::regex_search(file_name, core_cmod_regex))
    {
        return (_report("Excluding", "core module", 3), false);
    }

    // modules used by omnitrace and dependent libraries
    if(std::regex_search(file_name, dependlib_regex))
    {
        return (_report("Excluding", "dependency module", 3), false);
    }

    // known set of modules whose starting sequence of characters suggest it should not be
    // instrumented (wastes time)
    if(std::regex_search(file_name, prefix_regex))
    {
        return (_report("Excluding", "prefix match", 3), false);
    }

    _report("Including", "no constraint", 2);

    return true;
}

//======================================================================================//

extern const strset_t exclude_function_names;
bool
instrument_entity(const string_t& function_name)
{
    auto _report = [&function_name](const string_t& _action, const string_t& _reason,
                                    int _lvl) {
        static strset_t already_reported{};
        if(already_reported.count(function_name) == 0)
        {
            verbprintf(_lvl, "%s function [%s] : '%s'...\n", _action.c_str(),
                       _reason.c_str(), function_name.c_str());
            already_reported.insert(function_name);
        }
    };

    static std::regex exclude(
        "(omnitrace|tim::|N3tim|MPI_Init|MPI_Finalize|::__[A-Za-z]|"
        "dyninst|tm_clones|malloc$|calloc$|free$|realloc$|std::addressof)",
        regex_opts);
    static std::regex exclude_cxx("(std::_Sp_counted_base|std::use_facet)", regex_opts);
    static std::regex leading(
        "^(_|\\.|frame_dummy|\\(|targ|new|delete|operator new|operator delete|"
        "std::allocat|nvtx|gcov|TAU|tau|Tau|dyn|RT|sys|pthread|posix|clone|"
        "virtual thunk|non-virtual thunk|transaction clone|"
        "RtsLayer|DYNINST|PthreadLayer|threaded_func|PMPI|"
        "Kokkos::Impl::|Kokkos::Experimental::Impl::|Kokkos::impl_|"
        "Kokkos::[A-Za-z]+::impl_|Kokkos::Tools::|Kokkos::Profiling::)",
        regex_opts);
    static std::regex trailing("(\\.part\\.[0-9]+|\\.constprop\\.[0-9]+|\\.|\\.[0-9]+)$",
                               regex_opts);
    static strset_t   whole = []() {
        auto _v   = get_whole_function_names();
        auto _ret = _v;
        for(std::string _ext : { "64", "_l", "_r" })
            for(const auto& itr : _v)
                _ret.emplace(itr + _ext);
        return _ret;
    }();

    // don't instrument the functions when key is found anywhere in function name
    if(std::regex_search(function_name, exclude) ||
       std::regex_search(function_name, exclude_cxx))
    {
        _report("critical", function_name, 3);
        return false;
    }

    if(whole.count(function_name) > 0)
    {
        _report("critical", function_name, 3);
        return false;
    }

    // don't instrument the functions when key is found at the start of the function name
    if(std::regex_search(function_name, leading))
    {
        _report("recommended", function_name, 3);
        return false;
    }

    // don't instrument the functions when key is found at the end of the function name
    if(std::regex_search(function_name, trailing))
    {
        _report("recommended", function_name, 3);
        return false;
    }

    _report("Including function [no constraint] : '%s'...\n", function_name, 3);

    return true;
}

//======================================================================================//
// query_instr -- check whether there are one or more instrumentation points
//
bool
query_instr(procedure_t* funcToInstr, procedure_loc_t traceLoc, flow_graph_t* cfGraph,
            basic_loop_t* loopToInstrument, bool allow_traps)
{
    module_t* module = funcToInstr->getModule();
    if(!module) return false;

    bpvector_t<point_t*>* _points = nullptr;

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

    size_t _n = _points->size();
    for(auto& itr : *_points)
    {
        if(!itr)
            --_n;
        else if(itr && !allow_traps && itr->usesTrap_NP())
            --_n;
    }

    return (_n > 0);
}

//======================================================================================//
// insert_instr -- generic insert instrumentation function
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

    /*if(loop_level_instr)
    {
        flow_graph_t*                     flow = funcToInstr->getCFG();
        bpvector_t<basic_loop_t*> basicLoop;
        flow->getOuterLoops(basicLoop);
        for(auto litr = basicLoop.begin(); litr != basicLoop.end(); ++litr)
        {
            bpvector_t<point_t*>* _tmp;
            if(traceLoc == BPatch_entry)
                _tmp = cfGraph->findLoopInstPoints(BPatch_locLoopEntry, *litr);
            else if(traceLoc == BPatch_exit)
                _tmp = cfGraph->findLoopInstPoints(BPatch_locLoopExit, *litr);
            if(!_tmp)
                continue;
            for(auto& itr : *_tmp)
                _points->push_back(itr);
        }
    }*/

    // verbprintf(0, "Instrumenting |> [ %s ]\n", name.m_name.c_str());

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
        // else if(traceLoc == BPatch_exit)
        //    mutatee->insertSnippet(*_trace, *itr, BPatch_callAfter,
        //    BPatch_firstSnippet);
        else
            mutatee->insertSnippet(*_trace, *itr);
        ++_n;
    }

    return (_n > 0);
}

//======================================================================================//
// Constraints for instrumentation. Returns true for those modules that
// shouldn't be instrumented.
bool
module_constraint(char* fname)
{
    // fname is the name of module/file
    string_t _fname = fname;

    // never instrumentat any module matching omnitrace
    if(_fname.find("omnitrace") != string_t::npos) return true;

    // always instrument these modules
    if(_fname == "DEFAULT_MODULE" || _fname == "LIBRARY_MODULE") return false;

    if(instrument_module(_fname)) return false;

    // do not instrument
    return true;
}

//======================================================================================//
// Constraint for routines. The constraint returns true for those routines that
// should not be instrumented.
bool
routine_constraint(const char* fname)
{
    string_t _fname = fname;
    if(_fname.find("omnitrace") != string_t::npos) return true;

    auto npos = std::string::npos;
    if(_fname.find("FunctionInfo") != npos || _fname.find("_L_lock") != npos ||
       _fname.find("_L_unlock") != npos)
        return true;  // Don't instrument
    else
    {
        // Should the routine fname be instrumented?
        if(instrument_entity(string_t(fname)))
        {
            // Yes it should be instrumented. Return false
            return false;
        }
        else
        {
            // No. The selective instrumentation file says: don't instrument it
            return true;
        }
    }
}

namespace
{
//======================================================================================//
//
std::string
get_absolute_exe_filepath(std::string exe_name, const std::string& env_path)
{
    if(!exe_name.empty() && !file_exists(exe_name))
    {
        auto _exe_orig = exe_name;
        auto _paths    = tim::delimit(tim::get_env<std::string>(env_path, ""), ":");
        for(auto& pitr : _paths)
        {
            if(file_exists(TIMEMORY_JOIN('/', pitr, exe_name)))
            {
                exe_name = get_realpath(TIMEMORY_JOIN('/', pitr, exe_name));
                verbprintf(0, "Resolved '%s' to '%s'...\n", _exe_orig.c_str(),
                           exe_name.c_str());
                break;
            }
        }

        if(!file_exists(exe_name))
        {
            verbprintf(0, "Warning! File path to '%s' could not be determined...\n",
                       exe_name.c_str());
        }
    }
    else if(!exe_name.empty())
    {
        return get_realpath(exe_name);
    }

    return exe_name;
}

//======================================================================================//
//
std::string
get_absolute_lib_filepath(std::string lib_name, const std::string& env_path)
{
    if(!lib_name.empty() && (!file_exists(lib_name) ||
                             std::regex_match(lib_name, std::regex("^[A-Za-z0-9].*"))))
    {
        auto _lib_orig = lib_name;
        auto _paths    = tim::delimit(
            std::string{ ".:" } + tim::get_env<std::string>(env_path, ""), ":");
        for(auto& pitr : _paths)
        {
            if(file_exists(TIMEMORY_JOIN('/', pitr, lib_name)))
            {
                lib_name = get_realpath(TIMEMORY_JOIN('/', pitr, lib_name));
                verbprintf(0, "Resolved '%s' to '%s'...\n", _lib_orig.c_str(),
                           lib_name.c_str());
                break;
            }
        }

        if(!file_exists(lib_name))
        {
            verbprintf(0, "Warning! File path to '%s' could not be determined...\n",
                       lib_name.c_str());
        }
    }
    else if(!lib_name.empty())
    {
        return get_realpath(lib_name);
    }

    return lib_name;
}

//======================================================================================//
//
bool
file_exists(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

std::string
get_realpath(const std::string& _f)
{
    char _buffer[PATH_MAX];
    if(!::realpath(_f.c_str(), _buffer))
    {
        verbprintf(2, "Warning! realpath could not be found for %s\n", _f.c_str());
        return _f;
    }
    return std::string{ _buffer };
}

std::string
get_cwd()
{
    char cwd[PATH_MAX];
    return std::string{ getcwd(cwd, PATH_MAX) };
}
}  // namespace
