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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
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
bool                                       stl_func_instr          = false;
bool                                       use_mpi                 = false;
bool                                       is_static_exe           = false;
bool                                       is_driver               = false;
bool                                       allow_overlapping       = false;
bool                                       instr_dynamic_callsites = false;
bool                                       instr_traps             = false;
bool                                       instr_loop_traps        = false;
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
string_t                                   print_available         = {};
string_t                                   print_overlapping       = {};
std::string                                modfunc_dump_dir        = {};
auto regex_opts = std::regex_constants::egrep | std::regex_constants::optimize;
}  // namespace

std::string
get_absolute_exe_filepath(std::string exe_name);

std::string
get_absolute_lib_filepath(std::string lib_name);

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
        auto file_exists = [](const std::string& _fname) {
            struct stat _buffer;
            if(stat(_fname.c_str(), &_buffer) == 0)
                return (S_ISREG(_buffer.st_mode) != 0 || S_ISLNK(_buffer.st_mode) != 0);
            return false;
        };
        if(file_exists(itr))
            tim::set_env<string_t>("DYNINSTAPI_RT_LIB", itr, 0);
        else if(file_exists(TIMEMORY_JOIN('/', itr, "libdyninstAPI_RT.so")))
            tim::set_env<string_t>("DYNINSTAPI_RT_LIB",
                                   TIMEMORY_JOIN('/', itr, "libdyninstAPI_RT.so"), 0);
        else if(file_exists(TIMEMORY_JOIN('/', itr, "libdyninstAPI_RT.a")))
            tim::set_env<string_t>("DYNINSTAPI_RT_LIB",
                                   TIMEMORY_JOIN('/', itr, "libdyninstAPI_RT.a"), 0);
    }
    verbprintf(0, "[omnitrace][exe] DYNINST_API_RT: %s\n",
               tim::get_env<string_t>("DYNINSTAPI_RT_LIB", "").c_str());

    argv0 = argv[0];

    bpatch                              = std::make_shared<patch_t>();
    address_space_t*      addr_space    = nullptr;
    string_t              mutname       = {};
    string_t              outfile       = {};
    std::vector<string_t> inputlib      = { "libomnitrace" };
    std::vector<string_t> libname       = {};
    std::vector<string_t> sharedlibname = {};
    std::vector<string_t> staticlibname = {};
    tim::process::id_t    _pid          = -1;

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

    auto cmd_string = [](int _ac, char** _av) {
        stringstream_t ss;
        for(int i = 0; i < _ac; ++i)
            ss << _av[i] << " ";
        return ss.str();
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

    std::stringstream jump_description;
    jump_description
        << "Instrument with function pointers in OMNITRACE_JUMP_LIBRARY (default: "
        << tim::get_env<string_t>("OMNITRACE_JUMP_LIBRARY", "jump/libomnitrace.so")
        << ")";

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
        .add_argument(
            { "--print-dir" },
            "Output directory for diagnostic available/instrumented/overlapping module "
            "function lists, e.g. {print-dir}/available.txt")
        .count(1)
        .dtype("string")
        .action([](parser_t& p) { modfunc_dump_dir = p.get<std::string>("print-dir"); });
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
                      "Enable generation of a new executable (binary-rewrite)")
        .count(1)
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
    parser.add_argument({ "-j", "--jump" }, jump_description.str())
        .dtype("boolean")
        .max_count(1)
        .action([&inputlib](parser_t& p) {
            if(p.get<bool>("jump"))
            {
                for(auto& itr : inputlib)
                    itr += "-jump";
            }
        });
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
    parser.add_argument({ "-I", "-R", "--function-include" },
                        "Regex for selecting functions");
    parser.add_argument({ "-E", "--function-exclude" }, "Regex for excluding functions");
    parser.add_argument({ "-MI", "-MR", "--module-include" },
                        "Regex for selecting modules/files/libraries");
    parser.add_argument({ "-ME", "--module-exclude" },
                        "Regex for excluding modules/files/libraries");
    parser
        .add_argument({ "-S", "--stdlib" },
                      "Enable instrumentation of C++ standard library functions.")
        .max_count(1)
        .action([](parser_t& p) { stl_func_instr = p.get<bool>("stdlib"); });

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
    parser
        .add_argument({ "--dynamic-callsites" },
                      "Force instrumentation if a function has dynamic callsites (e.g. "
                      "function pointers)")
        .max_count(1)
        .dtype("boolean")
        .action([](parser_t& p) {
            instr_dynamic_callsites = p.get<bool>("dynamic-callsites");
        });
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
            "Dyninst supports batch insertion of multiple points. If one large batch "
            "insertion fails, this value will be used to create smaller batches")
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
    //  Helper function for adding regex expressions
    //
    auto add_regex = [](auto& regex_array, const string_t& regex_expr) {
        if(!regex_expr.empty())
            regex_array.emplace_back(std::regex(regex_expr, regex_opts));
    };

    add_regex(func_include, tim::get_env<string_t>("OMNITRACE_REGEX_INCLUDE", ""));
    add_regex(func_exclude, tim::get_env<string_t>("OMNITRACE_REGEX_EXCLUDE", ""));

    if(parser.exists("R"))
    {
        auto keys = parser.get<strvec_t>("R");
        for(const auto& itr : keys)
            add_regex(func_include, itr);
    }

    if(parser.exists("E"))
    {
        auto keys = parser.get<strvec_t>("E");
        for(const auto& itr : keys)
            add_regex(func_exclude, itr);
    }

    if(parser.exists("MI"))
    {
        auto keys = parser.get<strvec_t>("MI");
        for(const auto& itr : keys)
            add_regex(file_include, itr);
    }

    if(parser.exists("ME"))
    {
        auto keys = parser.get<strvec_t>("ME");
        for(const auto& itr : keys)
            add_regex(file_exclude, itr);
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
        fprintf(stderr, "[omnitrace][exe] Error! address space for dynamic "
                        "instrumentation was not created\n");
        exit(EXIT_FAILURE);
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

    auto _add_overlapping = [](module_t* mitr, procedure_t* pitr) {
        if(!pitr->isInstrumentable()) return;
        std::vector<procedure_t*> _overlapping{};
        if(pitr->findOverlapping(_overlapping))
        {
            overlapping_module_functions.insert(module_function{ mitr, pitr });
            for(auto* oitr : _overlapping)
            {
                if(!oitr->isInstrumentable()) continue;
                overlapping_module_functions.insert(
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
                    available_module_functions.insert(std::move(_modfn));
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
                available_module_functions.insert(std::move(_modfn));
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
                            TIMEMORY_JOIN("", _output_prefix, "available-instr.txt")),
              available_module_functions, 1, werror);
    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "overlapping-instr.txt")),
              overlapping_module_functions, 1, werror);

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
        fprintf(stderr, "No application thread or binary!...\n");
        throw std::runtime_error("Nullptr to BPatch_binaryEdit* and BPatch_process*");
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
            verbprintf(1, "loadLibrary(%s) result = %s\n", _libname.c_str(),
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
        fprintf(stderr, "[omnitrace][exe] Couldn't find '%s'\n", main_fname.c_str());
        if(!_mutatee_init || !_mutatee_fini)
        {
            fprintf(stderr, "[omnitrace][exe] Couldn't find '%s' or '%s', aborting\n",
                    "_init", "_fini");
            throw std::runtime_error("Could not find main function");
        }
        else
        {
            fprintf(stderr, "[omnitrace][exe] using '%s' and '%s' in lieu of '%s'...",
                    "_init", "_fini", main_fname.c_str());
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
            stringstream_t ss;
            ss << "Error! Couldn't find '" << itr.second.c_str() << "' function";
            fprintf(stderr, "[omnitrace][exe] %s\n", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }

    if(use_mpi && !(mpi_func || (mpi_init_func && mpi_fini_func)))
    {
        throw std::runtime_error("MPI support was requested but omnitrace was not built "
                                 "with MPI and GOTCHA support");
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

    auto main_call_args = omnitrace_call_expr(main_sign.get());
    auto init_call_args = omnitrace_call_expr(instr_mode, binary_rewrite, cmdv0);
    auto fini_call_args = omnitrace_call_expr();
    auto umpi_call_args = omnitrace_call_expr(use_mpi, is_attached);
    auto none_call_args = omnitrace_call_expr();

    verbprintf(2, "Done\n");
    verbprintf(2, "Getting call snippets... ");

    auto init_call = init_call_args.get(init_func);
    auto fini_call = fini_call_args.get(fini_func);
    auto umpi_call = umpi_call_args.get(mpi_func);

    auto main_beg_call = main_call_args.get(entr_trace);
    auto main_end_call = main_call_args.get(exit_trace);

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
        _libname = get_absolute_lib_filepath(itr);
    if(_libname.empty()) _libname = "libomnitrace.so";

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
            std::cerr << "Error! environment variable: " << itr
                      << " not in form VARIABLE=VALUE\n";
            throw std::runtime_error("Bad format");
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

    if(binary_rewrite)
    {
        verbprintf(2, "Adding main begin and end snippets...\n");
        if(main_beg_call) init_names.emplace_back(main_beg_call.get());
        if(main_end_call) fini_names.emplace_back(main_end_call.get());
    }
    else if(app_thread)
    {
        verbprintf(2, "Patching main function\n");
        if(main_beg_call)
            insert_instr(addr_space, main_func, main_beg_call, BPatch_entry, nullptr,
                         nullptr);
        if(main_end_call)
            insert_instr(addr_space, main_func, main_end_call, BPatch_exit, nullptr,
                         nullptr);
    }
    else
    {
        verbprintf(0, "No binary_rewrite and no app_thread!...\n");
    }

    if(fini_call) fini_names.emplace_back(fini_call.get());

    for(const auto& itr : end_expr)
    {
        if(itr.second) fini_names.emplace_back(itr.second.get());
    }

    //----------------------------------------------------------------------------------//
    //
    //  Lambda for instrumenting procedures. The first pass (usage_pass = true) will
    //  generate the hash_ids for each string so that these can be inserted in bulk
    //  with one operation and do not have to be calculated during runtime.
    //
    //----------------------------------------------------------------------------------//
    std::vector<std::function<void()>> instr_procedure_functions;
    auto instr_procedures = [&](const procedure_vec_t& procedures) {
        verbprintf(3, "Instrumenting %lu procedures...\n",
                   (unsigned long) procedures.size());
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

            if(itr == main_func && main_func->isInstrumentable())
            {
                hash_ids.emplace_back(std::hash<string_t>()(main_sign.get()),
                                      main_sign.get());
                auto main_mf = module_function{ modname, fname, main_sign, itr };
                available_module_functions.insert(main_mf);
                instrumented_module_functions.insert(main_mf);
                continue;
            }

            if(!itr->isInstrumentable())
            {
                verbprintf(2, "Skipping uninstrumentable function: %s\n", fname);
                continue;
            }

            if(std::string{ modname }.find("libdyninst") != std::string::npos) continue;

            if(module_constraint(modname)) continue;
            if(!instrument_module(modname)) continue;

            auto name = get_func_file_line_info(mod, itr);

            if(name.get().empty())
            {
                verbprintf(2, "Skipping function [empty name]: %s\n", fname);
                continue;
            }

            if(routine_constraint(name.m_name.c_str())) continue;
            if(!instrument_entity(name.m_name)) continue;

            if(is_static_exe && has_debug_info && string_t{ fname } == "_fini" &&
               string_t{ modname } == "DEFAULT_MODULE")
            {
                verbprintf(2, "Skipping function [DEFAULT_MODULE]: %s\n", fname);
                continue;
            }

            _add_overlapping(mod, itr);

            if(!allow_overlapping &&
               overlapping_module_functions.find(module_function{ mod, itr }) !=
                   overlapping_module_functions.end())
            {
                verbprintf(2, "Skipping function [overlapping]: %s / %s\n",
                           name.m_name.c_str(), name.get().c_str());
                continue;
            }

            // directly try to get loop entry points
            const std::vector<point_t*>* _loop_entries =
                itr->findPoint(BPatch_locLoopEntry);

            // try to get loops via the control flow graph
            flow_graph_t*    cfg = itr->getCFG();
            basic_loop_vec_t basic_loop{};
            if(cfg) cfg->getOuterLoops(basic_loop);

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
            auto _skip_loop_range =
                (_has_loop_entries) ? (_address_range < min_loop_address_range) : false;

            if(_force_instr && _skip_range)
            {
                verbprintf(
                    1,
                    "Instrumenting function [dynamic-callsite]: %s / %s despite not "
                    "satisfy minimum address range (address range = %lu, minimum "
                    "= %lu) because contains dynamic callsites\n",
                    name.m_name.c_str(), name.get().c_str(),
                    (unsigned long) _address_range, (unsigned long) min_address_range);
            }
            else if(_force_instr && _skip_loop_range)
            {
                verbprintf(
                    1,
                    "Instrumenting function [dynamic-callsite]: %s / %s despite not "
                    "satisfy minimum loop address range (address range = %lu, minimum "
                    "= %lu) because contains dynamic callsites\n",
                    name.m_name.c_str(), name.get().c_str(),
                    (unsigned long) _address_range,
                    (unsigned long) min_loop_address_range);
            }
            else if(_skip_range)
            {
                verbprintf(1,
                           "Skipping function [min-address-range]: %s / %s (address "
                           "range = %lu, minimum = %lu)\n",
                           name.m_name.c_str(), name.get().c_str(),
                           (unsigned long) _address_range,
                           (unsigned long) min_address_range);
                continue;
            }
            else if(_skip_loop_range)
            {
                verbprintf(1,
                           "Skipping function [min-loop-address-range]: %s / %s (address "
                           "range = %lu, minimum = %lu)\n",
                           name.m_name.c_str(), name.get().c_str(),
                           (unsigned long) _address_range,
                           (unsigned long) min_loop_address_range);
                continue;
            }

            bool _entr_success =
                query_instr(itr, BPatch_entry, nullptr, nullptr, instr_traps);
            bool _exit_success =
                query_instr(itr, BPatch_exit, nullptr, nullptr, instr_traps);
            if(!_entr_success && !_exit_success)
            {
                verbprintf(2,
                           "Skipping function [insert-instr]: %s / %s. Either no entry "
                           "instrumentation points were found or instrumentation "
                           "required traps and instrumenting via traps were disabled.\n",
                           name.m_name.c_str(), name.get().c_str());
                continue;
            }
            else if(_entr_success && !_exit_success)
            {
                verbprintf(2,
                           "Skipping function [insert-instr]: %s / %s. Function can be "
                           "only partially instrumented: entry = %s, exit = %s\n",
                           name.m_name.c_str(), name.get().c_str(),
                           _entr_success ? "y" : "n", _exit_success ? "y" : "n");
                continue;
            }

            hash_ids.emplace_back(std::hash<string_t>()(name.get()), name.get());
            available_module_functions.insert(module_function{ mod, itr });
            instrumented_module_functions.insert(module_function{ mod, itr });

            auto _f = [=]() {
                verbprintf(1, "Instrumenting |> [ %s ] -> [ %s ]\n", modname,
                           name.m_name.c_str());
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

            if(loop_level_instr)
            {
                verbprintf(1, "Instrumenting at the loop level: %s\n",
                           name.m_name.c_str());

                for(auto* litr : basic_loop)
                {
                    bool _lentr_success =
                        query_instr(itr, BPatch_entry, cfg, litr, instr_loop_traps);
                    bool _lexit_success =
                        query_instr(itr, BPatch_exit, cfg, litr, instr_loop_traps);
                    if(!_lentr_success && !_lexit_success)
                    {
                        verbprintf(
                            2,
                            "Skipping function [insert-instr-loop]: %s / %s. Either no "
                            "entry instrumentation points were found or instrumentation "
                            "required traps and instrumenting via traps were disabled.\n",
                            name.m_name.c_str(), name.get().c_str());
                        continue;
                    }
                    else if(_lentr_success && !_lexit_success)
                    {
                        verbprintf(
                            2,
                            "Skipping function [insert-instr-loop]: %s / %s. Function "
                            "can be only partially instrumented: entry = %s, exit = %s\n",
                            name.m_name.c_str(), name.get().c_str(),
                            _lentr_success ? "y" : "n", _lexit_success ? "y" : "n");
                        continue;
                    }

                    auto lname  = get_loop_file_line_info(mod, itr, cfg, litr);
                    auto _lname = lname.get();
                    auto _lhash = std::hash<string_t>()(_lname);
                    hash_ids.emplace_back(_lhash, _lname);
                    auto _lf = [=]() {
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
                }
            }
        }
    };

    //----------------------------------------------------------------------------------//
    //
    //  Do a first pass through all procedures to generate the hash ids
    //
    //----------------------------------------------------------------------------------//

    if(instr_mode == "trace")
    {
        verbprintf(2, "Beginning loop over modules [hash id generation pass]\n");
        for(auto& m : modules)
        {
            char modname[1024];
            m->getName(modname, 1024);
            if(strstr(modname, "libdyninst") != nullptr) continue;

            if(!m->getProcedures())
            {
                verbprintf(1, "Skipping module w/ no procedures: '%s'\n", modname);
                continue;
            }

            verbprintf(3, "Parsing module: %s\n", modname);
            bpvector_t<procedure_t*>* p = m->getProcedures();
            if(!p) continue;

            instr_procedures(*p);
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

    if(app_thread)
    {
        assert(app_thread != nullptr);
        verbprintf(1, "Executing initial snippets...\n");
        for(auto* itr : init_names)
            app_thread->oneTimeCode(*itr);
    }
    else
    {
        verbprintf(1, "Adding main entry snippets...\n");
        if(main_entr_points)
            addr_space->insertSnippet(BPatch_sequence(init_names), *main_entr_points,
                                      BPatch_callBefore, BPatch_firstSnippet);
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
    for(auto& instr_procedure : instr_procedure_functions)
        instr_procedure();

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

    bool _dump_and_exit = ((print_available.length() + print_instrumented.length() +
                            print_overlapping.length()) > 0);

    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "available-instr.txt")),
              available_module_functions, 0, werror);
    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "instrumented-instr.txt")),
              instrumented_module_functions, 0, werror);
    dump_info(TIMEMORY_JOIN('/', modfunc_dump_dir,
                            TIMEMORY_JOIN("", _output_prefix, "overlapping-instr.txt")),
              overlapping_module_functions, 0, werror);

    auto _dump_info = [](const string_t& _mode, const fmodset_t& _modset) {
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
                _insert(itr.module, itr.module);
        }
        else if(_mode == "functions")
        {
            for(const auto& itr : _modset)
                _insert(itr.module, itr.function);
        }
        else if(_mode == "functions+")
        {
            for(const auto& itr : _modset)
                _insert(itr.module, itr.signature.get());
        }
        else if(_mode == "pair")
        {
            for(const auto& itr : _modset)
            {
                std::stringstream _ss{};
                _ss << std::boolalpha;
                _ss << "[" << itr.module << "] --> [ " << itr.address_range << " ]["
                    << itr.function << "]";
                _insert(itr.module, _ss.str());
            }
        }
        else if(_mode == "pair+")
        {
            for(const auto& itr : _modset)
            {
                std::stringstream _ss{};
                _ss << std::boolalpha;
                _ss << "[" << itr.module << "] --> [ " << itr.address_range << " ]["
                    << itr.signature.get() << "]";
                _insert(itr.module, _ss.str());
            }
        }
        else
        {
            throw std::runtime_error("Unknown mode " + _mode);
        }
        for(auto& mitr : _data)
        {
            if(_mode != "modules") std::cout << "\n" << mitr.first << ":\n";
            for(auto& itr : mitr.second)
            {
                std::cout << "    " << itr << "\n";
            }
        }
    };

    if(!print_available.empty()) _dump_info(print_available, available_module_functions);
    if(!print_instrumented.empty())
        _dump_info(print_instrumented, instrumented_module_functions);
    if(!print_overlapping.empty())
        _dump_info(print_overlapping, overlapping_module_functions);

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
            if(outfile.find('/') != 0)
            {
                char  cwd[FUNCNAMELEN];
                auto* ret = getcwd(cwd, FUNCNAMELEN);
                consume_parameters(ret);
                printf("\nThe instrumented executable image is stored in '%s/%s'\n", cwd,
                       outfile.c_str());
            }
            else
                printf("\nThe instrumented executable image is stored in '%s'\n",
                       outfile.c_str());
        }

        if(main_func)
        {
            printf("[omnitrace][exe] Getting linked libraries for %s...\n",
                   cmdv0.c_str());
            printf("[omnitrace][exe] Consider instrumenting the relevant "
                   "libraries...\n\n");

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
            app_thread->detach(true);
            pid_t cpid = app_thread->getPid();
            pid_t w;
            int   status = 0;
            do
            {
                w = waitpid(cpid, &status, WUNTRACED);
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

    auto is_include = [&](bool _if_empty) {
        if(file_include.empty()) return _if_empty;
        // NOLINTNEXTLINE(readability-use-anyofallof)
        for(auto& itr : file_include)
        {
            if(std::regex_search(file_name, itr)) return true;
        }
        return false;
    };

    auto is_exclude = [&]() {
        // NOLINTNEXTLINE(readability-use-anyofallof)
        for(auto& itr : file_exclude)
        {
            if(std::regex_search(file_name, itr)) return true;
        }
        return false;
    };

    auto _user_include = is_include(false);
    auto _user_exclude = is_exclude();

    if(_user_include && !_user_exclude)
        return (_report("Including", "user-regex", 2), true);

    string_t          ext_str = "\\.(s|S)$";
    static std::regex ext_regex(ext_str, regex_opts);
    static std::regex sys_regex("^(s|k|e|w)_[A-Za-z_0-9\\-]+\\.(c|C)$", regex_opts);
    static std::regex userlib_regex(
        "^(lib|)(omnitrace|caliper|gotcha|papi|cupti|TAU|likwid|dyninst|pfm|nvtx|upcxx|"
        "nvperf|hsa|amdhip64|pthread|sem_|malloc|RT[A-Za-z_0-9\\-]+\\.c$|\\.\\./sysdeps/"
        "|/build/)",
        regex_opts);
    static std::regex corelib_regex(
        "^lib(c|z|rt|dl|dw|util|zstd|elf|pthread|open[\\-]rte|open[\\-]pal|"
        "hwloc|numa|event|udev|dyninstAPI_RT|gcc_s|tcmalloc|profiler|tbbmalloc|"
        "tbbmalloc_proxy|event_pthreads|ltdl)(-|\\.)",
        regex_opts);
    // these are all due to TAU
    static std::regex prefix_regex(
        "^(_|\\.[a-zA-Z0-9]|RT|Tau|Profiler|Rts|Papi|Py|Comp_xl\\.cpp|Comp_gnu\\.cpp|"
        "UserEvent\\.cpp|FunctionInfo\\.cpp|PthreadLayer\\.cpp|"
        "Comp_intel[0-9]\\.cpp|Tracer\\.cpp)",
        regex_opts);
    static std::regex suffix_regex(
        "(printf|gettext|^sig[a-z]+|^exit|^setenv|on_exit|quick_exit|_crypt|^str[a-z_]+|"
        "mmap[0-9]+|^err|getu[a-z]+|^call_once|^sendto|^timer_[a-z]+|^read|^close|^recv|^"
        "lseek[0-9]+|^open[a-z0-9]+|^nlist|^fclrexcpt|^conj[a-z0-9]*|^cimag[a-"
        "z0-9]*|^creal[a-z0-9]*|^cabs[a-z0-9]*|^wmem[a-z_]+|^mem[a-z_]+|^asctime|time|"
        "timeofday|timespec_get|locale|^abort|scanf|tmpfile|getline|fseek|putc|rewind|"
        "vscanf|memmove|uid|tsz|gid|cvt|cvt_r|^error|_r|[a-z]64|^f[a-z]+|^makecontext|^"
        "basename|^wcp[a-z]+|[a-z]+dir|^mb[a-z]+|^dir[a-z]+|euid[a-z]+|^c[36][24][a-z]+|^"
        "set[a-z_]+|^get[a-z_]+|^shm[a-z]+|^wc[a-z_]+|brk|^write[a-z]+|RTcommon|"
        "RTfreebsd|RTheap|RTheap-freebsd|RTheap-linux|RTheap-win|RTlinux|RTmemEmulator|"
        "RTposix|RTsignal|RTstatic_ctors_dtors-aarch64|RTstatic_ctors_dtors_begin|"
        "RTstatic_ctors_dtors_end|RTstatic_ctors_dtors-ppc32|RTstatic_ctors_dtors-ppc64|"
        "RTstatic_ctors_dtors-x86|RTthread-aarch64|RTthread|RTthread-powerpc|RTthread-"
        "x86-64|RTthread-x86|RTwinnt|unwind)\\.c$",
        regex_opts);

    if(std::regex_search(file_name, ext_regex))
    {
        return (_report("Excluding", "file extension", 3), false);
    }

    if(std::regex_search(file_name, sys_regex))
    {
        return (_report("Excluding", "system library", 3), false);
    }

    if(std::regex_search(file_name, corelib_regex))
    {
        return (_report("Excluding", "core library", 3), false);
    }

    if(std::regex_search(file_name, userlib_regex))
    {
        return (_report("Excluding", "instrumentation", 3), false);
    }

    if(std::regex_search(file_name, prefix_regex))
    {
        return (_report("Excluding", "prefix match", 3), false);
    }

    if(std::regex_search(file_name, suffix_regex))
    {
        return (_report("Excluding", "suffix match", 3), false);
    }

    if(_user_exclude) return (_report("Excluding", "user-regex", 2), false);

    _report("Including", "no constraint", 2);

    return true;
}

//======================================================================================//

extern const strset_t exclude_function_names;
bool
instrument_entity(const string_t& function_name)
{
    auto is_include = [&](bool _if_empty) {
        if(func_include.empty()) return _if_empty;
        // NOLINTNEXTLINE(readability-use-anyofallof)
        for(auto& itr : func_include)
        {
            if(std::regex_search(function_name, itr)) return true;
        }
        return false;
    };

    auto is_exclude = [&]() {
        // NOLINTNEXTLINE(readability-use-anyofallof)
        for(auto& itr : func_exclude)
        {
            if(std::regex_search(function_name, itr))
            {
                verbprintf(2, "Excluding function [user-regex] : '%s'...\n",
                           function_name.c_str());
                return true;
            }
        }
        return false;
    };

    auto _user_include = is_include(false) && !is_exclude();

    if(_user_include)
    {
        verbprintf(2, "Including function [user-regex] : '%s'...\n",
                   function_name.c_str());
        return true;
    }

    static std::regex exclude(
        "(omnitrace|tim::|cereal|N3tim|MPI_Init|MPI_Finalize|::__[A-Za-z]|"
        "dyninst|tm_clones|malloc$|calloc$|free$|realloc$|std::addressof)",
        regex_opts);
    static std::regex exclude_cxx("(std::_Sp_counted_base|std::use_facet)", regex_opts);
    static std::regex leading(
        "^(_|\\.|frame_dummy|\\(|targ|new|delete|operator new|operator delete|"
        "std::allocat|nvtx|gcov|main\\.cold|TAU|tau|Tau|dyn|RT|"
        "sys|pthread|posix|clone|"
        "virtual thunk|non-virtual thunk|transaction clone|"
        "RtsLayer|DYNINST|PthreadLayer|threaded_func|PMPI|"
        "Kokkos::Impl::|Kokkos::Experimental::Impl::|Kokkos::impl_|"
        "Kokkos::[A-Za-z]+::impl_|Kokkos::Tools::|Kokkos::Profiling::)",
        regex_opts);
    static std::regex trailing("(\\.part\\.[0-9]+|\\.constprop\\.[0-9]+|\\.|\\.[0-9]+)$",
                               regex_opts);
    static std::regex stlfunc("^std::", regex_opts);
    static strset_t   whole = get_whole_function_names();

    if(!stl_func_instr && std::regex_search(function_name, stlfunc))
    {
        verbprintf(3, "Excluding function [stl] : '%s'...\n", function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found anywhere in function name
    if(std::regex_search(function_name, exclude))
    {
        verbprintf(3, "Excluding function [critical, any match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found anywhere in function name
    if(std::regex_search(function_name, exclude_cxx))
    {
        verbprintf(3, "Excluding function [critical_cxx, any match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found at the start of the function name
    if(std::regex_search(function_name, leading))
    {
        verbprintf(3, "Excluding function [critical, leading match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    // don't instrument the functions when key is found at the end of the function name
    if(std::regex_search(function_name, trailing))
    {
        verbprintf(3, "Excluding function [critical, trailing match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    if(whole.count(function_name) > 0)
    {
        verbprintf(3, "Excluding function [critical, whole match] : '%s'...\n",
                   function_name.c_str());
        return false;
    }

    bool use = is_include(true) && !is_exclude();
    if(use)
        verbprintf(2, "Including function [no constraint] : '%s'...\n",
                   function_name.c_str());

    return use;
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

//======================================================================================//
//
std::string
get_absolute_exe_filepath(std::string exe_name)
{
    auto file_exists = [](const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    };

    if(!exe_name.empty() && !file_exists(exe_name))
    {
        auto _exe_orig = exe_name;
        auto _paths    = tim::delimit(tim::get_env<std::string>("PATH", ""), ":");
        for(auto& pitr : _paths)
        {
            if(file_exists(TIMEMORY_JOIN('/', pitr, exe_name)))
            {
                exe_name = TIMEMORY_JOIN('/', pitr, exe_name);
                verbprintf(0, "[omnitrace][exe] Resolved '%s' to '%s'...\n",
                           _exe_orig.c_str(), exe_name.c_str());
                break;
            }
        }

        if(!file_exists(exe_name))
        {
            verbprintf(0, "Warning! File path to '%s' could not be determined...\n",
                       exe_name.c_str());
        }
    }
    return exe_name;
}

//======================================================================================//
//
std::string
get_absolute_lib_filepath(std::string lib_name)
{
    auto file_exists = [](const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    };

    if(!lib_name.empty() && (!file_exists(lib_name) ||
                             std::regex_match(lib_name, std::regex("^[A-Za-z0-9].*"))))
    {
        auto _lib_orig = lib_name;
        auto _paths    = tim::delimit(
            std::string{ ".:" } + tim::get_env<std::string>("LD_LIBRARY_PATH", ""), ":");
        for(auto& pitr : _paths)
        {
            if(file_exists(TIMEMORY_JOIN('/', pitr, lib_name)))
            {
                lib_name = TIMEMORY_JOIN('/', pitr, lib_name);
                verbprintf(0, "[omnitrace][exe] Resolved '%s' to '%s'...\n",
                           _lib_orig.c_str(), lib_name.c_str());
                break;
            }
        }

        if(!file_exists(lib_name))
        {
            verbprintf(0, "Warning! File path to '%s' could not be determined...\n",
                       lib_name.c_str());
        }
    }
    return lib_name;
}
