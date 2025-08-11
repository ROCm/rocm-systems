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

#include "rocprof-sys-instrument.hpp"
#include "common/defines.h"
#include "common/join.hpp"
#include "dl/dl.hpp"
#include "fwd.hpp"
#include "internal_libs.hpp"
#include "log.hpp"

#include <timemory/backends/process.hpp>
#include <timemory/config.hpp>
#include <timemory/environment/types.hpp>
#include <timemory/hash.hpp>
#include <timemory/log/macros.hpp>
#include <timemory/manager.hpp>
#include <timemory/settings.hpp>
#include <timemory/signals/signal_mask.hpp>
#include <timemory/utility/console.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/signals.hpp>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

#if !defined(ROCPROFSYS_USE_MPI)
#    define ROCPROFSYS_USE_MPI 0
#endif

#if !defined(ROCPROFSYS_USE_MPI_HEADERS)
#    define ROCPROFSYS_USE_MPI_HEADERS 0
#endif

namespace
{
auto
get_default_min_instructions()
{
    // default to 1024
    return tim::get_env<size_t>("ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS", (1 << 10), false);
}
auto
get_default_min_address_range()
{
    // default to 4096
    return 4 * get_default_min_instructions();
}
}  // namespace

using InstrumentMode = ::rocprofsys::dl::InstrumentMode;

bool   use_return_info              = false;
bool   use_args_info                = false;
bool   use_file_info                = false;
bool   use_line_info                = false;
bool   allow_overlapping            = false;
bool   loop_level_instr             = false;
bool   instr_dynamic_callsites      = false;
bool   instr_traps                  = false;
bool   instr_loop_traps             = false;
bool   parse_all_modules            = false;
size_t min_address_range            = get_default_min_address_range();  // 4096
size_t min_loop_address_range       = get_default_min_address_range();  // 4096
size_t min_instructions             = get_default_min_instructions();   // 1024
size_t min_loop_instructions        = get_default_min_instructions();   // 1024
bool   werror                       = false;
bool   debug_print                  = false;
bool   instr_print                  = false;
bool   simulate                     = false;
bool   include_uninstr              = false;
bool   include_internal_linked_libs = false;
int    verbose_level   = tim::get_env<int>("ROCPROFSYS_VERBOSE_INSTRUMENT", 0);
int    num_log_entries = tim::get_env<int>(
    "ROCPROFSYS_LOG_COUNT", tim::get_env<bool>("ROCPROFSYS_CI", false) ? 20 : 50);
string_t main_fname     = "main";
string_t argv0          = {};
string_t cmdv0          = {};
string_t prefer_library = {};
//
//  global variables
//
patch_pointer_t  bpatch                        = {};
call_expr_t*     terminate_expr                = nullptr;
snippet_vec_t    init_names                    = {};
snippet_vec_t    fini_names                    = {};
fmodset_t        available_module_functions    = {};
fmodset_t        instrumented_module_functions = {};
fmodset_t        coverage_module_functions     = {};
fmodset_t        overlapping_module_functions  = {};
fmodset_t        excluded_module_functions     = {};
fixed_modset_t   fixed_module_functions        = {};
regexvec_t       func_include                  = {};
regexvec_t       func_exclude                  = {};
regexvec_t       file_include                  = {};
regexvec_t       file_exclude                  = {};
regexvec_t       file_restrict                 = {};
regexvec_t       func_restrict                 = {};
regexvec_t       caller_include                = {};
regexvec_t       func_internal_include         = {};
regexvec_t       file_internal_include         = {};
regexvec_t       instruction_exclude           = {};
CodeCoverageMode coverage_mode                 = CODECOV_NONE;

symtab_data_s                       symtab_data                = {};
std::set<symbol_linkage_t>          enabled_linkage            = {};
std::set<symbol_visibility_t>       enabled_visibility         = {};
const std::set<symbol_linkage_t>    default_enabled_linkage    = { SL_GLOBAL, SL_LOCAL,
                                                                   SL_UNIQUE };
const std::set<symbol_visibility_t> default_enabled_visibility = { SV_DEFAULT, SV_HIDDEN,
                                                                   SV_INTERNAL,
                                                                   SV_PROTECTED };

std::unique_ptr<std::ofstream> log_ofs = {};

namespace
{
namespace process  = tim::process;  // NOLINT
namespace signals  = tim::signals;
namespace filepath = tim::filepath;

using signal_settings = tim::signals::signal_settings;
using sys_signal      = tim::signals::sys_signal;

bool                                       binary_rewrite       = false;
bool                                       is_attached          = false;
bool                                       use_mpi              = false;
bool                                       is_static_exe        = false;
bool                                       force_config         = false;
size_t                                     batch_size           = 50;
strset_t                                   extra_libs           = {};
std::vector<std::pair<uint64_t, string_t>> hash_ids             = {};
std::map<string_t, bool>                   use_stubs            = {};
std::map<string_t, procedure_t*>           beg_stubs            = {};
std::map<string_t, procedure_t*>           end_stubs            = {};
strvec_t                                   init_stub_names      = {};
strvec_t                                   fini_stub_names      = {};
strset_t                                   used_stub_names      = {};
strvec_t                                   env_config_variables = {};
std::vector<call_expr_pointer_t>           env_variables        = {};
std::map<string_t, call_expr_pointer_t>    beg_expr             = {};
std::map<string_t, call_expr_pointer_t>    end_expr             = {};
const auto                                 npos_v               = string_t::npos;
string_t                                   instr_mode           = "trace";
string_t                                   print_coverage       = {};
string_t                                   print_instrumented   = {};
string_t                                   print_excluded       = {};
string_t                                   print_available      = {};
string_t                                   print_overlapping    = {};
strset_t                                   print_formats        = { "txt", "json" };
std::string                                modfunc_dump_dir     = {};
auto regex_opts = std::regex_constants::egrep | std::regex_constants::optimize;

std::string
get_internal_libpath()
{
    auto _exe = std::string_view{ ::realpath("/proc/self/exe", nullptr) };
    auto _pos = _exe.find_last_of('/');
    auto _dir = std::string{ "./" };
    if(_pos != std::string_view::npos) _dir = _exe.substr(0, _pos);
    return rocprofsys::common::join("/", _dir, "..", "lib");
}

strvec_t lib_search_paths = tim::delimit(
    JOIN(':', get_internal_libpath(), tim::get_env<std::string>("DYNINSTAPI_RT_LIB"),
         tim::get_env<std::string>("DYNINST_REWRITER_PATHS"),
         tim::get_env<std::string>("LD_LIBRARY_PATH")),
    ":");
strvec_t bin_search_paths = tim::delimit(tim::get_env<std::string>("PATH"), ":");

auto _dyn_api_rt_paths = tim::delimit(
    JOIN(":", get_internal_libpath(), JOIN("/", get_internal_libpath(), "rocprofsys")),
    ":");

std::string
get_absolute_filepath(std::string _name, const strvec_t& _paths);

std::string
get_absolute_filepath(std::string _name);

std::string
get_absolute_exe_filepath(std::string exe_name);

std::string
get_absolute_lib_filepath(std::string lib_name);

bool
exists(const std::string& name);

bool
is_file(std::string _name);

bool
is_directory(std::string _name);

std::string
get_realpath(const std::string&);

std::string
get_cwd();

void
find_dyn_api_rt();

void
activate_signal_handlers(const std::vector<sys_signal>& _signals)
{
    for(const auto& itr : _signals)
        signal_settings::enable(itr);

    static bool _protect     = false;
    auto        _exit_action = [](int nsig) {
        if(_protect) return;
        _protect = true;
        TIMEMORY_PRINTF_FATAL(
            stderr, "rocprof-sys exited with signal %i :: %s\n", nsig,
            signal_settings::str(static_cast<sys_signal>(nsig)).c_str());

        // print any forced entries
        print_log_entries(
            std::cerr, -1, [](const auto& _v) { return _v.forced(); },
            []() {
                tim::log::stream(std::cerr, tim::log::color::info())
                    << "\n[rocprof-sys][exe] Potentially important log entries:\n\n";
            });

        // print the last log entries
        print_log_entries(std::cerr, num_log_entries);

        TIMEMORY_PRINTF_FATAL(stderr, "\n");
        TIMEMORY_PRINTF_FATAL(
            stderr,
            "These were the last %i log entries from rocprof-sys. You can control the "
                   "number of log entries via the '--log <N>' option or ROCPROFSYS_LOG_COUNT "
                   "env variable.\n",
            num_log_entries);

        if(log_ofs) log_ofs->close();
        log_ofs.reset();

        _protect = false;
    };

    signal_settings::set_exit_action(_exit_action);
    signal_settings::check_environment();
    signals::enable_signal_detection(signal_settings::get_enabled());
}

// default signals to catch
auto _activate =
    (activate_signal_handlers({ sys_signal::Interrupt, sys_signal::FPE, sys_signal::Stop,
                                sys_signal::Quit, sys_signal::Illegal, sys_signal::Abort,
                                sys_signal::Bus, sys_signal::SegFault,
                                sys_signal::FileSize, sys_signal::CPUtime }),
     true);

auto
find(const std::string& itr, const strvec_t& _data)
{
    return std::any_of(_data.begin(), _data.end(),
                       [itr](const auto& _v) { return (itr == _v); });
}
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
    argv0 = argv[0];

    auto _omni_root = tim::get_env<std::string>(
        "rocprofiler_systems_ROOT", tim::get_env<std::string>("ROCPROFSYS_ROOT", ""));
    if(!_omni_root.empty() && exists(_omni_root))
    {
        bin_search_paths.emplace_back(JOIN('/', _omni_root, "bin"));
        bin_search_paths.emplace_back(
            JOIN('/', _omni_root, "lib", "rocprofiler-systems"));
        bin_search_paths.emplace_back(
            JOIN('/', _omni_root, "lib", "rocprofiler-systems", "bin"));
        lib_search_paths.emplace_back(JOIN('/', _omni_root, "lib"));
        lib_search_paths.emplace_back(
            JOIN('/', _omni_root, "lib", "rocprofiler-systems"));
        lib_search_paths.emplace_back(
            JOIN('/', _omni_root, "lib", "rocprofiler-systems", "lib"));
        lib_search_paths.emplace_back(
            JOIN('/', _omni_root, "lib", "rocprofiler-systems", "lib64"));
        ROCPROFSYS_ADD_LOG_ENTRY(argv[0],
                                 "::", "rocprofiler-systems root path: ", _omni_root);
    }

    auto _omni_exe_path = get_realpath(get_absolute_exe_filepath(argv[0]));
    if(!exists(_omni_exe_path))
        _omni_exe_path =
            get_realpath(get_absolute_exe_filepath(rocprofsys_get_exe_realpath()));
    bin_search_paths.emplace_back(filepath::dirname(_omni_exe_path));

    auto _omni_lib_path =
        JOIN('/', filepath::dirname(filepath::dirname(_omni_exe_path)), "lib");
    bin_search_paths.emplace_back(JOIN('/', _omni_lib_path, "rocprofiler-systems"));
    bin_search_paths.emplace_back(
        JOIN('/', _omni_lib_path, "rocprofiler-systems", "bin"));
    lib_search_paths.emplace_back(_omni_lib_path);
    lib_search_paths.emplace_back(JOIN('/', _omni_lib_path, "rocprofiler-systems"));
    lib_search_paths.emplace_back(
        JOIN('/', _omni_lib_path, "rocprofiler-systems", "lib"));
    lib_search_paths.emplace_back(
        JOIN('/', _omni_lib_path, "rocprofiler-systems", "lib64"));

    auto _omni_internal_libexec_path =
        JOIN('/', filepath::dirname(filepath::dirname(_omni_exe_path)), "libexec",
             "rocprofiler-systems");

    ROCPROFSYS_ADD_LOG_ENTRY(argv[0], "::", "rocprofsys bin path: ", _omni_exe_path);
    ROCPROFSYS_ADD_LOG_ENTRY(argv[0], "::", "rocprofsys lib path: ", _omni_lib_path);
    ROCPROFSYS_ADD_LOG_ENTRY(
        argv[0], "::", "rocprofsys libexec path: ", _omni_internal_libexec_path);

    for(const auto& itr : rocprofsys_get_link_map(nullptr))
    {
        if(itr.find("rocprofsys") != std::string::npos ||
           itr.find("rocprof-sys") != std::string::npos ||
           itr.find("rocprofiler-systems") != std::string::npos ||
           std::regex_search(
               itr, std::regex{
                        "lib(dyninstAPI|stackwalk|pcontrol|patchAPI|parseAPI|"
                        "instructionAPI|symtabAPI|dynDwarf|common|dynElf|tbb|tbbmalloc|"
                        "tbbmalloc_proxy|gotcha|libunwind|hsa-runtime|amdhip|"
                        "amd_comgr|amd_smi|rocprofiler-register|"
                        "rocprofiler-sdk|rocprofiler-sdk-roctx)\\.(so|a)" }))
        {
            if(!find(filepath::dirname(itr), lib_search_paths))
                lib_search_paths.emplace_back(filepath::dirname(itr));
        }
    }

    // DO NOT SORT! Just remove adjacent duplicates
    bin_search_paths.erase(std::unique(bin_search_paths.begin(), bin_search_paths.end()),
                           bin_search_paths.end());
    lib_search_paths.erase(std::unique(lib_search_paths.begin(), lib_search_paths.end()),
                           lib_search_paths.end());

    for(const auto& itr : bin_search_paths)
    {
        ROCPROFSYS_ADD_LOG_ENTRY("bin search path:", itr);
    }

    for(const auto& itr : lib_search_paths)
    {
        ROCPROFSYS_ADD_LOG_ENTRY("lib search path:", itr);
    }

    address_space_t*      addr_space    = nullptr;
    string_t              mutname       = {};
    string_t              outfile       = {};
    string_t              logfile       = {};
    std::vector<string_t> inputlib      = { "librocprof-sys-dl" };
    std::vector<string_t> libname       = {};
    std::vector<string_t> sharedlibname = {};
    std::vector<string_t> staticlibname = {};
    process::id_t         _pid          = -1;

    fixed_module_functions = {
        { &available_module_functions, false },
        { &instrumented_module_functions, false },
        { &coverage_module_functions, false },
        { &excluded_module_functions, false },
        { &overlapping_module_functions, false },
    };

    std::set<std::string> dyninst_defs = { "TypeChecking", "SaveFPR", "DelayedParsing",
                                           "DebugParsing", "MergeTramp" };

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
                auto _v =
                    std::regex_replace(argv[j], std::regex{ "(.*)([ \t\n\r]+)$" }, "$1");
                copy_str(_cmdv[k], _v.c_str());
            }
            if(_cmdc > 0) mutname = _cmdv[0];
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
        auto resolved_mutname = get_realpath(get_absolute_filepath(mutname));
        if(resolved_mutname != mutname)
        {
            mutname = resolved_mutname;
            free(_cmdv[0]);
            copy_str(_cmdv[0], resolved_mutname.c_str());
        }
    }

    if(verbose_level > 1)
    {
        std::cout << "[rocprof-sys][exe][original]: " << cmd_string(argc, argv)
                  << std::endl;
        std::cout << "[rocprof-sys][exe][cfg-args]: " << cmd_string(_argc, _argv)
                  << std::endl;
    }

    if(_cmdc > 0) cmdv0 = _cmdv[0];

    // now can loop through the options.  If the first character is '-', then we know
    // we have an option.  Check to see if it is one of our options and process it. If
    // it is unrecognized, then set the errflag to report an error.  When we come to a
    // non '-' charcter, then we must be at the application name.
    using parser_t = tim::argparse::argument_parser;
    parser_t parser("rocprof-sys-instrument");
    string_t extra_help = "-- <CMD> <ARGS>";

    parser.enable_help();
    parser.enable_version("rocprof-sys-instrument", ROCPROFSYS_ARGPARSE_VERSION_INFO);

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[DEBUG OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
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
    parser.add_argument({ "--debug" }, "Debug output")
        .max_count(1)
        .action([](parser_t& p) {
            debug_print = p.get<bool>("debug");
            if(debug_print && !p.exists("verbose")) verbose_level = 256;
        });
    parser
        .add_argument({ "--log" }, "Number of log entries to display after an error. Any "
                                   "value < 0 will emit the entire log")
        .count(1)
        .action([](parser_t& p) { num_log_entries = p.get<int>("log"); });
    parser
        .add_argument({ "--log-file" },
                      "Write the log out the specified file during the run")
        .count(1)
        .action([&logfile](parser_t& p) {
            auto _oname       = p.get<std::string>("log-file");
            auto _cfg         = tim::settings::compose_filename_config{};
            _cfg.subdirectory = "instrumentation";
            logfile = tim::settings::compose_output_filename(_oname, "log", _cfg);
        });
    parser
        .add_argument({ "--simulate" },
                      "Exit after outputting diagnostic "
                      "{available,instrumented,excluded,overlapping} module "
                      "function lists, e.g. available.txt")
        .max_count(1)
        .dtype("boolean")
        .action([](parser_t& p) { simulate = p.get<bool>("simulate"); });
    parser
        .add_argument({ "--print-format" },
                      "Output format for diagnostic "
                      "{available,instrumented,excluded,overlapping} module "
                      "function lists, e.g. {print-dir}/available.txt")
        .min_count(1)
        .max_count(3)
        .dtype("string")
        .choices({ "xml", "json", "txt" })
        .action([](parser_t& p) { print_formats = p.get<strset_t>("print-format"); });
    parser
        .add_argument({ "--print-dir" },
                      "Output directory for diagnostic "
                      "{available,instrumented,excluded,overlapping} module "
                      "function lists, e.g. {print-dir}/available.txt")
        .count(1)
        .dtype("string")
        .action([](parser_t& p) {
            tim::settings::output_path() = p.get<std::string>("print-dir");
        });
    parser
        .add_argument(
            { "--print-available" },
            "Print the available entities for instrumentation (functions, modules, or "
            "module-function pair) to stdout after applying regular expressions")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action(
            [](parser_t& p) { print_available = p.get<std::string>("print-available"); });
    parser
        .add_argument(
            { "--print-instrumented" },
            "Print the instrumented entities (functions, modules, or module-function "
            "pair) to stdout after applying regular expressions")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action([](parser_t& p) {
            print_instrumented = p.get<std::string>("print-instrumented");
        });
    parser
        .add_argument({ "--print-coverage" },
                      "Print the instrumented coverage entities (functions, modules, or "
                      "module-function "
                      "pair) to stdout after applying regular expressions")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action(
            [](parser_t& p) { print_coverage = p.get<std::string>("print-coverage"); });
    parser
        .add_argument({ "--print-excluded" },
                      "Print the entities for instrumentation (functions, modules, or "
                      "module-function "
                      "pair) which are excluded from the instrumentation to stdout after "
                      "applying regular expressions")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action(
            [](parser_t& p) { print_excluded = p.get<std::string>("print-excluded"); });
    parser
        .add_argument(
            { "--print-overlapping" },
            "Print the entities for instrumentation (functions, modules, or "
            "module-function pair) which overlap other function calls or have multiple "
            "entry points to stdout after applying regular expressions")
        .count(1)
        .choices({ "functions", "modules", "functions+", "pair", "pair+" })
        .action([](parser_t& p) {
            print_overlapping = p.get<std::string>("print-overlapping");
        });
    parser
        .add_argument(
            { "--print-instructions" },
            "Print the instructions for each basic-block in the JSON/XML outputs")
        .max_count(1)
        .action([](parser_t& p) { instr_print = p.get<bool>("print-instructions"); });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[MODE OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser
        .add_argument({ "-o", "--output" },
                      "Enable generation of a new executable (binary-rewrite). If a "
                      "filename is not provided, rocprof-sys will use the basename and "
                      "output to the cwd, unless the target binary is in the cwd. In the "
                      "latter case, rocprof-sys will either use ${PWD}/<basename>.inst "
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
        .choices({ "trace", "sampling", "coverage" })
        .count(1)
        .action([](parser_t& p) {
            instr_mode = p.get<string_t>("mode");
            if(instr_mode == "coverage" && !p.exists("coverage"))
                coverage_mode = CODECOV_FUNCTION;
        });
    parser
        .add_argument(
            { "-f", "--force" },
            "Force the command-line argument configuration, i.e. don't get cute. Useful "
            "for forcing runtime instrumentation of an executable that [A] Dyninst "
            "thinks is a library after reading ELF and [B] whose name makes it look like "
            "a library (e.g. starts with 'lib' and/or ends in '.so', '.so.*', or '.a')")
        .max_count(1)
        .action([](parser_t& p) { force_config = p.get<bool>("force"); });

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
                keys.at(0) = get_realpath(get_absolute_filepath(keys.at(0)));
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
            TIMEMORY_JOIN("", "Libraries with instrumentation routines (default: \"",
                          inputlib.front(), "\")"))
        .action([&inputlib](parser_t& p) { inputlib = p.get<strvec_t>("library"); });
    parser
        .add_argument({ "-m", "--main-function" },
                      "The primary function to instrument around, e.g. 'main'")
        .count(1)
        .action([](parser_t& p) { main_fname = p.get<string_t>("main-function"); });
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
                { "coverage_module_functions", &coverage_module_functions },
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
        .action([](parser_t& p) { fini_stub_names = p.get<strvec_t>("fini-functions"); });
    parser
        .add_argument(
            { "--all-functions" },
            "When finding functions, include the functions which are not instrumentable. "
            "This is purely diagnostic for the available/excluded functions output")
        .dtype("boolean")
        .max_count(1)
        .action([](parser_t& p) { include_uninstr = p.get<bool>("all-functions"); });

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
    parser.add_argument({ "--caller-include" },
                        "Regex(es) for including functions that call the "
                        "listed functions (despite heuristics)");
    parser.add_argument({ "-MI", "--module-include" },
                        "Regex(es) for selecting modules/files/libraries "
                        "(despite heuristics)");
    parser.add_argument({ "-ME", "--module-exclude" },
                        "Regex(es) for excluding modules/files/libraries "
                        "(always applied)");
    parser.add_argument({ "-MR", "--module-restrict" },
                        "Regex(es) for restricting modules/files/libraries only to those "
                        "that match the provided regular-expressions");
    parser.add_argument({ "--internal-function-include" },
                        "Regex(es) for including functions which are (likely) utilized "
                        "by rocprof-sys itself. Use this option with care.");
    parser.add_argument(
        { "--internal-module-include" },
        "Regex(es) for including modules/libraries which are (likely) utilized "
        "by rocprof-sys itself. Use this option with care.");
    parser.add_argument(
        { "--instruction-exclude" },
        "Regex(es) for excluding functions containing certain instructions");

    parser
        .add_argument({ "--internal-library-deps" },
                      "Treat the libraries linked to the internal libraries as internal "
                      "libraries. This increase the internal library processing time and "
                      "consume more memory (so use with care) but may be useful when the "
                      "application uses Boost libraries and Dyninst is dynamically "
                      "linked against the same boost libraries")
        .min_count(0)
        .max_count(1)
        .dtype("boolean")
        .action([](parser_t& p) {
            include_internal_linked_libs = p.get<bool>("internal-library-deps");
        });

    auto _internal_libs = get_internal_basic_libs();

    parser
        .add_argument({ "--internal-library-append" },
                      "Append to the list of libraries which rocprof-sys treats as being "
                      "used internally, e.g. rocprof-sys will find all the symbols in "
                      "this library and prevent them from being instrumented.")
        .action([](parser_t& p) {
            for(const auto& itr : p.get<strvec_t>("internal-library-append"))
                get_internal_basic_libs().emplace(itr);
        });

    parser
        .add_argument({ "--internal-library-remove" },
                      "Remove the specified libraries from being treated as being "
                      "used internally, e.g. rocprof-sys will permit all the symbols in "
                      "these libraries to be eligible for instrumentation.")
        .choices(_internal_libs)
        .action([](parser_t& p) {
            auto  _remove   = p.get<strset_t>("internal-library-remove");
            auto& _internal = get_internal_basic_libs();
            for(const auto& itr : _remove)
                _internal.erase(itr);
        });

    using timemory::join::array_config;
    using timemory::join::join;

    auto available_linkage    = std::vector<symbol_linkage_t>{};
    auto available_visibility = std::vector<symbol_visibility_t>{};

    for(int i = SL_UNKNOWN; i < SL_END_V; ++i)
        available_linkage.emplace_back(static_cast<symbol_linkage_t>(i));
    for(int i = SV_UNKNOWN; i < SV_END_V; ++i)
        available_visibility.emplace_back(static_cast<symbol_visibility_t>(i));

    auto _get_strvec = [](const auto& _inp) {
        auto _ret = std::vector<std::string>{};
        _ret.reserve(_inp.size());
        for(const auto& itr : _inp)
            _ret.emplace_back(std::to_string(itr));
        return _ret;
    };

    enabled_linkage = default_enabled_linkage;
    parser
        .add_argument(
            { "--linkage" },
            join("", "Only instrument functions with specified linkage (default: ",
                 join(array_config{ ", ", "", "" }, _get_strvec(default_enabled_linkage)),
                 ")"))
        .min_count(1)
        .choices(_get_strvec(available_linkage))
        .set_default(_get_strvec(default_enabled_linkage))
        .action([](parser_t& p) {
            auto selected_linkage = p.get<std::set<std::string>>("linkage");
            if(!selected_linkage.empty())
            {
                enabled_linkage.clear();
                for(const auto& itr : selected_linkage)
                    enabled_linkage.emplace(from_string<symbol_linkage_t>(itr));
            }
        });

    enabled_visibility = default_enabled_visibility;
    parser
        .add_argument(
            { "--visibility" },
            join("", "Only instrument functions with specified visibility (default: ",
                 join(array_config{ ", ", "", "" },
                      _get_strvec(default_enabled_visibility)),
                 ")"))
        .min_count(1)
        .choices(_get_strvec(available_visibility))
        .set_default(_get_strvec(default_enabled_visibility))
        .action([](parser_t& p) {
            auto selected_visibility = p.get<std::set<std::string>>("visibility");
            if(!selected_visibility.empty())
            {
                enabled_visibility.clear();
                for(const auto& itr : selected_visibility)
                    enabled_visibility.emplace(from_string<symbol_visibility_t>(itr));
            }
        });

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
        .names({ "-C", "--config" })
        .dtype("string")
        .min_count(1)
        .description("Read in a configuration file and encode these values as the "
                     "defaults in the executable");
    parser.add_argument({ "--env" },
                        "Environment variables to add to the runtime in form "
                        "VARIABLE=VALUE. E.g. use '--env ROCPROFSYS_PROFILE=ON' to "
                        "default to using timemory instead of perfetto");
    parser
        .add_argument({ "--mpi" },
                      "Enable MPI support (requires rocprof-sys built w/ full or partial "
                      "MPI support). NOTE: this will automatically be activated if "
                      "MPI_Init, MPI_Init_thread, MPI_Finalize, MPI_Comm_rank, or "
                      "MPI_Comm_size are found in the symbol table of target")
        .max_count(1)
        .action([](parser_t& p) {
            use_mpi = p.get<bool>("mpi");
#if ROCPROFSYS_USE_MPI == 0 && ROCPROFSYS_USE_MPI_HEADERS == 0
            errprintf(0, "rocprof-sys was not built with full or partial MPI support\n");
            use_mpi = false;
#endif
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[GRANULARITY OPTIONS]" }, "");
    parser.add_argument({ "" }, "");
    parser.add_argument({ "-l", "--instrument-loops" }, "Instrument at the loop level")
        .dtype("boolean")
        .max_count(1)
        .action([](parser_t& p) { loop_level_instr = p.get<bool>("instrument-loops"); });
    parser
        .add_argument({ "-i", "--min-instructions" },
                      "If the number of instructions in a function is less than this "
                      "value, exclude it from instrumentation")
        .count(1)
        .dtype("int")
        .action(
            [](parser_t& p) { min_instructions = p.get<size_t>("min-instructions"); });
    parser
        .add_argument({ "-r", "--min-address-range" },
                      "If the address range of a function is less than this value, "
                      "exclude it from instrumentation")
        .count(1)
        .dtype("int")
        .action(
            [](parser_t& p) { min_address_range = p.get<size_t>("min-address-range"); });
    parser
        .add_argument({ "--min-instructions-loop" },
                      "If the number of instructions in a function containing a loop is "
                      "less than this value, exclude it from instrumentation")
        .count(1)
        .dtype("int")
        .action([](parser_t& p) {
            min_loop_instructions = p.get<size_t>("min-instructions-loop");
        });
    parser
        .add_argument({ "--min-address-range-loop" },
                      "If the address range of a function containing a loop is less than "
                      "this value, exclude it from instrumentation")
        .count(1)
        .dtype("int")
        .action([](parser_t& p) {
            min_loop_address_range = p.get<size_t>("min-address-range-loop");
        });
    parser
        .add_argument(
            { "--coverage" },
            "Enable recording the code coverage. If instrumenting in coverage mode ('-M "
            "converage'), this simply specifies the granularity. If instrumenting in "
            "trace or sampling mode, this enables recording code-coverage in addition to "
            "the instrumentation of that mode (if any).")
        .max_count(1)
        .choices({ "none", "function", "basic_block" })
        .action([](parser_t& p) {
            auto _v = p.get<std::string>("coverage");
            if(_v == "function" || _v.empty())
                coverage_mode = CODECOV_FUNCTION;
            else if(_v == "basic_block")
                coverage_mode = CODECOV_BASIC_BLOCK;
            else
                coverage_mode = CODECOV_NONE;
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
        .dtype("boolean")
        .set_default(instr_traps)
        .action([](parser_t& p) { instr_traps = p.get<bool>("traps"); });
    parser
        .add_argument({ "--loop-traps" },
                      "Instrument points within a loop which require using a trap (only "
                      "relevant when --instrument-loops is enabled).")
        .max_count(1)
        .dtype("boolean")
        .set_default(instr_loop_traps)
        .action([](parser_t& p) { instr_loop_traps = p.get<bool>("loop-traps"); });
    parser
        .add_argument(
            { "--allow-overlapping" },
            "Allow dyninst to instrument either multiple functions which overlap (share "
            "part of same function body) or single functions with multiple entry points. "
            "For more info, see Section 2 of the DyninstAPI documentation.")
        .max_count(1)
        .action(
            [](parser_t& p) { allow_overlapping = p.get<bool>("allow-overlapping"); });
    parser
        .add_argument(
            { "--parse-all-modules" },
            "By default, rocprof-sys simply requests Dyninst to provide all the "
            "procedures "
            "in the application image. If this option is enabled, rocprof-sys will "
            "iterate "
            "over all the modules and extract the functions. Theoretically, it should be "
            "the same but the data is slightly different, possibly due to weak binding "
            "scopes. In general, enabling option will probably have no visible effect")
        .max_count(1)
        .action(
            [](parser_t& p) { parse_all_modules = p.get<bool>("parse-all-modules"); });

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
    parser.add_argument({ "--dyninst-rt" }, "Path(s) to the dyninstAPI_RT library")
        .dtype("filepath")
        .min_count(1)
        .action([](parser_t& _p) {
            auto _v = _p.get<strvec_t>("dyninst-rt");
            std::copy(_dyn_api_rt_paths.begin(), _dyn_api_rt_paths.end(),
                      std::back_inserter(_v));
            std::swap(_dyn_api_rt_paths, _v);
        });
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

    verbprintf(0, "\n");
    verbprintf(0, "command :: '%s'...\n", cmd_string(_cmdc, _cmdv).c_str());
    verbprintf(0, "\n");

    if(err)
    {
        std::cerr << err << std::endl;
        parser.print_help(extra_help);
        return -1;
    }

    if(parser.exists("config"))
    {
        struct rocprofsys_env_config_s
        {};
        auto _configs = parser.get<strvec_t>("config");
        for(auto&& itr : _configs)
        {
            auto _settings = tim::settings::push<rocprofsys_env_config_s>();
            for(auto&& iitr : *_settings)
            {
                if(iitr.second->get_updated()) iitr.second->set_user_updated();
            }
            _settings->read(itr);
            for(auto&& iitr : *_settings)
            {
                if(iitr.second && iitr.second->get_config_updated())
                {
                    env_config_variables.emplace_back(TIMEMORY_JOIN(
                        '=', iitr.second->get_env_name(), iitr.second->as_string()));
                    verbprintf(1, "Exporting known config value :: %s\n",
                               env_config_variables.back().c_str());
                }
            }
            for(auto&& iitr : _settings->get_unknown_configs())
            {
                env_config_variables.emplace_back(
                    TIMEMORY_JOIN('=', iitr.first, iitr.second));
                verbprintf(1, "Exporting unknown config value :: %s\n",
                           env_config_variables.back().c_str());
            }
            tim::settings::pop<rocprofsys_env_config_s>();
        }
    }

    auto _handle_heuristics = [&parser](std::string&& _exists, std::string&& _not_exists,
                                        auto& _field, auto _value, std::string&& _msg,
                                        bool _cond) {
        // if first is specified but second is not, to reduce verbosity of command-line
        // and increase simplicity, set _field to specified value
        if(parser.exists(_exists) && !parser.exists(_not_exists) && _cond)
        {
            verbprintf(3,
                       "Option '--%s' specified but '--%s <N>' was not specified. "
                       "Setting %s to %s...\n",
                       _exists.c_str(), _not_exists.c_str(), _msg.c_str(),
                       TIMEMORY_JOIN("", _value).c_str());
            _field = _value;
        }
    };

    // if instructions was specified and address range was not
    _handle_heuristics("min-instructions", "min-address-range", min_address_range, 0,
                       "minimum address range", true);
    _handle_heuristics("min-instructions-loop", "min-address-range-loop",
                       min_loop_address_range, 0, "minimum address range for loops",
                       true);

    // if address range was specified but instructions was not
    _handle_heuristics("min-address-range", "min-instructions", min_instructions, 0,
                       "minimum instructions", true);
    _handle_heuristics("min-address-range-loop", "min-instructions-loop",
                       min_loop_instructions, 0, "minimum instructions for loops", true);

    // if non-loop value was specified but loop value was not
    _handle_heuristics("min-instructions", "min-instructions-loop", min_loop_instructions,
                       min_instructions, "minimum instructions for loops", true);
    _handle_heuristics("min-address-range", "min-address-range-loop",
                       min_loop_address_range, min_address_range,
                       "minimum address range for loops", true);

    // if non-loop instructions was specified and loop address range was not specified
    // as long as non-loop address range and loop instructions were not specified
    _handle_heuristics("min-instructions", "min-address-range-loop",
                       min_loop_address_range, 0, "minimum address range for loops",
                       !parser.exists("min-address-range") &&
                           !parser.exists("min-instructions-loop"));

    // if non-loop address range was specified and loop instructions was not specified
    // as long as non-loop instructions and loop address range were not specified
    _handle_heuristics("min-address-range", "min-instructions-loop",
                       min_loop_instructions, 0, "minimum instructions for loops",
                       !parser.exists("min-instructions") &&
                           !parser.exists("min-address-range-loop"));

    auto _rocprofsys_exe_path = tim::dirname(::get_realpath("/proc/self/exe"));
    verbprintf(4, "rocprof-sys exe path: %s\n", _rocprofsys_exe_path.c_str());

    if(_cmdv && _cmdv[0] && strlen(_cmdv[0]) > 0)
    {
        auto _is_executable    = rocprofsys_get_is_executable(_cmdv[0], binary_rewrite);
        std::string _cmdv_base = ::basename(_cmdv[0]);
        auto        _has_lib_suffix = _cmdv_base.length() > 3 &&
                               (_cmdv_base.find(".so.") != std::string::npos ||
                                _cmdv_base.find(".so") == (_cmdv_base.length() - 3) ||
                                _cmdv_base.find(".a") == (_cmdv_base.length() - 2));
        auto _has_lib_prefix = _cmdv_base.length() > 3 && _cmdv_base.find("lib") == 0;
        if(!force_config && !_is_executable && !binary_rewrite &&
           (_has_lib_prefix || _has_lib_suffix))
        {
            fflush(stdout);
            std::stringstream _separator{};
            // 20 is approximate length of '[rocprof-sys][exe] '
            // 32 is approximate length of 'Warning! "" is not executable!'
            size_t _width =
                std::min<size_t>(std::get<0>(tim::utility::console::get_columns()) - 20,
                                 strlen(_cmdv[0]) + 32);
            _separator.fill('=');
            _separator << "#" << std::setw(_width - 2) << "" << "#";
            verbprintf(0, "%s\n", _separator.str().c_str());
            verbprintf(0, "\n");
            verbprintf(0, "Warning! '%s' is not executable!\n", _cmdv[0]);
            verbprintf(0, "Runtime instrumentation is not possible!\n");
            verbprintf(0, "Switching to binary rewrite mode and assuming '--simulate "
                          "--all-functions'\n");
            verbprintf(
                0, "(which will provide an approximation for runtime instrumentation)\n");
            verbprintf(1, "%s :: (^lib)=%s, (.so$|.a$|.so.*)=%s\n", _cmdv_base.c_str(),
                       (_has_lib_prefix) ? "true" : "false",
                       (_has_lib_suffix) ? "true" : "false");
            verbprintf(0, "\n");
            verbprintf(0, "%s\n", _separator.str().c_str());
            verbprintf(0, "\n");
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds{ 500 });
            binary_rewrite  = true;
            simulate        = true;
            include_uninstr = true;
        }
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

    if(binary_rewrite)
    {
        auto* _save = _cmdv[0];
        _cmdv[0]    = const_cast<char*>(outfile.c_str());
        tim::timemory_init(_cmdc, _cmdv, "rocprofsys-");
        _cmdv[0] = _save;
    }
    else
    {
        tim::timemory_init(_cmdc, _cmdv, "rocprofsys-");
    }

    if(!logfile.empty())
    {
        log_ofs = std::make_unique<std::ofstream>();
        verbprintf_bare(0, "%s", ::tim::log::color::source());
        verbprintf(0, "Opening '%s' for log output... ", logfile.c_str());
        if(!filepath::open(*log_ofs, logfile))
            throw std::runtime_error(JOIN(" ", "Error opening log output file", logfile));
        verbprintf_bare(0, "Done\n%s", ::tim::log::color::end());
        print_log_entries(*log_ofs, -1, {}, {}, "", false);
    }

    //----------------------------------------------------------------------------------//
    //
    //                              REGEX OPTIONS
    //
    //----------------------------------------------------------------------------------//
    //
    {
        //  Helper function for adding regex expressions
        auto add_regex = [](auto& regex_array, const string_t& regex_expr) {
            ROCPROFSYS_ADD_DETAILED_LOG_ENTRY("", "Adding regular expression \"",
                                              regex_expr, "\" to regex_array@",
                                              &regex_array);
            if(!regex_expr.empty())
                regex_array.emplace_back(std::regex(regex_expr, regex_opts));
        };

        add_regex(func_include, tim::get_env<string_t>("ROCPROFSYS_REGEX_INCLUDE", ""));
        add_regex(func_exclude, tim::get_env<string_t>("ROCPROFSYS_REGEX_EXCLUDE", ""));
        add_regex(func_restrict, tim::get_env<string_t>("ROCPROFSYS_REGEX_RESTRICT", ""));
        add_regex(caller_include,
                  tim::get_env<string_t>("ROCPROFSYS_REGEX_CALLER_INCLUDE"));
        add_regex(func_internal_include,
                  tim::get_env<string_t>("ROCPROFSYS_REGEX_INTERNAL_INCLUDE", ""));

        add_regex(file_include,
                  tim::get_env<string_t>("ROCPROFSYS_REGEX_MODULE_INCLUDE", ""));
        add_regex(file_exclude,
                  tim::get_env<string_t>("ROCPROFSYS_REGEX_MODULE_EXCLUDE", ""));
        add_regex(file_restrict,
                  tim::get_env<string_t>("ROCPROFSYS_REGEX_MODULE_RESTRICT", ""));
        add_regex(file_internal_include,
                  tim::get_env<string_t>("ROCPROFSYS_REGEX_MODULE_INTERNAL_INCLUDE", ""));

        add_regex(instruction_exclude,
                  tim::get_env<string_t>("ROCPROFSYS_REGEX_INSTRUCTION_EXCLUDE", ""));

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
        _parse_regex_option("caller-include", caller_include);
        _parse_regex_option("internal-function-include", func_internal_include);
        _parse_regex_option("module-include", file_include);
        _parse_regex_option("module-exclude", file_exclude);
        _parse_regex_option("module-restrict", file_restrict);
        _parse_regex_option("internal-module-include", file_internal_include);
        _parse_regex_option("instruction-exclude", instruction_exclude);
    }

    //----------------------------------------------------------------------------------//
    //
    //                              DYNINST OPTIONS
    //
    //----------------------------------------------------------------------------------//

    for(const auto& itr : _dyn_api_rt_paths)
    {
        lib_search_paths.emplace_back(itr);
        lib_search_paths.emplace_back(filepath::dirname(itr));
    }

    find_dyn_api_rt();

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

    bpatch = std::make_shared<patch_t>();
    bpatch->setTypeChecking(true);
    bpatch->setSaveFPR(true);
    bpatch->setDelayedParsing(true);
    bpatch->setDebugParsing(true);
    bpatch->setInstrStackFrames(false);
    bpatch->setLivenessAnalysis(false);
    bpatch->setBaseTrampDeletion(false);
    bpatch->setTrampRecursive(false);
    bpatch->setMergeTramp(true);

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
                        "\n\trocprof-sys-instrument <OPTIONS> -- <COMMAND> <ARGS>\nE.g. "
                        "\n\trocprof-sys-instrument -o foo.inst -- ./foo\nwill output an "
                        "instrumented version of 'foo' executable to 'foo.inst'\n");
        return EXIT_FAILURE;
    }

    verbprintf(1, "instrumentation target: %s\n", mutname.c_str());

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

    // prioritize the user environment arguments
    auto instr_mode_v     = (binary_rewrite) ? InstrumentMode::BinaryRewrite
                            : (_pid < 0)     ? InstrumentMode::ProcessCreate
                                             : InstrumentMode::ProcessAttach;
    auto instr_mode_v_int = static_cast<int>(instr_mode_v);
    auto env_vars         = parser.get<strvec_t>("env");
    env_vars.reserve(env_vars.size() + env_config_variables.size());
    for(auto&& itr : env_config_variables)
        env_vars.emplace_back(itr);
    env_vars.emplace_back(TIMEMORY_JOIN('=', "ROCPROFSYS_MODE", instr_mode));
    env_vars.emplace_back(
        TIMEMORY_JOIN('=', "ROCPROFSYS_INSTRUMENT_MODE", instr_mode_v_int));
    env_vars.emplace_back(TIMEMORY_JOIN('=', "ROCPROFSYS_MPI_INIT", "OFF"));
    env_vars.emplace_back(TIMEMORY_JOIN('=', "ROCPROFSYS_MPI_FINALIZE", "OFF"));
    env_vars.emplace_back(TIMEMORY_JOIN('=', "ROCPROFSYS_USE_CODE_COVERAGE",
                                        (coverage_mode != CODECOV_NONE) ? "ON" : "OFF"));
    addr_space = rocprofsys_get_address_space(bpatch, _cmdc, _cmdv, env_vars,
                                              binary_rewrite, _pid, mutname);

    // addr_space->allowTraps(instr_traps);

    if(!addr_space)
    {
        errprintf(-1, "address space for dynamic instrumentation was not created\n");
    }

    if(dynamic_cast<binary_edit_t*>(addr_space) != nullptr &&
       dynamic_cast<process_t*>(addr_space) != nullptr)
    {
        errprintf(-1, "address space statisfied dynamic_cast<%s> and dynamic_cast<%s>.\n",
                  tim::demangle<binary_edit_t*>().c_str(),
                  tim::demangle<process_t*>().c_str());
    }

    auto _rewrite = (dynamic_cast<binary_edit_t*>(addr_space) != nullptr &&
                     dynamic_cast<process_t*>(addr_space) == nullptr);
    if(_rewrite != binary_rewrite)
    {
        errprintf(-1, "binary rewrite was %s but has been deduced to be %s\n",
                  (binary_rewrite) ? "ON" : "OFF", (_rewrite) ? "ON" : "OFF");
    }

    process_t*     app_thread = nullptr;
    binary_edit_t* app_binary = nullptr;

    // get image
    verbprintf(1, "Getting the address space image, modules, and procedures...\n");
    image_t*                   app_image     = addr_space->getImage();
    std::vector<module_t*>*    app_modules   = app_image->getModules();
    std::vector<procedure_t*>* app_functions = app_image->getProcedures(include_uninstr);
    std::set<module_t*>        modules       = {};
    std::set<procedure_t*>     functions     = {};

    if(app_modules) process_modules(*app_modules);

    //----------------------------------------------------------------------------------//
    //
    //  Generate a log of all the available procedures and modules
    //
    //----------------------------------------------------------------------------------//
    std::set<std::string> module_names = {};

    static auto _insert_module_function = [](fmodset_t& _module_funcs, auto _v) {
        if(!fixed_module_functions.at(&_module_funcs)) _module_funcs.emplace(_v);
    };

    auto _add_overlapping = [](module_t* mitr, procedure_t* pitr) {
        ROCPROFSYS_ADD_LOG_ENTRY("Checking if procedure", get_name(pitr), "in module",
                                 get_name(mitr), "is overlapping");
        if(!pitr->isInstrumentable()) return;
        std::vector<procedure_t*> _overlapping{};
        if(pitr->findOverlapping(_overlapping))
        {
            ROCPROFSYS_ADD_LOG_ENTRY("Adding overlapping procedure", get_name(pitr),
                                     "and module", get_name(mitr));
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

    if(app_functions && !app_functions->empty())
    {
        for(auto* itr : *app_functions)
        {
            if(itr->getModule())
            {
                functions.emplace(itr);
                modules.emplace(itr->getModule());
            }
        }
        verbprintf(2, "Adding %zu procedures found in the app image...\n",
                   functions.size());
        for(auto* itr : functions)
        {
            if(itr->isInstrumentable() || (simulate && include_uninstr))
            {
                module_t* mod    = itr->getModule();
                auto      _modfn = module_function{ mod, itr };
                module_names.insert(_modfn.module_name);
                _insert_module_function(available_module_functions, _modfn);
                _add_overlapping(mod, itr);
            }
        }
    }
    else
    {
        verbprintf(
            0, "Warning! No functions in application. Enabling parsing all modules...\n");
        parse_all_modules = true;
    }

    if(parse_all_modules && app_modules && !app_modules->empty())
    {
        for(auto* itr : *app_modules)
            modules.emplace(itr);

        verbprintf(2,
                   "Adding the procedures from %zu modules found in the app image...\n",
                   modules.size());
        for(auto* itr : modules)
        {
            auto* procedures = itr->getProcedures(include_uninstr);
            if(procedures)
            {
                verbprintf(2, "Processing %zu procedures found in the %s module...\n",
                           procedures->size(), get_name(itr).data());
                for(auto* pitr : *procedures)
                {
                    if(!pitr->isInstrumentable() && !simulate && !include_uninstr)
                        continue;
                    functions.emplace(pitr);
                    auto _modfn = module_function{ itr, pitr };
                    module_names.insert(_modfn.module_name);
                    _insert_module_function(available_module_functions, _modfn);
                    _add_overlapping(itr, pitr);
                }
            }
        }
    }
    else if(parse_all_modules)
    {
        verbprintf(0, "Warning! No modules in application...\n");
    }

    verbprintf(1, "\n");
    verbprintf(1, "Found %zu functions in %zu modules in instrumentation target\n",
               functions.size(), modules.size());

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

    dump_info("available", available_module_functions, 1, werror, "available",
              print_formats);
    dump_info("overlapping", overlapping_module_functions, 1, werror,
              "overlapping_module_functions", print_formats);

    //----------------------------------------------------------------------------------//
    //
    //  Get the derived type of the address space
    //
    //----------------------------------------------------------------------------------//

    is_static_exe = addr_space->isStaticExecutable();

    ROCPROFSYS_ADD_LOG_ENTRY("address space is", (is_static_exe) ? "" : "not",
                             "a static executable");
    if(binary_rewrite)
        app_binary = static_cast<BPatch_binaryEdit*>(addr_space);
    else
        app_thread = static_cast<BPatch_process*>(addr_space);

    is_attached = (_pid >= 0 && app_thread != nullptr);

    ROCPROFSYS_ADD_LOG_ENTRY("address space is attached:", is_attached);

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
            ROCPROFSYS_ADD_LOG_ENTRY("Getting the absolute lib filepath to", _libname);
            _libname = get_realpath(get_absolute_lib_filepath(_libname));
            _tried_libs += string_t("|") + _libname;
            verbprintf(1, "loading library: '%s'...\n", _libname.c_str());
            result = (addr_space->loadLibrary(_libname.c_str()) != nullptr);
            verbprintf(2, "loadLibrary(%s) result = %s\n", _libname.c_str(),
                       (result) ? "success" : "failure");
            if(result)
            {
                ROCPROFSYS_ADD_LOG_ENTRY("Using library:", _libname);
                break;
            }
        }
        if(!result)
        {
            errprintf(-127,
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
        ROCPROFSYS_ADD_LOG_ENTRY("Using library:", lnames);
        return lnames;
    };

    //----------------------------------------------------------------------------------//
    //
    //  find _init and _fini before loading instrumentation library!
    //  These will be used for initialization and finalization if main is not found
    //
    //----------------------------------------------------------------------------------//

    auto* main_func       = find_function(app_image, main_fname.c_str());
    auto* user_start_func = find_function(app_image, "rocprofsys_user_start_trace",
                                          { "rocprofsys_user_start_thread_trace" });
    auto* user_stop_func  = find_function(app_image, "rocprofsys_user_stop_trace",
                                          { "rocprofsys_user_stop_thread_trace" });
#if ROCPROFSYS_USE_MPI > 0 || ROCPROFSYS_USE_MPI_HEADERS > 0
    // if any of the below MPI functions are found, enable MPI support
    for(const auto* itr :
        { "MPI_Init", "MPI_Init_thread", "MPI_Finalize", "MPI_Comm_rank", "MPI_Comm_size",
          "MPI_INIT", "mpi_init", "mpi_init_", "mpi_init__", "MPI_INIT_THREAD",
          "mpi_init_thread", "mpi_init_thread_", "mpi_init_thread__" })
    {
        if(find_function(app_image, itr) != nullptr)
        {
            verbprintf(0, "Found '%s' in '%s'. Enabling MPI support...\n", itr, _cmdv[0]);
            use_mpi = true;
            break;
        }
    }
#endif

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

    verbprintf(0, "Finding instrumentation functions...\n");

    auto* init_func      = find_function(app_image, "rocprofsys_init");
    auto* fini_func      = find_function(app_image, "rocprofsys_finalize");
    auto* env_func       = find_function(app_image, "rocprofsys_set_env");
    auto* mpi_func       = find_function(app_image, "rocprofsys_set_mpi");
    auto* entr_trace     = find_function(app_image, "rocprofsys_push_trace");
    auto* exit_trace     = find_function(app_image, "rocprofsys_pop_trace");
    auto* reg_src_func   = find_function(app_image, "rocprofsys_register_source");
    auto* reg_cov_func   = find_function(app_image, "rocprofsys_register_coverage");
    auto* set_instr_func = find_function(app_image, "rocprofsys_set_instrumented");

    if(!main_func && main_fname == "main") main_func = find_function(app_image, "_main");

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
        _pos = _name.find("librocprof-sys-");
        if(_pos != npos_v)
            _name = _name.erase(_pos, std::string("librocprof-sys-").length());
        _pos = _name.find("lib");
        if(_pos == 0) _name = _name.substr(_pos + std::string("lib").length());
        while((_pos = _name.find('-')) != npos_v)
            _name.replace(_pos, 1, "_");

        verbprintf(2,
                   "Supplemental instrumentation library '%s' is named '%s' after "
                   "removing everything before last '/', everything after first '.', and "
                   "'librocprof-sys-'...\n",
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
        if(add_instr_library(_name, TIMEMORY_JOIN("", "rocprofsys_register_" + _name),
                             TIMEMORY_JOIN("", "rocprofsys_deregister_" + _name)))
            continue;

    found_instr_functions:
        continue;
    }

    //----------------------------------------------------------------------------------//
    //
    //  Check for any issues finding the required functions
    //
    //----------------------------------------------------------------------------------//

    using pair_t = std::pair<procedure_t*, string_t>;

    for(const auto& itr : { pair_t{ entr_trace, "rocprofsys_push_trace" },
                            pair_t{ exit_trace, "rocprofsys_pop_trace" },
                            pair_t{ init_func, "rocprofsys_init" },
                            pair_t{ fini_func, "rocprofsys_finalize" },
                            pair_t{ env_func, "rocprofsys_set_env" },
                            pair_t{ set_instr_func, "rocprofsys_set_instrumented" },
                            pair_t{ reg_src_func, "rocprofsys_register_source" },
                            pair_t{ reg_cov_func, "rocprofsys_register_coverage" } })
    {
        if(!itr.first)
        {
            errprintf(-1, "could not find required function :: '%s'\n",
                      itr.second.c_str());
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Find the entry/exit point of either the main (if executable) or the _init
    //  and _fini functions of the library
    //
    //----------------------------------------------------------------------------------//

    std::vector<point_t*>* main_entr_points = nullptr;
    std::vector<point_t*>* main_exit_points = nullptr;

    if(main_func)
    {
        verbprintf(2, "Finding main function entry/exit... ");
        main_entr_points = main_func->findPoint(BPatch_entry);
        main_exit_points = main_func->findPoint(BPatch_exit);
        verbprintf(2, "Done\n");
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

    auto main_call_args = rocprofsys_call_expr(main_sign.get());
    auto init_call_args = rocprofsys_call_expr(instr_mode, binary_rewrite, "");
    auto fini_call_args = rocprofsys_call_expr();
    auto umpi_call_args = rocprofsys_call_expr(use_mpi, is_attached);
    auto none_call_args = rocprofsys_call_expr();
    auto set_instr_args = rocprofsys_call_expr(instr_mode_v_int);

    verbprintf(2, "Done\n");
    verbprintf(2, "Getting call snippets... ");

    auto init_call      = init_call_args.get(init_func);
    auto fini_call      = fini_call_args.get(fini_func);
    auto umpi_call      = umpi_call_args.get(mpi_func);
    auto set_instr_call = set_instr_args.get(set_instr_func);
    auto main_beg_call  = main_call_args.get(entr_trace);

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
        if(_libname.empty()) _libname = get_absolute_lib_filepath(itr);
    }
    for(auto&& itr : staticlibname)
    {
        if(_libname.empty()) _libname = get_absolute_lib_filepath(itr);
    }
    if(_libname.empty()) _libname = "librocprof-sys-dl.so";

    if(!binary_rewrite && !is_attached) env_vars.clear();

    env_vars.emplace_back(
        TIMEMORY_JOIN('=', "ROCPROFSYS_INIT_ENABLED",
                      (user_start_func && user_stop_func) ? "OFF" : "ON"));
    env_vars.emplace_back(TIMEMORY_JOIN('=', "ROCPROFSYS_USE_MPIP",
                                        (binary_rewrite && use_mpi) ? "ON" : "OFF"));
    if(use_mpi) env_vars.emplace_back(TIMEMORY_JOIN('=', "ROCPROFSYS_USE_PID", "ON"));

    env_vars.emplace_back(
        TIMEMORY_JOIN('=', "ROCPROFSYS_SCRIPT_PATH", _omni_internal_libexec_path));

    for(auto& itr : env_vars)
    {
        auto _pos = itr.find('=');
        if(_pos == std::string::npos)
        {
            errprintf(0, "environment variable %s not in form VARIABLE=VALUE\n",
                      itr.c_str());
        }
        auto _var = itr.substr(0, _pos);
        auto _val = itr.substr(_pos + 1);
        tim::set_env(_var, _val);
        auto _expr = rocprofsys_call_expr(_var, _val);
        env_variables.emplace_back(_expr.get(env_func));
    }

    //----------------------------------------------------------------------------------//
    //
    //  Configure the initialization and finalization routines
    //
    //----------------------------------------------------------------------------------//

    // call into rocprof-sys-dl to notify that instrumentation is occurring
    if(binary_rewrite)
    {
        init_names.emplace(init_names.begin(), set_instr_call.get());
    }

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
    if(!binary_rewrite && init_call) init_names.emplace_back(init_call.get());
    if(is_attached && main_func && main_beg_call)
        init_names.emplace_back(main_beg_call.get());

    for(const auto& itr : end_expr)
        if(itr.second) fini_names.emplace_back(itr.second.get());
    if(fini_call) fini_names.emplace_back(fini_call.get());

    //----------------------------------------------------------------------------------//
    //
    //  Sort the available module functions into appropriate containers
    //
    //----------------------------------------------------------------------------------//

    if(instr_mode != "sampling")
    {
        for(const auto& itr : available_module_functions)
        {
            if(itr.should_instrument())
            {
                _insert_module_function(instrumented_module_functions, itr);
            }
            else
            {
                _insert_module_function(excluded_module_functions, itr);
            }
            if(coverage_mode != CODECOV_NONE)
            {
                if(itr.should_coverage_instrument())
                    _insert_module_function(coverage_module_functions, itr);
            }
            if(itr.is_overlapping())
                _insert_module_function(overlapping_module_functions, itr);
        }
    }
    else
    {
        // in sampling mode, we instrument either main or add init and fini callbacks
        if(main_func)
            _insert_module_function(instrumented_module_functions,
                                    module_function{ main_func->getModule(), main_func });

        for(const auto& itr : available_module_functions)
        {
            _insert_module_function(excluded_module_functions, itr);
            if(coverage_mode != CODECOV_NONE)
            {
                if(itr.should_coverage_instrument())
                    _insert_module_function(coverage_module_functions, itr);
            }
            if(itr.is_overlapping())
                _insert_module_function(overlapping_module_functions, itr);
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Insert the initialization and finalization routines into the main entry and
    //  exit points
    //
    //----------------------------------------------------------------------------------//

    auto _objs = std::vector<object_t*>{};
    addr_space->getImage()->getObjects(_objs);
    auto _init_sequence = sequence_t{ init_names };
    auto _fini_sequence = sequence_t{ fini_names };

    if(!is_attached)
    {
        auto _insert_init_callbacks = std::function<bool()>{};
        auto _insert_init_snippets  = std::function<bool()>{};

        _insert_init_snippets = [&]() {
            if(main_entr_points)
            {
                verbprintf(1, "Adding main entry snippets...\n");
                addr_space->insertSnippet(_init_sequence, *main_entr_points);
            }
            else
            {
                errprintf(0, "Dyninst error inserting main entry snippets: no main entry "
                             "points\n");
                return false;
            }
            return true;
        };

        _insert_init_callbacks = [&]() {
            verbprintf(1, "Adding main init callbacks...\n");
            size_t _ninits = 0;
            for(auto* itr : _objs)
            {
                if(itr->name().find("librocprof-sys") != std::string::npos) continue;
                try
                {
                    verbprintf(2, "Adding main init callbacks (via %s)...\n",
                               itr->name().c_str());
                    if(itr->insertInitCallback(_init_sequence))
                    {
                        ++_ninits;
                    }
                } catch(std::runtime_error& _e)
                {
                    errprintf(0, "Dyninst error inserting init callback: %s\n",
                              _e.what());
                }
            }
            return (_ninits > 0);
        };

        if(binary_rewrite)
        {
            if(!_insert_init_callbacks()) _insert_init_snippets();
        }
        else
        {
            if(!_insert_init_snippets()) _insert_init_callbacks();
        }
    }

    if(main_exit_points)
    {
        verbprintf(1, "Adding main exit snippets...\n");
        // insert_instr(addr_space, *main_exit_points, _fini_sequence, BPatch_exit);
        addr_space->insertSnippet(_fini_sequence, *main_exit_points);
    }
    else
    {
        for(auto* itr : _objs)
        {
            try
            {
                itr->insertFiniCallback(_fini_sequence);
            } catch(std::runtime_error& _e)
            {
                errprintf(0, "Dyninst error inserting fini callback: %s\n", _e.what());
            }
        }
    }

    //----------------------------------------------------------------------------------//
    //
    //  Actually insert the instrumentation into the procedures
    //
    //----------------------------------------------------------------------------------//
    if(app_thread)
    {
        verbprintf(2, "Beginning insertion set...\n");
        addr_space->beginInsertionSet();
    }

    verbprintf(2, "Beginning instrumentation loop...\n");
    auto _report_info = [](int _lvl, const string_t& _action, const string_t& _type,
                           const string_t& _reason, const string_t& _name,
                           const std::string& _extra = {}) {
        static std::map<std::string, strset_t> already_reported{};
        auto _key = TIMEMORY_JOIN('_', _type, _action, _reason, _name, _extra);
        if(already_reported[_key].count(_name) == 0)
        {
            verbprintf(_lvl, "[%s][%s] %s :: '%s'", _type.c_str(), _action.c_str(),
                       _reason.c_str(), _name.c_str());
            if(!_extra.empty()) verbprintf_bare(_lvl, " (%s)", _extra.c_str());
            verbprintf_bare(_lvl, "...\n");
            already_reported[_key].insert(_name);
        }
    };

    if(instr_mode != "coverage")
    {
        auto      _pass_info        = std::map<std::string, std::pair<size_t, size_t>>{};
        const int _pass_verbose_lvl = 0;
        for(const auto& itr : instrumented_module_functions)
        {
            if(itr.function == main_func) continue;
            auto _count = itr(addr_space, entr_trace, exit_trace);
            _pass_info[itr.module_name].first += _count.first;
            _pass_info[itr.module_name].second += _count.second;

            for(const auto& mitr : itr.messages)
                _report_info(std::get<0>(mitr), std::get<1>(mitr), std::get<2>(mitr),
                             std::get<3>(mitr), std::get<4>(mitr));
        }

        // report the trace instrumented functions
        for(auto& itr : _pass_info)
        {
            auto _valid = (verbose_level >= _pass_verbose_lvl ||
                           (itr.second.first + itr.second.second) > 0);
            if(!_valid) continue;
            verbprintf(_pass_verbose_lvl, "%4zu instrumented funcs in %s\n",
                       itr.second.first, itr.first.c_str());
            _valid = (loop_level_instr &&
                      (verbose_level >= _pass_verbose_lvl || itr.second.second > 0));
            if(_valid)
            {
                verbprintf(_pass_verbose_lvl, "%4zu instrumented loops in %s\n",
                           itr.second.second, itr.first.c_str());
            }
        }
    }

    if(coverage_mode != CODECOV_NONE)
    {
        std::map<std::string, std::pair<size_t, size_t>> _covr_info        = {};
        const int                                        _covr_verbose_lvl = 1;
        for(const auto& itr : coverage_module_functions)
        {
            if(itr.function == main_func) continue;
            itr.register_source(addr_space, reg_src_func, *main_entr_points);
            auto _count = itr.register_coverage(addr_space, reg_cov_func);
            _covr_info[itr.module_name].first += _count.first;
            _covr_info[itr.module_name].second += _count.second;

            for(const auto& mitr : itr.messages)
                _report_info(std::get<0>(mitr), std::get<1>(mitr), std::get<2>(mitr),
                             std::get<3>(mitr), std::get<4>(mitr));
        }

        // report the coverage instrumented functions
        for(auto& itr : _covr_info)
        {
            auto _valid = (verbose_level > _covr_verbose_lvl ||
                           (itr.second.first + itr.second.second) > 0);
            if(!_valid) continue;
            switch(coverage_mode)
            {
                case CODECOV_NONE:
                {
                    break;
                }
                case CODECOV_FUNCTION:
                {
                    verbprintf(_covr_verbose_lvl, "%4zu coverage functions in %s\n",
                               itr.second.first, itr.first.c_str());
                    break;
                }
                case CODECOV_BASIC_BLOCK:
                {
                    verbprintf(_covr_verbose_lvl, "%4zu coverage basic blocks in %s\n",
                               itr.second.second, itr.first.c_str());
                    break;
                }
            }
        }
    }
    verbprintf(1, "\n");

    if(app_thread)
    {
        verbprintf(2, "Finalizing insertion set...\n");
        bool modified = true;
        bool success  = addr_space->finalizeInsertionSet(true, &modified);
        if(!success)
        {
            verbprintf(
                1,
                "Using insertion set failed. Restarting with individual insertion...\n");
            auto _execute_batch = [&addr_space, &entr_trace, &exit_trace](size_t _beg,
                                                                          size_t _end) {
                verbprintf(1, "Instrumenting batch of functions [%lu, %lu)\n",
                           (unsigned long) _beg, (unsigned long) _end);
                addr_space->beginInsertionSet();
                auto itr = instrumented_module_functions.begin();
                std::advance(itr, _beg);
                for(size_t i = _beg; i < _end; ++i, ++itr)
                    (*itr)(addr_space, entr_trace, exit_trace);
                bool _modified = true;
                bool _success  = addr_space->finalizeInsertionSet(true, &_modified);
                return _success;
            };

            auto execute_batch = [&_execute_batch, &addr_space, &entr_trace,
                                  &exit_trace](size_t _beg) {
                if(!_execute_batch(_beg, _beg + batch_size))
                {
                    verbprintf(1,
                               "Batch instrumentation of functions [%lu, %lu) failed. "
                               "Beginning non-batched instrumentation for this set\n",
                               (unsigned long) _beg, (unsigned long) _beg + batch_size);
                    auto itr  = instrumented_module_functions.begin();
                    auto _end = instrumented_module_functions.end();
                    std::advance(itr, _beg);
                    for(size_t i = _beg; i < _beg + batch_size && itr != _end; ++i, ++itr)
                    {
                        (*itr)(addr_space, entr_trace, exit_trace);
                    }
                }
                return _beg + batch_size;
            };

            size_t nidx = 0;
            while(nidx < instrumented_module_functions.size())
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

    dump_info("available", available_module_functions, 0, werror,
              "available_module_functions", print_formats);
    dump_info("instrumented", instrumented_module_functions, 0, werror,
              "instrumented_module_functions", print_formats);
    dump_info("excluded", excluded_module_functions, 0, werror,
              "excluded_module_functions", print_formats);
    if(coverage_mode != CODECOV_NONE)
        dump_info("coverage", coverage_module_functions, 0, werror,
                  "coverage_module_functions", print_formats);
    dump_info("overlapping", overlapping_module_functions, 0, werror,
              "overlapping_module_functions", print_formats);

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
                _insert(itr.module_name, TIMEMORY_JOIN("", "[", itr.module_name, "]"));
        }
        else if(_mode == "functions")
        {
            for(const auto& itr : _modset)
                _insert(itr.module_name, TIMEMORY_JOIN("", "[", itr.function_name, "][",
                                                       itr.num_instructions, "]"));
        }
        else if(_mode == "functions+")
        {
            for(const auto& itr : _modset)
                _insert(itr.module_name, TIMEMORY_JOIN("", "[", itr.signature.get(), "][",
                                                       itr.num_instructions, "]"));
        }
        else if(_mode == "pair")
        {
            for(const auto& itr : _modset)
            {
                _insert(itr.module_name, TIMEMORY_JOIN("", "[", itr.module_name,
                                                       "] --> [", itr.function_name, "][",
                                                       itr.num_instructions, "]"));
            }
        }
        else if(_mode == "pair+")
        {
            for(const auto& itr : _modset)
            {
                _insert(itr.module_name, TIMEMORY_JOIN("", "[", itr.module_name,
                                                       "] --> [", itr.signature.get(),
                                                       "][", itr.num_instructions, "]"));
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
    if(!print_coverage.empty())
        _dump_info("coverage", print_coverage, coverage_module_functions);
    if(!print_overlapping.empty())
        _dump_info("overlapping", print_overlapping, overlapping_module_functions);

    if(simulate) exit(EXIT_SUCCESS);

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
            verbprintf(0, "Consider instrumenting the relevant libraries...\n");
            verbprintf(0, "\n");

            auto cmdv_envp = std::array<char*, 2>{};
            cmdv_envp.fill(nullptr);
            cmdv_envp.at(0) = strdup("LD_TRACE_LOADED_OBJECTS=1");
            auto ldd        = tim::popen::popen(cmdv0.c_str(), nullptr, cmdv_envp.data());
            auto linked_libs = tim::popen::read_ldd_fork(ldd);
            auto perr        = tim::popen::pclose(ldd);
            for(auto& itr : cmdv_envp)
                ::free(itr);

            if(perr != 0) perror("Error in rocprofsys_fork");

            for(const auto& itr : linked_libs)
                verbprintf(0, "\t%s\n", itr.c_str());

            verbprintf(0, "\n");
        }
    }
    else
    {
        verbprintf(0, "Executing...\n");

#define WAITPID_DEBUG_MESSAGE(QUERY)                                                     \
    {                                                                                    \
        QUERY;                                                                           \
        verbprintf(3,                                                                    \
                   "waitpid (%i, %i) returned [%s:%i] :: %s :: code: %i, status %i. "    \
                   "WIFEXITED(status) = "                                                \
                   "%i, WIFSIGNALED(status) = %i\n",                                     \
                   cpid, w, __FILE__, __LINE__, #QUERY, code, status, WIFEXITED(status), \
                   WIFSIGNALED(status));                                                 \
    }

        auto _compute_exit_code = [app_thread, &code]() {
            if(app_thread->terminationStatus() == ExitedNormally)
            {
                if(app_thread->isTerminated()) verbprintf(0, "End of rocprof-sys\n");
            }
            else if(app_thread->terminationStatus() == ExitedViaSignal)
            {
                auto sign = app_thread->getExitSignal();
                fprintf(stderr, "\nApplication exited with signal: %i\n", int(sign));
            }
            code = app_thread->getExitCode();
        };

        if(!app_thread->isTerminated() && !is_attached)
        {
            pid_t cpid   = app_thread->getPid();
            int   status = 0;
            app_thread->detach(true);
            do
            {
                status  = 0;
                pid_t w = waitpid(cpid, &status, WUNTRACED);
                if(w == -1)
                {
                    perror("waitpid");
                    exit(EXIT_FAILURE);
                }

                if(WIFEXITED(status))
                {
                    WAITPID_DEBUG_MESSAGE(code = WEXITSTATUS(status));
                }
                else if(WIFSIGNALED(status))
                {
                    WAITPID_DEBUG_MESSAGE(code = WTERMSIG(status));
                }
                else if(WIFSTOPPED(status))
                {
                    WAITPID_DEBUG_MESSAGE(code = WSTOPSIG(status));
                }
                else if(WIFCONTINUED(status))
                {
                    WAITPID_DEBUG_MESSAGE(code = WIFCONTINUED(status));
                }
            } while(WIFEXITED(status) == 0 && WIFSIGNALED(status) == 0);
        }
        else if(!app_thread->isTerminated() && is_attached)
        {
            bpatch->setDebugParsing(false);
            bpatch->setDelayedParsing(true);
            verbprintf(1, "Executing initial snippets...\n");
            for(auto* itr : init_names)
                app_thread->oneTimeCode(*itr);

            app_thread->continueExecution();
            while(!app_thread->isTerminated())
            {
                while(bpatch->waitForStatusChange())
                    app_thread->continueExecution();
            }
            _compute_exit_code();
        }
        else
        {
            _compute_exit_code();
        }
    }

    // cleanup
    for(int i = 0; i < argc; ++i)
        free(_argv[i]);
    delete[] _argv;
    for(int i = 0; i < _cmdc; ++i)
        free(_cmdv[i]);
    delete[] _cmdv;

    verbprintf(0, "End of rocprof-sys\n");
    verbprintf((code != 0) ? 0 : 1, "Exit code: %i\n", code);

    if(log_ofs)
    {
        verbprintf(2, "Closing log file...\n");
        log_ofs->close();
    }
    return code;
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

    std::vector<point_t*>* _points = nullptr;

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

std::tuple<size_t, size_t>
query_instr(procedure_t* funcToInstr, procedure_loc_t traceLoc, flow_graph_t* cfGraph,
            basic_loop_t* loopToInstrument)
{
    module_t* module = funcToInstr->getModule();
    if(!module) return { 0, 0 };

    if(!cfGraph) cfGraph = funcToInstr->getCFG();

    std::vector<point_t*>* _points = nullptr;

    if((cfGraph && loopToInstrument) ||
       (traceLoc == BPatch_locLoopEntry || traceLoc == BPatch_locLoopExit))
    {
        if(!cfGraph) throw std::runtime_error("No control flow graph");
        if(!loopToInstrument) throw std::runtime_error("No loop to instrument");

        if(traceLoc == BPatch_entry || traceLoc == BPatch_locLoopEntry)
        {
            _points = cfGraph->findLoopInstPoints(BPatch_locLoopEntry, loopToInstrument);
        }
        else if(traceLoc == BPatch_exit || traceLoc == BPatch_locLoopExit)
        {
            _points = cfGraph->findLoopInstPoints(BPatch_locLoopExit, loopToInstrument);
        }
        else
        {
            throw std::runtime_error("unsupported trace location :: " +
                                     std::to_string(traceLoc));
        }
    }
    else
    {
        _points = funcToInstr->findPoint(traceLoc);
    }

    if(_points == nullptr) return { 0, 0 };
    if(_points->empty()) return { 0, 0 };

    size_t _n = 0;
    size_t _t = 0;
    for(auto& itr : *_points)
    {
        if(itr)
        {
            ++_n;
            if(itr->usesTrap_NP()) ++_t;
        }
    }

    return { _n, _t };
}

namespace
{
//======================================================================================//
//
std::string
canonicalize(std::string _path)
{
    if(_path.find("./") == 0)
        _path = _path.replace(0, 1, get_cwd());
    else if(_path.find("../") == 0)
        _path = _path.insert(0, get_cwd() + "/");

    auto _leading_dash = (_path.find('/') == 0);
    auto _pieces       = tim::delimit(_path, "/");
    std::reverse(_pieces.begin(), _pieces.end());
    auto _tree = std::vector<std::string>{};
    for(size_t i = 0; i < _pieces.size(); ++i)
    {
        const auto& itr = _pieces.at(i);
        if(itr == ".")
        {
            continue;
        }
        else if(itr == "..")
            ++i;
        else
            _tree.emplace_back(itr);
    }
    std::reverse(_tree.begin(), _tree.end());
    auto _cpath = std::string{ (_leading_dash) ? "/" : "" };
    for(size_t i = 0; i < _tree.size() - 1; ++i)
        _cpath += _tree.at(i) + "/";
    _cpath += _tree.back();
    return _cpath;
}

//======================================================================================//
//
std::string
absolute(std::string _path)
{
    if(_path.find('/') == 0) return canonicalize(_path);
    return canonicalize(JOIN('/', get_cwd(), _path));
}

//======================================================================================//
//
std::string
get_absolute_filepath(std::string _name, const strvec_t& _search_paths)
{
    if(!_name.empty() && (!exists(_name) || !is_file(_name)))
    {
        auto _orig = _name;
        for(auto itr : _search_paths)
        {
            if(!is_directory(itr) || is_file(itr)) itr = filepath::dirname(itr);

            auto _exists = false;
            ROCPROFSYS_ADD_LOG_ENTRY("searching", itr, "for", _name);
            for(const auto& pitr :
                { absolute(JOIN('/', itr, _name)),
                  absolute(JOIN('/', itr, filepath::basename(_name))) })
            {
                _exists = exists(pitr) && is_file(pitr);
                if(_exists)
                {
                    _name = pitr;
                    verbprintf(1, "Resolved '%s' to '%s'...\n", _orig.c_str(),
                               _name.c_str());
                    break;
                }
            }
            if(_exists) break;
        }

        if(!exists(_name))
        {
            using array_config_t = timemory::join::array_config;
            auto _search_paths_v =
                timemory::join::join(array_config_t{ ", ", "", "" }, bin_search_paths);
            verbprintf(
                0, "Warning! File path to '%s' could not be determined... search: %s\n",
                _name.c_str(), _search_paths_v.c_str());
        }
    }
    else if(!_name.empty())
    {
        auto _orig = _name;
        _name      = absolute(_name);
        verbprintf(1, "Resolved '%s' to '%s'...\n", _orig.c_str(), _name.c_str());
    }

    return _name;
}

//======================================================================================//
//
std::string
get_absolute_filepath(std::string _name)
{
    auto _search_paths  = strvec_t{};
    auto _combine_paths = std::vector<strvec_t>{ bin_search_paths, lib_search_paths };
    auto _base_name     = std::string_view{ filepath::basename(_name) };
    // if the name looks like a library, put the lib_search_paths first
    if(_base_name.find("lib") == 0 || _base_name.find(".so") != std::string::npos ||
       _base_name.find(".a") != std::string::npos)
        std::reverse(_combine_paths.begin(), _combine_paths.end());
    _search_paths.reserve(bin_search_paths.size() + lib_search_paths.size());
    for(const auto& pitr : _combine_paths)
        for(const auto& itr : pitr)
            _search_paths.emplace_back(itr);

    return get_absolute_filepath(std::move(_name), _search_paths);
}

//======================================================================================//
//
std::string
get_absolute_exe_filepath(std::string exe_name)
{
    return get_absolute_filepath(std::move(exe_name), bin_search_paths);
}

//======================================================================================//
//
std::string
get_absolute_lib_filepath(std::string lib_name)
{
    auto _orig_name = lib_name;
    lib_name        = get_absolute_filepath(std::move(lib_name), lib_search_paths);
    if(_orig_name == lib_name && !exists(lib_name) &&
       lib_name.find(".so") == std::string::npos &&
       lib_name.find(".a") == std::string::npos)
    {
        lib_name = get_absolute_filepath(lib_name + ".so", lib_search_paths);
    }
    return lib_name;
}

bool
exists(const std::string& name)
{
    return filepath::exists(absolute(name));
}

bool
is_file(std::string _name)
{
    _name = get_realpath(_name);
    struct stat buffer;
    return (stat(_name.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode) != 0);
}

bool
is_directory(std::string _name)
{
    _name = get_realpath(_name);
    struct stat buffer;
    return (stat(_name.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode) != 0);
}

std::string
get_realpath(const std::string& _f)
{
    return filepath::realpath(_f, nullptr, false);
}

std::string
get_cwd()
{
#if defined(__GNUC__)
    auto* _cwd_c = getcwd(nullptr, 0);
    auto  _cwd_v = std::string{ _cwd_c };
    free(_cwd_c);
    return _cwd_v;
#else
    char  cwd[PATH_MAX];
    auto* _cwd_v = getcwd(cwd, PATH_MAX);
    return (_cwd_v) ? std::string{ _cwd_v } : get_env<std::string>("PWD");
#endif
}

using tim::dirname;

void
find_dyn_api_rt()
{
#if defined(ROCPROFSYS_BUILD_DYNINST)
    std::string _dyn_api_rt_base =
        (binary_rewrite) ? "librocprof-sys-rt" : "libdyninstAPI_RT";
#else
    std::string _dyn_api_rt_base = "libdyninstAPI_RT";
#endif

    auto _dyn_api_rt_env =
        tim::get_env<std::string>("DYNINSTAPI_RT_LIB", _dyn_api_rt_base + ".so");
    auto _dyn_api_rt_abs = get_absolute_lib_filepath(_dyn_api_rt_env);

    if(!exists(_dyn_api_rt_abs))
        _dyn_api_rt_abs = get_absolute_lib_filepath(_dyn_api_rt_base + ".a");

    if(exists(_dyn_api_rt_abs))
    {
        namespace join = ::timemory::join;
        tim::set_env<string_t>("DYNINSTAPI_RT_LIB", _dyn_api_rt_abs, 1);
        tim::set_env<string_t>("DYNINST_REWRITER_PATHS",
                               join::join(join::array_config{ ":", "", "" },
                                          dirname(_dyn_api_rt_abs), lib_search_paths),
                               1);
    }

    auto _v = tim::get_env<string_t>("DYNINSTAPI_RT_LIB", "");
    verbprintf(0, "DYNINST_API_RT: %s\n", (_v.empty()) ? "<unknown>" : _v.c_str());
}
}  // namespace
