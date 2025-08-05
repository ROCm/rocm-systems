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

#include "avail.hpp"
#include "common.hpp"
#include "common/defines.h"
#include "component_categories.hpp"
#include "defines.hpp"
#include "enumerated_list.hpp"
#include "generate_config.hpp"
#include "get_availability.hpp"
#include "info_type.hpp"

#include "api.hpp"
#include "core/config.hpp"
#include "core/gpu.hpp"
#include "library/rocm.hpp"

#include <timemory/components.hpp>
#include <timemory/components/definition.hpp>
#include <timemory/components/placeholder.hpp>
#include <timemory/components/properties.hpp>
#include <timemory/components/skeletons.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/manager/manager.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/timemory.hpp>
#include <timemory/unwind/bfd.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/utility/utility.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(TIMEMORY_UNIX)
#    include <sys/ioctl.h>  // ioctl() and TIOCGWINSZ
#    include <unistd.h>     // for STDOUT_FILENO
#elif defined(TIMEMORY_WINDOWS)
#    include <windows.h>
#endif

using namespace tim;

//--------------------------------------------------------------------------------------//

namespace
{
template <typename IntArrayT, typename BoolArrayT>
IntArrayT
compute_max_columns(IntArrayT _widths, BoolArrayT _using);

template <typename Tp>
void
write_entry(std::ostream& os, const Tp& _entry, int64_t _w, bool center, bool mark);

template <typename Tp, typename IntArrayT, size_t N>
void
write_wrap_entry(std::ostream& os, const Tp& _entry, int64_t _w, bool center, bool mark,
                 size_t _idx, IntArrayT _breaks, std::array<bool, N> _use);

template <typename IntArrayT, size_t N>
string_t
banner(IntArrayT _breaks, std::array<bool, N> _use, char filler = '-', char delim = '|');

template <typename IntArrayT, size_t N>
string_t
wrap(size_t idx, IntArrayT _breaks, std::array<bool, N> _use, char filler = ' ',
     char delim = '|');
}  // namespace

template <size_t N = num_component_options>
void
write_component_info(std::ostream&, const array_t<bool, N>&, const array_t<bool, N>&,
                     const array_t<string_t, N>&);

template <size_t N = num_settings_options>
void
write_settings_info(std::ostream&, const array_t<bool, N>& = {},
                    const array_t<bool, N>& = {}, const array_t<string_t, N>& = {});

template <size_t N = num_hw_counter_options>
void
write_hw_counter_info(std::ostream&, const array_t<bool, N>& = {},
                      const array_t<bool, N>& = {}, const array_t<string_t, N>& = {});

namespace
{
// initialize HIP before main so that librocprof-sys is not HSA_TOOLS_LIB
int gpu_count = 0;

// statically allocated shared_ptrs to prevent use after free errors
auto timemory_manager      = tim::manager::master_instance();
auto timemory_hash_ids     = tim::hash::get_main_hash_ids();
auto timemory_hash_aliases = tim::hash::get_main_hash_aliases();
}  // namespace

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    (void) timemory_manager;       // suppress unused variables
    (void) timemory_hash_ids;      //
    (void) timemory_hash_aliases;  //

    tim::unwind::set_bfd_verbose(3);
    tim::set_env("ROCPROFSYS_INIT_TOOLING", "OFF", 1);
    rocprofsys_init_library();
    gpu_count = rocprofsys::gpu::device_count();

    std::set<std::string> _category_options = component_categories{}();
    {
        auto _settings = tim::settings::shared_instance();
        for(const auto& itr : *_settings)
        {
            if(exclude_setting(itr.second->get_env_name())) continue;
            auto _categories = itr.second->get_categories();
            if(_categories.find("native") != _categories.end())
            {
                _categories.erase("native");
                _categories.emplace("timemory");
                itr.second->set_categories(_categories);
            }
            for(const auto& eitr : itr.second->get_categories())
            {
                _category_options.emplace(TIMEMORY_JOIN("::", "settings", eitr));
            }
        }
    }
    _category_options.emplace("hw_counters::CPU");
    _category_options.emplace("hw_counters::GPU");

    array_t<bool, TOTAL> options    = { false, false, false, false, false, false, false };
    array_t<string_t, TOTAL> fields = {};
    array_t<bool, TOTAL>     use_mark = {};

    std::string cols_via{};
    std::tie(num_cols, cols_via) = tim::utility::console::get_columns();
    std::string col_msg =
        ". default: " + std::to_string(num_cols) + " [via " + cols_via + "]";

    fields[VAL]      = "VALUE_TYPE";
    fields[ENUM]     = "ENUMERATION";
    fields[LANG]     = "C++ ALIAS / PYTHON ENUMERATION";
    fields[FNAME]    = "FILENAME";
    fields[CID]      = "STRING_IDS";
    fields[DESC]     = "DESCRIPTION";
    fields[CATEGORY] = "CATEGORY";

    use_mark[VAL]      = true;
    use_mark[ENUM]     = true;
    use_mark[LANG]     = true;
    use_mark[FNAME]    = false;
    use_mark[CID]      = false;
    use_mark[DESC]     = false;
    use_mark[CATEGORY] = false;

    bool include_settings    = false;
    bool include_components  = false;
    bool include_hw_counters = false;

    std::string file = {};

    parser_t parser("rocprof-sys-avail");

    parser.set_help_width(40);
    auto _cols = std::get<0>(tim::utility::console::get_columns());
    if(_cols > parser.get_help_width() + 8)
        parser.set_description_width(
            std::min<int>(_cols - parser.get_help_width() - 8, 120));

    parser.enable_help();
    parser.enable_version("rocprof-sys-avail", ROCPROFSYS_ARGPARSE_VERSION_INFO);

    parser.start_group("DEBUG");

    parser.add_argument({ "--monochrome" }, "Disable colorized output")
        .max_count(1)
        .dtype("bool")
        .action([&](parser_t& p) {
            auto _monochrome       = p.get<bool>("monochrome");
            tim::log::monochrome() = _monochrome;
            p.set_use_color(!_monochrome);
        });
    parser.add_argument({ "--debug" }, "Enable debug messages")
        .max_count(1)
        .action([](parser_t& p) { debug_msg = p.get<bool>("debug"); });
    parser.add_argument({ "--verbose" }, "Enable informational messages")
        .max_count(1)
        .action([](parser_t& p) {
            verbose_level = (p.get_count("verbose") == 0) ? 1 : p.get<int>("verbose");
        });

    parser.start_group("INFO");

    parser
        .add_argument({ "-S", "--settings", "--print-settings" },
                      "Display the runtime settings")
        .max_count(1);
    parser
        .add_argument({ "-C", "--components", "--print-components" },
                      "Only display the components data")
        .max_count(1);
    parser
        .add_argument({ "-H", "--hw-counters", "--print-hw-counters" },
                      "Write the available hardware counters")
        .max_count(1);

    parser.add_argument({ "-a", "--all" }, "Print all available info")
        .max_count(1)
        .action([&](parser_t& p) {
            all_info = p.get<bool>("all");
            if(all_info)
            {
                for(auto& itr : options)
                    itr = true;
                options[ENUM]       = false;
                options[LANG]       = false;
                include_components  = true;
                include_settings    = true;
                include_hw_counters = true;
            }
        });

    parser
        .add_argument({ "--advanced" },
                      "Print advanced settings not relevant to most use cases")
        .max_count(1)
        .action([](parser_t& p) { print_advanced = p.get<bool>("advanced"); });

    parser
        .add_argument({ "--list-categories" },
                      "List the available categories for --categories option")
        .count(0)
        .action([_category_options](parser_t&) {
            std::cout << "Categories:\n";
            for(const auto& itr : _category_options)
                std::cout << "    " << itr << "\n";
        });
    parser.add_argument({ "--list-keys" }, "List the output keys")
        .max_count(1)
        .action([](parser_t& p) {
            auto _list = p.get<bool>("list-keys");
            auto _show = p.get<bool>("expand-keys");
            if(_list)
            {
                auto _keys = tim::settings::output_keys(
                    tim::settings::shared_instance()->get_tag());
                std::tuple<size_t, size_t, size_t> _w = { 0, 0, 0 };
                for(const auto& itr : _keys)
                {
                    if(!is_selected(itr.key)) continue;
                    if(_show && !is_selected(itr.value)) continue;
                    std::get<0>(_w) = std::max(std::get<0>(_w), itr.key.length());
                    std::get<1>(_w) = std::max(std::get<1>(_w), itr.value.length());
                    std::get<2>(_w) = std::max(std::get<2>(_w), itr.description.length());
                }
                std::stringstream _msg{};
                _msg << std::left;

                if(markdown)
                {
                    _msg << "| " << std::setw(std::get<0>(_w) + 2) << "String";
                    if(_show) _msg << " | " << std::setw(std::get<1>(_w)) << "Value";
                    _msg << " | " << std::setw(std::get<2>(_w)) << "Encoding" << " |\n";

                    auto _dashes = [](int64_t _n) {
                        std::stringstream _dss{};
                        _dss.fill('-');
                        _dss << std::setw(_n + 2) << "";
                        return _dss.str();
                    };

                    _msg << "|" << _dashes(std::get<0>(_w) + 2);
                    if(_show) _msg << "|" << _dashes(std::get<1>(_w));
                    _msg << "|" << _dashes(std::get<2>(_w)) << "|\n";

                    for(const auto& itr : _keys)
                    {
                        if(!is_selected(itr.key)) continue;
                        if(_show && !is_selected(itr.value)) continue;
                        _msg << "| " << std::setw(std::get<0>(_w) + 2)
                             << TIMEMORY_JOIN("", "`", itr.key, "`");
                        if(_show)
                            _msg << " | " << std::setw(std::get<1>(_w)) << itr.value;
                        _msg << " | " << std::setw(std::get<2>(_w)) << itr.description
                             << " |\n";
                    }
                }
                else
                {
                    _msg << "Output Keys:\n" << std::left;
                    for(const auto& itr : _keys)
                    {
                        if(!is_selected(itr.key)) continue;
                        if(_show && !is_selected(itr.value)) continue;
                        if(_show)
                            _msg << "    " << std::setw(std::get<0>(_w)) << itr.key
                                 << "  ::  " << std::setw(std::get<1>(_w)) << itr.value
                                 << "  ::  " << std::setw(std::get<2>(_w))
                                 << itr.description << "\n";
                        else
                            _msg << "    " << std::setw(std::get<0>(_w)) << itr.key
                                 << "  ::  " << std::setw(std::get<2>(_w))
                                 << itr.description << "\n";
                    }
                }
                std::cout << _msg.str() << std::flush;
            }
        });
    parser
        .add_argument({ "--expand-keys" },
                      "Expand the output keys to their current values")
        .max_count(1)
        .action([](parser_t& p) { expand_keys = p.get<bool>("expand-keys"); });

    parser.start_group("FILTER");

    parser
        .add_argument({ "-A", "--available" },
                      "Only display available components/settings/hw-counters")
        .max_count(1)
        .action([](parser_t& p) { available_only = p.get<bool>("available"); });
    parser
        .add_argument({ "-r", "--filter" },
                      "Filter the output according to provided regex (egrep + "
                      "case-sensitive) [e.g. -r \"true\"]. Prefix "
                      "with '~' to suppress matches")
        .min_count(1)
        .dtype("list of strings")
        .action([](parser_t& p) { regex_keys = p.get<str_vec_t>("filter"); });
    parser
        .add_argument({ "-R", "--category-filter" },
                      "Filter the output according to provided regex w.r.t. the "
                      "categories (egrep + case-sensitive) [e.g. -r \"true\"]. Prefix "
                      "with '~' to suppress matches")
        .min_count(1)
        .dtype("list of strings")
        .action([](parser_t& p) {
            category_regex_keys = p.get<str_vec_t>("category-filter");
        });
    parser.add_argument({ "-i", "--ignore-case" }, "Ignore case when filtering")
        .max_count(1)
        .dtype("bool")
        .action([](parser_t& p) { case_insensitive = p.get<bool>("ignore-case"); });
    parser
        .add_argument({ "-p", "--hl", "--highlight" },
                      "Highlight regex matches (only available on UNIX)")
        .max_count(1)
        .action([](parser_t&) { regex_hl = true; });
    parser.add_argument({ "--alphabetical" }, "Sort the output alphabetically")
        .max_count(1)
        .action([](parser_t& p) { alphabetical = p.get<bool>("alphabetical"); });

    parser.start_group("COLUMN");

    parser.add_argument({ "-b", "--brief" }, "Suppress availability/value info")
        .max_count(1)
        .action([](parser_t& p) { force_brief = p.get<bool>("brief"); });
    parser.add_argument({ "-d", "--description" }, "Display the component description")
        .max_count(1);
    parser.add_argument({ "-s", "--string" }, "Display all acceptable string identifiers")
        .max_count(1);
    parser
        .add_argument({ "-v", "--value" },
                      "Display the component data storage value type")
        .max_count(1);
    parser
        .add_argument({ "-f", "--filename" },
                      "Display the output filename for the component")
        .max_count(1);
    parser
        .add_argument({ "-c", "--categories" },
                      "Display the category information (use --list-categories to see "
                      "the available categories)")
        .dtype("string")
        .action([&_category_options](parser_t& p) {
            process_categories(p, _category_options);
        });

    parser.start_group("DISPLAY");

    parser
        .add_argument({ "-w", "--column-width" },
                      "if w > 0, truncate any columns greater than this width")
        .count(1)
        .dtype("int")
        .action([](parser_t& p) { max_width = p.get<int32_t>("column-width"); });
    parser
        .add_argument(
            { "-W", "--max-total-width" },
            std::string{ "if W > 0, truncate the total width of all the columns to this "
                         "value. Set '-w 0 -W 0' to remove all truncation" } +
                col_msg)
        .set_default(num_cols)
        .count(1)
        .dtype("int")
        .action([](parser_t& p) { num_cols = p.get<int32_t>("max-total-width"); });

    parser.start_group("OUTPUT");

    std::string           _config_file = {};
    std::set<std::string> _config_fmts = {};

    parser
        .add_argument({ "-G", "--generate-config" },
                      "Dump a configuration to a specified file.")
        .max_count(1)
        .dtype("filename")
        .set_default(std::string{ "rocprof-sys-config" })
        .action([&_config_file](parser_t& _p) {
            auto _out =
                (_p.exists("output")) ? _p.get<std::string>("output") : std::string{};
            if(_p.get_count("generate-config") == 0 && !_out.empty())
                _config_file = _out;
            else
            {
                _config_file = _p.get<std::string>("generate-config");
                if(get_bool(_config_file, false) && !_out.empty()) _config_file = _out;
            }
        });
    parser.add_argument({ "-F", "--config-format" }, "Configuration file format")
        .min_count(1)
        .max_count(3)
        .choices({ "txt", "json", "xml" })
        .set_default(std::set<std::string>{ "txt" })
        .dtype("filename")
        .action([&_config_fmts](parser_t& _p) {
            _config_fmts = _p.get<std::set<std::string>>("config-format");
        });
    parser.add_argument({ "-O", "--output" }, "Write results to file")
        .count(1)
        .dtype("filename");
    parser.add_argument({ "-t", "--tag" }, "Set the %tag% to a custom value")
        .count(1)
        .action([](parser_t& p) {
            settings::instance()->set_tag(p.get<std::string>("tag"));
        });
    parser.add_argument({ "-M", "--markdown" }, "Write data in markdown")
        .max_count(1)
        .action([](parser_t& p) { markdown = p.get<bool>("markdown"); });
    parser.add_argument({ "--csv" }, "Write data in csv")
        .max_count(1)
        .action([](parser_t& p) {
            csv = p.get<bool>("csv");
            if(!p.exists("csv-separator")) global_delim = ",";
        });
    parser
        .add_argument({ "--csv-separator" },
                      "Use the provided string instead of a ',' to separate values")
        .max_count(1)
        .action([](parser_t& p) { global_delim = p.get<std::string>("csv-separator"); });
    parser
        .add_argument({ "--force" },
                      "Force the generation of an configuration file even if it exists")
        .max_count(1)
        .action([](parser_t& p) { force_config = p.get<bool>("force"); });

    parser.end_group();

    parser.add_positional_argument("REGEX_FILTER").set_default(std::string{});

    auto err = parser.parse(argc, argv);

    if(parser.exists("help"))
    {
        parser.print_help();
        return EXIT_SUCCESS;
    }

    if(err)
    {
        std::cerr << err << std::endl;
        parser.print_help();
        return EXIT_FAILURE;
    }

#if ROCPROFSYS_USE_ROCM > 0
    if(gpu_count > 0)
    {
        size_t _num_metrics = 0;
        try
        {
            // call to rocm_events() will add choices to ROCPROFSYS_ROCM_EVENTS setting
            // so always perform this call even if list of HW counters is not requested
            _num_metrics = rocprofsys::rocm::rocm_events().size();
        } catch(std::runtime_error& _e)
        {
            verbprintf(0, "Retrieving the GPU HW counters failed: %s", _e.what());
        }
        verbprintf(1, "Found %i HIP devices and %zu GPU HW counters\n", gpu_count,
                   _num_metrics);
    }
    else
    {
        verbprintf(1, "No HIP devices found. GPU HW counters will not be available\n");
    }
#endif

    auto _parser_set_if_exists = [&parser](auto& _var, const std::string& _opt) {
        using Tp = decay_t<decltype(_var)>;
        if(parser.exists(_opt)) _var = parser.get<Tp>(_opt);
    };

    _parser_set_if_exists(options[FNAME], "filename");
    _parser_set_if_exists(options[DESC], "description");
    _parser_set_if_exists(options[VAL], "value");
    _parser_set_if_exists(options[CID], "string");
    _parser_set_if_exists(options[CATEGORY], "categories");
    _parser_set_if_exists(file, "output");
    _parser_set_if_exists(include_components, "components");
    _parser_set_if_exists(include_settings, "settings");
    _parser_set_if_exists(include_hw_counters, "hw-counters");

    if(parser.exists("generate-config"))
    {
        if(_config_file.empty())
            throw std::runtime_error("Error! No config output file specified!");
        if(_config_fmts.empty())
            throw std::runtime_error("Error! No config output formats specified!");
        try
        {
            generate_config(_config_file, _config_fmts, options);
        } catch(std::runtime_error& _e)
        {
            std::cerr << "[rocprof-sys-avail] " << _e.what() << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    if(parser.exists("markdown") && parser.exists("csv"))
    {
        std::cerr << "Error! both '--markdown' and '--csv' options cannot be specified\n";
        return EXIT_FAILURE;
    }

    if(parser.exists("list-categories") || parser.exists("list-keys"))
        return EXIT_SUCCESS;

    std::string _pos_regex{};
    if(parser.get_positional_count() > 0)
    {
        err = parser.get("REGEX_FILTER", _pos_regex);
        if(err)
        {
            std::cerr << err << std::endl;
            parser.print_help();
            return EXIT_FAILURE;
        }
    }

    if(!_pos_regex.empty())
    {
        regex_keys.emplace_back(_pos_regex);
        category_regex_keys.emplace_back(_pos_regex);
    }

    if(category_view.count("advanced") > 0 ||
       category_view.count("settings::advanced") > 0)
        print_advanced = true;

    if(category_view.empty()) category_view = _category_options;

    if(!include_components && !include_settings && !include_hw_counters)
        include_settings = true;

    if(markdown || include_hw_counters) padding = 6;

    std::ostream* os = nullptr;
    std::ofstream ofs;
    if(!file.empty())
    {
        ofs.open(file.c_str());
        if(ofs)
        {
            os = &ofs;
        }
        else
        {
            std::cerr << "Error opening output file: " << file << std::endl;
        }
    }

    // signal(SIGABRT, &dump_log_abort);
    // signal(SIGSEGV, &dump_log_abort);
    // signal(SIGQUIT, &dump_log_abort);

    if(!os) os = &std::cout;

    if(include_components)
    {
        write_component_info(*os, options, use_mark, fields);
    }
    dump_log();

    if(include_settings)
    {
        write_settings_info(
            *os, { options[VAL], options[LANG], options[DESC], options[CATEGORY] });
    }
    dump_log();

    if(include_hw_counters)
    {
        write_hw_counter_info(*os, { true, true, !force_brief && !available_only,
                                     !force_brief && !options[DESC], options[DESC] });
    }
    dump_log();

    const_cast<std::shared_ptr<tim::settings>&>(tim::settings::shared_instance()).reset();

    return 0;
}

//======================================================================================//
//
//                                  COMPONENT INFO
//
//======================================================================================//

template <size_t N>
void
write_component_info(std::ostream& os, const array_t<bool, N>& options,
                     const array_t<bool, N>& _mark, const array_t<string_t, N>& fields)
{
    static_assert(N >= num_component_options,
                  "Error! Too few component options + fields");

    std::vector<info_type> _info = get_component_info<TIMEMORY_COMPONENTS_END>();

    if(available_only)
        _info.erase(std::remove_if(_info.begin(), _info.end(),
                                   [](const auto& itr) { return !itr.is_available(); }),
                    _info.end());

    _info.erase(std::remove_if(_info.begin(), _info.end(),
                               [](const auto& itr) {
                                   // NOLINTNEXTLINE
                                   for(const auto& nitr :
                                       { "cuda", "cupti", "nvtx", "roofline", "_bundle",
                                         "data_integer", "data_unsigned", "data_floating",
                                         "printer" })
                                   {
                                       if(itr.name().find(nitr) != std::string::npos)
                                           return true;
                                   }
                                   auto _categories = tim::delimit(
                                       itr.categories(), ", ", [](const string_t& _v) {
                                           return "component::" + _v;
                                       });
                                   for(const auto& citr : _categories)
                                       if(category_view.count(citr) > 0) return false;
                                   return true;
                               }),
                _info.end());

    using width_type = std::vector<int64_t>;
    using width_bool = std::array<bool, N + 2>;

    auto       _available_column = !force_brief && !available_only;
    width_type _widths           = width_type{ 30, 12, 20, 20, 20, 40, 20, 40, 10 };
    width_bool _wusing           = width_bool{ true, _available_column };
    int64_t    pad               = padding;
    for(size_t i = 0; i < options.size(); ++i)
        _wusing[i + 2] = options[i];

    {
        constexpr size_t idx = 0;
        stringstream_t   ss;
        write_entry(ss, "COMPONENT", _widths.at(0), false, true);
        _widths.at(idx) = std::max<int64_t>(ss.str().length() + pad, _widths.at(idx));
    }

    {
        constexpr size_t idx = 1;
        stringstream_t   ss;
        write_entry(ss, "AVAILABLE", _widths.at(1), true, false);
        _widths.at(idx) = std::max<int64_t>(ss.str().length() + pad, _widths.at(idx));
    }

    for(size_t i = 0; i < fields.size(); ++i)
    {
        constexpr size_t idx = 2;
        stringstream_t   ss;
        if(!options[i]) continue;
        write_entry(ss, fields[i], _widths.at(i + 2), true, _mark.at(idx));
        _widths.at(idx + i) =
            std::max<int64_t>(ss.str().length() + pad, _widths.at(idx + i));
    }

    if(alphabetical)
    {
        std::sort(_info.begin(), _info.end(), [](const auto& lhs, const auto& rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });
    }

    // compute the widths
    for(const auto& itr : _info)
    {
        {
            int               _selected = 0;
            std::stringstream ss;
            _selected += (is_selected(std::get<0>(itr))) ? 1 : 0;
            write_entry(ss, std::get<0>(itr), _widths.at(0), false, true);
            if(_available_column)
            {
                std::stringstream _avss{};
                _avss << std::boolalpha << std::get<1>(itr);
                _selected += (is_selected(_avss.str())) ? 1 : 0;
            }
            write_entry(ss, std::get<1>(itr), _widths.at(1), true, false);
            for(size_t i = 0; i < std::get<2>(itr).size(); ++i)
            {
                if(!options[i]) continue;
                bool center = (i > 0) ? false : true;
                _selected += (is_selected(std::get<2>(itr).at(i))) ? 1 : 0;
                write_entry(ss, std::get<2>(itr).at(i), _widths.at(i + 2), center,
                            _mark.at(i));
            }

            if(!category_regex_keys.empty())
                _selected +=
                    (is_category_selected(std::get<2>(itr).at(CATEGORY))) ? 1 : 0;

            if(_selected == 0) continue;
        }

        {
            constexpr size_t idx = 0;
            stringstream_t   ss;
            write_entry(ss, std::get<idx>(itr), 0, true, true);
            _widths.at(idx) = std::max<int64_t>(ss.str().length() + pad, _widths.at(idx));
        }

        {
            constexpr size_t idx = 1;
            stringstream_t   ss;
            write_entry(ss, std::get<idx>(itr), 0, true, false);
            _widths.at(idx) = std::max<int64_t>(ss.str().length() + pad, _widths.at(idx));
        }

        constexpr size_t idx = 2;
        for(size_t i = 0; i < std::get<2>(itr).size(); ++i)
        {
            stringstream_t ss;
            write_entry(ss, std::get<idx>(itr)[i], 0, true, _mark.at(idx));
            _widths.at(idx + i) =
                std::max<int64_t>(ss.str().length() + pad, _widths.at(idx + i));
        }
    }

    dump_log();

    _widths = compute_max_columns(_widths, _wusing);

    if(!markdown && !csv) os << banner(_widths, _wusing, '-');

    if(!csv) os << global_delim;
    write_entry(os, "COMPONENT", _widths.at(0), true, false);
    if(_available_column) write_entry(os, "AVAILABLE", _widths.at(1), true, false);
    for(size_t i = 0; i < fields.size(); ++i)
    {
        if(!options[i]) continue;
        write_entry(os, fields[i], _widths.at(i + 2), true, false);
    }

    os << "\n" << banner(_widths, _wusing, '-');

    for(const auto& itr : _info)
    {
        int               _selected = 0;
        std::stringstream ss;
        _selected += (is_selected(std::get<0>(itr))) ? 1 : 0;
        write_entry(ss, std::get<0>(itr), _widths.at(0), false, true);
        if(_available_column)
        {
            std::stringstream _avss{};
            _avss << std::boolalpha << std::get<1>(itr);
            _selected += (is_selected(_avss.str())) ? 1 : 0;
            write_entry(ss, std::get<1>(itr), _widths.at(1), true, false);
        }
        for(size_t i = 0; i < std::get<2>(itr).size(); ++i)
        {
            if(!options[i]) continue;
            bool center = (i > 0) ? false : true;
            _selected += (is_selected(std::get<2>(itr).at(i))) ? 1 : 0;
            if(fields.at(i) == "DESCRIPTION")
                write_wrap_entry(ss, std::get<2>(itr).at(i), _widths.at(i + 2), center,
                                 _mark.at(i), i + 2, _widths, _wusing);
            else
                write_entry(ss, std::get<2>(itr).at(i), _widths.at(i + 2), center,
                            _mark.at(i));
        }

        if(!category_regex_keys.empty())
            _selected += (is_category_selected(std::get<2>(itr).at(CATEGORY))) ? 1 : 0;

        if(_selected > 0)
        {
            os << global_delim;
            os << hl_selected(ss.str());
            os << "\n";
        }
    }

    dump_log();

    if(!markdown) os << banner(_widths, _wusing, '-');
}

//======================================================================================//
//
//                                      SETTINGS
//
//======================================================================================//

template <size_t N>
void
write_settings_info(std::ostream& os, const array_t<bool, N>& opts,
                    const array_t<bool, N>&, const array_t<string_t, N>&)
{
    static_assert(N >= num_settings_options, "Error! Too few settings options + fields");

    static constexpr size_t size = 8;
    using archive_type           = cereal::SettingsTextArchive;
    using array_type             = typename archive_type::array_type;
    using width_type             = array_t<int64_t, size>;
    using width_bool             = array_t<bool, size>;

    width_type _widths = { 0, 0, 0, 0, 0, 0, 0, 0 };
    width_bool _wusing = {
        true, !force_brief, opts[0], opts[1], opts[1], opts[1], opts[2], opts[3],
    };
    width_bool _mark = { false, false, false, true, true, true, false, false };

    array_type _setting_output;
    auto       _settings = tim::settings::shared_instance();

    cereal::SettingsTextArchive settings_archive{ _setting_output, settings_exclude };
    settings::serialize_settings(settings_archive);

    if(expand_keys)
    {
        for(auto& itr : _setting_output)
        {
            itr["value"] = tim::settings::format(itr["value"], _settings->get_tag());
        }
    }
    _setting_output.erase(
        std::remove_if(_setting_output.begin(), _setting_output.end(),
                       [](const auto& itr) { return itr.find("environ") == itr.end(); }),
        _setting_output.end());

    // patch up the categories
    auto _not_in_category_view = str_set_t{};
    for(auto& itr : _setting_output)
    {
        auto _name = itr.find("environ")->second;
        auto sitr  = _settings->find(_name);
        if(sitr != _settings->end())
        {
            str_set_t _categories{};
            for(const auto& citr : sitr->second->get_categories())
                _categories.emplace(TIMEMORY_JOIN("::", "settings", citr));
            bool _found = false;
            for(const auto& citr : _categories)
            {
                if(category_view.count(citr) > 0) _found = true;
            }
            if(!print_advanced && _categories.count("settings::advanced") > 0)
            {
                if(!sitr->second->get_config_updated() &&
                   !sitr->second->get_environ_updated())
                    _not_in_category_view.emplace(_name);
            }
            if(!_found)
            {
                _not_in_category_view.emplace(_name);
                continue;
            }
            std::stringstream _ss{};
            for(const auto& citr : sitr->second->get_categories())
                _ss << ", " << citr;
            if(!_ss.str().empty())
            {
                itr["categories"] = _ss.str().substr(2);
            }
        }
    }

    // erase excluded settings and erase settings not in category view
    _setting_output.erase(
        std::remove_if(_setting_output.begin(), _setting_output.end(),
                       [&_not_in_category_view](const auto& itr) {
                           return (exclude_setting(itr.find("environ")->second) ||
                                   _not_in_category_view.count(
                                       itr.find("environ")->second) > 0);
                       }),
        _setting_output.end());

    _setting_output.erase(std::remove_if(_setting_output.begin(), _setting_output.end(),
                                         [](const auto& itr) {
                                             return !is_category_selected(
                                                 itr.find("categories")->second);
                                         }),
                          _setting_output.end());

    if(available_only)
    {
        _setting_output.erase(
            std::remove_if(_setting_output.begin(), _setting_output.end(),
                           [&_settings](const auto& itr) {
                               auto iitr = _settings->find(itr.at("environ"));
                               if(iitr != _settings->end())
                                   return (iitr->second->get_enabled() == false);
                               return true;
                           }),
            _setting_output.end());
    }

    if(alphabetical)
    {
        std::sort(_setting_output.begin(), _setting_output.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return (lhs.find("environ")->second < rhs.find("environ")->second);
                  });
    }

    array_t<string_t, size> _labels = {
        "ENVIRONMENT VARIABLE", "VALUE",           "DATA TYPE",   "C++ STATIC ACCESSOR",
        "C++ MEMBER ACCESSOR",  "Python ACCESSOR", "DESCRIPTION", "CATEGORIES",
    };
    array_t<string_t, size> _keys   = { "environ",         "value",
                                        "data_type",       "static_accessor",
                                        "member_accessor", "python_accessor",
                                        "description",     "categories" };
    array_t<bool, size>     _center = {
        false, true, true, false, false, false, false, false
    };

    for(size_t i = 0; i < _widths.size(); ++i)
    {
        if(_wusing.at(i))
            _widths.at(i) =
                std::max<uint64_t>(_widths.at(i), _labels.at(i).size() + padding);
        else
            _widths.at(i) = 0;
    }

    std::vector<array_t<string_t, size>> _results{};
    for(const auto& itr : _setting_output)
    {
        array_t<string_t, size> _tmp{};
        for(size_t j = 0; j < _keys.size(); ++j)
        {
            auto eitr = itr.find(_keys.at(j));
            if(eitr != itr.end()) _tmp.at(j) = eitr->second;
        }
        if(!_tmp.at(0).empty()) _results.push_back(_tmp);
    }

    for(const auto& itr : _results)
    {
        // save the widths in case this gets filtered
        auto              _last_widths = _widths;
        std::stringstream ss{};
        int               _selected = 0;
        for(size_t i = 0; i < itr.size(); ++i)
        {
            if(!_wusing.at(i)) continue;
            _widths.at(i) =
                std::max<uint64_t>(_widths.at(i), itr.at(i).length() + padding);
            _selected += (is_selected(itr.at(i))) ? 1 : 0;
            write_entry(ss, itr.at(i), _widths.at(i), _center.at(i), _mark.at(i));
        }

        if(_selected == 0)
        {
            _widths = _last_widths;
            continue;
        }
    }

    dump_log();

    _widths = compute_max_columns(_widths, _wusing);

    if(!markdown) os << banner(_widths, _wusing, '-');

    if(!csv) os << global_delim;
    for(size_t i = 0; i < _labels.size(); ++i)
    {
        if(!_wusing.at(i)) continue;
        write_entry(os, _labels.at(i), _widths.at(i), true, false);
    }
    os << "\n" << banner(_widths, _wusing, '-');

    for(const auto& itr : _results)
    {
        std::stringstream ss{};
        int               _selected = 0;
        for(size_t i = 0; i < itr.size(); ++i)
        {
            if(!_wusing.at(i)) continue;
            _selected += (is_selected(itr.at(i))) ? 1 : 0;
            if(_labels.at(i) == "DESCRIPTION")
                write_wrap_entry(ss, itr.at(i), _widths.at(i), _center.at(i), _mark.at(i),
                                 i, _widths, _wusing);
            else
                write_entry(ss, itr.at(i), _widths.at(i), _center.at(i), _mark.at(i));
        }

        if(_selected > 0)
        {
            if(!csv) os << global_delim;
            os << hl_selected(ss.str());
            os << "\n";
        }
    }

    dump_log();

    if(!markdown) os << banner(_widths, _wusing, '-');
}

//======================================================================================//
//
//                                  HARDWARE COUNTERS
//
//======================================================================================//

template <size_t N>
void
write_hw_counter_info(std::ostream& os, const array_t<bool, N>& options,
                      const array_t<bool, N>&, const array_t<string_t, N>&)
{
    static_assert(N >= num_hw_counter_options,
                  "Error! Too few hw counter options + fields");

    using width_type       = array_t<int64_t, N>;
    using width_bool       = array_t<bool, N>;
    using hwcounter_info_t = std::vector<tim::hardware_counters::info>;

    auto _papi_events = tim::papi::available_events_info();
    auto _rocm_events =
        (gpu_count > 0) ? rocprofsys::rocm::rocm_events() : hwcounter_info_t{};

    if(alphabetical)
    {
        auto _sorter = [](const auto& lhs, const auto& rhs) {
            return (lhs.symbol() < rhs.symbol());
        };
        std::sort(_papi_events.begin(), _papi_events.end(), _sorter);
        std::sort(_rocm_events.begin(), _rocm_events.end(), _sorter);
    }

    auto _process_counters = [](auto& _events_v, int32_t _offset_v) {
        for(auto& iitr : _events_v)
            iitr.offset() += _offset_v;
        return static_cast<int32_t>(_events_v.size());
    };

    int32_t _offset = 0;
    _offset += _process_counters(_papi_events, _offset);
    _offset += _process_counters(_rocm_events, _offset);

    auto fields =
        std::vector<std::pair<std::string, hwcounter_info_t>>{ { "CPU", _papi_events },
                                                               { "GPU", _rocm_events } };
    array_t<string_t, N> _labels = { "HARDWARE COUNTER", "DEVICE", "AVAILABLE", "SUMMARY",
                                     "DESCRIPTION" };
    array_t<bool, N>     _center = { false, true, true, false, false };

    auto _valid_symbols = std::set<std::string>{};
    for(auto& fitr : fields)
    {
        if(!category_view.empty() && category_view.count(fitr.first) == 0 &&
           category_view.count(std::string{ "hw_counters::" } + fitr.first) == 0)
            fitr.second.clear();

        if(!is_category_selected(fitr.first) &&
           !is_category_selected(std::string{ "hw_counters::" } + fitr.first))
            fitr.second.clear();

        for(const auto& itr : fitr.second)
        {
            if(available_only && !itr.available()) continue;
            std::stringstream ss;
            int               _selected = 0;
            if(options[0])
            {
                _selected += (is_selected(itr.symbol())) ? 1 : 0;
            }

            if(options[2])
            {
                std::stringstream _avss{};
                _avss << std::boolalpha << itr.available();
                _selected += (is_selected(_avss.str())) ? 1 : 0;
            }

            for(size_t i = 3; i < N; ++i)
            {
                if(options[i])
                {
                    _selected += (is_selected(itr.short_description()) ||
                                  is_selected(itr.long_description()))
                                     ? 1
                                     : 0;
                }
            }

            if(_selected > 0) _valid_symbols.emplace(itr.symbol());
        }
    }

    for(auto& fitr : fields)
    {
        fitr.second.erase(std::remove_if(fitr.second.begin(), fitr.second.end(),
                                         [&_valid_symbols](const auto& itr) {
                                             return (_valid_symbols.count(itr.symbol()) ==
                                                     0);
                                         }),
                          fitr.second.end());
    }

    width_type _widths;
    width_bool _wusing;
    width_bool _mark = { true, false, false, false, false };
    _widths.fill(0);
    _wusing.fill(false);
    for(size_t i = 0; i < _widths.size(); ++i)
    {
        // don't account for AVAILABLE or "DEVICE"
        if(i != 1 && i != 2) _widths.at(i) = _labels.at(i).length() + padding;
        _wusing.at(i) = options[i];
    }

    for(const auto& fitr : fields)
    {
        for(const auto& itr : fitr.second)
        {
            width_type _w = { { (int64_t) itr.symbol().length(), (int64_t) 4, (int64_t) 6,
                                (int64_t) itr.short_description().length(),
                                (int64_t) itr.long_description().length() } };
            for(auto& witr : _w)
                witr += padding;

            for(size_t i = 0; i < N; ++i)
            {
                if(_wusing.at(i))
                    _widths.at(i) = std::max<uint64_t>(_widths.at(i), _w.at(i));
            }
        }
    }

    _widths = compute_max_columns(_widths, _wusing);

    if(!markdown) os << banner(_widths, _wusing, '-');
    if(!csv) os << global_delim;

    for(size_t i = 0; i < _labels.size(); ++i)
    {
        if(options[i]) write_entry(os, _labels.at(i), _widths.at(i), true, false);
    }
    os << "\n" << banner(_widths, _wusing, '-');

    for(const auto& fitr : fields)
    {
        // if has a label and is empty, continue
        if(!fitr.first.empty() && fitr.second.empty()) continue;

        for(const auto& itr : fitr.second)
        {
            std::stringstream ss;
            if(options[0])
            {
                write_entry(ss, itr.symbol(), _widths.at(0), _center.at(0), _mark.at(0));
            }

            write_entry(ss, fitr.first, _widths.at(1), _center.at(1), _mark.at(1));

            if(options[2])
            {
                std::stringstream _avss{};
                _avss << std::boolalpha << itr.available();
                write_entry(ss, itr.available(), _widths.at(2), _center.at(2),
                            _mark.at(1));
            }

            array_t<string_t, N> _e = { { "", "", "", itr.short_description(),
                                          itr.long_description() } };
            for(size_t i = 3; i < N; ++i)
            {
                if(options[i])
                {
                    write_wrap_entry(ss, _e.at(i), _widths.at(i), _center.at(i),
                                     _mark.at(i), i, _widths, _wusing);
                }
            }

            if(!csv) os << global_delim;
            os << hl_selected(ss.str());
            os << "\n";
        }
    }

    dump_log();

    if(!markdown) os << banner(_widths, _wusing, '-');
}

//======================================================================================//
//
//                                  ANONYMOUS FUNCTIONS
//
//======================================================================================//

namespace
{
template <typename IntArrayT, typename BoolArrayT>
IntArrayT
compute_max_columns(IntArrayT _widths, BoolArrayT _using)
{
    using value_type = typename IntArrayT::value_type;

    if(num_cols == 0) return _widths;

    auto _get_sum = [&]() {
        value_type _sumv = 0;
        for(size_t i = 0; i < _widths.size(); ++i)
            if(_using.at(i)) _sumv += _widths.at(i);
        return _sumv;
    };
    auto _get_max = [&]() {
        auto       _midx = _widths.size();
        value_type _maxv = 0;
        for(size_t i = 0; i < _widths.size(); ++i)
        {
            if(_using.at(i) && _widths.at(i) > _maxv)
            {
                _midx = i;
                _maxv = _widths.at(i);
            }
        }

        if(_maxv <= min_width)
        {
            _midx = _widths.size();
            _maxv = min_width;
        }
        return std::make_pair(_midx, _maxv);
    };
    auto _decrement_max = [&]() {
        auto _midx = _get_max().first;
        if(_midx < _widths.size()) _widths.at(_midx) -= 1;
    };

    int32_t _max_width = num_cols;
    size_t  _n         = 0;
    size_t  _nmax      = std::numeric_limits<uint16_t>::max();
    while(_n++ < _nmax)
    {
        if(debug_msg)
        {
            std::stringstream _msg;
            for(size_t i = 0; i < _widths.size(); ++i)
                _msg << ", " << ((_using.at(i)) ? _widths.at(i) : 0);
            std::cerr << "[ temp]> sum_width = " << _get_sum()
                      << ", max_width = " << _max_width
                      << ", widths = " << _msg.str().substr(2) << std::endl;
        }

        if(_get_max().first == _widths.size() || _get_sum() <= _max_width) break;
        _decrement_max();
    }

    int32_t _maxw = _get_max().second;
    if(max_width == 0 || _maxw < max_width) max_width = _maxw;

    if(debug_msg)
    {
        std::stringstream _msg;
        for(size_t i = 0; i < _widths.size(); ++i)
            _msg << ", " << ((_using.at(i)) ? _widths.at(i) : 0);
        std::cerr << "[final]> sum_width = " << _get_sum()
                  << ", max_width = " << _max_width
                  << ", widths = " << _msg.str().substr(2)
                  << ", column max width = " << max_width << std::endl;
    }

    return _widths;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
write_entry(std::ostream& os, const Tp& _entry, int64_t _w, bool center, bool mark)
{
    if(max_width > 0 && _w > max_width) _w = max_width;

    stringstream_t ssentry;
    stringstream_t ss;
    if(csv)
        ssentry << std::boolalpha << _entry;
    else
        ssentry << ' ' << std::boolalpha << ((mark && markdown) ? "`" : "") << _entry;
    auto _sentry = remove(ssentry.str(), { "tim::", "component::" });

    auto _decr = (mark && markdown) ? 6 : 5;
    if(!csv && _w > 0 && _sentry.length() > static_cast<size_t>(_w - 2))
        _sentry = _sentry.substr(0, _w - _decr) + "...";

    if(mark && markdown)
    {
        _sentry += std::string{ "`" };
    }

    if(center && !csv)
    {
        size_t _n = 0;
        while(_sentry.length() + 2 < static_cast<size_t>(_w))
        {
            if(_n++ % 2 == 0)
            {
                _sentry += std::string{ " " };
            }
            else
            {
                _sentry.insert(0, " ");
            }
        }
        if(_w > 0 && _sentry.length() > static_cast<size_t>(_w - 1))
            _sentry = _sentry.substr(_w - 1);
        ss << std::left << std::setw(_w - 1) << _sentry << global_delim;
    }
    else
    {
        if(csv)
        {
            if(_sentry.find(global_delim) == std::string::npos)
                ss << _sentry << global_delim;
            else
            {
                if(_sentry.find('"') != std::string::npos)
                    ss << "'" << _sentry << "'" << global_delim;
                else
                    ss << "\"" << _sentry << "\"" << global_delim;
            }
        }
        else
            ss << std::left << std::setw(_w - 1) << _sentry << global_delim;
    }
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename IntArrayT, size_t N>
void
write_wrap_entry(std::ostream& os, const Tp& _entry, int64_t _w, bool center, bool mark,
                 size_t _idx, IntArrayT _breaks, std::array<bool, N> _use)
{
    if(csv)
    {
        write_entry(os, _entry, _w, center, mark);
        return;
    }

    auto _orig_w = _w;
    if(max_width > 0 && _w > max_width) _w = max_width;

    auto           _remainder = std::string{};
    stringstream_t ssentry;
    stringstream_t ss;
    ssentry << ' ' << std::boolalpha << ((mark && markdown) ? "`" : "") << _entry;
    auto _sentry = remove(ssentry.str(), { "tim::", "component::" });

    if(_w > 0 && _sentry.length() > static_cast<size_t>(_w - 2))
    {
        auto _decr   = (mark && markdown) ? 4 : 3;
        auto _lspace = _sentry.substr(0, _w - _decr).find_last_of(" \t");
        if(_lspace == std::string::npos || _lspace < static_cast<uint64_t>(_w / 2))
            _lspace = _w - _decr;
        _remainder = std::string{ " " } + _sentry.substr(_lspace);
        _sentry    = _sentry.substr(0, _lspace);
    }

    if(mark && markdown)
    {
        _sentry += std::string{ "`" };
    }

    if(center)
    {
        size_t _n = 0;
        while(_sentry.length() + 2 < static_cast<size_t>(_w))
        {
            if(_n++ % 2 == 0)
            {
                _sentry += std::string{ " " };
            }
            else
            {
                _sentry.insert(0, " ");
            }
        }
        if(_w > 0 && _sentry.length() > static_cast<size_t>(_w - 1))
            _sentry = _sentry.substr(_w - 1);
        ss << std::left << std::setw(_w - 1) << _sentry << global_delim;
    }
    else
    {
        ss << std::left << std::setw(_w - 1) << _sentry << global_delim;
    }

    if(!_remainder.empty())
    {
        ss << wrap(_idx, _breaks, _use);
        write_wrap_entry(ss, _remainder, _orig_w, center, mark, _idx, _breaks, _use);
    }

    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename IntArrayT, size_t N>
string_t
banner(IntArrayT _breaks, std::array<bool, N> _use, char filler, char delim)
{
    if(csv) return string_t{};

    if(debug_msg)
    {
        std::cerr << "[before]> Breaks: ";
        for(const auto& itr : _breaks)
            std::cerr << itr << " ";
        std::cerr << std::endl;
    }

    _breaks = compute_max_columns(_breaks, _use);

    if(debug_msg)
    {
        std::cerr << "[after]>  Breaks: ";
        for(const auto& itr : _breaks)
            std::cerr << itr << " ";
        std::cerr << std::endl;
    }

    for(auto& itr : _breaks)
    {
        if(max_width > 0 && itr > max_width) itr = max_width;
    }

    stringstream_t ss;
    ss.fill(filler);
    int64_t _remain = 0;
    for(size_t i = 0; i < _breaks.size(); ++i)
    {
        if(_use.at(i)) _remain += _breaks.at(i);
    }
    auto _total = _remain;
    ss << delim;
    for(size_t i = 0; i < _breaks.size(); ++i)
    {
        if(!_use.at(i)) continue;
        ss << std::setw(_breaks.at(i) - 1) << "" << delim;
        _remain -= _breaks.at(i);
    }
    ss << "\n";
    if(_remain != 0)
    {
        printf("[banner]> non-zero remainder: %i with total: %i\n", (int) _remain,
               (int) _total);
    }
    return ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename IntArrayT, size_t N>
string_t
wrap(size_t idx, IntArrayT _breaks, std::array<bool, N> _use, char filler, char delim)
{
    if(csv) return string_t{};

    _breaks = compute_max_columns(_breaks, _use);

    for(auto& itr : _breaks)
    {
        if(max_width > 0 && itr > max_width) itr = max_width;
    }

    stringstream_t ss;
    ss.fill(filler);
    int64_t _remain = 0;
    for(size_t i = 0; i < _breaks.size(); ++i)
    {
        if(_use.at(i)) _remain += _breaks.at(i);
    }

    for(size_t i = 1; i < _breaks.size(); ++i)
    {
        auto j = i + idx;
        auto k = j % _breaks.size();
        if(k == 0) ss << "\n" << delim;
        if(!_use.at(k)) continue;
        ss << std::setw(_breaks.at(k) - 1) << "" << delim;
        _remain -= _breaks.at(k);
    }

    return ss.str();
}
}  // namespace

//--------------------------------------------------------------------------------------//
