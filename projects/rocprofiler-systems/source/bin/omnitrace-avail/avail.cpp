//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#include "avail.hpp"
#include "library/api.hpp"
#include "library/components/backtrace.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/components/mpi_gotcha.hpp"
#include "library/components/omnitrace.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/components/roctracer.hpp"
#include "library/config.hpp"

#include <timemory/components.hpp>
#include <timemory/components/definition.hpp>
#include <timemory/components/placeholder.hpp>
#include <timemory/components/properties.hpp>
#include <timemory/components/skeletons.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/timemory.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/types.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <set>
#include <sstream>
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

template <typename Tp, size_t N>
using array_t        = std::array<Tp, N>;
using string_t       = std::string;
using stringstream_t = std::stringstream;
using str_vec_t      = std::vector<string_t>;
using str_set_t      = std::set<string_t>;
using info_type_base = std::tuple<string_t, bool, str_vec_t>;
using parser_t       = tim::argparse::argument_parser;

struct info_type : info_type_base
{
    TIMEMORY_DEFAULT_OBJECT(info_type)

    template <typename... Args>
    info_type(Args&&... _args)
    : info_type_base{ std::forward<Args>(_args)... }
    {}

    const auto& name() const { return std::get<0>(*this); }
    auto        is_available() const { return std::get<1>(*this); }
    const auto& info() const { return std::get<2>(*this); }
    const auto& data_type() const { return info().at(0); }
    const auto& enum_type() const { return info().at(1); }
    const auto& id_type() const { return info().at(2); }
    const auto& id_strings() const { return info().at(3); }
    const auto& label() const { return info().at(4); }
    const auto& description() const { return info().at(5); }
    const auto& categories() const { return info().at(6); }

    bool valid() const { return !name().empty() && info().size() >= 6; }

    bool operator<(const info_type& rhs) const { return name() < rhs.name(); }
    bool operator!=(const info_type& rhs) const { return !(*this == rhs); }
    bool operator==(const info_type& rhs) const
    {
        if(info().size() != rhs.info().size()) return false;
        for(size_t i = 0; i < info().size(); ++i)
        {
            if(info().at(i) != rhs.info().at(i)) return false;
        }
        return name() == rhs.name() && is_available() == rhs.is_available();
    }
};

//--------------------------------------------------------------------------------------//

enum
{
    VAL      = 0,
    ENUM     = 1,
    LANG     = 2,
    CID      = 3,
    FNAME    = 4,
    DESC     = 5,
    CATEGORY = 6,
    TOTAL    = 7
};

//--------------------------------------------------------------------------------------//

namespace
{
char              global_delim           = '|';
bool              markdown               = false;
bool              alphabetical           = false;
bool              available_only         = false;
bool              all_info               = false;
bool              force_brief            = false;
bool              debug_msg              = false;
bool              case_insensitive       = false;
bool              regex_hl               = false;
int32_t           max_width              = 0;
int32_t           num_cols               = 0;
int32_t           min_width              = 40;
int32_t           padding                = 4;
str_vec_t         regex_keys             = {};
str_vec_t         category_regex_keys    = {};
str_set_t         category_view          = {};
constexpr size_t  num_component_options  = 7;
constexpr size_t  num_settings_options   = 4;
constexpr size_t  num_hw_counter_options = 4;
std::stringstream lerr{};

// explicit setting names to exclude
std::set<std::string> settings_exclude = {
    "OMNITRACE_ENVIRONMENT", "OMNITRACE_COMMAND_LINE", "cereal_class_version", "settings",
#if !defined(TIMEMORY_USE_CRAYPAT)
    "OMNITRACE_CRAYPAT"
#endif
};

// exclude some timemory settings which are not relevant to omnitrace
//  exact matches, e.g. OMNITRACE_BANNER
std::string settings_rexclude_exact =
    "^OMNITRACE_(BANNER|DESTRUCTOR_REPORT|COMPONENTS|(GLOBAL|MPIP|NCCLP|OMPT|"
    "PROFILER|TRACE|KOKKOS)_COMPONENTS|PYTHON_EXE|PAPI_ATTACH|PLOT_OUTPUT|SEPARATOR_"
    "FREQ|"
    "STACK_CLEARING|TARGET_PID|THROTTLE_(COUNT|VALUE)|(AUTO|FLAMEGRAPH)_OUTPUT|"
    "(ENABLE|DISABLE)_ALL_SIGNALS|ALLOW_SIGNAL_HANDLER|CTEST_NOTES|INSTRUCTION_"
    "ROOFLINE)$";

//  leading matches, e.g. OMNITRACE_MPI_[A-Z_]+
std::string settings_rexclude_begin =
    "^OMNITRACE_(ERT|DART|MPI|UPCXX|ROOFLINE|CUDA|NVTX|CUPTI)_[A-Z_]+$";

bool
exclude_setting(const std::string&);
}  // namespace

//--------------------------------------------------------------------------------------//

void
dump_log()
{
    if(debug_msg)
    {
        std::cerr << lerr.str() << std::flush;
        lerr = std::stringstream{};
    }
}

template <typename IntArrayT, typename BoolArrayT>
static IntArrayT
compute_max_columns(IntArrayT _widths, BoolArrayT _using);

string_t
remove(string_t inp, const std::set<string_t>& entries);

template <typename Tp>
void
write_entry(std::ostream& os, const Tp& _entry, int64_t _w, bool center, bool mark);

template <typename IntArrayT, size_t N>
string_t
banner(IntArrayT _breaks, std::array<bool, N> _use, char filler = '-', char delim = '|');

bool
is_selected(const std::string& line);

bool
is_category_selected(const std::string& _line);

std::string
hl_selected(const std::string& line);

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

template <typename Type = void>
struct get_availability;

template <typename Type = void>
struct component_categories;

void
process_categories(parser_t&, const str_set_t&);

//--------------------------------------------------------------------------------------//

template <typename Type>
struct get_availability
{
    using this_type  = get_availability<Type>;
    using metadata_t = component::metadata<Type>;
    using property_t = component::properties<Type>;

    static info_type get_info();
    auto             operator()() const { return get_info(); }
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
struct get_availability<type_list<Types...>>
{
    using data_type = std::vector<info_type>;

    static data_type get_info(data_type& _v)
    {
        TIMEMORY_FOLD_EXPRESSION(_v.emplace_back(get_availability<Types>::get_info()));
        return _v;
    }

    static data_type get_info()
    {
        data_type _v{};
        return get_info(_v);
    }

    template <typename... Args>
    decltype(auto) operator()(Args&&... _args)
    {
        return get_info(std::forward<Args>(_args)...);
    }
};

//--------------------------------------------------------------------------------------//

template <>
struct get_availability<void>
{
    template <typename... Tp, typename... Args>
    decltype(auto) operator()(tim::type_list<Tp...>, Args&&... _args) const
    {
        return get_availability<tim::type_list<Tp...>>{}(std::forward<Args>(_args)...);
    }

    template <typename Tp, typename... Args>
    decltype(auto) operator()(Args&&... _args) const
    {
        return get_availability<tim::type_list<Tp>>{}(std::forward<Args>(_args)...);
    }
};

//--------------------------------------------------------------------------------------//

template <typename Type>
struct component_categories
{
    template <typename... Tp>
    void operator()(std::set<std::string>& _v, type_list<Tp...>) const
    {
        //
        auto _cleanup = [](std::string _type, const std::string& _pattern) {
            auto _pos = std::string::npos;
            while((_pos = _type.find(_pattern)) != std::string::npos)
                _type.erase(_pos, _pattern.length());
            return _type;
        };
        (void) _cleanup;  // unused but set if sizeof...(Tp) == 0

        TIMEMORY_FOLD_EXPRESSION(_v.emplace(
            TIMEMORY_JOIN("::", "component", _cleanup(demangle<Tp>(), "tim::"))));
    }

    void operator()(std::set<std::string>& _v) const
    {
        if constexpr(!concepts::is_placeholder<Type>::value)
            (*this)(_v, trait::component_apis_t<Type>{});
    }
};

template <>
struct component_categories<void>
{
    template <size_t... Idx>
    void operator()(std::set<std::string>& _v, std::index_sequence<Idx...>) const
    {
        TIMEMORY_FOLD_EXPRESSION(
            component_categories<component::enumerator_t<Idx>>{}(_v));
    }

    void operator()(std::set<std::string>& _v) const
    {
        (*this)(_v, std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
    }

    auto operator()() const
    {
        std::set<std::string> _categories{};
        (*this)(_categories);
        return _categories;
    }
};

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    omnitrace_init_library();

    std::set<std::string> _category_options = component_categories{}();
    {
        auto _settings = tim::settings::shared_instance();
        for(const auto& itr : *_settings)
        {
            if(exclude_setting(itr.second->get_env_name())) continue;
            for(const auto& eitr : itr.second->get_categories())
            {
                if(eitr == "native")
                    _category_options.emplace("settings::timemory");
                else
                    _category_options.emplace(TIMEMORY_JOIN("::", "settings", eitr));
            }
        }
    }

    array_t<bool, TOTAL> options    = { false, false, false, false, false, false, false };
    array_t<string_t, TOTAL> fields = {};
    array_t<bool, TOTAL>     use_mark = {};

    std::string cols_via{};
    std::tie(num_cols, cols_via) = tim::utility::console::get_columns();
    std::string col_msg =
        "(default: " + std::to_string(num_cols) + " [via " + cols_via + "])";

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

    parser_t parser("omnitrace-avail");

    parser.enable_help();
    parser.set_help_width(40);
    parser.add_argument({ "--debug" }, "Enable debug messages")
        .max_count(1)
        .action([](parser_t& p) { debug_msg = p.get<bool>("debug"); });
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

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[CATEGORIES]" }, "");
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

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[VIEW OPTIONS]" }, "");
    parser.add_argument({ "-A", "--available" }, "Only display available components")
        .max_count(1)
        .action([](parser_t& p) { available_only = p.get<bool>("available"); });
    parser
        .add_argument({ "-r", "--filter" },
                      "Filter the output according to provided regex (egrep + "
                      "case-sensitive) [e.g. -r \"true\"]")
        .min_count(1)
        .dtype("list of strings")
        .action([](parser_t& p) { regex_keys = p.get<str_vec_t>("filter"); });
    parser
        .add_argument({ "-R", "--category-filter" },
                      "Filter the output according to provided regex w.r.t. the "
                      "categories (egrep + "
                      "case-sensitive) [e.g. -r \"true\"]")
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
    parser
        .add_argument({ "--list-categories" },
                      "List the available categories for --categories option")
        .count(0)
        .action([_category_options](parser_t&) {
            std::cout << "Categories:\n";
            for(const auto& itr : _category_options)
                std::cout << "    " << itr << "\n";
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[COLUMN OPTIONS]" }, "");
    parser.add_argument({ "-b", "--brief" }, "Suppress availability/value info")
        .max_count(1)
        .action([](parser_t& p) { force_brief = p.get<bool>("brief"); });
    parser.add_argument({ "-d", "--description" }, "Display the component description")
        .max_count(1);
    parser
        .add_argument({ "--categories" },
                      "Display the category information (use --list-categories to see "
                      "the available categories)")
        .dtype("string")
        .action([&_category_options](parser_t& p) {
            process_categories(p, _category_options);
        });
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

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[WIDTH OPTIONS]" }, "");
    parser
        .add_argument({ "-w", "--width" },
                      "if w > 0, truncate any columns greater than this width")
        .count(1)
        .dtype("int")
        .action([](parser_t& p) { max_width = p.get<int32_t>("width"); });
    parser
        .add_argument(
            { "-c", "--columns" },
            std::string{ "if c > 0, truncate the total width of all the columns to this "
                         "value. Set '-w 0 -c 0' to remove all truncation" } +
                col_msg)
        .set_default(num_cols)
        .count(1)
        .dtype("int")
        .action([](parser_t& p) { num_cols = p.get<int32_t>("columns"); });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[OUTPUT OPTIONS]" }, "");
    parser.add_argument({ "-O", "--output" }, "Write results to file")
        .count(1)
        .dtype("filename");
    parser.add_argument({ "-M", "--markdown" }, "Write data in markdown")
        .max_count(1)
        .action([](parser_t& p) { markdown = p.get<bool>("markdown"); });

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

    if(parser.exists("list-categories")) return EXIT_SUCCESS;

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

    if(options[CATEGORY] && force_brief) options[CATEGORY] = false;

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

    if(!os) os = &std::cout;

    omnitrace_init_library();

    if(include_components) write_component_info(*os, options, use_mark, fields);

    dump_log();

    if(include_settings)
        write_settings_info(
            *os, { options[VAL], options[LANG], options[DESC], options[CATEGORY] });

    dump_log();

    if(include_hw_counters)
        write_hw_counter_info(*os, { true, !force_brief && !available_only,
                                     !options[DESC], options[DESC] });

    dump_log();

    return 0;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename I>
struct enumerated_list;

template <template <typename...> class TupT, typename... T>
struct enumerated_list<TupT<T...>, index_sequence<>>
{
    using type = type_list<T...>;
};

template <template <typename...> class TupT, size_t I, typename... T, size_t... Idx>
struct enumerated_list<TupT<T...>, index_sequence<I, Idx...>>
{
    using Tp                         = component::enumerator_t<I>;
    static constexpr bool is_nothing = concepts::is_placeholder<Tp>::value;
    using type                       = typename enumerated_list<
        tim::conditional_t<is_nothing, type_list<T...>, type_list<T..., Tp>>,
        index_sequence<Idx...>>::type;
};

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

    using index_seq_t = make_index_sequence<TIMEMORY_COMPONENTS_END>;
    using enum_list_t = typename enumerated_list<tim::type_list<>, index_seq_t>::type;

    std::vector<info_type> _info{};
    get_availability<>{}(enum_list_t{}, _info);

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

            _selected += (is_category_selected(std::get<2>(itr).at(CATEGORY))) ? 1 : 0;

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

    if(!markdown) os << banner(_widths, _wusing, '-');

    os << global_delim;
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
            write_entry(ss, std::get<2>(itr).at(i), _widths.at(i + 2), center,
                        _mark.at(i));
        }

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

    os << "\n" << std::flush;
    // os << banner(total_width) << std::flush;
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

    // this settings has delayed initialization. make sure it is generated
    (void) omnitrace::config::get_perfetto_output_filename();

    array_type _setting_output;

    cereal::SettingsTextArchive settings_archive{ _setting_output, settings_exclude };
    settings::serialize_settings(settings_archive);

    _setting_output.erase(
        std::remove_if(_setting_output.begin(), _setting_output.end(),
                       [](const auto& itr) { return itr.find("environ") == itr.end(); }),
        _setting_output.end());

    // patch up the categories
    str_set_t _not_in_category_view{};
    auto      _settings = tim::settings::shared_instance();
    for(auto& itr : _setting_output)
    {
        auto _name = itr.find("environ")->second;
        auto sitr  = _settings->find(_name);
        if(sitr != _settings->end())
        {
            str_set_t _categories{};
            for(const auto& citr : sitr->second->get_categories())
            {
                if(citr == "native")
                    _categories.emplace("settings::timemory");
                else
                    _categories.emplace(TIMEMORY_JOIN("::", "settings", citr));
            }
            bool _found = false;
            for(const auto& citr : _categories)
            {
                if(category_view.count(citr) > 0) _found = true;
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

    os << global_delim;
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
            write_entry(ss, itr.at(i), _widths.at(i), _center.at(i), _mark.at(i));
        }

        if(_selected > 0)
        {
            os << global_delim;
            os << hl_selected(ss.str());
            os << "\n";
        }
    }

    dump_log();

    if(!markdown) os << banner(_widths, _wusing, '-');

    os << "\n" << std::flush;
    // os << banner(total_width, '-') << std::flush;
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

    using width_type = array_t<int64_t, N>;
    using width_bool = array_t<bool, N>;

    auto _papi_events = tim::papi::available_events_info();

    auto _process_counters = [](auto& _events, int32_t _offset) {
        for(auto& itr : _events)
        {
            itr.offset() += _offset;
            itr.python_symbol() = "timemory.hardware_counters." + itr.python_symbol();
        }
        return static_cast<int32_t>(_events.size());
    };

    int32_t _offset = 0;
    _offset += _process_counters(_papi_events, _offset);

    using hwcounter_info_t             = std::vector<tim::hardware_counters::info>;
    auto                 fields        = std::vector<hwcounter_info_t>{ _papi_events };
    auto                 subcategories = std::vector<std::string>{ "CPU", "GPU", "" };
    array_t<string_t, N> _labels       = { "HARDWARE COUNTER", "AVAILABLE", "SUMMARY",
                                     "DESCRIPTION" };
    array_t<bool, N>     _center       = { false, true, false, false };

    width_type _widths;
    width_bool _wusing;
    width_bool _mark = { false, true, false, false };
    _widths.fill(0);
    _wusing.fill(false);
    for(size_t i = 0; i < _widths.size(); ++i)
    {
        _widths.at(i) = _labels.at(i).length() + padding;
        _wusing.at(i) = options[i];
    }

    for(const auto& fitr : fields)
    {
        for(const auto& itr : fitr)
        {
            if(available_only && !itr.available()) continue;
            width_type _w = { { (int64_t) itr.symbol().length(), (int64_t) 6,
                                (int64_t) itr.short_description().length(),
                                (int64_t) itr.long_description().length() } };
            for(auto& witr : _w)
                witr += padding;

            for(size_t i = 0; i < N; ++i)
                _widths.at(i) = std::max<uint64_t>(_widths.at(i), _w.at(i));
        }
    }

    _widths = compute_max_columns(_widths, _wusing);

    if(!markdown) os << banner(_widths, _wusing, '-');
    os << global_delim;

    for(size_t i = 0; i < _labels.size(); ++i)
    {
        if(options[i]) write_entry(os, _labels.at(i), _widths.at(i), true, false);
    }
    os << "\n" << banner(_widths, _wusing, '-');

    size_t nitr = 0;
    for(const auto& fitr : fields)
    {
        auto idx = nitr++;

        if(idx < subcategories.size())
        {
            if(!markdown && idx != 0) os << banner(_widths, _wusing, '-');
            if(subcategories.at(idx).length() > 0)
            {
                os << global_delim;
                if(options[0])
                {
                    write_entry(os, subcategories.at(idx), _widths.at(0), true,
                                _mark.at(0));
                }
                for(size_t i = 1; i < N; ++i)
                {
                    if(options[i])
                        write_entry(os, "", _widths.at(i), _center.at(i), _mark.at(i));
                }
                os << "\n";
                if(!markdown) os << banner(_widths, _wusing, '-');
            }
        }
        else
        {
            if(!markdown) os << banner(_widths, _wusing, '-');
        }

        for(const auto& itr : fitr)
        {
            if(available_only && !itr.available()) continue;
            std::stringstream ss;
            int               _selected = 0;
            if(options[0])
            {
                _selected += (is_selected(itr.symbol())) ? 1 : 0;
                write_entry(ss, itr.symbol(), _widths.at(0), _center.at(0), _mark.at(0));
            }

            if(options[1])
            {
                std::stringstream _avss{};
                _avss << std::boolalpha << itr.available();
                _selected += (is_selected(_avss.str())) ? 1 : 0;
                write_entry(ss, itr.available(), _widths.at(1), _center.at(1),
                            _mark.at(1));
            }

            array_t<string_t, N> _e = { { "", "", itr.short_description(),
                                          itr.long_description() } };
            for(size_t i = 2; i < N; ++i)
            {
                if(options[i])
                {
                    _selected += (is_selected(_e.at(i))) ? 1 : 0;
                    write_entry(ss, _e.at(i), _widths.at(i), _center.at(i), _mark.at(i));
                }
            }

            if(_selected > 0)
            {
                os << global_delim;
                os << hl_selected(ss.str());
                os << "\n";
            }
        }
    }

    dump_log();

    if(!markdown) os << banner(_widths, _wusing, '-');

    os << "\n" << std::flush;
}

//======================================================================================//

struct unknown
{};

template <typename T, typename U = typename T::value_type>
constexpr bool
available_value_type_alias(int)
{
    return true;
}

template <typename T, typename U = unknown>
constexpr bool
available_value_type_alias(long)
{
    return false;
}

template <typename Type, bool>
struct component_value_type;

template <typename Type>
struct component_value_type<Type, true>
{
    using type = typename Type::value_type;
};

template <typename Type>
struct component_value_type<Type, false>
{
    using type = unknown;
};

template <typename Type>
using component_value_type_t =
    typename component_value_type<Type, available_value_type_alias<Type>(0)>::type;

//--------------------------------------------------------------------------------------//

template <typename... Tp>
auto get_categories(type_list<Tp...>)
{
    auto _cleanup = [](std::string _type, const std::string& _pattern) {
        auto _pos = std::string::npos;
        while((_pos = _type.find(_pattern)) != std::string::npos)
            _type.erase(_pos, _pattern.length());
        return _type;
    };
    (void) _cleanup;  // unused but set if sizeof...(Tp) == 0

    auto _vec = str_vec_t{ _cleanup(demangle<Tp>(), "tim::")... };
    std::sort(_vec.begin(), _vec.end(), [](const auto& lhs, const auto& rhs) {
        // prioritize project category
        auto lpos = lhs.find("project::");
        auto rpos = rhs.find("project::");
        return (lpos == rpos) ? (lhs < rhs) : (lpos < rpos);
    });
    std::stringstream _ss{};
    for(auto&& itr : _vec)
    {
        _ss << ", " << itr;
    }
    std::string _v = _ss.str();
    if(!_v.empty()) return _v.substr(2);
    return _v;
}

//--------------------------------------------------------------------------------------//

template <typename Type>
info_type
get_availability<Type>::get_info()
{
    using value_type     = component_value_type_t<Type>;
    using category_types = typename trait::component_apis<Type>::type;

    auto _cleanup = [](std::string _type, const std::string& _pattern) {
        auto _pos = std::string::npos;
        while((_pos = _type.find(_pattern)) != std::string::npos)
            _type.erase(_pos, _pattern.length());
        return _type;
    };
    auto _replace = [](std::string _type, const std::string& _pattern,
                       const std::string& _with) {
        auto _pos = std::string::npos;
        while((_pos = _type.find(_pattern)) != std::string::npos)
            _type.replace(_pos, _pattern.length(), _with);
        return _type;
    };

    bool has_metadata   = metadata_t::specialized();
    bool has_properties = property_t::specialized();
    bool is_available   = trait::is_available<Type>::value;
    bool file_output    = trait::generates_output<Type>::value;
    auto name           = component::metadata<Type>::name();
    auto label          = (file_output)
                              ? ((has_metadata) ? metadata_t::label() : Type::get_label())
                              : std::string("");
    auto description =
        (has_metadata) ? metadata_t::description() : Type::get_description();
    auto     data_type = demangle<value_type>();
    string_t enum_type = property_t::enum_string();
    string_t id_type   = property_t::id();
    auto     ids_set   = property_t::ids();

    if(!has_properties)
    {
        enum_type = "";
        id_type   = "";
        ids_set.clear();
    }
    string_t ids_str = {};
    {
        auto     itr = ids_set.begin();
        string_t db  = (markdown) ? "`\"" : "\"";
        string_t de  = (markdown) ? "\"`" : "\"";
        if(has_metadata) description += ". " + metadata_t::extra_description();
        description += ".";
        while(itr->empty())
            ++itr;
        if(itr != ids_set.end())
            ids_str = TIMEMORY_JOIN("", TIMEMORY_JOIN("", db, *itr++, de));
        for(; itr != ids_set.end(); ++itr)
        {
            if(!itr->empty())
                ids_str = TIMEMORY_JOIN("  ", ids_str, TIMEMORY_JOIN("", db, *itr, de));
        }
    }

    string_t categories = get_categories(category_types{});

#if 0
    auto _remove_typelist = [](std::string _tmp) {
        if(_tmp.empty()) return _tmp;
        auto _key = std::string{ "type_list" };
        auto _idx = _tmp.find(_key);
        if(_idx == std::string::npos) return _tmp;
        _idx = _tmp.find('<', _idx);
        _tmp = _tmp.substr(_idx + 1);
        _idx = _tmp.find_last_of('>');
        _tmp = _tmp.substr(0, _idx);
        if(_tmp.empty()) return _tmp;
        // strip trailing whitespaces
        while((_idx = _tmp.find_last_of(' ')) == _tmp.length() - 1)
            _tmp = _tmp.substr(0, _idx);
        return _tmp;
    };
    auto     apis      = _remove_typelist(demangle<trait::component_apis_t<Type>>());
    if(!apis.empty()) description += ". APIs: " + apis;
#endif

    description = _replace(_replace(description, ". .", "."), "..", ".");
    data_type   = _replace(_cleanup(data_type, "::__1"), "> >", ">>");
    return info_type{ name, is_available,
                      str_vec_t{ data_type, enum_type, id_type, ids_str, label,
                                 description, categories } };
}

//--------------------------------------------------------------------------------------//

string_t
remove(string_t inp, const std::set<string_t>& entries)
{
    for(const auto& itr : entries)
    {
        auto idx = inp.find(itr);
        while(idx != string_t::npos)
        {
            inp.erase(idx, itr.length());
            idx = inp.find(itr);
        }
    }
    return inp;
}

//--------------------------------------------------------------------------------------//

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
    ssentry << ' ' << std::boolalpha << ((mark && markdown) ? "`" : "") << _entry;
    auto _sentry = remove(ssentry.str(), { "tim::", "component::" });

    auto _decr = (mark && markdown) ? 6 : 5;
    if(_w > 0 && _sentry.length() > static_cast<size_t>(_w - 2))
        _sentry = _sentry.substr(0, _w - _decr) + "...";

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
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename IntArrayT, size_t N>
string_t
banner(IntArrayT _breaks, std::array<bool, N> _use, char filler, char delim)
{
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

namespace regex_const = std::regex_constants;

namespace
{
const auto&
get_regex_constants()
{
    static auto _constants = []() {
        auto _v = regex_const::egrep | regex_const::optimize;
        if(case_insensitive) _v |= regex_const::icase;
        return _v;
    }();
    return _constants;
}

const std::string&
get_regex_pattern()
{
    static std::string _pattern = []() {
        std::string _v{};
        for(const auto& itr : regex_keys)
        {
            lerr << "Adding regex key: '" << itr << "'...\n";
            _v += "|" + itr;
        }
        return (_v.empty()) ? _v : _v.substr(1);
    }();
    return _pattern;
}

auto
get_regex()
{
    static auto _rc = std::regex(get_regex_pattern(), get_regex_constants());
    return _rc;
}

bool
regex_match(const std::string& _line)
{
    if(get_regex_pattern().empty()) return true;

    static size_t lerr_width = 0;
    lerr_width               = std::max<size_t>(lerr_width, _line.length());
    std::stringstream _line_ss;
    _line_ss << "'" << _line << "'";

    if(std::regex_match(_line, get_regex()))
    {
        lerr << std::left << std::setw(lerr_width) << _line_ss.str()
             << " matched pattern '" << get_regex_pattern() << "'...\n";
        return true;
    }
    if(std::regex_search(_line, get_regex()))
    {
        lerr << std::left << std::setw(lerr_width) << _line_ss.str() << " found pattern '"
             << get_regex_pattern() << "'...\n";
        return true;
    }

    lerr << std::left << std::setw(lerr_width) << _line_ss.str() << " missing pattern '"
         << get_regex_pattern() << "'...\n";
    return false;
}

std::string
regex_replace(const std::string& _line)
{
#if defined(TIMEMORY_UNIX)
    if(get_regex_pattern().empty()) return _line;
    if(regex_match(_line))
        return std::regex_replace(_line, get_regex(), "\33[01;04;36;40m$&\33[0m");
#endif
    return _line;
}

const std::string&
get_category_regex_pattern()
{
    static std::string _pattern = []() {
        std::string _v{};
        for(const auto& itr : category_regex_keys)
        {
            lerr << "Adding regex key: '" << itr << "'...\n";
            _v += "|" + itr;
        }
        return (_v.empty()) ? _v : _v.substr(1);
    }();
    return _pattern;
}

auto
get_category_regex()
{
    static auto _rc = std::regex(get_category_regex_pattern(), get_regex_constants());
    return _rc;
}

bool
category_regex_match(const std::string& _line)
{
    if(get_regex_pattern().empty()) return true;

    static size_t lerr_width = 0;
    lerr_width               = std::max<size_t>(lerr_width, _line.length());
    std::stringstream _line_ss;
    _line_ss << "'" << _line << "'";

    if(std::regex_match(_line, get_category_regex()))
    {
        lerr << std::left << std::setw(lerr_width) << _line_ss.str()
             << " matched pattern '" << get_category_regex_pattern() << "'...\n";
        return true;
    }
    if(std::regex_search(_line, get_category_regex()))
    {
        lerr << std::left << std::setw(lerr_width) << _line_ss.str() << " found pattern '"
             << get_category_regex_pattern() << "'...\n";
        return true;
    }

    lerr << std::left << std::setw(lerr_width) << _line_ss.str() << " missing pattern '"
         << get_category_regex_pattern() << "'...\n";
    return false;
}

bool
exclude_setting(const std::string& _v)
{
    bool _a = settings_exclude.find(_v) != settings_exclude.end();
    bool _b = std::regex_match(_v, std::regex{ settings_rexclude_exact });
    bool _c = std::regex_match(_v, std::regex{ settings_rexclude_begin });
    return (_a || _b || _c);
}
}  // namespace

//--------------------------------------------------------------------------------------//

bool
is_selected(const std::string& _line)
{
    return regex_match(_line);
}

//--------------------------------------------------------------------------------------//

bool
is_category_selected(const std::string& _line)
{
    return category_regex_match(_line);
}

//--------------------------------------------------------------------------------------//

std::string
hl_selected(const std::string& _line)
{
    return (regex_hl) ? regex_replace(_line) : _line;
}

//--------------------------------------------------------------------------------------//

void
process_categories(parser_t& p, const str_set_t& _category_options)
{
    category_view = p.get<str_set_t>("categories");
    std::vector<std::function<void()>> _shorthand_patches{};
    for(const auto& itr : category_view)
    {
        auto _is_shorthand = [&_shorthand_patches, &_category_options,
                              itr](const std::string& _prefix) {
            auto _opt = TIMEMORY_JOIN("::", _prefix, itr);
            if(_category_options.count(_opt) > 0)
            {
                _shorthand_patches.emplace_back([itr, _opt]() {
                    category_view.erase(itr);
                    category_view.emplace(_opt);
                });
                return true;
            }
            return false;
        };

        if(_category_options.count(itr) == 0)
        {
            if(!_is_shorthand("component") && !_is_shorthand("settings"))
                throw std::runtime_error(
                    itr + " is not a valid category. Use --list-categories to view "
                          "valid categories");
        }
    }
    for(auto&& itr : _shorthand_patches)
        itr();
}

//--------------------------------------------------------------------------------------//
