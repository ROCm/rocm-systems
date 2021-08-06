// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iosfwd>
#include <list>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tim
{
namespace argparse
{
namespace helpers
{
//
//--------------------------------------------------------------------------------------//
//
static inline bool
not_is_space(int ch)
{
    return std::isspace(ch) == 0;
}
//
//--------------------------------------------------------------------------------------//
//
static inline uint64_t
lcount(const std::string& s, bool (*f)(int) = not_is_space)
{
    uint64_t c = 0;
    for(size_t i = 0; i < s.length(); ++i, ++c)
    {
        if(f(s.at(i)))
            break;
    }
    return c;
}
//
//--------------------------------------------------------------------------------------//
//
static inline std::string
ltrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
static inline std::string
rtrim(std::string s, bool (*f)(int) = not_is_space)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
static inline std::string
trim(std::string s, bool (*f)(int) = not_is_space)
{
    ltrim(s, f);
    rtrim(s, f);
    return s;
}
//
//--------------------------------------------------------------------------------------//
//
static inline char*
strdup(const char* s)
{
    auto  slen   = strlen(s);
    auto* result = new char[slen + 1];
    if(result)
    {
        memcpy(result, s, slen * sizeof(char));
        result[slen] = '\0';
        return result;
    }
    return nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename InputIt>
static inline std::string
join(InputIt begin, InputIt end, const std::string& separator = " ")
{
    std::ostringstream ss;
    if(begin != end)
    {
        ss << *begin++;
    }
    while(begin != end)
    {
        ss << separator;
        ss << *begin++;
    }
    return ss.str();
}
//
//--------------------------------------------------------------------------------------//
//
static inline bool
is_numeric(const std::string& arg)
{
    auto _nidx = arg.find_first_of("0123456789");
    auto _oidx = arg.find_first_not_of("0123456789.Ee+-*/");

    // must have number somewhere
    if(_nidx == std::string::npos)
        return false;

    // if something other than number or scientific notation
    if(_oidx != std::string::npos)
        return false;

    // numbers + possible scientific notation
    return true;
}
//
//--------------------------------------------------------------------------------------//
//
static inline int
find_equiv(const std::string& s)
{
    for(size_t i = 0; i < s.length(); ++i)
    {
        // if find graph symbol before equal, end search
        // i.e. don't accept --asd)f=0 arguments
        // but allow --asd_f and --asd-f arguments
        if(std::ispunct(static_cast<int>(s[i])) != 0)
        {
            if(s[i] == '=')
            {
                return static_cast<int>(i);
            }
            if(s[i] == '_' || s[i] == '-')
            {
                continue;
            }
            return -1;
        }
    }
    return -1;
}
//
//--------------------------------------------------------------------------------------//
//
static inline size_t
find_punct(const std::string& s)
{
    size_t i;
    for(i = 0; i < s.length(); ++i)
    {
        if((std::ispunct(static_cast<int>(s[i])) != 0) && s[i] != '-')
        {
            break;
        }
    }
    return i;
}
//
//--------------------------------------------------------------------------------------//
//
namespace is_container_impl
{
//
template <typename T>
struct is_container : std::false_type
{};
//
template <typename... Args>
struct is_container<std::vector<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::set<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::deque<Args...>> : std::true_type
{};
template <typename... Args>
struct is_container<std::list<Args...>> : std::true_type
{};
//
template <typename T>
struct is_initializing_container : is_container<T>::type
{};
//
template <typename... Args>
struct is_initializing_container<std::initializer_list<Args...>> : std::true_type
{};
}  // namespace is_container_impl
//
//--------------------------------------------------------------------------------------//
//
// type trait to utilize the implementation type traits as well as decay the type
template <typename T>
struct is_container
{
    static constexpr bool const value =
        is_container_impl::is_container<decay_t<T>>::value;
};
//
//--------------------------------------------------------------------------------------//
//
// type trait to utilize the implementation type traits as well as decay the type
template <typename T>
struct is_initializing_container
{
    static constexpr bool const value =
        is_container_impl::is_initializing_container<decay_t<T>>::value;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace helpers

//
//--------------------------------------------------------------------------------------//
//
//                      argument vector
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::argparse::argument_vector
/// \brief This class exists to simplify creating argument arrays compatible with execv*
/// routines and MPI_Comm_spawn/MPI_Comm_spawn_multiple
///
struct argument_vector : std::vector<std::string>
{
    struct c_args : std::tuple<int, char**, std::string>
    {
        using base_type = std::tuple<int, char**, std::string>;
        template <typename... Args>
        c_args(Args&&... args)
        : base_type(std::forward<Args>(args)...)
        {}

        auto& argc() { return std::get<0>(*this); }
        auto& argv() { return std::get<1>(*this); }
        auto& args() { return std::get<2>(*this); }

        TIMEMORY_NODISCARD const auto& argc() const { return std::get<0>(*this); }
        TIMEMORY_NODISCARD const auto& argv() const { return std::get<1>(*this); }
        TIMEMORY_NODISCARD const auto& args() const { return std::get<2>(*this); }

        void clear()
        {
            // uses comma operator to execute delete and return nullptr
            for(int i = 0; i < argc(); ++i)
                argv()[i] = (delete[] argv()[i], nullptr);
            argv() = (delete[] argv(), nullptr);
        }
    };

    using base_type = std::vector<std::string>;
    using cargs_t   = c_args;

    template <typename... Args>
    argument_vector(Args&&... args)
    : base_type(std::forward<Args>(args)...)
    {}

    explicit argument_vector(int& argc, char**& argv);
    explicit argument_vector(int& argc, const char**& argv);
    explicit argument_vector(int& argc, const char* const*& argv);
    TIMEMORY_NODISCARD cargs_t
                       get_execv(const base_type& _prepend, size_t _beg = 0,
                                 size_t _end = std::numeric_limits<size_t>::max()) const;
    TIMEMORY_NODISCARD cargs_t
                       get_execv(size_t _beg = 0, size_t _end = std::numeric_limits<size_t>::max()) const;

    // helper function to free the memory created by get_execv, pass by reference
    // so that we can set values to nullptr and avoid multiple delete errors
    static void free_execv(cargs_t& itr) { itr.clear(); }
};
//
//--------------------------------------------------------------------------------------//
//
//                      argument parser
//
//--------------------------------------------------------------------------------------//
//
struct argument_parser
{
    struct arg_result;

    using this_type     = argument_parser;
    using result_type   = arg_result;
    using bool_func_t   = std::function<bool(this_type&)>;
    using action_func_t = std::function<void(this_type&)>;
    using action_pair_t = std::pair<bool_func_t, action_func_t>;
    using error_func_t  = std::function<void(this_type&, arg_result&)>;
    using known_args_t  = std::tuple<arg_result, int, char**>;
    using strvec_t      = std::vector<std::string>;
    using strset_t      = std::set<std::string>;
    //
    //----------------------------------------------------------------------------------//
    //
    struct arg_result
    {
        arg_result() = default;
        arg_result(std::string err) noexcept
        : m_error(true)
        , m_what(std::move(err))
        {}

        operator bool() const { return m_error; }

        friend std::ostream& operator<<(std::ostream& os, const arg_result& dt);

        TIMEMORY_NODISCARD const std::string& what() const { return m_what; }

    private:
        bool        m_error = false;
        std::string m_what  = {};
    };
    //
    //----------------------------------------------------------------------------------//
    //
    struct argument
    {
        using callback_t = std::function<void(void*&)>;

        enum Position : int
        {
            LastArgument   = -1,
            IgnoreArgument = -2
        };

        enum Count : int
        {
            ANY = -1
        };

        ~argument() { m_destroy(m_default); }

        argument& name(const std::string& name)
        {
            m_names.push_back(name);
            return *this;
        }

        argument& names(const std::vector<std::string>& names)
        {
            for(const auto& itr : names)
                m_names.push_back(itr);
            return *this;
        }

        argument& description(const std::string& description)
        {
            m_desc = description;
            return *this;
        }

        argument& dtype(const std::string& _dtype)
        {
            m_dtype = _dtype;
            return *this;
        }

        argument& required(bool req)
        {
            m_required = req;
            return *this;
        }

        argument& position(int position)
        {
            if(position != Position::LastArgument)
            {
                // position + 1 because technically argument zero is the name of the
                // executable
                m_position = position + 1;
            }
            else
            {
                m_position = position;
            }
            return *this;
        }

        argument& max_count(int count)
        {
            m_max_count = count;
            return *this;
        }

        argument& min_count(int count)
        {
            m_min_count = count;
            return *this;
        }

        argument& count(int count)
        {
            m_count = count;
            return *this;
        }

        template <typename T>
        argument& set_default(const T& val)
        {
            m_found        = true;
            m_default_tidx = std::type_index{ typeid(decay_t<T>) };
            m_callback     = [&](void*& obj) {
                m_destroy(obj);
                if(!obj)
                    obj = (void*) new T{};
                (*static_cast<T*>(obj)) = val;
            };
            m_destroy = [](void*& obj) {
                if(obj)
                    delete static_cast<T*>(obj);
            };
            return *this;
        }

        template <typename T>
        argument& set_default(T& val)
        {
            m_found        = true;
            m_default_tidx = std::type_index{ typeid(decay_t<T>) };
            m_callback     = [&](void*& obj) { obj = (void*) &val; };
            return *this;
        }

        template <typename T>
        argument& choices(const std::initializer_list<T>& _choices)
        {
            for(auto&& itr : _choices)
            {
                std::stringstream ss;
                ss << itr;
                m_choices.insert(ss.str());
            }
            return *this;
        }

        template <template <typename...> class ContainerT, typename T, typename... ExtraT,
                  typename ContT = ContainerT<T, ExtraT...>,
                  enable_if_t<helpers::is_container<ContT>::value> = 0>
        argument& choices(const ContainerT<T, ExtraT...>& _choices)
        {
            for(auto&& itr : _choices)
            {
                std::stringstream ss;
                ss << itr;
                m_choices.insert(ss.str());
            }
            return *this;
        }

        template <typename ActionFuncT>
        argument& action(ActionFuncT&& _func)
        {
            m_actions.push_back(std::forward<ActionFuncT>(_func));
            return *this;
        }

        TIMEMORY_NODISCARD bool found() const { return m_found; }

        template <typename T>
        std::enable_if_t<helpers::is_container<T>::value, T> get()
        {
            T                      t = T{};
            typename T::value_type vt;
            for(auto& s : m_values)
            {
                std::istringstream in(s);
                in >> vt;
                t.insert(t.end(), vt);
            }
            if(m_values.empty() && m_default &&
               m_default_tidx == std::type_index{ typeid(T) })
                t = (*static_cast<T*>(m_default));
            return t;
        }

        template <typename T>
        std::enable_if_t<
            !helpers::is_container<T>::value && !std::is_same<T, bool>::value, T>
        get()
        {
            auto               inp = get<std::string>();
            std::istringstream iss{ inp };
            T                  t = T{};
            iss >> t >> std::ws;
            if(inp.empty() && m_default && m_default_tidx == std::type_index{ typeid(T) })
                t = (*static_cast<T*>(m_default));
            return t;
        }

        template <typename T>
        std::enable_if_t<std::is_same<T, bool>::value, T> get()
        {
            if(m_count == 0)
                return found();

            auto inp = get<std::string>();
            if(inp.empty() && m_default && m_default_tidx == std::type_index{ typeid(T) })
                return (*static_cast<T*>(m_default));
            else if(inp.empty())
                return found();

            return get_bool(inp, found());
        }

        TIMEMORY_NODISCARD size_t size() const { return m_values.size(); }

        TIMEMORY_NODISCARD std::string get_name() const
        {
            std::stringstream ss;
            for(const auto& itr : m_names)
                ss << "/" << itr;
            return ss.str().substr(1);
        }

    private:
        argument(const std::string& name, std::string desc, bool required = false)
        : m_desc(std::move(desc))
        , m_required(required)
        {
            m_names.push_back(name);
        }

        argument() = default;

        arg_result check_choice(const std::string& value)
        {
            if(!m_choices.empty())
            {
                if(m_choices.find(value) == m_choices.end())
                {
                    std::stringstream ss;
                    ss << "Invalid choice: '" << value << "'. Valid choices: ";
                    for(const auto& itr : m_choices)
                        ss << "'" << itr << "' ";
                    return arg_result(ss.str());
                }
            }
            return arg_result{};
        }

        void execute_actions(argument_parser& p)
        {
            for(auto& itr : m_actions)
                itr(p);
        }

        friend std::ostream& operator<<(std::ostream& os, const argument& arg)
        {
            std::stringstream ss;
            ss << "names: ";
            for(const auto& itr : arg.m_names)
                ss << itr << " ";
            ss << ", index: " << arg.m_index << ", count: " << arg.m_count
               << ", min count: " << arg.m_min_count << ", max count: " << arg.m_max_count
               << ", found: " << std::boolalpha << arg.m_found
               << ", required: " << std::boolalpha << arg.m_required
               << ", position: " << arg.m_position << ", values: ";
            for(const auto& itr : arg.m_values)
                ss << itr << " ";
            os << ss.str();
            return os;
        }

        friend struct argument_parser;
        int                        m_position     = Position::IgnoreArgument;
        int                        m_count        = Count::ANY;
        int                        m_min_count    = Count::ANY;
        int                        m_max_count    = Count::ANY;
        std::vector<std::string>   m_names        = {};
        std::string                m_desc         = {};
        std::string                m_dtype        = {};
        bool                       m_found        = false;
        bool                       m_required     = false;
        int                        m_index        = -1;
        std::type_index            m_default_tidx = std::type_index{ typeid(void) };
        void*                      m_default      = nullptr;
        callback_t                 m_callback     = [](void*&) {};
        callback_t                 m_destroy      = [](void*&) {};
        std::set<std::string>      m_choices      = {};
        std::vector<std::string>   m_values       = {};
        std::vector<action_func_t> m_actions      = {};
    };
    //
    //----------------------------------------------------------------------------------//
    //
    argument_parser(std::string desc)
    : m_desc(std::move(desc))
    {}
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_argument()
    {
        m_arguments.push_back({});
        m_arguments.back().m_index = static_cast<int>(m_arguments.size()) - 1;
        return m_arguments.back();
    }
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_argument(const std::initializer_list<std::string>& _names,
                           const std::string& desc, bool req = false)
    {
        return add_argument().names(_names).description(desc).required(req);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_argument(const std::vector<std::string>& _names,
                           const std::string& desc, bool req = false)
    {
        return add_argument().names(_names).description(desc).required(req);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    argument& add_positional_argument(const std::string& _name)
    {
        m_positional_arguments.push_back({});
        auto& _entry = m_positional_arguments.back();
        _entry.name(_name);
        _entry.count(1);
        _entry.m_index = m_positional_arguments.size();
        return _entry;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename Tp>
    arg_result get(size_t _idx, Tp& _value)
    {
        if(m_positional_values.find(_idx) == m_positional_values.end())
            return arg_result{ "Positional value not found at index " +
                               std::to_string(_idx) };
        if(_idx >= m_positional_arguments.size())
            return arg_result{ "No positional argument was specified for index " +
                               std::to_string(_idx) };
        _value = m_positional_arguments.at(_idx).get<Tp>();
        return arg_result{};
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename Tp>
    arg_result get(const std::string& _name, Tp& _value)
    {
        // loop over parsed positional args
        for(size_t i = 0; i < m_positional_values.size(); ++i)
        {
            if(i >= m_positional_arguments.size())
                break;

            // loop over added positional args
            auto& itr = m_positional_arguments.at(i);
            for(auto& nitr : itr.m_names)
            {
                if(nitr == _name)
                    return get(i, _value);
            }
        }

        // not found, check if required
        for(auto& itr : m_positional_arguments)
        {
            for(auto& nitr : itr.m_names)
            {
                if(nitr == _name)
                {
                    if(itr.m_default &&
                       itr.m_default_tidx == std::type_index{ typeid(decay_t<Tp>) })
                        _value = (*static_cast<Tp*>(itr.m_default));
                    else if(itr.m_required)
                        return arg_result{
                            _name + " not parsed from the command line (required)"
                        };
                    return arg_result{};
                }
            }
        }

        return arg_result{ _name + " is not a named positional argument" };
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename BoolFuncT, typename ActionFuncT>
    this_type& add_action(BoolFuncT&& _b, ActionFuncT& _act)
    {
        m_actions.push_back(
            { std::forward<BoolFuncT>(_b), std::forward<ActionFuncT>(_act) });
        return *this;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ActionFuncT>
    this_type& add_action(const std::string& _name, ActionFuncT& _act)
    {
        auto _b = [=](this_type& p) { return p.exists(_name); };
        m_actions.push_back({ _b, std::forward<ActionFuncT>(_act) });
        return *this;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void print_help(const std::string& _extra = "");
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse_known_args(int argc, char** argv, const std::string& delim,
    ///                                 int verb)
    /// \param[in,out] argc Number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Array of strings (i.e. command-line)
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief Basic variant of \ref parse_known_args which does not replace argc/argv
    /// and does not provide an array of strings that it processed
    ///
    known_args_t parse_known_args(int argc, char** argv, const std::string& _delim = "--",
                                  int verbose_level = 0)
    {
        strvec_t _args{};
        return parse_known_args(argc, argv, _args, _delim, verbose_level);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result parse_known_args(int* argc, char*** argv, const std::string& _delim = "--",
                                int verbose_level = 0)
    {
        strvec_t args{};
        return parse_known_args(argc, argv, args, _delim, verbose_level);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse_known_args(int* argc, char*** argv, const std::string& delim,
    ///                                 int verb)
    /// \param[in,out] argc Pointer to number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Pointer to array of strings (i.e. command-line)
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief This variant calls \ref parse_known_args and replaces argc and argv with
    /// the argv[0] + anything after delimiter (if the delimiter is provided). If the
    /// delimiter does not exist, argc and argv are unchanged.
    ///
    arg_result parse_known_args(int* argc, char*** argv, strvec_t& _args,
                                const std::string& _delim = "--", int verbose_level = 0);
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse_known_args(int argc, char** argv, strvec_t& args, const
    ///                                 std::string& delim, int verb)
    /// \param[in,out] argc Number of arguments (i.e. # of command-line args)
    /// \param[in,out] argv Array of strings (i.e. command-line)
    /// \param[in,out] args Array of strings processed by this parser
    /// \param[in] delim Delimiter which separates this argparser's opts from user's
    /// arguments
    /// \param[in] verb verbosity
    ///
    /// \brief Parses all options until argv[argc-1] or delimiter is found.
    /// Returns a tuple containing an argument error object (operator bool will return
    /// true if there was an error) and the new argc and argv after the known arguments
    /// have been processed. This is slightly different from the Python
    /// argparse.ArgumentParser.parse_known_args: if the delimiter is not found, it will
    /// not remove the arguments that it recognizes.
    /// To distinguish this parsers options from user arguments, use the syntax:
    ///
    ///    ./<CMD> <PARSER_OPTIONS> -- <USER_ARGS>
    ///
    /// And std::get<1>(...) on the return value will be the new argc.
    /// and std::get<2>(...) on the return value will be the new argv.
    /// Other valid usages:
    ///
    ///    ./<CMD> --help       (will report this parser's help message)
    ///    ./<CMD> -- --help    (will report the applications help message, if supported)
    ///    ./<CMD> <USER_ARGS>
    ///    ./<CMD> <PARSER_OPTIONS>
    ///    ./<CMD> <PARSER_OPTIONS> <USER_ARGS> (intermixed)
    ///
    /// will not remove any of the known options.
    /// In other words, this will remove all arguments after <CMD> until the first "--" if
    /// reached and everything after the "--" will be placed in argv[1:]
    ///
    known_args_t parse_known_args(int argc, char** argv, strvec_t& _args,
                                  const std::string& _delim        = "--",
                                  int                verbose_level = 0);
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename... Args>
    arg_result parse_args(Args&&... args)
    {
        return parse(std::forward<Args>(args)...);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result parse(int argc, char** argv, int verbose_level = 0)
    {
        std::vector<std::string> _args;
        _args.reserve(argc);
        for(int i = 0; i < argc; ++i)
            _args.emplace_back((const char*) argv[i]);
        return parse(_args, verbose_level);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn arg_result parse(const std::vector<std::string>& args, int verb)
    /// \param[in] args Array of strings (i.e. command-line arguments)
    /// \param[in] verb Verbosity
    ///
    /// \brief This is the primary function for parsing the command line arguments.
    /// This is where the map of the options is built and the loop over the
    /// arguments is performed.
    arg_result parse(const std::vector<std::string>& _args, int verbose_level = 0);
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn argument& enable_help()
    /// \brief Add a help command
    argument& enable_help()
    {
        m_help_enabled = true;
        return add_argument()
            .names({ "-h", "-?", "--help" })
            .description("Shows this page")
            .count(0);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn bool exists(const std::string& name) const
    /// \brief Returns whether or not an option was found in the arguments. Only
    /// useful after a call to \ref parse or \ref parse_known_args.
    ///
    /// \code{.cpp}
    ///
    /// int main(int argc, char** argv)
    /// {
    ///     argument_parser p{ argv[0] };
    ///     p.add_argument()
    ///         .names({ "-h", "--help"})
    ///         .description("Help message")
    ///         .count(0);
    ///
    ///     auto ec = p.parse(argc, argv);
    ///     if(ec)
    ///     {
    ///         std::cerr << "Error: " << ec << std::endl;
    ///         exit(EXIT_FAILURE);
    ///     }
    ///
    ///     if(p.exists("help"))
    ///     {
    ///         p.print_help();
    ///         exit(EXIT_FAILURE);
    ///     }
    ///
    ///     // ...
    /// }
    /// \endcode
    TIMEMORY_NODISCARD bool exists(const std::string& name) const
    {
        std::string n = helpers::ltrim(
            name, [](int c) -> bool { return c != static_cast<int>('-'); });
        auto itr = m_name_map.find(n);
        if(itr != m_name_map.end())
            return m_arguments[static_cast<size_t>(itr->second)].m_found;
        return false;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    /// \fn T get(const std::string& name)
    /// \tparam T Data type to convert the argument into
    /// \param[in] name An identifier of the option
    ///
    /// \brief Get the value(s) associated with an argument. If option, it should
    /// be used in conjunction with \ref exists(name). Only useful after a call to \ref
    /// parse or \ref parse_known_args.
    ///
    /// \code{.cpp}
    ///
    /// int main(int argc, char** argv)
    /// {
    ///     argument_parser p{ argv[0] };
    ///     p.add_argument()
    ///         .names({ "-n", "--iterations"})
    ///         .description("Number of iterations")
    ///         .count(1);
    ///
    ///     // ... etc.
    ///
    ///     auto nitr = p.get<size_t>("iteration");
    ///
    ///     // ... etc.
    /// }
    /// \endcode
    template <typename T>
    T get(const std::string& name)
    {
        auto itr = m_name_map.find(name);
        if(itr != m_name_map.end())
            return m_arguments[static_cast<size_t>(itr->second)].get<T>();
        return T{};
    }
    //
    //----------------------------------------------------------------------------------//
    //
    int64_t get_count(const std::string& name)
    {
        auto itr = m_name_map.find(name);
        if(itr != m_name_map.end())
            return m_arguments[static_cast<size_t>(itr->second)].size();
        return 0;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    int64_t get_positional_count() const { return m_positional_values.size(); }
    //
    //----------------------------------------------------------------------------------//
    //
    static int64_t get_count(argument& a) { return a.m_values.size(); }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ErrorFuncT>
    void on_error(ErrorFuncT&& _func)
    {
        on_error_sfinae(std::forward<ErrorFuncT>(_func), 0);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    void set_help_width(int _v) { m_width = _v; }

private:
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename FuncT = std::function<bool(int, int)>>
    arg_result check_count(argument& a, const std::string& _do_str = "111",
                           const FuncT& _func = std::not_equal_to<int>{})
    {
        int _sz  = static_cast<int>(a.m_values.size());
        int _cnt = a.m_count;
        int _max = a.m_max_count;
        int _min = a.m_min_count;

        std::bitset<3> _do{ _do_str };
        std::bitset<3> _checks;
        _checks.reset();  // set all to false
        // if <val> > ANY AND <val does not satisfies condition> -> true
        _checks.set(0, _do.test(0) && _cnt > argument::Count::ANY && _func(_sz, _cnt));
        _checks.set(1, _do.test(1) && _max > argument::Count::ANY && _sz > _max);
        _checks.set(2, _do.test(2) && _min > argument::Count::ANY && _sz < _min);
        // if no checks failed, return non-error
        if(_checks.none())
            return arg_result{};
        // otherwise, compose an error message
        std::stringstream msg;
        msg << "Argument: " << a.get_name() << " failed to satisfy its argument count "
            << "requirements. Number of arguments: " << _sz << ".";
        if(_checks.test(0))
        {
            msg << "\n[" << a.get_name() << "]> Requires exactly " << _cnt << " values.";
        }
        else
        {
            if(_checks.test(1))
            {
                msg << "\n[" << a.get_name() << "]> Requires less than " << _max + 1
                    << " values.";
            }
            if(_checks.test(2))
            {
                msg << "\n[" << a.get_name() << "]> Requires more than " << _min - 1
                    << " values.";
            }
        }
        return arg_result(msg.str());
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename FuncT = std::function<bool(int, int)>>
    arg_result check_count(const std::string& name, const std::string& _do = "111",
                           const FuncT& _func = std::not_equal_to<int>{})
    {
        auto itr = m_name_map.find(name);
        if(itr != m_name_map.end())
            return check_count(m_arguments[static_cast<size_t>(itr->second)], _do, _func);
        return arg_result{};
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ErrorFuncT>
    auto on_error_sfinae(ErrorFuncT&& _func, int)
        -> decltype(_func(std::declval<this_type&>(), std::declval<result_type>()),
                    void())
    {
        m_error_func = std::forward<ErrorFuncT>(_func);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    template <typename ErrorFuncT>
    auto on_error_sfinae(ErrorFuncT&& _func, long)
        -> decltype(_func(std::declval<result_type>()), void())
    {
        auto _wrap_func = [=](this_type&, result_type ret) { _func(ret); };
        m_error_func    = _wrap_func;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    arg_result begin_argument(const std::string& arg, bool longarg, int position);
    arg_result add_value(const std::string& value, int location);
    arg_result end_argument();
    //
    //----------------------------------------------------------------------------------//
    //
private:
    bool                       m_help_enabled   = false;
    int                        m_current        = -1;
    int                        m_width          = 30;
    std::string                m_desc           = {};
    std::string                m_bin            = {};
    error_func_t               m_error_func     = [](this_type&, const result_type&) {};
    std::vector<argument>      m_arguments      = {};
    std::map<int, int>         m_positional_map = {};
    std::map<std::string, int> m_name_map       = {};
    std::vector<action_pair_t> m_actions        = {};
    std::vector<argument>      m_positional_arguments = {};
    std::map<int, std::string> m_positional_values    = {};
};
//
//--------------------------------------------------------------------------------------//
//
template <>
inline std::string
argument_parser::argument::get<std::string>()
{
    using T = std::string;
    if(m_values.empty() && m_default != nullptr &&
       m_default_tidx == std::type_index{ typeid(T) })
        return (*static_cast<T*>(m_default));
    return helpers::join(m_values.begin(), m_values.end());
}
//
//--------------------------------------------------------------------------------------//
//
template <>
inline std::vector<std::string>
argument_parser::argument::get<std::vector<std::string>>()
{
    using T = std::vector<std::string>;
    if(m_values.empty() && m_default != nullptr &&
       m_default_tidx == std::type_index{ typeid(T) })
        return (*static_cast<T*>(m_default));
    return m_values;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace argparse
}  // namespace tim

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/argparse.cpp"
#endif
