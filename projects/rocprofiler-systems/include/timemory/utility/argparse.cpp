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

#ifndef TIMEMORY_UTILITY_ARGPARSE_CPP_
#define TIMEMORY_UTILITY_ARGPARSE_CPP_

#include "timemory/utility/macros.hpp"

#if !defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/argparse.hpp"
#endif

namespace tim
{
namespace argparse
{
TIMEMORY_UTILITY_INLINE
argument_vector::argument_vector(int& argc, char**& argv)
: base_type()
{
    reserve(argc);
    for(int i = 0; i < argc; ++i)
        push_back(argv[i]);
}

TIMEMORY_UTILITY_INLINE
argument_vector::argument_vector(int& argc, const char**& argv)
: base_type()
{
    reserve(argc);
    for(int i = 0; i < argc; ++i)
        push_back(argv[i]);
}

TIMEMORY_UTILITY_INLINE
argument_vector::argument_vector(int& argc, const char* const*& argv)
{
    reserve(argc);
    for(int i = 0; i < argc; ++i)
        push_back(argv[i]);
}

TIMEMORY_UTILITY_INLINE argument_vector::cargs_t
                        argument_vector::get_execv(const base_type& _prepend, size_t _beg, size_t _end) const
{
    std::stringstream cmdss;
    // find the end if not specified
    _end = std::min<size_t>(size(), _end);
    // determine the number of arguments
    auto _argc = (_end - _beg) + _prepend.size();
    // create the new C argument array, add an extra entry at the end which will
    // always be a null pointer because that is how execv determines the end
    char** _argv = new char*[_argc + 1];

    // ensure all arguments are null pointers initially
    for(size_t i = 0; i < _argc + 1; ++i)
        _argv[i] = nullptr;

    // add the prepend list
    size_t _idx = 0;
    for(const auto& itr : _prepend)
        _argv[_idx++] = helpers::strdup(itr.c_str());

    // copy over the arguments stored internally from the range specified
    for(auto i = _beg; i < _end; ++i)
        _argv[_idx++] = helpers::strdup(this->at(i).c_str());

    // add check that last argument really is a nullptr
    assert(_argv[_argc] == nullptr);

    // create the command string
    for(size_t i = 0; i < _argc; ++i)
        cmdss << " " << _argv[i];
    auto cmd = cmdss.str().substr(1);

    // return a new (int argc, char** argv) and subtract 1 bc nullptr in last entry
    // does not count as argc
    return cargs_t(_argc - 1, _argv, cmd);
}

TIMEMORY_UTILITY_INLINE argument_vector::cargs_t
                        argument_vector::get_execv(size_t _beg, size_t _end) const
{
    return get_execv(base_type{}, _beg, _end);
}

TIMEMORY_UTILITY_INLINE void
argument_parser::print_help(const std::string& _extra)
{
    std::stringstream _usage;
    if(!m_desc.empty())
        _usage << "[" << m_desc << "] ";
    _usage << "Usage: " << m_bin;

    std::cerr << _usage.str();

    std::stringstream _sshort_desc;
    auto              _indent = _usage.str().length() + 2;
    size_t            _ncnt   = 0;
    for(auto& a : m_arguments)
    {
        std::string name = a.m_names.at(0);
        if(name.empty() || name.find_first_of('-') > name.find_first_not_of(" -"))
            continue;
        // select the first long option
        for(size_t n = 1; n < a.m_names.size(); ++n)
        {
            if(name.find("--") == 0)
                break;
            else if(a.m_names.at(n).find("--") == 0)
            {
                name = a.m_names.at(n);
                break;
            }
        }
        if(name.length() > 0)
        {
            if(_ncnt++ > 0)
                _sshort_desc << "\n " << std::setw(_indent) << " " << name;
            else
                _sshort_desc << " " << name;

            _sshort_desc << " (";
            if(a.m_count != argument::Count::ANY)
                _sshort_desc << "count: " << a.m_count;
            else if(a.m_min_count != argument::Count::ANY)
                _sshort_desc << "min: " << a.m_min_count;
            else if(a.m_max_count != argument::Count::ANY)
                _sshort_desc << "max: " << a.m_max_count;
            else
                _sshort_desc << "count: unlimited";
            if(!a.m_dtype.empty())
                _sshort_desc << ", dtype: " << a.m_dtype;
            else if(a.m_count == 0 ||
                    (a.m_count == argument::Count::ANY && a.m_max_count == 1))
                _sshort_desc << ", dtype: bool" << a.m_dtype;
            _sshort_desc << ")";
        }
    }

    std::string _short_desc;
    if(!_sshort_desc.str().empty())
    {
        _short_desc.append("[" + _sshort_desc.str());
        std::stringstream _tmp;
        _tmp << "\n" << std::setw(_indent) << "]";
        _short_desc.append(_tmp.str());
    }

    if(m_positional_arguments.empty())
    {
        std::cerr << " " << _short_desc << " " << _extra << std::endl;
    }
    else
    {
        std::cerr << " " << _short_desc;
        if(!_short_desc.empty())
            std::cerr << "\n" << std::setw(_indent - 2) << " ";
        for(auto& itr : m_positional_arguments)
        {
            std::cerr << " " << helpers::ltrim(itr.m_names.at(0), [](int c) -> bool {
                return c != static_cast<int>('-');
            });
        }

        int current = 0;
        for(auto& v : m_positional_map)
        {
            if(v.first != argument::Position::LastArgument)
            {
                for(; current < v.first; ++current)
                    std::cerr << " [" << current << "]";
                std::cerr << " ["
                          << helpers::ltrim(
                                 m_arguments[static_cast<size_t>(v.second)].m_names.at(0),
                                 [](int c) -> bool { return c != static_cast<int>('-'); })
                          << "]";
            }
            else
            {
                std::cerr << " ... ["
                          << helpers::ltrim(
                                 m_arguments[static_cast<size_t>(v.second)].m_names.at(0),
                                 [](int c) -> bool { return c != static_cast<int>('-'); })
                          << "]";
            }
        }
        std::cerr << " " << _extra << std::endl;
    }

    std::cerr << "\nOptions:" << std::endl;
    for(auto& a : m_arguments)
    {
        std::string name = a.m_names.at(0);
        for(size_t n = 1; n < a.m_names.size(); ++n)
            name.append(", " + a.m_names[n]);
        std::stringstream ss;
        ss << name;
        if(a.m_choices.size() > 0)
        {
            ss << " [";
            auto itr = a.m_choices.begin();
            ss << " " << *itr++;
            for(; itr != a.m_choices.end(); ++itr)
                ss << " | " << *itr;
            ss << " ] ";
        }
        std::stringstream prefix;
        prefix << "    " << std::setw(m_width) << std::left << ss.str();
        std::cerr << std::left << prefix.str();

        auto desc = a.m_desc;
        if(ss.str().length() >= static_cast<size_t>(m_width))
            desc = std::string("\n%{NEWLINE}%") + desc;

        {
            // replace %{INDENT}% with indentation
            const std::string indent_key = "%{INDENT}%";
            const auto        npos       = std::string::npos;
            auto              pos        = npos;
            std::stringstream indent;
            indent << std::setw(prefix.str().length()) << "";
            while((pos = desc.find(indent_key)) != npos)
                desc = desc.replace(pos, indent_key.length(), indent.str());
        }

        {
            // replace %{NEWLINE}% with indentation
            const std::string indent_key = "%{NEWLINE}%";
            const auto        npos       = std::string::npos;
            auto              pos        = npos;
            std::stringstream indent;
            indent << std::setw(m_width + 5) << "";
            while((pos = desc.find(indent_key)) != npos)
                desc = desc.replace(pos, indent_key.length(), indent.str());
        }

        std::cerr << " " << std::setw(m_width) << desc;

        if(a.m_required)
            std::cerr << " (Required)";
        std::cerr << std::endl;
    }
    std::cerr << '\n';
}

TIMEMORY_UTILITY_INLINE argument_parser::arg_result
                        argument_parser::parse_known_args(int* argc, char*** argv, strvec_t& _args,
                                  const std::string& _delim, int verbose_level)
{
    // check for help flag
    auto help_check = [&](int _argc, char** _argv) {
        strset_t help_args = { "-h", "--help", "-?" };
        auto     _help_req = (exists("help") ||
                          (_argc > 1 && help_args.find(_argv[1]) != help_args.end()));
        if(_help_req && !exists("help"))
        {
            for(auto hitr : help_args)
            {
                auto hstr = hitr.substr(hitr.find_first_not_of('-'));
                auto itr  = m_name_map.find(hstr);
                if(itr != m_name_map.end())
                    m_arguments[static_cast<size_t>(itr->second)].m_found = true;
            }
        }
        return _help_req;
    };

    // check for a dash in th command line
    bool _pdash = false;
    for(int i = 1; i < *argc; ++i)
    {
        if((*argv)[i] == std::string("--"))
            _pdash = true;
    }

    // parse the known args and get the remaining argc/argv
    auto _pargs = parse_known_args(*argc, *argv, _args, _delim, verbose_level);
    auto _perrc = std::get<0>(_pargs);
    auto _pargc = std::get<1>(_pargs);
    auto _pargv = std::get<2>(_pargs);

    // check if help was requested before the dash (if dash exists)
    if(help_check((_pdash) ? 0 : _pargc, _pargv))
        return arg_result{ "help requested" };

    // assign the argc and argv
    *argc = _pargc;
    *argv = _pargv;

    return _perrc;
}

TIMEMORY_UTILITY_INLINE argument_parser::known_args_t
                        argument_parser::parse_known_args(int argc, char** argv, strvec_t& _args,
                                  const std::string& _delim, int verbose_level)
{
    int    _cmdc = argc;  // the argc after known args removed
    char** _cmdv = argv;  // the argv after known args removed
    // _cmdv and argv are same pointer unless delimiter is found

    if(argc > 0)
    {
        m_bin = std::string((const char*) argv[0]);
        _args.push_back(std::string((const char*) argv[0]));
    }

    for(int i = 1; i < argc; ++i)
    {
        std::string _arg = argv[i];
        if(_arg == _delim)
        {
            _cmdc        = argc - i;
            _cmdv        = new char*[_cmdc + 1];
            _cmdv[_cmdc] = nullptr;
            _cmdv[0]     = helpers::strdup(argv[0]);
            int k        = 1;
            for(int j = i + 1; j < argc; ++j, ++k)
                _cmdv[k] = helpers::strdup(argv[j]);
            break;
        }
        else
        {
            _args.push_back(std::string((const char*) argv[i]));
        }
    }

    auto cmd_string = [](int _ac, char** _av) {
        std::stringstream ss;
        for(int i = 0; i < _ac; ++i)
            ss << _av[i] << " ";
        return ss.str();
    };

    if((_cmdc > 0 && verbose_level > 0) || verbose_level > 1)
        std::cerr << "\n";

    if(verbose_level > 1)
    {
        std::cerr << "[original]> " << cmd_string(argc, argv) << std::endl;
        std::cerr << "[cfg-args]> ";
        for(auto& itr : _args)
            std::cerr << itr << " ";
        std::cerr << std::endl;
    }

    if(_cmdc > 0 && verbose_level > 0)
        std::cerr << "[command]>  " << cmd_string(_cmdc, _cmdv) << "\n\n";

    return known_args_t{ parse(_args, verbose_level), _cmdc, _cmdv };
}

TIMEMORY_UTILITY_INLINE argument_parser::arg_result
                        argument_parser::parse(const std::vector<std::string>& _args, int verbose_level)
{
    if(verbose_level > 0)
    {
        std::cerr << "[argparse::parse]> parsing '";
        for(const auto& itr : _args)
            std::cerr << itr << " ";
        std::cerr << "'" << '\n';
    }

    for(auto& a : m_arguments)
        a.m_callback(a.m_default);
    for(auto& a : m_positional_arguments)
        a.m_callback(a.m_default);

    using argmap_t = std::map<std::string, argument*>;

    argmap_t   m_arg_map = {};
    arg_result err;
    int        argc = _args.size();
    // the set of options which use a single leading dash but are longer than
    // one character, e.g. -LS ...
    std::set<std::string> long_short_opts;
    if(_args.size() > 1)
    {
        auto is_leading_dash = [](int c) -> bool { return c != static_cast<int>('-'); };
        // build name map
        for(auto& a : m_arguments)
        {
            for(auto& n : a.m_names)
            {
                auto        nleading_dash = helpers::lcount(n, is_leading_dash);
                std::string name          = helpers::ltrim(n, is_leading_dash);
                if(name.empty())
                    continue;
                if(m_name_map.find(name) != m_name_map.end())
                    return arg_result("Duplicate of argument name: " + n);
                m_name_map[name] = a.m_index;
                m_arg_map[name]  = &a;
                if(nleading_dash == 1 && name.length() > 1)
                    long_short_opts.insert(name);
            }
            if(a.m_position >= 0 || a.m_position == argument::Position::LastArgument)
                m_positional_map.at(a.m_position) = a.m_index;
        }

        m_bin = _args.at(0);

        // parse
        std::string current_arg;
        size_t      arg_len;
        for(int argv_index = 1; argv_index < argc; ++argv_index)
        {
            current_arg = _args.at(argv_index);
            arg_len     = current_arg.length();
            if(arg_len == 0)
                continue;
            if(argv_index == argc - 1 &&
               m_positional_map.find(argument::Position::LastArgument) !=
                   m_positional_map.end())
            {
                err          = end_argument();
                arg_result b = err;
                err          = add_value(current_arg, argument::Position::LastArgument);
                if(b)
                    return b;
                // return (m_error_func(*this, b), b);
                if(err)
                    return (m_error_func(*this, err), err);
                continue;
            }

            // count number of leading dashes
            auto nleading_dash = helpers::lcount(current_arg, is_leading_dash);
            // ignores the case if the arg is just a '-'
            // look for -a (short) or --arg (long) args
            bool is_arg = (nleading_dash > 0 && arg_len > 1 && arg_len != nleading_dash)
                              ? true
                              : false;

            if(is_arg && !helpers::is_numeric(current_arg))
            {
                err = end_argument();
                if(err)
                    return (m_error_func(*this, err), err);

                auto name   = current_arg.substr(nleading_dash);
                auto islong = (nleading_dash > 1 || long_short_opts.count(name) > 0);
                err         = begin_argument(name, islong, argv_index);
                if(err)
                    return (m_error_func(*this, err), err);
            }
            else if(current_arg.length() > 0)
            {
                // argument value
                err = add_value(current_arg, argv_index);
                if(err)
                    return (m_error_func(*this, err), err);
            }
        }
    }

    // return the help
    if(m_help_enabled && exists("help"))
        return arg_result("help requested");

    err = end_argument();
    if(err)
        return (m_error_func(*this, err), err);

    // check requirements
    for(auto& a : m_arguments)
    {
        if(a.m_required && !a.m_found)
        {
            return arg_result("Required argument not found: " + a.m_names.at(0));
        }
        if(a.m_position >= 0 && argc >= a.m_position && !a.m_found)
        {
            return arg_result("argument " + a.m_names.at(0) + " expected in position " +
                              std::to_string(a.m_position));
        }
    }

    // check requirements
    for(auto& a : m_positional_arguments)
    {
        if(a.m_required && !a.m_found)
            return arg_result("Required argument not found: " + a.m_names.at(0));
    }

    // check all the counts have been satisfied
    for(auto& a : m_arguments)
    {
        if(a.m_found && a.m_default == nullptr)
        {
            auto cnt_err = check_count(a);
            if(cnt_err)
                return cnt_err;
        }
    }

    // execute the global actions
    for(auto& itr : m_actions)
    {
        if(itr.first(*this))
            itr.second(*this);
    }

    // execute the argument-specific actions
    for(auto& itr : m_arg_map)
    {
        if(exists(itr.first))
            itr.second->execute_actions(*this);
    }

    return arg_result{};
}

TIMEMORY_UTILITY_INLINE argument_parser::arg_result
                        argument_parser::begin_argument(const std::string& arg, bool longarg, int position)
{
    auto it = m_positional_map.find(position);
    if(it != m_positional_map.end())
    {
        arg_result err = end_argument();
        argument&  a   = m_arguments[static_cast<size_t>(it->second)];
        a.m_values.push_back(arg);
        a.m_found = true;
        return err;
    }
    if(m_current != -1)
    {
        return arg_result("Current argument left open");
    }
    size_t      name_end = helpers::find_punct(arg);
    std::string arg_name = arg.substr(0, name_end);
    if(longarg)
    {
        int  equal_pos = helpers::find_equiv(arg);
        auto nmf       = m_name_map.find(arg_name);
        if(nmf == m_name_map.end())
        {
            arg_name = arg.substr(0, equal_pos);
            nmf      = m_name_map.find(arg_name);
        }
        if(nmf == m_name_map.end())
        {
            return arg_result("Unrecognized command line option '" + arg_name + "'");
        }
        m_current                                             = nmf->second;
        m_arguments[static_cast<size_t>(nmf->second)].m_found = true;
        if(equal_pos == 0 || (equal_pos < 0 && arg_name.length() < arg.length()))
        {
            // malformed argument
            return arg_result("Malformed argument: " + arg);
        }
        else if(equal_pos > 0)
        {
            std::string arg_value = arg.substr(name_end + 1);
            add_value(arg_value, position);
        }
    }
    else
    {
        arg_result r;
        if(arg_name.length() == 1)
        {
            return begin_argument(arg, true, position);
        }
        else
        {
            for(char& c : arg_name)
            {
                r = begin_argument(std::string(1, c), true, position);
                if(r)
                {
                    return r;
                }
                r = end_argument();
                if(r)
                {
                    return r;
                }
            }
        }
    }
    return arg_result{};
}

TIMEMORY_UTILITY_INLINE argument_parser::arg_result
                        argument_parser::add_value(const std::string& value, int location)
{
    auto unnamed = [&]() {
        auto itr = m_positional_map.find(location);
        if(itr != m_positional_map.end())
        {
            argument& a = m_arguments[static_cast<size_t>(itr->second)];
            a.m_values.push_back(value);
            a.m_found = true;
        }
        else
        {
            auto idx = m_positional_values.size();
            m_positional_values.emplace(idx, value);
            if(idx < m_positional_arguments.size())
            {
                auto& a   = m_positional_arguments.at(idx);
                a.m_found = true;
                auto err  = a.check_choice(value);
                if(err)
                    return err;
                a.m_values.push_back(value);
                a.execute_actions(*this);
            }
        }
        return arg_result{};
    };

    if(m_current >= 0)
    {
        arg_result err;
        size_t     c = static_cast<size_t>(m_current);
        consume_parameters(c);
        argument& a = m_arguments[static_cast<size_t>(m_current)];

        err = a.check_choice(value);
        if(err)
            return err;

        auto num_values = [&]() { return static_cast<int>(a.m_values.size()); };

        // check {m_count, m_max_count} > COUNT::ANY && m_values.size() >= {value}
        if((a.m_count >= 0 && num_values() >= a.m_count) ||
           (a.m_max_count >= 0 && num_values() >= a.m_max_count))
        {
            err = end_argument();
            if(err)
                return err;
            return unnamed();
        }

        a.m_values.push_back(value);

        // check {m_count, m_max_count} > COUNT::ANY && m_values.size() >= {value}
        if((a.m_count >= 0 && num_values() >= a.m_count) ||
           (a.m_max_count >= 0 && num_values() >= a.m_max_count))
        {
            err = end_argument();
            if(err)
                return err;
        }
        return arg_result{};
    }

    return unnamed();
}

TIMEMORY_UTILITY_INLINE argument_parser::arg_result
                        argument_parser::end_argument()
{
    if(m_current >= 0)
    {
        argument& a = m_arguments[static_cast<size_t>(m_current)];
        m_current   = -1;
        if(static_cast<int>(a.m_values.size()) < a.m_count)
            return arg_result("Too few arguments given for " + a.m_names.at(0));
        if(a.m_max_count >= 0)
        {
            if(static_cast<int>(a.m_values.size()) > a.m_max_count)
                return arg_result("Too many arguments given for " + a.m_names.at(0));
        }
        else if(a.m_count >= 0)
        {
            if(static_cast<int>(a.m_values.size()) > a.m_count)
                return arg_result("Too many arguments given for " + a.m_names.at(0));
        }
    }
    return arg_result{};
}

TIMEMORY_UTILITY_INLINE std::ostream&
                        operator<<(std::ostream& os, const argument_parser::arg_result& r)
{
    os << r.what();
    return os;
}

}  // namespace argparse
}  // namespace tim

#endif
