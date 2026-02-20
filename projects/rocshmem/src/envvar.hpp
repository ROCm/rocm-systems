/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_ENVVAR_HPP_
#define LIBRARY_SRC_ENVVAR_HPP_

#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <mutex>
#include <ostream>
#include <source_location>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <sys/socket.h>
#include <unistd.h>

// forward declarations
namespace rocshmem {
namespace envvar {
  namespace _detail {
    template <typename T> class var;
  }  // namespace _detail

  namespace category {
    enum class tag;
  }  // namespace category

  namespace types {
    inline namespace _sf {
      enum class socket_family;
    }  // inline namespace _sf
    inline namespace _debug {
      enum class debug_level;
    }  // inline namespace _debug
  }  // namespace types

  template <typename T, category::tag> class var;

  template <typename... T>
  struct type_sequence {
    using variant = std::variant<T...>;
    using variant_ref = std::variant<std::reference_wrapper<T>...>;
    using variant_cref = std::variant<std::reference_wrapper<const T>...>;
    template <typename U> struct contains : std::disjunction<std::is_same<T, U>...> { };
    template <typename U> static constexpr bool contains_v = contains<U>::value;

    using var_variant = std::variant<_detail::var<T>...>;
    using var_variant_ref = std::variant<std::reference_wrapper<_detail::var<T>>...>;
    using var_variant_cref = std::variant<std::reference_wrapper<const _detail::var<T>>...>;
  };

  // primary template: start with an empty type_sequence<> on the left
  template <typename... T>
  struct unique_type_sequence {
    using type = typename unique_type_sequence<type_sequence<>, T...>::type;
  };

  // convenience alias
  template <typename... T>
  using unique_type_sequence_t = typename unique_type_sequence<T...>::type;

  // base case: the type of a type_sequence is a type_sequence
  template <typename... T>
  struct unique_type_sequence<type_sequence<T...>> {
    using type = type_sequence<T...>;
  };

  // recursion: type_sequence<T...> is already filtered
  //            if type_sequence<T...> contains U, discard U and recurse with remaining V...
  //            else, add U to form type_sequence<T..., U> and recurse with remaining V...
  template <typename... T, typename U, typename... V>
  struct unique_type_sequence<type_sequence<T...>, U, V...> {
    using type = std::conditional_t<type_sequence<T...>::template contains_v<U>,
                                    unique_type_sequence_t<type_sequence<T...>, V...>,
                                    unique_type_sequence_t<type_sequence<T..., U>, V...>>;
  };

  using var_types = unique_type_sequence_t<bool,
                                           uint8_t,
                                           int32_t,
                                           uint32_t,
                                           size_t,
                                           int64_t,
                                           uint64_t,
                                           useconds_t,
                                           std::string,
                                           types::socket_family,
                                           types::debug_level>;
}  // namespace envvar
}  // namespace rocshmem

namespace rocshmem {
namespace envvar {
  namespace category {
    // env var categories
    // when adding a new category, make sure to add prefix<tag::CATEGORY>
    enum class tag {
      ROCSHMEM,
      BOOTSTRAP,
      REVERSE_OFFLOAD,
      GDA,
    };

    // env var string prefixes
    // prevent instantiation of default template; require specializations for each tag
    // if P2041 (see https://wg21.link/P2041) gets merged, can be changed to just = delete instead
    template <tag C> inline constexpr std::enable_if_t<!std::is_enum_v<decltype(C)>> prefix;
    template <> inline constexpr const char* prefix<tag::ROCSHMEM> = "ROCSHMEM";
    template <> inline constexpr const char* prefix<tag::BOOTSTRAP> = "ROCSHMEM_BOOTSTRAP";
    template <> inline constexpr const char* prefix<tag::REVERSE_OFFLOAD> = "ROCSHMEM_RO";
    template <> inline constexpr const char* prefix<tag::GDA> = "ROCSHMEM_GDA";
  }  // namespace category

  namespace parser {
    inline namespace _type_traits {
      // see [basic.fundamental]/p7 for definition of 'narrow character type'
      inline namespace _narrow_character {
        template <typename T> struct _is_narrow_character      : std::false_type { };
        template <> struct _is_narrow_character<char>          : std::true_type  { };
        template <> struct _is_narrow_character<signed char>   : std::true_type  { };
        template <> struct _is_narrow_character<unsigned char> : std::true_type  { };
        template <> struct _is_narrow_character<char8_t>       : std::true_type  { };

        template <typename T>
        struct is_narrow_character
            : std::bool_constant<_is_narrow_character<std::remove_cv_t<T>>::value> { };

        template <typename T>
        inline constexpr bool is_narrow_character_v = is_narrow_character<T>::value;
      }

      // see [basic.fundamental]/p5 for definition of 'standard integer type'
      inline namespace _standard_integer {
        template <typename T> struct _is_standard_integer           : std::false_type { };
        template <> struct _is_standard_integer<  signed char>      : std::true_type  { };
        template <> struct _is_standard_integer<unsigned char>      : std::true_type  { };
        template <> struct _is_standard_integer<  signed short>     : std::true_type  { };
        template <> struct _is_standard_integer<unsigned short>     : std::true_type  { };
        template <> struct _is_standard_integer<  signed int>       : std::true_type  { };
        template <> struct _is_standard_integer<unsigned int>       : std::true_type  { };
        template <> struct _is_standard_integer<  signed long>      : std::true_type  { };
        template <> struct _is_standard_integer<unsigned long>      : std::true_type  { };
        template <> struct _is_standard_integer<  signed long long> : std::true_type  { };
        template <> struct _is_standard_integer<unsigned long long> : std::true_type  { };

        template <typename T>
        struct is_standard_integer
            : std::bool_constant<_is_standard_integer<std::remove_cv_t<T>>::value> { };

        template <typename T>
        inline constexpr bool is_standard_integer_v = is_standard_integer<T>::value;
      }
    }

    // base parser template
    // calls operator>>(std::istream&, T&)
    template <typename T, int = 0, typename = void>
    struct parse {
      std::istream& operator()(std::istream& is, T& value) const {
        return is >> value;
      }
    };

    // integer parser template
    //   * check for negative inputs to unsigned T
    //   * accept requested numeric bases: 0 = detect (default), 8 = octal, 10 = decimal, 16 = hex
    //   * parse {un}signed char as integer instead of character
    template <typename T, int Base>
    struct parse<T, Base, std::enable_if_t<is_standard_integer_v<T>>> {
      std::istream& operator()(std::istream& is, T& value) const {
        // check if input is negative: remove whitespace, then check if first char is '-'
        if constexpr (std::is_unsigned_v<T>) {
          is >> std::ws;
          auto first = is.peek();
          if (first == '-') {
            is.setstate(std::ios_base::failbit);
            return is;
          }
        }

        // accept requested numeric base
        is >> std::setbase(Base);

        // operator>>(std::istream&, {un}signed char&) parses input as a character
        // so signed or unsigned char need to be parsed as a larger type, then narrowed
        if constexpr (is_narrow_character_v<T>) {
          using parsechar_t = std::conditional_t<std::is_signed_v<T>, signed int, unsigned int>;
          parsechar_t parse_value = 0;
          is >> parse_value;
          if (parse_value < std::numeric_limits<T>::min()) {
            value = std::numeric_limits<T>::min();
            is.setstate(std::ios_base::failbit);
          } else if (parse_value > std::numeric_limits<T>::max()) {
            value = std::numeric_limits<T>::max();
            is.setstate(std::ios_base::failbit);
          } else {
            value = static_cast<T>(parse_value);
          }
        } else {
          is >> value;
        }

        return is;
      }
    };

    // string parser specialization, parse entire line
    // operator>>(std::istream&, std::string&) stops on the first whitespace character
    template <> inline
    std::istream& parse<std::string>::operator()(std::istream& is, std::string& value) const {
      std::getline(is, value);
      // std::getline sets failbit when no characters are extracted
      // setting ROCSHMEM_ENVVAR='' can be valid behavior, so clear failbit when this happens
      if (value.empty()) {
        is.clear();
      }
      return is;
    }

    // bool parser specialization, parse both false/true and 0/1
    // to accept true/false, True/False, on/off, On/Off, ON/OFF, 0/1, etc.,
    // can create a facet inheriting from std::num_get<char> and overriding do_get(..., bool& v)
    // then use the locale with that facet: is.imbue(std::locale(is.getloc(), new bool_get{}))
    // note that std::locale is responsible for reference-counting the facets, which is very silly
    // can the locale (and/or facets) have static storage duration?
    template <> inline
    std::istream& parse<bool>::operator()(std::istream& is, bool& value) const {
      auto pos = is.tellg();
      is >> std::boolalpha >> value;
      if (is.fail()) {
        is.clear();
        is.seekg(pos);
        is >> std::noboolalpha >> value;
      }
      return is;
    }

    // decimal integer parser
    template <typename T>
    using parse_decimal = parse<T, 10, std::enable_if_t<is_standard_integer_v<T>>>;

    // hexadecimal integer parser
    template <typename T>
    using parse_hex = parse<T, 16, std::enable_if_t<is_standard_integer_v<T>>>;
  }  // namespace parser

  // namespace for defining custom types, for parsing (mostly enums)
  namespace types {
    // namespace to contain socket_family stuff
    inline namespace _sf {
      enum class socket_family : int {
        UNSPEC = AF_UNSPEC,
        INET = AF_INET,
        INET6 = AF_INET6,
      };
      std::istream& operator>>(std::istream& is, socket_family& family);
      std::ostream& operator<<(std::ostream& os, const socket_family& family);
    }  // inline namespace _sf

    inline namespace _debug {
      enum class debug_level {
        NONE,
        VERSION,
        WARN,
        INFO,
        TRACE,
      };
      std::istream& operator>>(std::istream& is, debug_level& level);
      std::ostream& operator<<(std::ostream& os, const debug_level& level);
    }  // inline namespace _debug
  }  // namespace types

  namespace _detail {
    template <typename T>
    class var {
      static_assert(var_types::contains_v<T>,
                    "T is not in the list of environment variable types");
    public:
      using value_type = T;
      using reference = value_type&;
      using const_reference = const value_type&;

      // primary constructor
      template <typename Parser>
      var(const std::string& _prefix, const std::string& _name, const std::string& _doc,
          const_reference _default_value, Parser parse)
          : name(_prefix + "_" + _name),
            doc(_doc),
            default_value(_default_value),
            value(_default_value),
            value_set(false) {
        const char* env_value = std::getenv(name.c_str());
        if (env_value) {
          std::istringstream iss{std::string(env_value)};
          std::invoke(parse, iss, value);
          if (iss.fail()) {
            std::cerr << std::source_location::current().function_name() << ": invalid argument "
                      << name << "='" << env_value << "'" << std::endl;
            value = default_value;
          } else {
            value_set = true;
          }
        }
      }

      // can't figure out how to do an out-of-line definition for this
      template <typename CharT, typename Traits>
      friend
      std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                                    const var<value_type>& v) {
        return os << v.name << "=" << v.value;
      }

      // public accessors
      const std::string& get_name() const {
        return name;
      }
      const std::string& get_doc() const {
        return doc;
      }
      const_reference get_default() const {
        return default_value;
      }
      const_reference get_value() const {
        return value;
      }
      operator const_reference() const {
        return value;
      }
      bool is_default() const {
        return !value_set;
      }

    private:
      const std::string name;
      const std::string doc;
      const value_type default_value;
      value_type value;
      bool value_set;
    };

    // var_list is a list<variant<var<T>...>> for all valid var types
    // use std::visit for operations on the list elements
    using var_list_t = std::list<var_types::var_variant_cref>;

    // var_map is a map<category, var_list>
    using var_map_t = std::unordered_map<category::tag, var_list_t>;

    // returns a tuple<var_map&, mutex&>, where var_map& and mutex& are statically allocated
    // in particular, the map is allocated so as to fix the static initialization order problem
    // since these are used inside the constructor for envvar::var<T, C> to register variables
    // which are expected to be allocated statically as well
    std::tuple<var_map_t&, std::mutex&> get_var_map();

    // register the var<T, C> with the global variable map
    // map from category C to a list of variables in that category
    // returns a const_iterator to the inserted variable
    // list is heterogeneous over all valid variable types, using variant<_detail::var<T>&...>
    // locks mutex to ensure that there aren't race conditions due to parallel modifications
    template <typename T, category::tag C>
    auto register_variable(const envvar::var<T, C>& v) {
      auto [var_map, map_mutex] = _detail::get_var_map();
      std::lock_guard map_lock(map_mutex);

      // emplace variable to back of list
      // conversion sequence:
      //    const var<T, C>&
      // => const _detail::var<T>&
      // => std::reference_wrapper<const _detail::var<T>>
      // => std::variant<std::reference_wrapper<const _detail::var<T>>...>
      auto& var_list = var_map[C];
      var_list.emplace_back(v);
      // std::list::cend() returns iterator to past-the-end
      // since we just emplaced var to back of list, one before end() will be the back.
      return std::prev(var_list.cend());
    }

    // deregister the variable at the const_iterator pos from the list for category C
    // locks mutex to ensure that there aren't race conditions due to parallel modifications
    template <category::tag C>
    void deregister_variable(_detail::var_list_t::const_iterator pos) {
      auto [var_map, map_mutex] = _detail::get_var_map();
      std::lock_guard map_lock(map_mutex);
      var_map[C].erase(pos);
    }
  }  // namespace _detail

  // class var<Type, Category>
  // reads the specified environment variable using std::getenv()
  // if it set, the variable is parsed (using parser::parse<Type> by default)
  // if it is unset or parsing fails, a default value is used instead
  template <typename T, category::tag C = category::tag::ROCSHMEM>
  class var : public _detail::var<T> {
  public:
    // type aliases aren't inherited, for some reason?
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
    static constexpr category::tag category = C;

    // primary constructor
    // calls _detail::var<T>::var() with the category prefix
    // registers *this with var map and saves the iterator, so it can be deregistered later
    template <typename Parser>
    var(const std::string& name, const std::string& doc,
        const_reference default_value, Parser parse)
        : _detail::var<T>(category::prefix<C>, name, doc, default_value, parse),
          var_map_pos(_detail::register_variable(*this)) { }

    // convenience (delegating) constructors
    //
    // ensure that var(name, doc, default_value) is called instead of var(name, doc, parse)
    // remove the overload from consideration when Parser is not invocable
    template <typename Parser,
              typename = std::enable_if_t<std::is_invocable_v<Parser, std::istream&, reference>>>
    var(const std::string& name, const std::string& doc, Parser parse)
      : var(name, doc, T{}, parse) { }
    var(const std::string& name, const std::string& doc, const_reference default_value)
      : var(name, doc, default_value, parser::parse<T>{}) { }
    var(const std::string& name, const std::string& doc)
      : var(name, doc, T{}, parser::parse<T>{}) { }

    // deregister *this from var map using saved iterator pos
    ~var() {
      _detail::deregister_variable<C>(var_map_pos);
    }

  private:
    _detail::var_list_t::const_iterator var_map_pos;
  };

  inline namespace _base {
    extern const var<bool> uniqueid_with_mpi;
    extern const var<types::debug_level> debug_level;
    extern const var<size_t> heap_size;
    extern const var<size_t> max_num_teams;
    extern const var<std::string> backend;
    extern const var<bool> disable_mixed_ipc;
    extern const var<bool> disable_ipc;

    /**
     * @brief Maximum number of contexts for the application
     */
    extern const var<size_t> max_num_host_contexts;

    /**
     * @brief Maximum number of contexts used in library
     */
    extern const var<size_t> max_num_contexts;

    /**
     * @brief Maximum number of wavefront buffer arrays supported in the default
     * context.
     *
     * This value determines the size of the status flag, rocshmem_g return, and
     * rocshmem atomic return buffers.
     */
    extern const var<size_t> max_wavefront_buffers;

    extern const var<std::string> requested_nic;
    extern const var<std::string> hca_list;
    extern const var<uint32_t> sq_size;
  }  // inline namespace _base

  namespace bootstrap {
    template <typename T> using var = var<T, category::tag::BOOTSTRAP>;
    extern const var<int64_t> timeout;
    extern const var<std::string> hostid;
    extern const var<types::socket_family> socket_family;
    extern const var<std::string> socket_ifname;
  }  // namespace bootstrap

  namespace ro {
    template <typename T> using var = var<T, category::tag::REVERSE_OFFLOAD>;
    extern const var<bool> disable_ipc;
    extern const var<useconds_t> progress_delay;
    extern const var<bool> net_cpu_queue;
  }  // namespace ro

  namespace gda {
    template <typename T> using var = var<T, category::tag::GDA>;
    extern const var<std::string> provider;
    extern const var<bool> alternate_qp_ports;
    extern const var<uint8_t> traffic_class;
    extern const var<bool> pcie_relaxed_ordering;
    extern const var<bool> enable_dmabuf;
  }  // namespace gda
}  // namespace envvar
}  // namespace rocshmem

#endif  // LIBRARY_SRC_ENVVAR_HPP_
