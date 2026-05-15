/* Copyright (c) 2019-2024 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef AMD_DBGAPI_LOGGING_H
#define AMD_DBGAPI_LOGGING_H 1

#include "amd-dbgapi.h"
#include "debug.h"
#include "format_printf.h"
#include "utils.h"

#include <cinttypes>
#include <cstdarg>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#define dbgapi_log(level, format, ...)                                        \
  do                                                                          \
    {                                                                         \
      if ((level) <= amd::dbgapi::log_level)                                  \
        amd::dbgapi::detail::log (level, format, ##__VA_ARGS__);              \
    }                                                                         \
  while (0)

#define log_info(...) dbgapi_log (AMD_DBGAPI_LOG_LEVEL_INFO, __VA_ARGS__)
#define log_verbose(...) dbgapi_log (AMD_DBGAPI_LOG_LEVEL_VERBOSE, __VA_ARGS__)

#if defined (__linux__)
#define PROCESS_ID_FORMAT "%d"
#elif defined (_WIN32)
#define PROCESS_ID_FORMAT "%lu"
#endif

namespace amd::dbgapi
{
namespace detail
{

extern size_t log_indent_depth;

extern void log (amd_dbgapi_log_level_t level, const char *format, ...)
  ATTRIBUTE_PRINTF_FORMAT (2, 3);

} /* namespace detail */

extern void vlog (amd_dbgapi_log_level_t level, const char *format,
                  va_list va);

extern amd_dbgapi_log_level_t log_level;

template <typename T>
inline std::string
to_string (T v)
{
  if constexpr (std::is_pointer_v<T>)
    if (v == nullptr)
      return "nullptr";

  std::ostringstream ss;
  ss << v;
  return ss.str ();
}

template <typename T>
inline std::string
to_string (std::optional<T> v)
{
  if (v.has_value ())
    return std::string ("{") + to_string (*v) + "}";
  return "{}";
}

template <>
inline std::string
to_string (const char *v)
{
  return string_printf ("\"%s\"@%p", v, v);
}

template <>
inline std::string
to_string (char *v)
{
  return to_string (const_cast<const char *> (v));
}

template <typename T, typename D>
inline std::string
to_string (const std::unique_ptr<T, D> &v)
{
  return to_string (static_cast<void *> (v.get ()));
}

template <typename T, typename F>
inline std::string
to_string (
  const std::vector<T> &vector, F &&transform = [] (const T &x) { return x; })
{
  std::string str;
  for (auto &&x : vector)
    {
      if (!str.empty ())
        str += ", ";
      str += to_string (transform (x));
    }
  return str;
}

/* See comment for to_cstring (T &&) below.  */
template <typename T, typename F>
inline const char *
to_cstring (
  const std::vector<T> &vector, F &&transform = [] (const T &x) { return x; },
  std::string &&tmp = {})
{
  tmp = to_string (vector, std::forward<F> (transform));
  return tmp.c_str ();
}

namespace detail
{

enum class parameter_kind_t
{
  in,
  out
};

template <typename T, char const *name, parameter_kind_t kind>
struct parameter_t
{
  T m_value;

  explicit constexpr parameter_t (const T &value) : m_value (value) {}

  explicit constexpr parameter_t (T &&value)
    : m_value (std::forward<T> (value))
  {
  }
};

} /* namespace detail */

template <char const *name, detail::parameter_kind_t kind, typename T>
constexpr auto
make_param (T &&value)
{
  return detail::parameter_t<std::decay_t<T>, name, kind> (
    std::forward<T> (value));
}

#define param_in(param)                                                       \
  amd::dbgapi::make_param<STRING_LITERAL (#param),                            \
                          detail::parameter_kind_t::in> (param)

#define param_out(param)                                                      \
  amd::dbgapi::make_param<STRING_LITERAL (#param),                            \
                          detail::parameter_kind_t::out> (param)

template <typename T, char const *name, detail::parameter_kind_t kind>
std::string to_string (detail::parameter_t<T, name, kind> param);

namespace detail
{

template <typename T> struct hex
{
  T m_value;

  explicit constexpr hex (const T &value) : m_value (value) {}

  template <typename U>
  explicit constexpr hex (U &&value) : m_value (std::forward<U> (value))
  {
  }
};

template <typename T> struct query_ref
{
  T m_query;
  const void *m_memory;
};

/* Forward declare the ref struct so that we can define is_ref.  */
template <typename T> struct ref;

/* Check whether T is a ref struct.  */
template <typename T> struct is_ref : std::false_type
{
};

template <typename T> struct is_ref<ref<T>> : std::true_type
{
};

template <typename T> struct is_ref<hex<T>> : is_ref<T>
{
};

template <class T> inline constexpr bool is_ref_v = is_ref<T>::value;

template <typename T> struct ref
{
  T m_reference;
  std::optional<size_t> m_element_count;

  template <typename U = T, std::enable_if_t<is_ref_v<U>, int> = 0>
  explicit constexpr ref (U &&reference,
                          std::optional<size_t> element_count = std::nullopt)
    : m_reference (std::forward<U> (reference)),
      m_element_count (element_count)
  {
  }

  template <typename U,
            std::enable_if_t<std::is_convertible_v<T, U *>, int> = 0>
  explicit constexpr ref (U *reference,
                          std::optional<size_t> element_count = std::nullopt)
    : m_reference (static_cast<T> (reference)), m_element_count (element_count)
  {
  }

  auto *value () const
  {
    if constexpr (is_ref_v<T>)
      return m_reference.value ();
    else
      return m_reference;
  }

  std::optional<size_t> count () const
  {
    if constexpr (is_ref_v<T>)
      return m_reference.count ();
    else
      return m_element_count;
  }

  auto operator* () const
  {
    if constexpr (is_ref_v<T>)
      return ref<decltype (*m_reference)>{ *m_reference, m_element_count };
    else
      return *m_reference;
  }

  ref &operator++ ()
  {
    ++m_reference;
    return *this;
  }

  ref operator++ (int)
  {
    ref old_ref = *this;
    ++m_reference;
    return old_ref;
  }
};

/* void* references are always printed as array of bytes.  */
template <> struct ref<void *> : ref<uint8_t *>
{
  explicit constexpr ref (void *reference,
                          std::optional<size_t> element_count = std::nullopt)
    : ref<uint8_t *> (reference, element_count)
  {
  }
};
template <> struct ref<const void *> : ref<const uint8_t *>
{
  explicit constexpr ref (const void *reference,
                          std::optional<size_t> element_count = std::nullopt)
    : ref<const uint8_t *> (reference, element_count)
  {
  }
};

} /* namespace detail  */

/* Apply the hex modifier to the value of type T.  */
template <typename T>
constexpr auto
make_hex (T &&value)
{
  return detail::hex<std::decay_t<T>> (std::forward<T> (value));
}

template <typename T, char const *name, detail::parameter_kind_t kind>
constexpr auto
make_hex (detail::parameter_t<T, name, kind> &&value)
{
  return make_param<name, kind> (make_hex (std::move (value.m_value)));
}

template <typename T>
constexpr auto
make_query_ref (T query, const void *memory)
{
  return detail::query_ref<T>{ query, memory };
}

template <typename T, typename P, char const *name,
          detail::parameter_kind_t kind>
constexpr auto
make_query_ref (T query, detail::parameter_t<P *, name, kind> memory)
{
  return make_param<name, kind> (make_query_ref (query, memory.m_value));
}

template <typename T>
constexpr auto
make_ref (T *pointer, std::optional<size_t> count = std::nullopt)
{
  return detail::ref<T *> (pointer, count);
}

template <typename T, char const *name, detail::parameter_kind_t kind>
constexpr auto
make_ref (detail::parameter_t<T *, name, kind> pointer,
          std::optional<size_t> count = std::nullopt)
{
  return make_param<name, kind> (make_ref (pointer.m_value, count));
}

template <typename T>
constexpr auto
make_ref (detail::ref<T> &&reference,
          std::optional<size_t> count = std::nullopt)
{
  return detail::ref<detail::ref<T>> (std::move (reference), count);
}

template <typename T, char const *name, detail::parameter_kind_t kind>
constexpr auto
make_ref (detail::parameter_t<detail::ref<T>, name, kind> &&reference,
          std::optional<size_t> count = std::nullopt)
{
  return make_param<name, kind> (
    make_ref (std::move (reference.m_value), count));
}

template <typename T> std::string to_string (detail::hex<detail::ref<T>> hex);

template <typename T>
std::string
to_string (detail::hex<T> v)
{
  std::ostringstream ss;
  ss << "0x" << std::hex << +v.m_value;
  return ss.str ();
}

template <typename T, typename U = decltype (*std::declval<detail::ref<T>> ())>
std::string
to_string (detail::ref<T> ref)
{
  if (!ref.value ())
    return "null";

  std::string str;
  auto *address = ref.value ();

  std::optional<size_t> count = ref.count ();
  for (size_t i = 0; i < count.value_or (1); ++i, ++ref)
    {
      if (i != 0)
        str += ',';

      if (i >= 16)
        {
          str += string_printf ("... <%" PRId64 " more elements>", *count - i);
          break;
        }

      str += to_string (U{ *ref });
    }

  /* Add brackets if we are printing a tuple.  */
  if (count.has_value ())
    str = '[' + str + ']';

  return str + string_printf ("@%p", static_cast<const void *> (address));
}

template <typename T>
std::string
to_string (detail::hex<detail::ref<T>> hex)
{
  using Modifier = detail::hex<decltype (*std::declval<detail::ref<T>> ())>;
  return to_string<T, Modifier> (hex.m_value);
}

template <typename T, char const *name, detail::parameter_kind_t kind>
std::string
to_string (detail::parameter_t<T, name, kind> param)
{
  return std::string (name) + '=' + to_string (param.m_value);
}

template <typename T, char const *name, detail::parameter_kind_t kind,
          typename U = decltype (*std::declval<detail::ref<T>> ())>
std::string
to_string (detail::parameter_t<detail::ref<T>, name, kind> param)
{
  if (kind == detail::parameter_kind_t::out && !param.m_value.value ())
    return {};

  std::string ref_str = to_string<T, U> (param.m_value);

  /* For out parameters, simply print the name and it's value, no need to
     repeat the address.  */
  if (kind == detail::parameter_kind_t::out)
    {
      size_t pos = ref_str.rfind ("@");
      dbgapi_assert (pos != std::string::npos);
      return string_printf ("*%s=", name) + ref_str.substr (0, pos);
    }

  return string_printf ("%s=", name) + ref_str;
}

template <typename T, char const *name, detail::parameter_kind_t kind>
std::string
to_string (detail::parameter_t<detail::hex<detail::ref<T>>, name, kind> param)
{
  using Modifier = detail::hex<decltype (*std::declval<detail::ref<T>> ())>;
  return to_string<T, name, kind, Modifier> (
    make_param<name, kind> (param.m_value.m_value));
}

template <typename T, char const *name, detail::parameter_kind_t kind>
std::string
to_string (detail::parameter_t<detail::query_ref<T>, name, kind> param)
{
  static_assert (kind == detail::parameter_kind_t::out);
  std::string query_ref_str = to_string (param.m_value);

  /* Some queries may not return any information, for example queries to
     return the information for a given instruction kind.  */
  if (query_ref_str.empty ())
    return {};

  size_t pos = query_ref_str.rfind ("@");
  dbgapi_assert (pos != std::string::npos);

  return string_printf ("*%s=", name) + query_ref_str.substr (0, pos);
}

#define AMD_DBGAPI_TYPES_DO(F)                                                \
  F (amd_dbgapi_address_class_id_t)                                           \
  F (amd_dbgapi_address_class_info_t)                                         \
  F (amd_dbgapi_address_class_state_t)                                        \
  F (amd_dbgapi_address_space_access_t)                                       \
  F (amd_dbgapi_address_space_id_t)                                           \
  F (amd_dbgapi_address_space_info_t)                                         \
  F (amd_dbgapi_agent_id_t)                                                   \
  F (amd_dbgapi_agent_info_t)                                                 \
  F (amd_dbgapi_agent_state_t)                                                \
  F (amd_dbgapi_architecture_id_t)                                            \
  F (amd_dbgapi_architecture_info_t)                                          \
  F (amd_dbgapi_breakpoint_action_t)                                          \
  F (amd_dbgapi_breakpoint_id_t)                                              \
  F (amd_dbgapi_breakpoint_info_t)                                            \
  F (amd_dbgapi_changed_t)                                                    \
  F (amd_dbgapi_client_process_info_t)                                        \
  F (amd_dbgapi_code_object_id_t)                                             \
  F (amd_dbgapi_code_object_info_t)                                           \
  F (amd_dbgapi_core_state_data_t)                                            \
  F (amd_dbgapi_direct_call_register_pair_information_t)                      \
  F (amd_dbgapi_dispatch_barrier_t)                                           \
  F (amd_dbgapi_dispatch_fence_scope_t)                                       \
  F (amd_dbgapi_dispatch_id_t)                                                \
  F (amd_dbgapi_dispatch_info_t)                                              \
  F (amd_dbgapi_displaced_stepping_id_t)                                      \
  F (amd_dbgapi_displaced_stepping_info_t)                                    \
  F (amd_dbgapi_event_id_t)                                                   \
  F (amd_dbgapi_event_info_t)                                                 \
  F (amd_dbgapi_event_kind_t)                                                 \
  F (amd_dbgapi_exceptions_t)                                                 \
  F (amd_dbgapi_instruction_kind_t)                                           \
  F (amd_dbgapi_instruction_properties_t)                                     \
  F (amd_dbgapi_log_level_t)                                                  \
  F (amd_dbgapi_memory_precision_t)                                           \
  F (amd_dbgapi_alu_exceptions_precision_t)                                   \
  F (amd_dbgapi_os_queue_type_t)                                              \
  F (amd_dbgapi_process_id_t)                                                 \
  F (amd_dbgapi_process_info_t)                                               \
  F (amd_dbgapi_progress_t)                                                   \
  F (amd_dbgapi_queue_id_t)                                                   \
  F (amd_dbgapi_queue_info_t)                                                 \
  F (amd_dbgapi_queue_state_t)                                                \
  F (amd_dbgapi_register_class_id_t)                                          \
  F (amd_dbgapi_register_class_info_t)                                        \
  F (amd_dbgapi_register_class_state_t)                                       \
  F (amd_dbgapi_register_exists_t)                                            \
  F (amd_dbgapi_register_id_t)                                                \
  F (amd_dbgapi_register_info_t)                                              \
  F (amd_dbgapi_register_properties_t)                                        \
  F (amd_dbgapi_resume_mode_t)                                                \
  F (amd_dbgapi_runtime_state_t)                                              \
  F (amd_dbgapi_segment_address_dependency_t)                                 \
  F (amd_dbgapi_status_t)                                                     \
  F (amd_dbgapi_wave_creation_t)                                              \
  F (amd_dbgapi_wave_id_t)                                                    \
  F (amd_dbgapi_wave_info_t)                                                  \
  F (amd_dbgapi_wave_state_t)                                                 \
  F (amd_dbgapi_wave_stop_reasons_t)                                          \
  F (amd_dbgapi_watchpoint_id_t)                                              \
  F (amd_dbgapi_watchpoint_info_t)                                            \
  F (amd_dbgapi_watchpoint_kind_t)                                            \
  F (amd_dbgapi_watchpoint_share_kind_t)                                      \
  F (amd_dbgapi_workgroup_id_t)                                               \
  F (amd_dbgapi_workgroup_info_t)                                             \
  F (detail::query_ref<amd_dbgapi_agent_info_t>)                              \
  F (detail::query_ref<amd_dbgapi_address_class_info_t>)                      \
  F (detail::query_ref<amd_dbgapi_address_space_info_t>)                      \
  F (detail::query_ref<amd_dbgapi_architecture_info_t>)                       \
  F (detail::query_ref<amd_dbgapi_breakpoint_info_t>)                         \
  F (detail::query_ref<amd_dbgapi_client_process_info_t>)                     \
  F (detail::query_ref<amd_dbgapi_code_object_info_t>)                        \
  F (detail::query_ref<amd_dbgapi_dispatch_info_t>)                           \
  F (detail::query_ref<amd_dbgapi_displaced_stepping_info_t>)                 \
  F (detail::query_ref<amd_dbgapi_event_info_t>)                              \
  F (detail::query_ref<amd_dbgapi_instruction_kind_t>)                        \
  F (detail::query_ref<amd_dbgapi_process_info_t>)                            \
  F (detail::query_ref<amd_dbgapi_queue_info_t>)                              \
  F (detail::query_ref<amd_dbgapi_register_info_t>)                           \
  F (detail::query_ref<amd_dbgapi_register_class_info_t>)                     \
  F (detail::query_ref<amd_dbgapi_watchpoint_info_t>)                         \
  F (detail::query_ref<amd_dbgapi_wave_info_t>)                               \
  F (detail::query_ref<amd_dbgapi_workgroup_info_t>)

#define EXPLICIT_SPECIALIZATION(...)                                          \
  template <> std::string to_string (__VA_ARGS__);
AMD_DBGAPI_TYPES_DO (EXPLICIT_SPECIALIZATION)

#undef EXPLICIT_SPECIALIZATION
#undef AMD_DBGAPI_TYPES_DO

namespace detail
{

inline std::string
to_string_helper ()
{
  return {};
}

template <typename T, typename... Args>
inline std::string
to_string_helper (T &&first, Args &&...args)
{
  std::string str = to_string (std::forward<T> (first));
  if constexpr (sizeof...(Args) != 0)
    {
      std::string args_str = to_string_helper (std::forward<Args> (args)...);
      if (!args_str.empty ())
        {
          if (!str.empty ())
            str += ", ";
          str += args_str;
        }
    }
  return str;
}

} /* namespace detail */

template <typename... Args>
inline std::string
to_string (std::tuple<Args...> &&t)
{
  return std::apply (
    [] (auto &&...ts)
    { return detail::to_string_helper (std::forward<Args> (ts)...); }, t);
}

/* Construct a temporary std::string in the caller's frame to hold the result
   of to_string (T &&) and return a const char * that is equivalent to the
   content of that string.  This is a convenience wrapper used to shorten
   to_string (T &&).c_str ().  */
template <typename T>
inline const char *
to_cstring (T &&value, std::string &&tmp = {})
{
  tmp = to_string (std::forward<T> (value));
  return tmp.c_str ();
}

namespace detail
{

template <typename Functor,
          typename Result = decltype (std::declval<Functor> () ())>
struct tracer_closure
{
  Result m_result;

  tracer_closure (Functor &&f)
    : m_result (std::move (std::forward<Functor> (f) ())){};
  Result &&operator() () && { return std::move (m_result); }

  bool operator== (const Result &rhs) const { return m_result == rhs; }

  std::string str () const { return to_string (m_result); }
};

template <typename Functor> struct tracer_closure<Functor, void>
{
  tracer_closure (Functor &&f) { std::forward<Functor> (f) (); }
  void operator() () && {}

  std::string str () const { return "void"; }
};

} /* namespace detail */

template <amd_dbgapi_log_level_t LogLevel = AMD_DBGAPI_LOG_LEVEL_TRACE>
class tracer
{
private:
  const char *const m_prefix;
  const char *const m_function;
  bool const m_logging_enabled;

public:
  tracer (const char *prefix, const char *function)
    : m_prefix (prefix), m_function (function),
      m_logging_enabled (log_level >= LogLevel)
  {
  }

  template <typename... Args, typename Functor>
  detail::tracer_closure<Functor> enter (std::tuple<Args...> &&in_args,
                                         Functor &&func);

  template <typename... Args, typename Functor>
  decltype (std::declval<Functor> () ())
  leave (std::tuple<Args...> &&out_args,
         detail::tracer_closure<Functor> &&closure);
};

template <amd_dbgapi_log_level_t LogLevel>
template <typename... Args, typename Functor>
detail::tracer_closure<Functor>
tracer<LogLevel>::enter (std::tuple<Args...> &&in_args, Functor &&func)
{
  if (!m_logging_enabled)
    return detail::tracer_closure (std::forward<Functor> (func));

  detail::log (LogLevel, "%s%s (%s) {", m_prefix, m_function,
               to_cstring (std::move (in_args)));
  ++detail::log_indent_depth;

  try
    {
      return detail::tracer_closure (std::forward<Functor> (func));
    }
  catch (...)
    {
      --detail::log_indent_depth;
      dbgapi_log (LogLevel, "%s} throw", m_prefix);
      throw;
    }
}

template <amd_dbgapi_log_level_t LogLevel>
template <typename... Args, typename Functor>
decltype (std::declval<Functor> () ())
tracer<LogLevel>::leave (std::tuple<Args...> &&out_args,
                         detail::tracer_closure<Functor> &&result)
{
  if (m_logging_enabled)
    {
      std::string results_str = result.str ();

      /* Print the outargs unless the return type is amd_dbgapi_status_t and
         the result is not AMD_DBGAPI_STATUS_SUCCESS.  */
      bool print_out_args = true;
      if constexpr (std::is_same_v<decltype (std::declval<Functor> () ()),
                                   amd_dbgapi_status_t>)
        print_out_args = (result == AMD_DBGAPI_STATUS_SUCCESS);

      if (print_out_args && sizeof...(Args) != 0)
        if (auto &&out_args_str = to_string (std::move (out_args));
            !out_args_str.empty ())
          results_str += ", " + out_args_str;

      --detail::log_indent_depth;
      detail::log (LogLevel, "%s} = %s", m_prefix, results_str.c_str ());
    }

  return std::move (result) ();
}

#if defined(WITH_API_TRACING)

#define TRACE_BEGIN_HELPER(level, prefix, ...)                                \
  amd::dbgapi::tracer<level> _tracer (prefix, __FUNCTION__);                  \
  auto&& _closure = _tracer.enter (std::make_tuple (__VA_ARGS__), [&] () {

#define TRACE_END_HELPER(...)                                                 \
  });                                                                         \
  return _tracer.leave (std::make_tuple (__VA_ARGS__), std::move (_closure));

#define TRACE_BEGIN(...)                                                      \
  TRACE_BEGIN_HELPER (AMD_DBGAPI_LOG_LEVEL_TRACE, "", __VA_ARGS__)

#define TRACE_END(...) TRACE_END_HELPER (__VA_ARGS__)

#define TRACE_CALLBACK_BEGIN(...)                                             \
  TRACE_BEGIN_HELPER (AMD_DBGAPI_LOG_LEVEL_VERBOSE, "callback: ", __VA_ARGS__)

#define TRACE_CALLBACK_END(...) TRACE_END_HELPER (__VA_ARGS__)

#else /* !defined (WITH_API_TRACING) */

#define TRACE_BEGIN(...)
#define TRACE_END(...)
#define TRACE_CALLBACK_BEGIN(...)
#define TRACE_CALLBACK_END(...)

#endif /* !defined (WITH_API_TRACING) */

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_LOGGING_H */
