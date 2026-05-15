/* Copyright (c) 2019-2025 Advanced Micro Devices, Inc.

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

#ifndef AMD_DBGAPI_UTILS_H
#define AMD_DBGAPI_UTILS_H 1

#include "amd-dbgapi.h"
#include "debug.h"
#include "exception.h"
#include "format_printf.h"

#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define CONCAT_NX(x, y) x##y
#define CONCAT(x, y) CONCAT_NX (x, y)

#define DELIM_COMMA() ,
#define DELIM_SEMICOLON() ;

#define NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define ARGS_COUNT(...) NTH_ARG (__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define FOR_EACH_1(FUNCTION, DELIM, ARG, ...) FUNCTION (ARG)
#define FOR_EACH_2(FUNCTION, DELIM, ARG, ...)                                 \
  FUNCTION (ARG) DELIM () FOR_EACH_1 (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH_3(FUNCTION, DELIM, ARG, ...)                                 \
  FUNCTION (ARG) DELIM () FOR_EACH_2 (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH_4(FUNCTION, DELIM, ARG, ...)                                 \
  FUNCTION (ARG) DELIM () FOR_EACH_3 (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH_5(FUNCTION, DELIM, ARG, ...)                                 \
  FUNCTION (ARG) DELIM () FOR_EACH_4 (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH_6(FUNCTION, DELIM, ARG, ...)                                 \
  FUNCTION (ARG) DELIM () FOR_EACH_5 (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH_7(FUNCTION, DELIM, ARG, ...)                                 \
  FUNCTION (ARG) DELIM () FOR_EACH_6 (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH_8(FUNCTION, DELIM, ARG, ...)                                 \
  FUNCTION (ARG) DELIM () FOR_EACH_7 (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH_9(FUNCTION, DELIM, ARG, ...)                                 \
  FUNCTION (ARG) DELIM () FOR_EACH_8 (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH_10(FUNCTION, DELIM, ARG, ...)                                \
  FUNCTION (ARG) DELIM () FOR_EACH_9 (FUNCTION, DELIM, __VA_ARGS__)

#define FOR_EACH_N(FUNCTION, DELIM, N, ...)                                   \
  CONCAT (FOR_EACH_, N) (FUNCTION, DELIM, __VA_ARGS__)
#define FOR_EACH(FUNCTION, DELIM, ...)                                        \
  FOR_EACH_N (FUNCTION, DELIM, ARGS_COUNT (__VA_ARGS__), __VA_ARGS__)

namespace amd::dbgapi
{

class instruction_t;
class process_t;

using epoch_t = uint64_t;
using file_desc_t = int;

extern std::string string_vprintf (const char *format, va_list va);
extern std::string string_printf (const char *format, ...)
  ATTRIBUTE_PRINTF_FORMAT (1, 2);

namespace utils
{

/* Get the filename containing dbgapi's code.  */
const char *get_self_name ();

template <typename Integral = uint64_t>
constexpr Integral
bit_mask (int first, int last)
{
  dbgapi_assert (last >= first && "invalid argument");
  size_t num_bits = last - first + 1;
  return ((num_bits >= sizeof (Integral) * 8)
            ? ~Integral{ 0 } /* num_bits exceed the size of Integral */
            : ((Integral{ 1 } << num_bits) - 1))
         << first;
}

template <typename Integral>
constexpr int
bit_count (Integral x)
{
  /* Count the number of one bits using sideways additions. Works for integral
     types up to 128-bit wide.  */
  static_assert (sizeof (Integral) <= 16, "Integral type is too wide");
  using T = std::conditional_t<sizeof (Integral) >= sizeof (uint32_t),
                               Integral, uint32_t>;

  x = x - ((x >> 1) & ~T{ 0 } / 3);
  x = (x & ~T{ 0 } / 15 * 3) + ((x >> 2) & ~T{ 0 } / 15 * 3);
  x = (x + (x >> 4)) & ~T{ 0 } / 255 * 15;
  return (x * (~T{ 0 } / 255)) >> (sizeof (T) - 1) * 8;
}

template <typename Integral>
constexpr int
trailing_zeroes_count (Integral x)
{
  std::make_unsigned_t<Integral> v{ x };

  if (!v)
    return sizeof (v) * 8;

  /* Set v's trailing 0s to 1s, and zero rest.  */
  v = (v ^ (v - 1)) >> 1;

  return bit_count (v);
}

#if defined __has_builtin && __has_builtin(__builtin_ctzll)
template <>
constexpr int
trailing_zeroes_count<unsigned long long> (unsigned long long x)
{
  return __builtin_ctzll (x);
}
#endif

#if defined __has_builtin && __has_builtin(__builtin_ctzl)
template <>
constexpr int
trailing_zeroes_count<unsigned long> (unsigned long x)
{
  return __builtin_ctzl (x);
}
#endif

#if defined __has_builtin && __has_builtin(__builtin_ctz)
template <>
constexpr int
trailing_zeroes_count<unsigned int> (unsigned int x)
{
  return __builtin_ctz (x);
}
#endif

template <typename Integral>
constexpr std::make_unsigned_t<Integral>
zero_extend (Integral x, int width)
{
  return x & bit_mask (0, width - 1);
}

template <typename Integral>
constexpr std::make_signed_t<Integral>
sign_extend (Integral x, int width)
{
  Integral sign_mask = bit_mask (width - 1, sizeof (Integral) * 8 - 1);
  return (zero_extend (x, width) ^ sign_mask) - sign_mask;
}

/* Extract bits [last:first] from t.  */
template <typename Integral>
constexpr Integral
bit_extract (Integral x, int first, int last)
{
  return (x >> first) & bit_mask<Integral> (0, last - first);
}

template <typename Integral>
constexpr bool
is_power_of_two (Integral x)
{
  return x && (x & (x - 1)) == 0;
}

template <typename Integral>
constexpr std::make_unsigned_t<Integral>
next_power_of_two (Integral x)
{
  std::make_unsigned_t<Integral> v{ x };

  /* Edge case where v equals 0 should return 1.  */
  v += (v == 0);

  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  if (sizeof (v) > 1)
    v |= v >> 8;
  if (sizeof (v) > 2)
    v |= v >> 16;
  if (sizeof (v) > 4)
    v |= v >> 32;

  return ++v;
}

template <typename Integral>
constexpr Integral
align_down (Integral x, int alignment)
{
  dbgapi_assert (is_power_of_two (alignment));
  return x & -alignment;
}

template <typename Integral>
constexpr Integral
align_up (Integral x, int alignment)
{
  dbgapi_assert (is_power_of_two (alignment));
  return (x + alignment - 1) & -alignment;
}

template <typename Integral>
constexpr bool
is_aligned (Integral x, int alignment)
{
  dbgapi_assert (is_power_of_two (alignment));
  return x == align_down (x, alignment);
}

namespace detail
{

class not_copyable_t
{
protected:
  constexpr not_copyable_t () = default;
  ~not_copyable_t () = default;

  not_copyable_t (const not_copyable_t &) = delete;
  not_copyable_t &operator= (const not_copyable_t &) = delete;
};

} /* namespace detail */

using not_copyable_t = detail::not_copyable_t;

template <typename Functor,
          std::enable_if_t<std::is_invocable_v<Functor>, int> = 0>
class scope_exit_t : private not_copyable_t
{
private:
  Functor m_functor{};
  bool m_is_released{ false };

public:
  template <typename Other,
            std::enable_if_t<std::is_constructible_v<Functor, Other>, int> = 0>
  explicit scope_exit_t (Other &&other)
  try : m_functor (std::forward<Other> (other))
    {
    }
  catch (...)
    {
      other ();
    } /* there is an implicit "throw;" here.  */

  scope_exit_t (scope_exit_t &&other)
  {
    if (!other.m_is_released)
      {
        m_functor = std::move (other.m_functor);
        other.release ();
      }
  }

  scope_exit_t (const scope_exit_t &) = delete;

  ~scope_exit_t ()
  {
    if (!m_is_released)
      {
        m_is_released = true;
        m_functor ();
      }
  }

  void release () { m_is_released = true; }
};

template <typename Functor>
auto
make_scope_exit (Functor &&func)
{
  return scope_exit_t<std::decay_t<Functor>> (std::forward<Functor> (func));
}

template <typename Functor,
          std::enable_if_t<std::is_invocable_v<Functor>, int> = 0>
class scope_success_t
{
private:
  Functor m_functor{};
  bool m_is_released{ false };
  int m_uncaught_exceptions{ 0 };

public:
  template <typename Other,
            std::enable_if_t<std::is_constructible_v<Functor, Other>, int> = 0>
  explicit scope_success_t (Other &&other)
    : m_functor (std::forward<Other> (other)),
      m_uncaught_exceptions (std::uncaught_exceptions ())
  {
  }

  scope_success_t (scope_success_t &&other)
    : m_uncaught_exceptions (other.m_uncaught_exceptions)
  {
    if (!other.m_is_released)
      {
        m_functor = std::move (other.m_functor);
        other.release ();
      }
  }

  scope_success_t (const scope_success_t &) = delete;

  ~scope_success_t ()
  {
    if (!m_is_released && std::uncaught_exceptions () <= m_uncaught_exceptions)
      {
        m_is_released = true;
        m_functor ();
      }
  }

  void release () { m_is_released = true; }
};

template <typename Functor>
auto
make_scope_success (Functor &&func)
{
  return scope_success_t<std::decay_t<Functor>> (std::forward<Functor> (func));
}

template <typename Functor,
          std::enable_if_t<std::is_invocable_v<Functor>, int> = 0>
class scope_fail_t
{
private:
  Functor m_functor{};
  bool m_is_released{ false };
  int m_uncaught_exceptions{ 0 };

public:
  template <typename Other,
            std::enable_if_t<std::is_constructible_v<Functor, Other>, int> = 0>
  explicit scope_fail_t (Other &&other)
  try : m_functor (std::forward<Other> (other)),
    m_uncaught_exceptions (std::uncaught_exceptions ())
    {
    }
  catch (...)
    {
      other ();
    } /* there is an implicit "throw;" here.  */

  scope_fail_t (scope_fail_t &&other)
    : m_uncaught_exceptions (other.m_uncaught_exceptions)
  {
    if (!other.m_is_released)
      {
        m_functor = std::move (other.m_functor);
        other.release ();
      }
  }

  scope_fail_t (const scope_fail_t &) = delete;

  ~scope_fail_t ()
  {
    if (!m_is_released && std::uncaught_exceptions () > m_uncaught_exceptions)
      {
        m_is_released = true;
        m_functor ();
      }
  }

  void release () { m_is_released = true; }
};

template <typename Functor>
auto
make_scope_fail (Functor &&func)
{
  return scope_fail_t<std::decay_t<Functor>> (std::forward<Functor> (func));
}

/* A RAII wrapper for resources that are managed through a handle and disposed
   of with the provided deleter.  */
template <typename R /* Resource  */, typename D /* Deleter  */>
class unique_resource_t
{
private:
  /* The type of the resource stored in the wrapper.  */
  using RS
    = std::conditional_t<std::is_reference_v<R>,
                         std::reference_wrapper<std::remove_reference_t<R>>,
                         R>;

  RS m_resource;
  D m_deleter;
  bool m_execute_on_reset;

public:
  /* Constructors  */
  template <typename RRS = RS, typename DD = D,
            std::enable_if_t<std::is_default_constructible_v<RRS>
                               && std::is_default_constructible_v<DD>,
                             int>
            = 0>
  unique_resource_t ()
    : m_resource (), m_deleter (), m_execute_on_reset (false)
  {
  }

  template <
    typename RR, typename DD,
    std::enable_if_t<
      std::is_constructible_v<RS, RR> && std::is_constructible_v<D, DD>, int>
    = 0>
  unique_resource_t (RR &&resource, DD &&deleter) noexcept
    : m_resource (std::forward<RR> (resource)),
      m_deleter (std::forward<DD> (deleter)), m_execute_on_reset (true)
  {
  }

  unique_resource_t (unique_resource_t &&other) noexcept
    : m_resource (std::move (other.m_resource)),
      m_deleter (std::move (other.m_deleter)),
      m_execute_on_reset (other.m_execute_on_reset)
  {
    other.release ();
  }

  /* Destructor  */
  ~unique_resource_t () { reset (); }

  /* Assignment  */
  unique_resource_t &operator= (unique_resource_t &&other) noexcept
  {
    reset ();
    m_resource = std::move (other.m_resource);
    m_deleter = std::move (other.m_deleter);
    m_execute_on_reset = other.m_execute_on_reset;
    other.release ();
    return *this;
  }

  /* Modifiers:  */
  void reset ()
  {
    if (m_execute_on_reset)
      {
        m_execute_on_reset = false;
        get_deleter () (m_resource);
      }
  }

  template <class RR> void reset (RR &&resource)
  {
    reset ();
    m_resource = std::forward<RR> (resource);
    m_execute_on_reset = true;
  }

  void release () { m_execute_on_reset = false; }

  /* Accessors:  */
  const R &get () const { return m_resource; }

  template <
    typename RR = R,
    std::enable_if_t<
      std::is_pointer_v<RR> && !std::is_void_v<std::remove_pointer_t<RR>>, int>
    = 0>
  std::add_lvalue_reference_t<std::remove_pointer_t<R>> operator* () const
  {
    return *get ();
  }

  template <typename RR = R, std::enable_if_t<std::is_pointer_v<RR>, int> = 0>
  R operator->() const
  {
    return get ();
  }

  const D &get_deleter () const { return m_deleter; }
};

namespace detail
{

template <typename T, typename Tag>
struct doubly_linked_entry_t : private not_copyable_t
{
  doubly_linked_entry_t<T, Tag> *m_prev{ nullptr };
  doubly_linked_entry_t<T, Tag> *m_next{ nullptr };

  doubly_linked_entry_t (doubly_linked_entry_t &&rhs) noexcept
  {
    m_next = rhs.m_next;
    m_prev = rhs.m_prev;
    m_next->m_prev = this;
    m_prev->m_next = this;
    rhs.m_next = rhs.m_prev = nullptr;
  }
  doubly_linked_entry_t &operator= (doubly_linked_entry_t &&rhs) noexcept
  {
    m_next = rhs.m_next;
    m_prev = rhs.m_prev;
    m_next->m_prev = this;
    m_prev->m_next = this;
    rhs.m_next = rhs.m_prev = nullptr;
  }

  doubly_linked_entry_t () = default;
  bool is_inserted () const { return m_prev != nullptr; }
};

} /* namespace detail */

template <typename T, typename Tag = void>
class doubly_linked_list_t : private not_copyable_t
{
public:
  /* T must be derived from entry_type.  */
  using entry_type = detail::doubly_linked_entry_t<T, Tag>;

  struct iterator
  {
  private:
    entry_type *m_current;

  public:
    using self_type = iterator;
    using value_type = T;
    using difference_type = size_t;
    using reference = T &;
    using pointer = T *;
    using iterator_category = std::bidirectional_iterator_tag;

    iterator (entry_type *current) : m_current (current) {}
    self_type &operator++ ()
    {
      m_current = m_current->m_next;
      return *this;
    }
    self_type operator++ (int)
    {
      self_type i = *this;
      m_current = m_current->m_next;
      return i;
    }
    self_type &operator-- ()
    {
      m_current = m_current->m_prev;
      return *this;
    }
    self_type operator-- (int)
    {
      self_type i = *this;
      m_current = m_current->m_prev;
      return i;
    }
    reference operator* () { return static_cast<reference> (*m_current); }
    pointer operator->() { return static_cast<pointer> (m_current); }
    bool operator== (const self_type &rhs)
    {
      return m_current == rhs.m_current;
    }
    bool operator!= (const self_type &rhs)
    {
      return m_current != rhs.m_current;
    }
  };

private:
  size_t m_element_count{ 0 };
  entry_type m_head{};

public:
  doubly_linked_list_t ()
  {
    m_head.m_prev = &m_head;
    m_head.m_next = &m_head;
  }
  doubly_linked_list_t (doubly_linked_list_t &&rhs) noexcept
  {
    if (rhs.empty ())
      {
        m_head.m_prev = &m_head;
        m_head.m_next = &m_head;
      }
    else
      {
        m_head = rhs.m_head;
        m_head.m_next->m_prev = &m_head;
        m_head.m_prev->m_next = &m_head;
        m_element_count = rhs.m_element_count;
      }
  }

  doubly_linked_list_t &operator= (doubly_linked_list_t &&rhs) noexcept
  {
    if (rhs.empty ())
      {
        m_head.m_prev = &m_head;
        m_head.m_next = &m_head;
      }
    else
      {
        m_head = rhs.m_head;
        m_head.m_next->m_prev = &m_head;
        m_head.m_prev->m_next = &m_head;
        m_element_count = rhs.m_element_count;
      }
  }

  size_t size () const { return m_element_count; }
  bool empty () const { return size () == 0; }

  iterator begin () { return iterator{ m_head.m_next }; }
  iterator end () { return iterator{ &m_head }; }

  iterator insert (T &value)
  {
    entry_type &entry = static_cast<entry_type &> (value);
    dbgapi_assert (!entry.is_inserted () && "already inserted");

    entry.m_next = m_head.m_next;
    entry.m_prev = &m_head;
    m_head.m_next->m_prev = &entry;
    m_head.m_next = &entry;

    ++m_element_count;
    return iterator{ &entry };
  }

  iterator remove (T &value)
  {
    entry_type &entry = static_cast<entry_type &> (value);
    dbgapi_assert (entry.is_inserted () && "not inserted");

    auto next = entry.m_next;
    entry.m_prev->m_next = next;
    next->m_prev = entry.m_prev;
    entry.m_next = entry.m_prev = nullptr;

    --m_element_count;
    return iterator{ next };
  }

  void clear ()
  {
    for (auto it = begin (); it != end ();)
      it = remove (*it);
    dbgapi_assert (m_head.m_prev == &m_head && m_head.m_next == &m_head);
  }
};

namespace detail
{

template <typename Functor, typename Return, typename First, typename... Rest>
First first_argument_of_helper_t (Return (Functor::*) (First, Rest...));

template <typename Functor, typename Return, typename First, typename... Rest>
First first_argument_of_helper_t (Return (Functor::*) (First, Rest...) const);

} /* namespace detail */

/* Holds the type of the Functor's first argument.  */
template <typename Functor>
using first_argument_of_t
  = decltype (detail::first_argument_of_helper_t (&Functor::operator()));

namespace detail
{

template <typename AlwaysVoid, template <typename...> class Op,
          typename... Args>
struct detector : std::false_type
{
  static_assert (std::is_same_v<AlwaysVoid, void>, "must be void");
  struct type
  {
    type () = delete;
    ~type () = delete;
  };
};

template <template <typename...> class Op, typename... Args>
struct detector<std::void_t<Op<Args...>>, Op, Args...> : std::true_type
{
  using type = Op<Args...>;
};

} /* namespace detail */

template <template <typename...> class Op, typename... Args>
using is_detected = detail::detector<void, Op, Args...>;

template <template <typename...> class Op, typename... Args>
inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

template <template <typename...> class Op, typename... Args>
using detected_t = typename is_detected<Op, Args...>::type;

/* Check the size, and copy `value' into the memory pointed to by `ret'.  */
template <typename T>
void
get_info (size_t value_size, void *ret, const T &value)
{
  if (!ret)
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  if (value_size != sizeof (T))
    throw api_error_t (AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);

  memcpy (ret, &value, sizeof (T));
}

template <>
void get_info (size_t value_size, void *ret, const std::string &value);

template <>
void get_info (size_t value_size, void *ret, const instruction_t &value);

template <typename T>
void get_info (size_t value_size, void *ret, const std::vector<T> &value);

template <typename Object>
std::pair<typename Object::handle_type * /* objects */, size_t /* count */>
get_handle_list (const std::vector<process_t *> &processes,
                 amd_dbgapi_changed_t *changed);

template <char... Chars> struct string_literal
{
  using type = string_literal;
  static constexpr char value[sizeof...(Chars) + 1] = { Chars..., 0 };
};

namespace detail
{

template <class T, class U> struct concat;

/* Concatenate 2 string literals.  */
template <char... Ts, char... Us>
struct concat<string_literal<Ts...>, string_literal<Us...>>
  : string_literal<Ts..., Us...>
{
};

template <size_t N, char... Cs> struct make_string_literal;

/* string_literal terminator.  */
template <char... Cs>
struct make_string_literal<0, '\0', Cs...> : string_literal<>
{
};

/* Build a string literal one character at a time. Note, this could be made
   more efficient using power of twos if it becomes an issue.  */
template <size_t N, char C0, char... Cs>
struct make_string_literal<N, C0, Cs...>
  : concat<string_literal<C0>,
           typename make_string_literal<N - 1, Cs...>::type>
{
};

static constexpr size_t MAX_STRING_LITERAL_LENGTH = 32;

template <size_t N>
constexpr char
string_literal_at_or (const char (&str)[N], size_t n, char value)
{
  static_assert (N <= MAX_STRING_LITERAL_LENGTH, "increase tokenizer size");
  return (n < N - 1) ? str[n] : value;
}

} /* namespace detail */

#define TOKENIZE_STRING_LITERAL(s)                                            \
  utils::detail::string_literal_at_or ((s), 0, '\0'),                         \
    utils::detail::string_literal_at_or ((s), 1, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 2, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 3, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 4, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 5, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 6, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 7, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 8, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 9, '\0'),                       \
    utils::detail::string_literal_at_or ((s), 10, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 11, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 12, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 13, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 14, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 15, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 16, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 17, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 18, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 19, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 20, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 21, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 22, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 23, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 24, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 25, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 26, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 27, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 28, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 29, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 30, '\0'),                      \
    utils::detail::string_literal_at_or ((s), 31, '\0')

#define STRING_LITERAL(s)                                                     \
  utils::detail::make_string_literal<sizeof (s) - 1,                          \
                                     TOKENIZE_STRING_LITERAL (s)>::value

constexpr size_t KiB = 1024;
constexpr size_t MiB = KiB * KiB;
constexpr size_t GiB = KiB * KiB * KiB;

extern std::string human_readable_size (size_t size);

template <typename T> struct type_identity
{
  using type = T;
};

} /* namespace utils */

template <typename T> struct is_flag : std::false_type
{
};

/* To enable bitwise operators on enum classes use:
   template <> struct is_flag<my_enum_class> : std::true_type {};
*/

template <typename T> inline constexpr bool is_flag_v = is_flag<T>::value;

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr bool
operator!(T flag)
{
  using t = std::underlying_type_t<T>;
  return static_cast<t> (flag) == t{};
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr bool
operator== (T flag, std::underlying_type_t<T> value)
{
  return static_cast<std::underlying_type_t<T>> (flag) == value;
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr bool
operator!= (T flag, std::underlying_type_t<T> value)
{
  return !(flag == value);
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T
operator~(T flag)
{
  using t = std::underlying_type_t<T>;
  return static_cast<T> (~static_cast<t> (flag));
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T
operator| (T lhs, T rhs)
{
  using t = std::underlying_type_t<T>;
  return static_cast<T> (static_cast<t> (lhs) | static_cast<t> (rhs));
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T
operator<< (T flag, size_t shift)
{
  using t = std::underlying_type_t<T>;
  return static_cast<T> (static_cast<t> (flag) << shift);
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T
operator>> (T flag, size_t shift)
{
  using t = std::underlying_type_t<T>;
  return static_cast<T> (static_cast<t> (flag) >> shift);
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T &
operator|= (T &lhs, T rhs)
{
  return lhs = lhs | rhs;
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T
operator& (T lhs, T rhs)
{
  using t = std::underlying_type_t<T>;
  return static_cast<T> (static_cast<t> (lhs) & static_cast<t> (rhs));
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T &
operator&= (T &lhs, T rhs)
{
  return lhs = lhs & rhs;
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T
operator^ (T lhs, T rhs)
{
  using t = std::underlying_type_t<T>;
  return static_cast<T> (static_cast<t> (lhs) ^ static_cast<t> (rhs));
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T &
operator^= (T &lhs, T rhs)
{
  return lhs = lhs ^ rhs;
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T
operator- (T lhs, int rhs)
{
  using t = std::underlying_type_t<T>;
  return static_cast<T> (static_cast<t> (lhs) - static_cast<t> (rhs));
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T &
operator-= (T &lhs, int rhs)
{
  return lhs = lhs - rhs;
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T
operator+ (T lhs, int rhs)
{
  using t = std::underlying_type_t<T>;
  return static_cast<T> (static_cast<t> (lhs) + static_cast<t> (rhs));
}

template <typename T, std::enable_if_t<is_flag_v<T>, int> = 0>
constexpr T &
operator+= (T &lhs, int rhs)
{
  return lhs = lhs + rhs;
}

/* Enable bitwise operations for amd_dbgapi_exceptions_t.  */
template <> struct is_flag<amd_dbgapi_exceptions_t> : std::true_type
{
};

/* Enable bitwise operations for amd_dbgapi_instruction_properties_t.  */
template <>
struct is_flag<amd_dbgapi_instruction_properties_t> : std::true_type
{
};

/* Enable bitwise operations for amd_dbgapi_register_properties_t.  */
template <> struct is_flag<amd_dbgapi_register_properties_t> : std::true_type
{
};

/* Enable bitwise operations for amd_dbgapi_wave_stop_reasons_t.  */
template <> struct is_flag<amd_dbgapi_wave_stop_reasons_t> : std::true_type
{
};

/* A monotonic counter with wrap-around detection. Type must be an integral
   unsigned type. The WrapAroundCheck functor returns true if the counter has
   wrapped around. The default functor is (new_value > old_value).  */
template <typename Type, Type InitialValue = Type{},
          typename WrapAroundCheck = std::greater<Type>>
class monotonic_counter_t
{
  static_assert (std::is_integral_v<Type> && std::is_unsigned_v<Type>,
                 "Type is not an unsigned integral type");

public:
  using value_type = Type;

  /* Construct a new monotonic counter.  */
  monotonic_counter_t () { reset (); }

  /* Return the counter value. The counter is incremented each time it is
     accessed, guaranteeing a globally unique value until it is reset.  */
  value_type operator() ()
  {
    value_type old_value = m_value++;
    if (WrapAroundCheck{}(old_value, m_value))
      fatal_error ("monotonic counter wrapped around");

    return old_value;
  }

  /* Reset the counter.  */
  void reset () { m_value = InitialValue; }

private:
  value_type m_value{ 0 };
};

class notifier_t : private utils::not_copyable_t
{
public:
  static std::unique_ptr<notifier_t> create ();

  notifier_t () {}
  virtual ~notifier_t (){};

  /* The notifier_t is non-movable.  */
  notifier_t (notifier_t &&) = delete;
  notifier_t &operator= (notifier_t &&) = delete;

  virtual void open () = 0;
  virtual void close () = 0;
  virtual bool is_valid () const = 0;

  /* Return the platform-specific amd_dbgapi_notifier_t a producer can use
     to send notifications to the consumer.  */
  virtual amd_dbgapi_notifier_t producer_end () const = 0;

  /* Return the platform-specific amd_dbgapi_notifier_t a consumer can
     subscribe to.  */
  virtual amd_dbgapi_notifier_t consumer_end () const = 0;

  /* Notify the consumer of the notifier_t.  The true on success, false
     otherwise.  */
  virtual bool mark () = 0;

  /* Clear the notifier_t.  Return true on success, false otherwise.  */
  virtual bool clear () = 0;
};

} /* namespace amd::dbgapi */

#endif /* AMD_DBGAPI_UTILS_H */
