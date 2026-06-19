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

#ifndef AMD_DBGAPI_HANDLE_OBJECT_H
#define AMD_DBGAPI_HANDLE_OBJECT_H 1

#include "utils.h"

#include <algorithm>
#include <memory>
#include <type_traits>
#include <unordered_map>

namespace amd::dbgapi
{

/* Number of sentinel values reserved for each handle type.  */
constexpr int HANDLE_SENTINEL_COUNT = 5;

template <typename Handle, int N = 0>
constexpr Handle
sentinel_id ()
{
  static_assert (N < HANDLE_SENTINEL_COUNT, "invalid parameter");
  return Handle{ ~decltype (Handle::handle){ N } };
}

namespace detail
{

/* Base class for all handle objects, i.e. agents_t, queue_t, wave_t, ...
   Handle objects should only by allocated within their handle object sets,
   should not be copied nor moved.  */
template <typename Handle> class handle_object
{
public:
  using handle_type = Handle;

private:
  handle_type const m_id;

protected:
  explicit handle_object (const handle_type &id) : m_id (id) {}
  ~handle_object () = default;

  /* Disable copies and move assignments.  */
  handle_object (const handle_object &) = delete;
  handle_object (handle_object &&) = delete;
  handle_object &operator= (const handle_object &) = delete;
  handle_object &operator= (handle_object &&) = delete;

public:
  handle_type id () const { return m_id; }

  /* Since handle_object disallows copying & moving, two handle_objects are
     identical if they have the same address.  */
  bool operator== (const handle_object &other) const
  {
    return id () == other.id ();
  }
  bool operator!= (const handle_object &other) const
  {
    return id () != other.id ();
  }
};

/* Wrap-around check functor for handles. An error is thrown if the
   handle value reaches ~0 as it is reserved for a mark object.  */
template <typename Type> struct handle_has_wrapped_around
{
  bool operator() (const Type &, const Type &new_value) const
  {
    return new_value == ~Type{ HANDLE_SENTINEL_COUNT - 1 };
  }
};

/* Type of Object::is_valid ()  */
template <typename Object>
using is_valid_t = decltype (std::declval<Object> ().is_valid ());

} /* namespace detail */

/* Call is_valid () if Object implements it, return true otherwise.  */
template <typename Object>
constexpr bool
is_valid (Object &&object)
{
  if constexpr (utils::is_detected_v<detail::is_valid_t, Object>)
    return object.is_valid ();
  return true;
}

/* A handle type is a type that has an unsigned integral handle data member. */

namespace detail
{

/* Type of Handle::handle  */
template <typename Type>
using id_handle_t = decltype (std::declval<Type> ().handle);

} /* namespace detail */

template <typename Type>
inline constexpr bool is_handle_type_v = std::conjunction_v<
  std::is_integral<utils::detected_t<detail::id_handle_t, Type>>,
  std::is_unsigned<utils::detected_t<detail::id_handle_t, Type>>>;

/* Hash function for handle types.  */
template <typename Handle, std::enable_if_t<is_handle_type_v<Handle>, int> = 0>
struct hash
{
  std::size_t operator() (const Handle &rhs) const noexcept
  {
    return std::hash<decltype (rhs.handle)>{}(rhs.handle);
  }
};

/* A handle object type is a class that implements the id() member function,
   and the return type of the id() function is a handle type.  */

namespace detail
{

/* Type of Object::id ()  */
template <typename Type>
using object_id_t = decltype (std::declval<Type> ().id ());

} /* namespace detail */

template <typename Type>
inline constexpr bool is_handle_object_type_v
  = is_handle_type_v<utils::detected_t<detail::object_id_t, Type>>;

/* Specialize this struct to change the initial value of a monotonic counter
   for a given handle_object.  */
template <typename Type, std::enable_if_t<is_handle_type_v<Type>, int> = 0>
struct monotonic_counter_start_t
  : public std::integral_constant<decltype (Type::handle), 1>
{
};

template <typename Type>
inline constexpr decltype (Type::handle) monotonic_counter_start_v
  = monotonic_counter_start_t<Type>::value;

/* A set container type that holds objects that are referenced using handles.
   If the Object type supports the is_valid concept, then when creating objects
   an error is reported if the constructor creates an object that is not valid.
 */
template <typename Object> class handle_object_set_t
{
  static_assert (is_handle_object_type_v<Object>,
                 "Object is not a handle object");

public:
  using object_type = Object;
  using handle_type = utils::detected_t<detail::object_id_t, Object>;
  using map_type = std::unordered_map<handle_type, std::unique_ptr<Object>,
                                      hash<handle_type>>;

private:
  /* Flag to tell if the content of the map has changed.  */
  bool m_changed{ false };

  /* Map holding the objects, keyed by object::id ()'s return type. */
  map_type m_map{};

public:
  class iterator
  {
  private:
    typename map_type::iterator m_it;

  public:
    using self_type = iterator;
    using value_type = Object;
    using difference_type = size_t;
    using reference = Object &;
    using pointer = Object *;
    using iterator_category = std::forward_iterator_tag;

    iterator (typename map_type::iterator it) : m_it (it) {}
    typename map_type::iterator get () const { return m_it; }
    self_type &operator++ ()
    {
      ++m_it;
      return *this;
    }
    self_type operator++ (int)
    {
      self_type i = *this;
      ++m_it;
      return i;
    }
    reference operator* () { return *m_it->second; }
    pointer operator-> () { return m_it->second.get (); }
    bool operator== (const self_type &rhs) { return m_it == rhs.m_it; }
    bool operator!= (const self_type &rhs) { return m_it != rhs.m_it; }
  };

  class const_iterator
  {
  private:
    typename map_type::const_iterator m_it;

  public:
    using self_type = const_iterator;
    using value_type = Object;
    using difference_type = size_t;
    using const_reference = const Object &;
    using const_pointer = const Object *;
    using iterator_category = std::forward_iterator_tag;

    const_iterator (typename map_type::const_iterator it) : m_it (it) {}
    typename map_type::const_iterator get () const { return m_it; }
    self_type &operator++ ()
    {
      ++m_it;
      return *this;
    }
    self_type operator++ (int)
    {
      self_type i = *this;
      ++m_it;
      return i;
    }
    const_reference operator* () const { return *m_it->second; }
    const_pointer operator-> () const { return m_it->second.get (); }
    bool operator== (const self_type &rhs) const { return m_it == rhs.m_it; }
    bool operator!= (const self_type &rhs) const { return m_it != rhs.m_it; }
  };

  class range_t
  {
  public:
    explicit range_t (handle_object_set_t *objects) : m_objects (objects) {}
    iterator begin () { return iterator (m_objects->begin ()); }
    iterator end () { return iterator (m_objects->end ()); }

  private:
    handle_object_set_t *m_objects;
  };

  class const_range_t
  {
  public:
    explicit const_range_t (const handle_object_set_t *objects)
      : m_objects (objects)
    {
    }
    const_iterator begin () { return const_iterator (m_objects->begin ()); }
    const_iterator end () { return const_iterator (m_objects->end ()); }

  private:
    const handle_object_set_t *m_objects;
  };

  /* Delete copy and move.  */
  handle_object_set_t (const handle_object_set_t &) = delete;
  handle_object_set_t &operator= (const handle_object_set_t &) = delete;
  handle_object_set_t (handle_object_set_t &&) = delete;
  handle_object_set_t &operator= (handle_object_set_t &&) = delete;

  /* Default constructor.  */
  handle_object_set_t () = default;

  template <typename Derived = Object, typename... Args>
  inline Derived &create_object (std::optional<handle_type> id,
                                 Args &&...args);

  template <typename Derived = Object, typename... Args>
  Derived &create_object (Args &&...args)
  {
    return create_object<Derived> (std::optional<handle_type>{},
                                   std::forward<Args> (args)...);
  }

  void destroy (Object *object)
  {
    auto object_it = m_map.find (object->id ());
    dbgapi_assert (object_it != m_map.end ());

    m_changed = true;
    m_map.erase (object_it);
  }

  iterator destroy (iterator object_it)
  {
    m_changed = true;
    typename map_type::iterator it = object_it.get ();
    return iterator (m_map.erase (it));
  }

  /* The find() and find_if() operations hide any !is_valid() objects unless
    'all' is specified as 'true'.  This is important for the find_if() as the
    predicate may match multiple objects, of which only one is_valid().
    However, the range() operations will allow iteration of all objects
    including !is_valid().  As a consequence, the !is_valid() objects may stay
    in the map until range() is used to sweep over all the objects to destroy
    them.  */

  Object *find (handle_type id, bool all = false)
  {
    auto it = m_map.find (id);
    return (it != m_map.end () && (all || is_valid (*it->second)))
             ? it->second.get ()
             : nullptr;
  }

  const Object *find (handle_type id, bool all = false) const
  {
    auto it = m_map.find (id);
    return (it != m_map.end () && (all || is_valid (*it->second)))
             ? it->second.get ()
             : nullptr;
  }

  template <typename Functor>
  Object *find_if (Functor &&predicate, bool all = false)
  {
    auto it = std::find_if (m_map.begin (), m_map.end (),
                            [=] (const auto &value)
                            {
                              return bool{ (all || is_valid (*value.second))
                                           && predicate (*value.second) };
                            });
    return (it != m_map.end ()) ? it->second.get () : nullptr;
  }

  template <typename Functor>
  const Object *find_if (Functor &&predicate, bool all = false) const
  {
    auto it = std::find_if (m_map.begin (), m_map.end (),
                            [=] (const auto &value)
                            {
                              return bool{ (all || is_valid (*value.second))
                                           && predicate (*value.second) };
                            });
    return (it != m_map.end ()) ? it->second.get () : nullptr;
  }

  iterator begin () { return iterator (m_map.begin ()); }
  const_iterator begin () const { return const_iterator (m_map.cbegin ()); }

  iterator end () { return iterator (m_map.end ()); }
  const_iterator end () const { return const_iterator (m_map.cend ()); }

  range_t range () { return range_t{ this }; }
  const_range_t range () const { return const_range_t{ this }; }

  size_t size () const { return m_map.size (); }

  bool changed () const { return m_changed; }

  bool set_changed (bool changed)
  {
    bool ret = m_changed;
    m_changed = changed;
    return ret;
  }

  void clear ()
  {
    if (!m_map.empty ())
      {
        m_changed = true;
        m_map.clear ();
      }
  }

  static void reset_next_id () { next_id ().reset (); }

  static auto &next_id ()
  {
    /* Counter to assign ids to new objects. We start all the counters at 1,
       as 0 is the null object handle.  */
    using monotonic_counter_underlying_type_t = decltype (handle_type::handle);

    static monotonic_counter_t<
      monotonic_counter_underlying_type_t,
      monotonic_counter_start_v<handle_type>,
      detail::handle_has_wrapped_around<monotonic_counter_underlying_type_t>>
      counter;
    return counter;
  }
};

template <typename Object>
template <typename Derived, typename... Args>
inline Derived &
handle_object_set_t<Object>::create_object (std::optional<handle_type> id,
                                            Args &&...args)
{
  /* If id does not contain a value, request a new one.  This allows us to
     "re-create" objects that were place-holders (e.g. partially initialized
     queue object.  */
  if (!id)
    id = handle_type{ next_id () () };

  auto [it, success] = m_map.emplace (
    std::piecewise_construct, std::forward_as_tuple (*id),
    std::forward_as_tuple (new Derived (*id, std::forward<Args> (args)...)));

  if (!success)
    fatal_error ("could not create new object");

  if (!is_valid (it->second))
    {
      m_map.erase (it);
      fatal_error ("object is not valid");
    }

  m_changed = true;
  return *static_cast<Derived *> (it->second.get ());
}

namespace detail
{
/* Helper struct to return the object type associated with a handle type, e.g.
   process_t and amd_dbgapi_process_id_t.  It uses the handle_object_set tuple
   to match a given handle type with the return type of the id () function
   implemented by the tuple elements.  */

template <typename Handle, typename Tuple, size_t I>
struct object_type_from_handle;

template <typename Handle, typename Tuple>
struct object_type_from_handle<Handle, Tuple, 0>
{
  using element_0_type = typename std::tuple_element_t<0, Tuple>;
  using type = std::conditional_t<
    std::is_same_v<Handle, typename element_0_type::handle_type>,
    typename element_0_type::object_type, void>;
};

template <typename Handle, typename Tuple, size_t I>
struct object_type_from_handle
{
  using element_I_type = typename std::tuple_element_t<I, Tuple>;
  using type = std::conditional_t<
    std::is_same_v<Handle, typename element_I_type::handle_type>,
    typename element_I_type::object_type,
    typename object_type_from_handle<Handle, Tuple, I - 1>::type>;
};

} /* namespace detail */

template <typename Handle, typename Tuple>
using object_type_from_handle_t =
  typename detail::object_type_from_handle<Handle, Tuple,
                                           std::tuple_size_v<Tuple> - 1>::type;

namespace detail
{
template <typename Object, std::size_t N, typename... Args>
struct get_base_type_index
{
  static constexpr auto value = N;
};

template <typename Object, std::size_t N, typename HandleObjectSet,
          typename... Args>
struct get_base_type_index<Object, N, HandleObjectSet, Args...>
{
  static constexpr auto value
    = std::is_base_of_v<typename HandleObjectSet::object_type, Object>
        ? N
        : get_base_type_index<Object, N + 1, Args...>::value;
};
} /* namespace detail */

template <typename Object, typename... Args>
auto &
get_base_type_element (std::tuple<Args...> &tuple)
{
  return std::get<detail::get_base_type_index<Object, 0, Args...>::value> (
    tuple);
}

} /* namespace amd::dbgapi */

template <typename Handle,
          std::enable_if_t<amd::dbgapi::is_handle_type_v<Handle>, int> = 0>
bool
operator== (const Handle &lhs, const Handle &rhs)
{
  return lhs.handle == rhs.handle;
}

template <typename Handle,
          std::enable_if_t<amd::dbgapi::is_handle_type_v<Handle>, int> = 0>
bool
operator!= (const Handle &lhs, const Handle &rhs)
{
  return !(lhs == rhs);
}

template <typename Handle,
          std::enable_if_t<amd::dbgapi::is_handle_type_v<Handle>, int> = 0>
bool
operator! (const Handle &handle)
{
  return handle == Handle{};
}

#endif /* AMD_DBGAPI_HANDLE_OBJECT_H */
