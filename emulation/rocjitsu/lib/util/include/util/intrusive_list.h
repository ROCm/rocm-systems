// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_INTRUSIVE_LIST_H_
#define UTIL_INTRUSIVE_LIST_H_

#include "util/meta_programming.h"

#include <concepts>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <typeinfo>

namespace util {

template <typename ParentT> struct IListParent;

namespace detail {

template <typename NodeT> struct IListNodeBase {
  NodeT *prev_ = nullptr;
  NodeT *next_ = nullptr;
};

template <typename ParentT> struct IListNodeParent {
  ParentT *parent_ = nullptr;
};

template <metaprogramming::IsVoid ParentT> struct IListNodeParent<ParentT> {};

template <typename> struct IsListParent : std::false_type {};
template <> struct IsListParent<IListParent<void>> : std::false_type {};
template <typename ParentT> struct IsListParent<IListParent<ParentT>> : std::true_type {};

template <typename OptionT>
concept ValidNodeOptions = IsListParent<OptionT>::value;

template <typename... Options> struct NodeOptions {
  using ParentT = metaprogramming::GetOption<IListParent, Options...>::Type;
};

} // namespace detail

template <typename T, typename... Options>
  requires(detail::ValidNodeOptions<Options> && ...)
class IListNode
    : public detail::IListNodeBase<IListNode<T, Options...>>,
      public detail::IListNodeParent<typename detail::NodeOptions<Options...>::ParentT> {
public:
  using ParentT = detail::NodeOptions<Options...>::ParentT;

  IListNode() = default;

  ParentT *parent()
    requires(metaprogramming::IsNonVoid<ParentT>)
  {
    return this->parent_;
  }

  const ParentT *parent() const
    requires(metaprogramming::IsNonVoid<ParentT>)
  {
    return this->parent_;
  }
};

template <typename T, typename... Options> class IListIterator {
public:
  // Standard C++ iterator types.
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = std::remove_const_t<T>;
  using pointer = T *;
  using reference = T &;
  // Custom iterator types.
  using NodeT = IListNode<std::remove_const_t<T>, Options...>;
  using NodePointer = std::conditional_t<std::is_const_v<T>, const NodeT *, NodeT *>;
  using ConstNodePointer = const NodeT *;
  using NodeReference = std::conditional_t<std::is_const_v<T>, const NodeT &, NodeT &>;

  IListIterator() = default;
  explicit IListIterator(pointer node_ptr) : node_ptr_(static_cast<NodePointer>(node_ptr)) {}
  explicit IListIterator(reference node_ref) : node_ptr_(static_cast<NodePointer>(&node_ref)) {}
  explicit IListIterator(NodeReference node_ref) : node_ptr_(&node_ref) {}
  template <typename U>
    requires(std::is_const_v<T> && std::same_as<U, std::remove_const_t<T>>)
  IListIterator(const IListIterator<U, Options...> &other) : node_ptr_(other.node_pointer()) {}

  NodePointer node_pointer() { return node_ptr_; }
  ConstNodePointer node_pointer() const { return node_ptr_; }

  reference operator*() const { return *static_cast<pointer>(node_ptr_); }

  friend bool operator==(const IListIterator &lhs, const IListIterator &rhs) {
    return lhs.node_ptr_ == rhs.node_ptr_;
  }

  friend bool operator!=(const IListIterator &lhs, const IListIterator &rhs) {
    return lhs.node_ptr_ != rhs.node_ptr_;
  }

  /// @brief Prefix increment operator.
  IListIterator &operator++() {
    node_ptr_ = node_ptr_->next_;
    return *this;
  }

  /// @brief Postfix increment operator.
  IListIterator operator++(int) {
    IListIterator tmp(*this);
    operator++();
    return tmp;
  }

  /// @brief Prefix decrement operator.
  IListIterator &operator--() {
    node_ptr_ = node_ptr_->prev_;
    return *this;
  }

  /// @brief Postfix decrement operator.
  IListIterator operator--(int) {
    IListIterator tmp(*this);
    operator--();
    return tmp;
  }

private:
  NodePointer node_ptr_ = nullptr;
};

template <typename T, typename... Options> class IntrusiveList {
public:
  using Iterator = IListIterator<T, Options...>;
  using ConstIterator = IListIterator<const T, Options...>;

  IntrusiveList() {
    sentinel_.next_ = &sentinel_;
    sentinel_.prev_ = &sentinel_;
  }

  ~IntrusiveList() {}

  /// @brief Inserts an element at a specified location in the list.
  /// @param it Iterator before which the element will be inserted.
  /// @param node Element to insert into the list.
  /// @returns Iterator pointing to the newly inserted element.
  Iterator insert(Iterator it, T &node) {
    NodeT &next = *it.node_pointer();
    NodeT &prev = *next.prev_;
    node.next_ = &next;
    node.prev_ = &prev;
    prev.next_ = &node;
    next.prev_ = &node;
    return Iterator(node);
  }

  /// @brief Erase the element at a specified location from the list.
  /// @param it Iterator pointing to the element to be erased.
  /// @returns Iterator pointing to the next element after the one pointed to by @p it. If @p it
  /// points to the last element in the list end() is returned.
  Iterator erase(Iterator it) {
    NodeT &node = *it.node_pointer();
    NodeT &prev = *node.prev_;
    NodeT &next = *node.next_;
    prev.next_ = &next;
    next.prev_ = &prev;
    node.prev_ = nullptr;
    node.next_ = nullptr;
    return Iterator(next);
  }
  /// @brief Return an iterator to the beginning of the list.
  /// @returns Iterator pointing to the first element in the list.
  Iterator begin() { return ++Iterator(sentinel_); }
  ConstIterator begin() const { return ++ConstIterator(sentinel_); }
  /// @brief Return an iterator past the end of the list.
  /// @details The iterator returned is the sentinel which is pointed to by the next pointer in the
  /// last element in the list. This iterator should not be dereferenced.
  /// @returns The iterator pointing to the sentinel.
  Iterator end() { return Iterator(sentinel_); }
  ConstIterator end() const { return ConstIterator(sentinel_); }
  ConstIterator cbegin() const { return begin(); }
  ConstIterator cend() const { return end(); }
  /// @brief Checks if the list is empty.
  /// @returns Bool indicating if the list is empty or not.
  bool empty() const { return sentinel_.next_ == &sentinel_; }
  /// @brief Adds an element to the end of the list.
  /// @param node Element to be added to the list.
  void push_back(T &node) { insert(end(), node); }
  /// @brief Adds an element to the beginning of the list.
  /// @param node Element to be added to the list.
  void push_front(T &node) { insert(begin(), node); }

private:
  using NodeT = IListNode<T, Options...>;
  NodeT sentinel_;
};

} // namespace util

#endif // UTIL_INTRUSIVE_LIST_H_
