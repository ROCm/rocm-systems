// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/intrusive_list.h"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>

namespace {

struct Owner {};

class Entry : public util::IListNode<Entry, util::IListParent<Owner>> {
public:
  explicit Entry(int value) : value_(value) {}

  int value() const { return value_; }

private:
  int value_;
};

using EntryList = util::IntrusiveList<Entry, util::IListParent<Owner>>;

static_assert(std::bidirectional_iterator<EntryList::Iterator>);
static_assert(std::bidirectional_iterator<EntryList::ConstIterator>);
static_assert(std::ranges::bidirectional_range<const EntryList>);
static_assert(std::same_as<std::iter_reference_t<EntryList::ConstIterator>, const Entry &>);
static_assert(std::same_as<std::iter_reference_t<EntryList::Iterator>, Entry &>);
static_assert(std::same_as<std::iterator_traits<EntryList::ConstIterator>::value_type, Entry>);
static_assert(std::convertible_to<EntryList::Iterator, EntryList::ConstIterator>);
static_assert(!std::convertible_to<EntryList::ConstIterator, EntryList::Iterator>);

TEST(IntrusiveListTest, IteratesConstList) {
  Entry first(1);
  Entry second(2);
  Entry third(3);
  EntryList list;
  const EntryList &const_list = list;

  EXPECT_EQ(const_list.begin(), const_list.end());
  EXPECT_EQ(const_list.cbegin(), const_list.cend());

  list.push_back(first);
  list.push_back(second);
  list.push_back(third);

  EntryList::ConstIterator converted = list.begin();
  EXPECT_EQ(converted, const_list.begin());

  const std::array<const Entry *, 3> expected{&first, &second, &third};
  std::size_t index = 0;
  for (const Entry &entry : const_list) {
    ASSERT_LT(index, expected.size());
    EXPECT_EQ(&entry, expected[index++]);
  }
  EXPECT_EQ(index, expected.size());

  EntryList::ConstIterator it = const_list.cend();
  EXPECT_EQ(&*--it, &third);
  EXPECT_EQ(&*--it, &second);
  EXPECT_EQ(&*--it, &first);
  EXPECT_EQ(it, const_list.cbegin());
  EXPECT_EQ((*it).value(), 1);
}

} // namespace
