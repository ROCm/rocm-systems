/* Copyright (c) 2026 Advanced Micro Devices, Inc.

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

/* Unit tests for src/handle_object.h.  Uses a synthetic handle/object
   pair (test_handle_t / test_obj_t) so we don't drag in agent_t,
   queue_t, or any architecture state.  Covers:
     - sentinel_id<>()
     - is_handle_type_v / is_handle_object_type_v concepts
     - hash<> for handle types
     - handle_object_set_t CRUD (create, find, find_if, destroy, clear,
       iteration, changed flag, next_id reset).  */

#include "handle_object.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <type_traits>

using namespace amd::dbgapi;

namespace handle_object_test_ns
{

/* A synthetic handle type: trivial struct with a single unsigned
   integral 'handle' member, which is what is_handle_type_v requires.  */
struct test_handle_t
{
  uint32_t handle;
};

inline bool
operator== (const test_handle_t &a, const test_handle_t &b)
{
  return a.handle == b.handle;
}

/* Synthetic handle-object that derives from detail::handle_object.
   The implicit destructor on test_obj_t is public (the base's
   protected destructor is accessible from the derived class scope),
   which is what unique_ptr<test_obj_t> inside the set needs.  */
class test_obj_t : public detail::handle_object<test_handle_t>
{
public:
  test_obj_t (const test_handle_t &id, int value = 0)
    : handle_object (id), m_value (value)
  {
  }
  int value () const { return m_value; }

private:
  int m_value;
};

} /* namespace handle_object_test_ns */

/* ------------------------------------------------------------------ */
/* sentinel_id                                                         */
/* ------------------------------------------------------------------ */

TEST (HandleObjectSentinel, ReservedValues)
{
  using handle_object_test_ns::test_handle_t;

  /* sentinel_id<H, N> is H{~N}.  Reserved range is [~0, ~4].  */
  EXPECT_EQ (sentinel_id<test_handle_t> ().handle, ~uint32_t{ 0 });
  EXPECT_EQ ((sentinel_id<test_handle_t, 1> ()).handle, ~uint32_t{ 1 });
  EXPECT_EQ ((sentinel_id<test_handle_t, 4> ()).handle, ~uint32_t{ 4 });
}

/* ------------------------------------------------------------------ */
/* is_handle_type_v / is_handle_object_type_v                          */
/* ------------------------------------------------------------------ */

TEST (HandleObjectConcepts, IsHandleType)
{
  using handle_object_test_ns::test_handle_t;

  static_assert (is_handle_type_v<test_handle_t>,
                 "test_handle_t must satisfy is_handle_type");
  static_assert (!is_handle_type_v<int>,
                 "plain int has no 'handle' member");

  struct signed_handle_t
  {
    int32_t handle;
  };
  static_assert (!is_handle_type_v<signed_handle_t>,
                 "is_handle_type requires unsigned integral 'handle'");
}

TEST (HandleObjectConcepts, IsHandleObjectType)
{
  using handle_object_test_ns::test_obj_t;

  static_assert (is_handle_object_type_v<test_obj_t>,
                 "test_obj_t::id() returns a handle type");
  static_assert (!is_handle_object_type_v<int>);
}

/* ------------------------------------------------------------------ */
/* hash<Handle>                                                        */
/* ------------------------------------------------------------------ */

TEST (HandleObjectHash, DistinctInputsHashDifferently)
{
  using handle_object_test_ns::test_handle_t;

  hash<test_handle_t> h;
  EXPECT_EQ (h (test_handle_t{ 1 }), h (test_handle_t{ 1 }));
  /* Not strictly guaranteed by std::hash, but holds for uint32_t in
     libstdc++/libc++ for the values 1 and 2 - if this ever fires on a
     new STL, the test should be reframed.  */
  EXPECT_NE (h (test_handle_t{ 1 }), h (test_handle_t{ 2 }));
}

/* ------------------------------------------------------------------ */
/* handle_object_set_t                                                 */
/* ------------------------------------------------------------------ */

namespace
{

/* Each test gets a clean monotonic counter; otherwise the per-type
   static counter inside next_id() leaks across tests.  */
struct HandleObjectSet : public ::testing::Test
{
  using set_t = handle_object_set_t<handle_object_test_ns::test_obj_t>;
  set_t set;

  void SetUp () override { set_t::reset_next_id (); }
};

} /* namespace */

TEST_F (HandleObjectSet, EmptyAfterConstruction)
{
  EXPECT_EQ (set.size (), 0u);
  EXPECT_FALSE (set.changed ());
  EXPECT_TRUE (set.begin () == set.end ());
}

TEST_F (HandleObjectSet, CreateAssignsMonotonicallyIncreasingIds)
{
  auto &a = set.create_object (10);
  auto &b = set.create_object (20);

  EXPECT_EQ (a.id ().handle, 1u);
  EXPECT_EQ (b.id ().handle, 2u);
  EXPECT_EQ (a.value (), 10);
  EXPECT_EQ (b.value (), 20);
  EXPECT_EQ (set.size (), 2u);
  EXPECT_TRUE (set.changed ());
}

TEST_F (HandleObjectSet, CreateWithExplicitIdHonorsRequest)
{
  using handle_object_test_ns::test_handle_t;

  /* The explicit-id overload takes std::optional<handle_type> as its
     first parameter.  Passing a bare test_handle_t would match the
     variadic forward overload instead (perfect match beats implicit
     conversion to optional), so wrap it explicitly.  */
  auto &obj = set.create_object (
    std::optional<test_handle_t>{ test_handle_t{ 42 } }, 99);
  EXPECT_EQ (obj.id ().handle, 42u);
  EXPECT_EQ (obj.value (), 99);
}

TEST_F (HandleObjectSet, FindReturnsRegisteredObject)
{
  auto &a = set.create_object (10);
  auto *p = set.find (a.id ());
  ASSERT_NE (p, nullptr);
  EXPECT_EQ (p, &a);
  EXPECT_EQ (p->value (), 10);
}

TEST_F (HandleObjectSet, FindReturnsNullForMissingId)
{
  using handle_object_test_ns::test_handle_t;
  EXPECT_EQ (set.find (test_handle_t{ 9999 }), nullptr);
}

TEST_F (HandleObjectSet, FindIfMatchesPredicate)
{
  set.create_object (10);
  auto &b = set.create_object (20);
  set.create_object (30);

  auto *match = set.find_if (
    [] (const auto &obj) { return obj.value () == 20; });
  ASSERT_NE (match, nullptr);
  EXPECT_EQ (match, &b);

  auto *miss = set.find_if (
    [] (const auto &obj) { return obj.value () == 12345; });
  EXPECT_EQ (miss, nullptr);
}

TEST_F (HandleObjectSet, DestroyByPointerRemovesObject)
{
  auto &a = set.create_object (10);
  auto id = a.id ();
  set.set_changed (false);

  set.destroy (&a);
  EXPECT_EQ (set.size (), 0u);
  EXPECT_EQ (set.find (id), nullptr);
  EXPECT_TRUE (set.changed ());
}

TEST_F (HandleObjectSet, ClearEmptiesTheSet)
{
  set.create_object (1);
  set.create_object (2);
  set.create_object (3);

  set.clear ();
  EXPECT_EQ (set.size (), 0u);
  EXPECT_TRUE (set.changed ());
}

TEST_F (HandleObjectSet, ChangedFlagRoundtrip)
{
  EXPECT_FALSE (set.changed ());
  set.create_object (1);
  EXPECT_TRUE (set.changed ());

  bool prev = set.set_changed (false);
  EXPECT_TRUE (prev);
  EXPECT_FALSE (set.changed ());
}

TEST_F (HandleObjectSet, IterationVisitsEveryObject)
{
  set.create_object (1);
  set.create_object (2);
  set.create_object (3);

  int sum = 0;
  size_t count = 0;
  for (auto &obj : set)
    {
      sum += obj.value ();
      ++count;
    }
  EXPECT_EQ (count, 3u);
  EXPECT_EQ (sum, 6);
}

TEST_F (HandleObjectSet, ResetNextIdRestartsCounter)
{
  auto &a = set.create_object (1);
  EXPECT_EQ (a.id ().handle, 1u);

  set.clear ();
  set_t::reset_next_id ();

  auto &b = set.create_object (2);
  EXPECT_EQ (b.id ().handle, 1u);
}
