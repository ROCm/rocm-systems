/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Around a base test include (with derived_transport_tests_end.h): TEST_F registers each base test unless excluded.
 * Excluded tests still compile, so a base test that no longer builds against the derived code fails the build.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "transport/derived_transport_tests.h"

#pragma push_macro("TEST_F")
#undef TEST_F
#define TEST_F(fixture, name) RCCL_DERIVED_TEST_F_(fixture, name)
#define RCCL_DERIVED_TEST_F_(fixture, name)                                                                     \
  class fixture##_##name##_BaseTest : public fixture {                                                         \
    void TestBody() override;                                                                                  \
  };                                                                                                           \
  [[maybe_unused]] static const bool fixture##_##name##_BaseRegistered =                                       \
      ::rccl_derived_tests::RegisterBaseTest<fixture, fixture##_##name##_BaseTest>(#fixture, #name, __FILE__, \
                                                                                   __LINE__);                  \
  void fixture##_##name##_BaseTest::TestBody()
