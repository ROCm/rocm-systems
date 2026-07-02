/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Plain destructor callback for user objects. It only increments the integer
// counter reachable through userData; it never calls HIP APIs and never blocks,
// which keeps destructor execution safe and its side effect trivially
// observable.
void DestroyCounter(void* userData) {
  int* counter = static_cast<int*>(userData);
  ++(*counter);
}

// Attempts to create a user object with a single initial reference and the
// no-destructor-sync flag so that destructor timing stays deterministic.
// Returns false (without touching the out object) when the runtime path does
// not support user objects, which is treated as an unsupported-capability skip
// rather than a contract failure.
bool TryCreateUserObject(hipUserObject_t* object, int* counter) {
  const hipError_t status = hipUserObjectCreate(
      object, counter, DestroyCounter, 1, hipUserObjectNoDestructorSync);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphUserObjects_CreateRelease_InvokesDestructorOnce) {
  int counter = 0;
  hipUserObject_t object = nullptr;

  if (!TryCreateUserObject(&object, &counter)) {
    HIP_SKIP_TEST("User objects are not supported by this runtime path.");
  }
  REQUIRE(object != nullptr);

  // The single initial reference is released, which must drop the refcount to
  // zero and run the destructor exactly once.
  HIP_CHECK(hipUserObjectRelease(object, 1));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(counter == 1);
}

HIP_TEST_CASE(Contract_GraphUserObjects_RetainRelease_BalancesRefcount) {
  int counter = 0;
  hipUserObject_t object = nullptr;

  if (!TryCreateUserObject(&object, &counter)) {
    HIP_SKIP_TEST("User objects are not supported by this runtime path.");
  }
  REQUIRE(object != nullptr);

  // Retain two extra references so the object holds three references total.
  HIP_CHECK(hipUserObjectRetain(object, 2));

  // Releasing the two retained references must not run the destructor while a
  // reference still remains.
  HIP_CHECK(hipUserObjectRelease(object, 2));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(counter == 0);

  // Releasing the final initial reference drops the refcount to zero and runs
  // the destructor exactly once.
  HIP_CHECK(hipUserObjectRelease(object, 1));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(counter == 1);
}

HIP_TEST_CASE(Contract_GraphUserObjects_GraphRetainRelease_TiedToGraphLifetime) {
  int counter = 0;
  hipUserObject_t object = nullptr;
  hipGraph_t graph = nullptr;

  if (!TryCreateUserObject(&object, &counter)) {
    HIP_SKIP_TEST("User objects are not supported by this runtime path.");
  }
  REQUIRE(object != nullptr);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  // The graph takes its own reference on the object, and then releases it. This
  // reference is independent of the standalone initial reference, so releasing
  // it must not run the destructor while the standalone reference remains.
  HIP_CHECK(hipGraphRetainUserObject(graph, object, 1, 0));
  HIP_CHECK(hipGraphReleaseUserObject(graph, object, 1));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(counter == 0);

  // Releasing the standalone initial reference drops the refcount to zero and
  // runs the destructor exactly once, independent of graph destruction.
  HIP_CHECK(hipUserObjectRelease(object, 1));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(counter == 1);

  HIP_CHECK(hipGraphDestroy(graph));
}

HIP_TEST_CASE(Contract_GraphUserObjects_CreateNullObject_ReturnsInvalidValue) {
  int counter = 0;

  // A null out-object pointer is an invalid argument, so create must report a
  // portable invalid-value error rather than an unsupported-capability code.
  const hipError_t status = hipUserObjectCreate(
      nullptr, &counter, DestroyCounter, 1, hipUserObjectNoDestructorSync);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("User objects are not supported by this runtime path.");
  }
  REQUIRE(status == hipErrorInvalidValue);
}
