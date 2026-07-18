/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace hip {
namespace contract {

// A scope-exit guard for HIP contract tests. Register a cleanup action right
// after acquiring a resource; the actions run in reverse order when the guard
// leaves scope, INCLUDING when a REQUIRE/HIP_CHECK throws and unwinds the stack.
//
// Contract tests use raw HIP handles (device allocations, streams, events,
// graphs, modules, arrays, VMM mappings, memory pools) and historically placed
// the matching hipFree/hipStreamDestroy/... after the assertions, so a failing
// assertion leaked the resource into Catch2's shared, randomized-order process.
// Registering the teardown with this guard makes cleanup exception-safe without
// spelling out a bespoke RAII class per resource type.
//
// Usage:
//   ContractCleanup cleanup;
//   int* ptr = nullptr;
//   HIP_CHECK(hipMalloc(&ptr, bytes));
//   cleanup.Add([ptr] { (void)hipFree(ptr); });
//   ...assertions that may throw...
//
// Capture the handle BY VALUE, not by reference. The guard is declared before
// the resources it releases, so locals are destroyed in reverse declaration
// order: a resource declared after `cleanup` ends its lifetime before
// ~ContractCleanup runs. A by-reference capture ([&]) would then read a
// dangling local; capturing the handle's value at registration is
// order-independent and equivalent to correct RAII. For a handle stored in a
// struct field, snapshot it with an init-capture: [p = obj.handle] { ... }.
// A cleanup that must observe state mutated AFTER registration (e.g. a pointer
// nulled once an explicit free runs) is the exception and captures by reference
// deliberately.
//
// Cleanup callables must not throw; wrap HIP calls so their result is ignored
// (the process is already unwinding on the failure path and a second fault in a
// destructor would call std::terminate).
class ContractCleanup {
 public:
  ContractCleanup() = default;

  ~ContractCleanup() {
    for (auto it = actions_.rbegin(); it != actions_.rend(); ++it) {
      (*it)();
    }
  }

  ContractCleanup(const ContractCleanup&) = delete;
  ContractCleanup& operator=(const ContractCleanup&) = delete;

  void Add(std::function<void()> action) { actions_.push_back(std::move(action)); }

 private:
  std::vector<std::function<void()>> actions_;
};

}  // namespace contract
}  // namespace hip
