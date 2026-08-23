// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Exercises a behavioral ConSan pair through HIP module load, alternating
/// subset unload/reload, four simultaneous independent code objects, and
/// launches on separate streams. The kernels are deliberately absent from
/// this executable's static image.

#include "consan_device_test_support.h"

#include <array>

#ifndef RJ_CONSAN_MODULE_LIFECYCLE_HSACO_A
#error "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_A must name the first module fixture"
#endif
#ifndef RJ_CONSAN_MODULE_LIFECYCLE_HSACO_B
#error "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_B must name the second module fixture"
#endif
#ifndef RJ_CONSAN_MODULE_LIFECYCLE_HSACO_C
#error "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_C must name the third module fixture"
#endif
#ifndef RJ_CONSAN_MODULE_LIFECYCLE_HSACO_D
#error "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_D must name the fourth module fixture"
#endif

namespace {

constexpr uint32_t kModuleCount = 4u;
constexpr uint32_t kGenerations = 3u;
constexpr uint32_t kRacyModule = 2u;
constexpr uint32_t kMaximumThreads = 128u;

struct ModuleSpec {
  const char *path;
  const char *correct_kernel;
  const char *incorrect_kernel;
  uint32_t tag;
};

constexpr ModuleSpec kModuleA{RJ_CONSAN_MODULE_LIFECYCLE_HSACO_A, "module_lifecycle_correct_a",
                              "module_lifecycle_incorrect_a", 0xa1000000u};
constexpr ModuleSpec kModuleB{RJ_CONSAN_MODULE_LIFECYCLE_HSACO_B, "module_lifecycle_correct_b",
                              "module_lifecycle_incorrect_b", 0xb2000000u};
constexpr ModuleSpec kModuleC{RJ_CONSAN_MODULE_LIFECYCLE_HSACO_C, "module_lifecycle_correct_c",
                              "module_lifecycle_incorrect_c", 0xc3000000u};
constexpr ModuleSpec kModuleD{RJ_CONSAN_MODULE_LIFECYCLE_HSACO_D, "module_lifecycle_correct_d",
                              "module_lifecycle_incorrect_d", 0xd4000000u};
constexpr std::array<ModuleSpec, kModuleCount> kModules{kModuleA, kModuleB, kModuleC, kModuleD};

struct ModuleResult {
  uint32_t observed;
  uint32_t control[kMaximumThreads];
};

struct ModuleInstance {
  hipModule_t module = nullptr;
  hipStream_t stream = nullptr;
  ModuleResult *device_result = nullptr;
};

void load_module(ModuleInstance &instance, const ModuleSpec &spec) {
  HIP_ASSERT(hipModuleLoad(&instance.module, spec.path));
}

void reload_module(ModuleInstance &instance, const ModuleSpec &spec) {
  HIP_ASSERT(hipModuleUnload(instance.module));
  instance.module = nullptr;
  load_module(instance, spec);
}

void launch_loaded_module(const ModuleInstance &instance, const ModuleSpec &spec, bool correct,
                          uint32_t marker) {
  hipFunction_t function = nullptr;
  HIP_ASSERT(hipModuleGetFunction(&function, instance.module,
                                  correct ? spec.correct_kernel : spec.incorrect_kernel));
  struct {
    ModuleResult *result;
    uint32_t marker;
  } arguments{instance.device_result, marker};
  size_t argument_size = sizeof(arguments);
  void *configuration[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &arguments,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &argument_size,
      HIP_LAUNCH_PARAM_END,
  };
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  HIP_ASSERT(hipModuleLaunchKernel(function, 1u, 1u, 1u, thread_count, 1u, 1u, 0u, instance.stream,
                                   nullptr, configuration));
}

void check_result(const ModuleResult &result, const ModuleSpec &spec, uint32_t marker,
                  uint32_t thread_count, bool require_observed) {
  for (uint32_t tid = 0u; tid < thread_count; ++tid) {
    EXPECT_EQ(result.control[tid], marker ^ spec.tag ^ tid)
        << "module_tag=" << spec.tag << " marker=" << marker << " thread=" << tid;
  }
  if (require_observed)
    EXPECT_EQ(result.observed, marker ^ spec.tag)
        << "module_tag=" << spec.tag << " marker=" << marker;
}

void run_module_lifecycle(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, kMaximumThreads);

  std::array<ModuleInstance, kModuleCount> instances{};
  for (uint32_t module = 0u; module < kModuleCount; ++module) {
    load_module(instances[module], kModules[module]);
    HIP_ASSERT(hipStreamCreate(&instances[module].stream));
    HIP_ASSERT(hipMalloc(&instances[module].device_result, sizeof(ModuleResult)));
  }

  for (uint32_t generation = 0u; generation < kGenerations; ++generation) {
    // Keep half of the independently instrumented objects live while the
    // other half is replaced. Generation one reloads odd modules; generation
    // two reloads even modules, including the designated racy object.
    if (generation != 0u) {
      const uint32_t parity = generation & 1u;
      for (uint32_t module = parity; module < kModuleCount; module += 2u)
        reload_module(instances[module], kModules[module]);
    }

    for (uint32_t module = 0u; module < kModuleCount; ++module) {
      const uint32_t marker = 0x005a0000u | (generation << 8u) | module;
      const bool racy = !synchronize && module == kRacyModule;
      HIP_ASSERT(hipMemsetAsync(instances[module].device_result, 0, sizeof(ModuleResult),
                                instances[module].stream));
      launch_loaded_module(instances[module], kModules[module], !racy, marker);
    }
    for (ModuleInstance &instance : instances)
      HIP_ASSERT(hipStreamSynchronize(instance.stream));

    for (uint32_t module = 0u; module < kModuleCount; ++module) {
      const uint32_t marker = 0x005a0000u | (generation << 8u) | module;
      const bool racy = !synchronize && module == kRacyModule;
      ModuleResult result{};
      HIP_ASSERT(hipMemcpy(&result, instances[module].device_result, sizeof(result),
                           hipMemcpyDeviceToHost));
      check_result(result, kModules[module], marker, thread_count, !racy);
    }
  }

  // Tear down in reverse order so load and unload order are not accidentally
  // coupled. Every stream is already synchronized above.
  for (uint32_t module = kModuleCount; module-- > 0u;) {
    HIP_ASSERT(hipFree(instances[module].device_result));
    HIP_ASSERT(hipStreamDestroy(instances[module].stream));
    HIP_ASSERT(hipModuleUnload(instances[module].module));
  }
}

TEST(ConSanDeviceModuleLifecycleTest, Correct) { run_module_lifecycle(true); }

TEST(ConSanDeviceModuleLifecycleTest, Incorrect) { run_module_lifecycle(false); }

} // namespace
