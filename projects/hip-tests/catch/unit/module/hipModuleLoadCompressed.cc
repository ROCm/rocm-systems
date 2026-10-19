// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "hip_module_common.hh"

#include <array>
#include <cstring>
#include <resource_guards.hh>

namespace {
struct LoadedModules {
  std::array<hipModule_t, 3> modules{};

  ~LoadedModules() {
    for (auto module : modules) {
      if (module != nullptr) {
        static_cast<void>(hipModuleUnload(module));
      }
    }
  }
};

void VerifyCompressedModule(hipModule_t module, int expected, int* output) {
  hipDeviceptr_t global = nullptr;
  size_t size = 0;
  HIP_CHECK(hipModuleGetGlobal(&global, &size, module, "module_value"));
  REQUIRE(size == sizeof(int));
  int value = 0;
  HIP_CHECK(hipMemcpy(&value, global, sizeof(value), hipMemcpyDeviceToHost));
  REQUIRE(value == expected);

  unsigned int count = 0;
  HIP_CHECK(hipModuleGetFunctionCount(&count, module));
  REQUIRE(count == 1);
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "read_module"));
  void* args[] = {&output};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(&value, output, sizeof(value), hipMemcpyDeviceToHost));
  REQUIRE(value == expected * 3 + 1);

  HIP_CHECK_ERROR(hipModuleGetFunction(&kernel, module, "missing_kernel"), hipErrorNotFound);
  HIP_CHECK_ERROR(hipModuleGetGlobal(&global, &size, module, "missing_global"), hipErrorNotFound);
}
}  // namespace

// Run with COMGR exposing amd_comgr_get_data_pointer to cover borrowed output,
// and with an older COMGR lacking that optional API to cover the copying path.
// Each load keeps three independent modules alive; survivors must remain usable
// after another module releases its code object and cached metadata/symbols.
HIP_TEST_CASE(Unit_hipModuleLoad_Compressed_Lifetime) {
  HIP_CHECK(hipFree(nullptr));
  enum class LoadMode { File, Memory, MemoryEx };
  const auto mode = GENERATE(LoadMode::File, LoadMode::Memory, LoadMode::MemoryEx);
  const bool reverse = GENERATE(false, true);
  CAPTURE(static_cast<int>(mode), reverse);

  constexpr auto filename = "compressed_module.code";
  const auto image = LoadModuleIntoBuffer(filename);
  REQUIRE(image.size() >= 4);
  REQUIRE(std::memcmp(image.data(), "CCOB", 4) == 0);
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, sizeof(int));

  for (int iteration = 0; iteration < 8; ++iteration) {
    CAPTURE(iteration);
    LoadedModules loaded;
    std::array<int, 3> expected{};
    for (size_t i = 0; i < loaded.modules.size(); ++i) {
      auto& module = loaded.modules[i];
      if (mode == LoadMode::File) {
        HIP_CHECK(hipModuleLoad(&module, filename));
      } else if (mode == LoadMode::Memory) {
        HIP_CHECK(hipModuleLoadData(&module, image.data()));
      } else {
        HIP_CHECK(hipModuleLoadDataEx(&module, image.data(), 0, nullptr, nullptr));
      }
      REQUIRE(module != nullptr);
      VerifyCompressedModule(module, 17, output.ptr());

      hipDeviceptr_t global = nullptr;
      size_t size = 0;
      HIP_CHECK(hipModuleGetGlobal(&global, &size, module, "module_value"));
      expected[i] = 100 * (iteration + 1) + static_cast<int>(i);
      HIP_CHECK(hipMemcpy(global, &expected[i], sizeof(int), hipMemcpyHostToDevice));
    }

    // Distinct values catch accidental sharing of globals between module loads.
    for (size_t i = 0; i < loaded.modules.size(); ++i) {
      VerifyCompressedModule(loaded.modules[i], expected[i], output.ptr());
    }
    for (size_t i = 0; i < loaded.modules.size(); ++i) {
      const auto index = reverse ? loaded.modules.size() - i - 1 : i;
      HIP_CHECK(hipModuleUnload(loaded.modules[index]));
      loaded.modules[index] = nullptr;
      for (size_t j = 0; j < loaded.modules.size(); ++j) {
        if (loaded.modules[j] != nullptr) {
          VerifyCompressedModule(loaded.modules[j], expected[j], output.ptr());
        }
      }
    }
  }
}
