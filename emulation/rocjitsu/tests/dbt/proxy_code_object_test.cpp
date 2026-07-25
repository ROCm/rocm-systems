// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file proxy_code_object_test.cpp
/// @brief Prove the lazy proxy emitter produces a valid, loadable stand-in ELF.
///
/// The lazy path loads a proxy in place of a deferred B0 object: the proxy keeps
/// the kernel symbols and descriptors the runtime queries at load, but every
/// kernel body is the minimal s_endpgm stub (the proxy is never meant to execute;
/// the dispatch interceptor redirects to the translated kernel first). This test
/// confirms the proxy for the real gfx1250 vector_add fixture is a valid ELF that
/// still exposes the kernel descriptor and loads into simulator memory.

#ifndef HAS_GFX1250_DEVICE_KERNELS
#error "proxy_code_object_test.cpp requires a gfx1250-capable device compiler"
#endif

#include "../test_paths.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/proxy_code_object.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using namespace rocjitsu;

// Load the compiled gfx1250 vector_add fixture as a parsed code object.
const AmdGpuCodeObject *load_source(Executable &executable) {
  if (!executable.is_valid())
    return nullptr;
  return executable.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
}

// The proxy is a valid ELF that preserves the kernel descriptor symbol, so the
// runtime's load-time symbol/descriptor queries succeed while translation is
// deferred to first dispatch.
TEST(ProxyCodeObject, PreservesKernelDescriptorAndIsValid) {
  Executable executable(test::kernel_path("vector_add_gfx1250"));
  const auto *source = load_source(executable);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(source->kernel_descriptor_offset("vector_add"), 0u);

  ProxyCodeObject proxy = build_proxy_code_object(*source, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(proxy.ok()) << (proxy.diagnostics.empty() ? "proxy build failed without diagnostics"
                                                        : proxy.diagnostics.front().message);

  AmdGpuCodeObject proxy_obj(proxy.elf_bytes.data(), proxy.elf_bytes.size());
  ASSERT_TRUE(proxy_obj.is_valid()) << "emitted proxy ELF did not reparse as a valid code object";
  EXPECT_NE(proxy_obj.kernel_descriptor_offset("vector_add"), 0u)
      << "proxy lost the vector_add kernel descriptor symbol";
  EXPECT_FALSE(proxy_obj.text_sections().empty()) << "proxy has no .text section";
}

// The proxy is a distinct object from the source (its .text was replaced with
// stubs), not an identity copy.
TEST(ProxyCodeObject, ProxyDiffersFromSource) {
  Executable executable(test::kernel_path("vector_add_gfx1250"));
  const auto *source = load_source(executable);
  ASSERT_NE(source, nullptr);

  ProxyCodeObject proxy = build_proxy_code_object(*source, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(proxy.ok());

  const auto *source_image = reinterpret_cast<const uint8_t *>(source->image_data());
  const std::vector<uint8_t> source_bytes(source_image, source_image + source->image_size());
  EXPECT_NE(proxy.elf_bytes, source_bytes)
      << "proxy is byte-identical to the source; kernel bodies were not stubbed";
}

// The proxy loads into GPU memory and its kernel descriptor resolves to a mapped
// address, i.e. the runtime could obtain a stable kernel_object handle at load
// time while translation is deferred. (No dispatch — the proxy body never runs.)
TEST(ProxyCodeObject, LoadsIntoMemoryWithResolvableDescriptor) {
  Executable executable(test::kernel_path("vector_add_gfx1250"));
  const auto *source = load_source(executable);
  ASSERT_NE(source, nullptr);

  ProxyCodeObject proxy = build_proxy_code_object(*source, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(proxy.ok());
  AmdGpuCodeObject proxy_obj(proxy.elf_bytes.data(), proxy.elf_bytes.size());
  ASSERT_TRUE(proxy_obj.is_valid());

  const uint64_t kd_offset = proxy_obj.kernel_descriptor_offset("vector_add");
  ASSERT_NE(kd_offset, 0u);

  constexpr uint64_t kBaseAddr = 0x10000;
  amdgpu::GpuMemory memory("proxy_test");
  proxy_obj.load_to_memory(&memory, kBaseAddr);

  // The kernel_object address the runtime would hand a dispatch: the descriptor's
  // group_segment_fixed_size (a mapped descriptor field) must be readable. A proxy
  // descriptor is stripped to the minimal stub plan, so it reads back as 0.
  const uint64_t kernel_object = kBaseAddr + kd_offset;
  const uint32_t group_segment_fixed_size = memory.read32(kernel_object);
  EXPECT_EQ(group_segment_fixed_size, 0u)
      << "proxy descriptor group_segment_fixed_size should be the stub's zeroed plan";
}

} // namespace
