// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <limits>

namespace {

constexpr int kHipSuccess = 0;
constexpr int kHipMemcpyDeviceToHost = 2;

uint32_t *g_fake_device_counters = nullptr;
size_t g_fake_device_counter_bytes = 0;
int g_device_synchronize_calls = 0;
uintptr_t g_next_module = 1;
int g_module_load_data_calls = 0;
int g_module_unload_calls = 0;
void *g_last_module = nullptr;
const uint8_t *g_last_module_image = nullptr;
size_t g_last_module_image_size = 0;

size_t infer_elf_image_size(const void *image) {
  if (image == nullptr)
    return 0;

  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image, sizeof(ehdr));
  if (std::memcmp(ehdr.e_ident, ELFMAG, SELFMAG) != 0 || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_shentsize != sizeof(Elf64_Shdr) || ehdr.e_shnum == 0)
    return 0;

  const uint64_t section_headers_bytes = ehdr.e_shnum * sizeof(Elf64_Shdr);
  if (ehdr.e_shoff > std::numeric_limits<uint64_t>::max() - section_headers_bytes)
    return 0;
  uint64_t image_size = ehdr.e_shoff + section_headers_bytes;

  if (ehdr.e_phnum != 0) {
    if (ehdr.e_phentsize != sizeof(Elf64_Phdr))
      return 0;
    const uint64_t program_headers_bytes = ehdr.e_phnum * sizeof(Elf64_Phdr);
    if (ehdr.e_phoff > std::numeric_limits<uint64_t>::max() - program_headers_bytes)
      return 0;
    image_size = std::max(image_size, ehdr.e_phoff + program_headers_bytes);
  }

  return image_size <= std::numeric_limits<size_t>::max() ? static_cast<size_t>(image_size) : 0;
}

} // namespace

extern "C" {

int hipModuleLoadData(void **module, const void *image) {
  if (module == nullptr)
    return 999;

  ++g_module_load_data_calls;
  g_last_module = reinterpret_cast<void *>(g_next_module++);
  g_last_module_image = static_cast<const uint8_t *>(image);
  g_last_module_image_size = infer_elf_image_size(image);
  *module = g_last_module;
  return kHipSuccess;
}

int hipModuleUnload(void *module) {
  ++g_module_unload_calls;
  if (module == g_last_module)
    g_last_module = nullptr;
  return kHipSuccess;
}

int hipMalloc(void **ptr, size_t size) {
  g_fake_device_counter_bytes = size;
  g_fake_device_counters = static_cast<uint32_t *>(std::calloc(1, size));
  *ptr = g_fake_device_counters;
  return g_fake_device_counters == nullptr ? 999 : kHipSuccess;
}

int hipFree(void *ptr) {
  std::free(ptr);
  if (ptr == g_fake_device_counters) {
    g_fake_device_counters = nullptr;
    g_fake_device_counter_bytes = 0;
  }
  return kHipSuccess;
}

int hipMemset(void *dst, int value, size_t size) {
  std::memset(dst, value, size);
  return kHipSuccess;
}

int hipMemcpy(void *dst, const void *src, size_t size, int kind) {
  if (kind != kHipMemcpyDeviceToHost)
    return 999;
  std::memcpy(dst, src, size);
  return kHipSuccess;
}

int hipDeviceSynchronize() {
  ++g_device_synchronize_calls;
  return kHipSuccess;
}

const char *hipGetErrorString(int) { return "fake hip error"; }

uint32_t *rocfuzz_fake_device_counters() { return g_fake_device_counters; }

size_t rocfuzz_fake_device_counter_bytes() { return g_fake_device_counter_bytes; }

int rocfuzz_fake_device_synchronize_calls() { return g_device_synchronize_calls; }

void rocfuzz_fake_hip_reset() {
  g_module_load_data_calls = 0;
  g_module_unload_calls = 0;
  g_last_module = nullptr;
  g_last_module_image = nullptr;
  g_last_module_image_size = 0;
}

int rocfuzz_fake_hip_module_load_data_calls() { return g_module_load_data_calls; }

int rocfuzz_fake_hip_module_unload_calls() { return g_module_unload_calls; }

const uint8_t *rocfuzz_fake_hip_last_module_image() { return g_last_module_image; }

size_t rocfuzz_fake_hip_last_module_image_size() { return g_last_module_image_size; }

} // extern "C"
