// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/x86_v4_provider.h"

#include <algorithm>
#include <bit>
#include <cpuid.h>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <limits.h>
#include <stdexcept>
#include <string>

#ifndef ROCJITSU_X86_V4_BUILD_DIGEST
#error "ROCJITSU_X86_V4_BUILD_DIGEST must match the shared v4 provider"
#endif

namespace rocjitsu {
namespace {

uint64_t xcr0() noexcept {
  uint32_t low = 0, high = 0;
  __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
  return (uint64_t{high} << 32) | low;
}

// Capture a relative dladdr spelling before an application can change cwd.
const unsigned char image_anchor = 0;
std::array<char, PATH_MAX> library_origin{};

__attribute__((constructor)) void capture_library_origin() noexcept {
  Dl_info image{};
  if (!dladdr(&image_anchor, &image) || !image.dli_fname ||
      !realpath(image.dli_fname, library_origin.data()))
    library_origin.front() = '\0';
}

std::filesystem::path provider_path() {
  if (library_origin.front() == '\0')
    throw std::runtime_error("cannot locate librocjitsu.so for x86-v4 provider");
  return std::filesystem::path(library_origin.data()).parent_path() / "librocjitsu_x86_v4.so";
}

void validate_provider(const X86V4ProviderDescriptor *provider) {
  if (!provider || provider->abi_version != kX86V4ProviderAbi || !provider->build_digest ||
      std::strcmp(provider->build_digest, ROCJITSU_X86_V4_BUILD_DIGEST) != 0 ||
      !provider->bind_wmma_backend || !provider->exp.execute || !provider->log.execute ||
      std::any_of(provider->wmma_callbacks.begin(), provider->wmma_callbacks.end(),
                  [](Instruction::ExecuteFn callback) { return callback == nullptr; }))
    throw std::runtime_error("x86-v4 provider is incompatible with librocjitsu.so");
}

struct LoadedProvider {
  const X86V4ProviderDescriptor *descriptor = nullptr;
  void *handle = nullptr; // Retain successful handles until process exit.
  std::exception_ptr failure;

  LoadedProvider() noexcept {
    try {
      const std::filesystem::path path = provider_path();
      handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!handle) {
        const char *error = dlerror();
        throw std::runtime_error(std::string("cannot load x86-v4 provider: ") +
                                 (error ? error : "unknown loader error"));
      }
      void *symbol = dlsym(handle, "rj_x86_v4_provider_v1");
      if (!symbol)
        throw std::runtime_error("x86-v4 provider has no v1 entry point");
      static_assert(sizeof(GetX86V4Provider) == sizeof(symbol));
      descriptor = std::bit_cast<GetX86V4Provider>(symbol)();
      validate_provider(descriptor);
    } catch (...) {
      descriptor = nullptr;
      if (handle) {
        dlclose(handle);
        handle = nullptr;
      }
      failure = std::current_exception();
    }
  }
};

} // namespace

bool cpu_and_os_support_x86_v4() noexcept {
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  constexpr unsigned leaf1_required =
      (1u << 12) | (1u << 22) | (1u << 27) | (1u << 28) | (1u << 29);
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || (ecx & leaf1_required) != leaf1_required)
    return false;
  if ((xcr0() & 0x6u) != 0x6u)
    return false;
  constexpr unsigned leaf7_v3 = (1u << 3) | (1u << 5) | (1u << 8);
  constexpr unsigned leaf7_v4 = (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) ||
      (ebx & (leaf7_v3 | leaf7_v4)) != (leaf7_v3 | leaf7_v4))
    return false;
  if ((xcr0() & 0xe6u) != 0xe6u)
    return false;
  return __get_cpuid(0x80000001u, &eax, &ebx, &ecx, &edx) && (ecx & (1u << 5));
}

const X86V4ProviderDescriptor *resolve_x86_v4_provider(bool required) {
  // Even the getter is v4 code. Never load/initialize it before qualification.
  if (!cpu_and_os_support_x86_v4()) {
    if (required)
      throw std::runtime_error("x86-v4 provider requires CPU and OS AVX-512 support");
    return nullptr;
  }
  static const LoadedProvider loaded;
  if (!loaded.descriptor && required)
    std::rethrow_exception(loaded.failure);
  return loaded.descriptor;
}

} // namespace rocjitsu
