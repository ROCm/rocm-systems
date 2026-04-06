//===- hotswap_comgr_client.cpp - dlopen binding to COMGR -----------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap_comgr_client.hpp"
#include <cstdio>
#include <dlfcn.h>
#include <mutex>

namespace rocr {
namespace hotswap {

namespace {

using comgr_hotswap_rewrite_fn = int (*)(
    const void *elf_data, size_t elf_size,
    const char *source_isa_name, const char *target_isa_name,
    void **out_elf, size_t *out_elf_size);

std::once_flag g_comgr_init_flag;
void* g_comgr_lib = nullptr;
comgr_hotswap_rewrite_fn g_comgr_rewrite_fn = nullptr;

void init_comgr() {
  g_comgr_lib = dlopen("libamd_comgr.so", RTLD_LAZY | RTLD_LOCAL);
  if (!g_comgr_lib) {
    g_comgr_lib = dlopen("libamd_comgr.so.2", RTLD_LAZY | RTLD_LOCAL);
  }
  if (!g_comgr_lib)
    return;

  g_comgr_rewrite_fn = reinterpret_cast<comgr_hotswap_rewrite_fn>(
      dlsym(g_comgr_lib, "amd_comgr_hotswap_rewrite"));
  if (!g_comgr_rewrite_fn) {
    dlclose(g_comgr_lib);
    g_comgr_lib = nullptr;
  }
}

} // anonymous namespace

bool ComgrHotswapAvailable() {
  std::call_once(g_comgr_init_flag, init_comgr);
  return g_comgr_rewrite_fn != nullptr;
}

int ComgrHotswapRewrite(const void *elf_data, size_t elf_size,
                        const char *source_isa, const char *target_isa,
                        void **out_elf, size_t *out_elf_size) {
  std::call_once(g_comgr_init_flag, init_comgr);
  if (!g_comgr_rewrite_fn) {
    if (out_elf)
      *out_elf = nullptr;
    if (out_elf_size)
      *out_elf_size = 0;
    return -1;
  }
  return g_comgr_rewrite_fn(elf_data, elf_size, source_isa, target_isa,
                            out_elf, out_elf_size);
}

} // namespace hotswap
} // namespace rocr
