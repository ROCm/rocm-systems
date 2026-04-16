// Salmon Layer 1: LD_PRELOAD shim for HIP integration.
//
// Intercepts hipModuleLoadData to patch ELF metadata (e_flags) before HIP's
// ISA compatibility check.  The MSGPACK metadata is left untouched so the
// HSA runtime can still detect the original ISA and trigger Salmon.
//
// This implements the same role as hip_fatbin.cpp in the legacy hotswap
// system, but as a standalone LD_PRELOAD library that works with any
// unmodified HIP build.
//
// Build:
//   g++ -shared -fPIC -O2 -o libsalmon_intercept.so salmon_intercept.cpp -ldl
//
// Usage:
//   LD_PRELOAD=./libsalmon_intercept.so \
//   LD_LIBRARY_PATH=$ROCR_BUILD/rocr/lib \
//   HSA_HOTSWAP_ISA_OVERRIDE=gfx942 \
//   HSA_HOTSWAP_IR_RAISER=1 \
//   HSA_HOTSWAP_RULES=/dev/null \
//   ./my_hip_program

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using hipError_t = int;
using hipModule_t = void*;

using hipModuleLoadData_t = hipError_t (*)(hipModule_t*, const void*);
using rocr_salmon_patch_elf_t = int (*)(void*, size_t, const char*);

static hipModuleLoadData_t g_real_hipModuleLoadData = nullptr;
static rocr_salmon_patch_elf_t g_patch_elf = nullptr;
static const char* g_target_isa = nullptr;

static void init() {
  static bool done = false;
  if (done) return;
  done = true;

  g_real_hipModuleLoadData = reinterpret_cast<hipModuleLoadData_t>(
      dlsym(RTLD_NEXT, "hipModuleLoadData"));
  if (!g_real_hipModuleLoadData) {
    fprintf(stderr, "salmon_intercept: cannot find real hipModuleLoadData: %s\n",
            dlerror());
    return;
  }

  g_patch_elf = reinterpret_cast<rocr_salmon_patch_elf_t>(
      dlsym(RTLD_DEFAULT, "rocr_salmon_patch_elf"));
  if (!g_patch_elf) {
    fprintf(stderr, "salmon_intercept: cannot find rocr_salmon_patch_elf "
                    "(is Salmon-enabled libhsa-runtime64.so loaded?): %s\n",
            dlerror());
    return;
  }

  g_target_isa = std::getenv("HSA_HOTSWAP_ISA_OVERRIDE");
  if (!g_target_isa || !g_target_isa[0]) {
    fprintf(stderr, "salmon_intercept: HSA_HOTSWAP_ISA_OVERRIDE not set\n");
    return;
  }

  fprintf(stderr, "salmon_intercept: active, target=%s\n", g_target_isa);
}

static bool is_elf(const void* data) {
  auto* b = static_cast<const uint8_t*>(data);
  return b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F';
}

static size_t elf_size(const void* data) {
  auto* b = static_cast<const uint8_t*>(data);
  if (b[4] != 2) return 0;  // must be ELF64
  uint64_t shoff;
  uint16_t shentsz, shnum;
  std::memcpy(&shoff, b + 40, 8);
  std::memcpy(&shentsz, b + 58, 2);
  std::memcpy(&shnum, b + 60, 2);
  return static_cast<size_t>(shoff + shentsz * shnum);
}

extern "C" hipError_t hipModuleLoadData(hipModule_t* module,
                                         const void* image) {
  init();

  if (!g_real_hipModuleLoadData) {
    fprintf(stderr, "salmon_intercept: no real hipModuleLoadData\n");
    return -1;
  }

  if (!g_patch_elf || !g_target_isa || !is_elf(image)) {
    return g_real_hipModuleLoadData(module, image);
  }

  size_t sz = elf_size(image);
  if (sz == 0) {
    return g_real_hipModuleLoadData(module, image);
  }

  // Copy the ELF so we can patch it (hipModuleLoadData takes const void*)
  auto* buf = static_cast<uint8_t*>(std::malloc(sz));
  std::memcpy(buf, image, sz);

  std::string target = std::string("amdgcn-amd-amdhsa--") + g_target_isa;
  int rc = g_patch_elf(buf, sz, target.c_str());
  if (rc != 0) {
    fprintf(stderr, "salmon_intercept: PatchElfIsa failed, passing original\n");
    std::free(buf);
    return g_real_hipModuleLoadData(module, image);
  }

  fprintf(stderr, "salmon_intercept: patched e_flags for %s (%zu bytes)\n",
          g_target_isa, sz);

  hipError_t err = g_real_hipModuleLoadData(module, buf);
  std::free(buf);
  return err;
}
