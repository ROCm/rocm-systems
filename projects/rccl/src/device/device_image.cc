/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Embeds the device image (RCCL_DEVICE_IMAGE, the bundled per-arch device.elf
// files) and registers it once, with every kernel and variable the -fgpu-rdc
// host objects list in llvm_offload_entries. This is the registration
// clang-linker-wrapper would generate; RCCL links its device code itself.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

// llvm::offloading::EntryTy, version 1.
struct OffloadEntry {
  uint64_t reserved;
  uint16_t version;
  uint16_t kind;
  uint32_t flags;
  void* addr;
  char* name;
  uint64_t size;
  uint64_t data;
  void* auxAddr;
};
static_assert(sizeof(OffloadEntry) == 56, "unexpected offload entry layout");

constexpr uint16_t kOffloadKindHip = 4;
enum : uint32_t { kGlobal = 0, kManaged = 1, kTypeMask = 0x7, kExtern = 1 << 3, kConstant = 1 << 4 };

struct FatbinWrapper {
  uint32_t magic;
  uint32_t version;
  const void* binary;
  void* unused;
};

}  // namespace

extern "C" {
void** __hipRegisterFatBinary(const void* data);
void __hipUnregisterFatBinary(void** modules);
void __hipRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction,
                           const char* deviceName, unsigned int threadLimit, void* tid, void* bid,
                           void* blockDim, void* gridDim, int* wSize);
void __hipRegisterVar(void** modules, void* var, char* hostVar, const char* deviceVar, int ext,
                      size_t size, int constant, int global);
void __hipRegisterManagedVar(void* hipModule, void** pointer, void* initValue, const char* name,
                             size_t size, unsigned align);

extern const OffloadEntry __start_llvm_offload_entries[] __attribute__((weak, visibility("hidden")));
extern const OffloadEntry __stop_llvm_offload_entries[] __attribute__((weak, visibility("hidden")));
extern const char rccl_device_image[];
}

asm(".section .hip_fatbin,\"a\",@progbits\n"
    ".p2align 12\n"
    "rccl_device_image:\n"
    ".incbin \"" RCCL_DEVICE_IMAGE "\"\n"
    ".previous\n");

namespace {

__attribute__((section(".hipFatBinSegment"), used, aligned(8)))
const FatbinWrapper wrapper = {0x48495046 /* "HIPF" */, 1, rccl_device_image, nullptr};

void** handle;

[[noreturn]] void fail(const char* what, const char* name) {
  fprintf(stderr, "RCCL: cannot register the device image: %s%s\n", what, name);
  abort();
}

void unregisterImage() { __hipUnregisterFatBinary(handle); }

__attribute__((constructor(101))) void registerImage() {
  if (__start_llvm_offload_entries == __stop_llvm_offload_entries)
    fail("no llvm_offload_entries; the compiler predates the version 1 offload entry format", "");
  handle = __hipRegisterFatBinary(&wrapper);
  for (const OffloadEntry* e = __start_llvm_offload_entries; e != __stop_llvm_offload_entries; ++e) {
    if (e->reserved != 0 || e->version != 1) fail("unknown offload entry version for ", e->name);
    if (e->kind != kOffloadKindHip) continue;
    if (e->size == 0) {
      __hipRegisterFunction(handle, e->addr, e->name, e->name, -1, nullptr, nullptr, nullptr, nullptr, nullptr);
      continue;
    }
    switch (e->flags & kTypeMask) {
      case kGlobal:
        __hipRegisterVar(handle, e->addr, e->name, e->name, (e->flags & kExtern) != 0, e->size,
                         (e->flags & kConstant) != 0, 0);
        break;
      case kManaged:
        __hipRegisterManagedVar(handle, static_cast<void**>(e->auxAddr), e->addr, e->name, e->size,
                                static_cast<unsigned>(e->data));
        break;
      default:
        fail("surface and texture entries are not supported: ", e->name);
    }
  }
  atexit(unregisterImage);
}

}  // namespace
