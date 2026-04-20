// Salmon Layer 1: LD_PRELOAD shim for HIP integration.
//
// Patches ELF metadata (e_flags) on every code-object load so HIP's
// per-device ISA compatibility check accepts a foreign-ISA binary, while
// leaving the MSGPACK metadata untouched so the Salmon-enabled libhsa can
// still detect the original ISA and trigger raising.
//
// Three different code paths reach into HIP for module loads, so we
// intercept all three:
//
//   1. hipModuleLoadData / hipModuleLoadDataEx — the obvious PLT route,
//      taken by anything that links against libamdhip64 normally
//      (compare_correctness, raw HIP test programs).
//   2. hipGetProcAddress — newer HIP runtimes return per-API-version
//      function pointers through this, and a caller (e.g. Triton's AMD
//      backend) may then call those pointers directly without ever
//      touching the PLT-visible loader symbols.  We rewrite the pointer
//      it would have returned for the loader symbols to point at our
//      wrapper.
//   3. dlsym(handle, "hipGetProcAddress") with an RTLD_LOCAL handle —
//      this is how Triton resolves hipGetProcAddress itself.  An
//      RTLD_LOCAL dlsym lookup is scoped to that handle's lib and never
//      consults LD_PRELOAD'd modules, so without a hook here the chain
//      above never engages.  We hook dlsym, replace just that one
//      lookup with our hipGetProcAddress wrapper, and let everything
//      else pass through.
//
// This implements the same role as hip_fatbin.cpp in the legacy hotswap
// system, but as a standalone LD_PRELOAD library that works with any
// unmodified HIP build, JIT or not.
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
//
// For Triton-via-Python callers the Salmon-enabled libhsa-runtime64.so.1
// must also be in LD_PRELOAD (and ahead of this shim), so its
// rocr_salmon_patch_elf symbol is in global scope when init() runs.

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using hipError_t = int;
using hipModule_t = void*;

using hipJitOption = int;
using hipDriverProcAddressQueryResult = int;
using hipModuleLoadData_t = hipError_t (*)(hipModule_t*, const void*);
using hipModuleLoadDataEx_t = hipError_t (*)(hipModule_t*, const void*,
                                              unsigned int, hipJitOption*,
                                              void**);
using hipGetProcAddress_t = hipError_t (*)(const char*, void**, int, uint64_t,
                                            hipDriverProcAddressQueryResult*);
using rocr_salmon_patch_elf_t = int (*)(void*, size_t, const char*);

// "Real" implementations populated via RTLD_NEXT dlsym (the symbols visible
// to the dynamic linker).  These cover any caller that uses the PLT — e.g.
// compare_correctness, raw HIP test programs.
static hipModuleLoadData_t g_real_hipModuleLoadData = nullptr;
static hipModuleLoadDataEx_t g_real_hipModuleLoadDataEx = nullptr;
static hipGetProcAddress_t g_real_hipGetProcAddress = nullptr;

// Runtimes such as Triton bypass the PLT entirely and resolve HIP entry
// points through hipGetProcAddress.  We intercept that call and stash the
// "true" function pointers it returned here, then hand the caller a pointer
// to our wrapper instead.  Our wrapper prefers these proc-address pointers
// when available because they correctly track HIP API versioning.
static hipModuleLoadData_t g_proc_hipModuleLoadData = nullptr;
static hipModuleLoadDataEx_t g_proc_hipModuleLoadDataEx = nullptr;

static rocr_salmon_patch_elf_t g_patch_elf = nullptr;
static const char* g_target_isa = nullptr;

static void init() {
  static bool done = false;
  if (done) return;
  done = true;

  // Best-effort resolution of the PLT-visible HIP symbols.  These are needed
  // for callers that link against libamdhip64 normally (e.g.
  // compare_correctness).  When the runtime opens libamdhip64 with
  // RTLD_LOCAL (Triton, torch with newer ROCm), these lookups fail, but we
  // can still service those callers through the dlsym + hipGetProcAddress
  // hooks below.  So we log and continue rather than bailing.
  //
  // We only assign on success, so we never clobber pointers that the dlsym
  // hook may have populated before init() ran.
  if (auto* fn = reinterpret_cast<hipModuleLoadData_t>(
          dlsym(RTLD_NEXT, "hipModuleLoadData"))) {
    g_real_hipModuleLoadData = fn;
  } else {
    fprintf(stderr,
            "salmon_intercept: hipModuleLoadData not visible via RTLD_NEXT "
            "(%s); will rely on the dlsym/hipGetProcAddress hooks instead\n",
            dlerror());
  }

  if (auto* fn = reinterpret_cast<hipModuleLoadDataEx_t>(
          dlsym(RTLD_NEXT, "hipModuleLoadDataEx"))) {
    g_real_hipModuleLoadDataEx = fn;
  } else {
    fprintf(stderr,
            "salmon_intercept: hipModuleLoadDataEx not visible via RTLD_NEXT "
            "(%s); will rely on the dlsym/hipGetProcAddress hooks instead\n",
            dlerror());
  }

  if (auto* fn = reinterpret_cast<hipGetProcAddress_t>(
          dlsym(RTLD_NEXT, "hipGetProcAddress"))) {
    g_real_hipGetProcAddress = fn;
  } else {
    fprintf(stderr,
            "salmon_intercept: hipGetProcAddress not visible via RTLD_NEXT "
            "(%s); will be captured lazily through the dlsym hook\n",
            dlerror());
  }

  // The Salmon ELF patcher must be loaded — this is the whole point of the
  // shim.  Refuse to continue without it.
  g_patch_elf = reinterpret_cast<rocr_salmon_patch_elf_t>(
      dlsym(RTLD_DEFAULT, "rocr_salmon_patch_elf"));
  if (!g_patch_elf) {
    fprintf(stderr,
            "salmon_intercept: cannot find rocr_salmon_patch_elf "
            "(is Salmon-enabled libhsa-runtime64.so loaded?): %s — aborting\n",
            dlerror());
    std::abort();
  }

  g_target_isa = std::getenv("HSA_HOTSWAP_ISA_OVERRIDE");
  if (!g_target_isa || !g_target_isa[0]) {
    fprintf(stderr,
            "salmon_intercept: HSA_HOTSWAP_ISA_OVERRIDE not set — aborting\n");
    std::abort();
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

// Patch the ELF e_flags so HIP's ISA compatibility check accepts the image,
// while preserving the original ISA's MACH byte in a side channel so the
// Salmon-enabled libhsa can detect the true source ISA and trigger raising.
//
// The side channel is the 7-byte EI_PAD region of the ELF identification
// header (bytes 9..15), which is reserved by the ELF spec and required to
// be zero.  We use:
//
//   e_ident[9]  = 'S'                  // magic byte 0
//   e_ident[10] = 'L'                  // magic byte 1
//   e_ident[11] = <original MACH byte> // i.e. (old e_flags & 0xff)
//
// leaving e_ident[12..15] untouched (zero).  The loader-side hotswap hook
// in libhsa-runtime reads this on every code-object load and treats it as
// the authoritative original ISA when present.  This is needed because
// Tensile-style .co files carry no `amdhsa.target` in the MSGPACK
// metadata, so after `PatchElfIsa` overwrites e_flags and the
// NT_AMDGPU_ISA note there is no other channel left for ROCR to recover
// the original ISA from.  The side channel is idempotent — if the magic
// is already present we don't overwrite it, so repeated loads of the
// same buffer keep the same original MACH.
//
// Returns nullptr if the image is not a patchable ELF (caller should
// forward the original image to the underlying HIP loader).  Aborts the
// process if the image *is* an ELF but patching fails — propagating that
// as a "load success" would silently bypass the hotswap path and yield
// meaningless runtime errors elsewhere.
static uint8_t* patch_image(const void* image, size_t* out_size) {
  if (!g_patch_elf || !g_target_isa || !is_elf(image)) {
    return nullptr;
  }
  size_t sz = elf_size(image);
  if (sz == 0) {
    fprintf(stderr,
            "salmon_intercept: ELF size could not be determined; aborting\n");
    std::abort();
  }
  auto* buf = static_cast<uint8_t*>(std::malloc(sz));
  std::memcpy(buf, image, sz);

  // Stash the original MACH byte into e_ident[9..11] BEFORE PatchElfIsa
  // overwrites e_flags.  The ELF header is at least 52 bytes for any 64-bit
  // ELF (we already verified that via elf_size()), so offsets 9..11 are
  // always safe to access.
  uint8_t orig_mach = buf[48];
  if (buf[9] == 0 && buf[10] == 0) {
    buf[9] = 'S';
    buf[10] = 'L';
    buf[11] = orig_mach;
  }

  std::string target = std::string("amdgcn-amd-amdhsa--") + g_target_isa;
  int rc = g_patch_elf(buf, sz, target.c_str());
  if (rc != 0) {
    fprintf(stderr,
            "salmon_intercept: PatchElfIsa failed (rc=%d) for target=%s; "
            "aborting\n",
            rc, g_target_isa);
    std::free(buf);
    std::abort();
  }
  fprintf(stderr,
          "salmon_intercept: patched e_flags for %s (%zu bytes, "
          "orig_mach=0x%02x)\n",
          g_target_isa, sz, orig_mach);
  *out_size = sz;
  return buf;
}

// Resolve the underlying HIP entry point.  Prefer the proc-address-resolved
// pointer (Triton path) over the RTLD_NEXT one (PLT path) so we never call
// a stale or version-mismatched symbol.
static hipModuleLoadData_t resolve_real_load_data() {
  if (g_proc_hipModuleLoadData) return g_proc_hipModuleLoadData;
  return g_real_hipModuleLoadData;
}

static hipModuleLoadDataEx_t resolve_real_load_data_ex() {
  if (g_proc_hipModuleLoadDataEx) return g_proc_hipModuleLoadDataEx;
  return g_real_hipModuleLoadDataEx;
}

// NOTE: we deliberately do NOT free the patched buffer.  The hotswap layer
// inside libhsa-runtime keeps pointers into the ELF (its .note section, the
// MSGPACK metadata, and the kernel text) for the lifetime of the loaded
// module.  Freeing here trips a use-after-free that surfaces as random heap
// corruption later in the host process.  The leak is bounded by the number
// of distinct kernel modules loaded, which in practice is small.
extern "C" hipError_t hipModuleLoadData(hipModule_t* module,
                                         const void* image) {
  init();

  hipModuleLoadData_t real = resolve_real_load_data();
  if (!real) {
    fprintf(stderr, "salmon_intercept: no real hipModuleLoadData\n");
    std::abort();
  }

  size_t sz = 0;
  uint8_t* buf = patch_image(image, &sz);
  if (!buf) {
    return real(module, image);
  }
  return real(module, buf);
}

extern "C" hipError_t hipModuleLoadDataEx(hipModule_t* module,
                                           const void* image,
                                           unsigned int numOptions,
                                           hipJitOption* options,
                                           void** optionValues) {
  init();

  hipModuleLoadDataEx_t real = resolve_real_load_data_ex();
  if (!real) {
    fprintf(stderr, "salmon_intercept: no real hipModuleLoadDataEx\n");
    std::abort();
  }

  size_t sz = 0;
  uint8_t* buf = patch_image(image, &sz);
  if (!buf) {
    return real(module, image, numOptions, options, optionValues);
  }
  return real(module, buf, numOptions, options, optionValues);
}

// Triton (and other runtimes built against newer ROCm) resolve HIP entry
// points via hipGetProcAddress instead of dlsym/PLT.  We intercept it,
// stash the real pointers it returned, and hand callers a pointer to our
// wrapper for the loader symbols.
extern "C" hipError_t hipGetProcAddress(
    const char* symbol, void** pfn, int hipVersion, uint64_t hipFlags,
    hipDriverProcAddressQueryResult* symbolStatus) {
  init();

  if (!g_real_hipGetProcAddress) {
    fprintf(stderr,
            "salmon_intercept: hipGetProcAddress called but real symbol is "
            "missing\n");
    std::abort();
  }

  hipError_t err = g_real_hipGetProcAddress(symbol, pfn, hipVersion, hipFlags,
                                             symbolStatus);
  if (err != 0 /* hipSuccess */ || !symbol || !pfn || !*pfn) {
    return err;
  }

  if (std::strcmp(symbol, "hipModuleLoadDataEx") == 0) {
    g_proc_hipModuleLoadDataEx =
        reinterpret_cast<hipModuleLoadDataEx_t>(*pfn);
    *pfn = reinterpret_cast<void*>(&hipModuleLoadDataEx);
    fprintf(stderr,
            "salmon_intercept: redirected hipGetProcAddress(\"%s\")\n",
            symbol);
  } else if (std::strcmp(symbol, "hipModuleLoadData") == 0) {
    g_proc_hipModuleLoadData = reinterpret_cast<hipModuleLoadData_t>(*pfn);
    *pfn = reinterpret_cast<void*>(&hipModuleLoadData);
    fprintf(stderr,
            "salmon_intercept: redirected hipGetProcAddress(\"%s\")\n",
            symbol);
  }
  return err;
}

// ─────────────────────────────────────────────────────────────────────────────
// dlsym hook
//
// Some runtimes (notably Triton's AMD backend) dlopen libamdhip64.so with
// RTLD_LOCAL and then resolve every HIP symbol via dlsym(handle, name).  That
// dlsym call is scoped to the lib handle, so neither LD_PRELOAD'd symbols nor
// our hipGetProcAddress wrapper above would ever be reached without a hook
// here.  We intercept dlsym so that the very first lookup — the one for
// "hipGetProcAddress" — returns our wrapper; from there our wrapper rewrites
// every loader symbol the runtime asks for.
//
// We resolve the real dlsym with dlvsym() (libdl can be queried for a
// versioned symbol), avoiding the obvious dlsym(RTLD_NEXT, "dlsym") recursion.
// ─────────────────────────────────────────────────────────────────────────────

using dlsym_t = void* (*)(void*, const char*);

static dlsym_t resolve_real_dlsym() {
  static dlsym_t cached = nullptr;
  if (cached) return cached;

  // glibc on x86_64 / aarch64 ships dlsym in libc with version GLIBC_2.34
  // (since glibc 2.34 — Ubuntu 22.04+ / RHEL 9 / etc.) and previously in
  // libdl with version GLIBC_2.2.5.  Try both.
  cached = reinterpret_cast<dlsym_t>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34"));
  if (!cached) {
    cached = reinterpret_cast<dlsym_t>(
        dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
  }
  if (!cached) {
    fprintf(stderr,
            "salmon_intercept: could not resolve real dlsym via dlvsym; "
            "aborting (last dlerror: %s)\n",
            dlerror());
    std::abort();
  }
  return cached;
}

extern "C" void* dlsym(void* handle, const char* name) {
  dlsym_t real = resolve_real_dlsym();
  void* addr = real(handle, name);
  if (addr && name && std::strcmp(name, "hipGetProcAddress") == 0) {
    // Cache the real proc-address resolver so our wrapper can call it,
    // because the wrapper itself may not be reachable via RTLD_NEXT when
    // the loader is libamdhip64 opened with RTLD_LOCAL.
    if (!g_real_hipGetProcAddress) {
      g_real_hipGetProcAddress = reinterpret_cast<hipGetProcAddress_t>(addr);
    }
    fprintf(stderr,
            "salmon_intercept: redirected dlsym(\"hipGetProcAddress\")\n");
    return reinterpret_cast<void*>(&hipGetProcAddress);
  }
  return addr;
}
