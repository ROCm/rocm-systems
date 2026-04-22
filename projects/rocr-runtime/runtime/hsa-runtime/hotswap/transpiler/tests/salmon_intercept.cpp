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
using hipFunction_t = void*;
using hipStream_t = void*;

using hipJitOption = int;
using hipDriverProcAddressQueryResult = int;
using hipModuleLoadData_t = hipError_t (*)(hipModule_t*, const void*);
using hipModuleLoadDataEx_t = hipError_t (*)(hipModule_t*, const void*,
                                              unsigned int, hipJitOption*,
                                              void**);
using hipGetProcAddress_t = hipError_t (*)(const char*, void**, int, uint64_t,
                                            hipDriverProcAddressQueryResult*);
using hipModuleLaunchKernel_t = hipError_t (*)(
    hipFunction_t, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int,
    hipStream_t, void**, void**);
// `hipFuncGetAttribute(int *value, hipFunction_attribute_t attrib, hipFunction_t f)`
// queries per-function attributes.  The enum value we care about is
// `HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0` — HIP reads it
// directly out of the kernel descriptor's `max_flat_workgroup_size`
// field (the same field the Salmon raiser's phantom-lane fallback
// keys on), so querying it at launch time is the lowest-cost way to
// reconstruct which projection `raiser.cpp` chose.  Spelled as `int`
// here so this shim stays header-free (same convention as the rest
// of the file).
using hipFuncGetAttribute_t =
    hipError_t (*)(int* /*value*/, int /*attrib*/, hipFunction_t /*f*/);
using rocr_salmon_patch_elf_t = int (*)(void*, size_t, const char*);

// HIP error-code constants we use when the intercept refuses a call
// rather than forwarding it.  The HIP header's `enum hipError_t`
// assigns `hipErrorInvalidConfiguration = 9`; we spell the value as an
// `int` constant here to avoid needing the HIP header at compile time
// (this shim is deliberately header-free so it can build against any
// HIP installation — see the "Build:" comment at the top of the file).
static constexpr int kHipErrorInvalidConfiguration = 9;

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

// hipModuleLaunchKernel resolution.  Populated by the RTLD_NEXT path in
// `init()` (the PLT route used by e.g. `compare_correctness`) and by
// the hipGetProcAddress hook (the runtime route used by Triton /
// hipGetProcAddress-resolving runtimes).  `resolve_real_launch_kernel()`
// prefers the proc-address variant when available for the same
// HIP-API-versioning reason the load_data resolver does.
static hipModuleLaunchKernel_t g_real_hipModuleLaunchKernel = nullptr;
static hipModuleLaunchKernel_t g_proc_hipModuleLaunchKernel = nullptr;

// hipFuncGetAttribute resolution — same three-path scheme as the
// launcher / loader pointers.  We don't hook this function; we just
// call through to it from inside `hipModuleLaunchKernel` to query
// the kernel's `max_flat_workgroup_size` attribute and recover the
// projection choice the raiser made.  See the big comment block
// above `hipModuleLaunchKernel` below.
static hipFuncGetAttribute_t g_real_hipFuncGetAttribute = nullptr;
static hipFuncGetAttribute_t g_proc_hipFuncGetAttribute = nullptr;

static rocr_salmon_patch_elf_t g_patch_elf = nullptr;
static const char* g_target_isa = nullptr;

// Target wavefront width in threads, derived once at init() from
// `g_target_isa` and cached here.  `0` means "unknown" (no refusal);
// populated to 32 on gfx10xx+ / gfx11xx+ / gfx12xx+ and 64 on gfx9xx /
// gfx8xx / gfx7xx / gfx6xx per the AMDGPU ISA wave-size convention
// (FeatureWavefrontSize32 on RDNA1+, FeatureWavefrontSize64 on the
// CDNA / GCN5 and earlier families).  See `target_wave_size()` below.
static unsigned g_target_wave_size = 0;

// Whether the Salmon IR raiser is enabled for this process, captured
// from `HSA_HOTSWAP_IR_RAISER` at `init()`.  The partial-wave launch
// refusal below gates on this because the phantom-lane miscompile
// class is specific to the IR-raise-and-retarget path: native-lane
// runs (source ISA matches device ISA — no wave widening) and the
// legacy cross-widen path (separate machinery, not an IR-raise) both
// leave the `init_whole_wave` vs partial-block invariant undisturbed.
// Refusing launches those paths emit would be an over-refusal that
// blocks correct kernels, which is why we narrow the check to the
// mode where `WaveNativeProjection` actually participates — namely
// the IR-raise mode set by `compare_correctness`'s Salmon lane and
// by any external caller that opts in to Salmon via the same env
// var.
static bool g_ir_raiser_active = false;

// Opt-out for the partial-wave launch refusal.  When set to a non-
// empty / non-zero value, `hipModuleLaunchKernel` forwards calls with
// `blockDim < targetWaveSize` to the real HIP loader instead of
// returning `hipErrorInvalidConfiguration`.  This exists for kernels
// that legitimately launch with partial waves AND are provably safe
// (no cross-lane ops — `ds_bpermute`, `readlane`, `writelane`, DPP /
// permlane* / ds_swizzle reductions), where the phantom-lane miscompile
// class doesn't apply.  The intended workflow is:
//
//   1. A raise-time phantom-lane fallback catches kernels with
//      `max_flat_workgroup_size < targetWaveSize` and switches them
//      to `ModuloReplicationProjection` so phantom lanes stay
//      hardware-inactive — see `raiser.cpp`'s fallback block.
//   2. The runtime check below catches the REMAINING case: a kernel
//      compiled WITHOUT `__launch_bounds__` (so
//      `max_flat_workgroup_size` defaults to something large enough
//      to bypass the raise-time fallback) but launched at runtime
//      with `blockDim < targetWaveSize`.  In that regime the raise
//      chose `WaveNativeProjection`, `init_whole_wave` sets HW
//      EXEC=-1 at kernel entry, and the phantom lanes WILL
//      contribute garbage to any cross-lane collective the kernel
//      uses.  The correct fix is the user's responsibility: add
//      `__launch_bounds__(N)` to the source kernel and recompile so
//      the raise-time fallback engages.
//   3. Users who know their kernel is partial-wave-safe set
//      HSA_HOTSWAP_ALLOW_PARTIAL_WAVE=1 to bypass this runtime
//      refusal.  The env-var opt-out is explicit acknowledgement,
//      not a default fallback.
static bool g_allow_partial_wave_launch = false;

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

  // Derive the target wavefront width from `HSA_HOTSWAP_ISA_OVERRIDE`.
  // AMDGPU ISA convention: RDNA (gfx10xx+) subtargets default to
  // FeatureWavefrontSize32; CDNA / GCN5 and earlier
  // (gfx9xx / gfx8xx / gfx7xx / gfx6xx) default to
  // FeatureWavefrontSize64.  We use the first digit after the "gfx"
  // prefix because the numbering is monotonic within each family
  // (gfx942, gfx908, gfx90a → wave64; gfx1030, gfx1100, gfx1250 →
  // wave32).  A non-prefixed string leaves `g_target_wave_size = 0`,
  // which disables the partial-wave launch refusal below (principled:
  // don't refuse a launch when we can't confidently derive the target
  // wave width).
  g_target_wave_size = 0;
  if (std::strncmp(g_target_isa, "gfx", 3) == 0 && g_target_isa[3] != 0) {
    const char* digits = g_target_isa + 3;
    // Largest currently-modelled gfx numbering is gfx12xx — 4 digits.
    // Any future gfx13xx+ will land on this same wave32 branch; any
    // gfx9xx / gfx8xx / gfx7xx / gfx6xx number is <1000 and hits
    // wave64.  Using atol (not atoi) so 4-digit values don't overflow
    // on 16-bit int targets — paranoid since int is 32-bit on every
    // platform this shim builds on, but it's a free guarantee.
    long major = std::atol(digits);
    if (major >= 1000) {
      g_target_wave_size = 32;
    } else if (major > 0) {
      g_target_wave_size = 64;
    }
  }

  const char* allow_partial = std::getenv("HSA_HOTSWAP_ALLOW_PARTIAL_WAVE");
  if (allow_partial && allow_partial[0] && allow_partial[0] != '0') {
    g_allow_partial_wave_launch = true;
  }

  // The IR raiser gate for the partial-wave refusal (see the
  // `g_ir_raiser_active` comment above for the rationale).  Captured
  // once here because `compare_correctness`'s three-lane harness sets
  // the env var BEFORE forking each child and doesn't flip it mid-run,
  // so a cached read at intercept init suffices.
  const char* ir_raiser = std::getenv("HSA_HOTSWAP_IR_RAISER");
  g_ir_raiser_active = ir_raiser && ir_raiser[0] && ir_raiser[0] != '0';

  // Best-effort PLT resolution for hipModuleLaunchKernel, same pattern
  // as the load_data resolvers above.  Populating this here lets the
  // RTLD_NEXT path (`compare_correctness`, raw HIP programs) reach the
  // real launcher.  The Triton / hipGetProcAddress route populates the
  // `g_proc_hipModuleLaunchKernel` sibling pointer below when a caller
  // asks for the launch symbol.
  if (auto* fn = reinterpret_cast<hipModuleLaunchKernel_t>(
          dlsym(RTLD_NEXT, "hipModuleLaunchKernel"))) {
    g_real_hipModuleLaunchKernel = fn;
  } else {
    fprintf(stderr,
            "salmon_intercept: hipModuleLaunchKernel not visible via "
            "RTLD_NEXT (%s); will rely on the dlsym/hipGetProcAddress "
            "hooks instead\n",
            dlerror());
  }

  // hipFuncGetAttribute is used by `hipModuleLaunchKernel`'s gate to
  // query the kernel's `max_flat_workgroup_size` attribute at launch
  // time; see the big comment above `hipModuleLaunchKernel` for the
  // rationale.  Best-effort resolution here mirrors the launcher
  // resolution — callers using `hipGetProcAddress` populate
  // `g_proc_hipFuncGetAttribute` when they query the symbol.
  if (auto* fn = reinterpret_cast<hipFuncGetAttribute_t>(
          dlsym(RTLD_NEXT, "hipFuncGetAttribute"))) {
    g_real_hipFuncGetAttribute = fn;
  } else {
    fprintf(stderr,
            "salmon_intercept: hipFuncGetAttribute not visible via "
            "RTLD_NEXT (%s); partial-wave launch gate will conservatively "
            "refuse blockDim < target_wave_size launches without the "
            "max-threads attribute query to identify MODREP-projected "
            "kernels (dlsym/hipGetProcAddress hook may populate later)\n",
            dlerror());
  }

  fprintf(stderr,
          "salmon_intercept: active, target=%s (wave_size=%u, "
          "ir_raiser=%s)%s\n",
          g_target_isa, g_target_wave_size,
          g_ir_raiser_active ? "on" : "off",
          g_allow_partial_wave_launch
              ? ", partial-wave launches ALLOWED via "
                "HSA_HOTSWAP_ALLOW_PARTIAL_WAVE"
              : "");
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

// Resolve the real hipModuleLaunchKernel.  Same preference order as
// the load_data resolvers: proc-address path wins over the RTLD_NEXT
// path so we never call a stale / version-mismatched symbol.  Returns
// nullptr only when `init()` has never succeeded in populating either
// pointer — in which case the caller aborts, because a launch wrapper
// with no real launcher to forward to is broken by construction.
static hipModuleLaunchKernel_t resolve_real_launch_kernel() {
  if (g_proc_hipModuleLaunchKernel) return g_proc_hipModuleLaunchKernel;
  return g_real_hipModuleLaunchKernel;
}

// Resolve the real hipFuncGetAttribute.  Unlike the launcher, this
// pointer can legitimately be `nullptr` at call time (if neither
// `RTLD_NEXT` nor `hipGetProcAddress` has populated it yet) — the
// caller handles that by falling back to the conservative default
// policy (refuse the partial-wave launch).  See the MODREP-detection
// block comment on `hipModuleLaunchKernel` for why this is correct.
static hipFuncGetAttribute_t resolve_real_func_get_attribute() {
  if (g_proc_hipFuncGetAttribute) return g_proc_hipFuncGetAttribute;
  return g_real_hipFuncGetAttribute;
}

// HIP's `hipFunction_attribute_t` enum value for
// `HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`.  HIP exposes this as
// `enum hipFunction_attribute` element 0 — stable since HIP 1.0 and
// mirrored from the CUDA driver API's `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`.
// Hard-coded to 0 here to keep the shim header-free.
static constexpr int kHipFuncAttributeMaxThreadsPerBlock = 0;

// ─────────────────────────────────────────────────────────────────────────────
// hipModuleLaunchKernel — partial-wave launch refusal gate.
//
// This hook implements the runtime-side half of the Salmon phantom-lane
// safety contract.  See `raiser.cpp`'s phantom-lane fallback (the
// `max_flat_workgroup_size < targetWaveSize` branch that switches the
// projection from `WaveNativeProjection` to `ModuloReplicationProjection`)
// for the raise-time half: that catch fires when the statically-visible
// workgroup-size attribute is below the target wavefront width, forcing
// MODREP so phantom target lanes stay hardware-inactive.
//
// The raise-time catch is necessary but not sufficient.  A kernel
// compiled WITHOUT `__launch_bounds__(N)` has `max_flat_workgroup_size`
// default to the archit-wide maximum (e.g. 1024), so the raise-time
// fallback never engages, and `WaveNativeProjection` is chosen —
// `init_whole_wave` sets HW EXEC = -1 on every target wave so all 64
// hardware lanes execute the kernel body.  If the runtime launch then
// picks `blockDim < targetWaveSize` (e.g. `blockDim.x = 32` on gfx942
// wave64), the HARDWARE runs 32 real source threads plus 32 phantom
// lanes — and any cross-lane collective the kernel issues (DPP
// reduction, `ds_bpermute`, `readlane`, permlane16 butterfly, DS
// swizzle) pulls undef bits from the phantom lanes and mixes them into
// the collective's output on the real lanes.  The miscompile is
// silent: there is no crash, no wrong numerics at most elements — just
// corrupted collective outputs on lanes that happen to read from
// phantom-lane participants.
//
// This wrapper closes that runtime gap by refusing
// `hipModuleLaunchKernel` calls where
// `blockDim.x * blockDim.y * blockDim.z < targetWaveSize` AND the
// kernel was NOT already raised under `ModuloReplicationProjection`.
// The MODREP-or-not distinction matters: when the source kernel
// declares a small `__launch_bounds__(N)` (Triton kernels with
// `num_warps=1` do this by default, setting `max_flat_workgroup_size
// = N` in the kernel descriptor), the raiser's phantom-lane fallback
// in `raiser.cpp` has ALREADY engaged MODREP at lift time so no
// `init_whole_wave` is emitted — phantom target lanes stay
// hardware-inactive regardless of the runtime launch shape, and a
// partial-wave launch is correct by construction.  Refusing those
// kernels here would be an over-refusal that blocks correctly-raised
// kernels from running.
//
// We reconstruct the projection choice at launch time by querying
// HIP for the kernel's `MAX_THREADS_PER_BLOCK` attribute — HIP reads
// that directly out of the ELF kernel descriptor's
// `max_flat_workgroup_size` field, which is the exact same field
// `raiser.cpp`'s phantom-lane fallback keys on.  If
// `max_flat_workgroup_size < targetWaveSize`, the raiser chose MODREP
// and the launch is safe; otherwise it chose WaveNative, which in
// combination with a runtime partial-wave launch produces the
// phantom-lane miscompile this gate is designed to catch.
//
// The user has two principled workarounds when the refusal fires:
//
//   1. Add `__launch_bounds__(N)` to the source kernel (with
//      `N < targetWaveSize`) and recompile so the raise-time
//      phantom-lane fallback engages `ModuloReplicationProjection`
//      for this launch shape.  This is the CORRECT fix for kernels
//      that use cross-lane primitives.
//   2. Set `HSA_HOTSWAP_ALLOW_PARTIAL_WAVE=1` in the environment
//      to disable this runtime check entirely.  This is the OPT-IN
//      ACCEPT-THE-RISK path for kernels that are provably
//      phantom-lane-safe at the ISA level (no cross-lane primitives
//      at all — pure per-lane VALU work with SPE-gated loads /
//      stores).  The kernel's correctness under partial-wave launch
//      in that case is provable by construction: phantom lanes
//      compute undef VGPRs locally, but every write to memory is
//      SPE-gated (see the load/store gating commit `ebe575dcdc`),
//      so the undef never externalizes.
//
// Refusing with `hipErrorInvalidConfiguration` (= 9 in the HIP error
// enum — the `InvalidConfiguration` path HIP already uses for "launch
// dims rejected" conditions such as oversubscription) gives the caller
// a standard HIP error to branch on rather than a process abort.  A
// stderr diagnostic spells out the refusal reason and both workarounds
// above so the user has a single place to read about the fix.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" hipError_t hipModuleLaunchKernel(
    hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra) {
  init();

  hipModuleLaunchKernel_t real = resolve_real_launch_kernel();
  if (!real) {
    fprintf(stderr, "salmon_intercept: no real hipModuleLaunchKernel\n");
    std::abort();
  }

  // Phantom-lane launch gate.  Uses uint64_t for the product to stay
  // immune to unsigned-int overflow on pathologically large launches —
  // even though three 32-bit multiplies overflowing is already a
  // launch-size validation failure HIP would reject downstream, we
  // don't want to silently bypass the check by wrapping to a tiny
  // product.
  //
  // The `g_ir_raiser_active` guard narrows the check to the Salmon
  // raise path (the only mode where `WaveNativeProjection` +
  // `init_whole_wave` +`HSA_HOTSWAP_IR_RAISER` combine to produce the
  // phantom-lane miscompile class).  Native-lane runs and the legacy
  // pre-salmon cross-widen path don't share that invariant, so a
  // blanket refusal would over-reject correct kernels on those paths.
  if (g_ir_raiser_active && g_target_wave_size > 0 &&
      !g_allow_partial_wave_launch) {
    const uint64_t block_threads =
        static_cast<uint64_t>(blockDimX) *
        static_cast<uint64_t>(blockDimY) *
        static_cast<uint64_t>(blockDimZ);
    if (block_threads < static_cast<uint64_t>(g_target_wave_size)) {
      // MODREP-detection shortcut: if the kernel descriptor's
      // `max_flat_workgroup_size` attribute is below the target
      // wave size, the raiser has ALREADY engaged
      // `ModuloReplicationProjection` — no `init_whole_wave` was
      // emitted, phantom target lanes stay hardware-inactive, and a
      // partial-wave launch is correct by construction.  Pass
      // through without refusing.  See the block comment above for
      // the full projection-reconstruction rationale.
      //
      // We only consult `hipFuncGetAttribute` when it's available;
      // when it isn't (the attribute-resolver returned `nullptr` in
      // init() and the Triton-proc-address path hasn't populated it
      // either), we conservatively fall through to refusing, which
      // is the safer default: over-refusal is a test-harness
      // annoyance, but under-refusal is a silent miscompile.
      if (auto* fn_attr = resolve_real_func_get_attribute()) {
        int max_threads = 0;
        hipError_t attr_err =
            fn_attr(&max_threads, kHipFuncAttributeMaxThreadsPerBlock, f);
        if (attr_err == 0 && max_threads > 0 &&
            static_cast<unsigned>(max_threads) < g_target_wave_size) {
          // `max_flat_workgroup_size < targetWaveSize` → raiser
          // engaged MODREP → phantom-lane-safe.  Pass through.
          return real(f, gridDimX, gridDimY, gridDimZ,
                      blockDimX, blockDimY, blockDimZ,
                      sharedMemBytes, stream, kernelParams, extra);
        }
      }
      fprintf(stderr,
              "salmon_intercept: REFUSING hipModuleLaunchKernel: "
              "block dims %u x %u x %u = %llu thread(s) per block, "
              "target=%s (wave_size=%u), kernel's "
              "max_flat_workgroup_size does not satisfy the MODREP "
              "phantom-lane-safe condition (< %u) — partial-wave "
              "launch puts the kernel in the phantom-lane regime "
              "where `init_whole_wave` sets HW EXEC=-1 on a "
              "partially-filled wave, so any cross-lane primitive "
              "(DPP reductions, ds_bpermute, readlane/writelane, "
              "permlane16, ds_swizzle) reads undef from the unused "
              "lanes and silently miscompiles.  Principled fixes: "
              "(1) add `__launch_bounds__(%llu)` to the source "
              "kernel and recompile so the raise-time phantom-lane "
              "fallback engages ModuloReplicationProjection; (2) set "
              "HSA_HOTSWAP_ALLOW_PARTIAL_WAVE=1 if the kernel is "
              "provably cross-lane-free (pure per-lane VALU work).  "
              "Returning hipErrorInvalidConfiguration (=%d).\n",
              blockDimX, blockDimY, blockDimZ,
              static_cast<unsigned long long>(block_threads), g_target_isa,
              g_target_wave_size, g_target_wave_size,
              static_cast<unsigned long long>(block_threads),
              kHipErrorInvalidConfiguration);
      return kHipErrorInvalidConfiguration;
    }
  }

  return real(f, gridDimX, gridDimY, gridDimZ,
              blockDimX, blockDimY, blockDimZ,
              sharedMemBytes, stream, kernelParams, extra);
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
  } else if (std::strcmp(symbol, "hipModuleLaunchKernel") == 0) {
    // Route the launch symbol through our partial-wave refusal gate.
    // Same contract as the load_data redirects above: we stash the
    // real proc-address-resolved pointer for forward dispatch, and
    // hand the caller a pointer to our wrapper so every launch path
    // flows through the blockDim check.
    g_proc_hipModuleLaunchKernel =
        reinterpret_cast<hipModuleLaunchKernel_t>(*pfn);
    *pfn = reinterpret_cast<void*>(&hipModuleLaunchKernel);
    fprintf(stderr,
            "salmon_intercept: redirected hipGetProcAddress(\"%s\")\n",
            symbol);
  } else if (std::strcmp(symbol, "hipFuncGetAttribute") == 0) {
    // hipFuncGetAttribute is NOT hooked — we just stash the real
    // proc-address pointer so `resolve_real_func_get_attribute()`
    // can call through to it from inside
    // `hipModuleLaunchKernel`'s gate.  We leave `*pfn` pointing at
    // the real implementation so callers see no wrapper at all.
    g_proc_hipFuncGetAttribute = reinterpret_cast<hipFuncGetAttribute_t>(*pfn);
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
