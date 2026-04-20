// libaiter_arch_spoof.so — LD_PRELOAD shim that does **two** things
// to make AITER's asm-kernel code path visible to Salmon's
// foreign-ISA transpiler on a gfx942 host:
//
//   1. Spoof gcnArchName.  AITER's C++ asm-kernel loader
//      (csrc/include/aiter_hip_common.h::get_gpu_arch) chooses which
//      hsa/<arch>/<kernel>.co binary to read based purely on the
//      device's reported gcnArchName.  On a gfx942 box that always
//      picks hsa/gfx942/* — which means the gfx950 .co files never
//      get loaded and Salmon is never engaged for AITER's asm path.
//      We intercept hipGetDeviceProperties{R0000,R0600} and rewrite
//      the returned gcnArchName in place.
//
//   2. Reroute __hipRegisterFatBinary -> hipModuleLoadData for the
//      specific case where the caller is wrapping a raw HSACO ELF
//      in the FatBinaryWrapper struct (this is exactly what AITER's
//      AiterAsmKernelFast::init does — see aiter_hip_common.h:166).
//      HIP's real __hipRegisterFatBinary silently accepts raw-ELF
//      wrappers but never calls the HSA code-object-load path, so
//      (a) ROCR's LoadCodeObject hook never sees the foreign-ISA
//      bytes, (b) Salmon's hipModuleLoadData intercept never fires,
//      and (c) hipGetFuncBySymbol returns NULL which launch rejects
//      with "invalid resource handle".  Instead we spot the ELF
//      prefix and call the public hipModuleLoadData — Salmon sees
//      the gfx950 bytes, transpiles them, and the ensuing
//      hipModuleGetFunction yields a valid function handle that
//      works on gfx942.
//
// The reroute is scoped to wrappers whose .magic equals the AITER/hipcc
// private sentinel 0x48495046 ("HIPF") *and* whose .binary points at
// an amdgpu ELF64 header (\x7fELF + e_machine==EM_AMDGPU).  Real
// hipcc-generated fat binaries start with __CLANG_OFFLOAD_BUNDLE__
// and fall through to the real HIP impl untouched; non-HIPF wrappers
// or ELFs from other architectures are left alone as well.  That
// keeps us out of the way of every other C++ Python extension in
// the process that also uses the registration ABI for its device
// code.
//
// Important properties
// --------------------
//
//   * No fallback / silent-skip path — every reroute is logged once
//     (the first HSACO we catch prints a diagnostic), and a
//     hipModuleLoadData failure returns nullptr from
//     __hipRegisterFatBinary so AITER's AITER_CHECK(module !=
//     nullptr) aborts with a loud message rather than silently
//     proceeding with a broken kernel handle.
//
//   * Bounded state — we keep two small unordered_maps (opaque
//     handle -> hipModule_t, hostFn -> hipFunction_t) protected by a
//     single mutex, both wrapped in Meyers singletons so they can
//     never be used before construction (no static-init-order
//     fiasco even if another preloaded constructor calls our hooks).
//     Registration happens at AITER op-test startup; we deliberately
//     do NOT add a process-exit drain (no
//     ``__attribute__((destructor))``) — see the long comment near
//     the ``state()`` Meyers singleton for why (short version: the
//     shim is first in LD_PRELOAD and therefore last to tear down,
//     by which time libamdhip64 has already released the chunks
//     backing our boxed handles, so any work there trips the heap).
//     Leaking the bookkeeping at exit is the correct trade-off; the
//     kernel reclaims address space and GPU resources anyway.
//
//   * No interference with the rest of Salmon — we never call any
//     Salmon APIs directly.  We call **public** HIP entry points
//     (hipModuleLoadData, hipModuleGetFunction, hipModuleUnload)
//     which Salmon already intercepts via LD_PRELOAD, so the
//     transpile path is unchanged.
//
//   * Startup self-check — a ``constructor`` priority hook asks the
//     dynamic linker to resolve each of our exported versioned
//     symbols via dlsym(RTLD_DEFAULT, ...) and compares the returned
//     pointer to our own address.  A mismatch means a later
//     LD_PRELOAD entry (or the version-script DAG being wrong)
//     shadowed our export — we abort loudly instead of pretending
//     to be active.
//
// Build
// -----
//   make           (uses /opt/rocm by default; override ROCM= for
//                  another HIP install)
//
// Usage
// -----
//   AITER_CORPUS_SPOOF_ARCH=gfx950
//   LD_PRELOAD=libaiter_arch_spoof.so:libhsa-runtime64.so.1:libsalmon_intercept.so
//       python my_aiter_test.py
//
// The shim must be early in LD_PRELOAD so its dlsym(RTLD_NEXT, ...)
// resolves to the real HIP impl and not back to itself.

#include <dlfcn.h>
#include <elf.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_deprecated.h>

// hip_runtime_api.h installs a `#define hipGetDeviceProperties
// hipGetDevicePropertiesR0600` so that source compiled against
// modern HIP transparently resolves to the R0600 entry point.  We
// define both R0000 and R0600 explicitly below — the macro would
// rewrite our R0600 definition into a redefinition of itself, so
// drop it.  Callers that compile through *this* shim's source
// already use the explicit names.
#ifdef hipGetDeviceProperties
#undef hipGetDeviceProperties
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

// Every exported entry point below gets an explicit default-visibility
// annotation; combined with -fvisibility=hidden in the Makefile this
// means we only expose the HIP hooks we meant to expose, not every
// incidental namespace-less helper the compiler emits.
#define AITER_SHIM_EXPORT __attribute__((visibility("default")))

namespace {

using hipGetDevicePropertiesR0000_t =
    hipError_t (*)(hipDeviceProp_tR0000* prop, int device);
using hipGetDevicePropertiesR0600_t =
    hipError_t (*)(hipDeviceProp_tR0600* prop, int device);

// The FatBinaryWrapper layout below is private to clang/HIP (no
// public header exports it).  0x48495046 ("HIPF") is the sentinel
// clang writes into the .magic field for every hipcc-emitted fat
// binary as well as AITER's hand-rolled wrapper around raw HSACO
// bytes.  We *require* that sentinel before even looking at
// .binary so non-HIPF wrappers from other toolchains (e.g.
// CUDA/nvcc's 0x466243b1) fall through to the real HIP impl
// untouched.
constexpr uint32_t kFatBinMagicHIPF = 0x48495046u;

// AMDGPU ELF e_machine — as written by llvm/clang for any HSACO
// we might see.  See LLVM's include/llvm/BinaryFormat/ELF.h
// (EM_AMDGPU = 224).
constexpr uint16_t kEmAmdgpu = 224u;

struct FatBinaryWrapper {
  uint32_t magic;      // 0x48495046 ("HIPF") for AITER + hipcc
  uint32_t version;
  const void* binary;
  intptr_t _pad;
};

using register_fatbin_t = void* (*)(const FatBinaryWrapper*);
using unregister_fatbin_t = void (*)(void*);
using register_function_t = void (*)(void*,
                                     const void*,
                                     const char*,
                                     const char*,
                                     int,
                                     void*,
                                     void*,
                                     void*,
                                     void*,
                                     void*);
using get_func_by_symbol_t = hipError_t (*)(hipFunction_t*, const void*);

// ---------------------------------------------------------------
// Real-symbol resolution.  std::atomic wrappers + std::call_once
// give us a data-race-free lazy init — the benign "two threads both
// read nullptr and both do the resolve" race of the plain-pointer
// globals would work in practice but is technically UB.
// std::atomic<T*> load/CAS is free on x86 and cheap on aarch64, so
// this costs us nothing on the hot path.
template <typename Real>
class LazyRealSymbol {
 public:
  explicit LazyRealSymbol(const char* name) : name_(name), ptr_(nullptr) {}

  Real get() {
    Real p = ptr_.load(std::memory_order_acquire);
    if (p != nullptr) return p;
    std::call_once(once_, [this]() {
      Real resolved = resolve();
      ptr_.store(resolved, std::memory_order_release);
    });
    return ptr_.load(std::memory_order_acquire);
  }

  const char* name() const { return name_; }

 private:
  Real resolve();
  const char* name_;
  std::atomic<Real> ptr_;
  std::once_flag once_;
};

// AITER_CORPUS_SPOOF_ARCH controls what we rewrite gcnArchName to.
// We snapshot it once on first use; the value is a sticky property
// of the process and an inconsistent mid-run change would just
// confuse triage.
const char* spoof_target() {
  static const char* cached = []() -> const char* {
    const char* v = std::getenv("AITER_CORPUS_SPOOF_ARCH");
    if (v == nullptr || v[0] == '\0') return nullptr;
    std::fprintf(stderr,
                 "[aiter_arch_spoof] active: every "
                 "hipGetDeviceProperties().gcnArchName will be rewritten "
                 "to %s (set AITER_CORPUS_SPOOF_ARCH= to disable)\n",
                 v);
    std::fflush(stderr);
    return v;
  }();
  return cached;
}

void rewrite_arch_name(char* dst, size_t cap) {
  const char* spoof = spoof_target();
  if (spoof == nullptr) return;
  // ``cap`` comes from ``sizeof(hipDeviceProp_t::gcnArchName)`` at
  // the call sites below — both R0000 and R0600 define it as a
  // 256-byte array, so cap is always > 0 at runtime.  The check is
  // belt-and-braces for a future HIP ABI that might hand us a
  // zero-length field: dropping the write is still safer than a
  // null-dst deref.
  if (cap == 0) return;
  // gcnArchName is a 256-byte char array per HIP's public ABI.  We
  // null-clear the entire field before writing the spoof, so AITER's
  // ::find(':') / ::substr logic sees a clean prefix and no stale
  // tail bytes leak through.
  size_t n = std::strlen(spoof);
  if (n >= cap) n = cap - 1;
  std::memset(dst, 0, cap);
  std::memcpy(dst, spoof, n);
  // One-shot log so we can prove the hook fired at least once per
  // process from the stderr alone — without spamming for every
  // AITER/torch/HIP internal property query.  If the corresponding
  // "active" line prints but this line never does, the versioned
  // symbol wasn't actually resolved to us (see arch_spoof.ver).
  static std::atomic<bool> announced{false};
  if (!announced.exchange(true, std::memory_order_relaxed)) {
    std::fprintf(stderr,
                 "[aiter_arch_spoof] first rewrite fired: gcnArchName "
                 "-> %s (subsequent rewrites silent)\n",
                 spoof);
    std::fflush(stderr);
  }
}

// Open libamdhip64 explicitly the first time we need a real HIP
// symbol.  RTLD_NEXT alone is not enough: AITER's C++ ops invoke
// hipGetDeviceProperties from constructors that fire during the
// initial ``import aiter`` — at that point libamdhip64 may not be
// in the process image yet, RTLD_NEXT walks the link map past us
// and finds nothing, dlsym returns NULL, and the runtime crashes.
// dlopen-ing here forces HIP to load and gives us a stable handle
// to query.  We try the SONAME and the unversioned name; the
// former is what every modern ROCm install actually exports.
void* hip_handle() {
  static void* h = []() -> void* {
    for (const char* name :
         {"libamdhip64.so.7", "libamdhip64.so.6",
          "libamdhip64.so.5", "libamdhip64.so"}) {
      void* p = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
      if (p != nullptr) {
        std::fprintf(stderr,
                     "[aiter_arch_spoof] dlopen(%s) succeeded\n", name);
        std::fflush(stderr);
        return p;
      }
    }
    std::fprintf(stderr,
                 "[aiter_arch_spoof] could not dlopen any "
                 "libamdhip64.so.* — hipGetDeviceProperties spoof "
                 "will be inactive\n");
    std::fflush(stderr);
    return nullptr;
  }();
  return h;
}

template <typename Real>
Real LazyRealSymbol<Real>::resolve() {
  // Try RTLD_NEXT first: cheapest path when libamdhip64 is already
  // in the link map past us (typical once any HIP call has run).
  // Fall back to a dlopen'd handle on the SONAME for the early
  // import-time call where RTLD_NEXT walks past us into nothing.
  // Lastly try RTLD_DEFAULT so a host that pre-loaded libamdhip64
  // with RTLD_GLOBAL still resolves.  Any path that succeeds wins;
  // if all three fail we report the original RTLD_NEXT error and
  // let the caller decide — an honest fatal (hipErrorNotSupported)
  // is strictly better than a silent pass-through that would mask
  // a missing HIP runtime.
  if (void* p = dlsym(RTLD_NEXT, name_)) return reinterpret_cast<Real>(p);
  const char* err_next = dlerror();

  if (void* h = hip_handle()) {
    if (void* p = dlsym(h, name_)) return reinterpret_cast<Real>(p);
  }
  if (void* p = dlsym(RTLD_DEFAULT, name_)) return reinterpret_cast<Real>(p);

  std::fprintf(stderr,
               "[aiter_arch_spoof] could not resolve real %s "
               "(RTLD_NEXT err: %s)\n",
               name_, err_next ? err_next : "<null>");
  std::fflush(stderr);
  return nullptr;
}

LazyRealSymbol<hipGetDevicePropertiesR0000_t>& real_R0000() {
  static LazyRealSymbol<hipGetDevicePropertiesR0000_t> s(
      "hipGetDevicePropertiesR0000");
  return s;
}
LazyRealSymbol<hipGetDevicePropertiesR0600_t>& real_R0600() {
  static LazyRealSymbol<hipGetDevicePropertiesR0600_t> s(
      "hipGetDevicePropertiesR0600");
  return s;
}
LazyRealSymbol<register_fatbin_t>& real_register_fatbin() {
  static LazyRealSymbol<register_fatbin_t> s("__hipRegisterFatBinary");
  return s;
}
LazyRealSymbol<unregister_fatbin_t>& real_unregister_fatbin() {
  static LazyRealSymbol<unregister_fatbin_t> s("__hipUnregisterFatBinary");
  return s;
}
LazyRealSymbol<register_function_t>& real_register_function() {
  static LazyRealSymbol<register_function_t> s("__hipRegisterFunction");
  return s;
}
LazyRealSymbol<get_func_by_symbol_t>& real_get_func_by_symbol() {
  static LazyRealSymbol<get_func_by_symbol_t> s("hipGetFuncBySymbol");
  return s;
}

// -------------------------------------------------------------------
// Fat-binary registration reroute.
//
// AITER's AiterAsmKernelFast::init (aiter_hip_common.h) wraps raw
// HSACO ELF bytes in a ``struct FatBinaryWrapper{magic=0x48495046}``
// and passes it to __hipRegisterFatBinary.  On a mismatched-ISA
// binary (gfx950 bytes on a gfx942 host) the real HIP impl never
// reaches the HSA code-object-load path, so Salmon's
// hipModuleLoadData hook never sees the foreign-ISA bytes, and
// hipGetFuncBySymbol later returns NULL which trips
// "invalid resource handle" at launch.
//
// We detect that case by (a) asserting the wrapper magic equals the
// HIPF sentinel and (b) sniffing the ELF header at .binary[0..20]
// for an AMDGPU-class ELF.  Either check failing means we hand the
// wrapper back to the real HIP impl unchanged.  Subsequent
// __hipRegisterFunction calls on our opaque handle are serviced
// from hipModuleGetFunction, and hipGetFuncBySymbol returns the
// cached hipFunction_t.

// State shared across the reroute hooks.  Wrapped in a Meyers
// singleton so construction happens on first use from hook code —
// never before, never in an order relative to other translation-unit
// statics.  This rules out the static-initialisation-order fiasco
// that namespace-scope objects with non-trivial constructors would
// otherwise risk if another LD_PRELOAD entry's constructor called
// our hooks before our globals had been constructed.
struct ShimState {
  std::mutex mtx;
  // Opaque handles we mint for AITER-style registrations.  Value
  // is the hipModule_t we got back from hipModuleLoadData.  Keys
  // are pointers we allocate ourselves; they cannot collide with
  // handles produced by the real HIP impl because those are
  // internal structures owned by libamdhip64.
  std::unordered_map<void*, hipModule_t> opaque_to_hipmod;
  // hostFn -> kernel function handle registered via
  // __hipRegisterFunction.  Populated by our reroute, queried by
  // hipGetFuncBySymbol.  Multiple kernels in the same module share
  // one hipModule_t but each gets its own entry keyed on its
  // unique hostFn pointer (AITER uses ``static_cast<void*>(this)``,
  // which is unique per AiterAsmKernelFast instance).
  std::unordered_map<const void*, hipFunction_t> hostfn_to_func;
};
ShimState& state() {
  static ShimState s;
  return s;
}

// True iff ``data`` points to at least the first 20 bytes of a
// little-endian ELF64 header whose ``e_machine`` field is
// EM_AMDGPU.  We cannot validate the buffer length in general
// because FatBinaryWrapper has no size field — the caller is
// expected to hand us a self-consistent HSACO or a bog-standard
// clang-offload-bundle wrapped in the same struct.  The AMDGPU
// class check is the strongest sniff that stays read-only in the
// first page of the payload: every AITER/hipcc HSACO is a signed
// ELF64 amdgpu; every clang-offload-bundle fails the magic match
// at offset 0 (`__CLANG_OFFLOAD_BUNDLE__` starts with '_').
bool looks_like_amdgpu_hsaco(const void* data) {
  if (data == nullptr) return false;
  const auto* b = static_cast<const uint8_t*>(data);
  // e_ident[EI_MAG0..EI_MAG3] = "\x7fELF"
  if (b[EI_MAG0] != ELFMAG0) return false;
  if (b[EI_MAG1] != ELFMAG1) return false;
  if (b[EI_MAG2] != ELFMAG2) return false;
  if (b[EI_MAG3] != ELFMAG3) return false;
  // e_ident[EI_CLASS] = ELFCLASS64 — every AMDGPU code object is
  // 64-bit.  Rejecting 32-bit here also weeds out any stray
  // payloads from other toolchains.
  if (b[EI_CLASS] != ELFCLASS64) return false;
  if (b[EI_DATA] != ELFDATA2LSB) return false;
  // e_machine is a 16-bit little-endian field immediately after the
  // 16-byte e_ident[] array.  We use ``offsetof(Elf64_Ehdr, ...)`` so
  // the constant is self-documenting and the compiler enforces it
  // against the system <elf.h> layout.  Reading as two bytes avoids
  // alignment concerns on a pointer we don't own the provenance of.
  constexpr size_t kOffEMachine = offsetof(Elf64_Ehdr, e_machine);
  static_assert(sizeof(Elf64_Ehdr::e_machine) == 2,
                "ELF64 e_machine is expected to be a 16-bit field");
  uint16_t e_machine =
      static_cast<uint16_t>(b[kOffEMachine]) |
      (static_cast<uint16_t>(b[kOffEMachine + 1]) << 8);
  return e_machine == kEmAmdgpu;
}

void log_reroute_once(const char* kind) {
  static std::atomic<bool> announced{false};
  if (announced.exchange(true, std::memory_order_relaxed)) return;
  std::fprintf(stderr,
               "[aiter_arch_spoof] first %s reroute fired: raw HSACO "
               "ELF detected in FatBinaryWrapper -> hipModuleLoadData "
               "(Salmon will see it); subsequent reroutes silent\n",
               kind);
  std::fflush(stderr);
}

// No process-exit drain.  Earlier revisions added a
// __attribute__((destructor)) that called hipModuleUnload (or
// just delete'd our boxed handles), reasoning that matching the
// module load/unload counts might satisfy a debug-build leak
// check.  Both shapes observably corrupted the heap on exit —
// "double free or corruption" in the first variant, "corrupted
// size vs. prev_size in fastbins" in the second — because
// destructor order at process exit is the reverse of load order,
// libaiter_arch_spoof.so is first in LD_PRELOAD (so last to tear
// down), and by the time our destructor runs libamdhip64's own
// destructors have already released the chunks backing the
// opaque handles AITER handed us.  Freeing them again (or even
// just indexing the map state, which allocates in std::atomic /
// unordered_map internals) trips the heap.  The kernel reclaims
// the process's address space and any GPU resources on exit
// regardless, so leaking the boxed-handle bookkeeping is the
// correct trade-off here.  If a well-behaved shutdown path is
// ever needed (e.g. for a long-lived embedded host), it belongs
// as a Python atexit hook in _bootstrap.py that runs while the
// interpreter and HIP are still fully alive — not a C++
// destructor at the very end of library teardown.

}  // namespace

extern "C" {

// Modern (ROCm >= 6.0) ABI.  AITER built against ROCm 7's HIP
// header resolves the unversioned name to this symbol via the
// macro at hip_runtime_api.h:103, so this is the usual hot path.
AITER_SHIM_EXPORT
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop,
                                       int device) {
  auto fn = real_R0600().get();
  if (fn == nullptr) return hipErrorNotSupported;
  hipError_t rc = fn(prop, device);
  if (rc == hipSuccess && prop != nullptr) {
    rewrite_arch_name(prop->gcnArchName, sizeof(prop->gcnArchName));
  }
  return rc;
}

// Legacy (pre-ROCm 6.0) ABI.  Different struct layout; gcnArchName
// is at a different offset, so we use the typed field access.
AITER_SHIM_EXPORT
hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop,
                                       int device) {
  auto fn = real_R0000().get();
  if (fn == nullptr) return hipErrorNotSupported;
  hipError_t rc = fn(prop, device);
  if (rc == hipSuccess && prop != nullptr) {
    rewrite_arch_name(prop->gcnArchName, sizeof(prop->gcnArchName));
  }
  return rc;
}

AITER_SHIM_EXPORT
void* __hipRegisterFatBinary(const FatBinaryWrapper* data) {
  // Require the HIPF sentinel *and* an AMDGPU ELF64 header before
  // rerouting.  Every other shape — ROCm's own handled fat binary,
  // an nvcc-style wrapper, an empty or malformed pointer — goes
  // straight to the real impl.
  if (data != nullptr &&
      data->magic == kFatBinMagicHIPF &&
      looks_like_amdgpu_hsaco(data->binary)) {
    // Raw HSACO wrapped for fake-fat-binary registration.  Route
    // through the public module API so Salmon's hipModuleLoadData
    // hook can transpile it.
    log_reroute_once("__hipRegisterFatBinary");
    hipModule_t mod = nullptr;
    hipError_t rc = hipModuleLoadData(&mod, data->binary);
    if (rc != hipSuccess) {
      // Propagate as a registration failure — AITER's init does
      // AITER_CHECK(module != nullptr, ...) which aborts loudly.
      std::fprintf(stderr,
                   "[aiter_arch_spoof] hipModuleLoadData failed: %s "
                   "(%d) — returning nullptr from "
                   "__hipRegisterFatBinary so AITER's AITER_CHECK "
                   "trips clearly\n",
                   hipGetErrorString(rc),
                   static_cast<int>(rc));
      std::fflush(stderr);
      return nullptr;
    }
    // Allocate our own opaque handle — the pointer itself is
    // cheap and unique for the life of the registration.  It
    // doubles as the key into opaque_to_hipmod.
    void* handle = new hipModule_t(mod);
    {
      auto& s = state();
      std::lock_guard<std::mutex> g(s.mtx);
      s.opaque_to_hipmod.emplace(handle, mod);
    }
    return handle;
  }
  auto fn = real_register_fatbin().get();
  if (fn == nullptr) return nullptr;
  return fn(data);
}

AITER_SHIM_EXPORT
void __hipRegisterFunction(void* module,
                           const void* hostFn,
                           const char* deviceFunction,
                           const char* deviceName,
                           int threadLimit,
                           void* tid,
                           void* bid,
                           void* blockDim,
                           void* gridDim,
                           void* wSize) {
  // Concurrency contract: AITER's fat-binary registration path is
  // driven from a single thread during ``import aiter``/module
  // initialisation (clang emits the __hipRegisterFatBinary / For
  // Function pair into a static ctor for the wrapping .so, and that
  // ctor runs serialised by the dynamic loader's per-library lock).
  // We therefore allow the map lookup here to release the mutex
  // before calling hipModuleGetFunction — a concurrent
  // __hipUnregisterFatBinary from another thread could in principle
  // race and invalidate ``hipmod`` between lookup and call, but
  // AITER never does that.  ROCm's hipModuleGetFunction is itself
  // re-entrancy-safe and does not call back into our hooks, so
  // holding the lock across it would only risk deadlocks against
  // future HIP versions without buying real safety today.  If a
  // multi-threaded caller ever shows up, tighten this here.
  hipModule_t hipmod = nullptr;
  {
    auto& s = state();
    std::lock_guard<std::mutex> g(s.mtx);
    auto it = s.opaque_to_hipmod.find(module);
    if (it != s.opaque_to_hipmod.end()) hipmod = it->second;
  }
  if (hipmod != nullptr) {
    // One of ours — resolve the kernel via hipModuleGetFunction
    // and cache the (hostFn -> func) mapping that AITER's
    // hipGetFuncBySymbol call will later lookup.
    hipFunction_t f = nullptr;
    hipError_t rc = hipModuleGetFunction(&f, hipmod, deviceName);
    if (rc != hipSuccess) {
      std::fprintf(stderr,
                   "[aiter_arch_spoof] hipModuleGetFunction(%s) "
                   "failed: %s (%d) — launch will fail at "
                   "hipGetFuncBySymbol\n",
                   deviceName ? deviceName : "<null>",
                   hipGetErrorString(rc),
                   static_cast<int>(rc));
      std::fflush(stderr);
      return;
    }
    auto& s = state();
    std::lock_guard<std::mutex> g(s.mtx);
    s.hostfn_to_func[hostFn] = f;
    return;
  }
  auto fn = real_register_function().get();
  if (fn == nullptr) return;
  fn(module, hostFn, deviceFunction, deviceName,
     threadLimit, tid, bid, blockDim, gridDim, wSize);
}

AITER_SHIM_EXPORT
void __hipUnregisterFatBinary(void* module) {
  hipModule_t hipmod = nullptr;
  bool ours = false;
  {
    auto& s = state();
    std::lock_guard<std::mutex> g(s.mtx);
    auto it = s.opaque_to_hipmod.find(module);
    if (it != s.opaque_to_hipmod.end()) {
      hipmod = it->second;
      ours = true;
      s.opaque_to_hipmod.erase(it);
    }
  }
  if (ours) {
    // Best effort — a failure here would only leak a module; we
    // still want to free the boxed handle.  The error is loud so
    // triage can notice leaks without needing extra tooling.
    hipError_t rc = hipModuleUnload(hipmod);
    if (rc != hipSuccess) {
      std::fprintf(stderr,
                   "[aiter_arch_spoof] hipModuleUnload failed: %s "
                   "(%d) — module leaked\n",
                   hipGetErrorString(rc),
                   static_cast<int>(rc));
      std::fflush(stderr);
    }
    delete static_cast<hipModule_t*>(module);
    return;
  }
  auto fn = real_unregister_fatbin().get();
  if (fn == nullptr) return;
  fn(module);
}

AITER_SHIM_EXPORT
hipError_t hipGetFuncBySymbol(hipFunction_t* out, const void* hostFn) {
  if (out != nullptr) {
    auto& s = state();
    std::lock_guard<std::mutex> g(s.mtx);
    auto it = s.hostfn_to_func.find(hostFn);
    if (it != s.hostfn_to_func.end()) {
      *out = it->second;
      return hipSuccess;
    }
  }
  auto fn = real_get_func_by_symbol().get();
  if (fn == nullptr) return hipErrorNotSupported;
  return fn(out, hostFn);
}

}  // extern "C"

// -------------------------------------------------------------------
// Startup self-check.  Runs as a GCC constructor — after all
// translation-unit statics in *this* library are initialised (fine:
// the Meyers singletons here are function-local) but *before* any
// HIP / AITER code gets a chance to call our hooks.
//
// We ask the dynamic linker to resolve each exported entry via
// dlsym(RTLD_DEFAULT, ...) and compare the returned address to our
// own address for that symbol.  A mismatch means the linker is
// resolving past us to the real libamdhip64 — e.g. because the
// version-script DAG does not match the node the caller's
// ``nm -D --with-symbol-versions`` reports, or because a later
// LD_PRELOAD entry shadowed ours.  In that case the rest of the
// shim silently does nothing and runs of the corpus runner would
// mis-report "native"-style results under "legacy"/"salmon".  We
// refuse to run in that state.
namespace {

struct ExportCheck {
  const char* name;
  void* shim_addr;
};

void verify_export_wins(const ExportCheck& e) {
  // RTLD_DEFAULT walks the global search list *from the start* —
  // which is exactly what an undefined-versioned reference in a
  // third-party module uses.  If our export is not the winner here,
  // it won't be the winner for AITER either.
  void* resolved = dlsym(RTLD_DEFAULT, e.name);
  if (resolved == nullptr) {
    const char* err = dlerror();
    std::fprintf(stderr,
                 "[aiter_arch_spoof] startup self-check: dlsym(%s) "
                 "returned NULL: %s — aborting; something has gone "
                 "very wrong with LD_PRELOAD ordering\n",
                 e.name, err ? err : "<no error>");
    std::fflush(stderr);
    std::abort();
  }
  if (resolved != e.shim_addr) {
    std::fprintf(stderr,
                 "[aiter_arch_spoof] startup self-check FAILED: "
                 "dlsym(RTLD_DEFAULT, \"%s\") resolved to %p, but "
                 "our export is at %p.  Another LD_PRELOAD entry "
                 "has shadowed us — the runner's results would "
                 "silently be wrong.  Aborting.\n",
                 e.name, resolved, e.shim_addr);
    std::fflush(stderr);
    std::abort();
  }
}

__attribute__((constructor))
void aiter_arch_spoof_self_check() {
  // Only verify the exports whose interception is load-bearing for
  // the corpus runner.  hipGetDevicePropertiesR0000 is legacy and
  // may not have a default-version undefined reference in any
  // modern AITER build; skipping it keeps the self-check silent on
  // pure-ROCm-7 installs while still catching the common failure
  // modes.
  const ExportCheck checks[] = {
      {"hipGetDevicePropertiesR0600",
       reinterpret_cast<void*>(&hipGetDevicePropertiesR0600)},
      {"__hipRegisterFatBinary",
       reinterpret_cast<void*>(&__hipRegisterFatBinary)},
      {"__hipRegisterFunction",
       reinterpret_cast<void*>(&__hipRegisterFunction)},
      {"__hipUnregisterFatBinary",
       reinterpret_cast<void*>(&__hipUnregisterFatBinary)},
      {"hipGetFuncBySymbol",
       reinterpret_cast<void*>(&hipGetFuncBySymbol)},
  };
  for (const auto& c : checks) verify_export_wins(c);
  std::fprintf(stderr,
               "[aiter_arch_spoof] startup self-check: all %zu "
               "exported hooks win dlsym(RTLD_DEFAULT, ...) — OK\n",
               sizeof(checks) / sizeof(checks[0]));
  std::fflush(stderr);
}

}  // namespace
