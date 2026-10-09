/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * hip_capture.cpp — Hand-written capture shims for complex HIP APIs.
 *
 * Covers the MANUAL_CAPTURE_APIS set defined in gen_hrr_api_args.py:
 *   - Memcpy H2D variants with blob snapshotting (hipMemcpy, hipMemcpyAsync,
 *     hipMemcpyHtoD, hipMemcpyHtoDAsync, hipMemcpyWithStream)
 *   - Module load with code object snapshotting (hipModuleLoad*)
 *   - Module unload (hipModuleUnload) to drop the program hash cache entry
 *   - Kernel launch with arg introspection via kernel->signature()
 *   - Fat binary registration (__hipRegisterFatBinary)
 *   - Host memory registration (hipHostRegister / hipHostUnregister)
 *
 * Everything else (malloc/free, stream/event, memset, device sync, etc.)
 * is auto-generated in hip_capture_generated.cpp.
 *
 * All shims write hrr_args_* structs as event payloads (uniform format).
 * Kernel launch is the exception — it uses a variable-length binary payload
 * (the format defined in hrr_reader.h parse_kernel_launch).
 *
 * g_real_table, g_cap_table, g_compiler_installed are defined here (non-static)
 * so hip_capture_generated.cpp can extern them.
 *
 * Independence rule: zero dependency on hipamd/src/profiler/.
 */

#include "hip_capture.h"
#include "hip_capture_metadata.h"
#include "hip_capture_writer.h"

// hrr_api_args.h — for hrr_args_* struct types and hrr_api_id_t enum
#include "hrr/hrr_api_args.h"

// HIP runtime internals
#include "../hip_global.hpp"       // hip::asKernel()
#include "../hip_internal.hpp"
#include "hip/amd_detail/hip_api_trace.hpp"
#include "utils/flags.hpp"         // HIP_HRR_CAPTURE_OUTPUT flag

// ROCclr kernel introspection
#include "device/devkernel.hpp"    // amd::Kernel, KernelParameterDescriptor
#include "platform/kernel.hpp"     // amd::KernelSignature
#include "opencl/amdocl/cl_kernel.h"  // T_POINTER enum
#include "os/os.hpp"               // amd::Os::installExceptionHandlers()

// Fat binary format structs (ClangOffloadBundleUncompressedHeader, etc.)
#include "../hip_code_object.hpp"
#include "../hip_platform.hpp"   // PlatformState::Instance()

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <map>
#include <memory>
#include <new>
#include <tuple>
#include <unordered_map>
#include <vector>

static std::once_flag g_hrr_atexit_once;


// GetHipDispatchTable / GetHipCompilerDispatchTable
namespace hip {
const HipDispatchTable*         GetHipDispatchTable();
const HipCompilerDispatchTable* GetHipCompilerDispatchTable();
}

// ---------------------------------------------------------------------------
// Global tables — non-static: extern'd in hip_capture_generated.cpp
// ---------------------------------------------------------------------------

HipDispatchTable         g_real_table{};
HipDispatchTable         g_cap_table{};
std::atomic<bool>        g_installed{false};
std::atomic<bool>        g_table_built{false};  // guard for hip_capture_build_table()
std::atomic<bool>        g_cap_table_ready{false};  // g_cap_table fully populated

HipCompilerDispatchTable g_real_compiler_table{};
std::atomic<bool>        g_compiler_installed{false};  // guard for hip_capture_build_compiler_table()

// TLS dims saved by __hipPushCallConfiguration — used only as a fallback by
// hipLaunchByPtr when the exec stack is empty (the exec stack top() is the
// authoritative source for both the <<<>>> and legacy hipConfigureCall paths).
static thread_local dim3        g_pushed_grid{};
static thread_local dim3        g_pushed_block{};
static thread_local size_t      g_pushed_shared{};
static thread_local hipStream_t g_pushed_stream{};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// The kernel sets AT_SECURE for a set-user-ID, set-group-ID or file-capability
// exec, and for an LSM transition. HIP_HRR_CAPTURE_OUTPUT comes from whoever
// launched the process, so such a process must not write where it says, nor
// record its memory for them.
static bool hrr_secure_exec() {
  static const bool secure = hrr_cap::metadata::secure_exec();
  return secure;
}

// CLR's flag parser stores an exported empty variable as a single space, so a
// value that is only blanks counts as unset rather than as a directory named " ".
static bool hrr_capture_requested() {
  if (flagIsDefault(HIP_HRR_CAPTURE_OUTPUT)) return false;
  for (const char* p = HIP_HRR_CAPTURE_OUTPUT; *p != '\0'; ++p) {
    if (!std::isspace(static_cast<unsigned char>(*p))) return true;
  }
  return false;
}

bool hip_capture_enabled() { return hrr_capture_requested() && !hrr_secure_exec(); }

const char* hip_capture_output_dir() {
  return HIP_HRR_CAPTURE_OUTPUT;
}

// HIP_HRR_DEBUG_ARGS also enables provenance tracing: every H2D memcpy
// destination is logged so a kernel pointer arg can be matched to the copy that
// filled it. Routed through the CLR flags registry (flags.hpp) like every other
// HRR flag, so it is discoverable and honors AMD_LOG_LEVEL log routing.
static bool hrr_dbg_args_enabled() {
  return HIP_HRR_DEBUG_ARGS;
}

static void hrr_trace_h2d(const char* api, const void* dst, size_t sz) {
  if (hrr_dbg_args_enabled())
    LogPrintfInfo("[HRR h2d] %s dst=0x%llx size=%zu", api,
                  (unsigned long long)reinterpret_cast<uintptr_t>(dst),
                  sz);
}

// Wait for an async D2H copy before its host buffer is snapshotted. Once the
// writer has closed, for instance after capture stopped for lack of space, the
// snapshot would be dropped, so the copy is left asynchronous as the app asked.
static hipError_t sync_for_d2h_snapshot(hipStream_t stream) {
  if (!hrr_cap::writer::is_open()) return hipSuccess;
  return g_real_table.hipStreamSynchronize_fn(stream);
}

// Parse the extra[] sentinel format for packed kernarg buffers.
static bool parse_kernel_extra(void** extra, const void*& out_buf, size_t& out_size) {
  if (!extra) return false;
  if (extra[0] != HIP_LAUNCH_PARAM_BUFFER_POINTER) return false;
  if (extra[2] != HIP_LAUNCH_PARAM_BUFFER_SIZE)    return false;
  if (extra[4] != HIP_LAUNCH_PARAM_END)             return false;
  out_buf  = extra[1];
  out_size = *reinterpret_cast<const size_t*>(extra[3]);
  return out_buf != nullptr && out_size > 0;
}

// ---------------------------------------------------------------------------
// Kernel launch event serialization
//
// Binary layout of KERNEL_LAUNCH payload (matches hrr_reader.cpp parse_kernel_launch):
//
//   u64  stream_handle (recorded hipStream_t cast to uint64_t)
//   u16  name_len
//   u8[] kernel_name (name_len bytes, no NUL)
//   u64  co_hash_lo   (0 = unknown)
//   u64  co_hash_hi
//   u32[3] grid
//   u32[3] block
//   u32  shared_mem
//   u16  num_args
//   u16  num_snapshots (always 0)
//   for each arg:
//     u8   value_kind  (0=scalar, 1=pointer/gpu addr, 2=hidden,
//                       3=scalar/struct with embedded gpu pointer(s))
//     u16  size
//     u8[] data (size bytes)
//     if value_kind == 3:
//       u16   n_ptrs
//       u16[] ptr_offset (n_ptrs entries: byte offset of each 8-byte gpu ptr)
// ---------------------------------------------------------------------------

// Locate device pointers embedded inside a by-value kernel argument.
//
// KernelSignature marks an argument as a plain scalar (desc.type_ != T_POINTER)
// whenever the pointer is a *member* of a struct passed by value — e.g. the
// std::array<char*,N> that ATen's vectorized_elementwise_kernel takes through
// the <<<>>> path, or the ~720-byte ReduceOp config struct that mixes scalar
// shapes/strides with device pointers (input, output, and the global
// scratch/semaphore buffers used by multi-block reductions). The type metadata
// cannot say "bytes o..o+7 of this argument are an address", so the only robust
// way to find such pointers is to ask the runtime whether a candidate word is a
// live device allocation.
//
// `scan_embedded_ptr_offsets` produces the offset list for one launch, caching a
// per-word verdict (POINTER / SCALAR) keyed by (kernel, arg-index, size, offset)
// to avoid re-probing the runtime on every launch — that probe is a locked
// allocation-tree walk, and issuing one per candidate word of every by-value arg
// of every launch (millions on an LLM workload) both slows capture and perturbs
// the very timing/ordering HRR records.
//
// CRITICAL: a verdict is cached only from an *actual probe* (value >= 0x10000).
// A null/small word (a pointer field that is null on this launch, or a small
// scalar) is skipped WITHOUT caching, so a conditionally-populated pointer field
// — e.g. the multi-block reduction scratch/semaphore pointers, which are null on
// single-block launches — is still detected on the first launch where it is
// actually non-null, instead of being frozen as a scalar from an early launch.
// Device VAs are always huge, so a non-null embedded pointer never looks "small"
// and is therefore never skipped.
//
// Notes:
// - Scanning is byte-by-byte (skipping a full pointer width on a hit, since
//   pointers do not overlap) so a pointer at an unaligned offset inside a packed
//   struct is still found.
// - This is a heuristic: a large scalar/double whose bit pattern lands in a live
//   device VA is a false positive. Replay guards against corrupting such a
//   scalar by only overwriting a flagged word when the recorded value actually
//   resolves to a known allocation (see hip_playback.cpp).
// - Scope: only hipMemoryTypeDevice/Unified words are flagged. A pinned/host
//   (hipMemoryTypeHost) pointer embedded by value keeps its capture-time host VA
//   at replay (invalid in the replay process); translating such pointers is
//   intentionally out of scope.
enum class PtrVerdict : uint8_t { Pointer, Scalar };

// A cached per-word verdict. For a Scalar verdict we also remember the exact
// 8-byte value that was probed: a struct slot can legitimately hold a harmless
// scalar/garbage value on one launch and a real device pointer on a later one
// (e.g. the reused addresses[] slots in ATen's multi_tensor_apply metadata
// across launch waves). Freezing such an offset as "scalar" forever would leave
// the later real pointer unflagged and therefore untranslated at replay — a
// guaranteed GPU VM fault. So a Scalar verdict is trusted only while the word
// value is unchanged; any change forces a re-probe.
struct PtrVerdictEntry { PtrVerdict verdict; uint64_t probed_value; };

static void scan_embedded_ptr_offsets(const void* func_key, uint32_t arg_idx,
                                      const uint8_t* data, uint16_t size,
                                      std::vector<uint16_t>& offsets) {
  if (!data || !g_real_table.hipPointerGetAttributes_fn) return;
  // Per-word verdict cache. thread_local keeps the hot launch path lock-free;
  // verdicts are identical across threads so per-thread duplication is harmless.
  thread_local std::map<std::tuple<const void*, uint32_t, uint16_t, uint16_t>,
                        PtrVerdictEntry> verdicts;
  const bool cacheable = (func_key != nullptr);
  const hipError_t saved_cmd = hip::tls.last_command_error_;
  const hipError_t saved_err = hip::tls.last_error_;
  size_t j = 0;
  while (j + sizeof(void*) <= size) {
    const uint16_t off = static_cast<uint16_t>(j);
    uint64_t cand;
    std::memcpy(&cand, data + j, sizeof(cand));
    if (cacheable) {
      auto it = verdicts.find(std::make_tuple(func_key, arg_idx, size, off));
      if (it != verdicts.end()) {
        if (it->second.verdict == PtrVerdict::Pointer) {
          // Pointer verdicts are sticky: flagging an offset is harmless even if
          // it later holds a scalar, since replay only rewrites a flagged word
          // when its recorded value actually resolves to a live allocation.
          offsets.push_back(off);
          j += sizeof(void*);
          continue;
        }
        // Scalar verdict: trust it only while the value is unchanged. A changed
        // value may have transitioned from scalar/garbage to a real pointer.
        if (it->second.probed_value == cand) {
          j += 1;
          continue;
        }
        // else: value changed — fall through and re-probe.
      }
    }
    // Null/small word: never a device VA. Skip without caching a verdict — a
    // pointer field that is merely null on this launch must stay re-checkable.
    if (cand < 0x10000ULL) { j += 1; continue; }
    // Packed-integer false positive: two adjacent uint32 struct fields {lo, hi}
    // — e.g. an ATen OffsetCalculator's per-arg stride/size and IntDivider magic
    // constants embedded in an elementwise functor — can form an 8-byte value
    // where `hi` carries the device-VA prefix (0x7e../0x7f..) and `lo` is a small
    // scalar. That value lands inside a real allocation and would be mis-flagged
    // as an embedded pointer; translating it at replay corrupts the calculator
    // and faults. A genuine 48-bit device pointer never has such tiny low 32
    // bits, so skip these (stay re-checkable; do not cache a sticky verdict).
    if ((cand & 0xFFFFFFFFULL) < 0x10000ULL) { j += 1; continue; }
    hipPointerAttribute_t attr{};
    const bool is_ptr =
        g_real_table.hipPointerGetAttributes_fn(
            &attr, reinterpret_cast<const void*>(cand)) == hipSuccess &&
        (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeUnified);
    if (cacheable)
      verdicts[std::make_tuple(func_key, arg_idx, size, off)] =
          PtrVerdictEntry{ is_ptr ? PtrVerdict::Pointer : PtrVerdict::Scalar, cand };
    if (is_ptr) {
      offsets.push_back(off);
      j += sizeof(void*);
    } else {
      j += 1;
    }
  }
  hip::tls.last_command_error_ = saved_cmd;
  hip::tls.last_error_         = saved_err;
}

// api_id selects the event type the launch is written under. It is
// HRR_API_HIPMODULELAUNCHKERNEL for the plain launches, and the API's own id
// for the extensible spellings, whose events must appear under their own name
// so that a replay of them is attributable to the API the program called.
//
// attrs / num_attrs carry the launch-attribute list of an extensible launch.
// It is a second pointer hanging off the descriptor, so recording the
// descriptor alone would still leave replay with a capture-time address; the
// list is appended to the payload instead.
static void serialize_kernel_launch(
    const char*                 kernel_name,
    const void*                 func_key,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t shared_mem,
    hipStream_t stream,
    const amd::KernelSignature& sig,
    void**                      kernel_params,
    const void*                 kbuf,
    size_t                      ksz,
    hrr_cap::Hash128            co_hash,
    uint16_t                    api_id = HRR_API_HIPMODULELAUNCHKERNEL,
    const hipLaunchAttribute*   attrs = nullptr,
    uint32_t                    num_attrs = 0)
{
  // Reserve space for hrr_event_header at front; payload body follows.
  std::vector<uint8_t> payload(sizeof(hrr_event_header), 0);

  auto push_u8  = [&](uint8_t  v) { payload.push_back(v); };
  auto push_u16 = [&](uint16_t v) {
    payload.push_back(static_cast<uint8_t>(v));
    payload.push_back(static_cast<uint8_t>(v >> 8));
  };
  auto push_u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; i++) payload.push_back(static_cast<uint8_t>(v >> (i*8)));
  };
  auto push_u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; i++) payload.push_back(static_cast<uint8_t>(v >> (i*8)));
  };
  auto push_bytes = [&](const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    payload.insert(payload.end(), p, p + n);
  };

  // raw stream handle as first payload field after header (for replay stream routing)
  push_u64(reinterpret_cast<uint64_t>(stream));

  // name_len is a uint16_t on the wire; a longer name is dropped loudly below
  // rather than recorded truncated.
  const size_t kernel_name_len = std::strlen(kernel_name);
  const bool name_oversized = kernel_name_len > UINT16_MAX;
  uint16_t name_len = name_oversized ? 0 : static_cast<uint16_t>(kernel_name_len);
  push_u16(name_len);
  push_bytes(kernel_name, name_len);
  push_u64(co_hash.lo); push_u64(co_hash.hi);  // code object identity (0 = unknown)
  push_u32(gx); push_u32(gy); push_u32(gz);
  push_u32(bx); push_u32(by); push_u32(bz);
  push_u32(shared_mem);

  // When both kbuf and kernel_params are null (hipLaunchByPtr path) we have
  // no access to the argument values — write num_args=0 so parse_kernel_launch
  // accepts the event.  The replay will launch with null params (device memory
  // already populated from prior H2D transfers).
  uint32_t n_all = sig.numParametersAll();
  uint16_t num_args = 0;
  if (kbuf || kernel_params) {
    for (uint32_t i = 0; i < n_all; i++)
      if (!sig.at(i).info_.hidden_) num_args++;
  }
  push_u16(num_args);
  push_u16(0);  // num_snapshots

  // HIP_HRR_DEBUG_ARGS dumps every captured arg (kind, size, full bytes,
  // detected embedded-pointer offsets) — used to confirm pointer layout. Use the
  // single cached accessor rather than re-reading the env var here.
  const bool dbg = hrr_dbg_args_enabled();

  // Serialize one argument: a whole-arg pointer (kind 1), or a scalar/struct
  // that we additionally scan for embedded device pointers (kind 3 if any are
  // found, else kind 0). `bytes` may be null (unavailable) -> zero-filled.
  auto emit_arg = [&](uint32_t idx, bool is_ptr,
                      const uint8_t* bytes, uint16_t sz) {
    if (is_ptr) {
      push_u8(1); push_u16(sz);
      if (bytes) push_bytes(bytes, sz);
      else for (uint16_t j = 0; j < sz; j++) push_u8(0);
      if (dbg) {
        uint64_t v = 0; if (bytes && sz >= 8) std::memcpy(&v, bytes, 8);
        LogPrintfInfo("[HRR args] %s arg[%u] kind=1(ptr) size=%u value=0x%llx%s",
                      kernel_name, idx, sz, (unsigned long long)v,
                      bytes ? "" : " [TRUNCATED:no-bytes]");
      }
      return;
    }
    std::vector<uint16_t> offs;
    scan_embedded_ptr_offsets(func_key, idx, bytes, sz, offs);
    uint8_t kind = offs.empty() ? 0 : 3;
    push_u8(kind); push_u16(sz);
    if (bytes) push_bytes(bytes, sz);
    else for (uint16_t j = 0; j < sz; j++) push_u8(0);
    if (kind == 3) {
      push_u16(static_cast<uint16_t>(offs.size()));
      for (uint16_t o : offs) push_u16(o);
    }
    if (dbg) {
      std::string hex; char tmp[16];
      for (uint16_t b = 0; bytes && b < sz; b++) {
        std::snprintf(tmp, sizeof(tmp), "%02x", bytes[b]);
        hex += tmp;
      }
      std::string off_str;
      for (uint16_t o : offs) { std::snprintf(tmp, sizeof(tmp), "%u ", o); off_str += tmp; }
      LogPrintfInfo("[HRR args] %s arg[%u] kind=%u size=%u bytes=%s ptr_off=[%s]%s",
                    kernel_name, idx, kind, sz, hex.c_str(), off_str.c_str(),
                    bytes ? "" : " [TRUNCATED:no-bytes]");
    }
  };

  // Each arg's size is recorded as a uint16_t on the wire. A by-value struct
  // argument >= 64 KiB cannot be represented and would otherwise wrap mod 65536,
  // silently corrupting the event. Detect it and drop the launch loudly below.
  bool arg_oversized = false;

  if (kbuf && ksz > 0) {
    const auto* buf_bytes = static_cast<const uint8_t*>(kbuf);
    if (dbg) {
      uint32_t need = 0;
      for (uint32_t i = 0; i < n_all; i++) {
        const auto& desc = sig.at(i);
        if (desc.info_.hidden_) continue;
        need = std::max<uint32_t>(need,
                 static_cast<uint32_t>(desc.offset_ + desc.size_));
      }
      LogPrintfInfo("[HRR args] %s kbuf path: ksz=%zu need=%u n_all=%u%s",
                    kernel_name, ksz, need, n_all,
                    (ksz < need) ? " [KBUF-TOO-SMALL]" : "");
    }
    for (uint32_t i = 0; i < n_all; i++) {
      const auto& desc = sig.at(i);
      if (desc.info_.hidden_) continue;
      if (desc.size_ > UINT16_MAX) { arg_oversized = true; break; }
      uint16_t sz = static_cast<uint16_t>(desc.size_);
      const uint8_t* bytes =
          (desc.offset_ + sz <= ksz) ? buf_bytes + desc.offset_ : nullptr;
      emit_arg(i, desc.type_ == T_POINTER, bytes, sz);
    }
  } else if (kernel_params) {
    uint32_t param_idx = 0;
    for (uint32_t i = 0; i < n_all; i++) {
      const auto& desc = sig.at(i);
      if (desc.info_.hidden_) { continue; }
      if (desc.size_ > UINT16_MAX) { arg_oversized = true; break; }
      uint16_t sz = static_cast<uint16_t>(desc.size_);
      emit_arg(i, desc.type_ == T_POINTER,
               static_cast<const uint8_t*>(kernel_params[param_idx]), sz);
      param_idx++;
    }
  }

  // Launch-attribute tail. Written for every launch so the payload has one
  // shape: a plain launch records a count of zero. The stride lets replay
  // reject a recording made against a different hipLaunchAttribute layout
  // rather than reinterpreting its bytes.
  push_u32(num_attrs);
  push_u32(num_attrs ? static_cast<uint32_t>(sizeof(hipLaunchAttribute)) : 0u);
  if (attrs && num_attrs)
    push_bytes(attrs, static_cast<size_t>(num_attrs) * sizeof(hipLaunchAttribute));

  // payload_length is a uint32_t (wire v4), so the practical ceiling is ~4 GiB —
  // a real launch never approaches it. A per-arg size that exceeds the uint16_t
  // arg-size field (arg_oversized) is the genuine remaining limit. In either
  // case the launch cannot be recorded faithfully: drop it LOUDLY and mark the
  // whole archive incomplete so replay/validation can never silently treat a
  // capture missing a GPU launch (and its downstream writes) as faithful.
  if (arg_oversized || name_oversized || payload.size() > UINT32_MAX) {
    // log_printf truncates at 4 KiB, so the name goes last and is capped.
    LogPrintfError(
        "[HRR capture] Kernel launch cannot be serialized "
        "(name_len=%zu, payload=%zu bytes, oversized_by_value_arg=%s): dropping "
        "the event and marking the capture INCOMPLETE. Replay of this "
        "archive will be unfaithful (this launch and its effects are "
        "absent). Kernel: '%.1024s'",
        kernel_name_len, payload.size(), arg_oversized ? "yes" : "no", kernel_name);
    const char* reason = name_oversized ? "kernel name exceeds the uint16 wire-format length"
                                        : "kernel launch payload exceeds wire-format limits";
    hrr_cap::writer::mark_incomplete(reason);
    return;
  }
  hrr_cap::writer::write_event_raw(api_id,
                                   reinterpret_cast<hrr_event_header*>(payload.data()),
                                   static_cast<uint32_t>(payload.size()));
}

static hrr_cap::Hash128 kernel_code_object_hash(amd::Kernel* kernel);

// ---------------------------------------------------------------------------
// Graph kernel nodes
//
// hipKernelNodeParams names its kernel by a host function address and points
// at a void** argument array — exactly the two things a launch records by
// name and by value instead. So a kernel node event is its ordinary
// hrr_args_* struct followed by this tail, which is the launch encoding minus
// the parts a node does not have (no stream, and grid/block/shared are
// already fields of the struct).
//
// Tail layout: u16 name_len, name bytes, u64 co_hash_lo, u64 co_hash_hi,
// u16 num_args, then per argument: u8 kind, u16 size, size bytes, and for
// kind 3 a u16 count of embedded-pointer offsets followed by that many u16.
// Kinds match the launch encoding (0 scalar, 1 whole-arg pointer, 3 scalar
// with embedded pointers) so replay can decode both with one decoder.
// ---------------------------------------------------------------------------
static void append_kernel_node_tail(std::vector<uint8_t>& payload,
                                    const void* host_func,
                                    void** kernel_params, void** extra) {
  auto push_u8  = [&](uint8_t v)  { payload.push_back(v); };
  auto push_u16 = [&](uint16_t v) {
    payload.push_back(static_cast<uint8_t>(v));
    payload.push_back(static_cast<uint8_t>(v >> 8));
  };
  auto push_u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; i++) payload.push_back(static_cast<uint8_t>(v >> (i*8)));
  };
  auto push_bytes = [&](const void* d, size_t n) {
    const auto* q = static_cast<const uint8_t*>(d);
    payload.insert(payload.end(), q, q + n);
  };

  // A node names its kernel the way hipLaunchKernel does — a host stub — so
  // it needs the same resolution to reach the signature and the real name.
  hipFunction_t f = nullptr;
  amd::Kernel*  kernel = nullptr;
  if (host_func && g_real_table.hipGetFuncBySymbol_fn &&
      g_real_table.hipGetFuncBySymbol_fn(&f, host_func) == hipSuccess && f)
    kernel = hip::asKernel(f);

  if (!kernel) {
    // Nothing to name the kernel by. Write an empty tail; replay reports the
    // node as unreconstructable and marks its graph incomplete rather than
    // building a node that points at nothing.
    push_u16(0);
    push_u64(0); push_u64(0);
    push_u16(0);
    return;
  }

  const amd::KernelSignature& sig = kernel->signature();
  const std::string& name = kernel->name();
  push_u16(static_cast<uint16_t>(name.size()));
  push_bytes(name.data(), name.size());
  const hrr_cap::Hash128 co = kernel_code_object_hash(kernel);
  push_u64(co.lo); push_u64(co.hi);

  const void* kbuf = nullptr;
  size_t      ksz  = 0;
  if (!kernel_params && extra) parse_kernel_extra(extra, kbuf, ksz);

  const uint32_t n_all = sig.numParametersAll();
  uint16_t num_args = 0;
  if (kbuf || kernel_params)
    for (uint32_t i = 0; i < n_all; i++)
      if (!sig.at(i).info_.hidden_) num_args++;
  push_u16(num_args);

  auto emit_arg = [&](uint32_t idx, bool is_ptr, const uint8_t* bytes,
                      uint16_t sz) {
    if (is_ptr) {
      push_u8(1); push_u16(sz);
      if (bytes) push_bytes(bytes, sz);
      else for (uint16_t j = 0; j < sz; j++) push_u8(0);
      return;
    }
    std::vector<uint16_t> offs;
    scan_embedded_ptr_offsets(kernel, idx, bytes, sz, offs);
    push_u8(offs.empty() ? 0 : 3); push_u16(sz);
    if (bytes) push_bytes(bytes, sz);
    else for (uint16_t j = 0; j < sz; j++) push_u8(0);
    if (!offs.empty()) {
      push_u16(static_cast<uint16_t>(offs.size()));
      for (uint16_t o : offs) push_u16(o);
    }
  };

  if (kbuf && ksz > 0) {
    const auto* buf_bytes = static_cast<const uint8_t*>(kbuf);
    for (uint32_t i = 0; i < n_all; i++) {
      const auto& desc = sig.at(i);
      if (desc.info_.hidden_ || desc.size_ > UINT16_MAX) continue;
      uint16_t sz = static_cast<uint16_t>(desc.size_);
      const uint8_t* bytes =
          (desc.offset_ + sz <= ksz) ? buf_bytes + desc.offset_ : nullptr;
      emit_arg(i, desc.type_ == T_POINTER, bytes, sz);
    }
  } else if (kernel_params) {
    uint32_t param_idx = 0;
    for (uint32_t i = 0; i < n_all; i++) {
      const auto& desc = sig.at(i);
      if (desc.info_.hidden_ || desc.size_ > UINT16_MAX) continue;
      emit_arg(i, desc.type_ == T_POINTER,
               static_cast<const uint8_t*>(kernel_params[param_idx]),
               static_cast<uint16_t>(desc.size_));
      param_idx++;
    }
  }
}

static void record_launch(
    hipFunction_t f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared_mem,
    hipStream_t stream,
    void** kernel_params, void** extra,
    uint16_t api_id = HRR_API_HIPMODULELAUNCHKERNEL,
    const hipLaunchAttribute* attrs = nullptr,
    uint32_t num_attrs = 0)
{
  amd::Kernel* kernel = hip::asKernel(f);
  if (!kernel) return;

  const amd::KernelSignature& sig = kernel->signature();
  const void* kbuf = nullptr;
  size_t      ksz  = 0;

  if (!kernel_params && extra)
    parse_kernel_extra(extra, kbuf, ksz);

  serialize_kernel_launch(
      kernel->name().c_str(),
      kernel,  // stable per-kernel key for the embedded-ptr offset cache
      gx, gy, gz, bx, by, bz,
      static_cast<uint32_t>(shared_mem),
      stream, sig, kernel_params, kbuf, ksz,
      kernel_code_object_hash(kernel),
      api_id, attrs, num_attrs);
}

// ---------------------------------------------------------------------------
// Helper macro: fill common hrr_args_* header fields
// ---------------------------------------------------------------------------

// HRR_FILL_HDR removed — write_event_raw() stamps thread_id/sequence_id into
// the payload's hrr_event_header prefix automatically.

// ---------------------------------------------------------------------------
// Memcpy shims — with H2D blob snapshotting
// ---------------------------------------------------------------------------

hipError_t capture_hipMemcpy(void* dst, const void* src,
                                     size_t sizeBytes, hipMemcpyKind kind) {
  hipError_t r = g_real_table.hipMemcpy_fn(dst, src, sizeBytes, kind);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (kind == hipMemcpyHostToDevice && src && sizeBytes > 0) {
      h = hrr_cap::writer::write_blob(src, sizeBytes);
      hrr_trace_h2d("hipMemcpy", dst, sizeBytes);
    } else if (kind == hipMemcpyDeviceToHost && dst && sizeBytes > 0)
      h = hrr_cap::writer::write_blob(dst, sizeBytes);  // host dst valid after sync call
    hrr_args_hipMemcpy a{};
    a.ret           = static_cast<int32_t>(r);
    a.dst           = reinterpret_cast<uint64_t>(dst);
    a.src           = reinterpret_cast<uint64_t>(src);
    a.sizeBytes     = static_cast<uint64_t>(sizeBytes);
    a.kind          = static_cast<int32_t>(kind);
    a.blob_hash_lo  = h.lo;
    a.blob_hash_hi  = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPY, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyAsync(void* dst, const void* src,
                                          size_t sizeBytes, hipMemcpyKind kind,
                                          hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyAsync_fn(dst, src, sizeBytes, kind, stream);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (kind == hipMemcpyHostToDevice && src && sizeBytes > 0) {
      h = hrr_cap::writer::write_blob(src, sizeBytes);
      hrr_trace_h2d("hipMemcpyAsync", dst, sizeBytes);
    } else if (kind == hipMemcpyDeviceToHost && dst && sizeBytes > 0) {
      // Sync the stream so host dst is valid before we snapshot it.
      hipError_t sync_r = sync_for_d2h_snapshot(stream);
      if (sync_r == hipSuccess)
        h = hrr_cap::writer::write_blob(dst, sizeBytes);
      else
        LogPrintfWarning("[HRR capture] hipStreamSynchronize failed (%d) — D2H blob skipped",
                         sync_r);
    }
    hrr_args_hipMemcpyAsync a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.kind         = static_cast<int32_t>(kind);
    a.stream       = reinterpret_cast<uint64_t>(stream);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYASYNC, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyHtoD(hipDeviceptr_t dst, const void* src, size_t sizeBytes) {
  hipError_t r = g_real_table.hipMemcpyHtoD_fn(dst, src, sizeBytes);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (src && sizeBytes > 0) h = hrr_cap::writer::write_blob(src, sizeBytes);
    hrr_trace_h2d("hipMemcpyHtoD", reinterpret_cast<const void*>(dst), sizeBytes);
    hrr_args_hipMemcpyHtoD a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYHTOD, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void* src,
                                      size_t sizeBytes, hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyHtoDAsync_fn(dst, src, sizeBytes, stream);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (src && sizeBytes > 0) h = hrr_cap::writer::write_blob(src, sizeBytes);
    hrr_trace_h2d("hipMemcpyHtoDAsync", reinterpret_cast<const void*>(dst), sizeBytes);
    hrr_args_hipMemcpyHtoDAsync a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.stream       = reinterpret_cast<uint64_t>(stream);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYHTODASYNC, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes) {
  hipError_t r = g_real_table.hipMemcpyDtoH_fn(dst, src, sizeBytes);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (dst && sizeBytes > 0)
      h = hrr_cap::writer::write_blob(dst, sizeBytes);  // dst is host ptr, valid after sync call
    hrr_args_hipMemcpyDtoH a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYDTOH, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src,
                                      size_t sizeBytes, hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyDtoHAsync_fn(dst, src, sizeBytes, stream);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (dst && sizeBytes > 0) {
      // Sync the stream so host dst is valid before we snapshot it.
      hipError_t sync_r = sync_for_d2h_snapshot(stream);
      if (sync_r == hipSuccess)
        h = hrr_cap::writer::write_blob(dst, sizeBytes);
      else
        LogPrintfWarning("[HRR capture] hipStreamSynchronize failed (%d) — DtoH async blob skipped",
                         sync_r);
    }
    hrr_args_hipMemcpyDtoHAsync a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.stream       = reinterpret_cast<uint64_t>(stream);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYDTOHASYNC, &a.hdr, sizeof(a));
  }
  return r;
}

// hipMemcpyWithStream — synchronous copy with an explicit stream.
// Semantics: blocks until the copy completes (like hipMemcpy but with stream).
// H2D: host src is valid immediately on return — snapshot directly.
// D2H: host dst is valid immediately on return — snapshot directly.
// D2D: no blob data needed (both ends are GPU pointers).
hipError_t capture_hipMemcpyWithStream(void* dst, const void* src,
                                       size_t sizeBytes, hipMemcpyKind kind,
                                       hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyWithStream_fn(dst, src, sizeBytes, kind, stream);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (kind == hipMemcpyHostToDevice && src && sizeBytes > 0) {
      h = hrr_cap::writer::write_blob(src, sizeBytes);
      hrr_trace_h2d("hipMemcpyWithStream", dst, sizeBytes);
    } else if (kind == hipMemcpyDeviceToHost && dst && sizeBytes > 0) {
      // Call is synchronous — dst is valid immediately after return.
      h = hrr_cap::writer::write_blob(dst, sizeBytes);
    }
    hrr_args_hipMemcpyWithStream a{};
    a.ret          = static_cast<int32_t>(r);
    a.dst          = reinterpret_cast<uint64_t>(dst);
    a.src          = reinterpret_cast<uint64_t>(src);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.kind         = static_cast<int32_t>(kind);
    a.stream       = reinterpret_cast<uint64_t>(stream);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMEMCPYWITHSTREAM, &a.hdr, sizeof(a));
  }
  return r;
}

// ---------------------------------------------------------------------------
// Module shims
// ---------------------------------------------------------------------------

// Is this buffer something hipModuleLoadData can load at replay?
//
// amd::Program::binary() hands back a bare pointer and length with no promise
// the memory is still the program's own. A code object whose buffer the
// runtime has already released reads back as whatever the allocator put there
// next, at exactly the recorded length, and nothing downstream notices: the
// bytes hash cleanly, land in code_objects/, and only fail at replay with
// "device kernel image is invalid" (HIP 200). Checking the magic turns that
// into a capture-time warning and a launch that resolves by name instead.
static bool image_looks_loadable(const void* data, size_t size) {
  if (!data || size < 8) return false;
  const auto* p = static_cast<const char*>(data);
  if (std::memcmp(p, "\x7f" "ELF", 4) == 0) return true;
  if (size >= hip::symbols::kOffloadBundleCompressedMagicStrSize - 1 &&
      std::memcmp(p, hip::symbols::kOffloadBundleCompressedMagicStr,
                  hip::symbols::kOffloadBundleCompressedMagicStrSize - 1) == 0)
    return true;
  if (size >= hip::symbols::kOffloadBundleUncompressedMagicStrSize - 1 &&
      std::memcmp(p, hip::symbols::kOffloadBundleUncompressedMagicStr,
                  hip::symbols::kOffloadBundleUncompressedMagicStrSize - 1) == 0)
    return true;
  return false;
}

// Hash of the code object behind an amd::Program, recorded once and reused.
//
// Keyed by program because a launch has to name the same bytes the module load
// recorded: Triton/inductor emits the generic entry symbol "triton_" from
// dozens of distinct code objects, so name-only resolution at replay binds
// every "triton_" launch to one arbitrary kernel and faults (HIP error 719).
//
// The cache is also what keeps the read *timely*. Populating it lazily at first
// launch meant re-reading binary() long after the module was built, which is
// where the stale-buffer problem above showed up in practice — an inductor
// module's 6.5 KB code object came back as Triton MLIR source text. Module load
// seeds the cache now, so the launch path only re-reads for programs that were
// never loaded through a module shim (fat binaries registered at static init).
//
// An entry records which image it hashed, not just the hash, because an
// amd::Program is freed when its module is unloaded and the next one can land
// on the same address — inductor churns through modules, so this happens. A
// hash keyed on the address alone then hands the later program the earlier
// one's code object, and its launches resolve to nothing at replay ("not found
// in any loaded module"). Comparing the image pointer and length is a second
// check, but both are recyclable; capture_hipModuleUnload erases the entry so
// a recycled (program, buffer, length) triple cannot hit a stale hash.
struct ProgramCodeObject {
  const void*      image;
  size_t           size;
  hrr_cap::Hash128 hash;
};
// g_prog_hash_mu guards g_prog_hash *and* every call HRR makes into
// amd::Program::binary(). The latter is `return binary_[&device];` — a
// non-const, inserting std::unordered_map::operator[] that the runtime does
// not lock. Reading a span is therefore a write as far as binary_ is
// concerned: the first read for a device the program was not built for
// inserts an empty entry and can rehash the bucket array. Two capture threads
// launching kernels from the same program would otherwise do that
// concurrently. Serializing HRR's reads here keeps us off that path.
static std::mutex g_prog_hash_mu;
static std::unordered_map<const amd::Program*, ProgramCodeObject> g_prog_hash;
static std::atomic<bool> g_span_no_device_warned{false};

// Fetch a program's device binary as (pointer, length) without dereferencing
// the bytes (the call itself still goes through binary_'s inserting
// operator[] — see above).
// Caller must hold g_prog_hash_mu.
// Returns false when there is no current device (or the index is out of
// range): that is not "the program has no binary", so callers must not seed
// g_prog_hash with a {0,0} / (nullptr,0) entry.
static bool program_binary_span_locked(amd::Program* prog, const uint8_t*& data, size_t& size) {
  data = nullptr;
  size = 0;
  if (!prog) return false;
  const int dev = hip::ihipGetDevice();
  if (dev < 0 || static_cast<size_t>(dev) >= hip::g_devices.size()) {
    if (!g_span_no_device_warned.exchange(true)) {
      LogPrintfWarning("[HRR capture] program %p: no current HIP device "
                       "(ihipGetDevice()=%d, g_devices=%zu); not hashing a code object",
                       (const void*)prog, dev, hip::g_devices.size());
    }
    return false;
  }
  const amd::Device* device = hip::g_devices[dev]->devices()[0];
  const auto& bin = prog->binary(*device);
  data = std::get<0>(bin);
  size = std::get<1>(bin).first;
  return true;
}

static bool program_binary_span(amd::Program* prog, const uint8_t*& data, size_t& size) {
  std::lock_guard<std::mutex> lk(g_prog_hash_mu);
  return program_binary_span_locked(prog, data, size);
}

static void remember_program_hash(const amd::Program* prog, const void* image, size_t size,
                                  hrr_cap::Hash128 h) {
  if (!prog || !image || size == 0) return;
  std::lock_guard<std::mutex> lk(g_prog_hash_mu);
  g_prog_hash[prog] = ProgramCodeObject{image, size, h};
}

static void forget_program_hash(const amd::Program* prog) {
  if (!prog) return;
  std::lock_guard<std::mutex> lk(g_prog_hash_mu);
  g_prog_hash.erase(prog);
}

// Hash an already-read device binary span, refusing anything that does not
// look like a loadable image. Returns {0,0} when there is nothing safe to
// record. Takes the span rather than re-reading it so callers that already
// have one do not go back through amd::Program::binary(); must not be called
// with g_prog_hash_mu held (write_code_object takes the writer locks).
static hrr_cap::Hash128 hash_program_image(const amd::Program* prog, const uint8_t* data,
                                           size_t size) {
  if (!data || !size) return {0, 0};
  if (!image_looks_loadable(data, size)) {
    LogPrintfWarning("[HRR capture] program %p: %llu bytes at %p are not a loadable image"
                     " (no ELF/bundle magic) — not recording a code object for it;"
                     " its module event (if any) will not replay, and launches fall back"
                     " to name resolution",
                     (const void*)prog, (unsigned long long)size, (const void*)data);
    return {0, 0};
  }
  return hrr_cap::writer::write_code_object(data, size);
}

// Helper: get the actual device binary from a successfully loaded hipModule_t.
// The module may come from an in-memory image or from a file (possibly a
// bundle); either way the runtime stores the extracted device ELF in the
// amd::Program, and we read it back from there.
// It also seeds the launch-path hash cache (g_prog_hash) for this program.
static hrr_cap::Hash128 write_module_code_object(hipModule_t module) {
  amd::Program* prog = as_amd(reinterpret_cast<cl_program>(module));
  if (!prog) return {0, 0};
  const uint8_t* image = nullptr;
  size_t size = 0;
  if (!program_binary_span(prog, image, size)) return {0, 0};
  hrr_cap::Hash128 h = hash_program_image(prog, image, size);
  remember_program_hash(prog, image, size, h);
  return h;
}

static hrr_cap::Hash128 kernel_code_object_hash(amd::Kernel* kernel) {
  if (!kernel) return {0, 0};
  amd::Program* prog = &kernel->program();
  if (!prog) return {0, 0};
  const uint8_t* image = nullptr;
  size_t size = 0;
  bool got_span = false;
  // One critical section for the span read and the lookup: the span read is
  // the part that must not race (see g_prog_hash_mu above), and folding the
  // lookup in costs nothing since we hold the lock anyway.
  {
    std::lock_guard<std::mutex> lk(g_prog_hash_mu);
    got_span = program_binary_span_locked(prog, image, size);
    if (got_span) {
      auto it = g_prog_hash.find(prog);
      if (it != g_prog_hash.end() && it->second.image == image && it->second.size == size)
        return it->second.hash;
    }
  }
  if (!got_span) return {0, 0};
  // Miss. Hash outside the lock — write_code_object takes the writer's own
  // locks, and hashing the same image twice is harmless (content-addressed).
  hrr_cap::Hash128 h = hash_program_image(prog, image, size);
  remember_program_hash(prog, image, size, h);
  return h;
}

hipError_t capture_hipModuleLoadData(hipModule_t* module, const void* image) {
  hipError_t r = g_real_table.hipModuleLoadData_fn(module, image);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h = write_module_code_object(*module);
    hrr_args_hipModuleLoadData a{};
    a.ret        = static_cast<int32_t>(r);
    a.module     = reinterpret_cast<uint64_t>(*module);
    a.image      = reinterpret_cast<uint64_t>(image);
    a.co_hash_lo = h.lo;
    a.co_hash_hi = h.hi;
    a.module_id  = 0;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELOADDATA, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                       unsigned int numOptions,
                                       hipJitOption* options,
                                       void** optionValues) {
  hipError_t r = g_real_table.hipModuleLoadDataEx_fn(
      module, image, numOptions, options, optionValues);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h = write_module_code_object(*module);
    hrr_args_hipModuleLoadDataEx a{};
    a.ret          = static_cast<int32_t>(r);
    a.module       = reinterpret_cast<uint64_t>(*module);
    a.image        = reinterpret_cast<uint64_t>(image);
    a.numOptions   = static_cast<uint32_t>(numOptions);
    a.options      = reinterpret_cast<uint64_t>(options);
    a.optionValues = reinterpret_cast<uint64_t>(optionValues);
    a.co_hash_lo   = h.lo;
    a.co_hash_hi   = h.hi;
    a.module_id    = 0;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELOADDATAEX, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipModuleLoad(hipModule_t* module, const char* fname) {
  hipError_t r = g_real_table.hipModuleLoad_fn(module, fname);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h = write_module_code_object(*module);
    hrr_args_hipModuleLoad a{};
    a.ret        = static_cast<int32_t>(r);
    a.module     = reinterpret_cast<uint64_t>(*module);
    a.fname      = 0;  // not a valid cross-process address; hash identifies the code object
    a.co_hash_lo = h.lo;
    a.co_hash_hi = h.hi;
    a.module_id  = 0;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELOAD, &a.hdr, sizeof(a));
  }
  return r;
}

hipError_t capture_hipModuleUnload(hipModule_t module) {
  // Take the program pointer before the real unload frees it.
  amd::Program* prog = module ? as_amd(reinterpret_cast<cl_program>(module)) : nullptr;
  hipError_t r = g_real_table.hipModuleUnload_fn(module);
  if (r == hipSuccess) {
    forget_program_hash(prog);
    hrr_args_hipModuleUnload a{};
    a.ret    = static_cast<int32_t>(r);
    a.module = reinterpret_cast<uint64_t>(module);
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULEUNLOAD, &a.hdr, sizeof(a));
  }
  return r;
}

// ---------------------------------------------------------------------------
// Kernel launch shims — variable-length binary payload (not hrr_args_*)
// ---------------------------------------------------------------------------

hipError_t capture_hipModuleLaunchKernel(
    hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra) {
  hipError_t r = g_real_table.hipModuleLaunchKernel_fn(
      f, gridDimX, gridDimY, gridDimZ,
         blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams, extra);
  if (r == hipSuccess) {
    record_launch(f, gridDimX, gridDimY, gridDimZ,
                     blockDimX, blockDimY, blockDimZ,
                  sharedMemBytes, stream, kernelParams, extra);
  }
  return r;
}

// A cooperative launch carries the same payload as any other module launch —
// the only thing that makes it cooperative is which entry point runs it, and
// that is what the event id records.
hipError_t capture_hipModuleLaunchCooperativeKernel(
    hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams) {
  hipError_t r = g_real_table.hipModuleLaunchCooperativeKernel_fn(
      f, gridDimX, gridDimY, gridDimZ,
         blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams);
  if (r == hipSuccess) {
    record_launch(f, gridDimX, gridDimY, gridDimZ,
                     blockDimX, blockDimY, blockDimZ,
                  sharedMemBytes, stream, kernelParams, nullptr,
                  HRR_API_HIPMODULELAUNCHCOOPERATIVEKERNEL);
  }
  return r;
}

hipError_t capture_hipExtModuleLaunchKernel(
    hipFunction_t f,
    uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
    uint32_t localWorkSizeX,  uint32_t localWorkSizeY,  uint32_t localWorkSizeZ,
    size_t sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra,
    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags) {
  hipError_t r = g_real_table.hipExtModuleLaunchKernel_fn(
      f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
         localWorkSizeX,  localWorkSizeY,  localWorkSizeZ,
      sharedMemBytes, stream, kernelParams, extra, startEvent, stopEvent, flags);
  if (r == hipSuccess) {
    // hipExtModuleLaunchKernel takes *global work-item counts* (HSA/OpenCL
    // semantics), whereas the unified launch event — and the
    // hipModuleLaunchKernel replay path — expects *workgroup counts*. Convert
    // here so the recording is unambiguous and replays correctly. Recording the
    // raw global sizes would over-launch the grid by blockDim on replay and
    // deadlock persistent, co-resident kernels (e.g. hipBLASLt StreamK
    // producer/consumer flag handshakes).
    auto ceil_div = [](uint32_t a, uint32_t b) -> unsigned {
      return b ? static_cast<unsigned>((a + b - 1) / b) : static_cast<unsigned>(a);
    };
    record_launch(f,
                  ceil_div(globalWorkSizeX, localWorkSizeX),
                  ceil_div(globalWorkSizeY, localWorkSizeY),
                  ceil_div(globalWorkSizeZ, localWorkSizeZ),
                  localWorkSizeX,  localWorkSizeY,  localWorkSizeZ,
                  static_cast<unsigned>(sharedMemBytes), stream, kernelParams, extra);
  }
  return r;
}

hipError_t capture_hipLaunchKernel(const void* function_address,
                                           dim3 numBlocks, dim3 dimBlocks,
                                           void** args, size_t sharedMemBytes,
                                           hipStream_t stream) {
  hipError_t r = g_real_table.hipLaunchKernel_fn(
      function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
  if (r == hipSuccess) {
    // function_address is a host stub pointer, not hipFunction_t — resolve via dispatch table
    hipFunction_t f = nullptr;
    if (g_real_table.hipGetFuncBySymbol_fn &&
        g_real_table.hipGetFuncBySymbol_fn(&f, function_address) == hipSuccess && f) {
      record_launch(f,
                    numBlocks.x, numBlocks.y, numBlocks.z,
                    dimBlocks.x, dimBlocks.y, dimBlocks.z,
                    static_cast<unsigned>(sharedMemBytes), stream, args, nullptr);
    }
  }
  return r;
}

// The stream-per-thread spelling of the same launch. Any translation unit
// compiled with -fgpu-default-stream=per-thread calls this instead of
// hipLaunchKernel, and the generated shim recorded the host stub address, which
// replay then handed to the runtime — a segfault, not an error. It gets the
// same encoding as the plain spelling under its own event id, so a recording
// still says which entry point the program used.
hipError_t capture_hipLaunchKernel_spt(const void* function_address,
                                       dim3 numBlocks, dim3 dimBlocks,
                                       void** args, size_t sharedMemBytes,
                                       hipStream_t stream) {
  hipError_t r = g_real_table.hipLaunchKernel_spt_fn(
      function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
  if (r == hipSuccess) {
    hipFunction_t f = nullptr;
    if (g_real_table.hipGetFuncBySymbol_fn &&
        g_real_table.hipGetFuncBySymbol_fn(&f, function_address) == hipSuccess && f) {
      record_launch(f,
                    numBlocks.x, numBlocks.y, numBlocks.z,
                    dimBlocks.x, dimBlocks.y, dimBlocks.z,
                    static_cast<unsigned>(sharedMemBytes), stream, args, nullptr,
                    HRR_API_HIPLAUNCHKERNEL_SPT);
    }
  }
  return r;
}

// The cooperative launches by host stub, plain and stream-per-thread. Both
// carry the module-launch payload; the event id is what tells replay to go
// back through the cooperative entry point rather than the ordinary one.
hipError_t capture_hipLaunchCooperativeKernel(const void* f,
                                              dim3 gridDim, dim3 blockDimX,
                                              void** kernelParams,
                                              unsigned int sharedMemBytes,
                                              hipStream_t stream) {
  hipError_t r = g_real_table.hipLaunchCooperativeKernel_fn(
      f, gridDim, blockDimX, kernelParams, sharedMemBytes, stream);
  if (r == hipSuccess) {
    hipFunction_t fn = nullptr;
    if (g_real_table.hipGetFuncBySymbol_fn &&
        g_real_table.hipGetFuncBySymbol_fn(&fn, f) == hipSuccess && fn) {
      record_launch(fn,
                    gridDim.x, gridDim.y, gridDim.z,
                    blockDimX.x, blockDimX.y, blockDimX.z,
                    sharedMemBytes, stream, kernelParams, nullptr,
                    HRR_API_HIPLAUNCHCOOPERATIVEKERNEL);
    }
  }
  return r;
}

hipError_t capture_hipLaunchCooperativeKernel_spt(const void* f,
                                                  dim3 gridDim, dim3 blockDim,
                                                  void** kernelParams,
                                                  uint32_t sharedMemBytes,
                                                  hipStream_t hStream) {
  hipError_t r = g_real_table.hipLaunchCooperativeKernel_spt_fn(
      f, gridDim, blockDim, kernelParams, sharedMemBytes, hStream);
  if (r == hipSuccess) {
    hipFunction_t fn = nullptr;
    if (g_real_table.hipGetFuncBySymbol_fn &&
        g_real_table.hipGetFuncBySymbol_fn(&fn, f) == hipSuccess && fn) {
      record_launch(fn,
                    gridDim.x, gridDim.y, gridDim.z,
                    blockDim.x, blockDim.y, blockDim.z,
                    sharedMemBytes, hStream, kernelParams, nullptr,
                    HRR_API_HIPLAUNCHCOOPERATIVEKERNEL_SPT);
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// Extensible launches (hipDrvLaunchKernelEx / hipLaunchKernelExC)
//
// The descriptor is a const struct pointer and its attribute list is a second
// pointer hanging off it, so the generated shim recorded two capture-time host
// addresses and nothing else — Triton's and CK's default launch path replayed
// as an invalid resource handle. Both spellings record the same
// variable-length launch payload as every other launch, under their own event
// id, with the attribute list appended.
// ---------------------------------------------------------------------------

hipError_t capture_hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config,
                                        hipFunction_t f, void** params,
                                        void** extra) {
  hipError_t r = g_real_table.hipDrvLaunchKernelEx_fn(config, f, params, extra);
  if (r == hipSuccess && config) {
    record_launch(f,
                  config->gridDimX, config->gridDimY, config->gridDimZ,
                  config->blockDimX, config->blockDimY, config->blockDimZ,
                  config->sharedMemBytes, config->hStream, params, extra,
                  HRR_API_HIPDRVLAUNCHKERNELEX, config->attrs, config->numAttrs);
  }
  return r;
}

hipError_t capture_hipLaunchKernelExC(const hipLaunchConfig_t* config,
                                      const void* fPtr, void** args) {
  hipError_t r = g_real_table.hipLaunchKernelExC_fn(config, fPtr, args);
  if (r == hipSuccess && config) {
    // fPtr is a host stub address, not a hipFunction_t: the same resolution
    // hipLaunchKernel needs, and the same reason replay cannot use the
    // recorded value directly.
    hipFunction_t f = nullptr;
    if (g_real_table.hipGetFuncBySymbol_fn &&
        g_real_table.hipGetFuncBySymbol_fn(&f, fPtr) == hipSuccess && f) {
      record_launch(f,
                    config->gridDim.x, config->gridDim.y, config->gridDim.z,
                    config->blockDim.x, config->blockDim.y, config->blockDim.z,
                    static_cast<unsigned>(config->dynamicSmemBytes),
                    config->stream, args, nullptr,
                    HRR_API_HIPLAUNCHKERNELEXC, config->attrs, config->numAttrs);
    }
  }
  return r;
}

hipError_t capture___hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                              size_t sharedMem, hipStream_t stream) {
  hipError_t r = g_real_compiler_table.__hipPushCallConfiguration_fn(
      gridDim, blockDim, sharedMem, stream);
  if (r == hipSuccess) {
    // Save into TLS so capture_hipLaunchByPtr can read the launch dimensions.
    g_pushed_grid   = gridDim;
    g_pushed_block  = blockDim;
    g_pushed_shared = sharedMem;
    g_pushed_stream = stream;
    if (hip_capture_enabled()) {
      hrr_args___hipPushCallConfiguration a{};
      a.ret        = static_cast<int32_t>(r);
      a.gridDim_x  = gridDim.x;
      a.gridDim_y  = gridDim.y;
      a.gridDim_z  = gridDim.z;
      a.blockDim_x = blockDim.x;
      a.blockDim_y = blockDim.y;
      a.blockDim_z = blockDim.z;
      a.sharedMem  = static_cast<decltype(a.sharedMem)>(sharedMem);
      a.stream     = reinterpret_cast<uint64_t>(stream);
      hrr_cap::writer::write_event_raw(HRR_API_HIPPUSHCALLCONFIGURATION, &a.hdr, sizeof(a));
    }
  }
  return r;
}

hipError_t capture_hipLaunchByPtr(const void* func) {
  // Snapshot the WHOLE exec (dims + args) from the top of the per-thread exec
  // stack *before* the real hipLaunchByPtr, which PopExec's (std::move) it.
  //
  // top() is the authoritative source for launch config on BOTH launch paths,
  // because hipConfigureCall and __hipPushCallConfiguration both funnel into
  // PlatformState::ConfigureCall(), which pushes ihipExec_t{grid,block,shared,
  // stream}:
  //   - modern <<<>>>:  __hipPushCallConfiguration -> ConfigureCall (push exec)
  //   - legacy CUDA RT: hipConfigureCall -> ConfigureCall (push exec)
  //                     + hipSetupArgument (fills exec.arguments_)
  // The g_pushed_* shadow is only written by capture___hipPushCallConfiguration,
  // so on the legacy hipConfigureCall path it is stale/zero. Reading dims from
  // top() keeps dims and args from the same snapshot and is correct on both
  // paths; fall back to g_pushed_* only if the stack is unexpectedly empty.
  dim3        grid   = g_pushed_grid;
  dim3        block  = g_pushed_block;
  size_t      shared = g_pushed_shared;
  hipStream_t stream = g_pushed_stream;
  std::vector<char> kargs;
  if (hip_capture_enabled() && !hip::tls.exec_stack_.empty()) {
    const auto& e = hip::tls.exec_stack_.top();
    grid   = e.gridDim_;
    block  = e.blockDim_;
    shared = e.sharedMem_;
    stream = e.hStream_;
    kargs  = e.arguments_;  // copy (not move) to leave arguments_ intact for the real launch
  }

  hipError_t r = g_real_table.hipLaunchByPtr_fn(func);
  if (r == hipSuccess) {
    // func is a host stub pointer — resolve to real hipFunction_t first
    hipFunction_t f = nullptr;
    if (g_real_table.hipGetFuncBySymbol_fn &&
        g_real_table.hipGetFuncBySymbol_fn(&f, func) == hipSuccess && f) {
      amd::Kernel* kernel = hip::asKernel(f);
      if (kernel) {
        const amd::KernelSignature& sig = kernel->signature();
        // Feed the snapshot through the existing kbuf path (indexed by
        // desc.offset_). Empty buffer falls back to num_args=0.
        const void* kbuf = kargs.empty() ? nullptr : kargs.data();
        size_t      ksz  = kargs.size();
        serialize_kernel_launch(
            kernel->name().c_str(),
            kernel,  // stable per-kernel key for the embedded-ptr offset cache
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            static_cast<uint32_t>(shared), stream,
            sig, nullptr, kbuf, ksz,
            kernel_code_object_hash(kernel));
      }
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// Fat binary registration — record the binary blob
//
// We capture the clang offload bundle blob so the replay can load it via
// hipModuleLoadData, making all kernel names resolvable.
// ---------------------------------------------------------------------------

// __hipRegisterFatBinary receives a HIPF/HIPK wrapper, not the clang offload
// bundle itself; the live shim unwraps it before sizing.
//
// StatCO's keys are mixed, so the retroactive sweep at capture init sees both
// kinds of pointer: HIPF modules are keyed by the unwrapped bundle
// (AddFatBinary(fbwrapper->binary)), HIPK modules by the wrapper
// (AddKpackBinary(..., data)). fat_binary_blob_ptr() is therefore tolerant by
// design: it unwraps only on HIPF/HIPK magic and returns anything else as is.
// A bundle cannot be mistaken for a wrapper, since it starts with "__CL" or
// "CCOB" and is always longer than the 4-byte magic read. A HIPK wrapper
// unwraps to msgpack metadata rather than a bundle, so compute_bundle_size()
// returns 0 for it and no event is recorded.
struct __HRRFatBinaryWrapper {
  uint32_t magic;
  uint32_t version;
  const void* binary;
  const void* dummy;
};

static const void* fat_binary_blob_ptr(const void* data) {
  if (!data) return nullptr;
  const auto* wrapper = static_cast<const __HRRFatBinaryWrapper*>(data);
  if (wrapper->magic == 0x48495046u /* HIPF */ ||
      wrapper->magic == 0x4B504948u /* HIPK */)
    return wrapper->binary;
  return data;
}

// Byte size of a clang offload bundle, or 0 when the header is not one or fails
// the checks HIP applies to it. HIP receives a registered fat binary as a bare
// pointer, so it knows no image bound (image_bound in hip_fatbin.cpp) and these
// checks are all that stands between the header and the read.
static size_t compute_bundle_size(const void* blob) {
  if (!blob) return 0;
  const char* p = static_cast<const char*>(blob);

  // Compressed format: magic "CCOB", header contains totalSize at byte 8.
  if (std::memcmp(p, hip::symbols::kOffloadBundleCompressedMagicStr,
                  hip::symbols::kOffloadBundleCompressedMagicStrSize - 1) == 0) {
    constexpr size_t kHeaderSize =
        offsetof(hip::symbols::ClangOffloadBundleCompressedHeader, compressedBinarydesc);
    const auto* hdr = static_cast<const hip::symbols::ClangOffloadBundleCompressedHeader*>(blob);
    if (hdr->totalSize < kHeaderSize) {
      LogPrintfWarning(
          "[HRR capture] Fat binary at %p not recorded: compressed bundle "
          "totalSize %u is smaller than its %zu-byte header",
          blob, static_cast<unsigned>(hdr->totalSize), kHeaderSize);
      return 0;
    }
    return static_cast<size_t>(hdr->totalSize);
  }

  // Uncompressed format: magic "__CLANG_OFFLOAD_BUNDLE__" + numOfCodeObjects + entries.
  if (std::memcmp(p, hip::symbols::kOffloadBundleUncompressedMagicStr,
                  hip::symbols::kOffloadBundleUncompressedMagicStrSize - 1) != 0) {
    return 0;  // Unknown format
  }
  const auto* hdr = static_cast<const hip::symbols::ClangOffloadBundleUncompressedHeader*>(blob);
  uint64_t n = hdr->numOfCodeObjects;
  if (n == 0) return 0;

  // HIP hands COMGR only the first 4096 bytes, and COMGR reads the entry table from
  // that slice and fails if a read runs off its end. __hipRegisterFatBinary passes
  // no length, so this bound is also what keeps the walk below in range.
  constexpr size_t kEntryTableLimit = 4096;
  constexpr size_t kEntryHeaderSize = offsetof(hip::symbols::ClangOffloadBundleInfo, bundleEntryId);
  size_t pos = offsetof(hip::symbols::ClangOffloadBundleUncompressedHeader, desc);
  uint64_t end = 0;
  for (uint64_t i = 0; i < n; i++) {
    const auto* entry = reinterpret_cast<const hip::symbols::ClangOffloadBundleInfo*>(p + pos);
    if (kEntryTableLimit - pos < kEntryHeaderSize ||
        entry->bundleEntryIdSize > kEntryTableLimit - pos - kEntryHeaderSize) {
      LogPrintfWarning(
          "[HRR capture] Fat binary at %p not recorded: bundle entry %llu ends "
          "past the first %zu bytes, where HIP looks for it",
          blob, static_cast<unsigned long long>(i), kEntryTableLimit);
      return 0;
    }
    pos += kEntryHeaderSize + static_cast<size_t>(entry->bundleEntryIdSize);
    // HIP makes this check only for the code objects it picks for this host's GPUs.
    // Every entry counts towards the end here, so capture checks them all.
    const uint64_t entry_end = entry->offset + entry->size;
    if (entry_end < entry->offset) {
      LogPrintfWarning(
          "[HRR capture] Fat binary at %p not recorded: bundle entry %llu has "
          "an offset and size that overflow",
          blob, static_cast<unsigned long long>(i));
      return 0;
    }
    if (entry_end > end) end = entry_end;
  }
  static_assert(sizeof(size_t) >= sizeof(uint64_t), "a 64-bit bundle size must fit in size_t");
  return static_cast<size_t>(end);
}

void** capture___hipRegisterFatBinary(const void* data) {
  void** r = g_real_compiler_table.__hipRegisterFatBinary_fn(data);
  // Shim is only installed when capture is active — no hip_capture_enabled() check needed.

  const void* blob = fat_binary_blob_ptr(data);
  // PlatformState is initialized before this shim is installed, so HIP has already
  // digested the bundle, and no handle means it could not load it on this host: a bad
  // header or, more often, no code object for any GPU here. Then no launch from it can
  // succeed and be recorded, so replay never needs the blob.
  if (blob && !r) {
    LogPrintfWarning(
        "[HRR capture] Fat binary at %p not recorded: HIP did not register it "
        "(no code object for any GPU here, or a bad header)",
        blob);
    blob = nullptr;
  }
  size_t blob_size = blob ? compute_bundle_size(blob) : 0;

  hrr_args___hipRegisterFatBinary a{};
  a.ret      = reinterpret_cast<uint64_t>(r);
  a.blob_size = static_cast<uint64_t>(blob_size);
  if (blob && blob_size > 0) {
    auto h = hrr_cap::writer::write_blob(blob, blob_size);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
  }
  hrr_cap::writer::write_event_raw(HRR_API_HIPREGISTERFATBINARY, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// __hipUnregisterFatBinary — snapshot *modules before the real call
//
// The real call frees the node *modules points into, so we read *modules
// (used to correlate with the register event) first, then forward the call.
// ---------------------------------------------------------------------------
void capture___hipUnregisterFatBinary(void** modules) {
  hrr_args___hipUnregisterFatBinary a{};
  if (modules) a.modules = reinterpret_cast<uint64_t>(*modules);
  hrr_cap::writer::write_event_raw(HRR_API_HIPUNREGISTERFATBINARY, &a.hdr, sizeof(a));

  g_real_compiler_table.__hipUnregisterFatBinary_fn(modules);
}

// ---------------------------------------------------------------------------
// hipHostRegister / hipHostUnregister — sysmem blob snapshotting
//
// We snapshot the host memory at Register time so the replayer can restore it
// before calling hipHostRegister on a freshly allocated buffer.
// hipHostUnregister doesn't receive a size, so we track it in pinned_reg_map.
// ---------------------------------------------------------------------------

static std::mutex                              g_pinned_reg_mu;
static std::unordered_map<void*, size_t>       g_pinned_reg_map;

hipError_t capture_hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
  hipError_t r = g_real_table.hipHostRegister_fn(hostPtr, sizeBytes, flags);
  if (r == hipSuccess) {
    hrr_cap::Hash128 h{0, 0};
    if (hostPtr && sizeBytes > 0)
      h = hrr_cap::writer::write_blob(hostPtr, sizeBytes);
    hrr_args_hipHostRegister a{};
    a.ret          = static_cast<int32_t>(r);
    a.hostPtr      = reinterpret_cast<uint64_t>(hostPtr);
    a.sizeBytes    = static_cast<uint64_t>(sizeBytes);
    a.flags        = flags;
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
    hrr_cap::writer::write_event_raw(HRR_API_HIPHOSTREGISTER, &a.hdr, sizeof(a));
    {
      std::lock_guard<std::mutex> lk(g_pinned_reg_mu);
      g_pinned_reg_map[hostPtr] = sizeBytes;
    }
  }
  return r;
}

hipError_t capture_hipHostUnregister(void* hostPtr) {
  hipError_t r = g_real_table.hipHostUnregister_fn(hostPtr);
  if (r == hipSuccess) {
    hrr_args_hipHostUnregister a{};
    a.ret     = static_cast<int32_t>(r);
    a.hostPtr = reinterpret_cast<uint64_t>(hostPtr);
    hrr_cap::writer::write_event_raw(HRR_API_HIPHOSTUNREGISTER, &a.hdr, sizeof(a));
    {
      std::lock_guard<std::mutex> lk(g_pinned_reg_mu);
      g_pinned_reg_map.erase(hostPtr);
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
// hipMemPoolSetAttribute: value is void* to a scalar, stored inline in the low
// bytes of value_u64 (an int32_t for the three reuse policies, uint64_t otherwise).
// ---------------------------------------------------------------------------

hipError_t capture_hipMemPoolSetAttribute(hipMemPool_t mem_pool,
                                          hipMemPoolAttr attr,
                                          void* value) {
  hipError_t r = g_real_table.hipMemPoolSetAttribute_fn(mem_pool, attr, value);
  if (r != hipSuccess) return r;
  hrr_args_hipMemPoolSetAttribute a{};
  a.ret      = static_cast<int32_t>(r);
  a.mem_pool = reinterpret_cast<uint64_t>(mem_pool);
  a.attr     = static_cast<int32_t>(attr);
  a.value    = reinterpret_cast<uint64_t>(value);
  const bool int_valued = attr == hipMemPoolReuseFollowEventDependencies ||
                          attr == hipMemPoolReuseAllowOpportunistic ||
                          attr == hipMemPoolReuseAllowInternalDependencies;
  if (value) std::memcpy(&a.value_u64, value, int_valued ? sizeof(int32_t) : sizeof(a.value_u64));
  hrr_cap::writer::write_event_raw(HRR_API_HIPMEMPOOLSETATTRIBUTE, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// hipMemPoolCreate — pool_props is a struct pointer; copy it inline.
// ---------------------------------------------------------------------------

hipError_t capture_hipMemPoolCreate(hipMemPool_t* mem_pool,
                                    const hipMemPoolProps* pool_props) {
  hipError_t r = g_real_table.hipMemPoolCreate_fn(mem_pool, pool_props);
  if (r != hipSuccess) return r;
  hrr_args_hipMemPoolCreate a{};
  a.ret      = static_cast<int32_t>(r);
  a.mem_pool = reinterpret_cast<uint64_t>(*mem_pool);
  if (pool_props)
    std::memcpy(a.pool_props_bytes, pool_props, sizeof(a.pool_props_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPMEMPOOLCREATE, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// hipMemcpy3D / hipMemcpy3DAsync — inline parms + H2D blob + D2H expected blob
// ---------------------------------------------------------------------------

// Helper: compute byte count from 3D extent into *bytes (0 for a zero extent).
// Returns false when the product overflows size_t.
static bool memcpy3d_byte_count(const struct hipMemcpy3DParms* p, size_t* bytes) {
  const size_t w = p->extent.width, h = p->extent.height, d = p->extent.depth;
  *bytes = 0;
  if (w == 0 || h == 0 || d == 0) return true;
  if (h > SIZE_MAX / w || d > SIZE_MAX / (w * h)) return false;
  *bytes = w * h * d;
  return true;
}

// Defined with the driver-copy helpers below. The runtime widens a hipMemcpy3D
// to the same HIP_MEMCPY3D, so its host side is the same rect.
static hrr_cap::Hash128 write_host_rect_blob(const void* base, size_t pitch,
                                             size_t pitch_height, size_t x, size_t y,
                                             size_t z, size_t width, size_t height,
                                             size_t depth);

// Helper shared by all four 3D variants.
// Writes H2D blob (src host data) and D2H expected blob (dst host data after copy),
// then emits the event record.
template <typename T>
static void capture_memcpy3d_impl(
    T& a, hrr_api_id_t api_id,
    const struct hipMemcpy3DParms* p, hipStream_t stream, bool is_async) {
  if (!p) {
    hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
    return;
  }
  std::memcpy(a.parms_bytes, p, sizeof(a.parms_bytes));
  size_t byte_count = 0;
  const bool sized = memcpy3d_byte_count(p, &byte_count);
  const bool host_side = (p->kind == hipMemcpyHostToDevice && p->srcPtr.ptr) ||
                         (p->kind == hipMemcpyDeviceToHost && p->dstPtr.ptr);
  // The runtime validates with the same wrapping product, so it can accept an
  // extent that overflows. Without a blob, replay cannot perform a host-side copy.
  if (!sized && host_side) {
    LogPrintfError(
        "[HRR capture] 3D copy extent %zux%zux%zu overflows size_t, so no blob can be "
        "sized for it: dropping the event and marking the capture INCOMPLETE.",
        p->extent.width, p->extent.height, p->extent.depth);
    hrr_cap::writer::mark_incomplete("3D copy extent overflows size_t");
    return;
  }

  if (p->kind == hipMemcpyHostToDevice && p->srcPtr.ptr && byte_count > 0 &&
      a.ret == hipSuccess) {
    // H2D: host source is valid at call time, so no stream sync is needed.
    // The blob holds the copied rows of the source rect, as the driver copies'
    // does, and replay copies from it with the host side made dense.
    auto h = write_host_rect_blob(p->srcPtr.ptr, p->srcPtr.pitch, p->srcPtr.ysize,
                                  p->srcPos.x, p->srcPos.y, p->srcPos.z, p->extent.width,
                                  p->extent.height, p->extent.depth);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
  } else if (p->kind == hipMemcpyDeviceToHost && p->dstPtr.ptr && byte_count > 0 &&
             a.ret == hipSuccess) {
    // D2H: real call already completed (sync API) or stream sync done below;
    // host buffer now holds GPU result — capture it as the expected output.
    // The blob holds the copied rows of the destination rect, so replay can
    // compare exactly those rows. A rejected copy is skipped: nothing validated
    // its rect, and reading it can reach past the caller's buffer. The null
    // stream is synchronised too, as in the driver 3D path below.
    if (is_async) {
      hipError_t sync_r = sync_for_d2h_snapshot(stream);
      if (sync_r != hipSuccess) {
        LogPrintfWarning("[HRR capture] hipStreamSynchronize failed (%d) — D2H 3D blob skipped",
                         sync_r);
        hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
        return;
      }
    }
    auto h = write_host_rect_blob(p->dstPtr.ptr, p->dstPtr.pitch, p->dstPtr.ysize,
                                  p->dstPos.x, p->dstPos.y, p->dstPos.z, p->extent.width,
                                  p->extent.height, p->extent.depth);
    a.d2h_hash_lo = h.lo;
    a.d2h_hash_hi = h.hi;
  }
  hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
}

// Success-gated like the 2D and driver-style copies below. A copy the runtime
// rejected has an extent nothing validated, so blobbing w*h*d bytes of its host
// side would read past the caller's buffer.
hipError_t capture_hipMemcpy3D(const struct hipMemcpy3DParms* p) {
  hipError_t r = g_real_table.hipMemcpy3D_fn(p);
  if (r != hipSuccess) return r;
  hrr_args_hipMemcpy3D a{};
  a.ret = static_cast<int32_t>(r);
  capture_memcpy3d_impl(a, HRR_API_HIPMEMCPY3D, p, nullptr, false);
  return r;
}

hipError_t capture_hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpy3DAsync_fn(p, stream);
  if (r != hipSuccess) return r;
  hrr_args_hipMemcpy3DAsync a{};
  a.ret    = static_cast<int32_t>(r);
  a.stream = reinterpret_cast<uint64_t>(stream);
  capture_memcpy3d_impl(a, HRR_API_HIPMEMCPY3DASYNC, p, stream, true);
  return r;
}

hipError_t capture_hipMemcpy3D_spt(const struct hipMemcpy3DParms* p) {
  hipError_t r = g_real_table.hipMemcpy3D_spt_fn(p);
  if (r != hipSuccess) return r;
  hrr_args_hipMemcpy3D_spt a{};
  a.ret = static_cast<int32_t>(r);
  capture_memcpy3d_impl(a, HRR_API_HIPMEMCPY3D_SPT, p, nullptr, false);
  return r;
}

hipError_t capture_hipMemcpy3DAsync_spt(const struct hipMemcpy3DParms* p, hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpy3DAsync_spt_fn(p, stream);
  if (r != hipSuccess) return r;
  hrr_args_hipMemcpy3DAsync_spt a{};
  a.ret    = static_cast<int32_t>(r);
  a.stream = reinterpret_cast<uint64_t>(stream);
  // The null stream of an _spt call is this thread's default stream, and that
  // is the one the D2H blob has to wait for.
  capture_memcpy3d_impl(a, HRR_API_HIPMEMCPY3DASYNC_SPT, p,
                        stream ? stream : hipStreamPerThread, true);
  return r;
}

// ---------------------------------------------------------------------------
// hipDrvMemcpy3D / hipDrvMemcpy3DAsync / hipDrvMemcpy2DUnaligned
// Driver-style struct-pointer copies. Mirror the hipMemcpy3D path: inline the
// HIP_MEMCPY3D / hip_Memcpy2D struct + snapshot the H2D source (or D2H expected)
// blob. Keyed off srcMemoryType/dstMemoryType (no single "kind" field).
// ---------------------------------------------------------------------------

// Byte footprint of the host side of a driver-style copy, measured from the host
// base pointer (srcHost / dstHost).
//
// The runtime addresses the host buffer as a pitched rect (amd::BufferRect):
//   row   = pitch ? pitch : width
//   slice = pitch*pitch_height ? pitch*pitch_height : row*height
//   first byte = z*slice + y*row + x
//   last byte  = first + (depth-1)*slice + (height-1)*row + width - 1
// The blob holds only the copied rows (write_host_rect_blob), for the H2D
// source and the D2H expected output alike, but the footprint still locates
// them, and a copy whose footprint does not fit in size_t is not recorded.
// For a copy the runtime accepted, footprint >= width*height*depth.
//
// Every step is checked, as replay's hrr_host_rect is: a wrapped footprint would
// place rows outside the buffer the copy was given.
struct HostRect {
  size_t row, slice, first, bytes;  // bytes == 0: nothing is copied
  bool ok;                          // false: the footprint does not fit in size_t
};

// a*b + c, refusing to wrap.
static bool size_mad(size_t a, size_t b, size_t c, size_t* out) {
  if (a != 0 && b > (SIZE_MAX - c) / a) return false;
  *out = a * b + c;
  return true;
}

static HostRect host_rect(size_t pitch, size_t pitch_height, size_t x, size_t y, size_t z,
                          size_t width, size_t height, size_t depth) {
  HostRect r{0, 0, 0, 0, false};
  if (width == 0 || height == 0 || depth == 0) {
    r.ok = true;
    return r;
  }
  r.row = (pitch != 0) ? pitch : width;
  if (r.row < width) r.row = width;  // defensive: degenerate pitch
  size_t dense = 0, yx = 0, last = 0;
  if (!size_mad(r.row, height, 0, &dense) || !size_mad(pitch, pitch_height, 0, &r.slice))
    return r;
  if (r.slice < dense) r.slice = dense;  // 0 => runtime default
  // first = z*slice + y*row + x, the offset of the first byte; bytes = first +
  // (depth-1)*slice + (height-1)*row + width.
  if (!size_mad(y, r.row, x, &yx) || !size_mad(z, r.slice, yx, &r.first) ||
      !size_mad(height - 1, r.row, width, &last) ||
      !size_mad(depth - 1, r.slice, last, &last) || !size_mad(1, last, r.first, &r.bytes))
    return r;
  r.ok = true;
  return r;
}

// SIZE_MAX for a footprint that overflows, so the caller still reaches
// write_host_rect_blob, which records the loss.
static size_t drvmemcpy_host_byte_count(size_t pitch, size_t pitch_height,
                                        size_t x, size_t y, size_t z,
                                        size_t width, size_t height, size_t depth) {
  const HostRect r = host_rect(pitch, pitch_height, x, y, z, width, height, depth);
  return r.ok ? r.bytes : SIZE_MAX;
}

// Blob of a host rect: its copied rows packed end to end, width*height*depth
// bytes whatever the pitches and offsets, so capture costs what the copy moves.
// The bytes before the first row and between rows and slices are never read:
// the copy does not touch them, they can hold unrelated data or lie on an
// unmapped guard page, and a sparse pitch can put two rows gigabytes apart.
// The archive header carries HRR_FILE_FLAG_PACKED_HOST_RECTS, and replay reads
// the blob with the host side made dense: pitch == width and no offsets.
static hrr_cap::Hash128 write_host_rect_blob(const void* base, size_t pitch,
                                             size_t pitch_height, size_t x, size_t y,
                                             size_t z, size_t width, size_t height,
                                             size_t depth) {
  const HostRect r = host_rect(pitch, pitch_height, x, y, z, width, height, depth);
  if (!r.ok) {
    LogPrintfWarning("[HRR capture] host copy rect %zux%zux%zu at pitch %zu overflows size_t",
                     width, height, depth, pitch);
    hrr_cap::writer::mark_incomplete("pitched host copy not recorded");
    return {0, 0};
  }
  if (r.bytes == 0) return {0, 0};
  // Neither product wraps: both are at most r.bytes, which fits.
  const size_t n = width * height * depth;
  const auto* src = static_cast<const uint8_t*>(base) + r.first;
  if ((height == 1 || r.row == width) && (depth == 1 || r.slice == width * height)) {
    return hrr_cap::writer::write_blob(src, n);  // dense: the rows are one run
  }
  std::unique_ptr<uint8_t[]> buf(new (std::nothrow) uint8_t[n]);
  if (!buf) {
    LogPrintfWarning("[HRR capture] cannot allocate %zu bytes to snapshot a pitched host copy",
                     n);
    hrr_cap::writer::mark_incomplete("pitched host copy not recorded");
    return {0, 0};
  }
  uint8_t* out = buf.get();
  for (size_t k = 0; k < depth; ++k) {
    for (size_t j = 0; j < height; ++j, out += width) {
      std::memcpy(out, src + k * r.slice + j * r.row, width);
    }
  }
  return hrr_cap::writer::write_blob(buf.get(), n);
}

template <typename T>
static void capture_drvmemcpy3d_impl(T& a, hrr_api_id_t api_id,
    const HIP_MEMCPY3D* p, hipStream_t stream, bool is_async) {
  if (!p) { hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a)); return; }
  std::memcpy(a.drv3d_bytes, p, sizeof(HIP_MEMCPY3D));
  if (p->srcMemoryType == hipMemoryTypeHost && p->srcHost) {
    size_t n = drvmemcpy_host_byte_count(p->srcPitch, p->srcHeight, p->srcXInBytes,
                                         p->srcY, p->srcZ, p->WidthInBytes,
                                         p->Height, p->Depth);
    if (n > 0) {
      auto h = write_host_rect_blob(p->srcHost, p->srcPitch, p->srcHeight, p->srcXInBytes,
                                    p->srcY, p->srcZ, p->WidthInBytes, p->Height, p->Depth);
      a.blob_hash_lo = h.lo; a.blob_hash_hi = h.hi;
    }
  } else if (p->dstMemoryType == hipMemoryTypeHost && p->dstHost &&
             p->srcMemoryType != hipMemoryTypeArray) {
    // D2H expected output, the copied rows of the destination rect.
    // An array source is skipped because playback declines array-typed rects,
    // so the blob would be an expected output nothing ever validates.
    size_t n = drvmemcpy_host_byte_count(p->dstPitch, p->dstHeight, p->dstXInBytes,
                                         p->dstY, p->dstZ, p->WidthInBytes,
                                         p->Height, p->Depth);
    if (n > 0) {
      // The null stream needs the sync too: hipDrvMemcpy3DAsync(p, nullptr) is
      // still asynchronous, so skipping it can snapshot dstHost before the copy
      // lands and record a stale expected output. hipStreamSynchronize(nullptr)
      // is valid and waits on the blocking streams, which is what the app itself
      // would have to do before reading dstHost.
      if (is_async) {
        hipError_t sync_r = sync_for_d2h_snapshot(stream);
        if (sync_r != hipSuccess) {
          LogPrintfWarning("[HRR capture] hipStreamSynchronize failed (%d): D2H drv blob skipped",
                           sync_r);
          hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
          return;
        }
      }
      auto h = write_host_rect_blob(p->dstHost, p->dstPitch, p->dstHeight, p->dstXInBytes,
                                    p->dstY, p->dstZ, p->WidthInBytes, p->Height, p->Depth);
      a.d2h_hash_lo = h.lo; a.d2h_hash_hi = h.hi;
    }
  }
  hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
}

template <typename T>
static void capture_drvmemcpy2d_impl(T& a, hrr_api_id_t api_id, const hip_Memcpy2D* p,
                                     bool dst_ready = true) {
  if (!p) { hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a)); return; }
  std::memcpy(a.drv2d_bytes, p, sizeof(hip_Memcpy2D));
  // hip_Memcpy2D is widened to a HIP_MEMCPY3D by the runtime with Depth == 1,
  // srcHeight/dstHeight == 0 (slice pitch defaults to row*Height) and a pitch
  // that falls back to x + WidthInBytes. See hip::getDrvMemcpy3DDesc().
  if (p->srcMemoryType == hipMemoryTypeHost && p->srcHost) {
    size_t pitch = p->srcPitch ? p->srcPitch : p->srcXInBytes + p->WidthInBytes;
    size_t n = drvmemcpy_host_byte_count(pitch, /*pitch_height=*/0, p->srcXInBytes,
                                         p->srcY, /*z=*/0, p->WidthInBytes,
                                         p->Height, /*depth=*/1);
    if (n > 0) {
      auto h = write_host_rect_blob(p->srcHost, pitch, /*pitch_height=*/0, p->srcXInBytes,
                                    p->srcY, /*z=*/0, p->WidthInBytes, p->Height, /*depth=*/1);
      a.blob_hash_lo = h.lo; a.blob_hash_hi = h.hi;
    }
  } else if (dst_ready && p->dstMemoryType == hipMemoryTypeHost && p->dstHost &&
             p->srcMemoryType != hipMemoryTypeArray) {
    size_t pitch = p->dstPitch ? p->dstPitch : p->dstXInBytes + p->WidthInBytes;
    size_t n = drvmemcpy_host_byte_count(pitch, /*pitch_height=*/0, p->dstXInBytes,
                                         p->dstY, /*z=*/0, p->WidthInBytes,
                                         p->Height, /*depth=*/1);
    if (n > 0) {
      auto h = write_host_rect_blob(p->dstHost, pitch, /*pitch_height=*/0, p->dstXInBytes,
                                    p->dstY, /*z=*/0, p->WidthInBytes, p->Height, /*depth=*/1);
      a.d2h_hash_lo = h.lo; a.d2h_hash_hi = h.hi;
    }
  }
  hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
}

// The `r == hipSuccess` gate matches capture_hipMemcpy2D and the generated
// shims these replaced: a failed copy is not recorded at all. Recording it
// would make playback dispatch a handler that returns the failure, and any
// non-success handler return is fatal to the whole replay. For D2H it would
// also blob a buffer the copy never wrote as "expected output".
hipError_t capture_hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy) {
  hipError_t r = g_real_table.hipDrvMemcpy3D_fn(pCopy);
  if (r != hipSuccess) return r;
  hrr_args_hipDrvMemcpy3D a{};
  a.ret = static_cast<int32_t>(r);
  capture_drvmemcpy3d_impl(a, HRR_API_HIPDRVMEMCPY3D, pCopy, nullptr, false);
  return r;
}

hipError_t capture_hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream) {
  hipError_t r = g_real_table.hipDrvMemcpy3DAsync_fn(pCopy, stream);
  if (r != hipSuccess) return r;
  hrr_args_hipDrvMemcpy3DAsync a{};
  a.ret    = static_cast<int32_t>(r);
  a.stream = reinterpret_cast<uint64_t>(stream);
  capture_drvmemcpy3d_impl(a, HRR_API_HIPDRVMEMCPY3DASYNC, pCopy, stream, true);
  return r;
}

hipError_t capture_hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy) {
  hipError_t r = g_real_table.hipDrvMemcpy2DUnaligned_fn(pCopy);
  if (r != hipSuccess) return r;
  hrr_args_hipDrvMemcpy2DUnaligned a{};
  a.ret = static_cast<int32_t>(r);
  capture_drvmemcpy2d_impl(a, HRR_API_HIPDRVMEMCPY2DUNALIGNED, pCopy);
  return r;
}

// The same descriptor under the runtime-API spellings. Both were recording the
// pointer and nothing behind it, which is why they were no-ops at replay.
hipError_t capture_hipMemcpyParam2D(const hip_Memcpy2D* pCopy) {
  hipError_t r = g_real_table.hipMemcpyParam2D_fn(pCopy);
  if (r != hipSuccess) return r;
  hrr_args_hipMemcpyParam2D a{};
  a.ret = static_cast<int32_t>(r);
  capture_drvmemcpy2d_impl(a, HRR_API_HIPMEMCPYPARAM2D, pCopy);
  return r;
}

hipError_t capture_hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy,
                                         hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpyParam2DAsync_fn(pCopy, stream);
  if (r != hipSuccess) return r;
  hrr_args_hipMemcpyParam2DAsync a{};
  a.ret = static_cast<int32_t>(r);
  a.stream = reinterpret_cast<uint64_t>(stream);
  // A D2H blob taken before the copy lands would record a stale expected
  // output, the same reason the async 3D spelling synchronises first. The null
  // stream is asynchronous too, and a failed sync leaves no blob to take.
  bool dst_ready = true;
  if (pCopy && pCopy->dstMemoryType == hipMemoryTypeHost) {
    hipError_t sync_r = sync_for_d2h_snapshot(stream);
    if (sync_r != hipSuccess) {
      LogPrintfWarning("[HRR capture] hipStreamSynchronize failed (%d): D2H param 2D blob skipped",
                       sync_r);
      dst_ready = false;
    }
  }
  capture_drvmemcpy2d_impl(a, HRR_API_HIPMEMCPYPARAM2DASYNC, pCopy, dst_ready);
  return r;
}

// ---------------------------------------------------------------------------
// hipMemcpy2D / hipMemcpy2DAsync — pitched H2D blob + D2H expected blob
//
// The 2D copy moves `height` rows of `width` bytes, source rows spaced by
// `spitch` and destination rows by `dpitch`. The host side is a contiguous
// buffer whose byte extent is pitch*(height-1)+width — the bytes that actually
// belong to the copy. We snapshot exactly that so replay can substitute the
// (capture-time, untranslatable) host VA, and validate D2H output.
// ---------------------------------------------------------------------------

// SIZE_MAX when the extent overflows, as drvmemcpy_host_byte_count.
static size_t memcpy2d_host_byte_count(size_t pitch, size_t width, size_t height) {
  if (height == 0 || width == 0) return 0;
  if (pitch < width) pitch = width;  // defensive: degenerate pitch
  size_t n = 0;
  return size_mad(pitch, height - 1, width, &n) ? n : SIZE_MAX;
}

// Shared blob logic for both 2D variants. Writes the H2D source blob or the D2H
// expected-output blob into `a`, then emits the event.
template <typename T>
static void capture_memcpy2d_impl(
    T& a, hrr_api_id_t api_id, const void* dst, size_t dpitch,
    const void* src, size_t spitch, size_t width, size_t height,
    hipMemcpyKind kind, hipStream_t stream, bool is_async) {
  if (kind == hipMemcpyHostToDevice && src) {
    size_t n = memcpy2d_host_byte_count(spitch, width, height);
    if (n > 0) {
      auto h = write_host_rect_blob(src, spitch, 0, 0, 0, 0, width, height, 1);
      a.blob_hash_lo = h.lo;
      a.blob_hash_hi = h.hi;
      hrr_trace_h2d(is_async ? "hipMemcpy2DAsync" : "hipMemcpy2D", dst, n);
    }
  } else if (kind == hipMemcpyDeviceToHost && dst) {
    size_t n = memcpy2d_host_byte_count(dpitch, width, height);
    if (n > 0) {
      if (is_async) {
        hipError_t sync_r = sync_for_d2h_snapshot(stream);
        if (sync_r != hipSuccess) {
          LogPrintfWarning("[HRR capture] hipStreamSynchronize failed (%d) — D2H 2D blob skipped",
                           sync_r);
          hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
          return;
        }
      }
      auto h = write_host_rect_blob(dst, dpitch, 0, 0, 0, 0, width, height, 1);
      a.d2h_hash_lo = h.lo;
      a.d2h_hash_hi = h.hi;
    }
  }
  hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
}

hipError_t capture_hipMemcpy2D(void* dst, size_t dpitch, const void* src,
                               size_t spitch, size_t width, size_t height,
                               hipMemcpyKind kind) {
  hipError_t r = g_real_table.hipMemcpy2D_fn(dst, dpitch, src, spitch, width, height, kind);
  hrr_args_hipMemcpy2D a{};
  a.ret    = static_cast<int32_t>(r);
  a.dst    = reinterpret_cast<uint64_t>(dst);
  a.dpitch = static_cast<uint64_t>(dpitch);
  a.src    = reinterpret_cast<uint64_t>(src);
  a.spitch = static_cast<uint64_t>(spitch);
  a.width  = static_cast<uint64_t>(width);
  a.height = static_cast<uint64_t>(height);
  a.kind   = static_cast<int32_t>(kind);
  if (r == hipSuccess)
    capture_memcpy2d_impl(a, HRR_API_HIPMEMCPY2D, dst, dpitch, src, spitch,
                          width, height, kind, nullptr, false);
  return r;
}

hipError_t capture_hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                                    size_t spitch, size_t width, size_t height,
                                    hipMemcpyKind kind, hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpy2DAsync_fn(dst, dpitch, src, spitch,
                                                  width, height, kind, stream);
  hrr_args_hipMemcpy2DAsync a{};
  a.ret    = static_cast<int32_t>(r);
  a.dst    = reinterpret_cast<uint64_t>(dst);
  a.dpitch = static_cast<uint64_t>(dpitch);
  a.src    = reinterpret_cast<uint64_t>(src);
  a.spitch = static_cast<uint64_t>(spitch);
  a.width  = static_cast<uint64_t>(width);
  a.height = static_cast<uint64_t>(height);
  a.kind   = static_cast<int32_t>(kind);
  a.stream = reinterpret_cast<uint64_t>(stream);
  if (r == hipSuccess)
    capture_memcpy2d_impl(a, HRR_API_HIPMEMCPY2DASYNC, dst, dpitch, src, spitch,
                          width, height, kind, stream, true);
  return r;
}

// ---------------------------------------------------------------------------
// hipArrayCreate / hipArray3DCreate — inline descriptor; record handle
// ---------------------------------------------------------------------------

hipError_t capture_hipArrayCreate(hipArray_t* pHandle,
                                  const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  hipError_t r = g_real_table.hipArrayCreate_fn(pHandle, pAllocateArray);
  if (r != hipSuccess) return r;
  hrr_args_hipArrayCreate a{};
  a.ret     = static_cast<int32_t>(r);
  a.pHandle = reinterpret_cast<uint64_t>(*pHandle);
  if (pAllocateArray)
    std::memcpy(a.array_desc_bytes, pAllocateArray, sizeof(a.array_desc_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPARRAYCREATE, &a.hdr, sizeof(a));
  return r;
}

hipError_t capture_hipArray3DCreate(hipArray_t* array,
                                    const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  hipError_t r = g_real_table.hipArray3DCreate_fn(array, pAllocateArray);
  if (r != hipSuccess) return r;
  hrr_args_hipArray3DCreate a{};
  a.ret   = static_cast<int32_t>(r);
  a.array = reinterpret_cast<uint64_t>(*array);
  if (pAllocateArray)
    std::memcpy(a.array3d_desc_bytes, pAllocateArray, sizeof(a.array3d_desc_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPARRAY3DCREATE, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// hipStreamSetAttribute — inline hipStreamAttrValue (64 bytes)
// ---------------------------------------------------------------------------

hipError_t capture_hipStreamSetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                         const hipStreamAttrValue* value) {
  hipError_t r = g_real_table.hipStreamSetAttribute_fn(stream, attr, value);
  if (r != hipSuccess) return r;
  hrr_args_hipStreamSetAttribute a{};
  a.ret    = static_cast<int32_t>(r);
  a.stream = reinterpret_cast<uint64_t>(stream);
  a.attr   = static_cast<int32_t>(attr);
  if (value) std::memcpy(a.stream_attr_bytes, value, sizeof(a.stream_attr_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPSTREAMSETATTRIBUTE, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// hipMemGetAllocationGranularity — inline hipMemAllocationProp (32 bytes)
// ---------------------------------------------------------------------------

hipError_t capture_hipMemGetAllocationGranularity(size_t* granularity,
                                                   const hipMemAllocationProp* prop,
                                                   hipMemAllocationGranularity_flags option) {
  hipError_t r = g_real_table.hipMemGetAllocationGranularity_fn(granularity, prop, option);
  if (r != hipSuccess) return r;
  hrr_args_hipMemGetAllocationGranularity a{};
  a.ret         = static_cast<int32_t>(r);
  a.granularity = reinterpret_cast<uint64_t>(granularity);
  a.option      = static_cast<uint32_t>(option);
  if (prop) std::memcpy(a.alloc_prop_bytes, prop, sizeof(a.alloc_prop_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPMEMGETALLOCATIONGRANULARITY, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// hipMemPoolSetAccess / hipMemSetAccess — inline first hipMemAccessDesc entry
// ---------------------------------------------------------------------------

hipError_t capture_hipMemPoolSetAccess(hipMemPool_t mem_pool,
                                       const hipMemAccessDesc* desc_list, size_t count) {
  hipError_t r = g_real_table.hipMemPoolSetAccess_fn(mem_pool, desc_list, count);
  if (r != hipSuccess) return r;
  hrr_args_hipMemPoolSetAccess a{};
  a.ret      = static_cast<int32_t>(r);
  a.mem_pool = reinterpret_cast<uint64_t>(mem_pool);
  a.count    = static_cast<uint64_t>(count);
  if (desc_list && count > 0)
    std::memcpy(a.access_desc_bytes, desc_list, sizeof(a.access_desc_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPMEMPOOLSETACCESS, &a.hdr, sizeof(a));
  return r;
}

hipError_t capture_hipMemSetAccess(void* ptr, size_t size,
                                   const hipMemAccessDesc* desc, size_t count) {
  hipError_t r = g_real_table.hipMemSetAccess_fn(ptr, size, desc, count);
  if (r != hipSuccess) return r;
  hrr_args_hipMemSetAccess a{};
  a.ret   = static_cast<int32_t>(r);
  a.ptr   = reinterpret_cast<uint64_t>(ptr);
  a.size  = static_cast<uint64_t>(size);
  a.count = static_cast<uint64_t>(count);
  if (desc && count > 0)
    std::memcpy(a.access_desc_bytes, desc, sizeof(a.access_desc_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPMEMSETACCESS, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// Graph kernel nodes — struct fields plus the kernel tail
//
// The generated shim already carries the dependency array and the 64 bytes of
// hipKernelNodeParams (DEREF_FIELDS), which is enough for the dimensions and
// the shared-memory size. What it cannot carry is `func` (a host address) and
// `kernelParams` (a host array of pointers to argument values); those are the
// tail, written the same way a launch writes them.
// ---------------------------------------------------------------------------

template <typename A>
static void write_kernel_node_event(A& a, hrr_api_id_t api_id,
                                    const hipKernelNodeParams* p) {
  std::vector<uint8_t> payload(sizeof(A));
  std::memcpy(payload.data(), &a, sizeof(A));
  append_kernel_node_tail(payload, p ? p->func : nullptr,
                          p ? p->kernelParams : nullptr, p ? p->extra : nullptr);
  hrr_cap::writer::write_event_raw(
      api_id, reinterpret_cast<hrr_event_header*>(payload.data()),
      static_cast<uint32_t>(payload.size()));
}

hipError_t capture_hipGraphAddKernelNode(hipGraphNode_t* pGraphNode,
                                         hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies,
                                         const hipKernelNodeParams* pNodeParams) {
  hipError_t r = g_real_table.hipGraphAddKernelNode_fn(
      pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  hrr_args_hipGraphAddKernelNode a{};
  a.ret              = static_cast<int32_t>(r);
  a.graph            = reinterpret_cast<uint64_t>(graph);
  a.numDependencies  = static_cast<uint64_t>(numDependencies);
  a.pGraphNode       = reinterpret_cast<uint64_t>(
      (r == hipSuccess && pGraphNode) ? *pGraphNode : nullptr);
  if (pDependencies && numDependencies > 0) {
    uint32_t n = static_cast<uint32_t>(numDependencies);
    const uint32_t cap = sizeof(a.pDependencies_bytes) / sizeof(hipGraphNode_t);
    if (n > cap) n = cap;
    std::memcpy(a.pDependencies_bytes, pDependencies, n * sizeof(hipGraphNode_t));
    a.pDependencies_n       = n;
    a.pDependencies_present = 1;
  }
  if (pNodeParams) {
    std::memcpy(a.pNodeParams_bytes, pNodeParams, sizeof(hipKernelNodeParams));
    a.pNodeParams_present = 1;
  }
  write_kernel_node_event(a, HRR_API_HIPGRAPHADDKERNELNODE, pNodeParams);
  return r;
}

// A copy naming a __device__ global. The symbol is a host shadow address the
// replay cannot use, but the symbol sweep records its name, so all this shim
// has to add is the host buffer on the other side of the copy — which exists
// only when the copy is to the symbol from host memory.
static bool memcpy_src_is_host(const void* src, hipMemcpyKind kind) {
  if (kind == hipMemcpyHostToDevice || kind == hipMemcpyHostToHost) return true;
  if (kind != hipMemcpyDefault) return false;
  if (!src || !g_real_table.hipPointerGetAttributes_fn) return false;
  hipPointerAttribute_t attr{};
  if (g_real_table.hipPointerGetAttributes_fn(&attr, src) != hipSuccess)
    return true;  // unregistered memory is ordinary host memory
  return attr.type != hipMemoryTypeDevice && attr.type != hipMemoryTypeUnified;
}

template <typename A>
static void fill_node_deps(A& a, const hipGraphNode_t* deps, size_t n_deps) {
  if (!deps || n_deps == 0) return;
  uint32_t n = static_cast<uint32_t>(n_deps);
  const uint32_t cap = sizeof(a.pDependencies_bytes) / sizeof(hipGraphNode_t);
  if (n > cap) n = cap;
  std::memcpy(a.pDependencies_bytes, deps, n * sizeof(hipGraphNode_t));
  a.pDependencies_n       = n;
  a.pDependencies_present = 1;
}

hipError_t capture_hipGraphAddMemcpyNodeToSymbol(
    hipGraphNode_t* pGraphNode, hipGraph_t graph,
    const hipGraphNode_t* pDependencies, size_t numDependencies,
    const void* symbol, const void* src, size_t count, size_t offset,
    hipMemcpyKind kind) {
  hipError_t r = g_real_table.hipGraphAddMemcpyNodeToSymbol_fn(
      pGraphNode, graph, pDependencies, numDependencies, symbol, src, count,
      offset, kind);
  hrr_args_hipGraphAddMemcpyNodeToSymbol a{};
  a.ret             = static_cast<int32_t>(r);
  a.graph           = reinterpret_cast<uint64_t>(graph);
  a.numDependencies = static_cast<uint64_t>(numDependencies);
  a.symbol          = reinterpret_cast<uint64_t>(symbol);
  a.src             = reinterpret_cast<uint64_t>(src);
  a.count           = static_cast<uint64_t>(count);
  a.offset          = static_cast<uint64_t>(offset);
  a.kind            = static_cast<int32_t>(kind);
  a.pGraphNode      = reinterpret_cast<uint64_t>(
      (r == hipSuccess && pGraphNode) ? *pGraphNode : nullptr);
  fill_node_deps(a, pDependencies, numDependencies);
  if (src && count && memcpy_src_is_host(src, kind)) {
    auto h = hrr_cap::writer::write_blob(src, count);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
  }
  hrr_cap::writer::write_event_raw(HRR_API_HIPGRAPHADDMEMCPYNODETOSYMBOL,
                                   &a.hdr, sizeof(a));
  return r;
}

hipError_t capture_hipGraphAddMemcpyNodeFromSymbol(
    hipGraphNode_t* pGraphNode, hipGraph_t graph,
    const hipGraphNode_t* pDependencies, size_t numDependencies, void* dst,
    const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
  hipError_t r = g_real_table.hipGraphAddMemcpyNodeFromSymbol_fn(
      pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count,
      offset, kind);
  hrr_args_hipGraphAddMemcpyNodeFromSymbol a{};
  a.ret             = static_cast<int32_t>(r);
  a.graph           = reinterpret_cast<uint64_t>(graph);
  a.numDependencies = static_cast<uint64_t>(numDependencies);
  a.dst             = reinterpret_cast<uint64_t>(dst);
  a.symbol          = reinterpret_cast<uint64_t>(symbol);
  a.count           = static_cast<uint64_t>(count);
  a.offset          = static_cast<uint64_t>(offset);
  a.kind            = static_cast<int32_t>(kind);
  a.pGraphNode      = reinterpret_cast<uint64_t>(
      (r == hipSuccess && pGraphNode) ? *pGraphNode : nullptr);
  fill_node_deps(a, pDependencies, numDependencies);
  hrr_cap::writer::write_event_raw(HRR_API_HIPGRAPHADDMEMCPYNODEFROMSYMBOL,
                                   &a.hdr, sizeof(a));
  return r;
}

// A JIT linker input. The image is the whole point of the call and was
// recorded as an address, so replay had nothing to link even once the linker
// state itself was tracked.
hipError_t capture_hipLinkAddData(hipLinkState_t state, hipJitInputType type,
                                  void* data, size_t size, const char* name,
                                  unsigned int numOptions,
                                  hipJitOption* options, void** optionValues) {
  hipError_t r = g_real_table.hipLinkAddData_fn(state, type, data, size, name,
                                                numOptions, options,
                                                optionValues);
  hrr_args_hipLinkAddData a{};
  a.ret        = static_cast<int32_t>(r);
  a.state      = reinterpret_cast<uint64_t>(state);
  a.type       = static_cast<int32_t>(type);
  a.data       = reinterpret_cast<uint64_t>(data);
  a.size       = static_cast<uint64_t>(size);
  a.name       = reinterpret_cast<uint64_t>(name);
  a.numOptions = static_cast<uint32_t>(numOptions);
  a.options    = reinterpret_cast<uint64_t>(options);
  a.optionValues = reinterpret_cast<uint64_t>(optionValues);
  if (name) {
    size_t n = std::strlen(name);
    if (n > 255u) n = 255u;
    std::memcpy(a.name_bytes, name, n);
    a.name_bytes[n] = '\0';
    a.name_present = 1;
  }
  if (options && numOptions) {
    uint32_t n = numOptions > 32u ? 32u : numOptions;
    std::memcpy(a.options_bytes, options, n * sizeof(hipJitOption));
    a.options_n = n;
    a.options_present = 1;
  }
  if (optionValues && numOptions) {
    uint32_t n = numOptions > 32u ? 32u : numOptions;
    std::memcpy(a.optionValues_bytes, optionValues, n * sizeof(void*));
    a.optionValues_n = n;
    a.optionValues_present = 1;
  }
  if (data && size) {
    auto h = hrr_cap::writer::write_blob(data, size);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
  }
  hrr_cap::writer::write_event_raw(HRR_API_HIPLINKADDDATA, &a.hdr, sizeof(a));
  return r;
}

// The mutation spellings of the same copy, node-level and exec-level. Only the
// to-symbol direction has a host buffer worth recording.
hipError_t capture_hipGraphMemcpyNodeSetParamsToSymbol(
    hipGraphNode_t node, const void* symbol, const void* src, size_t count,
    size_t offset, hipMemcpyKind kind) {
  hipError_t r = g_real_table.hipGraphMemcpyNodeSetParamsToSymbol_fn(
      node, symbol, src, count, offset, kind);
  hrr_args_hipGraphMemcpyNodeSetParamsToSymbol a{};
  a.ret    = static_cast<int32_t>(r);
  a.node   = reinterpret_cast<uint64_t>(node);
  a.symbol = reinterpret_cast<uint64_t>(symbol);
  a.src    = reinterpret_cast<uint64_t>(src);
  a.count  = static_cast<uint64_t>(count);
  a.offset = static_cast<uint64_t>(offset);
  a.kind   = static_cast<int32_t>(kind);
  if (src && count && memcpy_src_is_host(src, kind)) {
    auto h = hrr_cap::writer::write_blob(src, count);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
  }
  hrr_cap::writer::write_event_raw(
      HRR_API_HIPGRAPHMEMCPYNODESETPARAMSTOSYMBOL, &a.hdr, sizeof(a));
  return r;
}

hipError_t capture_hipGraphExecMemcpyNodeSetParamsToSymbol(
    hipGraphExec_t hGraphExec, hipGraphNode_t node, const void* symbol,
    const void* src, size_t count, size_t offset, hipMemcpyKind kind) {
  hipError_t r = g_real_table.hipGraphExecMemcpyNodeSetParamsToSymbol_fn(
      hGraphExec, node, symbol, src, count, offset, kind);
  hrr_args_hipGraphExecMemcpyNodeSetParamsToSymbol a{};
  a.ret        = static_cast<int32_t>(r);
  a.hGraphExec = reinterpret_cast<uint64_t>(hGraphExec);
  a.node       = reinterpret_cast<uint64_t>(node);
  a.symbol     = reinterpret_cast<uint64_t>(symbol);
  a.src        = reinterpret_cast<uint64_t>(src);
  a.count      = static_cast<uint64_t>(count);
  a.offset     = static_cast<uint64_t>(offset);
  a.kind       = static_cast<int32_t>(kind);
  if (src && count && memcpy_src_is_host(src, kind)) {
    auto h = hrr_cap::writer::write_blob(src, count);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
  }
  hrr_cap::writer::write_event_raw(
      HRR_API_HIPGRAPHEXECMEMCPYNODESETPARAMSTOSYMBOL, &a.hdr, sizeof(a));
  return r;
}

hipError_t capture_hipGraphKernelNodeSetParams(
    hipGraphNode_t node, const hipKernelNodeParams* pNodeParams) {
  hipError_t r = g_real_table.hipGraphKernelNodeSetParams_fn(node, pNodeParams);
  hrr_args_hipGraphKernelNodeSetParams a{};
  a.ret  = static_cast<int32_t>(r);
  a.node = reinterpret_cast<uint64_t>(node);
  if (pNodeParams) {
    std::memcpy(a.pNodeParams_bytes, pNodeParams, sizeof(hipKernelNodeParams));
    a.pNodeParams_present = 1;
  }
  write_kernel_node_event(a, HRR_API_HIPGRAPHKERNELNODESETPARAMS, pNodeParams);
  return r;
}

hipError_t capture_hipGraphExecKernelNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t node,
    const hipKernelNodeParams* pNodeParams) {
  hipError_t r = g_real_table.hipGraphExecKernelNodeSetParams_fn(
      hGraphExec, node, pNodeParams);
  hrr_args_hipGraphExecKernelNodeSetParams a{};
  a.ret        = static_cast<int32_t>(r);
  a.hGraphExec = reinterpret_cast<uint64_t>(hGraphExec);
  a.node       = reinterpret_cast<uint64_t>(node);
  if (pNodeParams) {
    std::memcpy(a.pNodeParams_bytes, pNodeParams, sizeof(hipKernelNodeParams));
    a.pNodeParams_present = 1;
  }
  write_kernel_node_event(a, HRR_API_HIPGRAPHEXECKERNELNODESETPARAMS,
                          pNodeParams);
  return r;
}

// ---------------------------------------------------------------------------
// Graph batch-memory-operation nodes — struct fields plus the op-array tail
//
// hipBatchMemOpNodeParams is {ctx, count, paramArray, flags}: the operations
// are a second pointer hop, so the 32 bytes DEREF_FIELDS carries name an array
// in the capturing process and nothing else. The array follows the struct as
// a tail: u32 count, u32 stride, then count entries.
// ---------------------------------------------------------------------------

template <typename A>
static void write_batch_memop_node_event(A& a, hrr_api_id_t api_id,
                                         const hipBatchMemOpNodeParams* p) {
  std::vector<uint8_t> payload(sizeof(A));
  std::memcpy(payload.data(), &a, sizeof(A));
  const uint32_t n = (p && p->paramArray) ? p->count : 0u;
  const uint32_t stride =
      n ? static_cast<uint32_t>(sizeof(hipStreamBatchMemOpParams)) : 0u;
  const auto push_u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; i++) payload.push_back(static_cast<uint8_t>(v >> (i*8)));
  };
  push_u32(n);
  push_u32(stride);
  if (n) {
    const auto* q = reinterpret_cast<const uint8_t*>(p->paramArray);
    payload.insert(payload.end(), q, q + static_cast<size_t>(n) * stride);
  }
  hrr_cap::writer::write_event_raw(
      api_id, reinterpret_cast<hrr_event_header*>(payload.data()),
      static_cast<uint32_t>(payload.size()));
}

// The dependency array is the same shape in every hipGraphAdd*Node, so the
// hand-written shims fill it the same way the generated ones do.
template <typename A>
static void fill_node_dependencies(A& a, const hipGraphNode_t* deps,
                                   size_t num_deps) {
  if (!deps || num_deps == 0) return;
  uint32_t n = static_cast<uint32_t>(num_deps);
  const uint32_t cap = sizeof(a.dependencies_bytes) / sizeof(hipGraphNode_t);
  if (n > cap) n = cap;
  std::memcpy(a.dependencies_bytes, deps, n * sizeof(hipGraphNode_t));
  a.dependencies_n       = n;
  a.dependencies_present = 1;
}

hipError_t capture_hipGraphAddBatchMemOpNode(
    hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
    const hipGraphNode_t* dependencies, size_t numDependencies,
    const hipBatchMemOpNodeParams* nodeParams) {
  hipError_t r = g_real_table.hipGraphAddBatchMemOpNode_fn(
      phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
  hrr_args_hipGraphAddBatchMemOpNode a{};
  a.ret             = static_cast<int32_t>(r);
  a.hGraph          = reinterpret_cast<uint64_t>(hGraph);
  a.numDependencies = static_cast<uint64_t>(numDependencies);
  a.phGraphNode     = reinterpret_cast<uint64_t>(
      (r == hipSuccess && phGraphNode) ? *phGraphNode : nullptr);
  fill_node_dependencies(a, dependencies, numDependencies);
  if (nodeParams) {
    std::memcpy(a.nodeParams_bytes, nodeParams, sizeof(hipBatchMemOpNodeParams));
    a.nodeParams_present = 1;
  }
  write_batch_memop_node_event(a, HRR_API_HIPGRAPHADDBATCHMEMOPNODE, nodeParams);
  return r;
}

hipError_t capture_hipGraphBatchMemOpNodeSetParams(
    hipGraphNode_t hNode, hipBatchMemOpNodeParams* nodeParams) {
  hipError_t r =
      g_real_table.hipGraphBatchMemOpNodeSetParams_fn(hNode, nodeParams);
  hrr_args_hipGraphBatchMemOpNodeSetParams a{};
  a.ret   = static_cast<int32_t>(r);
  a.hNode = reinterpret_cast<uint64_t>(hNode);
  if (nodeParams) {
    std::memcpy(a.nodeParams_bytes, nodeParams, sizeof(hipBatchMemOpNodeParams));
    a.nodeParams_present = 1;
  }
  write_batch_memop_node_event(
      a, HRR_API_HIPGRAPHBATCHMEMOPNODESETPARAMS, nodeParams);
  return r;
}

hipError_t capture_hipGraphExecBatchMemOpNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipBatchMemOpNodeParams* nodeParams) {
  hipError_t r = g_real_table.hipGraphExecBatchMemOpNodeSetParams_fn(
      hGraphExec, hNode, nodeParams);
  hrr_args_hipGraphExecBatchMemOpNodeSetParams a{};
  a.ret        = static_cast<int32_t>(r);
  a.hGraphExec = reinterpret_cast<uint64_t>(hGraphExec);
  a.hNode      = reinterpret_cast<uint64_t>(hNode);
  if (nodeParams) {
    std::memcpy(a.nodeParams_bytes, nodeParams, sizeof(hipBatchMemOpNodeParams));
    a.nodeParams_present = 1;
  }
  write_batch_memop_node_event(
      a, HRR_API_HIPGRAPHEXECBATCHMEMOPNODESETPARAMS, nodeParams);
  return r;
}

// ---------------------------------------------------------------------------
// Install / uninstall (build_table functions live in hip_capture_generated.cpp)
// ---------------------------------------------------------------------------

void hip_capture_install(HipDispatchTable* target) {
  // A caller that lost the hip_capture_build_table() guard race would otherwise
  // publish a still-zeroed g_cap_table, nulling every slot of the live table.
  if (!g_cap_table_ready.load(std::memory_order_acquire)) return;
  if (g_installed.exchange(true)) return;
  if (!target) target = const_cast<HipDispatchTable*>(hip::GetHipDispatchTable());
  std::memcpy(target, &g_cap_table, sizeof(HipDispatchTable));
}

void hip_capture_uninstall() {
  if (!g_installed.exchange(false)) return;
  std::memcpy(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()),
              &g_real_table, sizeof(HipDispatchTable));
  // The compiler shims go as well, or a forked child whose archive failed to
  // open keeps running them with capture off.
  if (g_compiler_installed.exchange(false))
    std::memcpy(const_cast<HipCompilerDispatchTable*>(hip::GetHipCompilerDispatchTable()),
                &g_real_compiler_table, sizeof(HipCompilerDispatchTable));
}

// ---------------------------------------------------------------------------
// Init / Shutdown
// ---------------------------------------------------------------------------

// Record a single fat binary blob as a HRR_API_HIPREGISTERFATBINARY event.
// blob_ptr is a StatCO module key: the unwrapped bundle for HIPF, the wrapper
// for HIPK. fat_binary_blob_ptr() accepts either (see its comment).
// These were registered before HIP initialized and are not digested yet, so
// unlike the live shim there is no HIP verdict to check, only the header, and a
// bundle with no code object for this host is still recorded.
static void record_fat_binary_blob(const void* blob_ptr) {
  blob_ptr = fat_binary_blob_ptr(blob_ptr);
  if (!blob_ptr) return;
  size_t blob_size = compute_bundle_size(blob_ptr);
  if (blob_size == 0) return;

  hrr_args___hipRegisterFatBinary a{};
  a.ret      = 0;  // handle not meaningful at init time
  a.blob_size = static_cast<uint64_t>(blob_size);
  auto h = hrr_cap::writer::write_blob(blob_ptr, blob_size);
  a.blob_hash_lo = h.lo;
  a.blob_hash_hi = h.hi;
  hrr_cap::writer::write_event_raw(HRR_API_HIPREGISTERFATBINARY, &a.hdr, sizeof(a));
}

// Record one registered __device__ global as a HRR_API_HIPREGISTERVAR event.
//
// The name is what survives the process boundary; the host shadow address and
// the capture-time device address are recorded beside it so replay can map
// both onto the global it resolves by name. Without the device address a
// recorded hipMemcpy reading the symbol has a source that translates to
// nothing, which is what made hipMemcpyFromSymbolAsync fail as an inner copy.
static void record_registered_var(const void* host_var, const char* name,
                                  size_t size, const void* dev_ptr) {
  if (!host_var || !name || !*name) return;

  hrr_args___hipRegisterVar a{};
  a.var       = reinterpret_cast<uint64_t>(host_var);
  a.hostVar   = reinterpret_cast<uint64_t>(host_var);
  a.deviceVar = reinterpret_cast<uint64_t>(name);
  a.size      = static_cast<uint64_t>(size);
  a.global    = 1;
  a.dev_addr  = reinterpret_cast<uint64_t>(dev_ptr);

  size_t n = std::strlen(name);
  if (n > 255u) n = 255u;
  std::memcpy(a.deviceVar_bytes, name, n);
  a.deviceVar_bytes[n] = '\0';
  a.deviceVar_present = 1;

  hrr_cap::writer::write_event_raw(HRR_API_HIPREGISTERVAR, &a.hdr, sizeof(a));
}

// ---------------------------------------------------------------------------
// Where runtime dispatch-table capture gets installed, and why it is here.
//
// The exported entry points in hip_table_interface.cpp read the table and then
// call through it:
//
//     return hip::GetHipDispatchTable()->hipMalloc_fn(ptr, size);
//
// hip::init() — and with it hip_capture_init() — does not run until the callee
// is already executing, inside HIP_INIT_API. Installing from hip_capture_init()
// therefore always misses the process's first HIP call: its slot was loaded
// before the shims existed. When that first call is a hipMalloc the allocation
// is absent from the archive and replay aborts translating a pointer it never
// saw.
//
// Installing from UpdateDispatchTable() closes that window. It is the point
// where every slot has just been given its real function pointer, and it runs
// inside the initialiser of the function-local static that GetHipDispatchTable()
// returns, so no caller can have loaded a slot yet.
//
// This is deliberately not the static-init hook that used to live here. That
// one ran a DSO constructor which called GetHipDispatchTable() at library load,
// forcing the table build and rocprofiler registration to happen before
// amd::Runtime::init() / Flag::init() and pulling host stacks (Python import
// torch, then multiprocessing spawn) into an ordering where the GPU runtime
// looked "already initialized" before child processes started. The hook below
// forces nothing early: it runs only when something already asked for the
// table, so a process that never touches HIP still never builds one.
//
// The compiler table stays in hip_capture_init(). Installing it this early
// races compiler-table setup and leaves ModuleInfo() null, which surfaces as
// hipErrorInvalidDeviceFunction on the first kernel launch.
//
// Note this runs before ToolsInit() registers the table with rocprofiler, so a
// profiler attaching to a capturing process wraps HRR's shims as its
// "originals" rather than the reverse, and hip_capture_uninstall() restores a
// pre-registration snapshot, dropping the tool's wrappers. That only matters
// when capture and a profiler tool run together, which HRR does not support.
// ---------------------------------------------------------------------------
void hip_capture_install_early(HipDispatchTable* table) {
#if defined(HIP_HRR_CAPTURE_ENABLED)
  // Flag::init() has not run, so HIP_HRR_CAPTURE_OUTPUT is not populated yet and
  // hip_capture_enabled() cannot be trusted here. getenv is safe this early and
  // reads the same variable the flag is later initialised from.
  const char* out = std::getenv("HIP_HRR_CAPTURE_OUTPUT");
  if (!out || out[0] == '\0') return;

  hip_capture_build_table(table);
  hip_capture_install(table);
#else
  // Capture is compiled out: leave the dispatch table untouched even when
  // HIP_HRR_CAPTURE_OUTPUT is set, so no shim is ever installed (R-06).
  (void)table;
#endif
}

// ---------------------------------------------------------------------------
// Crash-time finalize through CLR exception handling.
//
// The archive is normally finalized from std::atexit(hip_capture_shutdown). If
// the recorded process dies on a fatal signal/exception, atexit does not run.
// Use CLR's cross-platform crash hook instead of an HRR-owned Linux-only
// sigaction chain. The callback intentionally writes an incomplete manifest
// without the clean trailer; periodic writer checkpoints still bound event loss
// if the process dies before the callback can run.
// ---------------------------------------------------------------------------
namespace {
std::once_flag g_hrr_exception_once;

void hrr_clr_crash_callback() {
  hrr_cap::writer::emergency_finalize(/*clean_shutdown=*/false);
}

void hrr_install_clr_exception_handler() {
  std::call_once(g_hrr_exception_once, [] {
    if (!amd::Os::installExceptionHandlers(hrr_clr_crash_callback)) {
      LogPrintfWarning("[HRR capture] CLR crash handler installation failed; "
                       "periodic checkpoints remain enabled");
    }
  });
}

}  // namespace

void hip_capture_init() {
  #if defined(HIP_HRR_CAPTURE_ENABLED)
  if (!hip_capture_enabled()) {
    // hip_capture_install_early() gates on getenv() because Flag::init() has
    // not run that early. If the flag disagrees with what getenv() saw, the
    // shims are already latched with no writer behind them, which would drop
    // every HIP API silently and never restore the table. Undo it instead.
    if (g_installed) hip_capture_uninstall();
    if (hrr_secure_exec() && hrr_capture_requested())
      fprintf(stderr,
              "[HRR capture] HIP_HRR_CAPTURE_OUTPUT ignored: the program was started in "
              "secure-execution mode (set-user-ID, set-group-ID, file capabilities or an "
              "LSM transition).\n");
    return;
  }

  // HIP_HRR_DEBUG_ARGS traces are emitted via LogPrintfInfo (amd::LOG_INFO).
  // ClPrint filters anything above AMD_LOG_LEVEL, so a user who set the trace
  // flag but left AMD_LOG_LEVEL below LOG_INFO would see nothing. Raise the
  // level to LOG_INFO (never lower an already-higher level) so enabling
  // HIP_HRR_DEBUG_ARGS alone is enough to get the traces, as it was when they
  // went through raw fprintf(stderr).
  if (hrr_dbg_args_enabled() && AMD_LOG_LEVEL < amd::LOG_INFO) {
    AMD_LOG_LEVEL = amd::LOG_INFO;
    LogPrintfInfo("[HRR capture] HIP_HRR_DEBUG_ARGS set — raised AMD_LOG_LEVEL "
                  "to %d (LOG_INFO) so argument traces are visible",
                  static_cast<int>(amd::LOG_INFO));
  }

  // Build the runtime dispatch table before touching the capture guards.
  // Leaving it to hip_capture_build_table(), which called GetHipDispatchTable()
  // itself, meant that when this was the first use of that table its static
  // initialiser ran UpdateDispatchTable() -> hip_capture_install_early()
  // underneath us with g_table_built already set: the nested build returned on
  // the guard and installed a still-zeroed g_cap_table over the live table.
  // Forcing the table here means install_early() has already run to completion,
  // leaving the block below a fallback for when it declined to install.
  HipDispatchTable* live = const_cast<HipDispatchTable*>(hip::GetHipDispatchTable());
  if (!g_installed) {
    hip_capture_build_table(live);
    hip_capture_install(live);
  }

    // Open the events writer now — Flag::init() has run so output_dir is valid.
    // A refused open leaves capture off. Without a writer the shims would only
    // cost time, and the D2H ones still synchronize streams, so take them out
    // of the dispatch table again.
    if (!hrr_cap::writer::open(hip_capture_output_dir())) {
      hip_capture_uninstall();
      return;
    }

    hrr_cap::writer::set_capture_metadata_json(
        hrr_cap::metadata::collect_json());

    hrr_install_clr_exception_handler();

    // Install compiler dispatch shims now — hip::init() has completed so
    // the compiler dispatch table is fully populated.
    hip_capture_build_compiler_table();

    // Retroactively record fat binaries that fired before our shims were live.
    // __hipRegisterFatBinary fires at app static-init, before hip_capture_init().
    hip::PlatformState::Instance().StatCO().ForEachFatBinaryBlob(record_fat_binary_blob);

    // And the __device__ globals registered against them, for the same reason
    // and in this order: the fat binaries have to be in the archive before the
    // symbols that live inside them.
    hip::PlatformState::Instance().StatCO().ForEachGlobalVar(record_registered_var);

    std::call_once(g_hrr_atexit_once, [] { std::atexit(hip_capture_shutdown); });

    // Deliberately not gated on AMD_LOG_LEVEL: whoever runs the process must be
    // able to see that it is being recorded. The path must match the pid-<pid>
    // directory writer::open() creates. A fork() child records to a sibling
    // pid-<pid> directory without printing, because hip::init() does not run again.
    fprintf(stderr,
            "[HRR capture] Recording this process's HIP calls, with their host buffers, kernel "
            "arguments and code objects, to %s/pid-%d (child processes record to their own "
            "pid-* directories in %s)\n",
            hip_capture_output_dir(), amd::Os::getProcessId(), hip_capture_output_dir());
  #else
    // HRR capture is disabled
    // below to avoid -Wunused-function / -Wunused-variable
    (void)&record_fat_binary_blob;
    (void)&record_registered_var;
    (void)&hrr_install_clr_exception_handler;
    (void)&g_hrr_atexit_once;
  #endif
}

void hip_capture_shutdown() {
  hip_capture_uninstall();
  hrr_cap::writer::flush(hip_capture_output_dir());
  hrr_cap::writer::close();

  LogPrintfInfo("[HRR capture] Wrote %llu events, %llu blobs to: %s/pid-%d",
                static_cast<unsigned long long>(hrr_cap::writer::event_count()),
                static_cast<unsigned long long>(hrr_cap::writer::blob_count()),
                hip_capture_output_dir(), amd::Os::getProcessId());
}
