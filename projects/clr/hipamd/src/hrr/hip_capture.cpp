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
#include "hip_capture_writer.h"

// hrr_api_args.h — for hrr_args_* struct types and hrr_api_id_t enum
#include "hrr_api_args.h"

// HIP runtime internals
#include "../hip_global.hpp"       // hip::asKernel()
#include "../hip_internal.hpp"
#include "hip/amd_detail/hip_api_trace.hpp"
#include "utils/flags.hpp"         // HIP_HRR_CAPTURE_OUTPUT flag

// ROCclr kernel introspection
#include "device/devkernel.hpp"    // amd::Kernel, KernelParameterDescriptor
#include "platform/kernel.hpp"     // amd::KernelSignature
#include "opencl/amdocl/cl_kernel.h"  // T_POINTER enum

// Fat binary format structs (ClangOffloadBundleUncompressedHeader, etc.)
#include "../hip_code_object.hpp"
#include "../hip_platform.hpp"   // PlatformState::Instance()

#include <algorithm>
#include <atomic>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
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

bool hip_capture_enabled() {
  return !flagIsDefault(HIP_HRR_CAPTURE_OUTPUT) &&
         HIP_HRR_CAPTURE_OUTPUT[0] != '\0';
}

const char* hip_capture_output_dir() {
  return HIP_HRR_CAPTURE_OUTPUT;
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
// the <<<>>> path. The type metadata cannot say "bytes 0..7 are an address",
// so the only robust way to find such pointers is to ask the runtime whether
// each 8-byte aligned word is a live device allocation. Probing perturbs the
// calling thread's HIP last-error state, so we save and restore it.
static void find_embedded_ptr_offsets(const uint8_t* data, size_t size,
                                      std::vector<uint16_t>& offsets) {
  if (!data || !g_real_table.hipPointerGetAttributes_fn) return;
  const hipError_t saved_cmd = hip::tls.last_command_error_;
  const hipError_t saved_err = hip::tls.last_error_;
  for (size_t j = 0; j + sizeof(void*) <= size; j += sizeof(void*)) {
    uint64_t cand;
    std::memcpy(&cand, data + j, sizeof(cand));
    if (cand < 0x10000ULL) continue;  // small scalars are never device VAs
    hipPointerAttribute_t attr{};
    if (g_real_table.hipPointerGetAttributes_fn(
            &attr, reinterpret_cast<const void*>(cand)) == hipSuccess &&
        (attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeUnified)) {
      offsets.push_back(static_cast<uint16_t>(j));
    }
  }
  hip::tls.last_command_error_ = saved_cmd;
  hip::tls.last_error_         = saved_err;
}

static void serialize_kernel_launch(
    const char*                 kernel_name,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t shared_mem,
    hipStream_t stream,
    const amd::KernelSignature& sig,
    void**                      kernel_params,
    const void*                 kbuf,
    size_t                      ksz)
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

  uint16_t name_len = static_cast<uint16_t>(std::strlen(kernel_name));
  push_u16(name_len);
  push_bytes(kernel_name, name_len);
  push_u64(0); push_u64(0);  // co_hash (unknown at capture time)
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

  // HIP_HRR_DEBUG_ARGS=<non-empty> dumps every captured arg (kind, size, full
  // bytes, detected embedded-pointer offsets) — used to confirm pointer layout.
  static const bool dbg = []() {
    const char* e = std::getenv("HIP_HRR_DEBUG_ARGS");
    return e != nullptr && e[0] != '\0';
  }();

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
        std::fprintf(stderr, "[HRR args] %s arg[%u] kind=1(ptr) size=%u value=0x%llx%s\n",
                     kernel_name, idx, sz, (unsigned long long)v,
                     bytes ? "" : " [TRUNCATED:no-bytes]");
      }
      return;
    }
    std::vector<uint16_t> offs;
    find_embedded_ptr_offsets(bytes, sz, offs);
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
      std::fprintf(stderr, "[HRR args] %s arg[%u] kind=%u size=%u bytes=%s ptr_off=[%s]%s\n",
                   kernel_name, idx, kind, sz, hex.c_str(), off_str.c_str(),
                   bytes ? "" : " [TRUNCATED:no-bytes]");
    }
  };

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
      std::fprintf(stderr, "[HRR args] %s kbuf path: ksz=%zu need=%u n_all=%u%s\n",
                   kernel_name, ksz, need, n_all,
                   (ksz < need) ? " [KBUF-TOO-SMALL]" : "");
    }
    for (uint32_t i = 0; i < n_all; i++) {
      const auto& desc = sig.at(i);
      if (desc.info_.hidden_) continue;
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
      uint16_t sz = static_cast<uint16_t>(desc.size_);
      emit_arg(i, desc.type_ == T_POINTER,
               static_cast<const uint8_t*>(kernel_params[param_idx]), sz);
      param_idx++;
    }
  }

  if (payload.size() > UINT16_MAX) {
    LogPrintfWarning("[HRR capture] Kernel launch payload too large (%zu bytes) — dropping event for '%s'",
                     payload.size(), kernel_name);
    return;
  }
  hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELAUNCHKERNEL,
                                   reinterpret_cast<hrr_event_header*>(payload.data()),
                                   static_cast<uint16_t>(payload.size()));
}

static void record_launch(
    hipFunction_t f,
    unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared_mem,
    hipStream_t stream,
    void** kernel_params, void** extra)
{
  if (!hip_capture_enabled()) return;
  amd::Kernel* kernel = hip::asKernel(f);
  if (!kernel) return;

  const amd::KernelSignature& sig = kernel->signature();
  const void* kbuf = nullptr;
  size_t      ksz  = 0;

  if (!kernel_params && extra)
    parse_kernel_extra(extra, kbuf, ksz);

  serialize_kernel_launch(
      kernel->name().c_str(),
      gx, gy, gz, bx, by, bz,
      static_cast<uint32_t>(shared_mem),
      stream, sig, kernel_params, kbuf, ksz);
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
    if (kind == hipMemcpyHostToDevice && src && sizeBytes > 0)
      h = hrr_cap::writer::write_blob(src, sizeBytes);
    else if (kind == hipMemcpyDeviceToHost && dst && sizeBytes > 0)
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
  if (r == hipSuccess && hip_capture_enabled()) {
    hrr_cap::Hash128 h{0, 0};
    if (kind == hipMemcpyHostToDevice && src && sizeBytes > 0) {
      h = hrr_cap::writer::write_blob(src, sizeBytes);
    } else if (kind == hipMemcpyDeviceToHost && dst && sizeBytes > 0) {
      // Sync the stream so host dst is valid before we snapshot it.
      hipError_t sync_r = g_real_table.hipStreamSynchronize_fn(stream);
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
  if (r == hipSuccess && hip_capture_enabled()) {
    hrr_cap::Hash128 h{0, 0};
    if (dst && sizeBytes > 0) {
      // Sync the stream so host dst is valid before we snapshot it.
      hipError_t sync_r = g_real_table.hipStreamSynchronize_fn(stream);
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

// Helper: get the actual device binary from a successfully loaded hipModule_t.
// The caller passes an in-memory image (which may be a fat binary bundle, not
// a raw ELF).  The runtime unbundles/extracts the device ELF internally and
// stores it in the amd::Program.  We read it back from there so we always
// capture the processed ELF, not the raw (possibly bundled) input image.
static hrr_cap::Hash128 write_module_code_object(hipModule_t module) {
  amd::Program* prog = as_amd(reinterpret_cast<cl_program>(module));
  if (!prog) return {0, 0};
  const int dev = hip::ihipGetDevice();
  const amd::Device* device = hip::g_devices[dev]->devices()[0];
  const auto& bin = prog->binary(*device);
  const uint8_t* data = std::get<0>(bin);
  const size_t   size = std::get<1>(bin).first;
  if (!data || !size) return {0, 0};
  return hrr_cap::writer::write_code_object(data, size);
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
    hrr_cap::Hash128 h{0, 0};
    // Read the file from disk and snapshot it as a code object so the replay
    // can load it by hash. Without this, fname is a capture-time address and
    // is useless at replay time.
    if (fname) {
      if (FILE* fh = fopen(fname, "rb")) {
        fseek(fh, 0, SEEK_END);
        long sz = ftell(fh);
        rewind(fh);
        if (sz > 0) {
          std::vector<uint8_t> buf(static_cast<size_t>(sz));
          if (fread(buf.data(), 1, buf.size(), fh) == buf.size())
            h = hrr_cap::writer::write_code_object(buf.data(), buf.size());
        }
        fclose(fh);
      }
    }
    hrr_args_hipModuleLoad a{};
    a.ret        = static_cast<int32_t>(r);
    a.module     = reinterpret_cast<uint64_t>(*module);
    a.fname      = 0;  // not a valid cross-process address; hash identifies the file
    a.co_hash_lo = h.lo;
    a.co_hash_hi = h.hi;
    a.module_id  = 0;
    hrr_cap::writer::write_event_raw(HRR_API_HIPMODULELOAD, &a.hdr, sizeof(a));
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
    record_launch(f,
                  globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
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
  if (r == hipSuccess && hip_capture_enabled()) {
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
  if (r == hipSuccess && hip_capture_enabled()) {
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
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            static_cast<uint32_t>(shared), stream,
            sig, nullptr, kbuf, ksz);
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

// Compute the total byte size of a clang offload bundle blob.
// Supports both uncompressed ("__CLANG_OFFLOAD_BUNDLE__") and compressed ("CCOB") formats.
// Returns 0 if the format is unrecognised.
static size_t compute_bundle_size(const void* blob) {
  if (!blob) return 0;
  const char* p = static_cast<const char*>(blob);

  // Compressed format: magic "CCOB", header contains totalSize at byte 8.
  if (std::memcmp(p, hip::symbols::kOffloadBundleCompressedMagicStr,
                  hip::symbols::kOffloadBundleCompressedMagicStrSize - 1) == 0) {
    const auto* hdr = static_cast<const hip::symbols::ClangOffloadBundleCompressedHeader*>(blob);
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

  // Walk entries to find the last offset + size (that is the blob end).
  // Guard against corrupt bundles: bundleEntryIdSize is read from untrusted
  // memory (fires at static-init before any error handler is installed).
  // A sane bundle ID is never > 4 KB; anything larger indicates corruption.
  static constexpr uint64_t kMaxBundleIdSize = 4096;
  static constexpr uint64_t kMaxEntries      = 4096;
  size_t end = 0;
  const uint8_t* cur = reinterpret_cast<const uint8_t*>(&hdr->desc[0]);
  uint64_t safe_n = (n < kMaxEntries) ? n : kMaxEntries;
  for (uint64_t i = 0; i < safe_n; i++) {
    const auto* entry = reinterpret_cast<const hip::symbols::ClangOffloadBundleInfo*>(cur);
    if (entry->bundleEntryIdSize > kMaxBundleIdSize) {
      LogPrintfWarning("[HRR capture] compute_bundle_size: bundleEntryIdSize %llu too large"
                       " — stopping walk at entry %llu",
                       (unsigned long long)entry->bundleEntryIdSize,
                       (unsigned long long)i);
      break;
    }
    size_t entry_end = static_cast<size_t>(entry->offset) + static_cast<size_t>(entry->size);
    if (entry_end > end) end = entry_end;
    // Advance past this entry: three uint64_t fields + bundleEntryIdSize bytes
    cur += 3 * sizeof(uint64_t) + entry->bundleEntryIdSize;
  }
  return end;
}

void** capture___hipRegisterFatBinary(const void* data) {
  void** r = g_real_compiler_table.__hipRegisterFatBinary_fn(data);
  // Shim is only installed when capture is active — no hip_capture_enabled() check needed.

  // data is a __CudaFatBinaryWrapper* { magic, version, binary, dummy }.
  // Capture the fat binary blob (binary field) so replay can load it via hipModuleLoadData.
  struct __HRRFatBinaryWrapper { uint32_t magic; uint32_t version; const void* binary; const void* dummy; };
  const auto* wrapper = static_cast<const __HRRFatBinaryWrapper*>(data);
  const void* blob = (wrapper && (wrapper->magic == 0x48495046u /*HIPF*/ ||
                                   wrapper->magic == 0x4B504948u /*HIPK*/))
                     ? wrapper->binary : nullptr;
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
// hipMemPoolSetAttribute — value is void* to a scalar; copy 8 bytes inline.
// ---------------------------------------------------------------------------

hipError_t capture_hipMemPoolSetAttribute(hipMemPool_t mem_pool,
                                          hipMemPoolAttr attr,
                                          void* value) {
  hipError_t r = g_real_table.hipMemPoolSetAttribute_fn(mem_pool, attr, value);
  hrr_args_hipMemPoolSetAttribute a{};
  a.ret      = static_cast<int32_t>(r);
  a.mem_pool = reinterpret_cast<uint64_t>(mem_pool);
  a.attr     = static_cast<int32_t>(attr);
  a.value    = reinterpret_cast<uint64_t>(value);
  if (value) std::memcpy(&a.value_u64, value, sizeof(a.value_u64));
  hrr_cap::writer::write_event_raw(HRR_API_HIPMEMPOOLSETATTRIBUTE, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// hipMemPoolCreate — pool_props is a struct pointer; copy it inline.
// ---------------------------------------------------------------------------

hipError_t capture_hipMemPoolCreate(hipMemPool_t* mem_pool,
                                    const hipMemPoolProps* pool_props) {
  hipError_t r = g_real_table.hipMemPoolCreate_fn(mem_pool, pool_props);
  hrr_args_hipMemPoolCreate a{};
  a.ret      = static_cast<int32_t>(r);
  a.mem_pool = reinterpret_cast<uint64_t>(r == hipSuccess ? *mem_pool : nullptr);
  if (pool_props)
    std::memcpy(a.pool_props_bytes, pool_props, sizeof(a.pool_props_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPMEMPOOLCREATE, &a.hdr, sizeof(a));
  return r;
}

// ---------------------------------------------------------------------------
// hipMemcpy3D / hipMemcpy3DAsync — inline parms + H2D blob + D2H expected blob
// ---------------------------------------------------------------------------

// Helper: compute byte count from 3D extent
static size_t memcpy3d_byte_count(const struct hipMemcpy3DParms* p) {
  return p->extent.width * p->extent.height * p->extent.depth;
}

// Helper shared by all four 3D variants.
// Writes H2D blob (src host data) and D2H expected blob (dst host data after copy),
// then emits the event record.
template <typename T>
static void capture_memcpy3d_impl(
    T& a, hrr_api_id_t api_id,
    const struct hipMemcpy3DParms* p, hipStream_t stream, bool is_async) {
  if (!hip_capture_enabled()) return;
  if (!p) {
    hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
    return;
  }
  std::memcpy(a.parms_bytes, p, sizeof(a.parms_bytes));
  size_t byte_count = memcpy3d_byte_count(p);

  if (p->kind == hipMemcpyHostToDevice && p->srcPtr.ptr && byte_count > 0) {
    // H2D: host source is valid at call time — no stream sync needed.
    auto h = hrr_cap::writer::write_blob(p->srcPtr.ptr, byte_count);
    a.blob_hash_lo = h.lo;
    a.blob_hash_hi = h.hi;
  } else if (p->kind == hipMemcpyDeviceToHost && p->dstPtr.ptr && byte_count > 0) {
    // D2H: real call already completed (sync API) or stream sync done below;
    // host buffer now holds GPU result — capture it as the expected output.
    if (is_async && stream) {
      hipError_t sync_r = g_real_table.hipStreamSynchronize_fn(stream);
      if (sync_r != hipSuccess) {
        LogPrintfWarning("[HRR capture] hipStreamSynchronize failed (%d) — D2H 3D blob skipped",
                         sync_r);
        hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
        return;
      }
    }
    auto h = hrr_cap::writer::write_blob(p->dstPtr.ptr, byte_count);
    a.d2h_hash_lo = h.lo;
    a.d2h_hash_hi = h.hi;
  }
  hrr_cap::writer::write_event_raw(api_id, &a.hdr, sizeof(a));
}

hipError_t capture_hipMemcpy3D(const struct hipMemcpy3DParms* p) {
  hipError_t r = g_real_table.hipMemcpy3D_fn(p);
  hrr_args_hipMemcpy3D a{};
  a.ret = static_cast<int32_t>(r);
  capture_memcpy3d_impl(a, HRR_API_HIPMEMCPY3D, p, nullptr, false);
  return r;
}

hipError_t capture_hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpy3DAsync_fn(p, stream);
  hrr_args_hipMemcpy3DAsync a{};
  a.ret    = static_cast<int32_t>(r);
  a.stream = reinterpret_cast<uint64_t>(stream);
  capture_memcpy3d_impl(a, HRR_API_HIPMEMCPY3DASYNC, p, stream, true);
  return r;
}

hipError_t capture_hipMemcpy3D_spt(const struct hipMemcpy3DParms* p) {
  hipError_t r = g_real_table.hipMemcpy3D_spt_fn(p);
  hrr_args_hipMemcpy3D_spt a{};
  a.ret = static_cast<int32_t>(r);
  capture_memcpy3d_impl(a, HRR_API_HIPMEMCPY3D_SPT, p, nullptr, false);
  return r;
}

hipError_t capture_hipMemcpy3DAsync_spt(const struct hipMemcpy3DParms* p, hipStream_t stream) {
  hipError_t r = g_real_table.hipMemcpy3DAsync_spt_fn(p, stream);
  hrr_args_hipMemcpy3DAsync_spt a{};
  a.ret    = static_cast<int32_t>(r);
  a.stream = reinterpret_cast<uint64_t>(stream);
  capture_memcpy3d_impl(a, HRR_API_HIPMEMCPY3DASYNC_SPT, p, stream, true);
  return r;
}

// ---------------------------------------------------------------------------
// hipArrayCreate / hipArray3DCreate — inline descriptor; record handle
// ---------------------------------------------------------------------------

hipError_t capture_hipArrayCreate(hipArray_t* pHandle,
                                  const HIP_ARRAY_DESCRIPTOR* pAllocateArray) {
  hipError_t r = g_real_table.hipArrayCreate_fn(pHandle, pAllocateArray);
  hrr_args_hipArrayCreate a{};
  a.ret     = static_cast<int32_t>(r);
  a.pHandle = reinterpret_cast<uint64_t>(r == hipSuccess ? *pHandle : nullptr);
  if (pAllocateArray)
    std::memcpy(a.array_desc_bytes, pAllocateArray, sizeof(a.array_desc_bytes));
  hrr_cap::writer::write_event_raw(HRR_API_HIPARRAYCREATE, &a.hdr, sizeof(a));
  return r;
}

hipError_t capture_hipArray3DCreate(hipArray_t* array,
                                    const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray) {
  hipError_t r = g_real_table.hipArray3DCreate_fn(array, pAllocateArray);
  hrr_args_hipArray3DCreate a{};
  a.ret   = static_cast<int32_t>(r);
  a.array = reinterpret_cast<uint64_t>(r == hipSuccess ? *array : nullptr);
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
// Install / uninstall (build_table functions live in hip_capture_generated.cpp)
// ---------------------------------------------------------------------------

void hip_capture_install() {
  if (g_installed.exchange(true)) return;
  std::memcpy(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()),
              &g_cap_table, sizeof(HipDispatchTable));
}

void hip_capture_uninstall() {
  if (!g_installed.exchange(false)) return;
  std::memcpy(const_cast<HipDispatchTable*>(hip::GetHipDispatchTable()),
              &g_real_table, sizeof(HipDispatchTable));
}

// ---------------------------------------------------------------------------
// Init / Shutdown
// ---------------------------------------------------------------------------

// Record a single fat binary blob as a HRR_API_HIPREGISTERFATBINARY event.
// blob_ptr is the fbwrapper->binary pointer (the actual clang offload bundle).
static void record_fat_binary_blob(const void* blob_ptr) {
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

// Runtime dispatch-table capture is installed only from hip_capture_init()
// (after hip::init() has built the live HipDispatchTable).  We intentionally
// do NOT hook at libamdhip64 static-init time when HIP_HRR_CAPTURE_OUTPUT is set:
// early install ran before amd::Runtime::init() / Flag::init(), forced every
// HIP API through capture shims from the moment the DSO loaded, and pulled host
// stacks (e.g. Python import torch → multiprocessing spawn) into ordering where
// the GPU runtime appeared "already initialized" before child processes start.
// Events before writer::open() were dropped anyway (write_event_raw no-ops when
// g_events_fd < 0), so deferring install loses no recorded events for that
// window while restoring a normal pre-init load path.

// ---------------------------------------------------------------------------
// Crash-time finalize — chained fatal-signal handlers.
//
// The archive is normally finalized from std::atexit(hip_capture_shutdown). If
// the recorded process dies on a fatal signal (e.g. a GPU memory fault aborts
// the host, or a host SIGSEGV), atexit never runs. These handlers best-effort
// flush events.bin and write a manifest, then CHAIN to the previously installed
// handler so ROCr still emits its GPU coredump (gpucore.<pid>) and the kernel
// still produces the host core.
//
// Crash signals (SIGSEGV/SIGABRT/SIGBUS/SIGILL/SIGFPE) do NOT write the clean
// trailer: its absence is exactly how the reader detects a crash-truncated
// archive. Orderly-termination signals (SIGTERM/SIGINT) DO write the clean
// trailer and a complete manifest, so killing a capture run with `kill -TERM`
// is not misreported as a crash (see emergency_finalize's clean_shutdown path).
//
// POSIX only; on Windows the writer's periodic checkpoint still bounds loss.
// ---------------------------------------------------------------------------
#ifndef _WIN32
namespace {

// Crash signals: the process is in an undefined state — the archive trailer is
// deliberately NOT written so its absence flags a crash-truncated archive.
// Clean-termination signals (SIGTERM/SIGINT): the process is being asked to exit
// in an orderly fashion — write the clean trailer so a `kill -TERM` of a capture
// run is not misreported as a crash (see hrr_signal_is_clean).
constexpr int    kHrrFatalSignals[] = {SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE, SIGTERM, SIGINT};
constexpr size_t kHrrNumSignals     = sizeof(kHrrFatalSignals) / sizeof(kHrrFatalSignals[0]);

struct sigaction g_hrr_prev_sa[kHrrNumSignals];
std::atomic_flag g_hrr_in_handler = ATOMIC_FLAG_INIT;
// Dedicated stack so a SIGSEGV from stack overflow can still run the handler.
char             g_hrr_altstack_mem[65536];

// SIGTERM/SIGINT are orderly-termination requests, not crashes.
inline bool hrr_signal_is_clean(int signo) {
  return signo == SIGTERM || signo == SIGINT;
}

void hrr_fatal_signal_handler(int signo, siginfo_t* info, void* uctx) {
  // Only the first thread into the handler finalizes the archive.
  if (!g_hrr_in_handler.test_and_set(std::memory_order_acq_rel)) {
    hrr_cap::writer::emergency_finalize(/*clean_shutdown=*/hrr_signal_is_clean(signo));
  }

  // Chain to the handler that was installed before us (e.g. ROCr's GPU-coredump
  // handler), so that behavior is preserved.
  int idx = -1;
  for (size_t i = 0; i < kHrrNumSignals; i++)
    if (kHrrFatalSignals[i] == signo) { idx = static_cast<int>(i); break; }

  if (idx >= 0) {
    const struct sigaction& prev = g_hrr_prev_sa[idx];
    if ((prev.sa_flags & SA_SIGINFO) && prev.sa_sigaction &&
        prev.sa_sigaction != hrr_fatal_signal_handler) {
      prev.sa_sigaction(signo, info, uctx);
      return;
    }
    if (prev.sa_handler == SIG_IGN) return;
    if (prev.sa_handler && prev.sa_handler != SIG_DFL &&
        prev.sa_handler != reinterpret_cast<void (*)(int)>(hrr_fatal_signal_handler)) {
      prev.sa_handler(signo);
      return;
    }
  }

  // No (usable) previous handler: restore the default disposition and re-raise
  // so the kernel terminates the process with a core dump for `signo`.
  signal(signo, SIG_DFL);
  raise(signo);
}

void hrr_install_signal_handlers() {
  stack_t ss{};
  ss.ss_sp    = g_hrr_altstack_mem;
  ss.ss_size  = sizeof(g_hrr_altstack_mem);
  ss.ss_flags = 0;
  sigaltstack(&ss, nullptr);

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = hrr_fatal_signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO | SA_NODEFER | SA_ONSTACK | SA_RESTART;
  for (size_t i = 0; i < kHrrNumSignals; i++)
    sigaction(kHrrFatalSignals[i], &sa, &g_hrr_prev_sa[i]);
}

}  // namespace
#endif  // !_WIN32

void hip_capture_init() {
  if (!hip_capture_enabled()) return;

  // Snapshot the fully-initialized dispatch table and install runtime shims here
  // only (see comment above — no static-init capture hook).
  if (!g_installed) {
    hip_capture_build_table();
    hip_capture_install();
  }

  // Open the events writer now — Flag::init() has run so output_dir is valid.
  if (!hrr_cap::writer::open(hip_capture_output_dir())) return;

#ifndef _WIN32
  // Install AFTER the runtime is up so we chain on top of ROCr's signal handlers
  // (ours runs first, flushes the archive, then forwards to ROCr for gpucore).
  hrr_install_signal_handlers();
#endif

  // Install compiler dispatch shims now — hip::init() has completed so
  // the compiler dispatch table is fully populated.
  hip_capture_build_compiler_table();

  // Retroactively record fat binaries that fired before our shims were live.
  // __hipRegisterFatBinary fires at app static-init, before hip_capture_init().
  hip::PlatformState::Instance().StatCO().ForEachFatBinaryBlob(record_fat_binary_blob);

  std::call_once(g_hrr_atexit_once, [] { std::atexit(hip_capture_shutdown); });
}

// Internal explicit-flush hook for multi-process hosts (e.g. a vLLM server
// shutdown callback). This is deliberately NOT a public ABI symbol: it has
// hidden visibility on Linux and is not exported on Windows, has no declaration
// in any public HIP header, and carries no stability contract. Normal shutdown
// is covered by the atexit(hip_capture_shutdown) hook; the periodic checkpoint
// bounds crash loss. Appends the clean trailer and manifest without uninstalling
// the capture shims so capture can continue afterwards.
#ifndef _WIN32
extern "C" __attribute__((visibility("hidden"))) void hipHrrCaptureFlush();
#endif
extern "C" void hipHrrCaptureFlush() {
  if (!hip_capture_enabled() || !hrr_cap::writer::is_open()) return;
  hrr_cap::writer::flush(hip_capture_output_dir());
}

void hip_capture_shutdown() {
  hip_capture_uninstall();
  hrr_cap::writer::flush(hip_capture_output_dir());
  hrr_cap::writer::close();

  LogPrintfInfo("[HRR capture] Wrote %llu events, %llu blobs to: %s",
               static_cast<unsigned long long>(hrr_cap::writer::event_count()),
               static_cast<unsigned long long>(hrr_cap::writer::blob_count()),
               hip_capture_output_dir());
}
