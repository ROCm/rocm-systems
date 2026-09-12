/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Python bindings for rocSHMEM via nanobind.  Framework-independent rocSHMEM
 * glue lives in rocshmem4py_common.hpp; this file is the thin nanobind
 * registration layer.  The compiled module name (_rocshmem4py), function
 * names, argument behavior, and return types are the stable public contract.
 */
#include <nanobind/nanobind.h>
#include <rocshmem/rocshmem.hpp>
#include <rocshmem/qp_introspect.hpp>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "rocshmem4py_common.hpp"

// Single binding framework: import nanobind's names directly.  A namespace
// prefix only earns its keep when a second framework shares the file.
using namespace nanobind;
using namespace rocshmem;
using rocshmem4py::resolve_team_handle;

namespace {

// QpInfo carries a tagged union: only the arm named by `vendor` holds a defined
// value. Reading another arm is undefined in C++, so the Python accessors gate
// on the tag and raise instead of handing back whatever bytes happen to be
// there. This is the whole reason the vendor fields are properties rather than
// plain def_ro members.
void require_vendor(const rocshmem::QpInfo &info, rocshmem::QpInfoVendor want,
                    const char *field) {
  if (info.vendor != want) {
    std::ostringstream oss;
    oss << "QpInfo." << field << " is only valid when vendor is "
        << static_cast<uint32_t>(want) << "; this QP reports vendor "
        << static_cast<uint32_t>(info.vendor);
    // AttributeError, not RuntimeError: these are properties, and Python
    // expects a failed attribute access to raise AttributeError. It is also
    // what makes hasattr() answer correctly -- hasattr swallows only
    // AttributeError, so with RuntimeError it propagates and
    // hasattr(qp, "mlx5_dbrec") raises instead of returning False.
    // nanobind maps attribute_error to Python's AttributeError.
    throw nanobind::attribute_error(oss.str().c_str());
  }
}

}  // namespace

NB_MODULE(_rocshmem4py, m) {
  m.doc() = "Python bindings for ROCSHMEM library";
  // Keep host-facing symbol coverage aligned with
  // python/rocshmem4py/__init__.py:_HOST_API_BINDINGS.

  // Version of the rocSHMEM library this extension was compiled and statically
  // linked against (from <rocshmem/rocshmem_config.h>). Baked in at build time
  // so it is authoritative even for a wheel copied to another machine.
  m.attr("__rocshmem_version__") = ROCSHMEM_VERSION;

  // Initialization
  m.def("rocshmem_init", []() { rocshmem_init(); });
  m.def("rocshmem_finalize", []() { rocshmem_finalize(); });

  m.def("rocshmem_hipmodule_init", [](intptr_t module, intptr_t stream) -> int {
    hipModule_t hip_module = reinterpret_cast<hipModule_t>(module);
    hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
    return rocshmem_hipmodule_init(hip_module, hip_stream);
  }, "Initialize rocSHMEM for HIP module (CUDA graph compatible)",
     arg("module"), arg("stream") = 0);

  // PE queries
  m.def("rocshmem_my_pe", []() -> int { return rocshmem_my_pe(); });
  m.def("rocshmem_n_pes", []() -> int { return rocshmem_n_pes(); });

  m.def("hip_device_synchronize", []() {
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      std::ostringstream err_msg;
      err_msg << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
      throw std::runtime_error(err_msg.str());
    }
  }, "Synchronize the current HIP device.");

  // Team queries
  m.def("rocshmem_team_my_pe", [](intptr_t team) -> int {
    return rocshmem_team_my_pe(resolve_team_handle(team));
  }, "Get PE number within a team", arg("team"));

  m.def("rocshmem_team_n_pes", [](intptr_t team) -> int {
    return rocshmem_team_n_pes(resolve_team_handle(team));
  }, "Get number of PEs in a team", arg("team"));

  // Memory management
  m.def("rocshmem_malloc", [](size_t size) -> intptr_t {
    void *ptr = rocshmem_malloc(size);
    if (ptr == nullptr) {
      throw std::runtime_error("rocshmem_malloc failed");
    }
    return (intptr_t)ptr;
  });
  m.def("rocshmem_free", [](intptr_t ptr) { rocshmem_free((void *)ptr); });

  m.def("rocshmem_calloc", [](size_t count, size_t size) -> intptr_t {
    void *ptr = rocshmem_calloc(count, size);
    if (ptr == nullptr) {
      throw std::runtime_error("rocshmem_calloc failed");
    }
    return (intptr_t)ptr;
  }, "Collective: allocate count*size zero-initialized bytes on the symmetric heap.",
     arg("count"), arg("size"));

  m.def("rocshmem_align", [](size_t alignment, size_t size) -> intptr_t {
    void *ptr = rocshmem_align(alignment, size);
    if (ptr == nullptr) {
      throw std::runtime_error(
          "rocshmem_align failed (invalid alignment or allocation failure)");
    }
    return (intptr_t)ptr;
  }, "Collective: allocate size bytes aligned to alignment on the symmetric "
     "heap. alignment must be a power of two and a multiple of sizeof(void*).",
     arg("alignment"), arg("size"));

  m.def("rocshmem_buffer_register", [](intptr_t addr, size_t length) -> int {
    return rocshmem_buffer_register((void *)addr, length);
  }, "Register a non-symmetric user buffer with the active backend. "
     "Returns ROCSHMEM_SUCCESS (0) on success.",
     arg("addr"), arg("length"));

  m.def("rocshmem_buffer_unregister", [](intptr_t addr) -> int {
    return rocshmem_buffer_unregister((void *)addr);
  }, "Deregister a previously registered non-symmetric buffer. "
     "Returns ROCSHMEM_SUCCESS (0) on success.",
     arg("addr"));

  m.def("rocshmem_buffer_unregister_all", []() {
    rocshmem_buffer_unregister_all();
  }, "Deregister all previously registered non-symmetric buffers.");

  m.def("rocshmem_ptr", [](intptr_t dest, int pe) -> intptr_t {
    void* remote_ptr = rocshmem_ptr((const void *)dest, pe);
    if (remote_ptr == nullptr) {
      return 0;
    }
    return (intptr_t)remote_ptr;
  }, "Get pointer to remote symmetric memory",
     arg("dest"), arg("pe"));

  // Synchronization
  m.def("rocshmem_barrier_all", []() { rocshmem_barrier_all(); });
  m.def("rocshmem_barrier", [](intptr_t team) {
    rocshmem_barrier(resolve_team_handle(team));
  }, "Barrier across all PEs in team.", arg("team"));

  m.def("rocshmem_barrier_all_on_stream", [](intptr_t stream) {
    rocshmem_barrier_all_on_stream((hipStream_t)stream);
  }, "Stream-ordered barrier across all PEs", arg("stream"));

  m.def("rocshmem_barrier_on_stream", [](intptr_t team, intptr_t stream) {
    rocshmem_barrier_on_stream(resolve_team_handle(team), (hipStream_t)stream);
  }, "Stream-ordered barrier across all PEs in team (ROCSHMEM_TEAM_INVALID is a no-op).",
     arg("team"), arg("stream"));

  m.def("rocshmem_fence", []() { rocshmem_fence(); });
  m.def("rocshmem_quiet", []() { rocshmem_quiet(); });

  // Unique ID
  m.def("rocshmem_get_uniqueid", []() -> bytes {
    rocshmem_uniqueid_t uid;
    CHECK_ROCSHMEM(rocshmem_get_uniqueid(&uid));
    // Construct from (buffer, size) to preserve the exact byte length and any
    // embedded NUL bytes in the binary unique-id blob.
    return bytes(reinterpret_cast<const char *>(&uid), sizeof(uid));
  });

  m.def("rocshmem_init_attr", [](int rank, int nranks, bytes uid_bytes) {
    rocshmem_uniqueid_t uid;
    if (uid_bytes.size() != sizeof(uid)) {
      throw std::runtime_error("rocshmem_init_attr: invalid unique ID size");
    }
    rocshmem_init_attr_t init_attr{};
    memcpy(&uid, uid_bytes.c_str(), uid_bytes.size());
    CHECK_ROCSHMEM(rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &init_attr));
    CHECK_ROCSHMEM(rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &init_attr));
  });

  // Data transfer
  m.def("rocshmem_putmem", [](intptr_t dest, intptr_t source, size_t nelems, int pe) {
    rocshmem_putmem((void *)dest, (const void *)source, nelems, pe);
  });
  m.def("rocshmem_getmem", [](intptr_t dest, intptr_t source, size_t nelems, int pe) {
    rocshmem_getmem((void *)dest, (const void *)source, nelems, pe);
  });
  m.def("rocshmem_putmem_nbi", [](intptr_t dest, intptr_t source, size_t nelems, int pe) {
    rocshmem_putmem_nbi((void *)dest, (const void *)source, nelems, pe);
  });
  m.def("rocshmem_getmem_nbi", [](intptr_t dest, intptr_t source, size_t nelems, int pe) {
    rocshmem_getmem_nbi((void *)dest, (const void *)source, nelems, pe);
  });

  // Stream-ordered operations
  m.def("rocshmem_putmem_on_stream", [](intptr_t dest, intptr_t source, size_t nelems, int pe, intptr_t stream) {
    rocshmem_putmem_on_stream((void *)dest, (const void *)source, nelems, pe, (hipStream_t)stream);
  }, "Stream-ordered put operation",
     arg("dest"), arg("source"), arg("nelems"), arg("pe"), arg("stream"));

  m.def("rocshmem_getmem_on_stream", [](intptr_t dest, intptr_t source, size_t nelems, int pe, intptr_t stream) {
    rocshmem_getmem_on_stream((void *)dest, (const void *)source, nelems, pe, (hipStream_t)stream);
  }, "Stream-ordered get operation",
     arg("dest"), arg("source"), arg("nelems"), arg("pe"), arg("stream"));

  m.def("rocshmem_putmem_signal_on_stream",
    [](intptr_t dest, intptr_t source, size_t nelems, intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t stream) {
      rocshmem_putmem_signal_on_stream(
        (void *)dest, (const void *)source, nelems,
        (uint64_t *)sig_addr, signal, sig_op, pe, (hipStream_t)stream);
    }, "Stream-ordered put with remote signaling",
    arg("dest"), arg("source"), arg("nelems"),
    arg("sig_addr"), arg("signal"), arg("sig_op"), arg("pe"), arg("stream"));

  m.def("rocshmem_signal_wait_until_on_stream",
    [](intptr_t sig_addr, int cmp, uint64_t cmp_value, intptr_t stream) {
      rocshmem_signal_wait_until_on_stream((uint64_t *)sig_addr, cmp, cmp_value, (hipStream_t)stream);
    }, "Stream-ordered wait on signal",
    arg("sig_addr"), arg("cmp"), arg("cmp_value"), arg("stream"));

  // -------------------------------------------------------------------------
  // Team APIs
  // -------------------------------------------------------------------------

  class_<rocshmem_team_config_t>(m, "TeamConfig",
      "Configuration record for rocshmem_team_split_strided.")
    .def(init<>())
    .def_rw("num_contexts", &rocshmem_team_config_t::num_contexts);

  m.def("rocshmem_team_split_strided",
    [](intptr_t parent, int start, int stride, int size,
       object config_obj, long mask) -> tuple {
      rocshmem_team_t new_team = ROCSHMEM_TEAM_INVALID;
      rocshmem_team_config_t cfg{};
      const rocshmem_team_config_t* cfg_ptr = nullptr;
      if (!config_obj.is_none()) {
        cfg = cast<rocshmem_team_config_t>(config_obj);
        cfg_ptr = &cfg;
      }
      int status = rocshmem_team_split_strided(
          resolve_team_handle(parent), start, stride, size, cfg_ptr, mask,
          &new_team);
      return make_tuple(status, (intptr_t)new_team);
    },
    "Split parent team into a strided sub-team. Returns (status, new_team_handle).",
    arg("parent"), arg("start"), arg("stride"), arg("size"),
    arg("config") = none(), arg("mask") = 0L);

  m.def("rocshmem_team_destroy", [](intptr_t team) {
    // Both Python sentinels (WORLD=0, INVALID=-1) are no-ops, matching
    // rocshmem_team_destroy()'s documented behavior for WORLD / INVALID
    // / SHARED.  Stale callers that pass 0 thinking it means INVALID
    // still get the expected no-op (silent compatibility).
    if (team == 0 || team == -1) return;
    rocshmem_team_destroy(reinterpret_cast<rocshmem_team_t>(team));
  }, "Destroy a team. Silently ignored for INVALID/WORLD/SHARED.",
     arg("team"));

  m.def("rocshmem_team_translate_pe",
    [](intptr_t src, int pe, intptr_t dst) -> int {
      return rocshmem_team_translate_pe(
          resolve_team_handle(src), pe, resolve_team_handle(dst));
    },
    "Translate a PE index from src_team to dst_team. Returns -1 if unmappable.",
    arg("src_team"), arg("src_pe"), arg("dest_team"));

  // -------------------------------------------------------------------------
  // sync_all (host-side ordering)
  // -------------------------------------------------------------------------
  //
  // TODO: ctx-scoped APIs (rocshmem_ctx_create / _destroy / _fence / _quiet)

  m.def("rocshmem_sync_all", []() { rocshmem_sync_all(); },
    "Lighter-weight partner to barrier_all (local-store visibility).");

  m.def("rocshmem_team_sync", [](intptr_t team) {
    rocshmem_team_sync(resolve_team_handle(team));
  }, "Lighter-weight sync across all PEs in team.", arg("team"));

  m.def("rocshmem_sync_all_on_stream", [](intptr_t stream) {
    rocshmem_sync_all_on_stream((hipStream_t)stream);
  }, "Stream-ordered sync_all.", arg("stream"));

  m.def("rocshmem_team_sync_on_stream", [](intptr_t team, intptr_t stream) {
    rocshmem_team_sync_on_stream(resolve_team_handle(team), (hipStream_t)stream);
  }, "Stream-ordered lighter-weight sync across all PEs in team (ROCSHMEM_TEAM_INVALID is a no-op).",
     arg("team"), arg("stream"));

  // -------------------------------------------------------------------------
  // Full host AMO matrix
  // -------------------------------------------------------------------------
  //
  // Generated via macro expansion from rocshmem_AMO.hpp.  Coverage matrix:
  //
  //   types: int, long, longlong, uint32_t, uint64_t, size_t, ptrdiff_t,
  //          float, double
  //   ops:   fetch, set, compare_swap, swap,
  //          fetch_add, fetch_inc, fetch_and, fetch_or, fetch_xor,
  //          add, inc, and, or, xor
  //
  // Bitwise + inc skipped on float/double (rocSHMEM does not provide them).
  // longlong has only set/inc/add per header; other ops omitted.
  //

#define AMO_FETCH_T(T, Tname)                                                  \
  m.def("rocshmem_" #Tname "_atomic_fetch",                                    \
    [](intptr_t dest, int pe) -> T {                                           \
      return rocshmem_##Tname##_atomic_fetch((T *)dest, pe);                   \
    }, arg("dest"), arg("pe"))

#define AMO_SET_T(T, Tname)                                                    \
  m.def("rocshmem_" #Tname "_atomic_set",                                      \
    [](intptr_t dest, T value, int pe) {                                       \
      rocshmem_##Tname##_atomic_set((T *)dest, value, pe);                     \
    }, arg("dest"), arg("value"), arg("pe"))

#define AMO_CAS_T(T, Tname)                                                    \
  m.def("rocshmem_" #Tname "_atomic_compare_swap",                             \
    [](intptr_t dest, T cond, T value, int pe) -> T {                          \
      return rocshmem_##Tname##_atomic_compare_swap(                           \
          (T *)dest, cond, value, pe);                                         \
    }, arg("dest"), arg("cond"), arg("value"), arg("pe"))

#define AMO_SWAP_T(T, Tname)                                                   \
  m.def("rocshmem_" #Tname "_atomic_swap",                                     \
    [](intptr_t dest, T value, int pe) -> T {                                  \
      return rocshmem_##Tname##_atomic_swap((T *)dest, value, pe);             \
    }, arg("dest"), arg("value"), arg("pe"))

#define AMO_FETCH_ADD_T(T, Tname)                                              \
  m.def("rocshmem_" #Tname "_atomic_fetch_add",                                \
    [](intptr_t dest, T value, int pe) -> T {                                  \
      return rocshmem_##Tname##_atomic_fetch_add((T *)dest, value, pe);        \
    }, arg("dest"), arg("value"), arg("pe"))

#define AMO_FETCH_INC_T(T, Tname)                                              \
  m.def("rocshmem_" #Tname "_atomic_fetch_inc",                                \
    [](intptr_t dest, int pe) -> T {                                           \
      return rocshmem_##Tname##_atomic_fetch_inc((T *)dest, pe);               \
    }, arg("dest"), arg("pe"))

#define AMO_ADD_T(T, Tname)                                                    \
  m.def("rocshmem_" #Tname "_atomic_add",                                      \
    [](intptr_t dest, T value, int pe) {                                       \
      rocshmem_##Tname##_atomic_add((T *)dest, value, pe);                     \
    }, arg("dest"), arg("value"), arg("pe"))

#define AMO_INC_T(T, Tname)                                                    \
  m.def("rocshmem_" #Tname "_atomic_inc",                                      \
    [](intptr_t dest, int pe) {                                                \
      rocshmem_##Tname##_atomic_inc((T *)dest, pe);                            \
    }, arg("dest"), arg("pe"))

#define AMO_FETCH_BITWISE_T(T, Tname, Op)                                      \
  m.def("rocshmem_" #Tname "_atomic_fetch_" #Op,                               \
    [](intptr_t dest, T value, int pe) -> T {                                  \
      return rocshmem_##Tname##_atomic_fetch_##Op((T *)dest, value, pe);       \
    }, arg("dest"), arg("value"), arg("pe"))

#define AMO_BITWISE_T(T, Tname, Op)                                            \
  m.def("rocshmem_" #Tname "_atomic_" #Op,                                     \
    [](intptr_t dest, T value, int pe) {                                       \
      rocshmem_##Tname##_atomic_##Op((T *)dest, value, pe);                    \
    }, arg("dest"), arg("value"), arg("pe"))

  // ------ fetch / set / cas / swap (full type set) ------
  // Note: numeric types only. CAS is missing on float/double per header.
  AMO_FETCH_T(int, int);
  AMO_FETCH_T(long, long);
  AMO_FETCH_T(uint32_t, uint32);
  AMO_FETCH_T(uint64_t, uint64);
  AMO_FETCH_T(size_t, size);
  AMO_FETCH_T(ptrdiff_t, ptrdiff);
  AMO_FETCH_T(float, float);
  AMO_FETCH_T(double, double);

  AMO_SET_T(int, int);
  AMO_SET_T(long, long);
  AMO_SET_T(long long, longlong);
  AMO_SET_T(uint32_t, uint32);
  AMO_SET_T(uint64_t, uint64);
  AMO_SET_T(size_t, size);
  AMO_SET_T(ptrdiff_t, ptrdiff);
  AMO_SET_T(float, float);
  AMO_SET_T(double, double);

  AMO_CAS_T(int, int);
  AMO_CAS_T(long, long);
  AMO_CAS_T(uint32_t, uint32);
  AMO_CAS_T(uint64_t, uint64);
  AMO_CAS_T(size_t, size);
  AMO_CAS_T(ptrdiff_t, ptrdiff);

  AMO_SWAP_T(int, int);
  AMO_SWAP_T(long, long);
  AMO_SWAP_T(uint32_t, uint32);
  AMO_SWAP_T(uint64_t, uint64);
  AMO_SWAP_T(size_t, size);
  AMO_SWAP_T(ptrdiff_t, ptrdiff);
  AMO_SWAP_T(float, float);
  AMO_SWAP_T(double, double);

  // ------ fetch_add / add (numeric types) ------
  AMO_FETCH_ADD_T(int, int);
  AMO_FETCH_ADD_T(long, long);
  AMO_FETCH_ADD_T(uint32_t, uint32);
  AMO_FETCH_ADD_T(uint64_t, uint64);
  AMO_FETCH_ADD_T(size_t, size);
  AMO_FETCH_ADD_T(ptrdiff_t, ptrdiff);

  AMO_ADD_T(int, int);
  AMO_ADD_T(long, long);
  AMO_ADD_T(long long, longlong);
  AMO_ADD_T(uint32_t, uint32);
  AMO_ADD_T(uint64_t, uint64);
  AMO_ADD_T(size_t, size);
  AMO_ADD_T(ptrdiff_t, ptrdiff);

  // ------ fetch_inc / inc (integer types) ------
  AMO_FETCH_INC_T(int, int);
  AMO_FETCH_INC_T(long, long);
  AMO_FETCH_INC_T(uint32_t, uint32);
  AMO_FETCH_INC_T(uint64_t, uint64);
  AMO_FETCH_INC_T(size_t, size);
  AMO_FETCH_INC_T(ptrdiff_t, ptrdiff);

  AMO_INC_T(int, int);
  AMO_INC_T(long, long);
  AMO_INC_T(long long, longlong);
  AMO_INC_T(uint32_t, uint32);
  AMO_INC_T(uint64_t, uint64);
  AMO_INC_T(size_t, size);
  AMO_INC_T(ptrdiff_t, ptrdiff);

  // ------ fetch_and / and / fetch_or / or / fetch_xor / xor (uint32/uint64) ------
  AMO_FETCH_BITWISE_T(uint32_t, uint32, and);
  AMO_FETCH_BITWISE_T(uint64_t, uint64, and);
  AMO_BITWISE_T(uint32_t, uint32, and);
  AMO_BITWISE_T(uint64_t, uint64, and);

  AMO_FETCH_BITWISE_T(uint32_t, uint32, or);
  AMO_FETCH_BITWISE_T(uint64_t, uint64, or);
  AMO_BITWISE_T(uint32_t, uint32, or);
  AMO_BITWISE_T(uint64_t, uint64, or);

  AMO_FETCH_BITWISE_T(uint32_t, uint32, xor);
  AMO_FETCH_BITWISE_T(uint64_t, uint64, xor);
  AMO_BITWISE_T(uint32_t, uint32, xor);
  AMO_BITWISE_T(uint64_t, uint64, xor);

#undef AMO_FETCH_T
#undef AMO_SET_T
#undef AMO_CAS_T
#undef AMO_SWAP_T
#undef AMO_FETCH_ADD_T
#undef AMO_FETCH_INC_T
#undef AMO_ADD_T
#undef AMO_INC_T
#undef AMO_FETCH_BITWISE_T
#undef AMO_BITWISE_T

  // -------------------------------------------------------------------------
  // Team-scoped collectives + small host primitives
  // -------------------------------------------------------------------------

  m.def("rocshmem_alltoallmem_on_stream",
    [](intptr_t team, intptr_t dest, intptr_t source, size_t bytes_per_pe,
       intptr_t stream) {
      rocshmem_alltoallmem_on_stream(
          resolve_team_handle(team), (void *)dest, (const void *)source,
          bytes_per_pe, (hipStream_t)stream);
    },
    "Stream-ordered all-to-all over a team. bytes_per_pe is the number of "
    "bytes transferred to each PE in the team.",
    arg("team"), arg("dest"), arg("source"),
    arg("bytes_per_pe"), arg("stream"));

  m.def("rocshmem_broadcastmem_on_stream",
    [](intptr_t team, intptr_t dest, intptr_t source, size_t nbytes,
       int pe_root, intptr_t stream) {
      rocshmem_broadcastmem_on_stream(
          resolve_team_handle(team), (void *)dest, (const void *)source,
          nbytes, pe_root, (hipStream_t)stream);
    },
    "Stream-ordered broadcast over a team. nbytes is the number of bytes "
    "broadcast. pe_root is in the team's PE space.",
    arg("team"), arg("dest"), arg("source"), arg("nbytes"),
    arg("pe_root"), arg("stream"));

  m.def("rocshmem_query_thread", []() -> int {
    int provided = 0;
    rocshmem_query_thread(&provided);
    return provided;
  }, "Return the threading mode provided by rocshmem_init_thread.");

  m.def("rocshmem_global_exit", [](int status) {
    rocshmem_global_exit(status);
  }, "Emergency abort hook (collective).", arg("status"));

  m.def("rocshmem_dump_stats", []() { rocshmem_dump_stats(); },
    "Dump runtime telemetry to stdout.");

  m.def("rocshmem_reset_stats", []() { rocshmem_reset_stats(); },
    "Reset runtime telemetry counters.");

  m.def("rocshmem_get_device_ctx", []() -> intptr_t {
    return (intptr_t)rocshmem_get_device_ctx();
  }, "Return the default device context as an intptr_t.");

  // Constants
  m.attr("ROCSHMEM_SUCCESS") = int_(0);

  m.attr("ROCSHMEM_SIGNAL_SET") = int_(static_cast<int>(ROCSHMEM_SIGNAL_SET));
  m.attr("ROCSHMEM_SIGNAL_ADD") = int_(static_cast<int>(ROCSHMEM_SIGNAL_ADD));

  m.attr("ROCSHMEM_CMP_EQ") = int_(static_cast<int>(ROCSHMEM_CMP_EQ));
  m.attr("ROCSHMEM_CMP_NE") = int_(static_cast<int>(ROCSHMEM_CMP_NE));
  m.attr("ROCSHMEM_CMP_GT") = int_(static_cast<int>(ROCSHMEM_CMP_GT));
  m.attr("ROCSHMEM_CMP_GE") = int_(static_cast<int>(ROCSHMEM_CMP_GE));
  m.attr("ROCSHMEM_CMP_LT") = int_(static_cast<int>(ROCSHMEM_CMP_LT));
  m.attr("ROCSHMEM_CMP_LE") = int_(static_cast<int>(ROCSHMEM_CMP_LE));

  // -------------------------------------------------------------------------
  // GDA queue-pair introspection
  // -------------------------------------------------------------------------

  enum_<QpInfoVendor>(m, "QpInfoVendor",
      "NIC provider owning a QpInfo. Selects which vendor-specific "
      "attributes are readable.")
    .value("UNKNOWN", QpInfoVendor::UNKNOWN)
    .value("IONIC", QpInfoVendor::IONIC)
    .value("BNXT", QpInfoVendor::BNXT)
    .value("MLX5", QpInfoVendor::MLX5);

  class_<QpInfo>(m, "QpInfo",
      "Device-visible resources of one peer's RC queue pair.\n\n"
      "Addresses are device virtual addresses in the address space of the GPU "
      "owning the QP; depths are in ring slots. The vendor-specific attributes "
      "raise AttributeError unless `vendor` matches, so hasattr() reports False "
      "for the arms that do not apply.")
    .def_ro("sq_buf", &QpInfo::sq_buf, "Send queue ring buffer.")
    .def_ro("sq_prod", &QpInfo::sq_prod,
        "Address of the live SQ producer counter. An external WQE builder must "
        "continue this sequence rather than restart it, or its posts collide "
        "with rocSHMEM's own.")
    .def_ro("cq_buf", &QpInfo::cq_buf, "Completion queue ring buffer.")
    .def_ro("base_heap", &QpInfo::base_heap,
        "Local symmetric heap base for this QP.")
    .def_ro("sq_depth", &QpInfo::sq_depth, "SQ capacity in WQE slots.")
    .def_ro("cq_depth", &QpInfo::cq_depth, "CQ capacity in CQE slots.")
    .def_ro("lkey", &QpInfo::lkey, "Local heap memory key.")
    .def_ro("rkey", &QpInfo::rkey, "Peer heap memory key for this connection.")
    .def_ro("qpn", &QpInfo::qpn, "QP number.")
    .def_ro("vendor", &QpInfo::vendor, "Provider owning this QP.")

    // ionic
    .def_prop_ro("ionic_db", [](const QpInfo &i) {
        require_vendor(i, QpInfoVendor::IONIC, "ionic_db");
        return i.ionic.db;
      }, "ionic: SQ doorbell register.")
    .def_prop_ro("ionic_dbval", [](const QpInfo &i) {
        require_vendor(i, QpInfoVendor::IONIC, "ionic_dbval");
        return i.ionic.dbval;
      }, "ionic: base doorbell value; the producer index is OR'ed in.")
    .def_prop_ro("ionic_sq_mask", [](const QpInfo &i) {
        require_vendor(i, QpInfoVendor::IONIC, "ionic_sq_mask");
        return i.ionic.sq_mask;
      }, "ionic: SQ index wrap mask (depth - 1).")
    .def_prop_ro("ionic_cq_mask", [](const QpInfo &i) {
        require_vendor(i, QpInfoVendor::IONIC, "ionic_cq_mask");
        return i.ionic.cq_mask;
      }, "ionic: CQ index wrap mask (depth - 1).")
    .def_prop_ro("ionic_udma_idx", [](const QpInfo &i) {
        require_vendor(i, QpInfoVendor::IONIC, "ionic_udma_idx");
        return i.ionic.udma_idx;
      }, "ionic: which of the NIC's two UDMA engines this QP is bound to.")

    // mlx5
    .def_prop_ro("mlx5_dbrec", [](const QpInfo &i) {
        require_vendor(i, QpInfoVendor::MLX5, "mlx5_dbrec");
        return i.mlx5.dbrec;
      }, "mlx5: SQ doorbell record.")
    .def_prop_ro("mlx5_bf", [](const QpInfo &i) {
        require_vendor(i, QpInfoVendor::MLX5, "mlx5_bf");
        return i.mlx5.bf;
      }, "mlx5: BlueFlame register; the WQE itself is written here.")

    // bnxt
    .def_prop_ro("bnxt_dbr", [](const QpInfo &i) {
        require_vendor(i, QpInfoVendor::BNXT, "bnxt_dbr");
        return i.bnxt.dbr;
      }, "bnxt: doorbell record.");

  m.def("rocshmem_qp_introspect_provider",
    []() { return rocshmem_qp_introspect_provider(); },
    "The active GDA provider, whether or not it is supported here.\n\n"
    "QpInfoVendor.UNKNOWN means rocSHMEM is not initialized or the backend is\n"
    "not GDA. A known vendor together with rocshmem_qp_introspect_available()\n"
    "== False means the NIC was detected but introspection is not implemented\n"
    "for it yet (mlx5 and bnxt are TODO). Those two situations need different\n"
    "fixes and are indistinguishable from a None result alone.\n\n"
    "Safe to call before rocshmem_init().");

  m.def("rocshmem_qp_introspect_available",
    []() { return rocshmem_qp_introspect_available(); },
    "True if QP introspection is usable here: rocSHMEM is initialized, the\n"
    "active backend is GDA, and its provider is one this API can describe\n"
    "(today, ionic). Check this to branch on capability rather than inferring\n"
    "it from a None result -- None also means 'this peer/ctx_id names no QP',\n"
    "which is a different situation.\n\n"
    "False on an mlx5 or bnxt GDA backend: the NIC is detected but unsupported.\n"
    "Use rocshmem_qp_introspect_provider() to tell that apart from 'no GDA'.\n\n"
    "Safe to call before rocshmem_init() (returns False).");

  m.def("rocshmem_query_qp_info",
    [](int peer, int ctx_id) -> object {
      // Caller errors raise; None is reserved for "not applicable in this
      // environment". Collapsing both into None makes a bad argument
      // indistinguishable from an unsupported backend, which is precisely the
      // ambiguity that lets a broken caller look like an inactive one.
      if (ctx_id <= 0) {
        throw value_error(
            "ctx_id must be > 0; ctx_id 0 is the default context whose "
            "producer index, send-queue lock and cached doorbell position "
            "rocSHMEM keeps to itself, and a queue shared with an external "
            "builder does not run a workload to completion");
      }
      // Check the environment before the peer bound: without an active backend
      // rocshmem_n_pes() is not meaningful, and "wrong backend" is the more
      // useful answer than "bad peer".
      if (!rocshmem_qp_introspect_available()) return none();

      const int n_pes = rocshmem_n_pes();
      if (peer < 0 || peer >= n_pes) {
        std::ostringstream oss;
        oss << "peer " << peer << " is out of range [0, " << n_pes << ")";
        throw value_error(oss.str().c_str());
      }

      QpInfo info{};
      if (!rocshmem_query_qp_info(peer, ctx_id, &info)) return none();
      return cast(info);
    },
    "Describe the RC QP to `peer` within `ctx_id`.\n\n"
    "Returns a QpInfo on success, or None when introspection does not apply in\n"
    "this environment -- the backend is not GDA, the provider is unsupported, or\n"
    "`ctx_id` names no QP (e.g. beyond the configured context count). Use\n"
    "rocshmem_qp_introspect_available() to distinguish 'wrong backend' up front.\n\n"
    "Raises ValueError for caller errors: ctx_id <= 0, or peer outside\n"
    "[0, n_pes). ctx_id 0 is the default context rocSHMEM drives itself; its\n"
    "queue state is not safely shared with an external descriptor builder.",
    arg("peer"), arg("ctx_id"));
}
