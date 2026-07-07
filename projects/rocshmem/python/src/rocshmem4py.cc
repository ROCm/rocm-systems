/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Python bindings for rocSHMEM via pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;
using namespace rocshmem;

namespace {
rocshmem_team_t resolve_team_handle(intptr_t team) {
  // Python sentinels are translated to rocSHMEM ABI handles:
  //   0   -> host::ROCSHMEM_TEAM_WORLD (runtime pointer set at init)
  //  -1   -> ROCSHMEM_TEAM_INVALID    (rocSHMEM ABI nullptr)
  // Anything else is a raw rocshmem_team_t (intptr_t-cast pointer)
  // returned by a previous successful split and passed back through.
  if (team == 0)  return ROCSHMEM_TEAM_WORLD;
  if (team == -1) return ROCSHMEM_TEAM_INVALID;
  return reinterpret_cast<rocshmem_team_t>(team);
}
}  // namespace

#define CHECK_ROCSHMEM(expr)                                                   \
  do {                                                                         \
    int status = expr;                                                         \
    if (status != ROCSHMEM_SUCCESS) {                                          \
      std::ostringstream err_msg;                                              \
      err_msg << "ROCSHMEM error in " << __FILE__ << ":" << __LINE__;        \
      throw std::runtime_error(err_msg.str());                                 \
    }                                                                          \
  } while (0)

PYBIND11_MODULE(_rocshmem4py, m) {
  m.doc() = "Python bindings for ROCSHMEM library";
  // Keep host-facing symbol coverage aligned with
  // python/rocshmem4py/__init__.py:_HOST_API_BINDINGS.

  // Initialization
  m.def("rocshmem_init", []() { rocshmem_init(); });
  m.def("rocshmem_finalize", []() { rocshmem_finalize(); });

  m.def("rocshmem_hipmodule_init", [](intptr_t module, intptr_t stream) -> int {
    hipModule_t hip_module = reinterpret_cast<hipModule_t>(module);
    hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
    return rocshmem_hipmodule_init(hip_module, hip_stream);
  }, "Initialize rocSHMEM for HIP module (CUDA graph compatible)",
     py::arg("module"), py::arg("stream") = 0);

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
  }, "Get PE number within a team", py::arg("team"));

  m.def("rocshmem_team_n_pes", [](intptr_t team) -> int {
    return rocshmem_team_n_pes(resolve_team_handle(team));
  }, "Get number of PEs in a team", py::arg("team"));

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
     py::arg("count"), py::arg("size"));

  m.def("rocshmem_align", [](size_t alignment, size_t size) -> intptr_t {
    void *ptr = rocshmem_align(alignment, size);
    if (ptr == nullptr) {
      throw std::runtime_error(
          "rocshmem_align failed (invalid alignment or allocation failure)");
    }
    return (intptr_t)ptr;
  }, "Collective: allocate size bytes aligned to alignment on the symmetric "
     "heap. alignment must be a power of two and a multiple of sizeof(void*).",
     py::arg("alignment"), py::arg("size"));

  m.def("rocshmem_buffer_register", [](intptr_t addr, size_t length) -> int {
    return rocshmem_buffer_register((void *)addr, length);
  }, "Register a non-symmetric user buffer with the active backend. "
     "Returns ROCSHMEM_SUCCESS (0) on success.",
     py::arg("addr"), py::arg("length"));

  m.def("rocshmem_buffer_unregister", [](intptr_t addr) -> int {
    return rocshmem_buffer_unregister((void *)addr);
  }, "Deregister a previously registered non-symmetric buffer. "
     "Returns ROCSHMEM_SUCCESS (0) on success.",
     py::arg("addr"));

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
     py::arg("dest"), py::arg("pe"));

  // Synchronization
  m.def("rocshmem_barrier_all", []() { rocshmem_barrier_all(); });
  m.def("rocshmem_barrier", [](intptr_t team) {
    rocshmem_barrier(resolve_team_handle(team));
  }, "Barrier across all PEs in team.", py::arg("team"));

  m.def("rocshmem_barrier_all_on_stream", [](intptr_t stream) {
    rocshmem_barrier_all_on_stream((hipStream_t)stream);
  }, "Stream-ordered barrier across all PEs", py::arg("stream"));

  m.def("rocshmem_barrier_on_stream", [](intptr_t team, intptr_t stream) {
    rocshmem_barrier_on_stream(resolve_team_handle(team), (hipStream_t)stream);
  }, "Stream-ordered barrier across all PEs in team (ROCSHMEM_TEAM_INVALID is a no-op).",
     py::arg("team"), py::arg("stream"));

  m.def("rocshmem_fence", []() { rocshmem_fence(); });
  m.def("rocshmem_quiet", []() { rocshmem_quiet(); });

  // Unique ID
  m.def("rocshmem_get_uniqueid", []() -> py::bytes {
    rocshmem_uniqueid_t uid;
    CHECK_ROCSHMEM(rocshmem_get_uniqueid(&uid));
    std::string bytes((char *)&uid, sizeof(uid));
    return py::bytes(bytes);
  });

  m.def("rocshmem_init_attr", [](int rank, int nranks, py::bytes bytes) {
    rocshmem_uniqueid_t uid;
    std::string uid_str = bytes;
    if (uid_str.size() != sizeof(uid)) {
      throw std::runtime_error("rocshmem_init_attr: invalid unique ID size");
    }
    rocshmem_init_attr_t init_attr{};
    memcpy(&uid, uid_str.data(), uid_str.size());
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
     py::arg("dest"), py::arg("source"), py::arg("nelems"), py::arg("pe"), py::arg("stream"));

  m.def("rocshmem_getmem_on_stream", [](intptr_t dest, intptr_t source, size_t nelems, int pe, intptr_t stream) {
    rocshmem_getmem_on_stream((void *)dest, (const void *)source, nelems, pe, (hipStream_t)stream);
  }, "Stream-ordered get operation",
     py::arg("dest"), py::arg("source"), py::arg("nelems"), py::arg("pe"), py::arg("stream"));

  m.def("rocshmem_putmem_signal_on_stream",
    [](intptr_t dest, intptr_t source, size_t nelems, intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t stream) {
      rocshmem_putmem_signal_on_stream(
        (void *)dest, (const void *)source, nelems,
        (uint64_t *)sig_addr, signal, sig_op, pe, (hipStream_t)stream);
    }, "Stream-ordered put with remote signaling",
    py::arg("dest"), py::arg("source"), py::arg("nelems"),
    py::arg("sig_addr"), py::arg("signal"), py::arg("sig_op"), py::arg("pe"), py::arg("stream"));

  m.def("rocshmem_signal_wait_until_on_stream",
    [](intptr_t sig_addr, int cmp, uint64_t cmp_value, intptr_t stream) {
      rocshmem_signal_wait_until_on_stream((uint64_t *)sig_addr, cmp, cmp_value, (hipStream_t)stream);
    }, "Stream-ordered wait on signal",
    py::arg("sig_addr"), py::arg("cmp"), py::arg("cmp_value"), py::arg("stream"));

  // -------------------------------------------------------------------------
  // Team APIs
  // -------------------------------------------------------------------------

  py::class_<rocshmem_team_config_t>(m, "TeamConfig",
      "Configuration record for rocshmem_team_split_strided.")
    .def(py::init<>())
    .def_readwrite("num_contexts", &rocshmem_team_config_t::num_contexts);

  m.def("rocshmem_team_split_strided",
    [](intptr_t parent, int start, int stride, int size,
       py::object config_obj, long mask) -> py::tuple {
      rocshmem_team_t new_team = ROCSHMEM_TEAM_INVALID;
      rocshmem_team_config_t cfg{};
      const rocshmem_team_config_t* cfg_ptr = nullptr;
      if (!config_obj.is_none()) {
        cfg = config_obj.cast<rocshmem_team_config_t>();
        cfg_ptr = &cfg;
      }
      int status = rocshmem_team_split_strided(
          resolve_team_handle(parent), start, stride, size, cfg_ptr, mask,
          &new_team);
      return py::make_tuple(status, (intptr_t)new_team);
    },
    "Split parent team into a strided sub-team. Returns (status, new_team_handle).",
    py::arg("parent"), py::arg("start"), py::arg("stride"), py::arg("size"),
    py::arg("config") = py::none(), py::arg("mask") = 0L);

  m.def("rocshmem_team_destroy", [](intptr_t team) {
    // Both Python sentinels (WORLD=0, INVALID=-1) are no-ops, matching
    // rocshmem_team_destroy()'s documented behavior for WORLD / INVALID
    // / SHARED.  Stale callers that pass 0 thinking it means INVALID
    // still get the expected no-op (silent compatibility).
    if (team == 0 || team == -1) return;
    rocshmem_team_destroy(reinterpret_cast<rocshmem_team_t>(team));
  }, "Destroy a team. Silently ignored for INVALID/WORLD/SHARED.",
     py::arg("team"));

  m.def("rocshmem_team_translate_pe",
    [](intptr_t src, int pe, intptr_t dst) -> int {
      return rocshmem_team_translate_pe(
          resolve_team_handle(src), pe, resolve_team_handle(dst));
    },
    "Translate a PE index from src_team to dst_team. Returns -1 if unmappable.",
    py::arg("src_team"), py::arg("src_pe"), py::arg("dest_team"));

  // -------------------------------------------------------------------------
  // sync_all (host-side ordering)
  // -------------------------------------------------------------------------
  //
  // TODO: ctx-scoped APIs (rocshmem_ctx_create / _destroy / _fence / _quiet)
  
  m.def("rocshmem_sync_all", []() { rocshmem_sync_all(); },
    "Lighter-weight partner to barrier_all (local-store visibility).");

  m.def("rocshmem_team_sync", [](intptr_t team) {
    rocshmem_team_sync(resolve_team_handle(team));
  }, "Lighter-weight sync across all PEs in team.", py::arg("team"));

  m.def("rocshmem_sync_all_on_stream", [](intptr_t stream) {
    rocshmem_sync_all_on_stream((hipStream_t)stream);
  }, "Stream-ordered sync_all.", py::arg("stream"));

  m.def("rocshmem_team_sync_on_stream", [](intptr_t team, intptr_t stream) {
    rocshmem_team_sync_on_stream(resolve_team_handle(team), (hipStream_t)stream);
  }, "Stream-ordered lighter-weight sync across all PEs in team (ROCSHMEM_TEAM_INVALID is a no-op).",
     py::arg("team"), py::arg("stream"));

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
    }, py::arg("dest"), py::arg("pe"))

#define AMO_SET_T(T, Tname)                                                    \
  m.def("rocshmem_" #Tname "_atomic_set",                                      \
    [](intptr_t dest, T value, int pe) {                                       \
      rocshmem_##Tname##_atomic_set((T *)dest, value, pe);                     \
    }, py::arg("dest"), py::arg("value"), py::arg("pe"))

#define AMO_CAS_T(T, Tname)                                                    \
  m.def("rocshmem_" #Tname "_atomic_compare_swap",                             \
    [](intptr_t dest, T cond, T value, int pe) -> T {                          \
      return rocshmem_##Tname##_atomic_compare_swap(                           \
          (T *)dest, cond, value, pe);                                         \
    }, py::arg("dest"), py::arg("cond"), py::arg("value"), py::arg("pe"))

#define AMO_SWAP_T(T, Tname)                                                   \
  m.def("rocshmem_" #Tname "_atomic_swap",                                     \
    [](intptr_t dest, T value, int pe) -> T {                                  \
      return rocshmem_##Tname##_atomic_swap((T *)dest, value, pe);             \
    }, py::arg("dest"), py::arg("value"), py::arg("pe"))

#define AMO_FETCH_ADD_T(T, Tname)                                              \
  m.def("rocshmem_" #Tname "_atomic_fetch_add",                                \
    [](intptr_t dest, T value, int pe) -> T {                                  \
      return rocshmem_##Tname##_atomic_fetch_add((T *)dest, value, pe);        \
    }, py::arg("dest"), py::arg("value"), py::arg("pe"))

#define AMO_FETCH_INC_T(T, Tname)                                              \
  m.def("rocshmem_" #Tname "_atomic_fetch_inc",                                \
    [](intptr_t dest, int pe) -> T {                                           \
      return rocshmem_##Tname##_atomic_fetch_inc((T *)dest, pe);               \
    }, py::arg("dest"), py::arg("pe"))

#define AMO_ADD_T(T, Tname)                                                    \
  m.def("rocshmem_" #Tname "_atomic_add",                                      \
    [](intptr_t dest, T value, int pe) {                                       \
      rocshmem_##Tname##_atomic_add((T *)dest, value, pe);                     \
    }, py::arg("dest"), py::arg("value"), py::arg("pe"))

#define AMO_INC_T(T, Tname)                                                    \
  m.def("rocshmem_" #Tname "_atomic_inc",                                      \
    [](intptr_t dest, int pe) {                                                \
      rocshmem_##Tname##_atomic_inc((T *)dest, pe);                            \
    }, py::arg("dest"), py::arg("pe"))

#define AMO_FETCH_BITWISE_T(T, Tname, Op)                                      \
  m.def("rocshmem_" #Tname "_atomic_fetch_" #Op,                               \
    [](intptr_t dest, T value, int pe) -> T {                                  \
      return rocshmem_##Tname##_atomic_fetch_##Op((T *)dest, value, pe);       \
    }, py::arg("dest"), py::arg("value"), py::arg("pe"))

#define AMO_BITWISE_T(T, Tname, Op)                                            \
  m.def("rocshmem_" #Tname "_atomic_" #Op,                                     \
    [](intptr_t dest, T value, int pe) {                                       \
      rocshmem_##Tname##_atomic_##Op((T *)dest, value, pe);                    \
    }, py::arg("dest"), py::arg("value"), py::arg("pe"))

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
    py::arg("team"), py::arg("dest"), py::arg("source"),
    py::arg("bytes_per_pe"), py::arg("stream"));

  m.def("rocshmem_broadcastmem_on_stream",
    [](intptr_t team, intptr_t dest, intptr_t source, size_t nbytes,
       int pe_root, intptr_t stream) {
      rocshmem_broadcastmem_on_stream(
          resolve_team_handle(team), (void *)dest, (const void *)source,
          nbytes, pe_root, (hipStream_t)stream);
    },
    "Stream-ordered broadcast over a team. nbytes is the number of bytes "
    "broadcast. pe_root is in the team's PE space.",
    py::arg("team"), py::arg("dest"), py::arg("source"), py::arg("nbytes"),
    py::arg("pe_root"), py::arg("stream"));

  m.def("rocshmem_query_thread", []() -> int {
    int provided = 0;
    rocshmem_query_thread(&provided);
    return provided;
  }, "Return the threading mode provided by rocshmem_init_thread.");

  m.def("rocshmem_global_exit", [](int status) {
    rocshmem_global_exit(status);
  }, "Emergency abort hook (collective).", py::arg("status"));

  m.def("rocshmem_dump_stats", []() { rocshmem_dump_stats(); },
    "Dump runtime telemetry to stdout.");

  m.def("rocshmem_reset_stats", []() { rocshmem_reset_stats(); },
    "Reset runtime telemetry counters.");

  m.def("rocshmem_get_device_ctx", []() -> intptr_t {
    return (intptr_t)rocshmem_get_device_ctx();
  }, "Return the default device context as an intptr_t.");

  // Constants
  m.attr("ROCSHMEM_SUCCESS") = py::int_(0);

  m.attr("ROCSHMEM_SIGNAL_SET") = py::int_(static_cast<int>(ROCSHMEM_SIGNAL_SET));
  m.attr("ROCSHMEM_SIGNAL_ADD") = py::int_(static_cast<int>(ROCSHMEM_SIGNAL_ADD));

  m.attr("ROCSHMEM_CMP_EQ") = py::int_(static_cast<int>(ROCSHMEM_CMP_EQ));
  m.attr("ROCSHMEM_CMP_NE") = py::int_(static_cast<int>(ROCSHMEM_CMP_NE));
  m.attr("ROCSHMEM_CMP_GT") = py::int_(static_cast<int>(ROCSHMEM_CMP_GT));
  m.attr("ROCSHMEM_CMP_GE") = py::int_(static_cast<int>(ROCSHMEM_CMP_GE));
  m.attr("ROCSHMEM_CMP_LT") = py::int_(static_cast<int>(ROCSHMEM_CMP_LT));
  m.attr("ROCSHMEM_CMP_LE") = py::int_(static_cast<int>(ROCSHMEM_CMP_LE));
}
