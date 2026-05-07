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

  // Team queries
  m.def("rocshmem_team_my_pe", [](intptr_t team) -> int {
    return rocshmem_team_my_pe((rocshmem_team_t)team);
  }, "Get PE number within a team", py::arg("team"));

  m.def("rocshmem_team_n_pes", [](intptr_t team) -> int {
    return rocshmem_team_n_pes((rocshmem_team_t)team);
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

  m.def("rocshmem_barrier_all_on_stream", [](intptr_t stream) {
    rocshmem_barrier_all_on_stream((hipStream_t)stream);
  }, "Stream-ordered barrier across all PEs", py::arg("stream"));

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

  // Atomics
  m.def("rocshmem_int_atomic_fetch_add", [](intptr_t dest, int value, int pe) -> int {
    return rocshmem_int_atomic_fetch_add((int *)dest, value, pe);
  });
  m.def("rocshmem_long_atomic_fetch_add", [](intptr_t dest, long value, int pe) -> long {
    return rocshmem_long_atomic_fetch_add((long *)dest, value, pe);
  });
  m.def("rocshmem_int_atomic_compare_swap", [](intptr_t dest, int cond, int value, int pe) -> int {
    return rocshmem_int_atomic_compare_swap((int *)dest, cond, value, pe);
  });

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
