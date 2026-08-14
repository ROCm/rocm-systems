// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// pybind11 module exposing the torch_trace_collector API to Python.

#include "capture_buffer.h"
#include "process_state.h"
#include "record_function_installation.h"
#include "snapshot_store.h"
#include "stats.h"
#include "user_scope.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace torch_trace_collector::detail
{

pybind11::dict dump_stats()
{
    const ProcessState& state = process_state();

    pybind11::dict stats_dict;
    stats_dict["installed"]             = is_installed();
    stats_dict["pushes"]                = state.stats.pushes.load();
    stats_dict["pops"]                  = state.stats.pops.load();
    stats_dict["user_scope_pushes"]     = state.stats.user_scope_pushes.load();
    stats_dict["user_scope_pops"]       = state.stats.user_scope_pops.load();
    stats_dict["user_scope_inherits"]   = state.stats.user_scope_inherits.load();
    stats_dict["snapshots_saved"]       = state.stats.snapshots_saved.load();
    stats_dict["snapshots_consumed"]    = state.stats.snapshots_consumed.load();
    stats_dict["snapshots_dropped"]     = state.stats.snapshots_dropped.load();
    stats_dict["snapshots_overwritten"] = state.stats.snapshots_overwritten.load();
    stats_dict["callback_errors"]       = state.stats.callback_errors.load();
    stats_dict["snapshots_pending"]     = state.snapshots.pending();
    return stats_dict;
}

}  // namespace torch_trace_collector::detail

PYBIND11_MODULE(torch_trace_collector, m)
{
    using namespace torch_trace_collector::detail;

    m.doc() = "Emits ROCTX ranges around PyTorch operators through a RecordFunction callback.";

    m.def("install", &install, "Install the global RecordFunction callback. Idempotent.");
    m.def("uninstall", &uninstall, "Remove the registered callback.");
    m.def("is_installed", &is_installed, "Return True if the callback is installed.");
    m.def("push_user_scope",
          &push_user_scope,
          pybind11::arg("marker"),
          pybind11::arg("context"),
          pybind11::arg("backend") = std::string(""),
          "Push a USER_SCOPE frame, emit a ROCTX range, publish chain into TLS DebugInfo.");
    m.def("pop_user_scope", &pop_user_scope, "Pop the most recent push_user_scope() frame on this thread.");
    m.def("dump_stats", &dump_stats, "Internal counters for tests/debugging.");
    m.def("start_capture", &start_capture, "Begin recording wire strings (test hook).");
    m.def("stop_capture", &stop_capture, "Stop and return captured wire strings.");
}
