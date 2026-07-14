// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// ROCTX bridge for PyTorch's RecordFunction callback. Subscribes to the
// FUNCTION and BACKWARD_FUNCTION scopes and propagates the main-thread
// USER_SCOPE chain into autograd workers via RecordFunction::seqNr()
// and c10::ThreadLocalDebugInfo.

#include "capture_buffer.h"
#include "install_state.h"
#include "record_function_bridge.h"
#include "snapshot_store.h"
#include "stats.h"
#include "user_scope.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace roctx_recordfn::detail
{

pybind11::dict dump_stats()
{
    pybind11::dict stats_dict;
    stats_dict["installed"]           = g_install.installed.load();
    stats_dict["pushes"]              = g_stats.pushes.load();
    stats_dict["pops"]                = g_stats.pops.load();
    stats_dict["user_scope_pushes"]   = g_stats.user_scope_pushes.load();
    stats_dict["user_scope_pops"]     = g_stats.user_scope_pops.load();
    stats_dict["user_scope_inherits"] = g_stats.user_scope_inherits.load();
    stats_dict["snapshots_saved"]     = g_stats.snapshots_saved.load();
    stats_dict["snapshots_consumed"]  = g_stats.snapshots_consumed.load();
    stats_dict["snapshots_dropped"]   = g_stats.snapshots_dropped.load();
    stats_dict["callback_errors"]     = g_stats.callback_errors.load();
    stats_dict["snapshots_pending"]   = g_snapshots.pending();
    return stats_dict;
}

void start_capture()
{
    g_capture.start();
}

std::vector<std::string> stop_capture()
{
    return g_capture.stop();
}

}  // namespace roctx_recordfn::detail

PYBIND11_MODULE(roctx_recordfn, m)
{
    using namespace roctx_recordfn::detail;

    m.doc() = "ROCTX bridge for PyTorch's RecordFunction callback.";

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
