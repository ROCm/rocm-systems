// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/emit_summary.hpp"

#include "core/config.hpp"
#include "output/process_metadata.hpp"
#include "output/run_metadata.hpp"
#include "output/summary_renderer.hpp"
#include "output_file_registry.hpp"

#include <unistd.h>

#include <ostream>
#include <utility>

namespace rocprofsys::output
{

void
emit_summary(std::ostream& os, output_file_registry& registry,
             std::chrono::steady_clock::time_point load_baseline)
{
    // Ensure the parent appears at the tree root even when its per-PID
    // metadata file is not loaded by cache_manager.
    process_metadata self{};
    self.pid     = getpid();
    self.ppid    = getppid();
    self.command = config::get_exe_name();
    registry.record_process(std::move(self));

    // output_dir_abs left empty: the renderer derives it from a
    // registered row's parent_path. settings::get_global_output_prefix
    // returns the unresolved %tag%/%timestamp% template.
    const auto meta = run_metadata::capture(load_baseline);
    print_summary(os, registry, meta);
}

}  // namespace rocprofsys::output
