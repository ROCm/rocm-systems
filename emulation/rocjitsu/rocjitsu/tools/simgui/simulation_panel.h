// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TOOLS_SIMGUI_SIMULATION_PANEL_H_
#define ROCJITSU_TOOLS_SIMGUI_SIMULATION_PANEL_H_

#include "rocjitsu/vm/soc.h"

#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace gui {

/// @brief Simulation control and state inspection panel.
class SimulationPanel {
public:
  /// @brief Render the simulation controls (Run, Step, Pause, Reset).
  /// @returns true if the user clicked Step (single-step request).
  bool render_controls(bool &running, uint64_t tick, int &steps_per_frame);

  /// @brief Render the wavefront state inspector.
  void render_wf_state(SoC *soc);

  /// @brief Render the timeline of wavefront activity using ImPlot.
  void render_timeline(SoC *soc, uint64_t tick);

  /// @brief Record a snapshot of the current wavefront activity.
  void record_tick(SoC *soc, uint64_t tick);

  /// @brief Clear all recorded timeline data.
  void clear_history();

private:
  struct TimelineSample {
    uint64_t tick;
    uint32_t active_wfs;
  };
  std::vector<TimelineSample> history_;
};

} // namespace gui
} // namespace rocjitsu

#endif // ROCJITSU_TOOLS_SIMGUI_SIMULATION_PANEL_H_
