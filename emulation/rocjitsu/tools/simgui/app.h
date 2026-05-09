// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TOOLS_SIMGUI_APP_H_
#define ROCJITSU_TOOLS_SIMGUI_APP_H_

#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include <cstdint>
#include <memory>
#include <string>

struct GLFWwindow;

namespace rocjitsu {
namespace gui {

/// @brief Main application state for the simulation GUI.
///
/// Manages the GLFW window, Dear ImGui context, and the simulation lifecycle.
class App {
public:
  App();
  ~App();

  /// @brief Initialize the window, rendering context, and ImGui.
  /// @returns true on success.
  bool init(int width = 1600, int height = 900, const char *title = "rocjitsu simgui");

  /// @brief Run the main event/render loop until the window is closed.
  void run();

  /// @brief Load a simulation config from a JSON file.
  void load_config(const std::string &json_path, const std::string &schema_path);

  /// @brief Build simulation from current editor state.
  void build_default_simulation();

  /// @brief Set the schema path for JSON config parsing.
  void set_schema_path(const std::string &path) { schema_path_ = path; }

private:
  void render_frame();
  void render_menu_bar();
  void reset_simulation();

  GLFWwindow *window_ = nullptr;

  // Simulation state (owned directly).
  std::unique_ptr<simdojo::SimulationEngine> engine_;
  simdojo::SimulationEngine::Config engine_config_{}; ///< Kept for checkpoint save.
  SoC *soc_ = nullptr;                                ///< Cached pointer into topology root.

  // Schema path for creating VMs from JSON.
  std::string schema_path_;

  // Simulation control.
  bool sim_running_ = false;
  uint64_t sim_tick_ = 0;
  int steps_per_frame_ = 1;
};

} // namespace gui
} // namespace rocjitsu

#endif // ROCJITSU_TOOLS_SIMGUI_APP_H_
