// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file main.cpp
/// @brief Entry point for the rocjitsu simulation GUI.
///
/// Launches a Dear ImGui window with a topology editor, simulation controls,
/// wavefront state inspector, and activity timeline. Optionally loads a JSON
/// config file passed as a command-line argument.

#include "app.h"

#include <cstdio>
#include <string>

int main(int argc, char *argv[]) {
  rocjitsu::gui::App app;

  if (!app.init()) {
    std::fprintf(stderr, "Failed to initialize GUI\n");
    return 1;
  }

  // Schema path: required for building VMs from JSON.
  std::string schema_path;
  if (argc >= 3) {
    schema_path = argv[2];
  } else {
    // Default schema path relative to working directory.
    schema_path = "schemas/simulation_config.fbs";
  }
  app.set_schema_path(schema_path);

  // If a JSON config was provided, load it.
  if (argc >= 2) {
    try {
      app.load_config(argv[1], schema_path);
      std::printf("Loaded config: %s\n", argv[1]);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "Error loading config: %s\n", e.what());
    }
  } else {
    // Start with a default simulation.
    app.build_default_simulation();
  }

  app.run();
  return 0;
}
