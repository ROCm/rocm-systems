// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_TOOLS_SIMGUI_TOPOLOGY_EDITOR_H_
#define ROCJITSU_TOOLS_SIMGUI_TOPOLOGY_EDITOR_H_

#include "rocjitsu/vm/soc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu {
namespace gui {

/// @brief Component types that can appear in the topology editor.
enum class ComponentType : uint8_t {
  COMMAND_PROCESSOR,
  COMPUTE_UNIT,
  GPU_MEMORY,
  SHADER_ENGINE,
};

/// @brief A node in the topology editor representing a simulation component.
struct EditorNode {
  int id;
  ComponentType type;
  std::string name;
  int input_attr_id;
  int output_attr_id;
};

/// @brief A link between two nodes in the topology editor.
struct EditorLink {
  int id;
  int start_attr;
  int end_attr;
};

/// @brief CU configuration for the topology editor (local value type).
struct EditorCuConfig {
  uint32_t num_wf_slots = 10;
  uint32_t sgprs_per_wf = 104;
  uint32_t vgprs_per_wf = 256;
  uint32_t lds_size_kb = 64;
};

/// @brief Dear ImGui + imnodes topology editor for crafting simulation configs.
class TopologyEditor {
public:
  TopologyEditor();
  ~TopologyEditor();

  void init();
  void render();

  /// @brief Populate the editor from an existing simulation.
  void load_from_simulation(SoC *soc);

  uint32_t num_shader_engines() const { return num_shader_engines_; }
  uint32_t num_cus_per_se() const { return num_cus_per_se_; }

  EditorCuConfig &cu_config() { return cu_config_; }
  const EditorCuConfig &cu_config() const { return cu_config_; }

private:
  bool validate_link(int start_attr, int end_attr) const;
  void rebuild_nodes();
  const EditorNode *find_node_by_attr(int attr_id) const;

  std::vector<EditorNode> nodes_;
  std::vector<EditorLink> links_;
  int next_id_ = 1;

  uint32_t num_shader_engines_ = 1;
  uint32_t num_cus_per_se_ = 4;
  EditorCuConfig cu_config_;
};

} // namespace gui
} // namespace rocjitsu

#endif // ROCJITSU_TOOLS_SIMGUI_TOPOLOGY_EDITOR_H_
