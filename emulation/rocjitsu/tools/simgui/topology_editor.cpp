// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "topology_editor.h"

#include "imgui.h"
#include "imnodes.h"

#include <algorithm>

namespace rocjitsu {
namespace gui {

TopologyEditor::TopologyEditor() = default;

TopologyEditor::~TopologyEditor() { ImNodes::DestroyContext(); }

void TopologyEditor::init() {
  ImNodes::CreateContext();
  ImNodes::StyleColorsDark();
  rebuild_nodes();
}

void TopologyEditor::load_from_simulation(SoC *soc) {
  if (!soc)
    return;
  auto *xcd = soc->xcd(0);
  num_shader_engines_ = xcd->num_shader_engines();
  if (num_shader_engines_ > 0)
    num_cus_per_se_ = xcd->shader_engine(0)->num_compute_units();
  rebuild_nodes();
}

void TopologyEditor::rebuild_nodes() {
  nodes_.clear();
  links_.clear();
  next_id_ = 1;

  int mem_node_id = next_id_++;
  int mem_in_attr = next_id_++;
  nodes_.push_back({mem_node_id, ComponentType::GPU_MEMORY, "VRAM", mem_in_attr, -1});

  for (uint32_t se = 0; se < num_shader_engines_; ++se) {
    std::string se_prefix = "SE" + std::to_string(se);

    int cp_node_id = next_id_++;
    int cp_out_attr = next_id_++;
    nodes_.push_back(
        {cp_node_id, ComponentType::COMMAND_PROCESSOR, se_prefix + " CP", -1, cp_out_attr});

    for (uint32_t cu = 0; cu < num_cus_per_se_; ++cu) {
      int cu_node_id = next_id_++;
      int cu_in_attr = next_id_++;
      int cu_out_attr = next_id_++;
      nodes_.push_back({cu_node_id, ComponentType::COMPUTE_UNIT,
                        se_prefix + " CU" + std::to_string(cu), cu_in_attr, cu_out_attr});

      links_.push_back({next_id_++, cp_out_attr, cu_in_attr});
      links_.push_back({next_id_++, cu_out_attr, mem_in_attr});
    }
  }
}

const EditorNode *TopologyEditor::find_node_by_attr(int attr_id) const {
  for (const auto &node : nodes_) {
    if (node.input_attr_id == attr_id || node.output_attr_id == attr_id)
      return &node;
  }
  return nullptr;
}

bool TopologyEditor::validate_link(int start_attr, int end_attr) const {
  const auto *src = find_node_by_attr(start_attr);
  const auto *dst = find_node_by_attr(end_attr);
  if (!src || !dst)
    return false;

  bool src_is_output = (src->output_attr_id == start_attr);
  bool dst_is_input = (dst->input_attr_id == end_attr);
  if (!src_is_output || !dst_is_input)
    return false;

  if (src->type == ComponentType::COMMAND_PROCESSOR && dst->type == ComponentType::COMPUTE_UNIT)
    return true;

  if (src->type == ComponentType::COMPUTE_UNIT && dst->type == ComponentType::GPU_MEMORY)
    return true;

  return false;
}

void TopologyEditor::render() {
  bool changed = false;
  int ses = static_cast<int>(num_shader_engines_);
  int cus = static_cast<int>(num_cus_per_se_);
  if (ImGui::SliderInt("Shader Engines", &ses, 1, 8)) {
    num_shader_engines_ = static_cast<uint32_t>(ses);
    changed = true;
  }
  if (ImGui::SliderInt("CUs per SE", &cus, 1, 16)) {
    num_cus_per_se_ = static_cast<uint32_t>(cus);
    changed = true;
  }
  if (changed)
    rebuild_nodes();

  ImGui::Separator();

  ImNodes::BeginNodeEditor();

  for (const auto &node : nodes_) {
    switch (node.type) {
    case ComponentType::COMMAND_PROCESSOR:
      ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(180, 80, 80, 255));
      break;
    case ComponentType::COMPUTE_UNIT:
      ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(80, 120, 180, 255));
      break;
    case ComponentType::GPU_MEMORY:
      ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(80, 180, 80, 255));
      break;
    default:
      break;
    }

    ImNodes::BeginNode(node.id);

    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted(node.name.c_str());
    ImNodes::EndNodeTitleBar();

    if (node.input_attr_id >= 0) {
      ImNodes::BeginInputAttribute(node.input_attr_id);
      const char *label = (node.type == ComponentType::COMPUTE_UNIT) ? "cpl" : "cpl";
      ImGui::Text("%s", label);
      ImNodes::EndInputAttribute();
    }

    if (node.output_attr_id >= 0) {
      ImNodes::BeginOutputAttribute(node.output_attr_id);
      const char *label = (node.type == ComponentType::COMMAND_PROCESSOR) ? "req" : "req";
      ImGui::Text("%s", label);
      ImNodes::EndOutputAttribute();
    }

    ImNodes::EndNode();
    ImNodes::PopColorStyle();
  }

  for (const auto &link : links_) {
    ImNodes::Link(link.id, link.start_attr, link.end_attr);
  }

  ImNodes::EndNodeEditor();

  int start_attr, end_attr;
  if (ImNodes::IsLinkCreated(&start_attr, &end_attr)) {
    if (validate_link(start_attr, end_attr)) {
      links_.push_back({next_id_++, start_attr, end_attr});
    }
  }

  int destroyed_link_id;
  if (ImNodes::IsLinkDestroyed(&destroyed_link_id)) {
    links_.erase(std::remove_if(links_.begin(), links_.end(),
                                [destroyed_link_id](const EditorLink &l) {
                                  return l.id == destroyed_link_id;
                                }),
                 links_.end());
  }
}

} // namespace gui
} // namespace rocjitsu
