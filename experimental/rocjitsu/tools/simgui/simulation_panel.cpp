// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simulation_panel.h"

#include "imgui.h"
#include "implot.h"

#include <cstdint>
#include <cstdio>

namespace rocjitsu {
namespace gui {

bool SimulationPanel::render_controls(bool &running, uint64_t tick, int &steps_per_frame) {
  bool step_requested = false;

  ImGui::Text("Tick: %lu", static_cast<unsigned long>(tick));
  ImGui::Separator();

  if (running) {
    if (ImGui::Button("Pause"))
      running = false;
  } else {
    if (ImGui::Button("Run"))
      running = true;
    ImGui::SameLine();
    if (ImGui::Button("Step"))
      step_requested = true;
  }

  ImGui::SameLine();
  if (ImGui::Button("Reset")) {
    running = false;
  }

  ImGui::SliderInt("Steps/Frame", &steps_per_frame, 1, 1000);

  return step_requested;
}

void SimulationPanel::render_wf_state(SoC *soc) {
  if (!soc)
    return;

  for (auto *xcd : soc->xcds()) {
    for (uint32_t se_idx = 0; se_idx < xcd->num_shader_engines(); ++se_idx) {
      auto *se = xcd->shader_engine(se_idx);
      for (uint32_t cu_idx = 0; cu_idx < se->num_compute_units(); ++cu_idx) {
        auto *cu = se->compute_unit(cu_idx);
        char label[64];
        std::snprintf(label, sizeof(label), "SE%u CU%u", se_idx, cu_idx);
        if (!ImGui::TreeNode(label))
          continue;

        ImGui::Text("Wavefronts: %zu / %u", cu->num_wfs(), cu->num_wf_slots());
        ImGui::Text("Active: %s", cu->has_active_wfs() ? "yes" : "no");

        if (ImGui::BeginTable("wf_table", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
          ImGui::TableSetupColumn("Slot");
          ImGui::TableSetupColumn("PC");
          ImGui::TableSetupColumn("EXEC");
          ImGui::TableSetupColumn("VCC");
          ImGui::TableSetupColumn("Halted");
          ImGui::TableHeadersRow();

          for (uint32_t w = 0; w < cu->num_wf_slots(); ++w) {
            auto *slot = cu->wf(w);
            if (!slot)
              continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%u", w);
            ImGui::TableNextColumn();
            ImGui::Text("0x%lx", static_cast<unsigned long>(slot->pc));
            ImGui::TableNextColumn();
            ImGui::Text("0x%lx", static_cast<unsigned long>(slot->exec()));
            ImGui::TableNextColumn();
            ImGui::Text("0x%lx", static_cast<unsigned long>(slot->vcc()));
            ImGui::TableNextColumn();
            const char *state_str = "?";
            switch (slot->state()) {
            case amdgpu::WfState::HALTED:
              state_str = "halted";
              break;
            case amdgpu::WfState::RUNNING:
              state_str = "running";
              break;
            case amdgpu::WfState::WAITCNT:
              state_str = "waitcnt";
              break;
            case amdgpu::WfState::BARRIER:
              state_str = "barrier";
              break;
            case amdgpu::WfState::ENDING:
              state_str = "ending";
              break;
            }
            ImGui::Text("%s", state_str);
          }
          ImGui::EndTable();
        }
        ImGui::TreePop();
      }
    }
  }
}

void SimulationPanel::render_timeline(SoC *, uint64_t) {
  if (history_.empty()) {
    ImGui::Text("No timeline data. Run the simulation to record activity.");
    return;
  }

  std::vector<double> ticks(history_.size());
  std::vector<double> active(history_.size());
  for (size_t i = 0; i < history_.size(); ++i) {
    ticks[i] = static_cast<double>(history_[i].tick);
    active[i] = static_cast<double>(history_[i].active_wfs);
  }

  if (ImPlot::BeginPlot("Active Wavefronts", ImVec2(-1, 200))) {
    ImPlot::SetupAxes("Tick", "Wavefronts");
    ImPlot::PlotLine("Active", ticks.data(), active.data(), static_cast<int>(ticks.size()));
    ImPlot::EndPlot();
  }
}

void SimulationPanel::record_tick(SoC *soc, uint64_t tick) {
  if (!soc)
    return;

  uint32_t active = 0;
  for (auto *xcd : soc->xcds()) {
    for (auto *se : xcd->shader_engines()) {
      for (uint32_t ci = 0; ci < se->num_compute_units(); ++ci) {
        auto *cu = se->compute_unit(ci);
        for (uint32_t w = 0; w < cu->num_wf_slots(); ++w) {
          auto *slot = cu->wf(w);
          if (slot && slot->state() != amdgpu::WfState::HALTED)
            ++active;
        }
      }
    }
  }
  history_.push_back({tick, active});

  if (history_.size() > 100000)
    history_.erase(history_.begin(), history_.begin() + 50000);
}

void SimulationPanel::clear_history() { history_.clear(); }

} // namespace gui
} // namespace rocjitsu
