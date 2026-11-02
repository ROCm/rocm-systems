// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/topology_assembly.h"

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "simdojo/sim/topology.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {

util::FailureOr<VirtualMachine *>
assemble_virtual_machine_topology(config::LoadedConfig &loaded, simdojo::Topology &topology,
                                  bool daemon_mode, util::DiagnosticEmitter emit_error) {
  SoC *primary_soc = loaded.soc();
  if (primary_soc == nullptr) {
    return emit_error.emit() << "topology root must be an SoC";
  }
  if (loaded.num_gpus > 1) {
    for (const auto &extra_build : loaded.extra_gpu_builds) {
      if (!dynamic_cast<SoC *>(extra_build.root.get())) {
        return emit_error.emit() << "each GPU topology root must be an SoC";
      }
    }
  }
  if (loaded.num_gpus > 1 && !loaded.extra_gpu_builds.empty()) {
    std::vector<std::unique_ptr<SoC>> socs;
    std::vector<uint32_t> gpu_ids;

    std::unique_ptr<simdojo::CompositeComponent> primary_root = loaded.take_root();
    primary_root.release();
    socs.push_back(std::unique_ptr<SoC>(primary_soc));
    gpu_ids.push_back(loaded.devices.empty() ? 0 : loaded.devices[0].gpu_id);

    for (std::size_t gpu_index = 0; gpu_index < loaded.extra_gpu_builds.size(); ++gpu_index) {
      config::TopologyBuildResult &extra_build = loaded.extra_gpu_builds[gpu_index];
      auto *extra_soc = dynamic_cast<SoC *>(extra_build.root.get());
      extra_build.root.release();
      socs.push_back(std::unique_ptr<SoC>(extra_soc));
      gpu_ids.push_back(gpu_index + 1 < loaded.devices.size() ? loaded.devices[gpu_index + 1].gpu_id
                                                              : 0);
    }

    auto machine =
        std::make_unique<VirtualMachine>(std::move(socs), std::move(gpu_ids), daemon_mode);
    VirtualMachine *machine_ptr = machine.get();
    topology.set_root(std::move(machine));

    auto prefix_specs = [](std::vector<simdojo::LinkSpec> &specs, const std::string &prefix) {
      for (auto &spec : specs) {
        spec.src = prefix + spec.src;
        spec.dst = prefix + spec.dst;
      }
    };
    prefix_specs(loaded.build_result.link_specs, "gpu0.");
    if (loaded.wire_links(topology, emit_error).failed()) {
      topology = simdojo::Topology{};
      return util::Result::failure();
    }
    primary_soc->wire_backing(topology);
    for (std::size_t gpu_index = 0; gpu_index < loaded.extra_gpu_builds.size(); ++gpu_index) {
      config::TopologyBuildResult &extra_build = loaded.extra_gpu_builds[gpu_index];
      prefix_specs(extra_build.link_specs, "gpu" + std::to_string(gpu_index + 1) + ".");
      if (topology.wire_links(extra_build.link_specs, loaded.exec_mode, emit_error).failed()) {
        topology = simdojo::Topology{};
        return util::Result::failure();
      }
      auto *extra_soc = machine_ptr->soc(static_cast<uint32_t>(gpu_index + 1));
      if (extra_soc)
        extra_soc->wire_backing(topology);
    }
    return machine_ptr;
  }

  std::unique_ptr<simdojo::CompositeComponent> root = loaded.take_root();
  root.release();
  auto machine = std::make_unique<VirtualMachine>(std::unique_ptr<SoC>(primary_soc), daemon_mode);
  VirtualMachine *machine_ptr = machine.get();
  topology.set_root(std::move(machine));
  if (loaded.wire_links(topology, emit_error).failed()) {
    topology = simdojo::Topology{};
    return util::Result::failure();
  }
  primary_soc->wire_backing(topology);
  return machine_ptr;
}

} // namespace rocjitsu
