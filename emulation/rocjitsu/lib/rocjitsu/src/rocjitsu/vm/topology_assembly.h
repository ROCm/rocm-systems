// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/diagnostic.h"
#include "util/result.h"

namespace simdojo {
class Topology;
} // namespace simdojo

namespace rocjitsu {
class VirtualMachine;

namespace config {
struct LoadedConfig;
} // namespace config

/// @brief Transfer loaded SoCs into a VirtualMachine and wire their topology.
/// @details Consumes the root trees in loaded. The returned machine is owned by
/// topology and remains valid until its root is replaced or destroyed.
/// @param loaded A configuration with a primary SoC root.
/// @param topology The topology that will own the assembled machine.
/// @param daemon_mode Whether the machine's KFD driver serves daemon requests.
/// @return The assembled machine, owned by topology, or a diagnosed failure.
/// Invalid roots leave loaded untouched. Wiring failure consumes loaded and
/// clears topology, so no partially wired machine can be used.
util::FailureOr<VirtualMachine *>
assemble_virtual_machine_topology(config::LoadedConfig &loaded, simdojo::Topology &topology,
                                  bool daemon_mode, util::DiagnosticEmitter emit_error);

} // namespace rocjitsu
