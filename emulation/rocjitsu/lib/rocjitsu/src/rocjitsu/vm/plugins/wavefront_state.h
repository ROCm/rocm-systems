// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocjitsu {

/// Per-wavefront state owned by a plugin. Each plugin subclasses this to store
/// its own data on the wavefront, accessed via Wavefront::plugin_states_[slot_index()].
struct WavefrontState {
  virtual ~WavefrontState() = default;
};

} // namespace rocjitsu
