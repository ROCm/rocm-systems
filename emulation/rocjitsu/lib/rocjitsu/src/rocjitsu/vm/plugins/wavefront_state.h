// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocjitsu {

/// Per-wavefront state owned by a plugin. Each plugin subclasses this to store
/// its own data on the wavefront. Access and install it through
/// ExecutionPlugin::wavefront_state<T>() and set_wavefront_state(); those
/// helpers preserve concrete-instance ownership when plugin groups replace one
/// another and reuse slot indices.
struct WavefrontState {
  virtual ~WavefrontState() = default;
};

} // namespace rocjitsu
