// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/kmd/linux/linux_kfd.h"

#include <cstdint>
#include <mutex>

namespace rocjitsu {

class SimulatedKfd;

/// KFD context selection belongs to the open file description, including its aliases.
class KfdOpenFile final : public LinuxKfd::OpenFile {
public:
  KfdOpenFile(SimulatedKfd &driver, uint32_t primary_process_id);
  ~KfdOpenFile() override;

  /// Select a fresh secondary context once; an existing selection returns EINVAL.
  int create_process();
  /// Snapshot the current context while selection is serialized.
  uint32_t process_id() const;

private:
  /// Backend outlives its file table and every reserved file operation.
  SimulatedKfd &driver_;
  /// Shared primary context, retained by the backend's operational references.
  const uint32_t primary_process_id_;
  /// Serializes the single primary-to-secondary transition.
  mutable std::mutex mutex_;
  /// Selected context; a secondary context is released by this file's destructor.
  uint32_t process_id_;
};

} // namespace rocjitsu
