// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_IOD_H_
#define ROCJITSU_VM_AMDGPU_IOD_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/hbm_controller.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"

#include "simdojo/sim/component.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief I/O Die — contains memory-side cache, memory controllers, and fabric routing.
///
/// @details Each IOD serves a subset of XCDs and owns a portion of the memory-side cache
/// and HBM stacks. The number of IODs and their XCD assignments are driven by config.
class Iod : public simdojo::CompositeComponent {
public:
  struct Config {
    uint32_t num_hbm_stacks; ///< Number of HBM stacks on this IOD.
  };

  Iod(std::string name, const Config &config, GpuMemory *memory);

  void initialize() override;

  /// @brief Create a completer port for an XCD's L2 req connection.
  simdojo::Port *create_cpl_port(const std::string &xcd_name);

  /// @brief Peer requester port (for cross-IOD traffic).
  simdojo::Port *peer_req_port() { return peer_req_; }

  /// @brief Peer completer port (for cross-IOD traffic).
  simdojo::Port *peer_cpl_port() { return peer_cpl_; }

  /// @brief Return the memory-side cache.
  MemorySideCache *msc() { return msc_; }

  /// @brief Return the HBM controller for this IOD.
  HbmController *hbm_controller() { return hbm_; }

  /// @brief Return the HBM requester ports (structural, for topology visibility).
  const std::vector<simdojo::Port *> &req_ports() const { return req_ports_; }

private:
  MemorySideCache *msc_ = nullptr;
  HbmController *hbm_ = nullptr;
  std::vector<simdojo::Port *> cpl_ports_;
  simdojo::Port *peer_req_ = nullptr;
  simdojo::Port *peer_cpl_ = nullptr;
  std::vector<simdojo::Port *> req_ports_;
  std::unique_ptr<simdojo::Link> msc_hbm_link_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_IOD_H_
