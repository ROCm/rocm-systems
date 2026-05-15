// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_CORE_HARDWARE_ARCHITECTURE_HPP_
#define SRC_CORE_HARDWARE_ARCHITECTURE_HPP_

#include <cstdint>
#include <cstring>

#include "core/hw/hardware_config.hpp"
#include "def/gpu_block_info.h"

namespace aql_profile {

/// Abstract base class providing architecture-specific metadata for GFX12+.
/// Initially adopted for gfx1200/gfx1201/gfx1250 only; older GPUs continue
/// to use the legacy Pm4Factory subclass pattern (no HardwareArchitecture).
///
/// This layer is intentionally narrow: it covers block-info lookup and
/// capability flags.  It does NOT own PM4 builder objects — those remain
/// in the templated GpuPmcBuilder/GpuSpmBuilder/GpuSqttBuilder hierarchy.
class HardwareArchitecture {
 public:
  virtual ~HardwareArchitecture() = default;

  virtual const HardwareConfig& GetConfig() const = 0;

  virtual bool IsGFX12() const { return false; }

  /// Block info lookup — mirrors the old BlockInfoMap interface.
  virtual const GpuBlockInfo* GetBlockInfo(uint32_t block_id) const = 0;
  virtual uint32_t FindBlockByName(const char* name) const = 0;
  virtual uint32_t GetBlockCount() const = 0;
  virtual const GpuBlockInfo** GetBlockTable() const = 0;

  /// Number of counter samples for a given block (accounts for SE/SA/WGP/XCC).
  virtual size_t GetNumEventsForBlock(uint32_t block_id) const;

  /// Total bytes needed to store one read of a given block across all XCCs.
  virtual size_t GetBytesNeededForBlock(uint32_t block_id) const;

  /// WGPs per shader array — matches GpuPmcBuilder::wgp_per_sa_ used in Read().
  virtual int GetNumWGPs() const {
    const auto& c = GetConfig();
    uint32_t wgp_per_sa =
        (c.cu_count / 2 + c.sa_per_se_count * c.GetSEPerXCC() - 1) /
        (c.GetSEPerXCC() * c.sa_per_se_count);
    return static_cast<int>(wgp_per_sa / c.xcc_count);
  }

  /// Accumulator register IDs for SQ counters (GFX12 returns 1/1).
  virtual int GetAccumLowID() const { return -1; }
  virtual int GetAccumHiID() const  { return -1; }

  /// SPM sample delay maximum (0 means "no limit / not applicable").
  virtual uint32_t GetSpmSampleDelayMax() const {
    return GetConfig().spm_sample_delay_max;
  }

 protected:
  HardwareArchitecture() = default;
};

}  // namespace aql_profile

#endif  // SRC_CORE_HARDWARE_ARCHITECTURE_HPP_
