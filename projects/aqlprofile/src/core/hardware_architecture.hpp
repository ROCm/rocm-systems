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

#include <memory>
#include <string>
#include <vector>

#include "core/hardware_config.hpp"
#include "core/register_schema.hpp"
#include "def/gpu_block_info.h"

namespace pm4_builder {
class CmdBuilder;
class PrimitivesProvider;
}

namespace aql_profile {

/// Abstract base class representing a GPU hardware architecture
/// This is the central abstraction that encapsulates all architecture-specific
/// knowledge, replacing the previous factory hierarchy and hardcoded conditionals
class HardwareArchitecture {
 public:
  virtual ~HardwareArchitecture() = default;

  /// Get the hardware configuration for this architecture
  virtual const HardwareConfig& GetConfig() const = 0;

  /// Get the register schema for this architecture
  virtual const RegisterSchema& GetRegisterSchema() const = 0;

  /// Get block information for a given block ID
  /// Returns nullptr if block is not supported
  virtual const GpuBlockInfo* GetBlockInfo(uint32_t block_id) const = 0;

  /// Find block ID by name
  /// Returns UINT32_MAX if not found
  virtual uint32_t FindBlockByName(const char* name) const = 0;

  /// Get total number of blocks supported
  virtual uint32_t GetBlockCount() const = 0;

  /// Create a command builder for this architecture
  /// The caller takes ownership of the returned pointer
  virtual pm4_builder::CmdBuilder* CreateCmdBuilder() const = 0;

  /// Create a primitives provider for this architecture
  /// The caller takes ownership of the returned pointer
  virtual pm4_builder::PrimitivesProvider* CreatePrimitivesProvider() const = 0;

  /// Architecture version queries
  virtual bool IsGFX9() const { return false; }
  virtual bool IsGFX10() const { return false; }
  virtual bool IsGFX11() const { return false; }
  virtual bool IsGFX12() const { return false; }

  /// Specialized architecture queries
  virtual bool IsMI100() const { return false; }
  virtual bool IsMI200() const { return false; }
  virtual bool IsMI300() const { return false; }
  virtual bool IsMI350() const { return false; }

  /// Get the number of WGPs (Work Group Processors)
  /// For GFX10+, this is architecture-dependent
  /// For earlier architectures, approximated from CU count
  virtual uint32_t GetNumWGPs() const {
    return GetConfig().GetTotalWGPs();
  }

  /// Get accumulator register IDs (for SQ counters)
  /// Throws if not supported by this architecture
  virtual uint32_t GetAccumLowID() const {
    throw std::runtime_error("Accumulator registers not supported by this architecture");
  }

  virtual uint32_t GetAccumHiID() const {
    throw std::runtime_error("Accumulator registers not supported by this architecture");
  }

  /// Get SPM sample delay maximum
  virtual uint32_t GetSpmSampleDelayMax() const {
    return GetConfig().spm_sample_delay_max;
  }

  /// Get SQTT header version2 field (only meaningful when needs_sqtt_header_packet is set)
  virtual uint32_t GetSqttHeaderVersion() const {
    return GetConfig().sqtt_header_version;
  }

  /// Get the number of events/samples for a given block
  /// Accounts for SE/SA/WGP/XCC distribution
  virtual size_t GetNumEventsForBlock(uint32_t block_id) const;

  /// Get bytes needed for counter data for a given block
  virtual size_t GetBytesNeededForBlock(uint32_t block_id) const;

  /// Get a human-readable description of this architecture
  virtual std::string GetDescription() const {
    const auto& config = GetConfig();
    return config.name + " (" + config.gfxip + ")";
  }

 protected:
  // Protected constructor - only derived classes can instantiate
  HardwareArchitecture() = default;
};

}  // namespace aql_profile

#endif  // SRC_CORE_HARDWARE_ARCHITECTURE_HPP_
