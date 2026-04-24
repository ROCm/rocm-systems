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

#ifndef SRC_CORE_REGISTER_SCHEMA_HPP_
#define SRC_CORE_REGISTER_SCHEMA_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <optional>
#include <stdexcept>

namespace aql_profile {

/// Register identifier for architecture-agnostic register access
enum class RegisterId {
  // Graphics Ring Buffer Manager (GRBM)
  GRBM_GFX_INDEX,
  GRBM_PERFMON_CNTL,

  // Command Processor (CP)
  CP_PERFMON_CNTL,

  // Memory Controller (MC/UMC)
  MC_CONFIG,
  MC_SEQ_PMG_TIMING,

  // Shader Profiling (SQG/SQ)
  SQ_PERFCOUNTER_CTRL,
  SQG_PERFCOUNTER0_SELECT,
  SQG_PERFCOUNTER0_LO,
  SQG_PERFCOUNTER0_HI,

  // Thread Trace
  SQTT_BUF_BASE,
  SQTT_BUF_SIZE,
  SQTT_BUF_STATUS,

  // SPM (Streaming Performance Monitor)
  SPM_RING_BASE,
  SPM_RING_SIZE,
  SPM_PERFMON_SEGMENT_SIZE,

  // Add more as needed...
};

/// Register definition with address and metadata
struct RegisterDef {
  uint32_t offset;            // Register offset (byte address)
  uint32_t default_value;     // Default/reset value
  uint32_t write_mask;        // Bits that can be written (0xFFFFFFFF = all)
  std::string description;    // Human-readable description

  RegisterDef()
      : offset(0), default_value(0), write_mask(0xFFFFFFFF), description("") {}

  RegisterDef(uint32_t off, uint32_t def_val = 0, uint32_t mask = 0xFFFFFFFF,
              const std::string& desc = "")
      : offset(off), default_value(def_val), write_mask(mask), description(desc) {}
};

/// Register schema: maps register IDs to their definitions
/// Each architecture provides its own schema, potentially inheriting
/// from a base schema and overriding specific registers
class RegisterSchema {
 public:
  RegisterSchema() = default;
  virtual ~RegisterSchema() = default;

  /// Register a new register definition
  void DefineRegister(RegisterId id, const RegisterDef& def) {
    registers_[id] = def;
  }

  /// Register with simple offset only
  void DefineRegister(RegisterId id, uint32_t offset) {
    registers_[id] = RegisterDef(offset);
  }

  /// Get register definition (throws if not found)
  const RegisterDef& GetRegister(RegisterId id) const {
    auto it = registers_.find(id);
    if (it == registers_.end()) {
      throw std::runtime_error("Register not defined in schema");
    }
    return it->second;
  }

  /// Get register offset
  uint32_t GetOffset(RegisterId id) const {
    return GetRegister(id).offset;
  }

  /// Try to get register (returns nullopt if not found)
  std::optional<RegisterDef> TryGetRegister(RegisterId id) const {
    auto it = registers_.find(id);
    if (it == registers_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  /// Check if register is defined
  bool HasRegister(RegisterId id) const {
    return registers_.find(id) != registers_.end();
  }

  /// Get all defined registers
  const std::map<RegisterId, RegisterDef>& GetAllRegisters() const {
    return registers_;
  }

  /// Merge another schema into this one (for inheritance/overrides)
  void Merge(const RegisterSchema& other) {
    for (const auto& [id, def] : other.registers_) {
      registers_[id] = def;
    }
  }

 protected:
  std::map<RegisterId, RegisterDef> registers_;
};

/// Helper functions for common register operations

/// Create a GRBM broadcast value (architecture-specific)
inline uint32_t MakeGrbmBroadcastValue(uint32_t se_mask = 0x3FF,
                                       uint32_t sa_mask = 0x3,
                                       uint32_t instance_mask = 0x3) {
  return (se_mask << 16) | (sa_mask << 8) | instance_mask;
}

/// Create a GRBM index value for targeting specific SE/SA/instance
inline uint32_t MakeGrbmIndexValue(uint32_t se_index,
                                   uint32_t sa_index = 0,
                                   uint32_t instance_index = 0,
                                   bool broadcast_se = false,
                                   bool broadcast_sa = false,
                                   bool broadcast_instance = false) {
  uint32_t value = 0;
  if (!broadcast_se) value |= se_index;
  if (!broadcast_sa) value |= (sa_index << 8);
  if (!broadcast_instance) value |= (instance_index << 16);
  if (broadcast_se) value |= (1 << 30);
  if (broadcast_sa) value |= (1 << 29);
  if (broadcast_instance) value |= (1 << 28);
  return value;
}

}  // namespace aql_profile

#endif  // SRC_CORE_REGISTER_SCHEMA_HPP_
