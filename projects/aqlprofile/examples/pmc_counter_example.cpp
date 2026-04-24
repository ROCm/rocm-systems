// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// End-to-end example demonstrating the new architecture abstraction system
// for PMC (Performance Monitor Counter) profiling

#include <iostream>
#include <memory>
#include <vector>

#include "core/architecture_registry.hpp"
#include "core/architectures/gfx9_architecture.hpp"
#include "core/architectures/gfx10_architecture.hpp"
#include "core/architectures/gfx11_architecture.hpp"
#include "aqlprofile-sdk/aql_profile_v2.h"

using namespace aql_profile;

/// Register all supported architectures
void RegisterArchitectures() {
  auto& registry = ArchitectureRegistry::Instance();

  // Create mock agent info for demonstration
  AgentInfo gfx90a_info = {};
  gfx90a_info.gfxip = "gfx90a";
  gfx90a_info.name = "MI200";
  gfx90a_info.se_num = 8;
  gfx90a_info.shader_arrays_per_se = 2;
  gfx90a_info.cu_num = 104;
  gfx90a_info.xcc_num = 1;

  AgentInfo gfx1100_info = {};
  gfx1100_info.gfxip = "gfx1100";
  gfx1100_info.name = "Navi31";
  gfx1100_info.se_num = 6;
  gfx1100_info.shader_arrays_per_se = 2;
  gfx1100_info.cu_num = 96;
  gfx1100_info.xcc_num = 1;

  // Register architectures
  registry.Register("gfx90a", std::make_unique<Gfx9Architecture>(&gfx90a_info));
  registry.Register("gfx1100", std::make_unique<Gfx11Architecture>(&gfx1100_info));
}

/// Example: Query architecture capabilities
void PrintArchitectureInfo(const char* gfxip) {
  auto& registry = ArchitectureRegistry::Instance();
  const auto* arch = registry.Lookup(gfxip);

  if (!arch) {
    std::cout << "Architecture " << gfxip << " not found!\n";
    return;
  }

  const auto& config = arch->GetConfig();

  std::cout << "\n=== Architecture: " << arch->GetDescription() << " ===\n";
  std::cout << "  GFXIP: " << config.gfxip << "\n";
  std::cout << "  Shader Engines: " << config.se_count << "\n";
  std::cout << "  Compute Units: " << config.cu_count << "\n";
  std::cout << "  Work Group Processors: " << arch->GetNumWGPs() << "\n";
  std::cout << "  XCC Count: " << config.xcc_count << "\n";
  std::cout << "  Multi-XCC: " << (config.IsMultiXCC() ? "Yes" : "No") << "\n";
  std::cout << "\n  Capabilities:\n";
  std::cout << "    PMC Support: " << (config.supports_pmc ? "Yes" : "No") << "\n";
  std::cout << "    SPM Support: " << (config.supports_spm ? "Yes" : "No") << "\n";
  std::cout << "    SQTT Support: " << (config.supports_sqtt ? "Yes" : "No") << "\n";
  std::cout << "    Concurrent Mode: " << (config.supports_concurrent ? "Yes" : "No") << "\n";
}

/// Example: Use register schema
void DemonstrateRegisterSchema(const char* gfxip) {
  auto& registry = ArchitectureRegistry::Instance();
  const auto* arch = registry.Lookup(gfxip);

  if (!arch) return;

  std::cout << "\n=== Register Schema for " << gfxip << " ===\n";

  const auto& schema = arch->GetRegisterSchema();

  // Query specific registers
  if (schema.HasRegister(RegisterId::GRBM_GFX_INDEX)) {
    uint32_t offset = schema.GetOffset(RegisterId::GRBM_GFX_INDEX);
    std::cout << "  GRBM_GFX_INDEX: 0x" << std::hex << offset << std::dec << "\n";
  }

  if (schema.HasRegister(RegisterId::SQ_PERFCOUNTER_CTRL)) {
    uint32_t offset = schema.GetOffset(RegisterId::SQ_PERFCOUNTER_CTRL);
    std::cout << "  SQ_PERFCOUNTER_CTRL: 0x" << std::hex << offset << std::dec << "\n";
  }
}

/// Example: Query block information
void DemonstrateBlockInfo(const char* gfxip) {
  auto& registry = ArchitectureRegistry::Instance();
  const auto* arch = registry.Lookup(gfxip);

  if (!arch) return;

  std::cout << "\n=== Block Information for " << gfxip << " ===\n";
  std::cout << "  Total blocks: " << arch->GetBlockCount() << "\n";

  // Find specific blocks
  const char* blocks_to_find[] = {"SQ", "TCP", "TCC", "GRBM"};

  for (const char* block_name : blocks_to_find) {
    uint32_t block_id = arch->FindBlockByName(block_name);
    if (block_id != UINT32_MAX) {
      const GpuBlockInfo* block = arch->GetBlockInfo(block_id);
      if (block) {
        std::cout << "  Block '" << block_name << "' (ID " << block_id << "):\n";
        std::cout << "    Instances: " << block->instance_count << "\n";
        std::cout << "    Counters: " << block->counter_count << "\n";
        std::cout << "    Max Event ID: " << block->event_id_max << "\n";

        // Calculate required bytes for this block
        size_t bytes = arch->GetBytesNeededForBlock(block_id);
        std::cout << "    Buffer size needed: " << bytes << " bytes\n";
      }
    }
  }
}

/// Example: PMC profiling workflow (simplified)
void DemonstratePMCWorkflow(const char* gfxip) {
  auto& registry = ArchitectureRegistry::Instance();
  const auto* arch = registry.Lookup(gfxip);

  if (!arch) return;

  std::cout << "\n=== PMC Profiling Workflow for " << gfxip << " ===\n";

  // Step 1: Check capabilities
  const auto& config = arch->GetConfig();
  if (!config.supports_pmc) {
    std::cout << "  PMC not supported on this architecture!\n";
    return;
  }

  std::cout << "  Step 1: PMC support confirmed\n";

  // Step 2: Find counters of interest
  uint32_t sq_block_id = arch->FindBlockByName("SQ");
  if (sq_block_id != UINT32_MAX) {
    std::cout << "  Step 2: Found SQ block (ID " << sq_block_id << ")\n";

    const GpuBlockInfo* sq_block = arch->GetBlockInfo(sq_block_id);
    if (sq_block) {
      std::cout << "    Available counters: " << sq_block->counter_count << "\n";
      std::cout << "    Event IDs: 0-" << sq_block->event_id_max << "\n";
    }
  }

  // Step 3: Allocate buffers
  uint32_t tcp_block_id = arch->FindBlockByName("TCP");
  if (tcp_block_id != UINT32_MAX) {
    size_t buffer_size = arch->GetBytesNeededForBlock(tcp_block_id);
    std::cout << "  Step 3: TCP block requires " << buffer_size << " byte buffer\n";
  }

  // Step 4: Create command builder
  pm4_builder::CmdBuilder* cmd_builder = arch->CreateCmdBuilder();
  if (cmd_builder) {
    std::cout << "  Step 4: Created PM4 command builder\n";
    delete cmd_builder;
  }

  std::cout << "  Workflow complete!\n";
}

int main() {
  std::cout << "=================================================\n";
  std::cout << "AQLProfile Architecture Abstraction Example\n";
  std::cout << "=================================================\n";

  // Step 1: Register all architectures
  RegisterArchitectures();

  auto& registry = ArchitectureRegistry::Instance();
  auto prefixes = registry.GetRegisteredPrefixes();

  std::cout << "\nRegistered architectures: ";
  for (const auto& prefix : prefixes) {
    std::cout << prefix << " ";
  }
  std::cout << "\n";

  // Step 2: Demonstrate architecture queries
  PrintArchitectureInfo("gfx90a");
  PrintArchitectureInfo("gfx1100");

  // Step 3: Demonstrate register schema
  DemonstrateRegisterSchema("gfx90a");
  DemonstrateRegisterSchema("gfx1100");

  // Step 4: Demonstrate block information
  DemonstrateBlockInfo("gfx90a");

  // Step 5: Demonstrate PMC workflow
  DemonstratePMCWorkflow("gfx90a");

  std::cout << "\n=================================================\n";
  std::cout << "Example complete!\n";
  std::cout << "=================================================\n";

  return 0;
}
