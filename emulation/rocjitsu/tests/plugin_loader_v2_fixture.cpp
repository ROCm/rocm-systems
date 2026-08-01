// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This fixture snapshots the v2 ExecutionPlugin contract. It intentionally
// does not include the current execution_plugin.h or plugin_abi.h: doing so
// would compile against the host's v3 vtable and fail to model a stale module.

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/plugins/wavefront_state.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <utility>

namespace rocjitsu {

// Preserved from plugin ABI v2, before requires_serial_execution() was added.
class ExecutionPlugin {
public:
  static constexpr uint8_t kFullByteMask = 0xF;
  static constexpr uint8_t kLowHalfByteMask = 0b0011u;
  static constexpr uint8_t kHighHalfByteMask = 0b1100u;

  explicit ExecutionPlugin(std::string name) : name_(std::move(name)) {}
  virtual ~ExecutionPlugin() = default;

  const std::string &name() const { return name_; }
  uint32_t slot_index() const { return slot_index_; }
  PluginSink &sink() { return *sink_; }

  virtual void onInit() {}
  virtual void onShutdown() {}
  virtual void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &,
                                                amdgpu::Wavefront &) {}
  virtual void onAmdgpuAfterExecuteInstruction(uint64_t, const Instruction &, amdgpu::Wavefront &) {
  }
  virtual void onAmdgpuRouteMemoryInstruction(const Instruction &, amdgpu::Wavefront &) {}
  virtual void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &) {}
  virtual void onAmdgpuDispatchExecutionBegin(uint32_t) {}
  virtual void onAmdgpuDispatchExecutionEnd(uint32_t) {}
  virtual void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                           std::span<amdgpu::Wavefront *>) {}
  virtual void onAmdgpuWorkgroupCompleted(uint32_t, uint32_t) {}
  virtual void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &) {}
  virtual void onAmdgpuWavefrontHalted(amdgpu::Wavefront &) {}
  virtual void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *, uint32_t, uint64_t,
                                     uint8_t = kFullByteMask) {}
  virtual void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *, uint32_t, uint64_t,
                                      uint8_t = kFullByteMask) {}
  virtual void onAmdgpuReadSgpr(const amdgpu::Wavefront *, uint32_t) {}
  virtual void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *>) {}

private:
  std::string name_;
  uint32_t slot_index_ = 0;
  PluginSink *sink_ = &StderrSink::instance();
};

// Preserved from plugin ABI v2. Keep the contract type and exported function
// signature identical to the host so UBSan can safely call the metadata
// accessor before the loader rejects the stale ABI version.
struct PluginMetadata {
  int abi;
  const char *name;
  const char *contact;
  const char *version;
  const char *config_schema;
};

} // namespace rocjitsu

namespace {

void trace_create() {
  const char *path = std::getenv("ROCJITSU_PLUGIN_TEST_TRACE");
  if (!path)
    return;
  if (FILE *file = std::fopen(path, "a")) {
    std::fputs("legacy_v2:create\n", file);
    std::fclose(file);
  }
}

class LegacyV2Plugin final : public rocjitsu::ExecutionPlugin {
public:
  LegacyV2Plugin() : ExecutionPlugin("legacy_v2") {}
};

} // namespace

extern "C" RJ_API_EXPORT const rocjitsu::PluginMetadata *rocjitsu_plugin_metadata() {
  static const rocjitsu::PluginMetadata metadata{2, "legacy_v2", "rocjitsu-tests", "1", "{}"};
  return &metadata;
}

extern "C" RJ_API_EXPORT void *rocjitsu_plugin_create(const char *) {
  trace_create();
  return new LegacyV2Plugin();
}

extern "C" RJ_API_EXPORT void rocjitsu_plugin_destroy(void *handle) {
  delete static_cast<rocjitsu::ExecutionPlugin *>(handle);
}
