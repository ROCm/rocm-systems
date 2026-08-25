// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/data_hazard/plugin.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include "flatbuffers/flexbuffers.h"
#include "flatbuffers/idl.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::data_hazard {

namespace {

constexpr uint32_t kBytesPerDword = 4;

bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

/// The hazard counter a store is outstanding on, taken from the counter the
/// decoder assigned it. gfx9 and CDNA count stores on the same vmcnt as loads;
/// gfx10 and later count them on vscnt/storecnt of their own.
hazard_core::WaitCntType make_store_wait(amdgpu::WaitCounterType counter) {
  switch (counter) {
  case amdgpu::WaitCounterType::VSCNT:
  case amdgpu::WaitCounterType::STORECNT:
    return hazard_core::WaitCntType::STORE;
  default:
    return hazard_core::WaitCntType::VMEM;
  }
}

bool is_vector_load_mnemonic(std::string_view mnemonic) {
  return starts_with(mnemonic, "buffer_load") || starts_with(mnemonic, "global_load") ||
         starts_with(mnemonic, "flat_load") || starts_with(mnemonic, "scratch_load");
}

/// LDS addresses are reported absolute; hazard tracking wants them relative to
/// the wave's LDS allocation.
uint32_t normalize_lds_addr(const amdgpu::Wavefront &wf, uint32_t addr) {
  const uint32_t base = wf.lds_base();
  return addr >= base ? addr - base : addr;
}

bool has_routed_memory_state(const Instruction &inst) {
  const auto *state = inst.data();
  if (state == nullptr)
    return false;

  switch (state->tag()) {
  case amdgpu::SCALAR_MEM:
  case amdgpu::GLOBAL_MEM:
  case amdgpu::LOCAL_MEM:
    return true;
  default:
    return false;
  }
}

/// The wait an instruction expresses, read from the operand text because the
/// encoded field layout differs across GFX generations.
WaitInfo parse_wait(const Instruction &inst) {
  const WaitKind kind = make_wait_kind(inst.mnemonic());
  if (kind == WaitKind::None || inst.num_src_operands() == 0)
    return WaitInfo{kind, 0, 0};

  const auto *op = inst.src_operand(0);
  const std::string text = op != nullptr ? op->name() : std::string{};
  return make_wait_info(kind, text);
}

RegisterClass make_register_class(RegClass cls) {
  switch (cls) {
  case RegClass::SGPR:
    return RegisterClass::Scalar;
  case RegClass::VGPR:
    return RegisterClass::Vector;
  case RegClass::ACC_VGPR:
    return RegisterClass::AccumVector;
  default:
    return RegisterClass::None;
  }
}

bool is_vector_register_class(RegisterClass reg_class) {
  return reg_class == RegisterClass::Vector || reg_class == RegisterClass::AccumVector;
}

/// Hazard tracking is per logical register, but callbacks report physical file
/// indices, so subtract the wave's allocation base.
uint32_t logical_vgpr_base(const amdgpu::Wavefront &wf, uint32_t physical_reg) {
  return physical_reg >= wf.vgpr_alloc().base ? physical_reg - wf.vgpr_alloc().base : physical_reg;
}

uint32_t logical_sgpr_base(const amdgpu::Wavefront &wf, uint32_t physical_reg) {
  return physical_reg >= wf.sgpr_alloc().base ? physical_reg - wf.sgpr_alloc().base : physical_reg;
}

hazard_core::ExecutionKey make_wave_key(const amdgpu::Wavefront &wf) {
  return hazard_core::ExecutionKey{wf.dispatch_id(), 0, wf.wg_id(), wf.wf_id()};
}

std::array<uint32_t, 4> copy_raw_isa(const Instruction &inst) {
  std::array<uint32_t, 4> raw{};
  const uint32_t *words = inst.raw_encoding();
  if (words == nullptr)
    return raw;
  const size_t word_count =
      std::min<size_t>(raw.size(), static_cast<size_t>(inst.size()) / sizeof(uint32_t));
  std::copy(words, words + word_count, raw.begin());
  return raw;
}

} // namespace

// -- InstructionFormatter ----------------------------------------------------

void InstructionFormatter::remember(const InstructionView &view, const Instruction &inst) {
  disassembly_.record(view.pc, inst);

  std::lock_guard<std::mutex> lock(mutex_);
  const auto [it, inserted] = instructions_.try_emplace(view.instruction_id);
  if (inserted)
    insertion_order_.push_back(view.instruction_id);
  it->second = view;
  while (insertion_order_.size() > kMaxCachedInstructions) {
    instructions_.erase(insertion_order_.front());
    insertion_order_.pop_front();
  }
}

std::string InstructionFormatter::format_instruction(
    const hazard_core::InstructionDescriptor &instruction) const {
  InstructionView view;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = instructions_.find(instruction.instruction_id);
    if (it == instructions_.end())
      return format_fallback(instruction);
    view = it->second;
  }
  return format(view, disassembly_.lookup(view.pc));
}

void InstructionFormatter::clear() {
  disassembly_.clear();

  std::lock_guard<std::mutex> lock(mutex_);
  instructions_.clear();
  insertion_order_.clear();
}

std::string InstructionFormatter::format(const InstructionView &view,
                                         const std::string &disassembly) {
  std::ostringstream out;
  if (!disassembly.empty())
    out << disassembly << "  // ";
  out << "pc=0x" << std::hex << view.pc << std::dec << ", id=" << view.instruction_id << ", raw=["
      << format_raw_isa_hex(view.raw_isa) << "]";
  return out.str();
}

std::string
InstructionFormatter::format_fallback(const hazard_core::InstructionDescriptor &instruction) {
  std::ostringstream out;
  out << "pc=0x" << std::hex << instruction.pc << std::dec << ", id=" << instruction.instruction_id
      << ", raw=[" << format_raw_isa_hex(instruction.raw_isa) << "]";
  return out.str();
}

// -- DataHazardPlugin --------------------------------------------------------

namespace {

/// Reads the resolved config object. The loader has already applied schema
/// defaults, so absent keys simply leave the default in place.
PluginConfig parse_config(const char *config_json) {
  PluginConfig config;
  if (config_json == nullptr || *config_json == '\0')
    return config;

  flatbuffers::Parser parser;
  flexbuffers::Builder builder;
  if (!parser.ParseFlexBuffer(config_json, nullptr, &builder)) {
    util::Logger::warn("data_hazard: could not parse plugin configuration; using defaults");
    return config;
  }

  auto root = flexbuffers::GetRoot(builder.GetBuffer());
  if (!root.IsMap())
    return config;

  auto map = root.AsMap();
  auto path = map["report_path"];
  if (path.IsString())
    config.report_path = path.AsString().str();
  auto verbose_value = map["verbose"];
  if (verbose_value.IsBool())
    config.verbose = verbose_value.AsBool();
  return config;
}

/// @brief Flushes hazard reports when the process exits without a VM teardown.
///
/// A launched application detaches the VM thread and usually exits without
/// destroying the VM, so neither onShutdown nor the plugin destructor is
/// guaranteed to run. Live plugins register here and this object's destructor,
/// which runs while the plugin library is still loaded, writes whatever has not
/// been written yet. Reports are idempotent, so an earlier flush wins.
///
/// This is a best-effort flush. Detached partition threads can still be running
/// when it fires and the plugin cannot join them, so it marks each plugin as
/// shutting down first: callbacks entered afterwards return without touching
/// engine state. A callback already in flight still finishes against the engine
/// and workgroup locks, so the report may miss a hazard detected in that window,
/// but it cannot observe half-written state.
class ExitFlush {
public:
  ~ExitFlush() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (DataHazardPlugin *plugin : plugins_)
      plugin->begin_shutdown();
    for (DataHazardPlugin *plugin : plugins_)
      plugin->write_report();
  }

  static ExitFlush &instance() {
    static ExitFlush flush;
    return flush;
  }

  void add(DataHazardPlugin *plugin) {
    std::lock_guard<std::mutex> lock(mutex_);
    plugins_.push_back(plugin);
  }

  void remove(DataHazardPlugin *plugin) {
    std::lock_guard<std::mutex> lock(mutex_);
    plugins_.erase(std::remove(plugins_.begin(), plugins_.end(), plugin), plugins_.end());
  }

private:
  std::mutex mutex_;
  std::vector<DataHazardPlugin *> plugins_;
};

} // namespace

DataHazardPlugin::DataHazardPlugin(const char *config_json)
    : ExecutionPlugin("data_hazard"), config_(parse_config(config_json)),
      collector_(config_.verbose), warning_sink_(collector_, &formatter_), adapter_(engine_) {
  engine_.set_instruction_formatter(&formatter_);
  engine_.set_warning_sink(&warning_sink_);
  engine_.set_diagnostic_handler(
      [](const std::string &message) { util::Logger::warn("data_hazard: ", message); });
  // Stream findings as they are made. The JSON report and the summary are only
  // written when the run ends, which for a launched application is process
  // exit, so this is the only output available while the run is in progress.
  collector_.set_on_new_warning(
      [this](const HazardWarning &warning) { write_to_sink(format_warning_line(warning) + "\n"); });
  ExitFlush::instance().add(this);
}

DataHazardPlugin::~DataHazardPlugin() {
  ExitFlush::instance().remove(this);
  // The group may destroy the plugin without a preceding onShutdown (for
  // example when the VM tears down early), so make sure the report still lands.
  begin_shutdown();
  write_report();
  engine_.set_diagnostic_handler(nullptr);
}

void DataHazardPlugin::write_to_sink(const std::string &text) {
  std::lock_guard<std::mutex> lock(sink_mutex_);
  sink().write(text);
}

void DataHazardPlugin::begin_shutdown() { shutting_down_.store(true, std::memory_order_release); }

void DataHazardPlugin::onInit() {
  engine_.reset();
  adapter_.reset();
  collector_.clear();
  formatter_.clear();
  next_instruction_id_.store(0);
  shutting_down_.store(false, std::memory_order_release);
}

void DataHazardPlugin::onShutdown() {
  adapter_.on_shutdown();
  begin_shutdown();
  write_report();
}

void DataHazardPlugin::write_report() {
  std::call_once(report_once_, [this] {
    write_to_sink(collector_.to_summary());
    if (config_.report_path.empty())
      return;
    if (!collector_.write_json_file(config_.report_path)) {
      util::Logger::warn("data_hazard: could not write hazard report to '", config_.report_path,
                         "'");
    }
  });
}

void DataHazardPlugin::seed_wave_state(amdgpu::Wavefront &wf) {
  wf.set_plugin_state(slot_index(), std::make_unique<DataHazardWavefrontState>());
}

DataHazardWavefrontState *DataHazardPlugin::wave_state(amdgpu::Wavefront &wf) const {
  return static_cast<DataHazardWavefrontState *>(wf.plugin_state(slot_index()));
}

const DataHazardWavefrontState *DataHazardPlugin::wave_state(const amdgpu::Wavefront *wf) const {
  if (wf == nullptr)
    return nullptr;
  return static_cast<const DataHazardWavefrontState *>(wf->plugin_state(slot_index()));
}

void DataHazardPlugin::onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
  if (shutting_down())
    return;
  adapter_.on_dispatch_begin(dispatch_id);
}

void DataHazardPlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  if (shutting_down())
    return;
  adapter_.on_dispatch_end(dispatch_id);
}

void DataHazardPlugin::onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                                   uint32_t /*physical_vgpr_count*/,
                                                   uint32_t /*sgpr_count*/,
                                                   std::span<amdgpu::Wavefront *> wavefronts) {
  if (shutting_down())
    return;
  adapter_.on_workgroup_begin(hazard_core::ExecutionKey{dispatch_id, 0, wg_id, 0});
  for (amdgpu::Wavefront *wf : wavefronts) {
    if (wf != nullptr)
      seed_wave_state(*wf);
  }
}

void DataHazardPlugin::onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) {
  if (shutting_down())
    return;
  adapter_.on_workgroup_end(hazard_core::ExecutionKey{dispatch_id, 0, wg_id, 0});
}

void DataHazardPlugin::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
  if (shutting_down())
    return;
  // Wavefront slots are recycled, so clear any state left by a prior wave.
  if (DataHazardWavefrontState *state = wave_state(wf))
    state->reset();
  adapter_.on_wave_begin(make_wave_key(wf));
}

void DataHazardPlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  if (shutting_down())
    return;
  adapter_.on_wave_end(make_wave_key(wf));
  if (DataHazardWavefrontState *state = wave_state(wf))
    state->reset();
}

void DataHazardPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                        amdgpu::Wavefront &wf) {
  if (shutting_down())
    return;
  InstructionView view;
  view.execution = make_wave_key(wf);
  view.instruction_id = next_instruction_id();
  view.pc = pc;
  view.raw_isa = copy_raw_isa(inst);
  view.wait = parse_wait(inst);

  const bool is_memory_op = inst.is_memory_op() || has_routed_memory_state(inst);

  formatter_.remember(view, inst);
  adapter_.on_instruction(view);

  if (DataHazardWavefrontState *state = wave_state(wf)) {
    state->current = view;
    state->is_memory_op = is_memory_op;
  }

  emit_source_reads(view, inst, is_memory_op);
  if (inst.mnemonic() == "tensor_load_to_lds")
    emit_tensor_lds_write(view, wf);
  // Destination writes of memory instructions are emitted when the access is
  // routed, where the wait counter that guards them is known.
  if (!is_memory_op)
    emit_destination_writes(view, inst);
}

void DataHazardPlugin::onAmdgpuRouteMemoryInstruction(const Instruction &inst,
                                                      amdgpu::Wavefront &wf) {
  if (shutting_down())
    return;
  const DataHazardWavefrontState *state = wave_state(wf);
  if (state == nullptr || state->current.instruction_id == 0 || inst.data() == nullptr)
    return;

  if (inst.data()->tag() == amdgpu::SCALAR_MEM) {
    route_scalar_memory(inst, wf, state->current);
    return;
  }
  route_vector_memory(inst, wf, state->current);
}

void DataHazardPlugin::route_scalar_memory(const Instruction &inst, amdgpu::Wavefront &wf,
                                           const InstructionView &current) {
  const auto *smem = inst.data_as<amdgpu::ScalarMemState>();
  if (smem == nullptr)
    return;

  MemoryRouteView route;
  route.instruction = current;
  route.resource_kind = hazard_core::ResourceKind::ScalarRegister;
  route.register_class = RegisterClass::Scalar;
  route.register_base = logical_sgpr_base(wf, smem->dst_register.index);
  route.size_bytes = std::max<uint32_t>(1, smem->num_dwords) * kBytesPerDword;
  route.is_load = smem->is_load;
  route.is_store = !smem->is_load;
  route.address = smem->addr;
  adapter_.on_memory_route(route);
}

void DataHazardPlugin::route_vector_memory(const Instruction &inst, amdgpu::Wavefront &wf,
                                           const InstructionView &current) {
  const auto *vmem = inst.data_as<amdgpu::VectorMemState>();
  if (vmem == nullptr)
    return;

  MemoryRouteView route;
  route.instruction = current;
  route.register_class = RegisterClass::Vector;
  route.register_base = logical_vgpr_base(wf, vmem->dst_reg_base);
  route.size_bytes =
      std::max<uint32_t>(1, vmem->elem_size * std::max<uint32_t>(1, vmem->num_elems));
  route.is_load = vmem->is_load;
  route.is_store = !vmem->is_load;
  route.is_atomic = vmem->atomic_op != amdgpu::AtomicOp::NONE;
  route.exec_mask = vmem->lane_mask != 0 ? vmem->lane_mask : vmem->exec_mask;
  route.is_flat = starts_with(inst.mnemonic(), "flat_");
  route.store_wait = make_store_wait(vmem->wait_counter_type);

  const size_t lanes = std::min<size_t>(vmem->wf_size, vmem->per_lane_addr.size());
  route.per_lane_addresses.reserve(lanes);
  for (size_t lane = 0; lane < lanes; ++lane)
    route.per_lane_addresses.push_back(vmem->per_lane_addr[lane]);

  if (inst.data()->tag() == amdgpu::LOCAL_MEM) {
    route.resource_kind = hazard_core::ResourceKind::LocalMemory;
    for (auto &addr : route.per_lane_addresses)
      addr = normalize_lds_addr(wf, static_cast<uint32_t>(addr));
    adapter_.on_memory_route(route);
    return;
  }

  if (inst.data()->tag() != amdgpu::GLOBAL_MEM)
    return;

  route.resource_kind = starts_with(inst.mnemonic(), "scratch_")
                            ? hazard_core::ResourceKind::ScratchMemory
                            : hazard_core::ResourceKind::GlobalMemory;

  // A global load with an LDS destination writes LDS asynchronously; the LDS
  // write is guarded by the load counter rather than the DS counter.
  if (vmem->lds_dst) {
    route.writes_local_memory = true;
    route.local_write_wait = hazard_core::WaitCntType::VMEM;
    route.local_size_bytes = route.size_bytes;
    route.per_lane_local_addresses.reserve(lanes);
    const uint32_t per_lane_bytes = route.size_bytes;
    for (size_t lane = 0; lane < lanes; ++lane) {
      const uint32_t addr = vmem->lds_per_lane_addr
                                ? vmem->per_lane_lds_addr[lane]
                                : vmem->lds_base + static_cast<uint32_t>(lane) * per_lane_bytes;
      route.per_lane_local_addresses.push_back(normalize_lds_addr(wf, addr));
    }
  }
  adapter_.on_memory_route(route);
}

void DataHazardPlugin::onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                             uint64_t /*lane_mask*/, uint8_t /*byte_mask*/) {
  if (shutting_down())
    return;
  const DataHazardWavefrontState *state = wave_state(wf);
  if (state == nullptr || state->current.instruction_id == 0 || state->is_memory_op)
    return;

  RegisterAccessView access;
  access.instruction = state->current;
  access.register_class = RegisterClass::Vector;
  access.physical_reg = logical_vgpr_base(*wf, physical_reg);
  access.size_bytes = kBytesPerDword;
  access.is_read = true;
  adapter_.on_register_access(access);
}

void DataHazardPlugin::onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) {
  if (shutting_down())
    return;
  const DataHazardWavefrontState *state = wave_state(wf);
  if (state == nullptr || state->current.instruction_id == 0 || state->is_memory_op)
    return;

  RegisterAccessView access;
  access.instruction = state->current;
  access.register_class = RegisterClass::Scalar;
  access.physical_reg = logical_sgpr_base(*wf, physical_reg);
  access.size_bytes = kBytesPerDword;
  access.is_read = true;
  adapter_.on_register_access(access);
}

void DataHazardPlugin::onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
  if (shutting_down())
    return;
  for (amdgpu::Wavefront *wf : wavefronts) {
    if (wf != nullptr)
      adapter_.on_workgroup_barrier(make_wave_key(*wf));
  }
}

void DataHazardPlugin::emit_tensor_lds_write(const InstructionView &view,
                                             const amdgpu::Wavefront &wf) {
  MemoryRouteView route;
  route.instruction = view;
  route.writes_local_memory = true;
  route.local_write_wait = hazard_core::WaitCntType::TENSOR;
  route.local_address = 0;
  route.local_size_bytes = kBytesPerDword;
  route.exec_mask = wf.exec();
  route.is_tensor = true;
  adapter_.on_memory_route(route);
}

void DataHazardPlugin::emit_source_reads(const InstructionView &view, const Instruction &inst,
                                         bool is_memory_op) {
  auto register_key = [](const auto &ref) {
    return (static_cast<uint64_t>(ref.index) << 8) | static_cast<uint64_t>(ref.width) |
           (static_cast<uint64_t>(ref.cls) << 32);
  };

  std::unordered_set<uint64_t> seen;
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const auto *op = inst.src_operand(i);
    if (op == nullptr)
      continue;
    const auto ref = op->to_register_ref();
    if (!ref)
      continue;
    const auto reg_class = make_register_class(ref->cls);
    if (reg_class == RegisterClass::None)
      continue;
    // The address operand pair of a buffer store is not a data dependency.
    if (is_memory_op && inst.mnemonic() == "buffer_store_dword" &&
        reg_class == RegisterClass::Scalar && ref->width > 1)
      continue;
    // Vector loads name their destination as a source operand; treating it as
    // a read would manufacture a false WAR against the load itself.
    if (is_vector_load_mnemonic(inst.mnemonic()) && is_vector_register_class(reg_class))
      continue;
    if (!seen.insert(register_key(*ref)).second)
      continue;

    RegisterAccessView access;
    access.instruction = view;
    access.register_class = reg_class;
    access.physical_reg = ref->index;
    access.size_bytes = std::max<uint32_t>(1, ref->width) * kBytesPerDword;
    access.is_read = true;
    adapter_.on_register_access(access);
  }
}

void DataHazardPlugin::emit_destination_writes(const InstructionView &view,
                                               const Instruction &inst) {
  for (int i = 0; i < inst.num_dst_operands(); ++i) {
    const auto *op = inst.dst_operand(i);
    if (op == nullptr)
      continue;
    const auto ref = op->to_register_ref();
    if (!ref)
      continue;
    const auto reg_class = make_register_class(ref->cls);
    if (reg_class == RegisterClass::None)
      continue;

    RegisterAccessView access;
    access.instruction = view;
    access.register_class = reg_class;
    access.physical_reg = ref->index;
    access.size_bytes = std::max<uint32_t>(1, ref->width) * kBytesPerDword;
    access.is_write = true;
    adapter_.on_register_access(access);
  }
}

} // namespace rocjitsu::plugins::data_hazard
