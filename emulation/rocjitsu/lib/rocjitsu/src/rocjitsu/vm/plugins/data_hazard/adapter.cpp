// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/data_hazard/adapter.h"

#include "detail/waitcnt_decode.h"

#include <algorithm>
#include <charconv>
#include <system_error>

namespace rocjitsu::plugins::data_hazard {

using hazard_core::BarrierEvent;
using hazard_core::BarrierKind;
using hazard_core::DataHazardSimulatorApi;
using hazard_core::EntityId;
using hazard_core::ExecutionKey;
using hazard_core::InstructionDescriptor;
using hazard_core::InstructionEvent;
using hazard_core::RegisterKind;
using hazard_core::ResourceAccessEvent;
using hazard_core::ResourceKind;
using hazard_core::WaitAction;
using hazard_core::WaitCntType;

namespace {

constexpr EntityId kGeneratedIdStartValue = 1ull << 63;

bool is_vector_class(RegisterClass reg_class) {
  return reg_class == RegisterClass::Vector || reg_class == RegisterClass::AccumVector;
}

RegisterClass default_register_class(ResourceKind resource_kind, RegisterClass reg_class) {
  if (reg_class != RegisterClass::None)
    return reg_class;
  if (resource_kind == ResourceKind::ScalarRegister)
    return RegisterClass::Scalar;
  if (resource_kind == ResourceKind::GlobalMemory || resource_kind == ResourceKind::ScratchMemory ||
      resource_kind == ResourceKind::LocalMemory)
    return RegisterClass::Vector;
  return RegisterClass::None;
}

uint32_t nonzero_size_or_dword(uint32_t size_bytes) {
  return size_bytes == 0 ? hazard_core::BYTES_PER_DWORD : size_bytes;
}

void add_counter(WaitAction &action, WaitCntType kind, uint32_t keep_count) {
  action.counters.push_back({kind, keep_count});
}

/// The first run of digits in @p text, which is how a wait that names no field
/// states its count.
uint32_t first_decimal(std::string_view text, uint32_t fallback = 0) {
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] < '0' || text[i] > '9')
      continue;
    size_t end = i;
    while (end < text.size() && text[end] >= '0' && text[end] <= '9')
      ++end;
    uint32_t value = fallback;
    const auto result = std::from_chars(text.data() + i, text.data() + end, value);
    return result.ec == std::errc{} ? value : fallback;
  }
  return fallback;
}

/// The count @p field names in @p text, as in the `lgkmcnt(` of
/// `vmcnt(1) expcnt(0) lgkmcnt(3)`.
uint32_t parse_waitcnt_field(std::string_view text, std::string_view field, uint32_t fallback) {
  const size_t field_pos = text.find(field);
  if (field_pos == std::string_view::npos)
    return fallback;
  const size_t begin = field_pos + field.size();
  const size_t end = text.find(')', begin);
  if (end == std::string_view::npos || end <= begin)
    return fallback;

  uint32_t value = fallback;
  const auto result = std::from_chars(text.data() + begin, text.data() + end, value);
  return result.ec == std::errc{} ? value : fallback;
}

constexpr std::string_view kDscntField = "dscnt(";

/// The count pair a combined `s_wait_*cnt_dscnt` expresses. An assembler
/// listing names both fields, but the emulator's own decoder prints the bare
/// immediate the hardware encodes, which packs the memory counter above dscnt.
/// Read whole, that immediate is neither count: it leaves the memory counter
/// too large to drain anything and dscnt at zero, draining everything.
WaitInfo decode_paired_wait(WaitKind kind, std::string_view text, std::string_view memory_field) {
  WaitInfo wait;
  wait.kind = kind;

  if (text.find(memory_field) != std::string_view::npos ||
      text.find(kDscntField) != std::string_view::npos) {
    wait.count = parse_waitcnt_field(text, memory_field, 0);
    wait.paired_count = parse_waitcnt_field(text, kDscntField, 0);
    return wait;
  }

  const auto fields = data_hazard_waitcnt::decode_split_waitcnt(first_decimal(text));
  wait.count = fields.primary;
  wait.paired_count = fields.dscnt;
  return wait;
}

ResourceAccessEvent make_base_resource_event(const InstructionDescriptor &instruction,
                                             uint32_t size_bytes, bool is_atomic,
                                             uint64_t exec_mask) {
  ResourceAccessEvent event;
  event.instruction = instruction;
  event.size_bytes = nonzero_size_or_dword(size_bytes);
  event.is_atomic = is_atomic;
  event.exec_mask = exec_mask;
  return event;
}

bool lane_is_active(uint64_t exec_mask, size_t lane) {
  if (exec_mask == 0)
    return true;
  if (lane >= 64)
    return false;
  return (exec_mask & (1ull << lane)) != 0;
}

template <typename Emit>
void emit_per_lane_or_scalar(const std::vector<uint64_t> &per_lane_addresses,
                             uint64_t scalar_address, uint64_t exec_mask, Emit emit) {
  if (per_lane_addresses.empty()) {
    emit(scalar_address);
    return;
  }

  for (size_t lane = 0; lane < per_lane_addresses.size(); ++lane) {
    if (lane_is_active(exec_mask, lane))
      emit(per_lane_addresses[lane]);
  }
}

} // namespace

WaitAction make_wait_action(const WaitInfo &wait) {
  WaitAction action;
  action.is_wait_instruction = wait.kind != WaitKind::None;

  switch (wait.kind) {
  case WaitKind::None:
    break;
  case WaitKind::Waitcnt:
    // Drains the stores as well on gfx9 and CDNA, where they are outstanding
    // on this same counter. gfx10 and gfx11 kept the s_waitcnt mnemonic but
    // count stores on vscnt, drained by the separate s_waitcnt_vscnt, and
    // their stores are tracked against that counter instead.
    add_counter(action, WaitCntType::VMEM, wait.count);
    add_counter(action, WaitCntType::LDS, wait.paired_count);
    add_counter(action, WaitCntType::SMEM, wait.paired_count);
    break;
  case WaitKind::WaitLoadcnt:
    add_counter(action, WaitCntType::VMEM, wait.count);
    break;
  case WaitKind::WaitStorecnt:
  case WaitKind::WaitVscnt:
    add_counter(action, WaitCntType::STORE, wait.count);
    break;
  case WaitKind::WaitKmcnt:
    add_counter(action, WaitCntType::SMEM, wait.count);
    break;
  case WaitKind::WaitDscnt:
    add_counter(action, WaitCntType::LDS, wait.count);
    break;
  case WaitKind::WaitLoadcntDscnt:
    add_counter(action, WaitCntType::VMEM, wait.count);
    add_counter(action, WaitCntType::LDS, wait.paired_count);
    break;
  case WaitKind::WaitStorecntDscnt:
    add_counter(action, WaitCntType::STORE, wait.count);
    add_counter(action, WaitCntType::LDS, wait.paired_count);
    break;
  case WaitKind::WaitIdle:
    action.waits_for_idle = true;
    break;
  case WaitKind::WaitTensorcnt:
    add_counter(action, WaitCntType::TENSOR, wait.count);
    break;
  case WaitKind::BarrierWait:
    // Deliberately not a completed workgroup barrier: this runs before the
    // wave stalls, while other waves may still be issuing pre-barrier
    // accesses. The epoch is closed from onAmdgpuBarrierResolved, which runs
    // once every wave has arrived.
    break;
  case WaitKind::AddressTranslation:
    action.is_address_translation = true;
    break;
  }

  return action;
}

WaitInfo make_wait_info(WaitKind kind, std::string_view operand_text) {
  WaitInfo wait;
  wait.kind = kind;

  switch (kind) {
  case WaitKind::None:
    break;
  case WaitKind::Waitcnt:
    wait.count = parse_waitcnt_field(operand_text, "vmcnt(", 0);
    wait.paired_count = parse_waitcnt_field(operand_text, "lgkmcnt(", 0);
    break;
  case WaitKind::WaitLoadcntDscnt:
    return decode_paired_wait(kind, operand_text, "loadcnt(");
  case WaitKind::WaitStorecntDscnt:
    return decode_paired_wait(kind, operand_text, "storecnt(");
  default:
    wait.count = first_decimal(operand_text);
    break;
  }
  return wait;
}

WaitKind make_wait_kind(std::string_view mnemonic) {
  if (mnemonic == "s_waitcnt")
    return WaitKind::Waitcnt;
  if (mnemonic == "s_wait_loadcnt")
    return WaitKind::WaitLoadcnt;
  if (mnemonic == "s_wait_storecnt")
    return WaitKind::WaitStorecnt;
  if (mnemonic == "s_wait_kmcnt")
    return WaitKind::WaitKmcnt;
  if (mnemonic == "s_wait_dscnt")
    return WaitKind::WaitDscnt;
  if (mnemonic == "s_wait_loadcnt_dscnt")
    return WaitKind::WaitLoadcntDscnt;
  if (mnemonic == "s_wait_storecnt_dscnt")
    return WaitKind::WaitStorecntDscnt;
  if (mnemonic == "s_wait_idle")
    return WaitKind::WaitIdle;
  if (mnemonic == "s_waitcnt_vscnt")
    return WaitKind::WaitVscnt;
  if (mnemonic == "s_wait_tensorcnt")
    return WaitKind::WaitTensorcnt;
  if (mnemonic == "s_barrier_wait")
    return WaitKind::BarrierWait;
  if (mnemonic == "s_wait_xcnt")
    return WaitKind::AddressTranslation;
  return WaitKind::None;
}

RegisterKind make_register_kind(RegisterClass reg_class) {
  switch (reg_class) {
  case RegisterClass::Scalar:
    return RegisterKind::Scalar;
  case RegisterClass::Vector:
    return RegisterKind::Vector;
  case RegisterClass::AccumVector:
    return RegisterKind::AccumVector;
  case RegisterClass::None:
    break;
  }
  return RegisterKind::None;
}

InstructionDescriptor make_instruction_descriptor(const InstructionView &instruction,
                                                  EntityId fallback_id) {
  InstructionDescriptor descriptor;
  descriptor.instruction_id =
      instruction.instruction_id != 0 ? instruction.instruction_id : fallback_id;
  descriptor.execution = instruction.execution;
  descriptor.pc = instruction.pc;
  descriptor.raw_isa = instruction.raw_isa;
  return descriptor;
}

DataHazardAdapter::DataHazardAdapter(DataHazardSimulatorApi &engine) : api_(engine) {}

void DataHazardAdapter::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  next_instruction_id_ = kGeneratedIdStartValue;
  current_instruction_.clear();
}

void DataHazardAdapter::on_dispatch_begin(EntityId dispatch_id) {
  erase_dispatch(dispatch_id);
  api_.on_dispatch_begin(dispatch_id);
}

void DataHazardAdapter::on_dispatch_end(EntityId dispatch_id) {
  erase_dispatch(dispatch_id);
  api_.on_dispatch_end(dispatch_id);
}

void DataHazardAdapter::on_workgroup_begin(const ExecutionKey &workgroup) {
  api_.on_workgroup_begin(workgroup.dispatch_id, workgroup.cluster_id, workgroup.workgroup_id);
}

void DataHazardAdapter::on_workgroup_end(const ExecutionKey &workgroup) {
  api_.on_workgroup_end(workgroup.dispatch_id, workgroup.cluster_id, workgroup.workgroup_id);
}

void DataHazardAdapter::on_wave_begin(const ExecutionKey &wave) { api_.on_wave_begin(wave); }

void DataHazardAdapter::on_wave_end(const ExecutionKey &wave) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_instruction_.erase(wave);
  }
  api_.on_wave_end(wave);
}

void DataHazardAdapter::on_instruction(const InstructionView &instruction) {
  InstructionEvent event;
  event.instruction = resolve_instruction(instruction, true);
  event.wait_action = make_wait_action(instruction.wait);
  api_.on_instruction(event);
}

void DataHazardAdapter::on_register_access(const RegisterAccessView &access) {
  const InstructionDescriptor instruction = current_or_resolved_instruction(access.instruction);

  ResourceAccessEvent event = make_base_resource_event(instruction, access.size_bytes, false, 0);
  event.register_kind = make_register_kind(access.register_class);
  event.resource_index = access.physical_reg;
  event.is_read = access.is_read;
  event.is_write = access.is_write;

  if (access.register_class == RegisterClass::AccumVector) {
    event.resource_kind = ResourceKind::AccumVectorRegister;
  } else if (is_vector_class(access.register_class)) {
    event.resource_kind = ResourceKind::VectorRegister;
  } else if (access.register_class == RegisterClass::Scalar) {
    event.resource_kind = ResourceKind::ScalarRegister;
  } else {
    return;
  }

  api_.on_resource_access(event);
}

void DataHazardAdapter::on_memory_route(const MemoryRouteView &route) {
  const InstructionDescriptor instruction = current_or_resolved_instruction(route.instruction);
  const RegisterClass reg_class = default_register_class(route.resource_kind, route.register_class);
  const RegisterKind reg_kind = make_register_kind(reg_class);
  const uint32_t size_bytes = nonzero_size_or_dword(route.size_bytes);
  const uint32_t local_size_bytes = nonzero_size_or_dword(
      route.local_size_bytes == 0 ? route.size_bytes : route.local_size_bytes);

  if (route.resource_kind == ResourceKind::GlobalMemory ||
      route.resource_kind == ResourceKind::ScratchMemory) {
    // A store holds a slot of route.store_wait until it completes even though
    // it leaves no register pending, and where that counter is the one loads
    // use, the slot is what holds an older load back. Reported once for the
    // instruction rather than per lane, because the counter tracks issued
    // instructions and an all-lanes-off store still occupies its slot. An
    // atomic that returns a value is left to its destination register, which
    // already takes a slot of the load counter.
    if (route.is_store) {
      ResourceAccessEvent slot =
          make_base_resource_event(instruction, size_bytes, route.is_atomic, route.exec_mask);
      slot.resource_kind = route.resource_kind;
      slot.hazards.memory_op_wait = route.store_wait;
      api_.on_resource_access(slot);
    }

    emit_per_lane_or_scalar(
        route.per_lane_addresses, route.address, route.exec_mask, [&](uint64_t address) {
          ResourceAccessEvent memory =
              make_base_resource_event(instruction, size_bytes, route.is_atomic, route.exec_mask);
          memory.resource_kind = route.resource_kind;
          memory.address = address;
          memory.is_read = route.is_load || route.is_atomic;
          memory.is_write = route.is_store || route.is_atomic;
          api_.on_resource_access(memory);
        });
  }

  if (route.resource_kind == ResourceKind::LocalMemory) {
    emit_per_lane_or_scalar(
        route.per_lane_addresses, route.address, route.exec_mask, [&](uint64_t address) {
          ResourceAccessEvent memory =
              make_base_resource_event(instruction, size_bytes, route.is_atomic, route.exec_mask);
          memory.resource_kind = ResourceKind::LocalMemory;
          memory.address = address;
          memory.is_read = route.is_load || route.is_atomic;
          memory.is_write = route.is_store || route.is_tensor || route.is_atomic;
          if (memory.is_write)
            memory.hazards.write_wait = route.is_tensor ? WaitCntType::TENSOR : WaitCntType::LDS;
          api_.on_resource_access(memory);
        });
  }

  if (route.writes_local_memory && route.resource_kind != ResourceKind::LocalMemory) {
    emit_per_lane_or_scalar(route.per_lane_local_addresses, route.local_address, route.exec_mask,
                            [&](uint64_t address) {
                              ResourceAccessEvent local = make_base_resource_event(
                                  instruction, local_size_bytes, route.is_atomic, route.exec_mask);
                              local.resource_kind = ResourceKind::LocalMemory;
                              local.address = address;
                              local.is_write = true;
                              local.hazards.write_wait = route.local_write_wait != WaitCntType::NONE
                                                             ? route.local_write_wait
                                                             : WaitCntType::VMEM;
                              api_.on_resource_access(local);
                            });
  }

  if (route.is_load && !route.writes_local_memory && reg_kind != RegisterKind::None) {
    ResourceAccessEvent dest =
        make_base_resource_event(instruction, size_bytes, route.is_atomic, route.exec_mask);
    dest.register_kind = reg_kind;
    dest.resource_index = route.register_base;
    dest.is_write = true;

    if (reg_class == RegisterClass::Scalar) {
      dest.resource_kind = ResourceKind::ScalarRegister;
      dest.hazards.write_wait = WaitCntType::SMEM;
    } else if (reg_class == RegisterClass::AccumVector) {
      dest.resource_kind = ResourceKind::AccumVectorRegister;
      if (route.resource_kind == ResourceKind::LocalMemory) {
        dest.hazards.write_wait = WaitCntType::LDS;
      } else {
        dest.hazards.write_wait = WaitCntType::VMEM;
        dest.hazards.write_also_waits_lds = route.is_flat;
      }
    } else if (reg_class == RegisterClass::Vector) {
      dest.resource_kind = ResourceKind::VectorRegister;
      if (route.resource_kind == ResourceKind::LocalMemory) {
        dest.hazards.write_wait = WaitCntType::LDS;
      } else {
        dest.hazards.write_wait = WaitCntType::VMEM;
        dest.hazards.write_also_waits_lds = route.is_flat;
      }
    }

    if (dest.resource_kind != ResourceKind::Unknown)
      api_.on_resource_access(dest);
  }

  // VMEM store data sources are captured when the store issues. Unlike
  // async destination writes, they do not need to remain live until
  // s_wait_storecnt drains, so do not track them as pending WAR reads.
}

void DataHazardAdapter::on_workgroup_barrier(const ExecutionKey &wave) {
  BarrierEvent event;
  event.wave = wave;
  event.kind = BarrierKind::Workgroup;
  api_.on_barrier(event);
}

void DataHazardAdapter::on_local_memory_atomic_barrier(const ExecutionKey &wave, bool async) {
  BarrierEvent event;
  event.wave = wave;
  event.kind = async ? BarrierKind::LocalMemoryAtomicAsync : BarrierKind::LocalMemoryAtomic;
  api_.on_barrier(event);
}

void DataHazardAdapter::on_shutdown() { api_.on_shutdown(); }

InstructionDescriptor DataHazardAdapter::resolve_instruction(const InstructionView &instruction,
                                                             bool make_current) {
  std::lock_guard<std::mutex> lock(mutex_);

  const ExecutionKey &key = instruction.execution;
  EntityId id = instruction.instruction_id;
  if (id == 0)
    id = next_instruction_id_++;

  InstructionDescriptor descriptor = make_instruction_descriptor(instruction, id);
  if (make_current)
    current_instruction_[key] = descriptor;
  return descriptor;
}

InstructionDescriptor
DataHazardAdapter::current_or_resolved_instruction(const InstructionView &instruction) {
  std::lock_guard<std::mutex> lock(mutex_);

  const ExecutionKey &key = instruction.execution;
  if (instruction.instruction_id == 0 && instruction.pc == 0 &&
      std::all_of(instruction.raw_isa.begin(), instruction.raw_isa.end(),
                  [](uint32_t word) { return word == 0; })) {
    if (auto it = current_instruction_.find(key); it != current_instruction_.end())
      return it->second;
  }

  EntityId id = instruction.instruction_id;
  if (id == 0) {
    if (auto it = current_instruction_.find(key); it != current_instruction_.end())
      id = it->second.instruction_id;
    else
      id = next_instruction_id_++;
  }

  InstructionDescriptor descriptor = make_instruction_descriptor(instruction, id);
  current_instruction_[key] = descriptor;
  return descriptor;
}

void DataHazardAdapter::erase_dispatch(EntityId dispatch_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = current_instruction_.begin(); it != current_instruction_.end();) {
    if (it->first.dispatch_id == dispatch_id)
      it = current_instruction_.erase(it);
    else
      ++it;
  }
}

} // namespace rocjitsu::plugins::data_hazard
