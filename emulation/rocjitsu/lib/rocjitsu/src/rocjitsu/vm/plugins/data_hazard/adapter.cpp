// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/data_hazard/adapter.h"

#include "detail/waitcnt_decode.h"

#include <algorithm>

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
using hazard_core::SemaphoreEvent;
using hazard_core::SemaphoreKind;
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
  case WaitKind::Waitcnt: {
    const auto fields = data_hazard_waitcnt::decode_legacy_waitcnt(
        wait.immediate, data_hazard_waitcnt::LEGACY_LGKMCNT_4BIT_MASK);
    add_counter(action, WaitCntType::VMEM, fields.vmcnt);
    add_counter(action, WaitCntType::STORE, fields.vmcnt);
    add_counter(action, WaitCntType::LDS, fields.lgkmcnt);
    add_counter(action, WaitCntType::SMEM, fields.lgkmcnt);
    break;
  }
  case WaitKind::WaitLoadcnt:
    add_counter(action, WaitCntType::VMEM, wait.immediate);
    break;
  case WaitKind::WaitStorecnt:
  case WaitKind::WaitVscnt:
    add_counter(action, WaitCntType::STORE, wait.immediate);
    break;
  case WaitKind::WaitKmcnt:
    add_counter(action, WaitCntType::SMEM, wait.immediate);
    break;
  case WaitKind::WaitDscnt:
    add_counter(action, WaitCntType::LDS, wait.immediate);
    break;
  case WaitKind::WaitLoadcntDscnt: {
    const auto fields = data_hazard_waitcnt::decode_split_waitcnt(wait.immediate);
    add_counter(action, WaitCntType::VMEM, fields.primary);
    add_counter(action, WaitCntType::LDS, fields.dscnt);
    break;
  }
  case WaitKind::WaitStorecntDscnt: {
    const auto fields = data_hazard_waitcnt::decode_split_waitcnt(wait.immediate);
    add_counter(action, WaitCntType::STORE, fields.primary);
    add_counter(action, WaitCntType::LDS, fields.dscnt);
    break;
  }
  case WaitKind::WaitIdle:
    action.waits_for_idle = true;
    break;
  case WaitKind::WaitTensorcnt:
    add_counter(action, WaitCntType::TENSOR, wait.immediate);
    break;
  case WaitKind::BarrierWait:
    // Deliberately not a completed workgroup barrier: this runs before the
    // wave stalls, while other waves may still be issuing pre-barrier
    // accesses. The epoch is closed from onAmdgpuBarrierResolved, which runs
    // once every wave has arrived.
    break;
  case WaitKind::SemaphoreWait:
    // Unlike a barrier wait this can be classified before the wave stalls: the
    // epoch it closes holds only the accesses of its own wavegroup, and the
    // signal that releases the wait was issued before the wait was reached.
    action.is_wavegroup_semaphore_wait = true;
    break;
  case WaitKind::AddressTranslation:
    action.is_address_translation = true;
    break;
  }

  return action;
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
  if (mnemonic == "s_sema_wait")
    return WaitKind::SemaphoreWait;
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

void DataHazardAdapter::on_wavegroup_begin(const ExecutionKey &wavegroup) {
  api_.on_wavegroup_begin(wavegroup.dispatch_id, wavegroup.cluster_id, wavegroup.workgroup_id,
                          wavegroup.wavegroup_id);
}

void DataHazardAdapter::on_wavegroup_end(const ExecutionKey &wavegroup) {
  api_.on_wavegroup_end(wavegroup.dispatch_id, wavegroup.cluster_id, wavegroup.workgroup_id,
                        wavegroup.wavegroup_id);
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

void DataHazardAdapter::on_wavegroup_semaphore_signal(const ExecutionKey &wave) {
  SemaphoreEvent event;
  event.wave = wave;
  event.wavegroup_id = wave.wavegroup_id;
  event.kind = SemaphoreKind::Signal;
  api_.on_semaphore(event);
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
