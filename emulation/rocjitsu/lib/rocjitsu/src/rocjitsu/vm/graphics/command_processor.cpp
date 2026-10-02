// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/graphics/command_state.h"
#include "rocjitsu/vm/graphics/graphics_draw.h"
#include "rocjitsu/vm/graphics/pm4_clear_state.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <format>
#include <stdexcept>

namespace rocjitsu::amdgpu {

void CommandProcessor::draw_pm4(ComputeQueueRecord &queue, Pm4DispatchState &qs, uint32_t vertices,
                                std::vector<uint32_t> indices) {
  if (!vertices || !graphics_state(queue.commands).num_instances)
    return;
  if (cus_.empty())
    throw std::runtime_error("graphics draw requires a compute unit");
  auto draw = std::make_shared<GraphicsDraw>(queue.commands, cus_[0]->config().arch, vertices,
                                             std::move(indices));
  if (plugin_group_->empty() &&
      std::ranges::none_of(cus_, [](const auto *cu) { return cu->debug_active(); })) {
    if (auto access = snapshot_gpu_access(queue.address_space))
      draw->enable_vertex_batching(*access);
  }
  auto dp = draw->vertex_dispatch();
  graphics_state(queue.commands).draw = std::move(draw);
  dispatch_graphics_pm4(queue, qs, std::move(dp));
}

void CommandProcessor::dispatch_graphics_pm4(const ComputeQueueRecord &queue, Pm4DispatchState &qs,
                                             DispatchEntry dp) {
  if (dp.vgprs_per_wf > cus_[0]->vgpr_allocation_block_size())
    throw std::runtime_error("graphics launch exceeds available VGPRs");
  dp.kind = DispatchPacketKind::Kernel;
  dp.dispatch_id = allocate_dispatch_id();
  dp.process_id = queue.process_id;
  dp.queue_id = queue.queue_id;
  dp.sgprs_per_wf = cus_[0]->config().sgprs_per_wf;
  dp.pm4_abi = true;
  dp.pm4_failure = queue.commands.submissions.front().failure;
  dp.graphics_stage = queue.commands.graphics->draw;
  dp.address_space = queue.address_space;
  const auto &state = queue.commands;
  const uint32_t rsrc2 = state.sh_registers[state.graphics->draw->fragment_stage() ? 0xb : 0x8b];
  if (rsrc2 & 1) { // SPI_SHADER_PGM_RSRC2_PS/GS.SCRATCH_EN.
    // SPI_TMPRING_SIZE and SPI_GFX_SCRATCH_BASE share the compute descriptor
    // layout and granularity on the supported graphics targets (GFX11+).
    init_pm4_scratch(dp, state.context_registers[0x1ba], state.context_registers[0x1bb],
                     state.context_registers[0x1bc]);
  }
  flush_gpu_caches();
  util::Logger::cp("graphics dispatch pc=", std::hex, dp.kernel_entry_pc, std::dec,
                   " workgroups=", dp.total_wgs);
  KernelDispatchInfo info{};
  info.dispatch_id = dp.dispatch_id;
  info.entry_pc = dp.kernel_entry_pc;
  info.kernel_name =
      queue.commands.graphics->draw->fragment_stage() ? "PM4 fragment" : "PM4 vertex";
  info.code_target = cus_[0]->config().target;
  info.lds_size_bytes = dp.group_segment_fixed_size;
  info.wave_size = dp.kernel_wave_size;
  info.grid_size_x = dp.grid_size_x;
  info.workgroup_size_x = dp.kernel_wave_size;
  info.grid_size_y = info.grid_size_z = info.workgroup_size_y = info.workgroup_size_z = 1;
  info.workgroup_count = dp.total_wgs;
  info.wfs_per_workgroup = 1;
  info.sgprs_per_wf = dp.sgprs_per_wf;
  info.vgprs_per_wf = dp.vgprs_per_wf;
  plugin_group_->onAmdgpuDispatchPacketProcessed(info);
  ++total_dispatched_;
  qs.push_entry(std::move(dp));
}

std::vector<uint32_t> CommandProcessor::read_graphics_indices(ComputeCommandState &state,
                                                              const GpuVmAccess &memory,
                                                              uint64_t base, uint32_t available,
                                                              uint32_t count) {
  const auto *access = &memory;
  if (count > (1u << 20))
    throw std::runtime_error("unsupported graphics index count");
  const uint32_t type = state.uconfig_registers[0x243] & 3;
  if (type > 2)
    throw std::runtime_error("unsupported graphics index type");
  const uint32_t bytes = type == 0 ? 2 : type == 1 ? 4 : 1;
  const uint32_t valid = std::min(available, count);
  flush_gpu_caches();
  std::vector<uint8_t> data(valid * bytes);
  if (!data.empty() &&
      access->read(base, std::as_writable_bytes(std::span{data})) != VmAccessOutcome::Complete)
    throw std::runtime_error("graphics index read failed");
  std::vector<uint32_t> indices(count);
  for (uint32_t i = 0; i < valid; ++i)
    for (uint32_t b = 0; b < bytes; ++b)
      indices[i] |= uint32_t{data[i * bytes + b]} << (b * 8);
  return indices;
}

bool CommandProcessor::advance_graphics_pm4(ComputeQueueRecord &queue, Pm4DispatchState &qs,
                                            const GpuVmAccess &memory) {
  if (!queue.commands.graphics)
    return false;
  auto &state = queue.commands;
  const auto *access = &memory;
  if (graphics_state(state).draw) {
    flush_gpu_caches();
    CpuDispatchPool *raster_pool = nullptr;
    if (dispatch_threads_ > 1 && plugin_group_->empty()) {
      if (shared_dispatch_pool_) {
        raster_pool = shared_dispatch_pool_;
      } else {
        if (!local_dispatch_pool_ || local_dispatch_pool_->thread_count() < dispatch_threads_)
          local_dispatch_pool_ = std::make_unique<CpuDispatchPool>(dispatch_threads_);
        raster_pool = local_dispatch_pool_.get();
      }
    }
    const bool allow_ram_read_batching =
        plugin_group_->empty() &&
        std::ranges::none_of(cus_, [](const auto *cu) { return cu->debug_active(); });
    if (auto dp =
            graphics_state(state).draw->advance(*access, raster_pool, dispatch_threads_,
                                                allow_ram_read_batching, allow_ram_read_batching)) {
      dispatch_graphics_pm4(queue, qs, std::move(*dp));
      return true;
    }
    graphics_state(state).occlusion_samples += graphics_state(state).draw->occlusion_samples();
    graphics_state(state).draw.reset();
  }
  return false;
}

bool CommandProcessor::advance_indirect_graphics_pm4(ComputeQueueRecord &queue,
                                                     Pm4DispatchState &qs,
                                                     const GpuVmAccess &memory) {
  if (!queue.commands.graphics)
    return false;
  auto &state = queue.commands;
  const auto *access = &memory;
  if (graphics_state(state).indirect_draw) {
    auto &draw = *graphics_state(state).indirect_draw;
    if (draw.next == draw.count) {
      graphics_state(state).indirect_draw.reset();
      return true;
    }
    // Read each record after its predecessor retires, so shader writes to
    // subsequent indirect arguments observe the same command ordering.
    std::array<uint32_t, 5> arguments{};
    flush_gpu_caches();
    const uint64_t offset = uint64_t{draw.next} * draw.stride;
    if (draw.arguments > UINT64_MAX - offset ||
        access->read(draw.arguments + offset, std::as_writable_bytes(std::span{arguments})) !=
            VmAccessOutcome::Complete)
      throw std::runtime_error("PM4 indexed indirect arguments read failed");
    graphics_state(state).num_instances = arguments[1];
    state.sh_registers[draw.vertex_register] = arguments[3];
    state.sh_registers[draw.instance_register] = arguments[4];
    if (draw.first_index_register)
      state.sh_registers[*draw.first_index_register] = arguments[2];
    if (draw.draw_index_register)
      state.sh_registers[*draw.draw_index_register] = draw.next;
    ++draw.next;
    if (!arguments[0] || !arguments[1])
      return true;
    const uint32_t type = state.uconfig_registers[0x243] & 3;
    const uint32_t bytes = type == 0 ? 2 : type == 1 ? 4 : 1;
    const uint64_t first = uint64_t{arguments[2]} * bytes;
    if (graphics_state(state).index_base > UINT64_MAX - first)
      throw std::runtime_error("PM4 indexed indirect index address overflow");
    const uint32_t available = arguments[2] < graphics_state(state).index_buffer_size
                                   ? graphics_state(state).index_buffer_size - arguments[2]
                                   : 0;
    auto indices = read_graphics_indices(state, memory, graphics_state(state).index_base + first,
                                         available, arguments[0]);
    draw_pm4(queue, qs, arguments[0], std::move(indices));
    return true;
  }
  return false;
}

void CommandProcessor::reset_graphics_pm4(ComputeCommandState &state) {
  if (state.graphics) {
    state.graphics->draw.reset();
    state.graphics->indirect_draw.reset();
  }
}

bool CommandProcessor::execute_graphics_pm4(ComputeQueueRecord &queue, Pm4DispatchState &qs,
                                            const GpuVmAccess &memory, uint32_t opcode,
                                            std::vector<uint32_t> &words) {
  auto &state = queue.commands;
  const auto &submission = state.submissions.front();
  const auto *access = &memory;
  const auto require = [&](size_t size) {
    if (words.size() != size)
      throw std::runtime_error(std::format("PM4 opcode {:#x} expects {} payload words, got {}",
                                           opcode, size, words.size()));
  };
  const auto address = [&](size_t index) {
    return uint64_t{words[index]} | (uint64_t{words[index + 1]} << 32);
  };
  switch (static_cast<Pm4Opcode>(opcode)) {
  case Pm4Opcode::ContextControl:
  case Pm4Opcode::PfpSyncMe:
    if (!submission.graphics_engine)
      throw std::runtime_error("graphics state packet on compute engine");
    // The single CP retires preceding work before these packets.
    if (opcode == uint32_t(Pm4Opcode::ContextControl))
      require(2);
    else
      require(1);
    break;
  case Pm4Opcode::ClearState: {
    require(1);
    const uint32_t command = words[0];
    const bool reset = command == 0 || command == 3;
    const bool push = command == 1 || command == 3;
    // GFX12 has push/pop only. Reset values are qualified for GFX11.
    if (!submission.graphics_engine || command > 3 ||
        (reset && (cus_.empty() || (cus_[0]->config().arch != ROCJITSU_CODE_ARCH_RDNA3 &&
                                    cus_[0]->config().arch != ROCJITSU_CODE_ARCH_RDNA3_5))))
      throw std::runtime_error("unsupported CLEAR_STATE mode or engine");
    if (push) {
      if (graphics_state(state).saved_context_registers)
        throw std::runtime_error("nested CLEAR_STATE push is unsupported");
      graphics_state(state).saved_context_registers =
          std::make_unique<ComputeCommandState::ContextRegisters>(state.context_registers);
    }
    if (command == 2) {
      if (!graphics_state(state).saved_context_registers)
        throw std::runtime_error("CLEAR_STATE pop without a saved context");
      state.context_registers = *graphics_state(state).saved_context_registers;
      graphics_state(state).saved_context_registers.reset();
    } else if (reset) {
      reset_gfx11_context_registers(state.context_registers);
    }
    break;
  }
  case Pm4Opcode::SetPredication: {
    require(3);
    const uint32_t operation = (words[0] >> 16) & 7;
    // Boolean predicates use the common GFX9+ packet layout.
    // Query accumulation and its CONTINUE/HINT controls are not modeled.
    if (!submission.graphics_engine || (words[0] & ~0x70100u) ||
        (operation != 0 && operation != 3 && operation != 4))
      throw std::runtime_error("unsupported SET_PREDICATION control");
    state.predicate_pass = true;
    if (operation) {
      const size_t bytes = operation == 4 ? 4 : 8;
      if (address(1) % bytes)
        throw std::runtime_error("unaligned SET_PREDICATION address");
      flush_gpu_caches();
      uint64_t value = 0;
      if (access->read(address(1), {reinterpret_cast<std::byte *>(&value), bytes}) !=
          VmAccessOutcome::Complete)
        throw std::runtime_error("PM4 SET_PREDICATION read failed");
      state.predicate_pass = (value != 0) == bool(words[0] & (1u << 8));
    }
    break;
  }
  case Pm4Opcode::SetBase:
    require(3);
    if (!submission.graphics_engine || words[0] != 1)
      throw std::runtime_error("unsupported SET_BASE index");
    state.indirect_base = address(1);
    break;
  case Pm4Opcode::ContextRegRmw:
    require(3);
    if (!submission.graphics_engine || words[0] >= state.context_registers.size())
      throw std::runtime_error("invalid CONTEXT_REG_RMW register or engine");
    state.context_registers[words[0]] =
        (state.context_registers[words[0]] & ~words[1]) | (words[2] & words[1]);
    break;
  case Pm4Opcode::EventWriteZpass:
    require(2);
    if (!submission.graphics_engine || cus_.empty() ||
        (cus_[0]->config().arch != ROCJITSU_CODE_ARCH_RDNA3 &&
         cus_[0]->config().arch != ROCJITSU_CODE_ARCH_RDNA3_5 &&
         cus_[0]->config().arch != ROCJITSU_CODE_ARCH_RDNA4))
      throw std::runtime_error("unsupported PM4 EVENT_WRITE_ZPASS engine or architecture");
    // GFX11+ also encodes PIXEL_PIPE_STAT_DUMP with just its destination.
    words.insert(words.begin(), 57u | (1u << 8));
    [[fallthrough]];
  case Pm4Opcode::EventWrite: {
    const uint32_t event = words[0] & 0x3f;
    const uint32_t event_index = (words[0] >> 8) & 15;
    if (event == 15 && event_index >= 8 && event_index <= 11) {
      require(3);
      const bool supports_streamout_query =
          !cus_.empty() && (cus_[0]->config().arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                            cus_[0]->config().arch == ROCJITSU_CODE_ARCH_RDNA3_5);
      if (!submission.graphics_engine || !supports_streamout_query ||
          words[0] != (15 | (event_index << 8)) || (words[1] & 7))
        throw std::runtime_error("unsupported PM4 streamout query event");
      flush_gpu_caches();
      auto values = graphics_state(state).gs_registers->streamout_stats(event_index - 8);
      for (auto &value : values)
        value |= uint64_t{1} << 63;
      if (access->write(address(1), std::as_bytes(std::span{values})) != VmAccessOutcome::Complete)
        throw std::runtime_error("PM4 streamout query write failed");
      break;
    }
    if (submission.graphics_engine && event == 57) {
      // PIXEL_PIPE_STAT_DUMP writes enabled occlusion counter instances.
      require(3);
      if (words[0] != (57u | (1u << 8)) || (words[1] & 7) ||
          graphics_state(state).unsupported_pixel_counter_mode ||
          !graphics_state(state).pixel_counter_instances)
        throw std::runtime_error("unsupported PM4 occlusion query event");
      flush_gpu_caches();
      const uint64_t mask = graphics_state(state).pixel_counter_instances;
      if (address(1) > UINT64_MAX - 16 * std::bit_width(mask))
        throw std::runtime_error("PM4 occlusion query address overflow");
      for (int instance = 0; instance < std::bit_width(mask); ++instance) {
        if (!(mask & (uint64_t{1} << instance)))
          continue;
        // The rasterizer has one logical sample counter. Publish its sum
        // in the first enabled instance and valid zeroes in the others.
        const uint64_t value =
            (uint64_t{1} << 63) |
            (instance == std::countr_zero(mask) ? graphics_state(state).occlusion_samples : 0);
        if (access->write(address(1) + 16 * instance, std::as_bytes(std::span{&value, 1})) !=
            VmAccessOutcome::Complete)
          throw std::runtime_error("PM4 occlusion query write failed");
      }
      break;
    }
    if (submission.graphics_engine && event == 56) {
      // PIXEL_PIPE_STAT_CONTROL configures graphics counters, not a memory write.
      require(3);
      // Mesa's ordinary preamble selects counter 0 and a 128-bit stride.
      // Instance-enable bits do not mean an occlusion query is active.
      graphics_state(state).unsupported_pixel_counter_mode =
          words[0] != (56u | (1u << 8)) || (words[1] & 0x7ffu) != (2u << 9);
      graphics_state(state).pixel_counter_instances = (uint64_t{words[2]} << 21) | (words[1] >> 11);
      break;
    }
    require(1);
    // Counter START/STOP events also configure compute counters. SQ_NON_EVENT
    // drains graphics pipeline messages without writing a sampled result.
    if (event != 7 && event != 23 && event != 24 && event != 25 && event != 26 &&
        !(submission.graphics_engine && (event == 15 || event == 16 || event == 36 || event == 38 ||
                                         event == 44 || event == 46 || event == 49)))
      throw std::runtime_error(std::format("unsupported PM4 EVENT_WRITE event {}", event));
    if (event == 23 || event == 24)
      graphics_state(state).performance_counters_active = event == 23;
    flush_gpu_caches();
    break;
  }
  case Pm4Opcode::StreamoutStatsQuery: {
    require(5);
    if (!submission.graphics_engine || cus_.empty() ||
        cus_[0]->config().arch != ROCJITSU_CODE_ARCH_RDNA4 || (words[0] & 7) || (words[3] & 7) ||
        words[2] > 3 || address(0) > UINT64_MAX - 79)
      throw std::runtime_error("unsupported PM4 streamout statistics query");
    // GFX12 shaders maintain four needed/written counter pairs in ordinary memory,
    // following four dwords of streamout buffer offsets.
    flush_gpu_caches();
    std::array<uint64_t, 2> values{};
    if (access->read(address(0) + 16 + 16 * words[2], std::as_writable_bytes(std::span{values})) !=
        VmAccessOutcome::Complete)
      throw std::runtime_error("PM4 streamout query read failed");
    for (auto &value : values)
      value |= uint64_t{1} << 63;
    if (access->write(address(3), std::as_bytes(std::span{values})) != VmAccessOutcome::Complete)
      throw std::runtime_error("PM4 streamout query write failed");
    break;
  }
  case Pm4Opcode::NumInstances:
    require(1);
    if (!submission.graphics_engine)
      throw std::runtime_error("NUM_INSTANCES on compute engine");
    graphics_state(state).num_instances = words[0];
    break;
  case Pm4Opcode::DrawIndexAuto:
    require(2);
    if (!submission.graphics_engine || words[1] != 2)
      throw std::runtime_error("unsupported DRAW_INDEX_AUTO initiator");
    draw_pm4(queue, qs, words[0]);
    break;
  case Pm4Opcode::IndexBase:
    require(2);
    if (!submission.graphics_engine || (words[0] & 1))
      throw std::runtime_error("unsupported INDEX_BASE packet");
    graphics_state(state).index_base = address(0);
    break;
  case Pm4Opcode::IndexBufferSize:
    require(1);
    if (!submission.graphics_engine)
      throw std::runtime_error("INDEX_BUFFER_SIZE on compute engine");
    graphics_state(state).index_buffer_size = words[0];
    break;
  case Pm4Opcode::DrawIndexIndirect:
  case Pm4Opcode::DrawIndexIndirectMulti: {
    const bool multi = opcode == uint32_t(Pm4Opcode::DrawIndexIndirectMulti);
    require(multi ? 9 : 4);
    if (!submission.graphics_engine || words.back() || (words[0] & 3) ||
        (words[2] & (multi ? 0xffff0000u : 0xefff0000u)) ||
        (multi && ((words[3] & ~0xf000ffffu) || (words[7] & 3))))
      throw std::runtime_error("unsupported indexed indirect draw packet");
    GraphicsCommandState::IndirectDraw draw;
    if (state.indirect_base > UINT64_MAX - words[0])
      throw std::runtime_error("PM4 indexed indirect argument address overflow");
    draw.arguments = state.indirect_base + words[0];
    draw.vertex_register = words[1] & 0xffff;
    draw.instance_register = words[2] & 0xffff;
    draw.count = multi ? words[4] : 1;
    draw.stride = multi ? words[7] : 20;
    if ((multi ? words[3] : words[2]) & (1u << 28))
      draw.first_index_register = words[1] >> 16;
    else if (words[1] >> 16)
      throw std::runtime_error("unsupported indexed indirect first-index register");
    if (multi && (words[3] & (1u << 31)))
      draw.draw_index_register = words[3] & 0xffff;
    if (draw.vertex_register >= state.sh_registers.size() ||
        draw.instance_register >= state.sh_registers.size() ||
        (draw.first_index_register && *draw.first_index_register >= state.sh_registers.size()) ||
        (draw.draw_index_register && *draw.draw_index_register >= state.sh_registers.size()))
      throw std::runtime_error("indexed indirect draw register outside SH aperture");
    if (multi && (words[3] & (1u << 30))) {
      uint32_t count = 0;
      flush_gpu_caches();
      if ((words[5] & 3) ||
          access->read(address(5), std::as_writable_bytes(std::span{&count, 1})) !=
              VmAccessOutcome::Complete)
        throw std::runtime_error("PM4 indexed indirect count read failed");
      draw.count = std::min(draw.count, count);
    }
    if (draw.count > (1u << 20))
      throw std::runtime_error("indexed indirect draw count exceeds simulator limit");
    graphics_state(state).indirect_draw = draw;
    break;
  }
  case Pm4Opcode::DrawIndex2: {
    require(5);
    if (!submission.graphics_engine || words[4])
      throw std::runtime_error("unsupported DRAW_INDEX_2 initiator");
    auto indices = read_graphics_indices(state, memory, address(1), words[0], words[3]);
    draw_pm4(queue, qs, words[3], std::move(indices));
    break;
  }
  default:
    return false;
  }
  return true;
}

} // namespace rocjitsu::amdgpu
