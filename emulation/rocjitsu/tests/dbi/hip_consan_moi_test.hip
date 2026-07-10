// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_consan_moi_test.hip
/// @brief Real-HIP smoke tests for ConSan MOI DBI report-buffer instrumentation.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

#define RJ_CONSAN_MOI_RECORD_NOP_CAVE                                                              \
  ".rept 128\n\t"                                                                                  \
  "s_nop 0\n\t"                                                                                    \
  ".endr\n\t"

namespace {

constexpr uint32_t kMoiReportMagic = 0x494f4d43u;
constexpr uint32_t kMoiReportAbiVersion = 1;
constexpr uint32_t kMoiAccessKindRead = 1;
constexpr uint32_t kMoiAccessKindWrite = 2;
constexpr uint32_t kMoiAtomicKindRelease = 1;
constexpr uint32_t kMoiAtomicKindAcquire = 2;
constexpr uint32_t kRdna4ScopeDevice = 2;
constexpr uint32_t kRdna4AtomicNoReturnTh = 0;
constexpr uint32_t kRdna4AtomicReturnTh = 1;

struct alignas(8) MoiReportHeader {
  uint32_t magic = kMoiReportMagic;
  uint32_t abi_version = kMoiReportAbiVersion;
  uint32_t header_size = 0;
  uint32_t flags = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t access_record_capacity = 0;
  uint32_t access_record_count = 0;
  uint32_t barrier_record_capacity = 0;
  uint32_t barrier_record_count = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t atomic_record_count = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t diagnostic_count = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  uint32_t event_counter = 0;
  uint32_t reserved = 0;
};

struct alignas(8) MoiAccessRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t wave_id = 0;
  uint64_t lane_mask = 0;
  uint32_t instruction_offset = 0;
  uint32_t access_kind = 0;
  uint32_t lds_byte_offset = 0;
  uint32_t lds_byte_count = 0;
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
  uint32_t epoch = 0;
  uint32_t event_index = 0;
};

struct alignas(8) MoiBarrierRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t wave_id = 0;
  uint64_t lane_mask = 0;
  uint32_t instruction_offset = 0;
  uint32_t event_index = 0;
};

struct alignas(8) MoiAtomicRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t owner_id = 0;
  uint64_t atomic_address = 0;
  uint32_t instruction_offset = 0;
  uint32_t event_index = 0;
  uint32_t epoch = 0;
  uint32_t kind = 0;
  uint32_t scope = 0;
  uint32_t semantics = 0;
};

struct MoiReplayDiagnostic {
  bool conflict = false;
  uint32_t first_owner_id = 0;
  uint32_t second_owner_id = 0;
  uint32_t first_access_kind = 0;
  uint32_t second_access_kind = 0;
  uint32_t first_instruction_offset = 0;
  uint32_t second_instruction_offset = 0;
  uint32_t lds_byte_offset = 0;
  uint32_t lds_byte_count = 0;
};

struct MoiShadowCell {
  bool valid = false;
  uint64_t generation = 0;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t access_kind = 0;
  uint32_t instruction_offset = 0;
};

struct MoiAtomicRelease {
  bool valid = false;
  uint64_t generation = 0;
  uint64_t atomic_address = 0;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
};

struct MoiAcquiredEpoch {
  bool valid = false;
  uint64_t generation = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
};

struct MoiSampledEntry {
  bool valid = false;
  bool consumed = false;
  uint32_t kind = 0;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t generation = 0;
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
};

constexpr size_t moi_report_buffer_bytes(uint32_t access_record_capacity,
                                         uint32_t barrier_record_capacity = 0,
                                         uint32_t atomic_record_capacity = 0) {
  return sizeof(MoiReportHeader) + access_record_capacity * sizeof(MoiAccessRecord) +
         barrier_record_capacity * sizeof(MoiBarrierRecord) +
         atomic_record_capacity * sizeof(MoiAtomicRecord);
}

constexpr size_t moi_direct_sampled_report_buffer_bytes(uint32_t sampled_capacity) {
  return sizeof(MoiReportHeader) + sampled_capacity * sizeof(uint64_t);
}

MoiSampledEntry decode_moi_sampled_entry(uint64_t value) {
  return MoiSampledEntry{
      (value & 1u) != 0,
      (value & 2u) != 0,
      static_cast<uint32_t>((value >> 2u) & 0x7u),
      static_cast<uint32_t>((value >> 5u) & 0x3ffu),
      static_cast<uint32_t>((value >> 15u) & 0x3ffu),
      static_cast<uint32_t>((value >> 25u) & 0xfffffu),
      static_cast<uint32_t>((value >> 45u) & 0x3fffu),
      static_cast<uint32_t>(((value >> 59u) & 0x1fu) + 1u),
  };
}

constexpr size_t moi_atomic_records_offset(uint32_t access_record_capacity,
                                           uint32_t barrier_record_capacity = 0) {
  return sizeof(MoiReportHeader) + access_record_capacity * sizeof(MoiAccessRecord) +
         barrier_record_capacity * sizeof(MoiBarrierRecord);
}

struct ReportEnvGuard {
  ~ReportEnvGuard() {
    (void)unsetenv("RJ_CONSAN_MOI_REPORT_BUFFER");
    (void)unsetenv("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE");
  }
};

bool set_moi_report_buffer_env(void *d_report, size_t report_bytes) {
  char address[32];
  const int address_written =
      std::snprintf(address, sizeof(address), "0x%llx",
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(d_report)));
  if (address_written < 0 || static_cast<size_t>(address_written) >= sizeof(address)) {
    ADD_FAILURE() << "failed to format MOI report-buffer address";
    return false;
  }
  char size[32];
  const int size_written =
      std::snprintf(size, sizeof(size), "%llu", static_cast<unsigned long long>(report_bytes));
  if (size_written < 0 || static_cast<size_t>(size_written) >= sizeof(size)) {
    ADD_FAILURE() << "failed to format MOI report-buffer size";
    return false;
  }
  if (setenv("RJ_CONSAN_MOI_REPORT_BUFFER", address, 1) != 0) {
    ADD_FAILURE() << "failed to set RJ_CONSAN_MOI_REPORT_BUFFER";
    return false;
  }
  if (setenv("RJ_CONSAN_MOI_REPORT_BUFFER_SIZE", size, 1) != 0) {
    ADD_FAILURE() << "failed to set RJ_CONSAN_MOI_REPORT_BUFFER_SIZE";
    return false;
  }
  return true;
}

bool moi_access_kinds_conflict(uint32_t current, uint32_t prior) {
  if (current == 0 || prior == 0)
    return false;
  if (current == kMoiAccessKindRead)
    return prior != kMoiAccessKindRead;
  return true;
}

bool moi_sampled_entries_conflict(const MoiSampledEntry &current, const MoiSampledEntry &prior) {
  const uint32_t current_end = current.start_cell + current.cell_count;
  const uint32_t prior_end = prior.start_cell + prior.cell_count;
  return current.valid && prior.valid && !prior.consumed && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         moi_access_kinds_conflict(current.kind, prior.kind) && current.start_cell < prior_end &&
         prior.start_cell < current_end;
}

bool moi_acquired_epoch_orders(const std::vector<MoiAcquiredEpoch> &tokens, uint64_t generation,
                               uint32_t consumer_owner_id, const MoiShadowCell &prior) {
  for (const MoiAcquiredEpoch &token : tokens) {
    if (!token.valid)
      continue;
    if (token.generation != generation || token.generation != prior.generation)
      continue;
    if (token.consumer_owner_id != consumer_owner_id || token.producer_owner_id != prior.owner_id)
      continue;
    if (token.producer_epoch_plus_one >= prior.epoch + 1u)
      return true;
  }
  return false;
}

void replay_moi_atomic_release(std::vector<MoiAtomicRelease> &releases, uint64_t generation,
                               const MoiAtomicRecord &record, uint32_t epoch) {
  MoiAtomicRelease *empty = nullptr;
  for (MoiAtomicRelease &release : releases) {
    if (!release.valid) {
      if (empty == nullptr)
        empty = &release;
      continue;
    }
    if (release.generation != generation || release.atomic_address != record.atomic_address ||
        release.owner_id != record.owner_id)
      continue;
    if (epoch > release.epoch)
      release.epoch = epoch;
    return;
  }
  if (empty != nullptr) {
    *empty = MoiAtomicRelease{
        true, generation, record.atomic_address, record.owner_id, epoch,
    };
  }
}

void replay_moi_atomic_acquire(const std::vector<MoiAtomicRelease> &releases,
                               std::vector<MoiAcquiredEpoch> &tokens, uint64_t generation,
                               const MoiAtomicRecord &record) {
  for (const MoiAtomicRelease &release : releases) {
    if (!release.valid)
      continue;
    if (release.generation != generation || release.atomic_address != record.atomic_address ||
        release.owner_id == record.owner_id)
      continue;
    MoiAcquiredEpoch *empty = nullptr;
    for (MoiAcquiredEpoch &token : tokens) {
      if (!token.valid) {
        if (empty == nullptr)
          empty = &token;
        continue;
      }
      if (token.generation != generation || token.consumer_owner_id != record.owner_id ||
          token.producer_owner_id != release.owner_id)
        continue;
      token.producer_epoch_plus_one = std::max(token.producer_epoch_plus_one, release.epoch + 1u);
      empty = nullptr;
      break;
    }
    if (empty != nullptr) {
      *empty = MoiAcquiredEpoch{
          true, generation, record.owner_id, release.owner_id, release.epoch + 1u,
      };
    }
  }
}

MoiReplayDiagnostic replay_first_moi_conflict(const MoiReportHeader &header,
                                              const MoiAccessRecord *records,
                                              uint32_t record_count) {
  uint32_t required_cells = 0;
  for (uint32_t i = 0; i < record_count; ++i)
    required_cells = std::max(required_cells, records[i].start_cell + records[i].cell_count);

  struct WorkgroupState {
    uint32_t workgroup_x = 0;
    uint32_t workgroup_y = 0;
    uint32_t workgroup_z = 0;
    std::vector<MoiShadowCell> shadow;
  };
  std::vector<WorkgroupState> workgroups;
  auto find_workgroup = [&](const MoiAccessRecord &record) -> WorkgroupState & {
    for (WorkgroupState &workgroup : workgroups) {
      if (workgroup.workgroup_x == record.workgroup_x &&
          workgroup.workgroup_y == record.workgroup_y &&
          workgroup.workgroup_z == record.workgroup_z) {
        return workgroup;
      }
    }
    WorkgroupState workgroup;
    workgroup.workgroup_x = record.workgroup_x;
    workgroup.workgroup_y = record.workgroup_y;
    workgroup.workgroup_z = record.workgroup_z;
    workgroup.shadow.resize(required_cells);
    workgroups.push_back(std::move(workgroup));
    return workgroups.back();
  };

  for (uint32_t i = 0; i < record_count; ++i) {
    const MoiAccessRecord &record = records[i];
    WorkgroupState &workgroup = find_workgroup(record);
    const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
    (void)generation;
    const uint32_t end_cell = record.start_cell + record.cell_count;
    for (uint32_t cell = record.start_cell; cell < end_cell; ++cell) {
      const MoiShadowCell &prior = workgroup.shadow[cell];
      if (prior.valid && prior.epoch == record.epoch && prior.owner_id != record.wave_id &&
          moi_access_kinds_conflict(record.access_kind, prior.access_kind)) {
        return MoiReplayDiagnostic{
            true,
            prior.owner_id,
            record.wave_id,
            prior.access_kind,
            record.access_kind,
            prior.instruction_offset,
            record.instruction_offset,
            record.lds_byte_offset,
            record.lds_byte_count,
        };
      }
    }
    for (uint32_t cell = record.start_cell; cell < end_cell; ++cell) {
      workgroup.shadow[cell] = MoiShadowCell{
          true,         generation,         record.wave_id,
          record.epoch, record.access_kind, record.instruction_offset,
      };
    }
  }

  return {};
}

MoiReplayDiagnostic replay_first_moi_conflict_with_barriers(const MoiReportHeader &header,
                                                            const MoiAccessRecord *records,
                                                            uint32_t record_count,
                                                            const MoiBarrierRecord *barriers,
                                                            uint32_t barrier_count) {
  struct Event {
    uint32_t event_index = 0;
    uint32_t input_order = 0;
    uint32_t record_index = 0;
    bool is_barrier = false;
  };

  uint32_t required_cells = 0;
  for (uint32_t i = 0; i < record_count; ++i)
    required_cells = std::max(required_cells, records[i].start_cell + records[i].cell_count);
  std::vector<MoiShadowCell> shadow(required_cells);
  std::vector<uint32_t> owner_epochs(64);
  uint32_t current_barrier_epoch = 0;
  std::vector<Event> events;
  events.reserve(record_count + barrier_count);
  for (uint32_t i = 0; i < record_count; ++i)
    events.push_back({records[i].event_index, i, i, false});
  for (uint32_t i = 0; i < barrier_count; ++i)
    events.push_back({barriers[i].event_index, record_count + i, i, true});
  std::stable_sort(events.begin(), events.end(), [](const Event &lhs, const Event &rhs) {
    if (lhs.event_index != rhs.event_index)
      return lhs.event_index < rhs.event_index;
    return lhs.input_order < rhs.input_order;
  });

  bool in_barrier_run = false;
  for (const Event &event : events) {
    if (event.is_barrier) {
      if (!in_barrier_run) {
        ++current_barrier_epoch;
        for (uint32_t &epoch : owner_epochs)
          ++epoch;
        in_barrier_run = true;
      }
      continue;
    }

    const MoiAccessRecord &record = records[event.record_index];
    in_barrier_run = false;
    if (record.wave_id >= owner_epochs.size())
      owner_epochs.resize(record.wave_id + 1u, current_barrier_epoch);
    const uint32_t effective_epoch =
        record.epoch != 0 ? record.epoch : owner_epochs[record.wave_id];
    const uint32_t end_cell = record.start_cell + record.cell_count;
    for (uint32_t cell = record.start_cell; cell < end_cell; ++cell) {
      const MoiShadowCell &prior = shadow[cell];
      if (prior.valid && prior.epoch == effective_epoch && prior.owner_id != record.wave_id &&
          moi_access_kinds_conflict(record.access_kind, prior.access_kind)) {
        return MoiReplayDiagnostic{
            true,
            prior.owner_id,
            record.wave_id,
            prior.access_kind,
            record.access_kind,
            prior.instruction_offset,
            record.instruction_offset,
            record.lds_byte_offset,
            record.lds_byte_count,
        };
      }
    }
    for (uint32_t cell = record.start_cell; cell < end_cell; ++cell) {
      shadow[cell] = MoiShadowCell{
          true,
          record.generation != 0 ? record.generation : header.generation,
          record.wave_id,
          effective_epoch,
          record.access_kind,
          record.instruction_offset,
      };
    }
  }

  return {};
}

MoiReplayDiagnostic replay_first_moi_conflict_with_atomics(const MoiReportHeader &header,
                                                           const MoiAccessRecord *records,
                                                           uint32_t record_count,
                                                           const MoiAtomicRecord *atomics,
                                                           uint32_t atomic_count) {
  struct Event {
    enum class Kind {
      Access,
      Atomic,
    };

    uint32_t event_index = 0;
    uint32_t input_order = 0;
    uint32_t record_index = 0;
    Kind kind = Kind::Access;
  };

  uint32_t required_cells = 0;
  for (uint32_t i = 0; i < record_count; ++i)
    required_cells = std::max(required_cells, records[i].start_cell + records[i].cell_count);
  std::vector<MoiShadowCell> shadow(required_cells);
  std::vector<uint32_t> owner_epochs(64);
  std::vector<MoiAtomicRelease> releases(atomic_count);
  std::vector<MoiAcquiredEpoch> tokens(atomic_count);

  std::vector<Event> events;
  events.reserve(record_count + atomic_count);
  for (uint32_t i = 0; i < record_count; ++i)
    events.push_back({records[i].event_index, i, i, Event::Kind::Access});
  for (uint32_t i = 0; i < atomic_count; ++i)
    events.push_back({atomics[i].event_index, record_count + i, i, Event::Kind::Atomic});
  std::stable_sort(events.begin(), events.end(), [](const Event &lhs, const Event &rhs) {
    if (lhs.event_index != rhs.event_index)
      return lhs.event_index < rhs.event_index;
    return lhs.input_order < rhs.input_order;
  });

  for (const Event &event : events) {
    if (event.kind == Event::Kind::Atomic) {
      const MoiAtomicRecord &record = atomics[event.record_index];
      const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
      if (record.owner_id >= owner_epochs.size())
        owner_epochs.resize(record.owner_id + 1u);
      const uint32_t epoch = record.epoch != 0 ? record.epoch : owner_epochs[record.owner_id];
      if (record.kind == kMoiAtomicKindRelease)
        replay_moi_atomic_release(releases, generation, record, epoch);
      else if (record.kind == kMoiAtomicKindAcquire)
        replay_moi_atomic_acquire(releases, tokens, generation, record);
      continue;
    }

    const MoiAccessRecord &record = records[event.record_index];
    if (record.wave_id >= owner_epochs.size())
      owner_epochs.resize(record.wave_id + 1u);
    const uint64_t generation = record.generation != 0 ? record.generation : header.generation;
    const uint32_t effective_epoch =
        record.epoch != 0 ? record.epoch : owner_epochs[record.wave_id];
    const uint32_t end_cell = record.start_cell + record.cell_count;
    for (uint32_t cell = record.start_cell; cell < end_cell; ++cell) {
      const MoiShadowCell &prior = shadow[cell];
      if (prior.valid && prior.epoch == effective_epoch && prior.owner_id != record.wave_id &&
          moi_access_kinds_conflict(record.access_kind, prior.access_kind) &&
          !moi_acquired_epoch_orders(tokens, generation, record.wave_id, prior)) {
        return MoiReplayDiagnostic{
            true,
            prior.owner_id,
            record.wave_id,
            prior.access_kind,
            record.access_kind,
            prior.instruction_offset,
            record.instruction_offset,
            record.lds_byte_offset,
            record.lds_byte_count,
        };
      }
    }
    for (uint32_t cell = record.start_cell; cell < end_cell; ++cell) {
      shadow[cell] = MoiShadowCell{
          true,
          generation,
          record.wave_id,
          effective_epoch,
          record.access_kind,
          record.instruction_offset,
      };
    }
  }

  return {};
}

__global__ __launch_bounds__(
    64) void moi_forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);
  uint32_t values[8] = {
      0x11000000u ^ tid, 0x22000000u ^ tid, 0x33000000u ^ tid, 0x44000000u ^ tid,
      0x55000000u ^ tid, 0x66000000u ^ tid, 0x77000000u ^ tid, 0x88000000u ^ tid,
  };

  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));
  asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
               :
               : "v"(lds_addr), "v"(values[0])
               : "memory");
  asm volatile(""
               : "+v"(values[0]), "+v"(values[1]), "+v"(values[2]), "+v"(values[3]),
                 "+v"(values[4]), "+v"(values[5]), "+v"(values[6]), "+v"(values[7]));

  const uint32_t output_base = tid * 8u;
  for (uint32_t i = 0; i < 8u; ++i)
    out[output_base + i] = values[i];
}

__global__ __launch_bounds__(64) void moi_two_wave_lds_accesses(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid < 32) {
    const uint32_t value = 0xABCD0000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  if (tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__global__ __launch_bounds__(64) void gfx1201_vgpr_scratch_roundtrip(uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  volatile uint32_t private_words[4];
  private_words[tid & 3u] = tid;

  uint32_t value = 0x5A170000u ^ tid;
  asm volatile("scratch_store_b32 off, %0, off offset:0\n\t"
               "s_wait_storecnt 0\n\t"
               "v_mov_b32 %0, 0\n\t"
               "scratch_load_b32 %0, off, off offset:0\n\t"
               "s_wait_loadcnt 0\n\t"
               : "+v"(value)
               :
               : "memory");
  out[tid] = value;
}

__global__
__launch_bounds__(64) void moi_equivalent_lds_accesses_in_different_workgroups(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (blockIdx.x == 0 && tid < 32) {
    const uint32_t value = 0xCAFE0000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  if (blockIdx.x == 1 && tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__global__
__launch_bounds__(64) void moi_equivalent_lds_accesses_in_different_2d_workgroups(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (blockIdx.y == 0 && tid < 32) {
    const uint32_t value = 0x2D000000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  if (blockIdx.y == 1 && tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__global__
__launch_bounds__(64) void moi_equivalent_lds_accesses_in_different_3d_workgroups(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (blockIdx.z == 0 && tid < 32) {
    const uint32_t value = 0x3D000000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  if (blockIdx.z == 1 && tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__global__ __launch_bounds__(64) void moi_two_wave_lds_accesses_with_barrier(uint32_t *out) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid < 32) {
    const uint32_t value = 0x12340000u | tid;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
  }
  __syncthreads();
  if (tid >= 32) {
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    if (tid == 32)
      out[0] = loaded;
  }
}

__global__ __launch_bounds__(64) void moi_flat_atomic_add_record(uint32_t *out) {
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  if (tid == 0) {
    uint32_t one = 1;
    asm volatile("flat_atomic_add_u32 %0, %1 scope:SCOPE_DEV\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 :
                 : "v"(out), "v"(one)
                 : "memory");
  }
}

__global__ __launch_bounds__(64) void moi_atomic_handoff_lds_accesses(uint32_t *out,
                                                                      uint32_t *release_counter,
                                                                      uint32_t *acquire_counter,
                                                                      uint32_t *flag) {
  __shared__ uint32_t lds[1];
  const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
  const auto lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[0]) & 0xFFFFu);

  if (tid == 0) {
    constexpr uint32_t value = 0xA70C0001u;
    uint32_t one = 1;
    asm volatile("ds_store_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 :
                 : "v"(lds_addr), "v"(value)
                 : "memory");
    asm volatile("flat_atomic_add_u32 %0, %1 scope:SCOPE_DEV\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 :
                 : "v"(release_counter), "v"(one)
                 : "memory");
    *flag = 1;
  }

  if (tid == 32) {
    volatile uint32_t *volatile_flag = flag;
    while (*volatile_flag == 0) {
    }
    uint32_t one = 1;
    uint32_t old = 0;
    asm volatile("flat_atomic_add_u32 %0, %1, %2 th:TH_ATOMIC_RETURN scope:SCOPE_DEV\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 : "=&v"(old)
                 : "v"(acquire_counter), "v"(one)
                 : "memory");
    uint32_t loaded = 0;
    asm volatile("ds_load_b32 %0, %1\n\t" RJ_CONSAN_MOI_RECORD_NOP_CAVE
                 : "=v"(loaded)
                 : "v"(lds_addr)
                 : "memory");
    out[0] = loaded ^ old;
  }
}

void free_device_ptr(uint32_t *ptr) {
  if (ptr != nullptr)
    (void)hipFree(ptr);
}

struct LiveAtomicHandoffRecords {
  MoiReportHeader header;
  std::vector<MoiAccessRecord> access_records;
  std::vector<MoiAtomicRecord> atomic_records;
};

void run_live_atomic_handoff(bool same_atomic_address, LiveAtomicHandoffRecords &records) {
  constexpr uint32_t kAccessSlots = 8;
  constexpr uint32_t kAtomicSlots = 8;
  constexpr size_t kReportBytes =
      moi_report_buffer_bytes(kAccessSlots, /*barrier_record_capacity=*/0, kAtomicSlots);
  uint32_t *d_out = nullptr;
  uint32_t *d_release_counter = nullptr;
  uint32_t *d_acquire_counter = nullptr;
  uint32_t *d_flag = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_release_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_acquire_counter, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_flag, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_release_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_acquire_counter, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_flag, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  uint32_t *acquire_counter = same_atomic_address ? d_release_counter : d_acquire_counter;
  moi_atomic_handoff_lds_accesses<<<1, 64>>>(d_out, d_release_counter, acquire_counter, d_flag);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  EXPECT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_flag);
  free_device_ptr(d_acquire_counter);
  free_device_ptr(d_release_counter);
  free_device_ptr(d_out);

  std::memcpy(&records.header, report.data(), sizeof(records.header));
  records.access_records.resize(kAccessSlots);
  records.atomic_records.resize(kAtomicSlots);
  std::memcpy(records.access_records.data(), report.data() + sizeof(MoiReportHeader),
              kAccessSlots * sizeof(MoiAccessRecord));
  std::memcpy(records.atomic_records.data(),
              report.data() + moi_atomic_records_offset(kAccessSlots),
              kAtomicSlots * sizeof(MoiAtomicRecord));
  records.access_records.erase(
      std::remove_if(records.access_records.begin(), records.access_records.end(),
                     [](const MoiAccessRecord &record) { return record.access_kind == 0; }),
      records.access_records.end());
  records.atomic_records.erase(
      std::remove_if(records.atomic_records.begin(), records.atomic_records.end(),
                     [](const MoiAtomicRecord &record) {
                       return record.kind == 0 || record.atomic_address == 0;
                     }),
      records.atomic_records.end());
}

} // namespace

TEST(ConSanSpillHipTest, Gfx1201VgprScratchRoundTrip) {
  constexpr uint32_t kThreads = 64;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kThreads * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, kThreads * sizeof(uint32_t)));

  gfx1201_vgpr_scratch_roundtrip<<<1, kThreads>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, kThreads * sizeof(uint32_t), hipMemcpyDeviceToHost));
  free_device_ptr(d_out);
  for (uint32_t tid = 0; tid < kThreads; ++tid)
    EXPECT_EQ(out[tid], 0x5A170000u ^ tid) << "tid=" << tid;
}

TEST(ConSanMoiHipTest, ForcedSpillRecordReplayPreservesLiveVgprsAndWritesRecord) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  constexpr uint32_t kRecordSlots = 1;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kRecordSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));
  moi_forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation<<<1, kThreads>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  constexpr uint32_t kBases[kValuesPerThread] = {
      0x11000000u, 0x22000000u, 0x33000000u, 0x44000000u,
      0x55000000u, 0x66000000u, 0x77000000u, 0x88000000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], kBases[i] ^ tid)
          << "tid=" << tid << " value=" << i;
  }

  std::vector<MoiAccessRecord> records(kRecordSlots);
  std::memcpy(records.data(), report.data() + sizeof(MoiReportHeader),
              records.size() * sizeof(MoiAccessRecord));
  const auto record = std::find_if(records.begin(), records.end(), [](const MoiAccessRecord &item) {
    return item.access_kind != 0;
  });
  ASSERT_NE(record, records.end());
  EXPECT_EQ(record->access_kind, kMoiAccessKindWrite);
  EXPECT_EQ(record->lds_byte_count, sizeof(uint32_t));
  EXPECT_EQ(record->cell_count, 1u);
}

TEST(ConSanMoiHipTest, ForcedSpillSampledPreservesLiveVgprsAndWritesEntry) {
  constexpr uint32_t kThreads = 64;
  constexpr uint32_t kValuesPerThread = 8;
  constexpr uint32_t kSampledSlots = 1;
  constexpr size_t kReportBytes = moi_direct_sampled_report_buffer_bytes(kSampledSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, kThreads * kValuesPerThread * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));
  moi_forced_spill_preserves_live_vgprs_test_kernel_for_instrumentation<<<1, kThreads>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> out(kThreads * kValuesPerThread);
  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(out.data(), d_out, out.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  constexpr uint32_t kBases[kValuesPerThread] = {
      0x11000000u, 0x22000000u, 0x33000000u, 0x44000000u,
      0x55000000u, 0x66000000u, 0x77000000u, 0x88000000u,
  };
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kValuesPerThread; ++i)
      EXPECT_EQ(out[tid * kValuesPerThread + i], kBases[i] ^ tid)
          << "tid=" << tid << " value=" << i;
  }

  std::optional<MoiSampledEntry> live_entry;
  for (uint32_t i = 0; i < kSampledSlots; ++i) {
    uint64_t raw_entry = 0;
    std::memcpy(&raw_entry, report.data() + sizeof(MoiReportHeader) + i * sizeof(uint64_t),
                sizeof(raw_entry));
    const MoiSampledEntry entry = decode_moi_sampled_entry(raw_entry);
    if (entry.valid) {
      live_entry = entry;
      break;
    }
  }
  ASSERT_TRUE(live_entry);
  EXPECT_EQ(live_entry->kind, kMoiAccessKindWrite);
  EXPECT_EQ(live_entry->cell_count, 1u);
}

TEST(ConSanMoiHipTest, TwoWaveAccessesWriteDynamicRecords) {
  constexpr uint32_t kAccessRecords = 2;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kAccessRecords);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  moi_two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.access_record_count, kAccessRecords);
  EXPECT_GE(header.event_counter, kAccessRecords);

  MoiAccessRecord records[kAccessRecords];
  std::memcpy(records, report.data() + sizeof(MoiReportHeader), sizeof(records));
  EXPECT_EQ(records[0].access_kind, kMoiAccessKindWrite);
  EXPECT_EQ(records[1].access_kind, kMoiAccessKindRead);
  EXPECT_EQ(records[0].lds_byte_offset, records[1].lds_byte_offset);
  EXPECT_EQ(records[0].lds_byte_count, sizeof(uint32_t));
  EXPECT_EQ(records[1].lds_byte_count, sizeof(uint32_t));
  EXPECT_EQ(records[0].start_cell, records[1].start_cell);
  EXPECT_EQ(records[0].cell_count, 1u);
  EXPECT_EQ(records[1].cell_count, 1u);
  EXPECT_NE(records[0].wave_id, records[1].wave_id);
  EXPECT_EQ(records[0].workgroup_x, 0u);
  EXPECT_EQ(records[1].workgroup_x, 0u);
  EXPECT_NE(records[0].event_index, records[1].event_index);

  const MoiReplayDiagnostic diagnostic = replay_first_moi_conflict(header, records, kAccessRecords);
  EXPECT_TRUE(diagnostic.conflict);
  EXPECT_EQ(diagnostic.first_owner_id, records[0].wave_id);
  EXPECT_EQ(diagnostic.second_owner_id, records[1].wave_id);
  EXPECT_EQ(diagnostic.first_access_kind, kMoiAccessKindWrite);
  EXPECT_EQ(diagnostic.second_access_kind, kMoiAccessKindRead);
  EXPECT_EQ(diagnostic.first_instruction_offset, records[0].instruction_offset);
  EXPECT_EQ(diagnostic.second_instruction_offset, records[1].instruction_offset);
  EXPECT_EQ(diagnostic.lds_byte_offset, records[1].lds_byte_offset);
  EXPECT_EQ(diagnostic.lds_byte_count, sizeof(uint32_t));

  MoiAccessRecord ordered_records[kAccessRecords];
  std::memcpy(ordered_records, records, sizeof(ordered_records));
  ordered_records[1].epoch = ordered_records[0].epoch + 1;
  EXPECT_FALSE(replay_first_moi_conflict(header, ordered_records, kAccessRecords).conflict);
}

TEST(ConSanMoiHipTest, DynamicAccessRecordsAppendActiveLanes) {
  constexpr uint32_t kAccessSlots = 96;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kAccessSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  moi_two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_GE(header.access_record_count, 64u);
  EXPECT_LE(header.access_record_count, kAccessSlots);
  EXPECT_GE(header.event_counter, 64u);

  std::vector<MoiAccessRecord> records(kAccessSlots);
  std::memcpy(records.data(), report.data() + sizeof(MoiReportHeader),
              records.size() * sizeof(MoiAccessRecord));
  records.erase(
      std::remove_if(records.begin(), records.end(),
                     [](const MoiAccessRecord &record) { return record.access_kind == 0; }),
      records.end());
  EXPECT_GE(records.size(), 64u);

  const size_t write_count =
      std::count_if(records.begin(), records.end(), [](const MoiAccessRecord &record) {
        return record.access_kind == kMoiAccessKindWrite;
      });
  const size_t read_count =
      std::count_if(records.begin(), records.end(), [](const MoiAccessRecord &record) {
        return record.access_kind == kMoiAccessKindRead;
      });
  EXPECT_GE(write_count, 32u);
  EXPECT_GE(read_count, 32u);

  for (const MoiAccessRecord &record : records) {
    EXPECT_EQ(record.lds_byte_count, sizeof(uint32_t));
    EXPECT_EQ(record.cell_count, 1u);
    EXPECT_EQ(record.workgroup_x, 0u);
  }

  const MoiReplayDiagnostic diagnostic =
      replay_first_moi_conflict(header, records.data(), static_cast<uint32_t>(records.size()));
  EXPECT_TRUE(diagnostic.conflict);
}

TEST(ConSanMoiHipTest, TwoWaveAccessesRunWithAutoReportBuffer) {
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));

  moi_two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  EXPECT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  free_device_ptr(d_out);
}

TEST(ConSanMoiHipTest, SampledBackendWritesDirectEntries) {
  constexpr uint32_t kSampledSlots = 2;
  constexpr size_t kReportBytes = moi_direct_sampled_report_buffer_bytes(kSampledSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  moi_two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.access_record_count, 0u);
  EXPECT_EQ(header.barrier_record_count, 0u);
  EXPECT_EQ(header.atomic_record_count, 0u);
  EXPECT_EQ(header.event_counter, 0u);

  std::vector<MoiSampledEntry> live_entries;
  for (uint32_t i = 0; i < kSampledSlots; ++i) {
    uint64_t raw_entry = 0;
    std::memcpy(&raw_entry, report.data() + sizeof(MoiReportHeader) + i * sizeof(uint64_t),
                sizeof(raw_entry));
    MoiSampledEntry entry = decode_moi_sampled_entry(raw_entry);
    if (entry.valid)
      live_entries.push_back(entry);
  }

  ASSERT_EQ(live_entries.size(), 2u);
  std::stable_sort(
      live_entries.begin(), live_entries.end(),
      [](const MoiSampledEntry &lhs, const MoiSampledEntry &rhs) { return lhs.kind < rhs.kind; });
  EXPECT_EQ(live_entries[0].kind, kMoiAccessKindRead);
  EXPECT_EQ(live_entries[1].kind, kMoiAccessKindWrite);
  EXPECT_FALSE(live_entries[0].consumed);
  EXPECT_FALSE(live_entries[1].consumed);
  EXPECT_NE(live_entries[0].owner_id, live_entries[1].owner_id);
  EXPECT_EQ(live_entries[0].epoch, 0u);
  EXPECT_EQ(live_entries[1].epoch, 0u);
  EXPECT_EQ(live_entries[0].generation, 0u);
  EXPECT_EQ(live_entries[1].generation, 0u);
  EXPECT_EQ(live_entries[0].start_cell, live_entries[1].start_cell);
  EXPECT_EQ(live_entries[0].cell_count, 1u);
  EXPECT_EQ(live_entries[1].cell_count, 1u);
  EXPECT_TRUE(moi_sampled_entries_conflict(live_entries[1], live_entries[0]));
}

TEST(ConSanMoiHipTest, RuntimeSampledBackendSelectsOneWave) {
  constexpr uint32_t kSampledSlots = 2;
  constexpr size_t kReportBytes = moi_direct_sampled_report_buffer_bytes(kSampledSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  moi_two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  std::vector<MoiSampledEntry> live_entries;
  for (uint32_t i = 0; i < kSampledSlots; ++i) {
    uint64_t raw_entry = 0;
    std::memcpy(&raw_entry, report.data() + sizeof(MoiReportHeader) + i * sizeof(uint64_t),
                sizeof(raw_entry));
    MoiSampledEntry entry = decode_moi_sampled_entry(raw_entry);
    if (entry.valid)
      live_entries.push_back(entry);
  }

  ASSERT_EQ(live_entries.size(), 1u);
  EXPECT_EQ(live_entries.front().owner_id & 1u, 0u);
  EXPECT_EQ(live_entries.front().epoch, 0u);
  EXPECT_EQ(live_entries.front().cell_count, 1u);
}

TEST(ConSanMoiHipTest, EquivalentAccessesInDifferentWorkgroupsDoNotConflict) {
  constexpr uint32_t kAccessSlots = 8;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kAccessSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  moi_equivalent_lds_accesses_in_different_workgroups<<<2, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.access_record_count, kAccessSlots);

  MoiAccessRecord records[kAccessSlots];
  std::memcpy(records, report.data() + sizeof(MoiReportHeader), sizeof(records));
  std::vector<MoiAccessRecord> live_records;
  for (const MoiAccessRecord &record : records) {
    if (record.access_kind != 0)
      live_records.push_back(record);
  }
  ASSERT_EQ(live_records.size(), 2u);
  EXPECT_GE(header.event_counter, live_records.size());

  std::stable_sort(live_records.begin(), live_records.end(),
                   [](const MoiAccessRecord &lhs, const MoiAccessRecord &rhs) {
                     return lhs.workgroup_x < rhs.workgroup_x;
                   });
  EXPECT_EQ(live_records[0].access_kind, kMoiAccessKindWrite);
  EXPECT_EQ(live_records[1].access_kind, kMoiAccessKindRead);
  EXPECT_EQ(live_records[0].lds_byte_offset, live_records[1].lds_byte_offset);
  EXPECT_EQ(live_records[0].start_cell, live_records[1].start_cell);
  EXPECT_EQ(live_records[0].workgroup_x, 0u);
  EXPECT_EQ(live_records[1].workgroup_x, 1u);
  EXPECT_EQ(live_records[0].workgroup_y, 0u);
  EXPECT_EQ(live_records[1].workgroup_y, 0u);
  EXPECT_EQ(live_records[0].workgroup_z, 0u);
  EXPECT_EQ(live_records[1].workgroup_z, 0u);
  EXPECT_NE(live_records[0].wave_id, live_records[1].wave_id);
  EXPECT_FALSE(replay_first_moi_conflict(header, live_records.data(),
                                         static_cast<uint32_t>(live_records.size()))
                   .conflict);

  MoiAccessRecord same_workgroup_records[2];
  std::memcpy(same_workgroup_records, live_records.data(), sizeof(same_workgroup_records));
  same_workgroup_records[1].workgroup_x = same_workgroup_records[0].workgroup_x;
  EXPECT_TRUE(replay_first_moi_conflict(header, same_workgroup_records, 2).conflict);
}

TEST(ConSanMoiHipTest, EquivalentAccessesInDifferent2DWorkgroupsDoNotConflict) {
  constexpr uint32_t kAccessSlots = 10;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kAccessSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  const dim3 grid(1, 2, 1);
  moi_equivalent_lds_accesses_in_different_2d_workgroups<<<grid, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.access_record_count, kAccessSlots);

  MoiAccessRecord records[kAccessSlots];
  std::memcpy(records, report.data() + sizeof(MoiReportHeader), sizeof(records));
  std::vector<MoiAccessRecord> live_records;
  for (const MoiAccessRecord &record : records) {
    if (record.access_kind != 0)
      live_records.push_back(record);
  }
  ASSERT_EQ(live_records.size(), 2u);
  EXPECT_GE(header.event_counter, live_records.size());

  std::stable_sort(live_records.begin(), live_records.end(),
                   [](const MoiAccessRecord &lhs, const MoiAccessRecord &rhs) {
                     return lhs.workgroup_y < rhs.workgroup_y;
                   });
  EXPECT_EQ(live_records[0].access_kind, kMoiAccessKindWrite);
  EXPECT_EQ(live_records[1].access_kind, kMoiAccessKindRead);
  EXPECT_EQ(live_records[0].lds_byte_offset, live_records[1].lds_byte_offset);
  EXPECT_EQ(live_records[0].start_cell, live_records[1].start_cell);
  EXPECT_EQ(live_records[0].workgroup_x, 0u);
  EXPECT_EQ(live_records[1].workgroup_x, 0u);
  EXPECT_EQ(live_records[0].workgroup_y, 0u);
  EXPECT_EQ(live_records[1].workgroup_y, 1u);
  EXPECT_EQ(live_records[0].workgroup_z, 0u);
  EXPECT_EQ(live_records[1].workgroup_z, 0u);
  EXPECT_NE(live_records[0].wave_id, live_records[1].wave_id);
  EXPECT_FALSE(replay_first_moi_conflict(header, live_records.data(),
                                         static_cast<uint32_t>(live_records.size()))
                   .conflict);

  MoiAccessRecord same_workgroup_records[2];
  std::memcpy(same_workgroup_records, live_records.data(), sizeof(same_workgroup_records));
  same_workgroup_records[1].workgroup_y = same_workgroup_records[0].workgroup_y;
  EXPECT_TRUE(replay_first_moi_conflict(header, same_workgroup_records, 2).conflict);
}

TEST(ConSanMoiHipTest, EquivalentAccessesInDifferent3DWorkgroupsDoNotConflict) {
  constexpr uint32_t kAccessSlots = 12;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kAccessSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  const dim3 grid(1, 1, 2);
  moi_equivalent_lds_accesses_in_different_3d_workgroups<<<grid, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.access_record_count, kAccessSlots);

  MoiAccessRecord records[kAccessSlots];
  std::memcpy(records, report.data() + sizeof(MoiReportHeader), sizeof(records));
  std::vector<MoiAccessRecord> live_records;
  for (const MoiAccessRecord &record : records) {
    if (record.access_kind != 0)
      live_records.push_back(record);
  }
  ASSERT_EQ(live_records.size(), 2u);
  EXPECT_GE(header.event_counter, live_records.size());

  std::stable_sort(live_records.begin(), live_records.end(),
                   [](const MoiAccessRecord &lhs, const MoiAccessRecord &rhs) {
                     return lhs.workgroup_z < rhs.workgroup_z;
                   });
  EXPECT_EQ(live_records[0].access_kind, kMoiAccessKindWrite);
  EXPECT_EQ(live_records[1].access_kind, kMoiAccessKindRead);
  EXPECT_EQ(live_records[0].lds_byte_offset, live_records[1].lds_byte_offset);
  EXPECT_EQ(live_records[0].start_cell, live_records[1].start_cell);
  EXPECT_EQ(live_records[0].workgroup_x, 0u);
  EXPECT_EQ(live_records[1].workgroup_x, 0u);
  EXPECT_EQ(live_records[0].workgroup_y, 0u);
  EXPECT_EQ(live_records[1].workgroup_y, 0u);
  EXPECT_EQ(live_records[0].workgroup_z, 0u);
  EXPECT_EQ(live_records[1].workgroup_z, 1u);
  EXPECT_NE(live_records[0].wave_id, live_records[1].wave_id);
  EXPECT_FALSE(replay_first_moi_conflict(header, live_records.data(),
                                         static_cast<uint32_t>(live_records.size()))
                   .conflict);

  MoiAccessRecord same_workgroup_records[2];
  std::memcpy(same_workgroup_records, live_records.data(), sizeof(same_workgroup_records));
  same_workgroup_records[1].workgroup_z = same_workgroup_records[0].workgroup_z;
  EXPECT_TRUE(replay_first_moi_conflict(header, same_workgroup_records, 2).conflict);
}

TEST(ConSanMoiHipTest, OneRecordCapacityLimitsAccessPatches) {
  constexpr uint32_t kAccessRecords = 1;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kAccessRecords);
  constexpr size_t kSentinelBytes = 64;
  constexpr uint8_t kSentinel = 0xA5;
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes + kSentinelBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));
  auto *d_report_bytes = static_cast<uint8_t *>(d_report);
  HIP_ASSERT(hipMemset(d_report_bytes + kReportBytes, kSentinel, kSentinelBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  moi_two_wave_lds_accesses<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes + kSentinelBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.access_record_count, kAccessRecords);
  EXPECT_GE(header.event_counter, kAccessRecords);

  MoiAccessRecord record;
  std::memcpy(&record, report.data() + sizeof(MoiReportHeader), sizeof(record));
  EXPECT_NE(record.access_kind, 0u);
  EXPECT_EQ(record.lds_byte_count, sizeof(uint32_t));
  EXPECT_EQ(record.cell_count, 1u);

  for (size_t i = 0; i < kSentinelBytes; ++i)
    EXPECT_EQ(report[kReportBytes + i], kSentinel) << "sentinel byte " << i << " was modified";
}

TEST(ConSanMoiHipTest, BarrierRecordOrdersTwoWaveAccesses) {
  constexpr uint32_t kAccessSlots = 8;
  constexpr uint32_t kBarrierSlots = 8;
  constexpr size_t kReportBytes = moi_report_buffer_bytes(kAccessSlots, kBarrierSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  moi_two_wave_lds_accesses_with_barrier<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_EQ(header.access_record_count, kAccessSlots);
  EXPECT_GE(header.barrier_record_count, 1u);
  EXPECT_LE(header.barrier_record_count, kBarrierSlots);
  EXPECT_GE(header.event_counter, kAccessSlots + header.barrier_record_count);

  MoiAccessRecord records[kAccessSlots];
  std::memcpy(records, report.data() + sizeof(MoiReportHeader), sizeof(records));
  MoiBarrierRecord barriers[kBarrierSlots];
  std::memcpy(barriers, report.data() + sizeof(MoiReportHeader) + sizeof(records),
              sizeof(barriers));

  std::vector<MoiAccessRecord> live_records;
  for (const MoiAccessRecord &record : records) {
    if (record.access_kind != 0)
      live_records.push_back(record);
  }
  std::stable_sort(live_records.begin(), live_records.end(),
                   [](const MoiAccessRecord &lhs, const MoiAccessRecord &rhs) {
                     return lhs.event_index < rhs.event_index;
                   });
  ASSERT_EQ(live_records.size(), 2u);

  EXPECT_EQ(live_records[0].access_kind, kMoiAccessKindWrite);
  EXPECT_EQ(live_records[1].access_kind, kMoiAccessKindRead);
  EXPECT_EQ(live_records[0].lds_byte_offset, live_records[1].lds_byte_offset);
  EXPECT_EQ(live_records[0].start_cell, live_records[1].start_cell);
  EXPECT_NE(live_records[0].wave_id, live_records[1].wave_id);

  bool saw_barrier_between_accesses = false;
  for (uint32_t i = 0; i < header.barrier_record_count; ++i) {
    saw_barrier_between_accesses |= live_records[0].event_index < barriers[i].event_index &&
                                    barriers[i].event_index < live_records[1].event_index;
  }
  EXPECT_TRUE(saw_barrier_between_accesses);

  const uint32_t live_record_count = static_cast<uint32_t>(live_records.size());
  EXPECT_TRUE(replay_first_moi_conflict(header, live_records.data(), live_record_count).conflict);
  EXPECT_FALSE(replay_first_moi_conflict_with_barriers(header, live_records.data(),
                                                       live_record_count, barriers,
                                                       header.barrier_record_count)
                   .conflict);
}

TEST(ConSanMoiHipTest, FlatAtomicWritesDynamicRecord) {
  constexpr uint32_t kAccessSlots = 1;
  constexpr uint32_t kAtomicSlots = 1;
  constexpr size_t kReportBytes =
      moi_report_buffer_bytes(kAccessSlots, /*barrier_record_capacity=*/0, kAtomicSlots);
  uint32_t *d_out = nullptr;
  void *d_report = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, 2 * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&d_report, kReportBytes));
  HIP_ASSERT(hipMemset(d_out, 0, 2 * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_report, 0, kReportBytes));

  ReportEnvGuard report_env;
  ASSERT_TRUE(set_moi_report_buffer_env(d_report, kReportBytes));

  moi_flat_atomic_add_record<<<1, 64>>>(d_out);
  HIP_ASSERT(hipGetLastError());
  const hipError_t sync = hipDeviceSynchronize();
  ASSERT_EQ(sync, hipSuccess) << "HIP error: " << hipGetErrorString(sync);

  uint32_t out[2] = {};
  HIP_ASSERT(hipMemcpy(out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  EXPECT_EQ(out[0], 1u);
  EXPECT_EQ(out[1], 0u);

  std::vector<uint8_t> report(kReportBytes);
  HIP_ASSERT(hipMemcpy(report.data(), d_report, report.size(), hipMemcpyDeviceToHost));
  free_device_ptr(static_cast<uint32_t *>(d_report));
  free_device_ptr(d_out);

  MoiReportHeader header;
  std::memcpy(&header, report.data(), sizeof(header));
  EXPECT_LE(header.access_record_count, kAccessSlots);
  EXPECT_EQ(header.atomic_record_count, kAtomicSlots);
  EXPECT_GE(header.event_counter, kAtomicSlots);

  MoiAtomicRecord record;
  std::memcpy(&record, report.data() + moi_atomic_records_offset(kAccessSlots), sizeof(record));
  EXPECT_EQ(record.workgroup_x, 0u);
  EXPECT_EQ(record.workgroup_y, 0u);
  EXPECT_EQ(record.workgroup_z, 0u);
  EXPECT_EQ(record.kind, kMoiAtomicKindRelease);
  EXPECT_EQ(record.scope, kRdna4ScopeDevice);
  EXPECT_EQ(record.semantics, kRdna4AtomicNoReturnTh);
  EXPECT_NE(record.atomic_address, 0u);
  EXPECT_NE(record.instruction_offset, 0u);
}

TEST(ConSanMoiHipTest, AtomicHandoffOrdersTwoWaveAccesses) {
  LiveAtomicHandoffRecords same_address;
  run_live_atomic_handoff(/*same_atomic_address=*/true, same_address);
  EXPECT_GE(same_address.header.access_record_count, 2u);
  EXPECT_GE(same_address.header.atomic_record_count, 2u);
  EXPECT_GE(same_address.header.event_counter, 4u);

  ASSERT_EQ(same_address.access_records.size(), 2u);
  ASSERT_EQ(same_address.atomic_records.size(), 2u);
  std::sort(same_address.access_records.begin(), same_address.access_records.end(),
            [](const MoiAccessRecord &lhs, const MoiAccessRecord &rhs) {
              return lhs.event_index < rhs.event_index;
            });
  std::sort(same_address.atomic_records.begin(), same_address.atomic_records.end(),
            [](const MoiAtomicRecord &lhs, const MoiAtomicRecord &rhs) {
              return lhs.event_index < rhs.event_index;
            });
  EXPECT_EQ(same_address.access_records[0].access_kind, kMoiAccessKindWrite)
      << "access events " << same_address.access_records[0].event_index << ", "
      << same_address.access_records[1].event_index;
  EXPECT_EQ(same_address.access_records[1].access_kind, kMoiAccessKindRead);
  EXPECT_EQ(same_address.access_records[0].start_cell, same_address.access_records[1].start_cell);
  EXPECT_NE(same_address.access_records[0].wave_id, same_address.access_records[1].wave_id);
  EXPECT_EQ(same_address.atomic_records[0].kind, kMoiAtomicKindRelease)
      << "atomic events " << same_address.atomic_records[0].event_index << ", "
      << same_address.atomic_records[1].event_index;
  EXPECT_EQ(same_address.atomic_records[1].kind, kMoiAtomicKindAcquire);
  EXPECT_EQ(same_address.atomic_records[0].scope, kRdna4ScopeDevice);
  EXPECT_EQ(same_address.atomic_records[1].scope, kRdna4ScopeDevice);
  EXPECT_EQ(same_address.atomic_records[0].semantics, kRdna4AtomicNoReturnTh);
  EXPECT_EQ(same_address.atomic_records[1].semantics, kRdna4AtomicReturnTh);
  EXPECT_EQ(same_address.atomic_records[0].atomic_address,
            same_address.atomic_records[1].atomic_address);

  const MoiReplayDiagnostic unordered =
      replay_first_moi_conflict(same_address.header, same_address.access_records.data(),
                                static_cast<uint32_t>(same_address.access_records.size()));
  EXPECT_TRUE(unordered.conflict);

  const MoiReplayDiagnostic ordered = replay_first_moi_conflict_with_atomics(
      same_address.header, same_address.access_records.data(),
      static_cast<uint32_t>(same_address.access_records.size()), same_address.atomic_records.data(),
      static_cast<uint32_t>(same_address.atomic_records.size()));
  EXPECT_FALSE(ordered.conflict) << "access events " << same_address.access_records[0].event_index
                                 << ", " << same_address.access_records[1].event_index
                                 << "; atomic events " << same_address.atomic_records[0].event_index
                                 << ", " << same_address.atomic_records[1].event_index
                                 << "; access owners " << same_address.access_records[0].wave_id
                                 << ", " << same_address.access_records[1].wave_id
                                 << "; atomic owners " << same_address.atomic_records[0].owner_id
                                 << ", " << same_address.atomic_records[1].owner_id
                                 << "; access epochs " << same_address.access_records[0].epoch
                                 << ", " << same_address.access_records[1].epoch
                                 << "; atomic epochs " << same_address.atomic_records[0].epoch
                                 << ", " << same_address.atomic_records[1].epoch;

  LiveAtomicHandoffRecords wrong_address;
  run_live_atomic_handoff(/*same_atomic_address=*/false, wrong_address);
  EXPECT_GE(wrong_address.header.access_record_count, 2u);
  EXPECT_GE(wrong_address.header.atomic_record_count, 2u);
  ASSERT_EQ(wrong_address.access_records.size(), 2u);
  ASSERT_EQ(wrong_address.atomic_records.size(), 2u);
  std::sort(wrong_address.access_records.begin(), wrong_address.access_records.end(),
            [](const MoiAccessRecord &lhs, const MoiAccessRecord &rhs) {
              return lhs.event_index < rhs.event_index;
            });
  std::sort(wrong_address.atomic_records.begin(), wrong_address.atomic_records.end(),
            [](const MoiAtomicRecord &lhs, const MoiAtomicRecord &rhs) {
              return lhs.event_index < rhs.event_index;
            });
  EXPECT_NE(wrong_address.atomic_records[0].atomic_address,
            wrong_address.atomic_records[1].atomic_address);

  const MoiReplayDiagnostic wrong_address_replay = replay_first_moi_conflict_with_atomics(
      wrong_address.header, wrong_address.access_records.data(),
      static_cast<uint32_t>(wrong_address.access_records.size()),
      wrong_address.atomic_records.data(),
      static_cast<uint32_t>(wrong_address.atomic_records.size()));
  EXPECT_TRUE(wrong_address_replay.conflict);
}
