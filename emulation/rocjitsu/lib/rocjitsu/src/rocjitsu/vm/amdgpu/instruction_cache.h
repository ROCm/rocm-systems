// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm_access.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>

namespace rocjitsu {
namespace amdgpu {

/// @brief Per-compute-unit instruction cache (I$).
///
/// @details The issue path reads the 16 bytes at the PC for every instruction
/// it retires. Uncached, each of those reads walks the page table and takes a
/// reader lock on a GpuMemory page stripe. Every CU of a dispatch runs the same
/// kernel, so those reads all fall on the one stripe holding the code page and
/// contend on a single host cache line; that contention, not the emulation
/// work, dominates a multi-threaded run.
///
/// The cache is private to one CU and takes no lock. Lines are 64 bytes and
/// never straddle a page, so a fill costs one page-stripe lock instead of
/// sixteen word reads, and a kernel loop that fits in @ref kCacheBytes costs
/// none at all.
///
/// Coherence follows hardware: an AMDGPU I$ is not coherent with data writes,
/// and code that rewrites itself must issue s_icache_inv. The cache is
/// invalidated where the driver would issue one -- once per dispatch at kernel
/// launch, on s_icache_inv itself, at CU cache maintenance, and at the command
/// device-wide maintenance around direct backing writes. It is
/// bypassed entirely while a debugger is attached, because breakpoint writes
/// reach code memory without any of those, and dropped on every attach and
/// detach so a session cannot leave a stale line behind either way.
class InstructionCache {
public:
  InstructionCache() : coherence_(std::make_shared<DeviceCacheCoherence>()) {}

  /// @name Cache geometry
  /// @details Chosen for host speed, not copied from an ISA profile, and
  /// deliberately not derived from one. The CDNA5 manual gives a 64 KiB
  /// instruction cache per WGP; ComputeUnitCore::Config::l1_size_kb is the
  /// WGP *data* cache and describes nothing about this structure. What these
  /// numbers have to be is large enough that a hot kernel loop stops paying
  /// for GpuMemory lookups and small enough to stay in the host's own caches
  /// alongside the rest of a CU. 4 KiB direct-mapped does that; nothing in the
  /// emulated architecture is observable through them, because the I$ is
  /// modelled for its coherence behaviour and not for hit rates or timing.
  /// @{
  static constexpr uint32_t kLineSize = 64;
  static constexpr uint32_t kNumLines = 64;
  static constexpr uint32_t kCacheBytes = kLineSize * kNumLines;
  /// @}

  /// @brief Number of bytes the issue path reads at the PC.
  static constexpr uint32_t kFetchBytes = 16;

  static_assert(kLineSize >= kFetchBytes, "a fetch may straddle at most two lines");
  static_assert(GpuMemory::PAGE_SIZE % kLineSize == 0,
                "an aligned line must fall inside one page, so a fill is one page chunk");

  /// @brief Read @p byte_count at @p pc, filling from @p memory on a miss.
  /// @param memory Backing GPU memory.
  /// @param pc Program counter; four-byte aligned.
  /// @param[out] dst Buffer of at least @p byte_count bytes.
  /// @param byte_count Number of bytes to read, at most one cache line.
  void fetch(const GpuMemory &memory, uint64_t pc, uint8_t *dst,
             uint32_t byte_count = kFetchBytes) {
    synchronize_coherence_epoch();
    const uint32_t offset = static_cast<uint32_t>(pc) & (kLineSize - 1);
    const uint8_t *line = line_for(memory, pc);

    if (offset + byte_count <= kLineSize) {
      std::memcpy(dst, line + offset, byte_count);
      return;
    }

    // Straddles into the next line. Copy the head out first: the second lookup
    // may fill, and while adjacent lines never share a set today, relying on
    // that would make the geometry load-bearing.
    const uint32_t head = kLineSize - offset;
    std::memcpy(dst, line + offset, head);
    const uint64_t next = (pc & ~uint64_t{kLineSize - 1}) + kLineSize;
    std::memcpy(dst + head, line_for(memory, next), byte_count - head);
  }

  /// @brief Fetch through a retained VM binding and generation-safe namespace.
  ///
  /// This path is used by translated PCI/VFIO address spaces.  Lines from a
  /// replaced root or a reused address-space slot cannot alias because both
  /// the handle generation and translation epoch participate in the tag.
  [[nodiscard, gnu::always_inline]] VmAccessOutcome
  fetch(const GpuVmAccess &access, uint64_t pc, uint8_t *dst, uint32_t byte_count = kFetchBytes) {
    synchronize_coherence_epoch();
    const uint32_t line_offset = static_cast<uint32_t>(pc) & (kLineSize - 1);
    if (access.info().legacy_cache_compatible && byte_count && byte_count <= kLineSize) {
      const uint64_t base = pc & ~uint64_t{kLineSize - 1};
      const Line &first = lines_[(base / kLineSize) & (kNumLines - 1)];
      const uint32_t head = std::min(byte_count, kLineSize - line_offset);
      const auto resident = [&](const Line &line, uint64_t address) {
        return line.valid && line.translated && line.addr == address &&
               line.cache_namespace == access.cache_namespace();
      };
      if (resident(first, base)) {
        // Only the matching live snapshot owns the counters referenced by the
        // copied grant. Check its unique identity before touching those counters.
        const uint32_t bytes = head == byte_count ? kLineSize : 2 * kLineSize;
        if (code_grant_.identity() != access.cache_identity() || !code_grant_.covers(base, bytes))
          code_grant_ = access.local_grant_view(base, bytes);
        const auto &second = lines_[((base / kLineSize) + 1) & (kNumLines - 1)];
        if (code_grant_ && first.admission_generation == code_grant_.generation() &&
            (head == byte_count || (resident(second, base + kLineSize) &&
                                    second.admission_generation == code_grant_.generation()))) {
          std::memcpy(dst, first.data + line_offset, head);
          if (head != byte_count)
            std::memcpy(dst + head, second.data, byte_count - head);
          return VmAccessOutcome::Complete;
        }
      }
    }
    if (access.execute_admission_is_fixed() && byte_count <= kLineSize - line_offset) {
      // PCI/VFIO bindings have no live execute grant to recheck. A resident
      // line still belongs to its original VM generation and may not survive
      // revocation or an instruction-cache maintenance event.
      if (access.revoked())
        return VmAccessOutcome::Revoked;
      const uint64_t line_address = pc & ~uint64_t{kLineSize - 1};
      const Line &line = lines_[(line_address / kLineSize) & (kNumLines - 1)];
      if (line.valid && line.translated && line.addr == line_address &&
          line.cache_namespace == access.cache_namespace() && line.admission_generation == 0) {
        std::memcpy(dst, line.data + line_offset, byte_count);
        return VmAccessOutcome::Complete;
      }
    }
    const VmAccessAdmission admission = access.admit_access(pc, byte_count, VmAccessKind::Execute);
    if (!admission)
      return admission.outcome;
    uint32_t copied = 0;
    while (copied < byte_count) {
      // A translated hit otherwise bypasses GpuVmAccess entirely and could
      // continue executing cached code after its snapshot was revoked.
      if (access.revoked())
        return VmAccessOutcome::Revoked;
      const uint64_t address = pc + copied;
      const uint64_t line_address = address & ~uint64_t{kLineSize - 1};
      const uint32_t line_offset = static_cast<uint32_t>(address) & (kLineSize - 1);
      const uint32_t chunk = std::min(byte_count - copied, kLineSize - line_offset);
      const VmAccessAdmission chunk_admission =
          copied == 0 && chunk == byte_count
              ? admission
              : access.admit_access(address, chunk, VmAccessKind::Execute);
      if (!chunk_admission)
        return chunk_admission.outcome;
      const VmAccessAdmission line_admission =
          access.admit_access(line_address, kLineSize, VmAccessKind::Execute,
                              /*report_fault=*/false);
      const bool cache_line = line_admission && line_admission.cacheable &&
                              line_admission.generation == chunk_admission.generation;
      const VmCacheNamespace cache_namespace = access.cache_namespace();
      Line &line = lines_[(line_address / kLineSize) & (kNumLines - 1)];

      if (cache_line && line.valid && line.translated && line.addr == line_address &&
          line.cache_namespace == cache_namespace &&
          line.admission_generation == chunk_admission.generation) {
        std::memcpy(dst + copied, line.data + line_offset, chunk);
      } else if (!cache_line ||
                 access.query_access(line_address, kLineSize, VmAccessKind::Execute) !=
                     VmAccessOutcome::Complete) {
        const VmAccessOutcome outcome = access.read(
            address, std::span<std::byte>(reinterpret_cast<std::byte *>(dst + copied), chunk),
            VmAccessKind::Execute);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      } else {
        const uint8_t *line_data = nullptr;
        const VmAccessOutcome outcome =
            line_for(access, address, chunk_admission.generation, line_data);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
        std::memcpy(dst + copied, line_data + line_offset, chunk);
      }
      copied += chunk;
    }
    return VmAccessOutcome::Complete;
  }

  /// @brief Bind instruction invalidation to the same domain as the CU's L2.
  void set_l2(const L2Cache *l2) {
    coherence_ = l2 ? l2->coherence_domain() : std::make_shared<DeviceCacheCoherence>();
    invalidate_all();
  }

  const std::shared_ptr<DeviceCacheCoherence> &coherence_domain() const { return coherence_; }

  /// @brief Discard every cached line (s_icache_inv).
  void invalidate_all() {
    ++epoch_;
    if (has_cached_lines_) {
      for (Line &line : lines_)
        line.valid = false;
      has_cached_lines_ = false;
    }
    coherence_epoch_ = coherence_->current_instruction_epoch();
  }

  /// @brief Discard translated instructions belonging to one VM root.
  /// @details Completion advances the public epoch even if no matching line
  /// is resident. Other namespaces and legacy lines stay cached. A pending
  /// device-wide instruction invalidation still discards every line first.
  void invalidate_for_completion(const VmCacheNamespace &cache_namespace) {
    synchronize_coherence_epoch();
    ++epoch_;
    if (!has_cached_lines_)
      return;
    bool retained_line = false;
    for (Line &line : lines_) {
      if (!line.valid)
        continue;
      if (line.translated && line.cache_namespace == cache_namespace)
        line.valid = false;
      else
        retained_line = true;
    }
    has_cached_lines_ = retained_line;
  }

  uint64_t epoch() const { return epoch_; }

  /// @brief Read an already cached instruction word without fetching or filling.
  /// @details This is only a hint for choosing an execution path. The caller
  /// must still perform ordinary fetchability and debugger-coherence checks.
  bool peek_word(uint64_t pc, uint32_t vmid, uint32_t &word) const {
    const uint64_t addr = pc & ~uint64_t{kLineSize - 1};
    const auto &line = lines_[(addr / kLineSize) & (kNumLines - 1)];
    if ((pc & 3) || !line.valid || line.translated || line.addr != addr || line.vmid != vmid)
      return false;
    std::memcpy(&word, line.data + (pc & (kLineSize - 1)), sizeof(word));
    return true;
  }

  /// @brief Read a resident hint in the retained access's translated namespace.
  /// @details Like the VMID overload, this only selects an execution path and
  /// may return a stale word. Ordinary fetch must validate execution access,
  /// including revocation, and synchronize cache and debugger coherence.
  bool peek_word(const GpuVmAccess &access, uint64_t pc, uint32_t &word) const {
    const uint64_t addr = pc & ~uint64_t{kLineSize - 1};
    const auto &line = lines_[(addr / kLineSize) & (kNumLines - 1)];
    if ((pc & 3) || !line.valid || !line.translated || line.addr != addr ||
        line.cache_namespace != access.cache_namespace())
      return false;
    std::memcpy(&word, line.data + (pc & (kLineSize - 1)), sizeof(word));
    return true;
  }

private:
  void synchronize_coherence_epoch() {
    const uint64_t current_epoch = coherence_->current_instruction_epoch();
    if (coherence_epoch_ != current_epoch)
      invalidate_all();
  }

  struct Line {
    uint64_t addr = 0;
    uint64_t admission_generation = 0;
    uint32_t vmid = 0;
    VmCacheNamespace cache_namespace;
    bool translated = false;
    bool valid = false;
    uint8_t data[kLineSize] = {};
  };

  const uint8_t *line_for(const GpuMemory &memory, uint64_t addr) {
    const uint64_t line_addr = addr & ~uint64_t{kLineSize - 1};
    Line &line = lines_[(line_addr / kLineSize) & (kNumLines - 1)];
    if (line.valid && !line.translated && line.addr == line_addr)
      return line.data;

    memory.read_block(line_addr, std::span<uint8_t>(line.data, kLineSize));
    line.addr = line_addr;
    line.vmid = 0;
    line.cache_namespace = {};
    line.admission_generation = 0;
    line.translated = false;
    line.valid = true;
    has_cached_lines_ = true;
    return line.data;
  }

  VmAccessOutcome line_for(const GpuVmAccess &access, uint64_t addr, uint64_t admission_generation,
                           const uint8_t *&data) {
    const uint64_t line_addr = addr & ~uint64_t{kLineSize - 1};
    const VmCacheNamespace cache_namespace = access.cache_namespace();
    Line &line = lines_[(line_addr / kLineSize) & (kNumLines - 1)];
    if (line.valid && line.translated && line.addr == line_addr &&
        line.cache_namespace == cache_namespace &&
        line.admission_generation == admission_generation) {
      data = line.data;
      return VmAccessOutcome::Complete;
    }

    const VmAccessOutcome outcome = access.read(
        line_addr, std::span<std::byte>(reinterpret_cast<std::byte *>(line.data), kLineSize),
        VmAccessKind::Execute);
    if (outcome != VmAccessOutcome::Complete) {
      line.valid = false;
      data = nullptr;
      return outcome;
    }
    line.addr = line_addr;
    line.vmid = 0;
    line.cache_namespace = cache_namespace;
    line.admission_generation = admission_generation;
    line.translated = true;
    line.valid = true;
    has_cached_lines_ = true;
    data = line.data;
    return VmAccessOutcome::Complete;
  }

  VmLocalGrantView code_grant_;
  Line lines_[kNumLines];
  uint64_t epoch_ = 0;
  std::shared_ptr<DeviceCacheCoherence> coherence_;
  uint64_t coherence_epoch_ = 0;
  // Only successful fills make a line valid. Repeated maintenance still
  // advances the public epoch, but need not revisit already invalid lines.
  bool has_cached_lines_ = false;
};

} // namespace amdgpu
} // namespace rocjitsu
