// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/legacy_address_space.h"

#include <linux/fs.h>
#include <sys/ioctl.h>

// Linux 6.11 added this ioctl. Use the published prefix of its UAPI when
// compiling against older headers; older running kernels use the maps parser.
#ifndef PROCMAP_QUERY
struct procmap_query {
  uint64_t size, query_flags, query_addr;
  uint64_t vma_start, vma_end, vma_flags, vma_page_size, vma_offset, inode;
  uint32_t dev_major, dev_minor, vma_name_size, build_id_size;
  uint64_t vma_name_addr, build_id_addr;
};
#define PROCMAP_QUERY _IOWR('f', 17, struct procmap_query)
constexpr uint64_t PROCMAP_QUERY_VMA_READABLE = 1;
constexpr uint64_t PROCMAP_QUERY_VMA_WRITABLE = 2;
#endif

namespace rocjitsu::amdgpu {

bool LegacyAddressSpace::local_ingress_mapped(
    std::span<const MemoryIngressSpan> spans, uint64_t begin, uint64_t end, uint32_t vmid,
    const std::shared_ptr<util::DistributedSharedMutex> &expected,
    const std::shared_ptr<LegacyPageTableCacheState> &policies) const {
  // The general lookup drops its page-table lease around allocator metadata queries.
  if constexpr (kCheckAsanBacking)
    return false;
  // Cached coverage identifies mapped GPU pages. Only the driver owns stable
  // write permissions. Application pages need a fresh check on each ingress.
  struct Coverage {
    uint64_t memory = 0, generation = 0, begin = 0, end = 0;
    uint32_t vmid = 0;
    std::shared_ptr<LegacyPageTableCacheState> policies;
    bool driver_owned = false;
    uintptr_t host_begin = 0;
  };
  static thread_local std::array<Coverage, 8> runs;
  const bool writing =
      std::ranges::any_of(spans, [](const auto &s) { return s.access != VmAccessKind::Read; });
  // A contiguous application mapping still needs a fresh kernel permission
  // query for each ingress. Only reuse its numeric GPU-to-host offset.
  const auto admitted = [&](const Coverage &run) {
    if (!writing || run.driver_owned)
      return true;
    if (!run.host_begin)
      return false;
    const auto mapping = rocjitsu::host_mapping_lock().lock_shared();
    const auto host = run.host_begin + (begin - run.begin);
    uintptr_t vma_end = 0;
    return host_page_writability(reinterpret_cast<const uint8_t *>(host), &vma_end) ==
               PageWritability::Writable &&
           vma_end >= host && end - begin <= vma_end - host &&
           policies->generation() == run.generation;
  };
  const auto generation = policies->generation();
  for (const auto &run : runs)
    if (run.memory == instance_id_ && run.vmid == vmid && run.policies == policies &&
        run.generation == generation && run.begin <= begin && end <= run.end && admitted(run))
      return true;

  auto request = acquire_page_table_request(vmid, expected);
  if (!request.owns_lock())
    return false;
  std::shared_lock page_table_lock(*request.page_table_mutex_);
  // A replacement registration can share the request mutex but provide
  // different copied-policy state. Only publish coverage under its actual epoch.
  if (request.policies_ == policies.get()) {
    constexpr uint64_t max_pages = 128;
    const uint64_t first = begin >> PAGE_SHIFT;
    uint64_t page = first;
    bool driver_owned = true;
    uintptr_t host_begin = 0;
    for (; page - first < max_pages && page < (UINT64_MAX >> PAGE_SHIFT); ++page) {
      const auto entry = request.page_table_->find(page);
      if (entry == request.page_table_->end() || entry->second.host_extents.size() != 1)
        break;
      const auto &extent = entry->second.host_extents.front();
      if (!extent.host_ptr || extent.gpu_page_offset != 0 || extent.host_backed_bytes != PAGE_SIZE)
        break;
      const uintptr_t host = reinterpret_cast<uintptr_t>(extent.host_ptr);
      if (page == first)
        host_begin = host;
      else if (host - host_begin != (page - first) * PAGE_SIZE)
        host_begin = 0;
      driver_owned &= extent.owner != LegacyHostExtentOwner::Application;
    }
    if (page != first) {
      static thread_local unsigned next = 0;
      auto &run = runs[next++ % runs.size()];
      run = {instance_id_, policies->generation(), first << PAGE_SHIFT, page << PAGE_SHIFT, vmid,
             policies,     driver_owned,           host_begin};
      if (end <= run.end && admitted(run))
        return true;
    }
  }
  // A wide bounding span can include inactive lanes. If copied PTE coverage
  // did not prove it, let exact ingress inspect only the requested stores.
  if (writing && spans.size() == 1 && spans.front().size > PAGE_SIZE)
    return false;
  const auto mapping_lease = rocjitsu::host_mapping_lock().lock_shared();
  uintptr_t writable_page = UINTPTR_MAX;
  for (const auto &span : spans) {
    if (!for_each_page_chunk_until(
            span.address, span.size, [&](uint64_t address, size_t, size_t size) {
              const auto entry = request.page_table_->find(address >> PAGE_SHIFT);
              if (entry == request.page_table_->end() || entry->second.host_extents.size() != 1)
                return false;
              const size_t offset = address & PAGE_MASK;
              const auto *extent = host_extent_at(entry->second, offset);
              if (!extent || !extent->host_ptr ||
                  size > extent->host_backed_bytes - (offset - extent->gpu_page_offset))
                return false;
              if (span.access == VmAccessKind::Read ||
                  extent->owner != LegacyHostExtentOwner::Application)
                return true;
              auto *target = extent->host_ptr + (offset - extent->gpu_page_offset);
              const uintptr_t first = reinterpret_cast<uintptr_t>(target) & ~uintptr_t{PAGE_MASK};
              const uintptr_t last =
                  reinterpret_cast<uintptr_t>(target + size - 1) & ~uintptr_t{PAGE_MASK};
              if ((first != writable_page || last != writable_page) &&
                  host_range_writability(target, size) != PageWritability::Writable)
                return false;
              writable_page = last;
              return true;
            }))
      return false;
  }
  return true;
}

PageWritability LegacyAddressSpace::host_page_writability(const uint8_t *page, uintptr_t *vma_end) {
  if (page == nullptr)
    return PageWritability::Inaccessible;
  if (proc_maps_open_failure_for_test_.load(std::memory_order_relaxed))
    return PageWritability::Indeterminate;
  const auto address = reinterpret_cast<uintptr_t>(page);
  const long fd = ::syscall(SYS_openat, AT_FDCWD, "/proc/self/maps", O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return PageWritability::Indeterminate;

  // Ask the kernel for the VMA covering this page without walking unrelated mappings.
  struct procmap_query query {};
  query.size = sizeof(query);
  query.query_addr = address;
  if (::syscall(SYS_ioctl, fd, PROCMAP_QUERY, &query) == 0) {
    ::syscall(SYS_close, fd);
    if (vma_end)
      *vma_end = query.vma_end;
    return (query.vma_flags & PROCMAP_QUERY_VMA_WRITABLE)   ? PageWritability::Writable
           : (query.vma_flags & PROCMAP_QUERY_VMA_READABLE) ? PageWritability::ReadOnly
                                                            : PageWritability::Inaccessible;
  }
  if (errno == ENOENT) {
    ::syscall(SYS_close, fd);
    return PageWritability::Inaccessible;
  }

  // Parsed incrementally: the table can be large, and a line may straddle
  // reads, so keep any partial tail and prepend it to the next chunk.
  std::string pending;
  char chunk[8192];
  PageWritability answer = PageWritability::Indeterminate;
  for (bool reading = true; reading;) {
    const long got = ::syscall(SYS_read, static_cast<int>(fd), chunk, sizeof(chunk));
    if (got < 0) {
      // A signal arriving mid-read says nothing about the mapping.
      if (errno == EINTR)
        continue;
      break;
    }
    if (got == 0) {
      // Walked the whole table without covering the address: nothing is
      // mapped there, which is a definite answer rather than a failure.
      answer = PageWritability::Inaccessible;
      break;
    }
    pending.append(chunk, static_cast<size_t>(got));
    size_t line_begin = 0;
    for (size_t newline = pending.find('\n', line_begin); newline != std::string::npos;
         newline = pending.find('\n', line_begin)) {
      const std::string line(pending, line_begin, newline - line_begin);
      line_begin = newline + 1;
      uintptr_t begin = 0;
      uintptr_t end = 0;
      char permissions[5] = {};
      if (std::sscanf(line.c_str(), "%zx-%zx %4s", &begin, &end, permissions) != 3)
        continue;
      if (address < begin || address >= end)
        continue;
      // A mapping that permits neither read nor write -- a PROT_NONE
      // reservation, the shape the runtime leaves behind around an aperture
      // -- is absent as far as the GPU is concerned, not merely protected.
      if (vma_end)
        *vma_end = end;
      answer = permissions[1] == 'w'   ? PageWritability::Writable
               : permissions[0] == 'r' ? PageWritability::ReadOnly
                                       : PageWritability::Inaccessible;
      reading = false;
      break;
    }
    pending.erase(0, line_begin);
  }
  ::syscall(SYS_close, static_cast<int>(fd));
  return answer;
}

} // namespace rocjitsu::amdgpu
