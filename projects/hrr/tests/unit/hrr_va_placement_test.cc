/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR capture-address placement
 * @{
 * @ingroup HRRTest
 * CPU-only tests for capture-address placement: which recorded calls claim a
 * range, how the ranges are rounded and merged, what is left out, how the
 * placeholders are held, and the bookkeeping that needs no GPU. The GPU half
 * is covered by tests/integration/hrr_va_placement_test.cc.
 */

#include "hrr_test_common.hh"
#include "hrr_va_placement.h"
#include "hrr_event_order.h"
#include "hrr/hrr_api_args.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
// The placeholder and stderr tests below use mmap and dup2.
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using hrr::VaRange;

namespace {
constexpr uint64_t P = hrr::kPlacePage;
constexpr uint64_t B = 0x7f0000000000ull;  // a plausible device VA

// An Event whose raw payload is `args`, typed as `type`.
template <class Args>
hrr::Event place_event(uint16_t type, Args args) {
  args.hdr.event_type     = type;
  args.hdr.payload_length = sizeof(Args);
  hrr::Event ev;
  ev.raw_payload.resize(sizeof(Args));
  std::memcpy(ev.raw_payload.data(), &args, sizeof(Args));
  return ev;
}

hrr::Event ev_malloc(uint64_t ptr, uint64_t size, int32_t ret = 0) {
  hrr_args_hipMalloc a{};
  a.ret = ret; a.ptr = ptr; a.size = size;
  return place_event(HRR_API_HIPMALLOC, a);
}

hrr::Event ev_ext_malloc(uint64_t ptr, uint64_t size, uint32_t flags) {
  hrr_args_hipExtMallocWithFlags a{};
  a.ptr = ptr; a.sizeBytes = size; a.flags = flags;
  return place_event(HRR_API_HIPEXTMALLOCWITHFLAGS, a);
}

hrr::Event ev_malloc_async(uint64_t ptr, uint64_t size) {
  hrr_args_hipMallocAsync a{};
  a.dev_ptr = ptr; a.size = size;
  return place_event(HRR_API_HIPMALLOCASYNC, a);
}

hrr::Event ev_malloc_pool(uint64_t ptr, uint64_t size) {
  hrr_args_hipMallocFromPoolAsync a{};
  a.dev_ptr = ptr; a.size = size;
  return place_event(HRR_API_HIPMALLOCFROMPOOLASYNC, a);
}

hrr::Event ev_reserve(uint64_t ptr, uint64_t size) {
  hrr_args_hipMemAddressReserve a{};
  a.ptr = ptr; a.size = size;
  return place_event(HRR_API_HIPMEMADDRESSRESERVE, a);
}

hrr::Event ev_ipc_export(uint64_t ptr, int32_t ret = 0) {
  hrr_args_hipIpcGetMemHandle a{};
  a.ret = ret; a.devPtr = ptr;
  return place_event(HRR_API_HIPIPCGETMEMHANDLE, a);
}

hrr::Event ev_pool_export(uint64_t ptr) {
  hrr_args_hipMemPoolExportPointer a{};
  a.dev_ptr = ptr;
  return place_event(HRR_API_HIPMEMPOOLEXPORTPOINTER, a);
}

using Ranges = std::vector<VaRange>;
}  // namespace

HRR_TEST_CASE(Unit_HRR_VaPlacement_Rounding) {
  REQUIRE(hrr::va_floor(B + 1, P) == B);
  REQUIRE(hrr::va_floor(B, P) == B);
  REQUIRE(hrr::va_ceil(B + 1, P) == B + P);
  REQUIRE(hrr::va_ceil(B, P) == B);
  // Rounding up past the top of the address space reports 0, not a wrap.
  REQUIRE(hrr::va_ceil(UINT64_MAX - 5, P) == 0);
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_RoundMerge) {
  SECTION("a range is rounded out to whole pages") {
    REQUIRE(hrr::va_round_merge({{B + 0x10, B + 0x20}}, P) == Ranges{{B, B + P}});
    REQUIRE(hrr::va_round_merge({{B + 0x10, B + P + 1}}, P) == Ranges{{B, B + 2 * P}});
  }
  SECTION("ranges that overlap or touch merge, and gaps stay") {
    const Ranges got = hrr::va_round_merge(
        {{B + 4 * P, B + 5 * P},   // after a one-page gap
         {B, B + P},
         {B + P, B + 2 * P},       // touches the first
         {B + 0x80, B + 0x100},    // inside the first
         {B + 2 * P + 8, B + 3 * P}},  // shares a page boundary once rounded
        P);
    REQUIRE(got == Ranges{{B, B + 3 * P}, {B + 4 * P, B + 5 * P}});
  }
  SECTION("two allocations in one page become one range") {
    REQUIRE(hrr::va_round_merge({{B + 0x100, B + 0x108}, {B + 0x200, B + 0x208}}, P) ==
            Ranges{{B, B + P}});
  }
  SECTION("empty, wrapping and out-of-range ranges are dropped") {
    const Ranges got = hrr::va_round_merge(
        {{B, B},                                    // empty
         {B + P, B},                                // reversed
         {0x1000, 0x2000},                          // under the mmap floor
         {hrr::kPlaceVaLimit - P, hrr::kPlaceVaLimit + P},  // crosses the limit
         {UINT64_MAX - 8, UINT64_MAX},              // rounds past the top
         {B, B + P}},
        P);
    REQUIRE(got == Ranges{{B, B + P}});
  }
  SECTION("a coarser granularity rounds to it") {
    const uint64_t G = 2ull << 20;
    REQUIRE(hrr::va_round_merge({{B + P, B + 2 * P}}, G) == Ranges{{B, B + G}});
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_Subtract) {
  const Ranges from = {{B, B + 8 * P}, {B + 10 * P, B + 12 * P}};
  SECTION("a hole in the middle splits the range") {
    REQUIRE(hrr::va_subtract(from, {{B + 2 * P, B + 3 * P}}) ==
            Ranges{{B, B + 2 * P}, {B + 3 * P, B + 8 * P}, {B + 10 * P, B + 12 * P}});
  }
  SECTION("edges and a range removed whole") {
    REQUIRE(hrr::va_subtract(from, {{B - P, B + P}, {B + 7 * P, B + 12 * P}}) ==
            Ranges{{B + P, B + 7 * P}});
  }
  SECTION("several holes in one range") {
    REQUIRE(hrr::va_subtract(from, {{B + P, B + 2 * P}, {B + 4 * P, B + 5 * P}}) ==
            Ranges{{B, B + P}, {B + 2 * P, B + 4 * P}, {B + 5 * P, B + 8 * P},
                   {B + 10 * P, B + 12 * P}});
  }
  SECTION("nothing to remove") {
    REQUIRE(hrr::va_subtract(from, {}) == from);
    REQUIRE(hrr::va_subtract(from, {{B + 8 * P, B + 10 * P}}) == from);
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_FindContaining) {
  const Ranges r = {{B, B + 2 * P}, {B + 4 * P, B + 6 * P}};
  const VaRange* hit = hrr::va_find_containing(r, B + P, B + 2 * P);
  REQUIRE(hit != nullptr);
  REQUIRE(*hit == VaRange{B, B + 2 * P});
  hit = hrr::va_find_containing(r, B + 4 * P, B + 6 * P);
  REQUIRE(hit != nullptr);
  REQUIRE(hit->base == B + 4 * P);
  // Straddling the gap, before the first and after the last range all miss.
  REQUIRE(hrr::va_find_containing(r, B + P, B + 5 * P) == nullptr);
  REQUIRE(hrr::va_find_containing(r, B - P, B) == nullptr);
  REQUIRE(hrr::va_find_containing(r, B + 6 * P, B + 7 * P) == nullptr);
  REQUIRE(hrr::va_find_containing({}, B, B + P) == nullptr);
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_EventRange) {
  VaRange r;
  REQUIRE(hrr::placement_event_range(ev_malloc(B, 100), &r) == hrr::PlaceKind::Alloc);
  REQUIRE(r == VaRange{B, B + 100});
  REQUIRE(hrr::placement_event_range(ev_ext_malloc(B, 64, 0), &r) == hrr::PlaceKind::Alloc);
  REQUIRE(hrr::placement_event_range(ev_malloc_async(B, 64), &r) == hrr::PlaceKind::Alloc);
  REQUIRE(hrr::placement_event_range(ev_malloc_pool(B, 64), &r) == hrr::PlaceKind::Alloc);
  REQUIRE(hrr::placement_event_range(ev_reserve(B, 2 * P), &r) == hrr::PlaceKind::Vmm);
  REQUIRE(r == VaRange{B, B + 2 * P});

  // A failed call returned no address.
  REQUIRE(hrr::placement_event_range(ev_malloc(B, 100, /*ret=*/2), &r) ==
          hrr::PlaceKind::None);
  // Fine-grained and the other flagged kinds have no VMM equivalent.
  REQUIRE(hrr::placement_event_range(ev_ext_malloc(B, 64, 1), &r) == hrr::PlaceKind::None);
  // Null, empty and wrapping ranges.
  REQUIRE(hrr::placement_event_range(ev_malloc(0, 100), &r) == hrr::PlaceKind::None);
  REQUIRE(hrr::placement_event_range(ev_malloc(B, 0), &r) == hrr::PlaceKind::None);
  REQUIRE(hrr::placement_event_range(ev_malloc(UINT64_MAX - 4, 100), &r) ==
          hrr::PlaceKind::None);
  // A payload cut short by a crash is not read past its end.
  hrr::Event cut = ev_malloc(B, 100);
  cut.raw_payload.resize(cut.raw_payload.size() - 4);
  REQUIRE(hrr::placement_event_range(cut, &r) == hrr::PlaceKind::None);
  // Anything else claims nothing.
  hrr_args_hipMemAddressFree f{};
  f.devPtr = B; f.size = P;
  REQUIRE(hrr::placement_event_range(place_event(HRR_API_HIPMEMADDRESSFREE, f), &r) ==
          hrr::PlaceKind::None);
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_Plan) {
  // Event is move-only, so the list is built call by call.
  std::vector<hrr::Event> events;
  events.push_back(ev_malloc(B, P));  // alloc
  events.push_back(ev_malloc(B + 0x1000 + 8, 8));  // shares page B+P with the next
  events.push_back(ev_malloc(B + 0x1000 + 0x100, 8));
  events.push_back(ev_malloc(B + 16 * P, P, /*ret=*/2));  // failed: no range
  events.push_back(ev_ext_malloc(B + 20 * P, P, 1));  // fine-grained: not placed
  events.push_back(ev_malloc_async(B + 4 * P, P));
  events.push_back(ev_malloc_pool(B + 6 * P, 2 * P));
  events.push_back(ev_reserve(B + 32 * P, 8 * P));  // VMM
  events.push_back(ev_reserve(B + 6 * P, 4 * P));  // overlaps the pool allocation
  const std::vector<VaRange> segments = {{B + 64 * P, B + 66 * P}};

  SECTION("alloc, vmm and segments are separated and merged") {
    const hrr::PlacementPlan p = hrr::plan_placement(events, segments, {});
    REQUIRE(p.alloc_events == 5);
    REQUIRE(p.vmm_events == 2);
    REQUIRE(p.segments == 1);
    REQUIRE(p.denied.empty());
    REQUIRE(p.alloc == Ranges{{B, B + 2 * P},
                              {B + 4 * P, B + 5 * P},
                              {B + 6 * P, B + 8 * P},
                              {B + 64 * P, B + 66 * P}});
    // The reservation over the pool allocation keeps only what the
    // allocation does not cover.
    REQUIRE(p.vmm == Ranges{{B + 8 * P, B + 10 * P}, {B + 32 * P, B + 40 * P}});
    REQUIRE(p.exported.empty());
  }

  SECTION("an allocation exported to another process is left to the runtime") {
    // The IPC export names an address inside the hipMalloc at B; the pool
    // export names the pool allocation at B+6P. A failed export and an export
    // of a reservation take out nothing.
    events.push_back(ev_ipc_export(B + 16));
    events.push_back(ev_pool_export(B + 6 * P));
    events.push_back(ev_ipc_export(B + 4 * P, /*ret=*/1));
    events.push_back(ev_ipc_export(B + 33 * P));
    const hrr::PlacementPlan p = hrr::plan_placement(events, segments, {});
    REQUIRE(p.exported == Ranges{{B, B + P}, {B + 6 * P, B + 8 * P}});
    REQUIRE(p.alloc == Ranges{{B + P, B + 2 * P},
                              {B + 4 * P, B + 5 * P},
                              {B + 64 * P, B + 66 * P}});
    REQUIRE(p.vmm == Ranges{{B + 6 * P, B + 10 * P}, {B + 32 * P, B + 40 * P}});
    REQUIRE(hrr::va_overlaps(p.exported, B + 0x80, B + 0x90));
    REQUIRE_FALSE(hrr::va_overlaps(p.exported, B + P, B + 2 * P));
  }

  SECTION("a denied address takes out only its own allocation") {
    const hrr::PlacementPlan p = hrr::plan_placement(events, segments, {B + 4 * P + 16});
    REQUIRE(p.denied == Ranges{{B + 4 * P, B + 5 * P}});
    REQUIRE(p.alloc == Ranges{{B, B + 2 * P},
                              {B + 6 * P, B + 8 * P},
                              {B + 64 * P, B + 66 * P}});
  }

  SECTION("denying a reservation or a segment works the same way") {
    const hrr::PlacementPlan p =
        hrr::plan_placement(events, segments, {B + 33 * P, B + 65 * P});
    REQUIRE(p.denied == Ranges{{B + 32 * P, B + 40 * P}, {B + 64 * P, B + 66 * P}});
    REQUIRE(p.vmm == Ranges{{B + 8 * P, B + 10 * P}});
    REQUIRE(p.alloc.back() == VaRange{B + 6 * P, B + 8 * P});
  }

  SECTION("an address in no recorded range denies nothing") {
    const hrr::PlacementPlan p = hrr::plan_placement(events, segments, {B + 100 * P});
    REQUIRE(p.denied.empty());
    REQUIRE(p.alloc.size() == 4);
  }

  SECTION("an empty recording plans nothing") {
    const hrr::PlacementPlan p = hrr::plan_placement(std::vector<hrr::Event>{}, {}, {B});
    REQUIRE(p.alloc.empty());
    REQUIRE(p.vmm.empty());
    REQUIRE(p.denied.empty());
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_ParseDeny) {
  REQUIRE(hrr::parse_place_deny(nullptr).empty());
  REQUIRE(hrr::parse_place_deny("").empty());
  REQUIRE(hrr::parse_place_deny("0x7f0000001000,4096 0x10") ==
          std::vector<uint64_t>{0x7f0000001000ull, 4096, 0x10});
  REQUIRE(hrr::parse_place_deny("0x1000, ,0x2000") == std::vector<uint64_t>{0x1000, 0x2000});
  // Zero is no address, and parsing stops at the first thing that is not one.
  REQUIRE(hrr::parse_place_deny("0,0x3000,junk,0x4000") == std::vector<uint64_t>{0x3000});
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_AllocsRunInCaptureOrder) {
  // A placed allocation takes the address its recorded call returned. If two
  // threads replay out of capture order, one can take the other's range, so
  // every call that claims or releases a range has to be ordered.
  for (uint16_t api : {HRR_API_HIPMALLOC, HRR_API_HIPEXTMALLOCWITHFLAGS,
                       HRR_API_HIPMALLOCASYNC, HRR_API_HIPMALLOCFROMPOOLASYNC,
                       HRR_API_HIPMALLOCMANAGED, HRR_API_HIPMEMADDRESSRESERVE,
                       HRR_API_HIPFREE, HRR_API_HIPFREEASYNC,
                       HRR_API_HIPMEMADDRESSFREE}) {
    INFO(hrr_api_names[api]);
    REQUIRE(hrr_needs_ordering(api));
  }
  REQUIRE_FALSE(hrr_needs_ordering(HRR_API_HIPLAUNCHKERNEL));
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_CapturesOpenAndCloseInOrder) {
  // Whether a capture is open decides whether a placed free unmaps now or
  // waits. A free on one thread has to see the capture another thread opened
  // or closed before it, whichever API did that.
  for (uint16_t api : {HRR_API_HIPSTREAMBEGINCAPTURE, HRR_API_HIPSTREAMENDCAPTURE,
                       HRR_API_HIPSTREAMBEGINCAPTURETOGRAPH,
                       HRR_API_HIPSTREAMBEGINCAPTURE_SPT,
                       HRR_API_HIPSTREAMENDCAPTURE_SPT,
                       HRR_API_HIPSTREAMDESTROY}) {
    INFO(hrr_api_names[api]);
    REQUIRE(hrr_needs_ordering(api));
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_PlanSegmentsMinusReservations) {
  // The region sidecar declares a segment for every allocation it saw, so a
  // hipMemAddressReserve range comes back as a segment too. It belongs to the
  // replayed reserve: the segment loses it, not the reservation.
  std::vector<hrr::Event> events;
  events.push_back(ev_malloc(B, P));
  events.push_back(ev_reserve(B + 32 * P, 8 * P));
  const std::vector<VaRange> segments = {{B, B + P},
                                         {B + 32 * P, B + 40 * P},
                                         {B + 64 * P, B + 66 * P}};
  const hrr::PlacementPlan p = hrr::plan_placement(events, segments, {});
  REQUIRE(p.vmm == Ranges{{B + 32 * P, B + 40 * P}});
  REQUIRE(p.alloc == Ranges{{B, B + P}, {B + 64 * P, B + 66 * P}});
  // Each allocation and segment keeps its own range for reserve() to fall
  // back on; the reservation's segment is not one of them.
  REQUIRE(p.pieces == Ranges{{B, B + P}, {B, B + P}, {B + 64 * P, B + 66 * P}});
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_ReservePieces) {
  const uint64_t G = 16 * P;
  const Ranges pieces = {{B, B + P},              // granule 0
                         {B + 2 * P, B + 3 * P},  // granule 0 too: shares it
                         {B + G, B + G + P},      // granule 1: touches, stays apart
                         {B + 4 * G, B + 5 * G}}; // outside [b, e)
  SECTION("pieces in one granule share it, touching ones stay apart") {
    REQUIRE(hrr::va_reserve_pieces(pieces, B, B + 2 * G, G) ==
            Ranges{{B, B + G}, {B + G, B + 2 * G}});
  }
  SECTION("a piece is clipped to the range that failed") {
    REQUIRE(hrr::va_reserve_pieces(pieces, B, B + G + 4 * P, G) ==
            Ranges{{B, B + G}, {B + G, B + G + 4 * P}});
  }
  SECTION("nothing in the range, nothing to reserve") {
    REQUIRE(hrr::va_reserve_pieces(pieces, B + 2 * G, B + 3 * G, G).empty());
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_ParseProcMaps) {
  const std::string maps =
      "7f0000003000-7f0000004000 r--p 00000000 00:00 0\n"
      "55d0a0000000-55d0a0021000 rw-p 00000000 00:00 0   [heap]\n"
      "7f0000001000-7f0000003000 r-xp 00001000 08:01 42  /usr/lib/libx.so\n"
      "garbage line\n"
      "ffffffffff600000-ffffffffff601000 --xp 00000000 00:00 0 [vsyscall]";
  REQUIRE(hrr::parse_proc_maps(maps) ==
          Ranges{{0x55d0a0000000ull, 0x55d0a0021000ull},
                 {0x7f0000001000ull, 0x7f0000004000ull},
                 {0xffffffffff600000ull, 0xffffffffff601000ull}});
  REQUIRE(hrr::parse_proc_maps("").empty());
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_Unplaced) {
  hrr::Unplaced u;
  SECTION("managed memory is reported with its range") {
    hrr_args_hipMallocManaged a{};
    a.dev_ptr = B; a.size = 3 * P;
    REQUIRE(hrr::placement_unplaced(place_event(HRR_API_HIPMALLOCMANAGED, a), &u));
    REQUIRE(u.rec == B);
    REQUIRE(u.size == 3 * P);
    REQUIRE(std::string(u.api) == "hipMallocManaged");
    REQUIRE(u.why.find("managed") != std::string::npos);
    a.ret = 2;  // a failed call allocated nothing
    REQUIRE_FALSE(hrr::placement_unplaced(place_event(HRR_API_HIPMALLOCMANAGED, a), &u));
  }
  SECTION("hipExtMallocWithFlags only with a flag set") {
    REQUIRE(hrr::placement_unplaced(ev_ext_malloc(B, P, 1), &u));
    REQUIRE(u.why.find("0x1") != std::string::npos);
    REQUIRE_FALSE(hrr::placement_unplaced(ev_ext_malloc(B, P, 0), &u));
  }
  SECTION("pitched memory") {
    hrr_args_hipMallocPitch a{};
    a.ptr = B; a.pitch = 512; a.width = 500; a.height = 8;
    REQUIRE(hrr::placement_unplaced(place_event(HRR_API_HIPMALLOCPITCH, a), &u));
    REQUIRE(u.rec == B);
    REQUIRE(u.size == 512 * 8);
  }
  SECTION("graph memory nodes") {
    hrr_args_hipGraphAddMemAllocNode a{};
    hipMemAllocNodeParams np{};
    np.dptr = reinterpret_cast<void*>(B);
    np.bytesize = 2 * P;
    std::memcpy(a.pNodeParams_bytes, &np, std::min(sizeof(np), sizeof(a.pNodeParams_bytes)));
    a.pNodeParams_present = 1;
    REQUIRE(hrr::placement_unplaced(place_event(HRR_API_HIPGRAPHADDMEMALLOCNODE, a), &u));
    REQUIRE(u.rec == B);
    REQUIRE(u.size == 2 * P);
  }
  SECTION("placed and unrelated calls are not reported") {
    REQUIRE_FALSE(hrr::placement_unplaced(ev_malloc(B, P), &u));
    REQUIRE_FALSE(hrr::placement_unplaced(ev_reserve(B, P), &u));
    hrr::Event cut = ev_malloc(B, P);
    cut.raw_payload.resize(4);
    REQUIRE_FALSE(hrr::placement_unplaced(cut, &u));
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_NeedsPeerAccess) {
  auto set_device = [](int32_t d, int32_t ret = 0) {
    hrr_args_hipSetDevice a{};
    a.deviceId = d; a.ret = ret;
    return place_event(HRR_API_HIPSETDEVICE, a);
  };
  std::vector<hrr::Event> ev;
  ev.push_back(set_device(0));
  ev.push_back(set_device(0));
  ev.push_back(ev_malloc(B, P));
  REQUIRE_FALSE(hrr::placement_needs_peer_access(ev));
  ev.push_back(set_device(1, /*ret=*/101));  // failed: still one device
  REQUIRE_FALSE(hrr::placement_needs_peer_access(ev));
  ev.push_back(set_device(1));
  REQUIRE(hrr::placement_needs_peer_access(ev));

  std::vector<hrr::Event> peer;
  hrr_args_hipMemcpyPeer c{};
  peer.push_back(place_event(HRR_API_HIPMEMCPYPEER, c));
  REQUIRE(hrr::placement_needs_peer_access(peer));
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_MappingOverlap) {
  hrr::PlacedMap m;
  m[B] = {B + 2 * P, B + 16, {}};
  m[B + 4 * P] = {B + 5 * P, B + 4 * P, {}};
  const auto* hit = hrr::va_mapping_overlapping(m, B + P, B + P + 1);
  REQUIRE(hit != nullptr);
  REQUIRE(hit->rec == B + 16);
  hit = hrr::va_mapping_overlapping(m, B + 3 * P, B + 4 * P + 1);
  REQUIRE(hit != nullptr);
  REQUIRE(hit->rec == B + 4 * P);
  // The gap between them, and the edges, touch nothing.
  REQUIRE(hrr::va_mapping_overlapping(m, B + 2 * P, B + 4 * P) == nullptr);
  REQUIRE(hrr::va_mapping_overlapping(m, B - P, B) == nullptr);
  REQUIRE(hrr::va_mapping_overlapping(m, B + 5 * P, B + 6 * P) == nullptr);
  REQUIRE(hrr::va_mapping_overlapping({}, B, B + P) == nullptr);
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_VmmMappingsTracked) {
  // The warm-up reset unmaps every VMM mapping replay still tracks. A recorded
  // hipMemUnmap may span several hipMemMap pieces, or cover only part of one.
  // Each piece it touched must be forgotten, and only the part it covered.
  using M = std::map<uint64_t, size_t>;
  SECTION("one unmap over several pieces drops them all") {
    M m;
    for (int i = 0; i < 4; ++i) hrr::va_track_mapping(m, B + i * P, P);
    hrr::va_untrack_mapping(m, B, 4 * P);
    REQUIRE(m.empty());
  }
  SECTION("unmapping a large mapping piece by piece leaves the rest tracked") {
    M m;
    hrr::va_track_mapping(m, B, 4 * P);
    hrr::va_untrack_mapping(m, B, P);
    REQUIRE(m == M{{B + P, 3 * P}});
    hrr::va_untrack_mapping(m, B + 2 * P, P);
    REQUIRE(m == M{{B + P, P}, {B + 3 * P, P}});
    hrr::va_untrack_mapping(m, B + P, P);
    hrr::va_untrack_mapping(m, B + 3 * P, P);
    REQUIRE(m.empty());
  }
  SECTION("an unmap that straddles two pieces keeps their outer parts") {
    M m;
    hrr::va_track_mapping(m, B, 2 * P);
    hrr::va_track_mapping(m, B + 2 * P, 2 * P);
    hrr::va_untrack_mapping(m, B + P, 2 * P);
    REQUIRE(m == M{{B, P}, {B + 3 * P, P}});
  }
  SECTION("a range outside every mapping changes nothing") {
    M m;
    hrr::va_track_mapping(m, B + 2 * P, P);
    hrr::va_untrack_mapping(m, B, 2 * P);
    hrr::va_untrack_mapping(m, B + 3 * P, P);
    REQUIRE(m == M{{B + 2 * P, P}});
  }
  SECTION("a map over a tracked range keeps the entries disjoint") {
    M m;
    hrr::va_track_mapping(m, B, 4 * P);
    hrr::va_track_mapping(m, B + P, P);
    REQUIRE(m == M{{B, P}, {B + P, P}, {B + 2 * P, 2 * P}});
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_SubtractScalesWithTheAnswer) {
  // Startup holds each planned range minus what /proc/self/maps lists, one
  // range at a time. With a million mappings and fifty thousand ranges, a
  // walk from the first mapping on every call is tens of billions of steps.
  constexpr size_t M = 1000000, R = 50000;
  std::vector<VaRange> maps;
  maps.reserve(M);
  for (size_t i = 0; i < M; ++i) maps.push_back({B + 2 * i * P, B + (2 * i + 1) * P});
  const auto t0 = std::chrono::steady_clock::now();
  size_t pieces = 0;
  for (size_t r = 0; r < R; ++r) {
    // A range over the last mappings: two of them, and the gap between.
    const uint64_t b = B + 2 * (M - 2 - r % 100) * P;
    pieces += hrr::va_subtract({{b, b + 4 * P}}, maps).size();
  }
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  REQUIRE(pieces == 2 * R);
  REQUIRE(secs < 1.0);
}

namespace {
// A stand-in for hipMemUnmap that holds the unmap open until the test lets
// it finish.
std::atomic<bool> g_unmap_entered{false};
std::atomic<bool> g_unmap_go{false};
std::atomic<bool> g_unmap_done{false};

void reset_slow_unmap() {
  g_unmap_entered = false;
  g_unmap_go      = false;
  g_unmap_done    = false;
}

hipError_t slow_unmap(void*, size_t) {
  g_unmap_entered = true;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!g_unmap_go && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  g_unmap_done = true;
  return hipSuccess;
}
hipError_t no_release(hipMemGenericAllocationHandle_t) { return hipSuccess; }
hipError_t ok_map(void*, size_t, int, hipMemGenericAllocationHandle_t*,
                  const std::vector<int>&) {
  return hipSuccess;
}

// Wait up to `secs` seconds for `flag`.
bool wait_for(const std::atomic<bool>& flag, int secs = 10) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(secs);
  while (!flag && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  return flag;
}

// Stand-ins that count the maps and unmaps placement makes. Maps run out of
// memory until g_oom_until unmaps have happened.
std::atomic<int> g_maps{0};
std::atomic<int> g_unmaps{0};
std::atomic<int> g_oom_until{0};

void reset_vmm_counts() {
  g_maps = 0;
  g_unmaps = 0;
  g_oom_until = 0;
}
hipError_t counted_map(void*, size_t, int, hipMemGenericAllocationHandle_t*,
                       const std::vector<int>&) {
  ++g_maps;
  return g_unmaps < g_oom_until ? hipErrorOutOfMemory : hipSuccess;
}
hipError_t counted_unmap(void*, size_t) {
  ++g_unmaps;
  return hipSuccess;
}

void* at(uint64_t va) { return reinterpret_cast<void*>(va); }
}  // namespace

HRR_TEST_CASE(Unit_HRR_VaPlacement_UnmapHoldsTheLock) {
  // While one thread unmaps a placed allocation, the pages are still mapped.
  // Another thread asking about them waits until the unmap is over, rather
  // than hearing they are gone and placing over them.
  reset_slow_unmap();
  hrr::VaPlacement pl;
  pl.set_vmm_ops_for_test({ok_map, slow_unmap, no_release});
  pl.adopt_mapping_for_test(B, P);
  REQUIRE(pl.is_mapped(at(B)));

  SECTION("a free") {
    bool unmapped = false;  // Catch2 assertions are for the main thread only
    std::thread a([&] { unmapped = pl.unmap(at(B)); });
    if (!wait_for(g_unmap_entered)) {
      g_unmap_go = true;
      a.join();
      FAIL("unmap never reached hipMemUnmap");
    }
    bool done_when_answered = false;
    bool mapped = true;
    std::thread b([&] {
      mapped = pl.is_mapped(at(B));
      done_when_answered = g_unmap_done;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    g_unmap_go = true;
    a.join();
    b.join();
    REQUIRE(unmapped);
    REQUIRE(done_when_answered);
    REQUIRE_FALSE(mapped);
  }
  SECTION("a drain of deferred frees") {
    REQUIRE(pl.unmap(at(B), /*defer=*/true));
    size_t drained = 0;
    std::thread a([&] { drained = pl.drain_deferred(); });
    if (!wait_for(g_unmap_entered)) {
      g_unmap_go = true;
      a.join();
      FAIL("drain_deferred never reached hipMemUnmap");
    }
    bool done_when_answered = false;
    bool mapped = true;
    std::thread b([&] {
      mapped = pl.is_mapped(at(B));
      done_when_answered = g_unmap_done;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    g_unmap_go = true;
    a.join();
    b.join();
    REQUIRE(drained == 1);
    REQUIRE(done_when_answered);
    REQUIRE_FALSE(mapped);
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_UnmapRunsOutsideTheLock) {
  // hipMemUnmap waits for every stream on the device. Another thread's
  // allocation or free elsewhere must not wait for that too: a kernel still
  // queued may need that thread's next event to finish. An allocation over
  // the pages being unmapped waits, and is placed once they are gone.
  const uint64_t C = B + 8 * P;
  for (bool drain : {false, true}) {
    INFO((drain ? "drain_deferred" : "unmap"));
    reset_slow_unmap();
    hrr::VaPlacement pl;
    pl.set_vmm_ops_for_test({ok_map, slow_unmap, no_release});
    pl.adopt_reservation_for_test(B, B + 16 * P, 1);
    pl.adopt_mapping_for_test(B, P);
    pl.adopt_mapping_for_test(C, P);
    if (drain) REQUIRE(pl.unmap(at(B), /*defer=*/true));

    std::thread a([&] {
      if (drain) (void)pl.drain_deferred();
      else (void)pl.unmap(at(B));
    });
    if (!wait_for(g_unmap_entered)) {
      g_unmap_go = true;
      a.join();
      FAIL("hipMemUnmap never reached");
    }
    // Another mapping: answered while the unmap is still running.
    std::atomic<bool> answered{false};
    bool other_mapped = false, done_when_answered = true;
    std::thread b([&] {
      other_mapped = pl.is_mapped(at(C));
      done_when_answered = g_unmap_done;
      answered = true;
    });
    // The pages being unmapped: placed only once the unmap is over.
    bool placed = false, done_when_placed = false;
    void* live = nullptr;
    std::thread c([&] {
      placed = pl.map_at(B, P, 0, "hipMalloc", &live);
      done_when_placed = g_unmap_done;
    });
    const bool in_time = wait_for(answered, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    g_unmap_go = true;
    a.join();
    b.join();
    c.join();
    REQUIRE(in_time);
    REQUIRE(other_mapped);
    REQUIRE_FALSE(done_when_answered);
    REQUIRE(placed);
    REQUIRE(live == at(B));
    REQUIRE(done_when_placed);
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_FreedRangeTakenBack) {
  // hipFreeAsync defers the unmap, and a stream-ordered pool hands the same
  // address straight back. The allocation recorded there takes the mapping
  // back as it is: no device-wide wait for an unmap the recording never had.
  reset_vmm_counts();
  hrr::VaPlacement pl;
  pl.set_vmm_ops_for_test({counted_map, counted_unmap, no_release});
  pl.adopt_reservation_for_test(B, B + 16 * P, 2);
  void* live = nullptr;
  REQUIRE(pl.map_at(B, P, 0, "hipMallocAsync", &live));
  REQUIRE(pl.unmap(at(B), /*defer=*/true));
  REQUIRE(g_maps == 1);
  REQUIRE(g_unmaps == 0);

  SECTION("the same pages on the same device") {
    REQUIRE(pl.map_at(B + 64, P - 64, 0, "hipMallocAsync", &live));
    REQUIRE(live == at(B + 64));
    REQUIRE(g_maps == 1);
    REQUIRE(g_unmaps == 0);
    REQUIRE(pl.is_mapped(at(B + 64)));
    REQUIRE(pl.drain_deferred() == 0);
  }
  SECTION("the same pages inside a capture") {
    REQUIRE(pl.map_at(B, P, 0, "hipMallocAsync", &live, /*capturing=*/true));
    REQUIRE(g_unmaps == 0);
    REQUIRE(pl.fallbacks() == 0);
  }
  SECTION("the same pages on another device: unmapped, then mapped") {
    REQUIRE(pl.map_at(B, P, 1, "hipMallocAsync", &live));
    REQUIRE(g_unmaps == 1);
    REQUIRE(g_maps == 2);
  }
  SECTION("more pages over it: unmapped, then mapped") {
    REQUIRE(pl.map_at(B, 2 * P, 0, "hipMalloc", &live));
    REQUIRE(g_unmaps == 1);
    REQUIRE(g_maps == 2);
  }
  SECTION("more pages over it inside a capture: falls back, still mapped") {
    REQUIRE_FALSE(pl.map_at(B, 2 * P, 0, "hipMalloc", &live, /*capturing=*/true));
    REQUIRE(g_unmaps == 0);
    REQUIRE(pl.fallbacks() == 1);
    REQUIRE(pl.drain_deferred() == 1);
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_OutOfMemoryDrainsFreed) {
  // Deferred frees still hold their memory. A map that runs out of it unmaps
  // them and tries once more, unless a capture is open.
  reset_vmm_counts();
  hrr::VaPlacement pl;
  pl.set_vmm_ops_for_test({counted_map, counted_unmap, no_release});
  pl.adopt_reservation_for_test(B, B + 16 * P, 1);
  void* live = nullptr;
  REQUIRE(pl.map_at(B, P, 0, "hipMallocAsync", &live));
  REQUIRE(pl.unmap(at(B), /*defer=*/true));
  REQUIRE(g_maps == 1);

  SECTION("memory comes back once the freed mapping is unmapped") {
    g_oom_until = 1;
    REQUIRE(pl.map_at(B + 4 * P, P, 0, "hipMalloc", &live));
    REQUIRE(live == at(B + 4 * P));
    REQUIRE(g_unmaps == 1);
    REQUIRE(g_maps == 3);
    REQUIRE(pl.fallbacks() == 0);
  }
  SECTION("inside a capture nothing is unmapped") {
    g_oom_until = 1;
    REQUIRE_FALSE(pl.map_at(B + 4 * P, P, 0, "hipMalloc", &live, /*capturing=*/true));
    REQUIRE(g_unmaps == 0);
    REQUIRE(g_maps == 2);
    REQUIRE(pl.fallbacks() == 1);
  }
  SECTION("still out of memory after the drain: one retry, then a fallback") {
    g_oom_until = 100;
    REQUIRE_FALSE(pl.map_at(B + 4 * P, P, 0, "hipMalloc", &live));
    REQUIRE(g_unmaps == 1);
    REQUIRE(g_maps == 3);
    REQUIRE(pl.fallbacks() == 1);
  }
  SECTION("nothing freed: no retry") {
    REQUIRE(pl.drain_deferred() == 1);
    g_oom_until = 100;
    REQUIRE_FALSE(pl.map_at(B + 4 * P, P, 0, "hipMalloc", &live));
    REQUIRE(g_maps == 2);
  }
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_FallbackRetriesAfterOutOfMemory) {
  // An allocation placement did not place ran out of memory. Replay unmaps
  // the deferred frees and tries it once more, unless a capture is open or
  // the call failed for another reason.
  reset_vmm_counts();
  hrr::VaPlacement pl;
  pl.set_vmm_ops_for_test({counted_map, counted_unmap, no_release});
  pl.adopt_reservation_for_test(B, B + 16 * P, 1);
  void* live = nullptr;
  REQUIRE(pl.map_at(B, P, 0, "hipMallocAsync", &live));
  REQUIRE(pl.unmap(at(B), /*defer=*/true));

  SECTION("out of memory: the freed mapping is unmapped") {
    REQUIRE(pl.drain_for_retry(hipErrorOutOfMemory, /*capturing=*/false) == 1);
    REQUIRE(g_unmaps == 1);
    // Nothing left to give back, so no second retry.
    REQUIRE(pl.drain_for_retry(hipErrorOutOfMemory, false) == 0);
  }
  SECTION("inside a capture nothing is unmapped") {
    REQUIRE(pl.drain_for_retry(hipErrorOutOfMemory, /*capturing=*/true) == 0);
    REQUIRE(g_unmaps == 0);
  }
  SECTION("another error: nothing is unmapped") {
    REQUIRE(pl.drain_for_retry(hipErrorInvalidValue, false) == 0);
    REQUIRE(g_unmaps == 0);
  }
}

// Placement holds its placeholders with mmap, so it is off on Windows and
// these tests have nothing to hold there.
#ifndef _WIN32
namespace {
// A range of `pages` pages that nothing in this process maps right now.
uint64_t free_range(size_t pages) {
  void* p = mmap(nullptr, pages * P, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  REQUIRE(p != MAP_FAILED);
  munmap(p, pages * P);
  return reinterpret_cast<uint64_t>(p);
}

// Whether something already maps the page at `va`.
bool page_taken(uint64_t va) {
  void* want = reinterpret_cast<void*>(va);
  void* p = mmap(want, P, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | 0x100000 /*NOREPLACE*/, -1, 0);
  if (p == MAP_FAILED) return true;
  munmap(p, P);
  return p != want;
}
}  // namespace

HRR_TEST_CASE(Unit_HRR_VaPlacement_HoldFreePieces) {
  const uint64_t R = free_range(8);
  // Something else takes page 3.
  void* other = mmap(reinterpret_cast<void*>(R + 3 * P), P, PROT_READ,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  REQUIRE(other == reinterpret_cast<void*>(R + 3 * P));

  SECTION("the occupied page is skipped and the rest is held") {
    std::vector<VaRange> held;
    hrr::hold_free_pieces(R, R + 8 * P, hrr::read_proc_maps(), &held);
    REQUIRE(hrr::va_round_merge(held, P) == Ranges{{R, R + 3 * P}, {R + 4 * P, R + 8 * P}});
    REQUIRE(page_taken(R));
    REQUIRE(page_taken(R + 7 * P));
    for (const auto& r : held) munmap(reinterpret_cast<void*>(r.base), r.end - r.base);
  }
  SECTION("a stale address map still finds the free pieces") {
    std::vector<VaRange> held;
    hrr::hold_free_pieces(R, R + 8 * P, {}, &held);
    REQUIRE(hrr::va_round_merge(held, P) == Ranges{{R, R + 3 * P}, {R + 4 * P, R + 8 * P}});
    for (const auto& r : held) munmap(reinterpret_cast<void*>(r.base), r.end - r.base);
  }
  munmap(other, P);
}

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HRR_TEST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define HRR_TEST_ASAN 1
#endif

HRR_TEST_CASE(Unit_HRR_VaPlacement_HoldStopsOnARefusal) {
  // Under RLIMIT_AS every mmap fails with ENOMEM. Nothing is mapped in the
  // range, so splitting it would only fail again for each of its 16M pages.
  // In a child, since the limit cannot be lifted again by an unprivileged
  // process. The alarm turns a hang into a failure.
#ifdef HRR_TEST_ASAN
  // ASan reserves terabytes of shadow memory, and its allocator fails
  // under any address-space limit near what the process maps.
  SKIP("RLIMIT_AS cannot be used under AddressSanitizer");
#endif
  constexpr uint64_t kRange = 64ull << 30;
  const uint64_t b = free_range(kRange / P);
  const pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    alarm(60);
    rlimit lim{};
    getrlimit(RLIMIT_AS, &lim);
    // What the process maps now, so the next mapping is over the limit.
    size_t vm = 0;
    for (const auto& r : hrr::read_proc_maps()) vm += r.end - r.base;
    // A hard limit below that already: the soft one cannot be set to it.
    if (lim.rlim_max != RLIM_INFINITY && lim.rlim_max < vm) _exit(3);
    lim.rlim_cur = vm;
    if (setrlimit(RLIMIT_AS, &lim) != 0) _exit(3);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<VaRange> held;
    hrr::hold_free_pieces(b, b + kRange, {}, &held);
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    _exit(!held.empty() ? 1 : secs >= 1.0 ? 2 : 0);
  }
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  INFO("child status " << status << " (exit 1: something was held, 2: over 1 s)");
  REQUIRE(WIFEXITED(status));
  if (WEXITSTATUS(status) == 3) SKIP("the address-space limit could not be set here");
  REQUIRE(WEXITSTATUS(status) == 0);
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_VmmHoldRestoredAfterMiss) {
  // A recorded hipMemAddressReserve range is held until the replayed reserve
  // asks for it. If the runtime then reserves somewhere else, the recorded
  // range must be held again, or a later allocation can land in it.
  const uint64_t R = free_range(4);
  hrr::PlacementPlan plan;
  plan.vmm = {{R, R + 4 * P}};
  hrr::VaPlacement pl;
  REQUIRE(pl.hold(plan));
  REQUIRE(page_taken(R));

  REQUIRE(pl.release_vmm_hold(R, 4 * P));
  REQUIRE_FALSE(page_taken(R));
  REQUIRE_FALSE(page_taken(R + 3 * P));

  pl.vmm_reserved(R, 4 * P, /*held=*/true, /*live=*/R + (1ull << 32));
  REQUIRE(pl.fallbacks() == 1);
  REQUIRE(pl.placed() == 0);
  REQUIRE(page_taken(R));
  REQUIRE(page_taken(R + 3 * P));

  // A reserve that lands where it was recorded is counted as placed.
  REQUIRE(pl.release_vmm_hold(R, 4 * P));
  pl.vmm_reserved(R, 4 * P, true, R);
  REQUIRE(pl.placed() == 1);

  pl.release_all();
  REQUIRE_FALSE(pl.active());
  REQUIRE_FALSE(page_taken(R));
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_VmmHoldsRestoredWithOneRead) {
  // Between the warm-up and the timed pass every hipMemAddressReserve range is
  // held again. A trace with thousands of them must not read /proc/self/maps,
  // which can list a million mappings, once for each.
  const uint64_t R1 = free_range(12);
  const uint64_t R2 = R1 + 8 * P;
  hrr::PlacementPlan plan;
  plan.vmm = {{R1, R1 + 4 * P}, {R2, R2 + 4 * P}};
  hrr::VaPlacement pl;
  REQUIRE(pl.hold(plan));
  REQUIRE(pl.release_vmm_hold(R1, 4 * P));
  REQUIRE(pl.release_vmm_hold(R2, 4 * P));
  REQUIRE_FALSE(page_taken(R1));
  REQUIRE_FALSE(page_taken(R2));

  const size_t before = hrr::proc_maps_reads_for_test();
  pl.restore_vmm_holds({{R1, R1 + 4 * P}, {R2, R2 + 4 * P}});
  REQUIRE(hrr::proc_maps_reads_for_test() - before == 1);
  REQUIRE(page_taken(R1));
  REQUIRE(page_taken(R1 + 3 * P));
  REQUIRE(page_taken(R2));
  REQUIRE(page_taken(R2 + 3 * P));

  pl.release_all();
  REQUIRE_FALSE(page_taken(R1));
  REQUIRE_FALSE(page_taken(R2));
}

HRR_TEST_CASE(Unit_HRR_VaPlacement_DeniedHeldWithPlacementOff) {
  // With placement off, HIP_HRR_REPLAY_PLACE_DENY still holds the denied
  // ranges, so the runtime cannot return a recorded address by chance. The
  // holds leave placement inactive, and teardown drops them.
  const uint64_t R = free_range(4);
  hrr::VaPlacement pl;
  REQUIRE(pl.hold_denied({{R, R + 4 * P}}));
  REQUIRE_FALSE(pl.active());
  REQUIRE(page_taken(R));
  REQUIRE(page_taken(R + 3 * P));
  pl.release_all();
  REQUIRE_FALSE(page_taken(R));
  REQUIRE_FALSE(page_taken(R + 3 * P));
}

namespace {
// What `fn` writes to stderr.
template <class Fn>
std::string stderr_of(Fn fn) {
  fflush(stderr);
  FILE* tmp = tmpfile();
  REQUIRE(tmp != nullptr);
  const int saved = dup(2);
  dup2(fileno(tmp), 2);
  fn();
  fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::string text;
  rewind(tmp);
  char buf[4096];
  size_t n = 0;
  while ((n = fread(buf, 1, sizeof(buf), tmp)) > 0) text.append(buf, n);
  fclose(tmp);
  return text;
}

size_t count_of(const std::string& text, const std::string& what) {
  size_t n = 0;
  for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
  return n;
}
}  // namespace

HRR_TEST_CASE(Unit_HRR_VaPlacement_FallbackLinesCapped) {
  const std::string named = "not placed at its recorded address";
  const std::string more  = "further fallbacks are only counted; --verbose names every one";
  SECTION("16 lines, then one saying the rest are counted") {
    hrr::VaPlacement pl;
    const std::string err = stderr_of([&] {
      for (int i = 0; i < 40; ++i) pl.fell_back(B + i * P, 64, "hipMalloc", "a test");
    });
    REQUIRE(pl.fallbacks() == 40);
    REQUIRE(count_of(err, named) == 16);
    REQUIRE(count_of(err, more) == 1);
  }
  SECTION("--verbose names every one") {
    hrr::VaPlacement pl;
    pl.set_verbose(true);
    const std::string err = stderr_of([&] {
      for (int i = 0; i < 40; ++i) pl.fell_back(B + i * P, 64, "hipMalloc", "a test");
    });
    REQUIRE(pl.fallbacks() == 40);
    REQUIRE(count_of(err, named) == 40);
    REQUIRE(count_of(err, more) == 0);
  }
  SECTION("the first fallback says once that the H2D scan is on, and why") {
    hrr::VaPlacement pl;
    const std::string err = stderr_of([&] {
      pl.fell_back(B, 64, "hipMallocAsync", "a test");
      pl.fell_back(B + P, 64, "hipMalloc", "a test");
    });
    const std::string scan = "replay scans the payload of each host-to-device hipMemcpy,";
    REQUIRE(count_of(err, scan) == 1);
    char why[96];
    snprintf(why, sizeof(why), "because hipMallocAsync 0x%llx did not land",
             static_cast<unsigned long long>(B));
    REQUIRE(count_of(err, why) == 1);
    // The per-fallback line no longer repeats it.
    REQUIRE(count_of(err, "now scans") == 0);
  }
  SECTION("the timed pass after a warm-up names its own fallbacks again") {
    hrr::VaPlacement pl;
    (void)stderr_of([&] {
      for (int i = 0; i < 40; ++i) pl.fell_back(B + i * P, 64, "hipMalloc", "a test");
    });
    pl.reset_counts();
    const std::string err = stderr_of([&] {
      for (int i = 0; i < 3; ++i) pl.fell_back(B + i * P, 64, "hipMalloc", "a test");
    });
    REQUIRE(pl.fallbacks() == 3);
    REQUIRE(count_of(err, named) == 3);
  }
}
#endif  // _WIN32

/**
 * End doxygen group HRR.
 * @}
 */
