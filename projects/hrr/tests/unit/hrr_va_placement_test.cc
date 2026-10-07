/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR capture-address placement
 * @{
 * @ingroup HRRTest
 * CPU-only tests for the range arithmetic behind capture-address placement:
 * which recorded calls claim a range, how the ranges are rounded and merged,
 * and what is left out. The GPU half is covered by
 * tests/integration/hrr_va_placement_test.cc.
 */

#include "hrr_test_common.hh"
#include "hrr_va_placement.h"
#include "hrr/hrr_api_args.h"

#include <cstring>
#include <vector>

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

/**
 * End doxygen group HRR.
 * @}
 */
