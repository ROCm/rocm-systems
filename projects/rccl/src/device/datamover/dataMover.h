/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/// \file dataMover.h
/// \brief Backend-agnostic device data mover.
///
/// A DataMover is an ordered chain of backends. Each move goes to the FIRST
/// backend that
///   1. is compiled for the current target (kSupported),
///   2. can move between the two endpoint spaces (handles<S, D>(), compile time),
///   3. accepts this particular move (accepts<S, D>(team, bytes), runtime;
///      e.g. SDMA only above a size threshold, LDS DMA only with staging space).
/// Steps 1-2 prune the chain at compile time, so a chain written once compiles
/// for every target in a multi-arch build: on gfx950 the TDM and async-LDS
/// links simply vanish. The last eligible backend must accept; if it does not,
/// the move traps.
///
/// Backends (see their headers for details):
///   SdmaMover        Global->Global via device-initiated SDMA        (sdmaMover.h)
///   TdmMover<>       Global<->Lds, Global->Global staged, via TDM     (ldsDmaMover.h)
///   AsyncLdsMover<>  same, via global_{load,store}_async_*_lds        (ldsDmaMover.h)
///   VectorMover<>    any pair, through VGPRs with global_* / ds_*     (vectorMover.h)
///
/// \par Example
/// \code
///   using namespace rccl::dm;
///   __shared__ uint8_t staging[16384];
///   DataMover<SdmaMover, TdmMover<>, VectorMover<>> mover(
///       SdmaMover(queue, 1 << 20), TdmMover<>(staging, sizeof(staging)), VectorMover<>());
///   Team team = Team::block();
///
///   // Broadcast one peer buffer to the user buffer and the next peer.
///   mover.move(team, global(recvBuf), global(userBuf, sendBuf), bytes);
///
///   // Two independent pairs into LDS (TDM on gfx1250, vector+ds elsewhere).
///   mover.move(team, global(a, b), lds(ldsA, ldsB), bytes);
///   __syncthreads();
///
///   // Into registers and back out (always the vector backend).
///   RegTile<4> tile;
///   mover.move(team, lds(ldsA), tile, RegTile<4>::capacity(team));
///   mover.move(team, tile, global(out), RegTile<4>::capacity(team));
/// \endcode
///
/// \par Completion and visibility
/// issue() starts a move; wait() completes every move this thread (or, for the
/// per-wave DMA engines, this wave) issued through the mover, on all backends;
/// move() is issue() + wait(). Neither synchronizes the team: other threads see
/// the results after the caller's barrier. The mover object carries per-thread
/// completion state, so issue and wait through the same object.
#pragma once

#include "dmCore.h"
#include "vectorMover.h"
#include "ldsDmaMover.h"
#include "sdmaMover.h"

namespace rccl {
namespace dm {

template <class... Backends>
struct DataMover;

template <>
struct DataMover<> {
  __host__ __device__ DataMover() {}
  DM_INLINE void wait(const Team&) {}

  template <Space S, Space D>
  static constexpr bool anyEligible() {
    return false;
  }
};

template <class B, class... Rest>
struct DataMover<B, Rest...> {
  B head;
  DataMover<Rest...> tail;

  __host__ __device__ DataMover() {}
  __host__ __device__ DataMover(B b, Rest... rest) : head(b), tail(rest...) {}

  template <class Src, class Dst>
  DM_INLINE void issue(const Team& t, Src&& src, Dst&& dst, size_t bytes) {
    constexpr Space S = Endpoint<Src>::space;
    constexpr Space D = Endpoint<Dst>::space;
    if constexpr (eligible<S, D>()) {
      if constexpr (!DataMover<Rest...>::template anyEligible<S, D>()) {
        if (!head.template accepts<S, D>(t, bytes)) __builtin_trap();
        head.issue(t, src, dst, bytes);
      } else {
        if (head.template accepts<S, D>(t, bytes)) head.issue(t, src, dst, bytes);
        else tail.issue(t, src, dst, bytes);
      }
    } else if constexpr (DataMover<Rest...>::template anyEligible<S, D>()) {
      tail.issue(t, src, dst, bytes);
    } else {
      static_assert(detail::kAlwaysFalse<Src>,
                    "no backend in this DataMover can move between these spaces on this target");
    }
  }

  DM_INLINE void wait(const Team& t) {
    if constexpr (B::kSupported) head.wait(t);
    tail.wait(t);
  }

  template <class Src, class Dst>
  DM_INLINE void move(const Team& t, Src&& src, Dst&& dst, size_t bytes) {
    issue(t, src, dst, bytes);
    wait(t);
  }

  template <Space S, Space D>
  static constexpr bool eligible() {
    if constexpr (B::kSupported) return B::template handles<S, D>();
    else return false;
  }
  template <Space S, Space D>
  static constexpr bool anyEligible() {
    return eligible<S, D>() || DataMover<Rest...>::template anyEligible<S, D>();
  }
};

/// Preferred chain for gfx1250: SDMA for large Global->Global (when given a
/// queue), TDM for Global<->Lds and for Global->Global when given staging LDS,
/// async-to-LDS where the toolchain lacks TDM, VGPRs for everything else.
using DefaultMover = DataMover<SdmaMover, TdmMover<>, AsyncLdsMover<>, VectorMover<>>;

DM_INLINE DefaultMover makeDefaultMover(SdmaMover::Queue* sdmaQueue = nullptr, size_t sdmaMinBytes = 0,
                                        void* stagingLds = nullptr, uint32_t stagingLdsBytes = 0) {
  return DefaultMover(SdmaMover(sdmaQueue, sdmaMinBytes), TdmMover<>(stagingLds, stagingLdsBytes),
                      AsyncLdsMover<>(stagingLds, stagingLdsBytes), VectorMover<>());
}

}  // namespace dm
}  // namespace rccl
