/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/
#ifndef LIBRARY_SRC_SDMA_ANVIL_ENGINE_MAP_HPP_
#define LIBRARY_SRC_SDMA_ANVIL_ENGINE_MAP_HPP_

// Pure engine-selection and queue-budget helpers for AnvilLib. Kept free of HSA/KFD headers so the
// unit tests can build them without USE_SDMA.

#include <cstdint>
#include <string>

namespace sdma_anvil {

// A CPX/DPX partition owns host SDMA engines but no xGMI engines. Callers pass a total of 0 when
// the engine counts could not be queried, so a failed query is never read as a partition.
inline bool isSdmaPartition(uint32_t numSdmaXgmiEngines, uint32_t numSdmaEnginesTotal) {
  return numSdmaXgmiEngines == 0 && numSdmaEnginesTotal > 0;
}

// Engines Anvil may place queues on. On a partition only engine 0: the partition's second engine
// never sees a doorbell written from a GPU kernel (ROCM-32598), and Anvil rings every doorbell from
// the device, so a queue there hangs on its first packet. 0 when the counts are unknown.
inline uint32_t usableSdmaEngines(uint32_t numSdmaXgmiEngines, uint32_t numSdmaEnginesTotal) {
  return isSdmaPartition(numSdmaXgmiEngines, numSdmaEnginesTotal) ? 1 : numSdmaEnginesTotal;
}

// True when the doubled OAM-map id is past the engines this node reports. Partitions never get
// here: they always use engine 0. numSdmaEnginesTotal == 0 never folds, so the modulo below is not
// asked to divide by zero.
inline bool oamMapEngineNeedsFold(uint32_t numSdmaEnginesTotal, int doubledEngineId) {
  return numSdmaEnginesTotal > 0 && static_cast<uint32_t>(doubledEngineId) >= numSdmaEnginesTotal;
}

// Fold the undoubled OAM-map value and both PCI functions into the engines this
// node reports. numEngines must be > 0. An unreadable function (-1) counts as 0
// and therefore collides with function 0.
inline int foldOamMapEngine(int oamEngine, int srcFn, int dstFn, uint32_t numEngines) {
  const int src = srcFn < 0 ? 0 : srcFn;
  const int dst = dstFn < 0 ? 0 : dstFn;
  return (oamEngine + src + dst) % static_cast<int>(numEngines);
}

// PCI function and the function-0 BDF of the same device. function is -1 when the tail is not a
// function digit, so an unreadable id is not treated as function 0. The physical BDF is rewritten
// only in that case; a bad tail is left unchanged.
struct PciFunctionBus {
  std::string busId;
  std::string physBusId;
  int function;
};

// Split a BDF into its function digit and the function-0 BDF of the same device. Pure, and beside
// the two helpers above because it decides the same fallback they do: a tail outside '0'-'7'
// leaves function at -1 and physBusId equal to busId, which collapses getOamId's candidate list to
// one entry so the physical-BDF read never runs. PCI function numbers are three bits, so '0'-'7'
// is the whole range.
inline PciFunctionBus pciFunctionBus(const std::string& busId) {
  PciFunctionBus loc;
  loc.busId = busId;
  loc.physBusId = busId;
  loc.function = -1;
  if (busId.empty()) return loc;
  const char c = busId.back();
  if (c >= '0' && c <= '7') {
    loc.function = c - '0';
    loc.physBusId.back() = '0';
  }
  return loc;
}

// True when taking `requested` more queues would exceed the queue budget, counted across the usable
// engines as numEngines * queuesPerEngine. Either count being 0 means the budget is unknown, which
// is never reported as exhausted.
inline bool queueBudgetExceeded(uint32_t used, int requested, uint32_t numEngines,
                                uint32_t queuesPerEngine) {
  const uint32_t budget = numEngines * queuesPerEngine;
  return requested > 0 && budget > 0 && used + static_cast<uint32_t>(requested) > budget;
}

}  // namespace sdma_anvil

#endif  // LIBRARY_SRC_SDMA_ANVIL_ENGINE_MAP_HPP_
