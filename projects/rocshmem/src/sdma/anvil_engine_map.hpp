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

// Pure engine-selection helpers for AnvilLib. Kept free of HSA/KFD headers so the unit tests can
// build them without USE_SDMA.

#include <cstdint>
#include <string>

namespace sdma_anvil {

// A CPX/DPX partition owns host SDMA engines but no xGMI engines. Callers pass a total of 0 when
// the engine counts could not be queried, so a failed query is never read as a partition.
inline bool isSdmaPartition(uint32_t numSdmaXgmiEngines, uint32_t numSdmaEnginesTotal) {
  return numSdmaXgmiEngines == 0 && numSdmaEnginesTotal > 0;
}

// True when the doubled OAM-map id cannot be used as-is. A partition (no xGMI
// engines) folds even when the doubled id is in range: same-device peers share
// the map diagonal and would otherwise all land on engine 0. An id past the
// engines this node reports folds too. numSdmaEnginesTotal == 0 never folds,
// so the modulo below is not asked to divide by zero.
inline bool oamMapEngineNeedsFold(uint32_t numSdmaXgmiEngines, uint32_t numSdmaEnginesTotal,
                                  int doubledEngineId) {
  const bool partition = isSdmaPartition(numSdmaXgmiEngines, numSdmaEnginesTotal);
  const bool outOfRange =
      numSdmaEnginesTotal > 0 && static_cast<uint32_t>(doubledEngineId) >= numSdmaEnginesTotal;
  return partition || outOfRange;
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

}  // namespace sdma_anvil

#endif  // LIBRARY_SRC_SDMA_ANVIL_ENGINE_MAP_HPP_
