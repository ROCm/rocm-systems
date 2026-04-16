//===-- aegisbit/VariantCoverageAccumulator.h ------------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Pure (HSA-free) accumulator tracking which original PCs have been
/// covered so far across successive patched-ELF variants.  Drives the
/// plateau-termination decision in `KernelPatcher::getOrPatchVariants`.
///
/// Not thread-safe — scoped to one variant-building loop.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_VARIANT_COVERAGE_ACCUMULATOR_H
#define AEGISBIT_VARIANT_COVERAGE_ACCUMULATOR_H

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <unordered_set>

namespace aegisbit {

class VariantCoverageAccumulator {
public:
  /// True iff the given original PC has not yet been covered by a prior
  /// variant in this build loop.
  bool needsCoverage(uint64_t OriginalPC) const {
    return !Covered.count(OriginalPC);
  }

  /// Record that a newly-built variant covered `PatchedPCs`.  Returns the
  /// number of PCs that were *new* — i.e. not previously covered.  Zero
  /// means this variant contributed nothing new and the caller should
  /// stop iterating (plateau).
  std::size_t markCovered(llvm::ArrayRef<uint64_t> PatchedPCs) {
    std::size_t Added = 0;
    for (uint64_t PC : PatchedPCs) {
      if (Covered.insert(PC).second)
        ++Added;
    }
    LastAdded = Added;
    return Added;
  }

  /// Total distinct PCs covered across all variants so far.
  std::size_t coveredCount() const { return Covered.size(); }

  /// Read-only view of the full covered set — handy both for passing as
  /// `ExcludedPCs` into the next variant's `patchKernel` call and for
  /// tests asserting the union.
  const std::unordered_set<uint64_t> &covered() const { return Covered; }

  /// How many new PCs the most recent `markCovered` added.
  std::size_t lastAdded() const { return LastAdded; }

private:
  std::unordered_set<uint64_t> Covered;
  std::size_t LastAdded = 0;
};

} // namespace aegisbit

#endif // AEGISBIT_VARIANT_COVERAGE_ACCUMULATOR_H
