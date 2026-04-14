#ifndef ASTER_PROTO_LIFTER_HPP
#define ASTER_PROTO_LIFTER_HPP

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace aster_proto {

struct LiftResult {
  mlir::OwningOpRef<mlir::Operation *> module;
  int liftedCount = 0;
  int unsupportedCount = 0;
  std::vector<std::string> unsupportedMnemonics;
  /// (byte offset, full assembly text) for each unsupported instruction.
  std::vector<std::pair<uint64_t, std::string>> unsupportedInstructions;
  /// Informational notes about transformations applied during lifting.
  std::vector<std::string> notes;
  bool success = false;
};

/// Lift raw .text bytes from a gfx942 code object into Aster's amdgcn dialect.
/// Produces amdgcn.module > amdgcn.kernel > instruction ops.
/// Unsupported instructions are tracked in the result but not emitted as ops.
LiftResult liftToAster(mlir::MLIRContext &ctx,
                       const std::vector<uint8_t> &textBytes,
                       const std::string &targetISA,
                       const std::string &kernelName);

} // namespace aster_proto

#endif // ASTER_PROTO_LIFTER_HPP
