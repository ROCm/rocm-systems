#ifndef LLVM_MIR_PROTO_LIFTER_HPP
#define LLVM_MIR_PROTO_LIFTER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mir_proto {

struct LiftResult {
  int liftedCount = 0;
  int totalCount = 0;
  std::vector<std::string> notes;
  std::string mirText;
  bool success = false;

  /// Live in-memory MachineFunction and MC printing infrastructure.
  /// Kept alive so generateAssembly() can emit instructions directly
  /// from the lifted MIR rather than re-encoding the original bytes.
  struct LiveMF;
  std::unique_ptr<LiveMF> liveMF;

  LiftResult();
  ~LiftResult();
  LiftResult(LiftResult &&) noexcept;
  LiftResult &operator=(LiftResult &&) noexcept;

  /// Generate assembly text from the lifted MIR.  The instruction
  /// stream is emitted by walking the live MachineFunction and
  /// lowering each MachineInstr back to MCInst via MCInstPrinter.
  /// Kernel metadata is appended (hardcoded for vecadd).
  std::string generateAssembly(const std::string &targetISA,
                               const std::string &kernelName) const;
};

/// Lift raw .text bytes from a gfx942 code object into LLVM MIR
/// (MachineFunction with MachineInstrs using physical registers).
/// Implicit operands (VCC, SCC, EXEC) are populated automatically from
/// MCInstrDesc (TableGen).
///
/// The live MachineFunction is returned in LiftResult::liveMF so
/// generateAssembly() can emit from it.
LiftResult liftToMIR(const std::vector<uint8_t> &textBytes,
                     const std::string &targetISA,
                     const std::string &kernelName);

} // namespace mir_proto

#endif // LLVM_MIR_PROTO_LIFTER_HPP
