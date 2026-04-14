#ifndef LLVM_IR_PROTO_RAISER_HPP
#define LLVM_IR_PROTO_RAISER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace ir_proto {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<llvm::Module> module;
  int liftedCount = 0;
  int totalCount = 0;
  std::string irText;
  bool success = false;
};

/// Lift raw .text bytes from a gfx942 code object and raise to LLVM IR.
/// Internally lifts to MIR (MachineFunction), then pattern-matches
/// MachineInstrs to produce typed, SSA LLVM IR suitable for llc.
RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &targetISA,
                      const std::string &kernelName);

} // namespace ir_proto

#endif
