#ifndef LLVM_IR_PROTO_RAISER_HPP
#define LLVM_IR_PROTO_RAISER_HPP

#include "code_object_utils.hpp"
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
  std::string failMnemonic;
  std::string failFormat;
  bool success = false;
};

RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &targetISA,
                      const std::string &kernelName,
                      const KernelMeta &meta);

} // namespace ir_proto

#endif
