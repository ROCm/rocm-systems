#ifndef LLVM_IR_PROTO_PIPELINE_HPP
#define LLVM_IR_PROTO_PIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace ir_proto {

struct PipelineResult {
  std::vector<uint8_t> hsaco;
  std::string irText;
  std::string asmText;
  int liftedCount = 0;
  int totalCount = 0;
  bool success = false;
};

/// End-to-end pipeline: HSACO binary → raise to LLVM IR → llc → HSACO.
/// Single-ISA: raises and lowers using the same ISA.
PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &targetISA,
                           const std::string &kernelName);

/// Cross-architecture pipeline: raises using sourceISA, lowers to targetISA.
PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &sourceISA,
                           const std::string &targetISA,
                           const std::string &kernelName);

} // namespace ir_proto

#endif
