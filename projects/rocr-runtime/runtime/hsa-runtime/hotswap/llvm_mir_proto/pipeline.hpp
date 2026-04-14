#ifndef LLVM_MIR_PROTO_PIPELINE_HPP
#define LLVM_MIR_PROTO_PIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace mir_proto {

struct PipelineResult {
  std::string mirText;
  std::string assemblyText;
  std::vector<char> hsaco;
  int liftedCount = 0;
  int totalCount = 0;
  bool success = false;
};

/// Full pipeline: binary .text bytes → LLVM MIR → assembly → HSACO.
PipelineResult runPipeline(const std::vector<uint8_t> &textBytes,
                           const std::string &targetISA,
                           const std::string &kernelName);

} // namespace mir_proto

#endif // LLVM_MIR_PROTO_PIPELINE_HPP
