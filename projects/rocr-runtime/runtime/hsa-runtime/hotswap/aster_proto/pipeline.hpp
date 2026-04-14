#ifndef ASTER_PROTO_PIPELINE_HPP
#define ASTER_PROTO_PIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace aster_proto {

struct PipelineResult {
  std::string assemblyText;
  std::vector<char> hsaco;
  int liftedCount = 0;
  int unsupportedCount = 0;
  std::vector<std::string> unsupportedMnemonics;
  bool success = false;
};

/// Full pipeline: binary .text bytes → Aster IR → assembly → HSACO.
/// Uses Aster's translateModule for assembly emission and compileAsm+linkBinary
/// for in-process HSACO generation (no subprocess calls).
PipelineResult runPipeline(const std::vector<uint8_t> &textBytes,
                           const std::string &targetISA,
                           const std::string &kernelName);

} // namespace aster_proto

#endif // ASTER_PROTO_PIPELINE_HPP
