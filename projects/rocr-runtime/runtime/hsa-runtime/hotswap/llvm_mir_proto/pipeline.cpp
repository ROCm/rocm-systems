#include "pipeline.hpp"
#include "lifter.hpp"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <fstream>

namespace mir_proto {

PipelineResult runPipeline(const std::vector<uint8_t> &textBytes,
                           const std::string &targetISA,
                           const std::string &kernelName) {
  PipelineResult pResult;

  // Step 1: Lift binary → live MachineFunction
  auto liftResult = liftToMIR(textBytes, targetISA, kernelName);
  if (!liftResult.success) {
    llvm::errs() << "mir_proto: Lifting failed\n";
    return pResult;
  }

  pResult.liftedCount = liftResult.liftedCount;
  pResult.totalCount = liftResult.totalCount;
  pResult.mirText = liftResult.mirText;

  llvm::errs() << "mir_proto: Lifted " << liftResult.liftedCount << " / "
               << liftResult.totalCount << " instructions to MIR\n";

  llvm::errs() << "\n=== LLVM MIR (lifted) ===\n"
               << liftResult.mirText << "\n=== End LLVM MIR ===\n\n";

  // Step 2: Generate assembly FROM the live MachineFunction.
  // The instruction stream is produced by walking MachineInstrs and
  // lowering each one to MCInst via MCInstPrinter — the original bytes
  // are NOT re-disassembled.
  pResult.assemblyText = liftResult.generateAssembly(targetISA, kernelName);
  if (pResult.assemblyText.empty()) {
    llvm::errs() << "mir_proto: Assembly generation from MIR failed\n";
    return pResult;
  }

  if (std::getenv("MIR_DUMP_ASM"))
    llvm::errs() << "\n=== Generated Assembly (from MIR) ===\n"
                 << pResult.assemblyText
                 << "\n=== End Assembly ===\n\n";

  // Step 3: Assembly → object → HSACO  (llvm-mc + ld.lld)
  auto asmPath = std::string("/tmp/mir_proto.s");
  auto objPath = std::string("/tmp/mir_proto.o");
  auto hsacoPath = std::string("/tmp/mir_proto.hsaco");

  {
    std::error_code ec;
    llvm::raw_fd_ostream f(asmPath, ec);
    if (ec) {
      llvm::errs() << "mir_proto: Cannot write asm: " << ec.message() << "\n";
      return pResult;
    }
    f << pResult.assemblyText;
  }

  std::string mcBin = "/opt/rocm/llvm/bin/llvm-mc";
  std::string mcCmd = mcBin + " -triple=amdgcn-amd-amdhsa -mcpu=" + targetISA +
                      " -filetype=obj " + asmPath + " -o " + objPath;
  int ret = std::system(mcCmd.c_str());
  if (ret != 0) {
    llvm::errs() << "mir_proto: llvm-mc failed (exit " << ret << ")\n";
    return pResult;
  }

  std::string lldBin = std::string(LLVM_TOOLS_DIR) + "/ld.lld";
  std::string lldCmd = lldBin + " -shared " + objPath + " -o " + hsacoPath;
  ret = std::system(lldCmd.c_str());
  if (ret != 0) {
    llvm::errs() << "mir_proto: ld.lld failed (exit " << ret << ")\n";
    return pResult;
  }

  std::ifstream hf(hsacoPath, std::ios::binary | std::ios::ate);
  if (!hf.is_open()) {
    llvm::errs() << "mir_proto: Cannot read HSACO\n";
    return pResult;
  }
  auto hsacoSize = hf.tellg();
  hf.seekg(0);
  pResult.hsaco.resize(hsacoSize);
  hf.read(pResult.hsaco.data(), hsacoSize);

  pResult.success = true;
  llvm::errs() << "mir_proto: HSACO generated (" << pResult.hsaco.size()
               << " bytes)\n";
  return pResult;
}

} // namespace mir_proto
