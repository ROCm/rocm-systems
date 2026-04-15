#include "pipeline.hpp"
#include "code_object_utils.hpp"
#include "raiser.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#ifndef LLVM_TOOLS_DIR
#define LLVM_TOOLS_DIR "/usr/bin"
#endif

namespace ir_proto {

namespace {

bool writeFile(const std::string &path, const std::string &contents) {
  std::ofstream f(path);
  if (!f.is_open()) {
    llvm::errs() << "ir_proto: Cannot write file: " << path << "\n";
    return false;
  }
  f << contents;
  return true;
}

bool writeFile(const std::string &path, const std::vector<uint8_t> &data) {
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    llvm::errs() << "ir_proto: Cannot write file: " << path << "\n";
    return false;
  }
  f.write(reinterpret_cast<const char *>(data.data()), data.size());
  return true;
}

int runCommand(const std::string &cmd) {
  llvm::errs() << "ir_proto: Running: " << cmd << "\n";
  int ret = std::system(cmd.c_str());
  if (ret != 0)
    llvm::errs() << "ir_proto: Command failed (exit " << ret << "): " << cmd
                 << "\n";
  return ret;
}

} // anonymous namespace

PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &targetISA,
                           const std::string &kernelName) {
  return runPipeline(codeObjectData, targetISA, targetISA, kernelName);
}

PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &sourceISA,
                           const std::string &targetISA,
                           const std::string &kernelName) {
  PipelineResult result;

  // Step 1: Extract .text section
  auto text = extractTextSection(codeObjectData);
  if (!text.valid) {
    llvm::errs() << "ir_proto: Failed to extract .text section\n";
    return result;
  }
  llvm::errs() << "ir_proto: .text section: " << text.bytes.size()
               << " bytes\n";

  // Step 1b: Extract kernel metadata
  auto meta = extractKernelMeta(codeObjectData, kernelName);
  if (meta.args.empty()) {
    llvm::errs() << "ir_proto: WARNING: No metadata found for '" << kernelName
                 << "', using empty metadata\n";
  }

  // Step 2: Raise to LLVM IR (using source ISA for disassembly)
  auto raised = raiseToIR(text.bytes, sourceISA, kernelName, meta);
  if (!raised.success) {
    llvm::errs() << "ir_proto: Raising to LLVM IR failed\n";
    return result;
  }
  result.irText = raised.irText;
  result.liftedCount = raised.liftedCount;
  result.totalCount = raised.totalCount;

  llvm::errs() << "ir_proto: Raised " << raised.liftedCount << "/"
               << raised.totalCount << " instructions to LLVM IR\n";
  llvm::errs() << "--- Raised LLVM IR ---\n" << raised.irText << "\n";

  // Step 3: Write IR to file
  std::string tmpDir = "/tmp/ir_proto_" + std::to_string(getpid());
  runCommand("mkdir -p " + tmpDir);

  std::string irPath = tmpDir + "/kernel.ll";
  if (!writeFile(irPath, raised.irText)) {
    llvm::errs() << "ir_proto: Failed to write IR file\n";
    return result;
  }

  // Step 4: llc — compile LLVM IR to assembly
  std::string llcBin = std::string(LLVM_TOOLS_DIR) + "/llc";
  std::string asmPath = tmpDir + "/kernel.s";
  std::string llcCmd = llcBin + " -march=amdgcn -mcpu=" + targetISA +
                       " -filetype=asm -o " + asmPath + " " + irPath +
                       " 2>&1";
  if (runCommand(llcCmd) != 0) {
    llvm::errs() << "ir_proto: llc failed\n";
    return result;
  }

  // Read the generated assembly for inspection
  {
    auto asmData = readFile(asmPath);
    result.asmText.assign(asmData.begin(), asmData.end());
    llvm::errs() << "--- llc output assembly ---\n" << result.asmText << "\n";
  }

  // Step 5: llvm-mc — assemble to object file
  std::string mcBin = std::string(LLVM_TOOLS_DIR) + "/llvm-mc";
  std::string objPath = tmpDir + "/kernel.o";
  std::string mcCmd = mcBin + " -triple=amdgcn-amd-amdhsa -mcpu=" + targetISA +
                      " -filetype=obj -o " + objPath + " " + asmPath +
                      " 2>&1";
  if (runCommand(mcCmd) != 0) {
    llvm::errs() << "ir_proto: llvm-mc failed\n";
    return result;
  }

  // Step 6: ld.lld — link to HSACO
  std::string lldBin = std::string(LLVM_TOOLS_DIR) + "/ld.lld";
  std::string hsacoPath = tmpDir + "/kernel.hsaco";
  std::string lldCmd = lldBin + " -shared -o " + hsacoPath + " " + objPath +
                       " 2>&1";
  if (runCommand(lldCmd) != 0) {
    llvm::errs() << "ir_proto: ld.lld failed\n";
    return result;
  }

  // Step 7: Read the generated HSACO
  result.hsaco = readFile(hsacoPath);
  if (result.hsaco.empty()) {
    llvm::errs() << "ir_proto: Failed to read HSACO\n";
    return result;
  }

  llvm::errs() << "ir_proto: HSACO generated: " << result.hsaco.size()
               << " bytes\n";

  result.success = true;
  return result;
}

} // namespace ir_proto
