#include "pipeline.hpp"
#include "code_object_utils.hpp"
#include "raiser.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"

#include <cstdio>
#include <fstream>
#include <string>

#define DEBUG_TYPE "ir-proto"

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
  f.flush();
  if (!f) {
    llvm::errs() << "ir_proto: write failed for: " << path << "\n";
    return false;
  }
  return true;
}

bool writeFile(const std::string &path, const std::vector<uint8_t> &data) {
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    llvm::errs() << "ir_proto: Cannot write file: " << path << "\n";
    return false;
  }
  f.write(reinterpret_cast<const char *>(data.data()), data.size());
  f.flush();
  if (!f) {
    llvm::errs() << "ir_proto: write failed for: " << path << "\n";
    return false;
  }
  return true;
}

int runTool(llvm::StringRef program, llvm::ArrayRef<llvm::StringRef> args) {
  llvm::errs() << "ir_proto: Running:";
  for (auto &a : args) llvm::errs() << " " << a;
  llvm::errs() << "\n";

  auto exeOrErr = llvm::sys::findProgramByName(program);
  if (!exeOrErr) {
    llvm::errs() << "ir_proto: tool not found: " << program << "\n";
    return -1;
  }

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(*exeOrErr, args, /*Env=*/std::nullopt,
                                     /*Redirects=*/{}, /*SecondsToWait=*/120,
                                     /*MemoryLimit=*/0, &errMsg);
  if (rc != 0)
    llvm::errs() << "ir_proto: " << program << " failed (exit " << rc << ")"
                 << (errMsg.empty() ? "" : ": " + errMsg) << "\n";
  return rc;
}

struct TempDir {
  llvm::SmallString<128> path;
  bool valid = false;

  TempDir() {
    if (auto ec = llvm::sys::fs::createUniqueDirectory("ir_proto", path)) {
      llvm::errs() << "ir_proto: failed to create temp dir: " << ec.message() << "\n";
    } else {
      valid = true;
    }
  }
  ~TempDir() {
    // Keep temp dirs for debugging
    // if (valid)
    //   llvm::sys::fs::remove_directories(path);
  }
  TempDir(const TempDir &) = delete;
  TempDir &operator=(const TempDir &) = delete;

  std::string filePath(llvm::StringRef name) const {
    llvm::SmallString<256> p(path);
    llvm::sys::path::append(p, name);
    return std::string(p);
  }
};

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
  LLVM_DEBUG(llvm::dbgs() << "ir_proto: .text section: " << text.bytes.size()
                          << " bytes\n");

  // Step 1b: Extract kernel metadata
  auto meta = extractKernelMeta(codeObjectData, kernelName);
  if (meta.args.empty()) {
    llvm::errs() << "ir_proto: WARNING: No metadata found for '" << kernelName
                 << "', using empty metadata\n";
  }

  // Step 1c: Find kernel symbol offset in .text section
  uint64_t kernelOffset = findKernelSymbolOffset(codeObjectData, kernelName);
  LLVM_DEBUG(if (kernelOffset > 0)
    llvm::dbgs() << "ir_proto: Kernel '" << kernelName
                 << "' at .text offset 0x" << llvm::utohexstr(kernelOffset)
                 << "\n");

  // Step 2: Raise to LLVM IR (using source ISA for disassembly)
  auto raised = raiseToIR(text.bytes, sourceISA, kernelName, meta, kernelOffset);
  if (!raised.success) {
    llvm::errs() << "ir_proto: Raising to LLVM IR failed\n";
    return result;
  }
  result.irText = raised.irText;
  result.liftedCount = raised.liftedCount;
  result.totalCount = raised.totalCount;

  LLVM_DEBUG(llvm::dbgs() << "ir_proto: Raised " << raised.liftedCount << "/"
                           << raised.totalCount << " instructions to LLVM IR\n");
  LLVM_DEBUG(llvm::dbgs() << "--- Raised LLVM IR ---\n" << raised.irText << "\n");

  // Step 3: Create temp directory and write IR to file
  TempDir tmpDir;
  if (!tmpDir.valid) {
    llvm::errs() << "ir_proto: Failed to create temp directory\n";
    return result;
  }

  std::string irPath   = tmpDir.filePath("kernel.ll");
  std::string asmPath  = tmpDir.filePath("kernel.s");
  std::string objPath  = tmpDir.filePath("kernel.o");
  std::string hsacoPath = tmpDir.filePath("kernel.hsaco");

  if (!writeFile(irPath, raised.irText)) {
    llvm::errs() << "ir_proto: Failed to write IR file\n";
    return result;
  }
  // Debug: also save a copy for inspection
  writeFile("/tmp/gfx1250_debug_" + kernelName + ".ll", raised.irText);

  // Step 4: llc — compile LLVM IR to assembly
  std::string llcBin = std::string(LLVM_TOOLS_DIR) + "/llc";
  if (runTool(llcBin, {llcBin, "-march=amdgcn",
                       "-mcpu=" + targetISA,
                       "-filetype=asm", "-o", asmPath, irPath}) != 0) {
    llvm::errs() << "ir_proto: llc failed\n";
    return result;
  }

  // Read the generated assembly for inspection
  {
    auto asmData = readFile(asmPath);
    result.asmText.assign(asmData.begin(), asmData.end());
    LLVM_DEBUG(llvm::dbgs() << "--- llc output assembly ---\n" << result.asmText << "\n");
    // Debug: save asm for inspection
    writeFile("/tmp/gfx1250_debug_" + kernelName + ".s", result.asmText);
  }

  // Step 5: llvm-mc — assemble to object file
  std::string mcBin = std::string(LLVM_TOOLS_DIR) + "/llvm-mc";
  if (runTool(mcBin, {mcBin, "-triple=amdgcn-amd-amdhsa",
                      "-mcpu=" + targetISA,
                      "-filetype=obj", "-o", objPath, asmPath}) != 0) {
    llvm::errs() << "ir_proto: llvm-mc failed\n";
    return result;
  }

  // Step 6: ld.lld — link to HSACO
  std::string lldBin = std::string(LLVM_TOOLS_DIR) + "/ld.lld";
  if (runTool(lldBin, {lldBin, "-shared", "-o", hsacoPath, objPath}) != 0) {
    llvm::errs() << "ir_proto: ld.lld failed\n";
    return result;
  }

  // Step 7: Read the generated HSACO
  result.hsaco = readFile(hsacoPath);
  if (result.hsaco.empty()) {
    llvm::errs() << "ir_proto: Failed to read HSACO\n";
    return result;
  }

  LLVM_DEBUG(llvm::dbgs() << "ir_proto: HSACO generated: " << result.hsaco.size()
                          << " bytes\n");

  result.success = true;
  return result;
}

} // namespace ir_proto
