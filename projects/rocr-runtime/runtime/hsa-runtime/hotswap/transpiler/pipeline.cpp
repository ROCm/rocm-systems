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
#include <cstdlib>
#include <fstream>
#include <string>

#define DEBUG_TYPE "transpiler"

#ifndef LLVM_TOOLS_DIR
#define LLVM_TOOLS_DIR "/usr/bin"
#endif

namespace transpiler {

namespace {

bool writeFile(const std::string &path, const std::string &contents) {
  std::ofstream f(path);
  if (!f.is_open()) {
    llvm::errs() << "transpiler: Cannot write file: " << path << "\n";
    return false;
  }
  f << contents;
  f.flush();
  if (!f) {
    llvm::errs() << "transpiler: write failed for: " << path << "\n";
    return false;
  }
  return true;
}

bool writeFile(const std::string &path, const std::vector<uint8_t> &data) {
  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    llvm::errs() << "transpiler: Cannot write file: " << path << "\n";
    return false;
  }
  f.write(reinterpret_cast<const char *>(data.data()), data.size());
  f.flush();
  if (!f) {
    llvm::errs() << "transpiler: write failed for: " << path << "\n";
    return false;
  }
  return true;
}

// Derive a filesystem-safe basename for an arbitrarily long kernel name.
// Most POSIX filesystems cap individual path components at 255 bytes, and
// Salmon generates sibling files off the same stem (e.g. `<stem>.ll`,
// `<stem>.s`, `<stem>.dis`), so we leave a small suffix budget and fold
// anything longer down to a deterministic truncated+hashed form so two
// kernels with a shared 240-byte prefix don't collide on disk.
//
// The returned basename preserves a readable prefix of the original name
// for debuggability; it's only intended for temp-dir scratch files —
// symbol names inside the IR itself are unaffected.
std::string makeSafeBasename(const std::string &kernelName,
                             size_t reservedSuffixBytes = 8) {
  constexpr size_t kMaxComponentBytes = 255;
  if (kernelName.size() + reservedSuffixBytes <= kMaxComponentBytes)
    return kernelName;

  // FNV-1a 64-bit hash — small, deterministic, no libstdc++ dep beyond cstdint.
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : kernelName) {
    h ^= c;
    h *= 0x100000001b3ull;
  }

  constexpr size_t kHashHexBytes = 16;   // "%016llx"
  constexpr size_t kSeparatorBytes = 1;  // '_'
  const size_t prefixBudget = kMaxComponentBytes - reservedSuffixBytes -
                              kHashHexBytes - kSeparatorBytes;
  std::string prefix = kernelName.substr(0, prefixBudget);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return prefix + "_" + buf;
}

int runTool(llvm::StringRef program, llvm::ArrayRef<llvm::StringRef> args) {
  LLVM_DEBUG({
    llvm::dbgs() << "transpiler: Running:";
    for (auto &a : args) llvm::dbgs() << " " << a;
    llvm::dbgs() << "\n";
  });

  auto exeOrErr = llvm::sys::findProgramByName(program);
  if (!exeOrErr) {
    llvm::errs() << "transpiler: tool not found: " << program << "\n";
    return -1;
  }

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(*exeOrErr, args, /*Env=*/std::nullopt,
                                     /*Redirects=*/{}, /*SecondsToWait=*/120,
                                     /*MemoryLimit=*/0, &errMsg);
  if (rc != 0)
    llvm::errs() << "transpiler: " << program << " failed (exit " << rc << ")"
                 << (errMsg.empty() ? "" : ": " + errMsg) << "\n";
  return rc;
}

struct DumpDir {
  llvm::SmallString<128> path;
  bool valid = false;
  bool persistent = false;

  DumpDir() {
    static const char *envDir = std::getenv("HSA_SALMON_DUMP_DIR");
    if (envDir && envDir[0]) {
      persistent = true;
      path = envDir;
      if (auto ec = llvm::sys::fs::create_directories(path)) {
        llvm::errs() << "salmon: failed to create dump dir '"
                     << path << "': " << ec.message() << "\n";
        return;
      }
      // Create a unique subdirectory per invocation so parallel runs
      // don't clobber each other.
      llvm::SmallString<128> sub;
      if (auto ec = llvm::sys::fs::createUniqueDirectory(
              path + "/salmon", sub)) {
        llvm::errs() << "salmon: failed to create subdir in '"
                     << path << "': " << ec.message() << "\n";
        return;
      }
      path = sub;
      valid = true;
    } else {
      if (auto ec =
              llvm::sys::fs::createUniqueDirectory("transpiler", path)) {
        llvm::errs() << "salmon: failed to create temp dir: "
                     << ec.message() << "\n";
      } else {
        valid = true;
      }
    }
  }

  ~DumpDir() {
    if (valid && !persistent)
      llvm::sys::fs::remove_directories(path);
  }

  DumpDir(const DumpDir &) = delete;
  DumpDir &operator=(const DumpDir &) = delete;

  std::string filePath(llvm::StringRef name) const {
    llvm::SmallString<256> p(path);
    llvm::sys::path::append(p, name);
    return std::string(p);
  }
};

} // anonymous namespace

bool isStrictMode() {
  // Parsed once on first call. The handler implementations call this on
  // every relevant instruction, so going through the OS allocator
  // (`std::getenv`) repeatedly would be wasteful; the result also cannot
  // change inside a process because the env var is read once at the
  // first transpile and reused for the rest of the process lifetime.
  // Treats any non-empty value as enabled to keep the runner side
  // (`HSA_SALMON_STRICT=1`) and the pipeline side decoupled — a future
  // shell that writes `HSA_SALMON_STRICT=true` still works.
  static const bool s_strict = []() {
    const char *v = std::getenv("HSA_SALMON_STRICT");
    return v && v[0] != '\0';
  }();
  return s_strict;
}

// Raise one kernel to IR, compile to a relocatable .o via llc + llvm-mc.
// On success, writes the .o to objPath and returns true.
static bool raiseAndCompileKernel(const TextSection &text,
                                  const std::vector<uint8_t> &codeObjectData,
                                  const std::string &kernelName,
                                  const std::string &sourceISA,
                                  const std::string &targetISA,
                                  const DumpDir &tmpDir,
                                  const std::string &objPath,
                                  PipelineResult &result,
                                  bool enableWritelaneRewrite,
                                  bool enableWaveNative,
                                  bool enablePermLane16Xor3PartnerRewrite) {
  auto meta = extractKernelMeta(codeObjectData, kernelName);
  if (meta.args.empty()) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << kernelName
                 << "', using empty metadata\n";
  }

  uint64_t kernelOffset = findKernelSymbolOffset(codeObjectData, kernelName);
  LLVM_DEBUG(if (kernelOffset > 0)
    llvm::dbgs() << "transpiler: Kernel '" << kernelName
                 << "' at .text offset 0x" << llvm::utohexstr(kernelOffset)
                 << "\n");

  auto raised = raiseToIR(text.bytes, sourceISA, kernelName, meta, kernelOffset,
                           targetISA, enableWritelaneRewrite,
                           enableWaveNative,
                           enablePermLane16Xor3PartnerRewrite);
  if (!raised.success) {
    llvm::errs() << "transpiler: Raising '" << kernelName << "' to LLVM IR failed";
    if (!raised.failure.mnemonic.empty()) {
      llvm::errs() << " (unsupported: " << raised.failure.mnemonic << ")";
      result.failMnemonic = raised.failure.mnemonic;
    }
    llvm::errs() << "\n";
    return false;
  }
  result.liftedCount += raised.liftedCount;
  result.totalCount += raised.totalCount;
  if (!result.irText.empty())
    result.irText += "\n";
  result.irText += raised.irText;

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raised '" << kernelName << "' "
                           << raised.liftedCount << "/"
                           << raised.totalCount << " instructions\n");

  // Kernel names from Tensile et al. routinely exceed 255 bytes, which is
  // the per-component limit on ext4/xfs/tmpfs.  makeSafeBasename() hashes
  // the tail and truncates the head when the full name would blow the
  // budget; the symbol name inside the IR stays untouched, so debug
  // tooling can still resolve the long name from the LLVM module.
  std::string fileStem = makeSafeBasename(kernelName, /*reservedSuffixBytes=*/5);
  std::string irPath  = tmpDir.filePath(fileStem + ".ll");
  std::string asmPath = tmpDir.filePath(fileStem + ".s");

  if (!writeFile(irPath, raised.irText))
    return false;

  static const char *s_dumpInput = std::getenv("HSA_SALMON_DUMP_INPUT");
  if (s_dumpInput && s_dumpInput[0] == '1' && !raised.disasmText.empty())
    writeFile(tmpDir.filePath(fileStem + ".dis"), raised.disasmText);

  std::string llcBin = std::string(LLVM_TOOLS_DIR) + "/llc";
  if (runTool(llcBin, {llcBin, "-march=amdgcn",
                       "-mcpu=" + targetISA,
                       "-filetype=asm", "-o", asmPath, irPath}) != 0) {
    llvm::errs() << "transpiler: llc failed for '" << kernelName << "'\n";
    return false;
  }

  {
    auto asmData = readFile(asmPath);
    if (!result.asmText.empty())
      result.asmText += "\n";
    result.asmText.append(asmData.begin(), asmData.end());
  }

  std::string mcBin = std::string(LLVM_TOOLS_DIR) + "/llvm-mc";
  if (runTool(mcBin, {mcBin, "-triple=amdgcn-amd-amdhsa",
                      "-mcpu=" + targetISA,
                      "-filetype=obj", "-o", objPath, asmPath}) != 0) {
    llvm::errs() << "transpiler: llvm-mc failed for '" << kernelName << "'\n";
    return false;
  }

  return true;
}

// Link one or more relocatable .o files into a shared HSACO.
static bool linkObjects(llvm::ArrayRef<std::string> objPaths,
                        const std::string &hsacoPath) {
  std::string lldBin = std::string(LLVM_TOOLS_DIR) + "/ld.lld";
  llvm::SmallVector<llvm::StringRef, 16> args;
  args.push_back(lldBin);
  args.push_back("-shared");
  args.push_back("-o");
  args.push_back(hsacoPath);
  for (auto &o : objPaths)
    args.push_back(o);
  if (runTool(lldBin, args) != 0) {
    llvm::errs() << "transpiler: ld.lld failed\n";
    return false;
  }
  return true;
}

PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &sourceISA,
                           const std::string &targetISA,
                           const std::string &kernelName,
                           bool enableWritelaneRewrite,
                           bool enableWaveNative,
                           bool enablePermLane16Xor3PartnerRewrite) {
  PipelineResult result;

  auto text = extractTextSection(codeObjectData);
  if (!text.valid) {
    llvm::errs() << "transpiler: Failed to extract .text section\n";
    return result;
  }

  DumpDir tmpDir;
  if (!tmpDir.valid)
    return result;

  {
    static const char *s_dumpInput = std::getenv("HSA_SALMON_DUMP_INPUT");
    if (s_dumpInput && s_dumpInput[0] == '1')
      writeFile(tmpDir.filePath("input.co"), codeObjectData);
  }

  std::string objPath   = tmpDir.filePath("kernel.o");
  std::string hsacoPath = tmpDir.filePath("kernel.hsaco");

  if (!raiseAndCompileKernel(text, codeObjectData, kernelName,
                             sourceISA, targetISA, tmpDir, objPath, result,
                             enableWritelaneRewrite, enableWaveNative,
                             enablePermLane16Xor3PartnerRewrite))
    return result;

  if (!linkObjects({objPath}, hsacoPath))
    return result;

  result.hsaco = readFile(hsacoPath);
  if (result.hsaco.empty()) {
    llvm::errs() << "transpiler: Failed to read HSACO\n";
    return result;
  }

  LLVM_DEBUG(llvm::dbgs() << "transpiler: HSACO generated: " << result.hsaco.size()
                          << " bytes\n");
  result.success = true;
  return result;
}

PipelineResult runPipelineAllKernels(const std::vector<uint8_t> &codeObjectData,
                                     const std::string &sourceISA,
                                     const std::string &targetISA,
                                     bool enableWritelaneRewrite,
                                     bool enableWaveNative,
                                     bool enablePermLane16Xor3PartnerRewrite) {
  PipelineResult result;

  auto kernelNames = listKernelNames(codeObjectData);
  if (kernelNames.empty()) {
    llvm::errs() << "transpiler: No kernels found in code object\n";
    return result;
  }

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raising " << kernelNames.size()
                          << " kernel(s) [" << sourceISA << " -> " << targetISA
                          << "]\n");

  auto text = extractTextSection(codeObjectData);
  if (!text.valid) {
    llvm::errs() << "transpiler: Failed to extract .text section\n";
    return result;
  }

  DumpDir tmpDir;
  if (!tmpDir.valid)
    return result;

  static const char *s_dumpInput = std::getenv("HSA_SALMON_DUMP_INPUT");
  if (s_dumpInput && s_dumpInput[0] == '1')
    writeFile(tmpDir.filePath("input.co"), codeObjectData);

  std::vector<std::string> objPaths;
  for (size_t i = 0; i < kernelNames.size(); ++i) {
    const auto &kName = kernelNames[i];
    std::string objPath = tmpDir.filePath("k" + std::to_string(i) + ".o");

    LLVM_DEBUG(llvm::dbgs() << "transpiler:   [" << (i + 1) << "/"
                            << kernelNames.size() << "] " << kName << " ... ");

    if (!raiseAndCompileKernel(text, codeObjectData, kName,
                               sourceISA, targetISA, tmpDir, objPath, result,
                               enableWritelaneRewrite, enableWaveNative,
                               enablePermLane16Xor3PartnerRewrite)) {
      LLVM_DEBUG(llvm::dbgs() << "FAILED\n");
      result.success = false;
      return result;
    }
    LLVM_DEBUG(llvm::dbgs() << "OK\n");
    objPaths.push_back(std::move(objPath));
  }

  std::string hsacoPath = tmpDir.filePath("merged.hsaco");
  if (!linkObjects(objPaths, hsacoPath))
    return result;

  result.hsaco = readFile(hsacoPath);
  if (result.hsaco.empty()) {
    llvm::errs() << "transpiler: Failed to read merged HSACO\n";
    return result;
  }

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Merged HSACO: " << result.hsaco.size()
                          << " bytes, " << kernelNames.size()
                          << " kernel(s)\n");
  result.success = true;
  return result;
}

} // namespace transpiler
