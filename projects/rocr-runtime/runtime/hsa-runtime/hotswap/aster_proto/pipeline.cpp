#include "pipeline.hpp"
#include "lifter.hpp"

#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Target/ASM/TranslateModule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <fstream>

using namespace mlir;
namespace amdgcn = mlir::aster::amdgcn;

namespace aster_proto {

PipelineResult runPipeline(const std::vector<uint8_t> &textBytes,
                           const std::string &targetISA,
                           const std::string &kernelName) {
  PipelineResult pResult;

  // Step 1: Lift binary → Aster IR
  MLIRContext ctx;
  ctx.loadDialect<amdgcn::AMDGCNDialect>();
  ctx.loadDialect<arith::ArithDialect>();

  auto liftResult = liftToAster(ctx, textBytes, targetISA, kernelName);
  if (!liftResult.success) {
    llvm::errs() << "aster_proto: Lifting failed\n";
    return pResult;
  }

  pResult.liftedCount = liftResult.liftedCount;
  pResult.unsupportedCount = liftResult.unsupportedCount;
  pResult.unsupportedMnemonics = liftResult.unsupportedMnemonics;

  llvm::errs() << "aster_proto: Lifted " << liftResult.liftedCount
               << " instructions, " << liftResult.unsupportedCount
               << " unsupported\n";
  if (!liftResult.unsupportedMnemonics.empty()) {
    llvm::errs() << "aster_proto: Unsupported mnemonics:";
    for (auto &m : liftResult.unsupportedMnemonics)
      llvm::errs() << " " << m;
    llvm::errs() << "\n";
  }

  // Print the MLIR for debugging
  llvm::errs() << "\n=== Aster MLIR (lifted) ===\n";
  liftResult.module->print(llvm::errs());
  llvm::errs() << "\n=== End Aster MLIR ===\n\n";

  // Step 2: Find the amdgcn.module inside the builtin module
  auto *builtinMod = liftResult.module.get();
  amdgcn::ModuleOp amdgcnMod;
  builtinMod->walk([&](Operation *op) {
    if (auto mod = dyn_cast<amdgcn::ModuleOp>(op)) {
      amdgcnMod = mod;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!amdgcnMod) {
    llvm::errs() << "aster_proto: No amdgcn.module found in lifted IR\n";
    return pResult;
  }

  // Step 3: Translate MLIR → assembly using Aster's translateModule
  std::string asmStr;
  llvm::raw_string_ostream asmOs(asmStr);
  auto translateResult =
      amdgcn::target::translateModule(amdgcnMod, asmOs, true);
  pResult.assemblyText = asmStr;
  if (failed(translateResult)) {
    llvm::errs() << "aster_proto: translateModule returned failure\n";
    if (pResult.assemblyText.empty()) {
      llvm::errs() << "aster_proto: No assembly produced\n";
      return pResult;
    }
    llvm::errs() << "aster_proto: Assembly was produced despite failure ("
                 << pResult.assemblyText.size() << " bytes), attempting HSACO\n";
  } else {
    llvm::errs() << "aster_proto: Assembly generated ("
                 << pResult.assemblyText.size() << " bytes)\n";
  }

  // Dump assembly when ASTER_DUMP_ASM environment variable is set
  if (std::getenv("ASTER_DUMP_ASM"))
    llvm::errs() << "\n=== Generated Assembly ===\n"
                 << pResult.assemblyText
                 << "\n=== End Assembly ===\n\n";

  // Step 4: Assemble → object → HSACO via external llvm-mc + ld.lld
  auto asmPath = std::string("/tmp/aster_proto.s");
  auto objPath = std::string("/tmp/aster_proto.o");
  auto hsacoPath = std::string("/tmp/aster_proto.hsaco");

  // Filter assembly: remove directives incompatible with architected flat
  // scratch on GFX942, fix kernarg size to include hidden args, and inject
  // hidden arg metadata for HIP runtime compatibility.
  //
  // The original kernel binary loads hidden_group_size_x from offset 44 in the
  // kernarg segment. Aster only emits explicit args (28 bytes), so we must
  // extend the kernarg_size to 288 and add hidden arg entries to the metadata
  // YAML so the HIP runtime allocates and populates the full kernarg buffer.
  std::string filteredAsm;
  {
    llvm::StringRef asmRef(pResult.assemblyText);
    llvm::SmallVector<llvm::StringRef> lines;
    asmRef.split(lines, '\n');
    llvm::raw_string_ostream fos(filteredAsm);
    for (auto &line : lines) {
      auto trimmed = line.ltrim();
      // Only remove directives that GFX942 with architected flat scratch
      // rejects. Keep all directives that configure user SGPRs (kernarg ptr,
      // dispatch ptr), system SGPRs (workgroup IDs), and system VGPRs
      // (workitem IDs) — the kernel needs these to function.
      if (trimmed.starts_with(".amdhsa_user_sgpr_private_segment_buffer"))
        continue;
      // Fix kernarg size: Aster emits 28 (explicit only), but the binary
      // code loads hidden args at offset 44+. Must match original layout.
      if (trimmed.starts_with(".amdhsa_kernarg_size")) {
        fos << "    .amdhsa_kernarg_size 288\n";
        continue;
      }
      fos << line << "\n";
    }
    // Patch the YAML metadata: replace kernarg_segment_size and inject
    // hidden args after the last explicit arg entry.
    std::string patched;
    llvm::StringRef filtRef(filteredAsm);
    auto pos = filtRef.find(".kernarg_segment_size: 28");
    if (pos != llvm::StringRef::npos) {
      llvm::raw_string_ostream patchOs(patched);
      patchOs << filtRef.substr(0, pos);
      patchOs << ".kernarg_segment_size: 288";
      patchOs << filtRef.substr(pos + strlen(".kernarg_segment_size: 28"));
      filteredAsm = std::move(patched);
    }
    // Inject hidden arg entries into the YAML args list
    llvm::StringRef filtRef2(filteredAsm);
    auto lastByVal = filtRef2.find(".value_kind: by_value");
    if (lastByVal == llvm::StringRef::npos)
      lastByVal = filtRef2.find(".value_kind:     by_value");
    if (lastByVal != llvm::StringRef::npos) {
      auto insertPoint = filtRef2.find('\n', lastByVal);
      if (insertPoint != llvm::StringRef::npos) {
        std::string injected;
        llvm::raw_string_ostream injOs(injected);
        injOs << filtRef2.substr(0, insertPoint + 1);
        injOs << "      - .offset:         32\n"
              << "        .size:           4\n"
              << "        .value_kind:     hidden_block_count_x\n"
              << "      - .offset:         36\n"
              << "        .size:           4\n"
              << "        .value_kind:     hidden_block_count_y\n"
              << "      - .offset:         40\n"
              << "        .size:           4\n"
              << "        .value_kind:     hidden_block_count_z\n"
              << "      - .offset:         44\n"
              << "        .size:           2\n"
              << "        .value_kind:     hidden_group_size_x\n"
              << "      - .offset:         46\n"
              << "        .size:           2\n"
              << "        .value_kind:     hidden_group_size_y\n"
              << "      - .offset:         48\n"
              << "        .size:           2\n"
              << "        .value_kind:     hidden_group_size_z\n"
              << "      - .offset:         50\n"
              << "        .size:           2\n"
              << "        .value_kind:     hidden_remainder_x\n"
              << "      - .offset:         52\n"
              << "        .size:           2\n"
              << "        .value_kind:     hidden_remainder_y\n"
              << "      - .offset:         54\n"
              << "        .size:           2\n"
              << "        .value_kind:     hidden_remainder_z\n"
              << "      - .offset:         72\n"
              << "        .size:           8\n"
              << "        .value_kind:     hidden_global_offset_x\n"
              << "      - .offset:         80\n"
              << "        .size:           8\n"
              << "        .value_kind:     hidden_global_offset_y\n"
              << "      - .offset:         88\n"
              << "        .size:           8\n"
              << "        .value_kind:     hidden_global_offset_z\n"
              << "      - .offset:         96\n"
              << "        .size:           2\n"
              << "        .value_kind:     hidden_grid_dims\n";
        injOs << filtRef2.substr(insertPoint + 1);
        filteredAsm = std::move(injected);
      }
    }
  }

  // Write filtered assembly to temp file
  {
    std::error_code ec;
    llvm::raw_fd_ostream f(asmPath, ec);
    if (ec) {
      llvm::errs() << "aster_proto: Cannot write asm file: "
                   << ec.message() << "\n";
      return pResult;
    }
    f << filteredAsm;
  }

  // Use the system ROCm llvm-mc which has full GFX942 support
  std::string mcBin = "/opt/rocm/llvm/bin/llvm-mc";
  std::string mcCmd = mcBin + " -triple=amdgcn-amd-amdhsa"
      " -mcpu=" + targetISA + " -filetype=obj " + asmPath + " -o " + objPath;
  int ret = std::system(mcCmd.c_str());
  if (ret != 0) {
    llvm::errs() << "aster_proto: llvm-mc failed (exit " << ret << ")\n";
    return pResult;
  }

  std::string lldBin = std::string(LLVM_TOOLS_DIR) + "/ld.lld";
  std::string lldCmd = lldBin + " -shared " + objPath +
                        " -o " + hsacoPath;
  ret = std::system(lldCmd.c_str());
  if (ret != 0) {
    llvm::errs() << "aster_proto: ld.lld failed (exit " << ret << ")\n";
    return pResult;
  }

  // Read back HSACO
  std::ifstream hf(hsacoPath, std::ios::binary | std::ios::ate);
  if (!hf.is_open()) {
    llvm::errs() << "aster_proto: Cannot read HSACO\n";
    return pResult;
  }
  auto hsacoSize = hf.tellg();
  hf.seekg(0);
  pResult.hsaco.resize(hsacoSize);
  hf.read(pResult.hsaco.data(), hsacoSize);

  pResult.success = true;
  llvm::errs() << "aster_proto: HSACO generated (" << pResult.hsaco.size()
               << " bytes)\n";
  return pResult;
}

} // namespace aster_proto
