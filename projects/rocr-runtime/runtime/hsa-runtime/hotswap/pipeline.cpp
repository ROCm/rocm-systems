////////////////////////////////////////////////////////////////////////////////
//
// MLIR-Based Transpilation Pipeline
//
////////////////////////////////////////////////////////////////////////////////

#include "pipeline.hpp"
#include "cross_target.hpp"
#include "emit_assembly.hpp"
#include "lifter.hpp"
#include "wave_width.hpp"

#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/MCAsmParser.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/MLIRContext.h>

using namespace hotswap;

//===----------------------------------------------------------------------===//
// LLVM MC Assembler
//===----------------------------------------------------------------------===//

std::vector<uint8_t> hotswap::assembleToBytes(const std::string &asmText,
                                              const std::string &targetISA) {
  auto mc = initMCState(targetISA);
  if (!mc.valid)
    return {};

  llvm::Triple triple("amdgcn-amd-amdhsa");
  llvm::MCTargetOptions mcOpts;

  mc.Ctx->reset();

  auto buf = llvm::MemoryBuffer::getMemBuffer(asmText, "", false);
  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());

  std::string data;
  auto dataStream = std::make_unique<llvm::raw_string_ostream>(data);
  auto bos = std::make_unique<llvm::buffer_ostream>(*dataStream);

  llvm::MCCodeEmitter *ce =
      mc.target->createMCCodeEmitter(*mc.MCII, *mc.Ctx);
  llvm::MCAsmBackend *mab =
      mc.target->createMCAsmBackend(*mc.STI, *mc.MRI, mcOpts);
  if (!ce || !mab)
    return {};

#if LLVM_VERSION_MAJOR > 20
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      mc.target->createMCObjectStreamer(
          triple, *mc.Ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *mc.STI));
#else
  auto streamer = std::unique_ptr<llvm::MCStreamer>(
      mc.target->createMCObjectStreamer(
          triple, *mc.Ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
          mab->createObjectWriter(*bos),
          std::unique_ptr<llvm::MCCodeEmitter>(ce), *mc.STI,
          mcOpts.MCRelaxAll, mcOpts.MCIncrementalLinkerCompatible, false));
#endif
  if (!streamer)
    return {};

  auto parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(srcMgr, *mc.Ctx, *streamer, *mc.MAI));
  auto tap = std::unique_ptr<llvm::MCTargetAsmParser>(
      mc.target->createMCAsmParser(*mc.STI, *parser, *mc.MCII, mcOpts));
  if (!tap)
    return {};
  parser->setTargetParser(*tap);

  bool failed = parser->Run(true);

  tap.reset();
  parser.reset();
  streamer.reset();
  bos.reset();
  dataStream->flush();

  if (failed || data.size() < 4)
    return {};

  // The assembled output is a full ELF object. Extract the .text section.
  // Parse minimal ELF to find .text offset/size.
  const uint8_t *elfData = reinterpret_cast<const uint8_t *>(data.data());
  size_t elfSize = data.size();

  // Minimal ELF64 header parsing
  if (elfSize < 64)
    return {};
  uint16_t e_shentsize = *reinterpret_cast<const uint16_t *>(elfData + 58);
  uint16_t e_shnum = *reinterpret_cast<const uint16_t *>(elfData + 60);
  uint16_t e_shstrndx = *reinterpret_cast<const uint16_t *>(elfData + 62);
  uint64_t e_shoff = *reinterpret_cast<const uint64_t *>(elfData + 40);

  if (e_shoff == 0 || e_shnum == 0 || e_shentsize < 64)
    return {};
  if (e_shoff + (uint64_t)e_shnum * e_shentsize > elfSize)
    return {};

  // Get section name string table
  const uint8_t *shstrtab_hdr = elfData + e_shoff + e_shstrndx * e_shentsize;
  uint64_t shstrtab_offset =
      *reinterpret_cast<const uint64_t *>(shstrtab_hdr + 24);
  uint64_t shstrtab_size =
      *reinterpret_cast<const uint64_t *>(shstrtab_hdr + 32);

  for (uint16_t i = 0; i < e_shnum; ++i) {
    const uint8_t *shdr = elfData + e_shoff + i * e_shentsize;
    uint32_t name_idx = *reinterpret_cast<const uint32_t *>(shdr);
    uint64_t sh_offset = *reinterpret_cast<const uint64_t *>(shdr + 24);
    uint64_t sh_size = *reinterpret_cast<const uint64_t *>(shdr + 32);

    if (name_idx < shstrtab_size) {
      const char *name =
          reinterpret_cast<const char *>(elfData + shstrtab_offset + name_idx);
      if (std::string(name) == ".text" && sh_size > 0 &&
          sh_offset + sh_size <= elfSize) {
        return std::vector<uint8_t>(elfData + sh_offset,
                                    elfData + sh_offset + sh_size);
      }
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Full Pipeline
//===----------------------------------------------------------------------===//

PipelineResult hotswap::runPipeline(const uint8_t *sourceBytes,
                                    size_t sourceSize,
                                    const std::string &sourceISA,
                                    const std::string &targetISA,
                                    const std::string &kernelName) {
  PipelineResult result;

  // Step 1: Lift binary to waveasm MLIR
  mlir::MLIRContext ctx;
  Lifter lifter(ctx);

  auto module = lifter.lift(
      llvm::ArrayRef<uint8_t>(sourceBytes, sourceSize), sourceISA, kernelName);
  if (!module) {
    result.errorMessage = "Lift failed: could not create MLIR module";
    return result;
  }

  auto &liftStats = lifter.getStats();
  result.stats.totalInstructions = liftStats.totalInstructions;
  result.stats.liftedInstructions = liftStats.liftedInstructions;
  result.stats.rawFallbacks = liftStats.rawFallbacks;
  result.stats.failedDisassembly = liftStats.failedDisassembly;

  // Step 2: Cross-target mnemonic mapping
  if (sourceISA != targetISA) {
    if (mlir::failed(retargetModule(*module, sourceISA, targetISA))) {
      result.errorMessage = "Cross-target mapping failed";
      return result;
    }
  }

  // Step 3: Wave width translation (wave32 → wave64)
  bool needsWaveWidth = false;
  llvm::StringRef srcRef(sourceISA), tgtRef(targetISA);
  if ((srcRef.starts_with("gfx12") || srcRef.starts_with("gfx1250")) &&
      tgtRef.starts_with("gfx9"))
    needsWaveWidth = true;

  if (needsWaveWidth) {
    if (mlir::failed(widenExecMask(*module))) {
      result.errorMessage = "Wave width translation failed";
      return result;
    }
  }

  // Step 4: Emit assembly text
  result.assemblyText = emitAssembly(*module);

  // Step 5: Assemble to bytes. Skip when cross-ISA since the full output
  // may contain preamble instructions from the source ISA that can't
  // assemble on the target. Callers do core extraction + rebuild instead.
  if (sourceISA == targetISA)
    result.bytes = assembleToBytes(".text\n" + result.assemblyText, targetISA);
  result.success = true;

  return result;
}
