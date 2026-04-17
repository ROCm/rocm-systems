#ifndef HOTSWAP_TRANSPILER_MC_STATE_HPP
#define HOTSWAP_TRANSPILER_MC_STATE_HPP
// Include LLVM MC headers needed for the unique_ptrs
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace transpiler {

struct MCState {
  const llvm::Target *target = nullptr;
  std::unique_ptr<llvm::MCInstrInfo> instrInfo;
  std::unique_ptr<llvm::MCRegisterInfo> regInfo;
  std::unique_ptr<llvm::MCSubtargetInfo> subtargetInfo;
  std::unique_ptr<const llvm::MCAsmInfo> asmInfo;
  std::unique_ptr<llvm::MCContext> ctx;
  std::unique_ptr<llvm::MCDisassembler> disasm;
  std::unique_ptr<llvm::MCInstPrinter> printer;
};

// The AMDGPU triple shared by every MC object we construct here.
extern const char kAMDGPUTriple[];

bool initMCState(MCState &state, const std::string &targetISA);

// Thin wrapper around Target::createMCSubtargetInfo for the AMDGPU triple.
// Returns a fully populated MCSubtargetInfo (feature bits honour the CPU
// name). Aborts via report_fatal_error on failure — we never want a silent
// empty subtarget.
std::unique_ptr<llvm::MCSubtargetInfo>
buildSubtargetInfo(const llvm::Target &target, llvm::StringRef isa);

std::string getMnemonic(const MCState &mc, const llvm::MCInst &inst);
llvm::StringRef stripEncoding(llvm::StringRef mn);

} // namespace transpiler

#endif
