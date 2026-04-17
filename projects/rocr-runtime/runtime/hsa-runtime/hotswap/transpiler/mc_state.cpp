#include "mc_state.hpp"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

namespace transpiler {

const char kAMDGPUTriple[] = "amdgcn-amd-amdhsa";

std::unique_ptr<MCSubtargetInfo>
buildSubtargetInfo(const Target &target, StringRef isa) {
  Triple triple(kAMDGPUTriple);
  std::unique_ptr<MCSubtargetInfo> sti(
      target.createMCSubtargetInfo(triple, isa, ""));
  if (!sti)
    report_fatal_error(Twine("transpiler: failed to create MCSubtargetInfo "
                             "for ISA '") +
                       isa + "'");
  return sti;
}

bool initMCState(MCState &state, const std::string &targetISA) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  Triple triple(kAMDGPUTriple);
  std::string error;
  state.target = TargetRegistry::lookupTarget(triple, error);
  if (!state.target)
    report_fatal_error(Twine("transpiler: Target lookup for '") +
                       kAMDGPUTriple + "' failed: " + error);

  state.instrInfo.reset(state.target->createMCInstrInfo());
  state.regInfo.reset(state.target->createMCRegInfo(triple));
  state.subtargetInfo = buildSubtargetInfo(*state.target, targetISA);
  state.asmInfo.reset(state.target->createMCAsmInfo(
      *state.regInfo, triple, MCTargetOptions()));
  state.ctx = std::make_unique<MCContext>(triple, state.asmInfo.get(),
                                         state.regInfo.get(),
                                         state.subtargetInfo.get());
  state.disasm.reset(
      state.target->createMCDisassembler(*state.subtargetInfo, *state.ctx));
  state.printer.reset(state.target->createMCInstPrinter(
      triple, 0, *state.asmInfo, *state.instrInfo, *state.regInfo));
  state.printer->setPrintImmHex(true);
  return true;
}

std::string getMnemonic(const MCState &mc, const MCInst &inst) {
  std::string s;
  raw_string_ostream os(s);
  mc.printer->printInst(&inst, 0, "", *mc.subtargetInfo, os);
  StringRef sr(s);
  sr = sr.ltrim();
  return sr.split('\t').first.split(' ').first.str();
}

StringRef stripEncoding(StringRef mn) {
  for (StringRef suffix : {"_e32", "_e64", "_vi"})
    if (mn.ends_with(suffix))
      return mn.drop_back(suffix.size());
  return mn;
}

} // namespace transpiler
