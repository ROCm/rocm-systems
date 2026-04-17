#include "mc_state.hpp"
#include "decoded_inst.hpp"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
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

  // Sanity-check that the hand-sized `DecodedInst::kMaxSrcs` is still large
  // enough for every opcode this MCInstrInfo knows about. We use
  // (NumOperands - NumDefs) as a conservative upper bound on the logical-
  // source count: the raiser's walk can only shrink this by skipping VOP3
  // input modifiers and the DPP/SDWA scalar fallback, never grow it. If a
  // future LLVM adds an encoding wider than our cap, stop here rather than
  // surprise the raiser mid-kernel. Runs once per MCState init.
  {
    const MCInstrInfo &ii = *state.instrInfo;
    unsigned worst = 0;
    for (unsigned opc = 0, end = ii.getNumOpcodes(); opc < end; ++opc) {
      const MCInstrDesc &d = ii.get(opc);
      unsigned nOps = d.getNumOperands();
      unsigned nDefs = d.getNumDefs();
      if (nOps > nDefs && (nOps - nDefs) > worst)
        worst = nOps - nDefs;
    }
    if (worst > DecodedInst::kMaxSrcs)
      report_fatal_error(Twine("transpiler: DecodedInst::kMaxSrcs (") +
                         Twine(DecodedInst::kMaxSrcs) +
                         ") is smaller than the widest AMDGPU operand list "
                         "in MCInstrInfo (NumOperands-NumDefs=" +
                         Twine(worst) + "); bump kMaxSrcs to at least " +
                         Twine(worst));
  }
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
