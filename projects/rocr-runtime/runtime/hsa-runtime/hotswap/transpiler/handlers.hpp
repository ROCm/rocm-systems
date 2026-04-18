#ifndef HOTSWAP_TRANSPILER_HANDLERS_HPP
#define HOTSWAP_TRANSPILER_HANDLERS_HPP

#include "raise_context.hpp"

namespace llvm {
class MCInstrInfo;
} // namespace llvm

namespace transpiler {

class OpcodeMap;

// Asserts every MFMA-format opcode the disassembler can decode has a SemOp
// handler entry. See `handle_mfma.cpp` for details.
void verifyMFMACoverage(const llvm::MCInstrInfo &MCII, const OpcodeMap &opcMap);

HandlerResult handleSOPP(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSMEM(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSOPC(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSOP1(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSOPK(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSOP2(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleVALU(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleFLAT(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleDS(RaiseContext &ctx, const DecodedInst &di,
                       OpResolver &op);
HandlerResult handleMUBUF(RaiseContext &ctx, const DecodedInst &di,
                          OpResolver &op);
HandlerResult handleMFMA(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleVOPD(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);

} // namespace transpiler

#endif
