#ifndef HOTSWAP_TRANSPILER_HANDLERS_HPP
#define HOTSWAP_TRANSPILER_HANDLERS_HPP

#include "raise_context.hpp"

namespace llvm {
class MCInstrInfo;
} // namespace llvm

namespace transpiler {

class OpcodeMap;
struct RaiseResult;

// Asserts every MFMA-format opcode the disassembler can decode has a SemOp
// handler entry. See `handle_mfma.cpp` for details.
void verifyMFMACoverage(const llvm::MCInstrInfo &MCII, const OpcodeMap &opcMap);

HandlerResult handleSOPP(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleSMEM(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleSOPC(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleSOP1(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleSOPK(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleSOP2(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleVALU(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleFLAT(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleDS(RaiseContext &ctx, const DecodedInst &di,
                       OpResolver &op, RaiseResult &result);
HandlerResult handleMUBUF(RaiseContext &ctx, const DecodedInst &di,
                          OpResolver &op, RaiseResult &result);
HandlerResult handleMFMA(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);
HandlerResult handleVOPD(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op, RaiseResult &result);

} // namespace transpiler

#endif
