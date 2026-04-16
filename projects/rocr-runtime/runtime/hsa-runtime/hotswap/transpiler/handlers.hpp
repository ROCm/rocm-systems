#ifndef HOTSWAP_TRANSPILER_HANDLERS_HPP
#define HOTSWAP_TRANSPILER_HANDLERS_HPP

#include "raise_context.hpp"

namespace transpiler {

struct RaiseResult;

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
