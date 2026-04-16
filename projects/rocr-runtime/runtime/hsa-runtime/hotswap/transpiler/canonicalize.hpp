#ifndef HOTSWAP_TRANSPILER_CANONICALIZE_HPP
#define HOTSWAP_TRANSPILER_CANONICALIZE_HPP

#include "llvm/ADT/StringRef.h"
#include <string>

namespace transpiler {

std::string canonicalizeMnemonic(llvm::StringRef mn);

} // namespace transpiler

#endif
