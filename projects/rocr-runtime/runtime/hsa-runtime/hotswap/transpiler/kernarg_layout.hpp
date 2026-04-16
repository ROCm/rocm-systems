#ifndef HOTSWAP_TRANSPILER_KERNARG_LAYOUT_HPP
#define HOTSWAP_TRANSPILER_KERNARG_LAYOUT_HPP

#include "code_object_utils.hpp"
#include <utility>
#include <vector>

namespace transpiler {

struct KernargParam {
  int byteOffset;
  int byteSize;
  int paramIdx;
  bool isPointer;
};

struct KernargLayout {
  std::vector<KernargParam> params;
  int implicitArgsBase = 0;

  void resolveLoad(
      int byteOffset, int loadBytes,
      std::vector<std::pair<int, int>> &out) const {
    int loadEnd = byteOffset + loadBytes;
    for (auto &p : params) {
      int pEnd = p.byteOffset + p.byteSize;
      if (p.byteOffset >= byteOffset && pEnd <= loadEnd)
        out.push_back({p.byteSize / 4, p.paramIdx});
    }
  }
};

} // namespace transpiler

#endif
