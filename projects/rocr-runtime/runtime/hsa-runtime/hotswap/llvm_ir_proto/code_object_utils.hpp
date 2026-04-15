#ifndef LLVM_IR_PROTO_CODE_OBJECT_UTILS_HPP
#define LLVM_IR_PROTO_CODE_OBJECT_UTILS_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace ir_proto {

struct TextSection {
  std::vector<uint8_t> bytes;
  uint64_t offset = 0;
  uint64_t size = 0;
  bool valid = false;
};

struct KernelArgMeta {
  std::string name;
  int offset = 0;
  int size = 0;
  std::string valueKind;
  int addressSpace = -1;
};

struct KernelMeta {
  std::string name;
  int kernargSegmentSize = 0;
  std::vector<KernelArgMeta> args;

  int implicitArgsBase() const {
    int maxEnd = 0;
    for (auto &a : args) {
      if (a.valueKind.rfind("hidden_", 0) == 0)
        continue;
      int end = a.offset + a.size;
      if (end > maxEnd) maxEnd = end;
    }
    return (maxEnd + 7) & ~7;
  }
};

std::vector<uint8_t> readFile(const std::string &path);
TextSection extractTextSection(const std::vector<uint8_t> &elfData);
std::vector<std::string> listKernelNames(const std::vector<uint8_t> &elfData);
KernelMeta extractKernelMeta(const std::vector<uint8_t> &elfData,
                             const std::string &kernelName);

} // namespace ir_proto

#endif
