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

std::vector<uint8_t> readFile(const std::string &path);
TextSection extractTextSection(const std::vector<uint8_t> &elfData);

} // namespace ir_proto

#endif
