#ifndef ASTER_PROTO_CODE_OBJECT_UTILS_HPP
#define ASTER_PROTO_CODE_OBJECT_UTILS_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace aster_proto {

struct TextSection {
  std::vector<uint8_t> bytes;
  uint64_t offset = 0;
  uint64_t size = 0;
  bool valid = false;
};

std::vector<uint8_t> readFile(const std::string &path);
TextSection extractTextSection(const std::vector<uint8_t> &elfData);

} // namespace aster_proto

#endif // ASTER_PROTO_CODE_OBJECT_UTILS_HPP
