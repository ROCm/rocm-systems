#include "code_object_utils.hpp"

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

namespace mir_proto {

std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    llvm::errs() << "mir_proto: Cannot open file: " << path << "\n";
    return {};
  }
  auto sz = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> data(sz);
  f.read(reinterpret_cast<char *>(data.data()), sz);
  return data;
}

TextSection extractTextSection(const std::vector<uint8_t> &elfData) {
  TextSection result;
  auto bufOrErr = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*bufOrErr);
  if (!objOrErr) {
    llvm::errs() << "mir_proto: Failed to parse ELF: "
                 << llvm::toString(objOrErr.takeError()) << "\n";
    return result;
  }
  auto &obj = *objOrErr;
  for (const auto &sec : obj->sections()) {
    auto nameOrErr = sec.getName();
    if (!nameOrErr)
      continue;
    if (*nameOrErr == ".text") {
      auto contentsOrErr = sec.getContents();
      if (!contentsOrErr)
        continue;
      result.bytes.assign(contentsOrErr->begin(), contentsOrErr->end());
      result.offset = sec.getAddress();
      result.size = sec.getSize();
      result.valid = true;
      return result;
    }
  }
  llvm::errs() << "mir_proto: .text section not found in ELF\n";
  return result;
}

} // namespace mir_proto
