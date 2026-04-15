#include "code_object_utils.hpp"

#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <fstream>

namespace ir_proto {

std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    llvm::errs() << "ir_proto: Cannot open file: " << path << "\n";
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
    llvm::errs() << "ir_proto: Failed to parse ELF: "
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
  llvm::errs() << "ir_proto: .text section not found in ELF\n";
  return result;
}

std::vector<std::string> listKernelNames(const std::vector<uint8_t> &elfData) {
  std::vector<std::string> names;

  auto bufOrErr = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*bufOrErr);
  if (!objOrErr) {
    llvm::errs() << "ir_proto: listKernelNames: Failed to parse ELF\n";
    return names;
  }
  auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(objOrErr->get());
  if (!elf) {
    llvm::errs() << "ir_proto: listKernelNames: Not ELF64LE\n";
    return names;
  }

  auto sectionsOrErr = elf->getELFFile().sections();
  if (!sectionsOrErr) return names;

  for (auto &shdr : *sectionsOrErr) {
    if (shdr.sh_type != 7) // SHT_NOTE
      continue;

    auto dataOrErr = elf->getELFFile().getSectionContents(shdr);
    if (!dataOrErr) continue;
    auto data = *dataOrErr;

    size_t off = 0;
    while (off + 12 <= data.size()) {
      uint32_t namesz = *(uint32_t *)(data.data() + off);
      uint32_t descsz = *(uint32_t *)(data.data() + off + 4);
      uint32_t type = *(uint32_t *)(data.data() + off + 8);
      off += 12;

      uint32_t nameAligned = (namesz + 3) & ~3;
      if (off + nameAligned + descsz > data.size()) break;

      const char *noteName = reinterpret_cast<const char *>(data.data() + off);
      off += nameAligned;

      if (type == 32 && namesz >= 5 &&
          std::memcmp(noteName, "AMDGPU", 6) == 0) {
        llvm::StringRef blob(reinterpret_cast<const char *>(data.data() + off),
                             descsz);
        llvm::msgpack::Document doc;
        if (!doc.readFromBlob(blob, false)) {
          off += (descsz + 3) & ~3;
          continue;
        }

        auto &root = doc.getRoot();
        if (!root.isMap()) { off += (descsz + 3) & ~3; continue; }
        auto &rootMap = root.getMap();

        auto kernelsIt = rootMap.find(doc.getNode("amdhsa.kernels"));
        if (kernelsIt == rootMap.end()) { off += (descsz + 3) & ~3; continue; }

        auto &kernelsNode = kernelsIt->second;
        if (!kernelsNode.isArray()) { off += (descsz + 3) & ~3; continue; }

        for (auto &kNode : kernelsNode.getArray()) {
          if (!kNode.isMap()) continue;
          auto &kMap = kNode.getMap();
          auto nameIt = kMap.find(doc.getNode(".name"));
          if (nameIt == kMap.end()) continue;
          names.push_back(nameIt->second.toString());
        }
        return names;
      }
      off += (descsz + 3) & ~3;
    }
  }

  return names;
}

KernelMeta extractKernelMeta(const std::vector<uint8_t> &elfData,
                             const std::string &kernelName) {
  KernelMeta meta;

  auto bufOrErr = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*bufOrErr);
  if (!objOrErr) {
    llvm::errs() << "ir_proto: extractKernelMeta: Failed to parse ELF\n";
    return meta;
  }
  auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(objOrErr->get());
  if (!elf) {
    llvm::errs() << "ir_proto: extractKernelMeta: Not ELF64LE\n";
    return meta;
  }

  // Find .note section
  auto sectionsOrErr = elf->getELFFile().sections();
  if (!sectionsOrErr) return meta;

  for (auto &shdr : *sectionsOrErr) {
    if (shdr.sh_type != 7) // SHT_NOTE
      continue;

    auto dataOrErr = elf->getELFFile().getSectionContents(shdr);
    if (!dataOrErr) continue;
    auto data = *dataOrErr;

    size_t off = 0;
    while (off + 12 <= data.size()) {
      uint32_t namesz = *(uint32_t *)(data.data() + off);
      uint32_t descsz = *(uint32_t *)(data.data() + off + 4);
      uint32_t type = *(uint32_t *)(data.data() + off + 8);
      off += 12;

      uint32_t nameAligned = (namesz + 3) & ~3;
      if (off + nameAligned + descsz > data.size()) break;

      const char *noteName = reinterpret_cast<const char *>(data.data() + off);
      off += nameAligned;

      if (type == 32 && namesz >= 5 &&
          std::memcmp(noteName, "AMDGPU", 6) == 0) {
        llvm::StringRef blob(reinterpret_cast<const char *>(data.data() + off),
                             descsz);
        llvm::msgpack::Document doc;
        if (!doc.readFromBlob(blob, false)) {
          off += (descsz + 3) & ~3;
          continue;
        }

        auto &root = doc.getRoot();
        if (!root.isMap()) { off += (descsz + 3) & ~3; continue; }
        auto &rootMap = root.getMap();

        auto kernelsIt = rootMap.find(doc.getNode("amdhsa.kernels"));
        if (kernelsIt == rootMap.end()) { off += (descsz + 3) & ~3; continue; }

        auto &kernelsNode = kernelsIt->second;
        if (!kernelsNode.isArray()) { off += (descsz + 3) & ~3; continue; }

        for (auto &kNode : kernelsNode.getArray()) {
          if (!kNode.isMap()) continue;
          auto &kMap = kNode.getMap();

          auto nameIt = kMap.find(doc.getNode(".name"));
          if (nameIt == kMap.end()) continue;
          std::string kName = nameIt->second.toString();
          if (kName != kernelName) continue;

          meta.name = kName;

          auto getNodeInt = [](llvm::msgpack::DocNode &n) -> int64_t {
            if (n.getKind() == llvm::msgpack::Type::Int) return n.getInt();
            if (n.getKind() == llvm::msgpack::Type::UInt) return (int64_t)n.getUInt();
            return 0;
          };

          auto kasIt = kMap.find(doc.getNode(".kernarg_segment_size"));
          if (kasIt != kMap.end())
            meta.kernargSegmentSize = getNodeInt(kasIt->second);

          auto gsfIt = kMap.find(doc.getNode(".group_segment_fixed_size"));
          if (gsfIt != kMap.end())
            meta.groupSegmentFixedSize = getNodeInt(gsfIt->second);

          auto mfwIt = kMap.find(doc.getNode(".max_flat_workgroup_size"));
          if (mfwIt != kMap.end())
            meta.maxFlatWorkgroupSize = getNodeInt(mfwIt->second);

          auto argsIt = kMap.find(doc.getNode(".args"));
          if (argsIt != kMap.end() && argsIt->second.isArray()) {
            for (auto &argNode : argsIt->second.getArray()) {
              if (!argNode.isMap()) continue;
              auto &aMap = argNode.getMap();
              KernelArgMeta am;
              auto f = [&](const char *key) -> llvm::msgpack::DocNode * {
                auto it = aMap.find(doc.getNode(key));
                return (it != aMap.end()) ? &it->second : nullptr;
              };
              if (auto *n = f(".name")) am.name = n->toString();
              if (auto *n = f(".offset")) am.offset = getNodeInt(*n);
              if (auto *n = f(".size")) am.size = getNodeInt(*n);
              if (auto *n = f(".value_kind")) am.valueKind = n->toString();
              if (auto *n = f(".address_space")) am.addressSpace = getNodeInt(*n);
              meta.args.push_back(am);
            }
          }
          return meta;
        }
      }
      off += (descsz + 3) & ~3;
    }
  }

  llvm::errs() << "ir_proto: extractKernelMeta: kernel '" << kernelName
               << "' not found in metadata\n";
  return meta;
}

} // namespace ir_proto
