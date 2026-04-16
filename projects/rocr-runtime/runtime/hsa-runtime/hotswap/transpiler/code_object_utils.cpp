#include "code_object_utils.hpp"

#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <fstream>

namespace transpiler {

namespace {
inline uint32_t readU32(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
} // namespace

std::vector<uint8_t> readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    llvm::errs() << "transpiler: Cannot open file: " << path << "\n";
    return {};
  }
  auto pos = f.tellg();
  if (pos < 0) {
    llvm::errs() << "transpiler: tellg failed for: " << path << "\n";
    return {};
  }
  auto sz = static_cast<size_t>(pos);
  f.seekg(0);
  std::vector<uint8_t> data(sz);
  f.read(reinterpret_cast<char *>(data.data()), sz);
  if (!f) {
    llvm::errs() << "transpiler: short read on: " << path << "\n";
    return {};
  }
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
    llvm::errs() << "transpiler: Failed to parse ELF: "
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
  llvm::errs() << "transpiler: .text section not found in ELF\n";
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
    llvm::errs() << "transpiler: listKernelNames: Failed to parse ELF\n";
    return names;
  }
  auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(objOrErr->get());
  if (!elf) {
    llvm::errs() << "transpiler: listKernelNames: Not ELF64LE\n";
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
      uint32_t namesz = readU32(data.data() + off);
      uint32_t descsz = readU32(data.data() + off + 4);
      uint32_t type   = readU32(data.data() + off + 8);
      off += 12;

      uint32_t nameAligned = (namesz + 3) & ~3u;
      uint64_t needed = static_cast<uint64_t>(nameAligned) + descsz;
      if (needed > data.size() - off) break;

      const char *noteName = reinterpret_cast<const char *>(data.data() + off);
      off += nameAligned;

      if (type == 32 && namesz >= 5 &&
          std::memcmp(noteName, "AMDGPU", 6) == 0) {
        llvm::StringRef blob(reinterpret_cast<const char *>(data.data() + off),
                             descsz);
        llvm::msgpack::Document doc;
        if (!doc.readFromBlob(blob, false)) {
          off += (descsz + 3) & ~3u;
          continue;
        }

        auto &root = doc.getRoot();
        if (!root.isMap()) { off += (descsz + 3) & ~3u; continue; }
        auto &rootMap = root.getMap();

        auto kernelsIt = rootMap.find(doc.getNode("amdhsa.kernels"));
        if (kernelsIt == rootMap.end()) { off += (descsz + 3) & ~3u; continue; }

        auto &kernelsNode = kernelsIt->second;
        if (!kernelsNode.isArray()) { off += (descsz + 3) & ~3u; continue; }

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
    llvm::errs() << "transpiler: extractKernelMeta: Failed to parse ELF\n";
    return meta;
  }
  auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(objOrErr->get());
  if (!elf) {
    llvm::errs() << "transpiler: extractKernelMeta: Not ELF64LE\n";
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
      uint32_t namesz = readU32(data.data() + off);
      uint32_t descsz = readU32(data.data() + off + 4);
      uint32_t type   = readU32(data.data() + off + 8);
      off += 12;

      uint32_t nameAligned = (namesz + 3) & ~3u;
      uint64_t needed = static_cast<uint64_t>(nameAligned) + descsz;
      if (needed > data.size() - off) break;

      const char *noteName = reinterpret_cast<const char *>(data.data() + off);
      off += nameAligned;

      if (type == 32 && namesz >= 5 &&
          std::memcmp(noteName, "AMDGPU", 6) == 0) {
        llvm::StringRef blob(reinterpret_cast<const char *>(data.data() + off),
                             descsz);
        llvm::msgpack::Document doc;
        if (!doc.readFromBlob(blob, false)) {
          off += (descsz + 3) & ~3u;
          continue;
        }

        auto &root = doc.getRoot();
        if (!root.isMap()) { off += (descsz + 3) & ~3u; continue; }
        auto &rootMap = root.getMap();

        auto kernelsIt = rootMap.find(doc.getNode("amdhsa.kernels"));
        if (kernelsIt == rootMap.end()) { off += (descsz + 3) & ~3u; continue; }

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

  llvm::errs() << "transpiler: extractKernelMeta: kernel '" << kernelName
               << "' not found in metadata\n";
  return meta;
}

uint64_t findKernelSymbolOffset(const std::vector<uint8_t> &elfData,
                                const std::string &kernelName) {
  auto bufOrErr = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*bufOrErr);
  if (!objOrErr) {
    llvm::errs() << "transpiler: findKernelSymbolOffset: Failed to parse ELF\n";
    return 0;
  }

  uint64_t textBase = UINT64_MAX;
  for (const auto &sec : (*objOrErr)->sections()) {
    auto nameOrErr = sec.getName();
    if (nameOrErr && *nameOrErr == ".text") {
      textBase = sec.getAddress();
      break;
    }
  }
  if (textBase == UINT64_MAX) {
    llvm::errs() << "transpiler: findKernelSymbolOffset: no .text section found\n";
    return 0;
  }

  for (const auto &sym : (*objOrErr)->symbols()) {
    auto nameOrErr = sym.getName();
    if (!nameOrErr) continue;
    if (*nameOrErr == kernelName) {
      auto addrOrErr = sym.getAddress();
      if (!addrOrErr) continue;
      if (*addrOrErr < textBase) {
        llvm::errs() << "transpiler: findKernelSymbolOffset: symbol address 0x"
                     << llvm::utohexstr(*addrOrErr) << " < .text base 0x"
                     << llvm::utohexstr(textBase) << "\n";
        return 0;
      }
      return *addrOrErr - textBase;
    }
  }

  llvm::errs() << "transpiler: findKernelSymbolOffset: symbol '" << kernelName
               << "' not found, defaulting to offset 0\n";
  return 0;
}

} // namespace transpiler
