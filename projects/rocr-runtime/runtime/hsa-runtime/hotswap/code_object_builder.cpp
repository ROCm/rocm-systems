////////////////////////////////////////////////////////////////////////////////
//
// Code Object Builder
//
////////////////////////////////////////////////////////////////////////////////

#include "code_object_builder.hpp"

#include <llvm/Support/raw_ostream.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace hotswap;

//===----------------------------------------------------------------------===//
// File I/O
//===----------------------------------------------------------------------===//

std::vector<uint8_t> hotswap::readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

std::string hotswap::readFileAsString(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    return {};
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

//===----------------------------------------------------------------------===//
// ELF .text extraction
//===----------------------------------------------------------------------===//

TextSection hotswap::extractTextSection(const std::vector<uint8_t> &elfData) {
  TextSection result;

  if (elfData.size() < 64)
    return result;

  // Verify ELF magic
  if (elfData[0] != 0x7f || elfData[1] != 'E' || elfData[2] != 'L' ||
      elfData[3] != 'F')
    return result;

  auto read16 = [&](size_t off) -> uint16_t {
    uint16_t v;
    memcpy(&v, elfData.data() + off, 2);
    return v;
  };
  auto read64 = [&](size_t off) -> uint64_t {
    uint64_t v;
    memcpy(&v, elfData.data() + off, 8);
    return v;
  };
  auto read32 = [&](size_t off) -> uint32_t {
    uint32_t v;
    memcpy(&v, elfData.data() + off, 4);
    return v;
  };

  uint64_t e_shoff = read64(40);
  uint16_t e_shentsize = read16(58);
  uint16_t e_shnum = read16(60);
  uint16_t e_shstrndx = read16(62);

  if (e_shoff == 0 || e_shnum == 0 || e_shentsize < 64)
    return result;
  if (e_shoff + (uint64_t)e_shnum * e_shentsize > elfData.size())
    return result;

  uint64_t shstrtab_offset = read64(e_shoff + e_shstrndx * e_shentsize + 24);
  uint64_t shstrtab_size = read64(e_shoff + e_shstrndx * e_shentsize + 32);

  for (uint16_t i = 0; i < e_shnum; ++i) {
    size_t shdr = e_shoff + i * e_shentsize;
    uint32_t name_idx = read32(shdr);
    uint64_t sh_offset = read64(shdr + 24);
    uint64_t sh_size = read64(shdr + 32);

    if (name_idx < shstrtab_size) {
      const char *name =
          reinterpret_cast<const char *>(elfData.data() + shstrtab_offset + name_idx);
      if (strcmp(name, ".text") == 0 && sh_size > 0 &&
          sh_offset + sh_size <= elfData.size()) {
        result.bytes.assign(elfData.data() + sh_offset,
                            elfData.data() + sh_offset + sh_size);
        result.offset = sh_offset;
        result.size = sh_size;
        result.valid = true;
        return result;
      }
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Assembly splicing
//===----------------------------------------------------------------------===//

std::string hotswap::spliceInstructions(const std::string &deviceAsmTemplate,
                                        const std::string &newInstructions,
                                        const std::string &kernelSymbol) {
  // Strategy: find the kernel label line, then find the s_endpgm line,
  // and replace everything in between with the new instructions.
  std::istringstream stream(deviceAsmTemplate);
  std::string result;
  std::string line;

  // The kernel label is: "kernelSymbol:" at the start of a line (with possible
  // whitespace). We look for the label followed by instructions.
  std::string labelPattern = kernelSymbol + ":";
  bool inKernelBody = false;
  bool foundEndpgm = false;
  bool alreadySpliced = false;

  while (std::getline(stream, line)) {
    if (!alreadySpliced) {
      // Look for the kernel function label
      auto stripped = line;
      auto firstNonSpace = stripped.find_first_not_of(" \t");
      if (firstNonSpace != std::string::npos)
        stripped = stripped.substr(firstNonSpace);

      if (!inKernelBody && stripped.find(labelPattern) == 0) {
        result += line + "\n";
        inKernelBody = true;
        continue;
      }

      if (inKernelBody && !foundEndpgm) {
        // Skip original instructions until we see s_endpgm
        auto content = stripped;
        // Strip comments
        auto commentPos = content.find(';');
        if (commentPos != std::string::npos)
          content = content.substr(0, commentPos);
        auto contentTrimmed = content;
        auto lastNonSpace = contentTrimmed.find_last_not_of(" \t\r\n");
        if (lastNonSpace != std::string::npos)
          contentTrimmed = contentTrimmed.substr(0, lastNonSpace + 1);

        if (contentTrimmed == "s_endpgm") {
          // Insert our new instructions (which include s_endpgm)
          result += newInstructions;
          // Ensure trailing newline
          if (!newInstructions.empty() && newInstructions.back() != '\n')
            result += "\n";
          foundEndpgm = true;
          alreadySpliced = true;
          continue;
        }
        // Skip this original instruction line
        continue;
      }
    }

    result += line + "\n";
  }

  if (!alreadySpliced) {
    llvm::errs() << "ERROR: Could not find kernel body to splice in template. "
                 << "Kernel symbol: " << kernelSymbol << "\n";
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Core extraction from pipeline output
//===----------------------------------------------------------------------===//

std::string
hotswap::extractCoreFromPipelineOutput(const std::string &fullAssembly) {
  std::istringstream stream(fullAssembly);
  std::string line;
  bool pastBranch = false;
  std::string core;

  while (std::getline(stream, line)) {
    auto stripped = line;
    auto first = stripped.find_first_not_of(" \t");
    if (first != std::string::npos)
      stripped = stripped.substr(first);

    if (!pastBranch) {
      if (stripped.find("s_cbranch_execz") == 0)
        pastBranch = true;
      continue;
    }

    // Stop before the label that precedes s_endpgm (e.g., "L_br0:")
    if (!stripped.empty() && stripped.back() == ':' &&
        stripped.find("s_") == std::string::npos)
      break;

    // Also stop at s_endpgm itself
    if (stripped == "s_endpgm")
      break;

    core += line + "\n";
  }

  return core;
}

//===----------------------------------------------------------------------===//
// Core-only splice for cross-ISA
//===----------------------------------------------------------------------===//

std::string
hotswap::spliceCoreInstructions(const std::string &deviceAsmTemplate,
                                const std::string &translatedCore,
                                const std::string &kernelSymbol) {
  // Strategy: in the template, find the s_cbranch_execz line.
  // Keep everything up to and including that line (the preamble).
  // Skip original core instructions until the branch target label (e.g. .LBB0_2:).
  // Insert translated core instructions.
  // Keep the target label and everything after it (s_endpgm + metadata).
  std::istringstream stream(deviceAsmTemplate);
  std::string result;
  std::string line;
  bool foundBranch = false;
  bool spliced = false;

  while (std::getline(stream, line)) {
    if (spliced) {
      result += line + "\n";
      continue;
    }

    if (!foundBranch) {
      result += line + "\n";
      auto stripped = line;
      auto first = stripped.find_first_not_of(" \t");
      if (first != std::string::npos)
        stripped = stripped.substr(first);
      if (stripped.find("s_cbranch_execz") == 0)
        foundBranch = true;
      continue;
    }

    // We're past the branch — skip original core lines until we see the
    // target label (starts with "." and ends with ":")
    auto stripped = line;
    auto first = stripped.find_first_not_of(" \t");
    if (first != std::string::npos)
      stripped = stripped.substr(first);

    if (!stripped.empty() && stripped[0] == '.' && stripped.back() == ':') {
      // This is the target label. Insert our translated core, then this label.
      result += translatedCore;
      if (!translatedCore.empty() && translatedCore.back() != '\n')
        result += "\n";
      result += line + "\n";
      spliced = true;
      continue;
    }
    // Skip original instruction lines
  }

  if (!spliced) {
    llvm::errs() << "ERROR: Could not find core splice point in template. "
                 << "Kernel symbol: " << kernelSymbol << "\n";
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Code object rebuild
//===----------------------------------------------------------------------===//

std::vector<uint8_t>
hotswap::rebuildCodeObject(const std::string &assemblyText,
                           const std::string &targetISA,
                           const std::string &llvmBinDir) {
  // Write assembly to a temp file
  std::string asmPath = "/tmp/hotswap_mve_rebuild.s";
  std::string objPath = "/tmp/hotswap_mve_rebuild.o";
  std::string coPath = "/tmp/hotswap_mve_rebuild.co";

  {
    std::ofstream f(asmPath);
    if (!f) {
      llvm::errs() << "ERROR: Cannot write assembly to " << asmPath << "\n";
      return {};
    }
    f << assemblyText;
  }

  // Step 1: Assemble with llvm-mc
  std::string mcCmd = llvmBinDir + "/llvm-mc"
                      " -triple=amdgcn-amd-amdhsa"
                      " -mcpu=" + targetISA +
                      " -filetype=obj"
                      " -o " + objPath +
                      " " + asmPath + " 2>&1";
  int ret = system(mcCmd.c_str());
  if (ret != 0) {
    llvm::errs() << "ERROR: llvm-mc failed (exit " << ret << ")\n"
                 << "  Command: " << mcCmd << "\n";
    // Read and print errors
    std::string errOut = readFileAsString(objPath + ".err");
    if (!errOut.empty())
      llvm::errs() << errOut << "\n";
    return {};
  }

  // Step 2: Link with ld.lld
  std::string ldCmd = llvmBinDir + "/ld.lld"
                      " -shared"
                      " -o " + coPath +
                      " " + objPath + " 2>&1";
  ret = system(ldCmd.c_str());
  if (ret != 0) {
    llvm::errs() << "ERROR: ld.lld failed (exit " << ret << ")\n"
                 << "  Command: " << ldCmd << "\n";
    return {};
  }

  // Read the resulting code object
  auto result = readFile(coPath);
  if (result.empty()) {
    llvm::errs() << "ERROR: Rebuilt code object is empty\n";
    return {};
  }

  return result;
}
