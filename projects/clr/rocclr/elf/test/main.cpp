/* Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All Rights Reserved.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <elf/elf.hpp>
#include <string>
#include <utils/flags.hpp>
#include <utils/debug.hpp>

using namespace amd::ELFIO;

static constexpr uint32_t target_ = 11;
static constexpr char comment_[] = "comment text";
static constexpr size_t commentSize_ = strlen(comment_) + 1;

// Elf::RODATA,         ".rodata",         1, SHT_PROGBITS, SHF_ALLOC,
static const amd::Elf::SymbolInfo rodataSymbolInfos_[] = {
    {".rodata", nullptr, 0, "data__fmetadata", "fmetatdata", strlen("fmetatdata") + 1},
    {".rodata", nullptr, 0, "data__amdil", "amdildata", strlen("amdildata") + 1},
    {".rodata", nullptr, 0, "data__metadata", "metadata", strlen("metadata") + 1},
    {".rodata", nullptr, 0, "data__header", "header", strlen("header") + 1},
    {".rodata", nullptr, 0, "data__global", "global", strlen("global") + 1},
    {".rodata", nullptr, 0, "data__randome0", "xu\0e\0\0l", sizeof("xu\0e\0\0l")},  // binary
    {".rodata", nullptr, 0, "data__randome1", "\0j\0\0w\0", sizeof("\0j\0\0w\0")},  // binary
};

static constexpr size_t rodataSymbolInfosSize_ =
    sizeof(rodataSymbolInfos_) / sizeof(rodataSymbolInfos_[0]);

// Elf::COMMENT,        ".comment",        1, SHT_PROGBITS, 0,
static const amd::Elf::SymbolInfo commentSymbolInfos_[] = {
    {".comment", nullptr, 0, "compile", "-g -I/opt/include", strlen("-g -I/opt/include") + 1},
    {".comment", nullptr, 0, "link", "-g -l/opt/rocm/lib", strlen("-g -l/opt/rocm/lib") + 1},
};
static constexpr size_t commentSymbolInfosSize_ =
    sizeof(commentSymbolInfos_) / sizeof(commentSymbolInfos_[0]);

struct NoteInfo {
  const char* noteName;
  const char* noteDesc;
  size_t descSize;
};

static constexpr NoteInfo noteInfos_[] = {
    {"notename0", "sjfasdfe2Afs", strlen("sjfasdfe2Afs") + 1},
    {"notename1", "AsdmvdfFfkd", strlen("AsdmvdfFfkd") + 1},
    {"notename2", "d\0kelH\0D", sizeof("d\0kelH\0D")},  // binary
    {"notename3", "\0F\0kA\0", sizeof("\0F\0kA\0")},    // binary
};

static const size_t noteInfosSize_ = sizeof(noteInfos_) / sizeof(noteInfos_[0]);

bool set(amd::Elf* elf) {
  if (!elf->setTarget(target_, amd::Elf::CPU_PLATFORM)) {
    LogError("elf->setTarget() failed");
    return false;
  }

  if (!elf->setType(ET_EXEC)) {
    LogError("elf->elf() failed");
    return false;
  }

  if (!elf->addSection(amd::Elf::COMMENT, comment_, commentSize_)) {
    LogError("elf->addSection() failed");
    return false;
  }

  size_t i = 0;
  LogInfo("writing rodataSymbolInfo");

  for (i = 0; i < rodataSymbolInfosSize_; i++) {
    auto& info = rodataSymbolInfos_[i];
    if (!elf->addSymbol(amd::Elf::RODATA, info.sym_name.c_str(), info.address, info.size)) {
      LogPrintfError("elf->addSymbol(RODATA) failed at index %zu", i);
      return false;
    }
  }

  LogInfo("Succeeded");
  LogInfo("writing commentSymbolInfo");

  for (i = 0; i < commentSymbolInfosSize_; i++) {
    auto& info = commentSymbolInfos_[i];
    if (!elf->addSymbol(amd::Elf::COMMENT, info.sym_name.c_str(), info.address, info.size)) {
      LogPrintfError("elf->addSymbol(COMMENT) failed at index %zu", i);
      return false;
    }
  }

  LogInfo("Succeeded");
  LogInfo("writing noteInfos");

  for (i = 0; i < noteInfosSize_; i++) {
    auto& info = noteInfos_[i];
    if (!elf->addNote(info.noteName, info.noteDesc, info.descSize)) {
      LogPrintfError("elf->addNote() failed at index %zu", i);
      return false;
    }
  }

  LogInfo("Succeeded");

  return true;
}

bool verify(amd::Elf* elf) {
  uint16_t machine = amd::Elf::OCL_TARGETS_LAST;
  amd::Elf::ElfPlatform platform = amd::Elf::LAST_PLATFORM;
  if (!elf->getTarget(machine, platform)) {
    LogError("elf->getTarget() failed");
    return false;
  }

  LogPrintfInfo("getTarget(machine=%u, platform=%d)", machine, platform);

  if (machine != target_) {
    LogPrintfError("machine(%u) != target_(%d)", machine, target_);
    return false;
  }

  if (platform != amd::Elf::CPU_PLATFORM) {
    LogPrintfError("platform(%d) != CAL_PLATFORM(%d)", platform, amd::Elf::CPU_PLATFORM);
    return false;
  }

  uint16_t type = ET_NONE;

  if (!elf->getType(type)) {
    LogError("elf->elf() failed");
    return false;
  }

  LogPrintfInfo("getType(%u)", type);

  if (type != ET_EXEC) {
    LogError("type != ET_EXEC");
    return false;
  }

  char* buffer = nullptr;
  size_t size = 0;

  if (!elf->getSection(amd::Elf::COMMENT, &buffer, &size)) {
    LogError("elf->getSection(COMMENT) failed");
    return false;
  }

  LogPrintfInfo("getSection(COMMENT, buffer=%s, size=%zu)", buffer, size);

  if (size < commentSize_ || memcmp(comment_, buffer, commentSize_) != 0) {
    LogPrintfError("Not matched section: size = %zu, buffer = %s, expected: %zu, %s", size, buffer,
                   commentSize_, comment_);
    return false;
  }

  LogInfo("Reading rodataSymbolInfo");

  size_t i = 0;
  buffer = nullptr;
  size = 0;
  for (i = 0; i < rodataSymbolInfosSize_; i++) {
    auto& info = rodataSymbolInfos_[i];
    if (!elf->getSymbol(amd::Elf::RODATA, info.sym_name.c_str(), &buffer, &size)) {
      LogPrintfError("elf->getSymbol(RODATA, %s) failed at index %zu", info.sym_name.c_str(), i);
      return false;
    }
    LogPrintfInfo("getSymbol(amd::Elf::RODATA, sym_name=%s, buffer=%s, size=%zu)",
                  info.sym_name.c_str(), buffer, size);  // Will possibly print part of buffer

    if (size != info.size || memcmp(buffer, info.address, info.size)) {
      LogPrintfError("Not matched symbol(%s): size = %zu, buff = %s, expected: %zu, %s",
                     info.sym_name.c_str(), size, buffer, info.size, info.address);
      return false;
    }
  }

  LogInfo("Succeeded");
  LogInfo("reading commentSymbolInfo");

  buffer = nullptr;
  size = 0;
  for (i = 0; i < commentSymbolInfosSize_; i++) {
    auto& info = commentSymbolInfos_[i];
    if (!elf->getSymbol(amd::Elf::COMMENT, info.sym_name.c_str(), &buffer, &size)) {
      LogPrintfError("elf->getSymbol(COMMENT, %s) failed at index %zu", info.sym_name.c_str(), i);
      return false;
    }
    LogPrintfInfo("getSymbol(COMMENT, sym_name=%s, buffer=%s, size=%zu)", info.sym_name.c_str(),
                  buffer, size);  // Will possibly print part of buffer
    if (size != info.size || memcmp(buffer, info.address, info.size)) {
      LogPrintfError("Not matched symbol(%s): size = %zu, buff = %s, expected: %zu, %s",
                     info.sym_name.c_str(), size, buffer, info.size, info.address);
      return false;
    }
  }

  // Test another way
  auto symbolNum = elf->getSymbolNum();
  if (symbolNum != (rodataSymbolInfosSize_ + commentSymbolInfosSize_)) {
    LogPrintfError(
        "Not matched: symbolNum(%u) != rodataSymbolInfosSize_(%u) +"
        " commentSymbolInfosSize_(%u)",
        symbolNum, rodataSymbolInfosSize_, commentSymbolInfosSize_);
    return false;
  }

  for (i = 0; i < rodataSymbolInfosSize_; i++) {
    auto& info = rodataSymbolInfos_[i];
    amd::Elf::SymbolInfo symInfo;

    if (!elf->getSymbolInfo(i, &symInfo)) {
      LogPrintfError("getSymbolInfo(%zu) failed", i);
      return false;
    }
    LogPrintfInfo(
        "getSymbolInfo(%zu): amd::Elf::RODATA: sec_name=%s, sym_name=%s, "
        "address=%s, size=%lu, sec_addr=%s, sec_size=%lu)",
        i, symInfo.sec_name.c_str(), symInfo.sym_name.c_str(),
        symInfo.address,  // Will possibly print part of buffer
        symInfo.size, symInfo.sec_addr, symInfo.sec_size);
    if (symInfo.sec_name == info.sec_name && symInfo.sym_name == info.sym_name &&
        symInfo.size == info.size && ::memcmp(symInfo.address, info.address, info.size) == 0) {
      continue;
    }
    LogPrintfError("getSymbolInfo(%zu) returned not matched", i);
    return false;
  }

  for (; i < symbolNum; i++) {
    auto& info = commentSymbolInfos_[i - rodataSymbolInfosSize_];
    amd::Elf::SymbolInfo symInfo;

    if (!elf->getSymbolInfo(i, &symInfo)) {
      LogPrintfError("getSymbolInfo(%zu) failed", i);
      return false;
    }
    LogPrintfInfo(
        "getSymbolInfo(%zu): amd::Elf::COMMENT: sec_name=%s, sym_name=%s, "
        "address=%s, size=%lu, sec_addr=%s, sec_size=%lu)",
        i, symInfo.sec_name.c_str(), symInfo.sym_name.c_str(),
        symInfo.address,  // Will possibly print part of buffer
        symInfo.size, symInfo.sec_addr, symInfo.sec_size);
    if (symInfo.sec_name == info.sec_name && symInfo.sym_name == info.sym_name &&
        symInfo.size == info.size && ::memcmp(symInfo.address, info.address, info.size) == 0) {
      continue;
    }
    LogPrintfError("getSymbolInfo(%zu) returned not matched", i);
    return false;
  }

  LogInfo("Succeeded");
  LogError("Reading noteInfos");

  buffer = nullptr;
  size = 0;
  for (i = 0; i < noteInfosSize_; i++) {
    auto& info = noteInfos_[i];
    if (!elf->getNote(info.noteName, &buffer, &size)) {
      LogPrintfError("elf->getNote(%s) failed at index %zu", info.noteName, i);
      return false;
    }
    // Will possibly print part of buffer
    LogPrintfInfo("getNote(noteName=%s, buffer=%s, size=%zu)", info.noteName, buffer, size);
    if (size != info.descSize || memcmp(buffer, info.noteDesc, info.descSize)) {
      LogPrintfError("Not matched note(%s): size = %zu, buff = %s, expected: %zu, %s",
                     info.noteName, size, buffer, info.descSize, info.noteDesc);
      return false;
    }
  }

  LogPrintfInfo("%s: Succeeded", __func__);

  return true;
}

bool test(unsigned char eclass = ELFCLASS64, const char* outFile = nullptr) {
  amd::Elf* writer = new amd::Elf(eclass, nullptr, 0, outFile, amd::Elf::ELF_C_WRITE);
  amd::Elf* reader = nullptr;
  bool ret = false;
  do {
    if ((writer == nullptr) || !writer->isSuccessful()) {
      LogError("Creating writter ELF object failed");
      break;
    }

    // Writing
    if (!set(writer)) {
      break;
    }

    // Verifying
    if (!verify(writer)) {
      break;
    }

    char* buff = nullptr;
    unsigned long len = 0;
    if (writer->dumpImage(&buff, &len)) {
      LogPrintfInfo("dumpImage succeed: buff=%p, len=%u)", buff, len);

      reader = new amd::Elf(eclass, buff, len, nullptr, amd::Elf::ELF_C_READ);

      delete[] buff;

      if ((reader == nullptr) || !reader->isSuccessful()) {
        LogError("Creating reader ELF object failed");
        break;
      }

      ret = verify(reader);

      delete reader;
    }
  } while (false);

  if (writer) {
    delete writer;
  }
  if (reader) {
    delete reader;
  }
  LogPrintfError("%s(%s, %s): %s", __func__, eclass == ELFCLASS64 ? "ELFCLASS64" : "ELFCLASS32",
                 outFile ? outFile : "nullptr", ret ? "Succeeded" : "Failed");
  return ret;
}

int main() {
  bool ret = false;
  amd::Flag::init();
  unsigned char eclass = LP64_SWITCH(ELFCLASS32, ELFCLASS64);
  const char* outFile = eclass == ELFCLASS32 ? "elf32.bin" : "elf64.bin";

  ret = test(eclass, outFile);
  printf("%s: test(%s, %s) %s!\n", __func__, eclass == ELFCLASS32 ? "ELFCLASS32" : "ELFCLASS64",
         outFile, ret ? "Succeeded" : "Failed");

  if (ret) {
    ret = test(eclass, nullptr);
    printf("%s: test(%s, nullptr) %s!\n", __func__,
           eclass == ELFCLASS32 ? "ELFCLASS32" : "ELFCLASS64", ret ? "Succeeded" : "Failed");
  }
  return 0;
}
