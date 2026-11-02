/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared by the src/misc/*symbols.cc tests: a builder's table of function pointers,
// each slot named by the symbol and version it is loaded from.

#ifndef RCCL_TEST_HOST_MISC_DL_SYMBOL_TABLE_H_
#define RCCL_TEST_HOST_MISC_DL_SYMBOL_TABLE_H_

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "fakes/dl_fakes.h"

struct DlSym {
  const char* name;
  const char* version;
  size_t offset;  // of the slot in the table
};

template <typename Table>
void* Slot(const Table& table, const DlSym& sym) {
  void* p;
  std::memcpy(&p, reinterpret_cast<const char*>(&table) + sym.offset, sizeof(p));
  return p;
}

// A distinct fake address per slot.
inline void* AddrOf(const DlSym& sym) { return reinterpret_cast<void*>(0x1000 + sym.offset); }

inline void* AddLibraryExporting(ScriptedDl& dl, const std::string& name, const std::vector<DlSym>& exports) {
  void* handle = dl.AddLibrary(name);
  for (const DlSym& sym : exports) dl.Export(handle, sym.name, sym.version, AddrOf(sym));
  return handle;
}

inline std::vector<std::string> OpenedFiles(const ScriptedDl& dl) {
  std::vector<std::string> files;
  for (const auto& open : dl.opens) files.push_back(open.file);
  return files;
}

template <typename Table>
void ExpectAllSlotsNull(const Table& table, const std::vector<DlSym>& syms) {
  for (const DlSym& sym : syms) EXPECT_EQ(Slot(table, sym), nullptr) << sym.name;
}

#endif  // RCCL_TEST_HOST_MISC_DL_SYMBOL_TABLE_H_
