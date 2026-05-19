/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLTestUtils.h"

#include <fstream>
#include <iostream>

bool loadFile(const char* filename, std::string& s) {
  size_t size;
  char* str;
  std::fstream f(filename, std::fstream::in | std::fstream::binary);

  if (f.is_open()) {
    size_t fileSize;
    f.seekg(0, std::fstream::end);
    size = fileSize = (size_t)f.tellg();
    f.seekg(0, std::fstream::beg);
    str = new char[size + 1];
    f.read(str, fileSize);
    f.close();
    str[size] = '\0';
    s = str;
    delete[] str;
    return true;
  }
  std::cerr << "Error: failed to open file: " << filename << '\n';
  return false;
}
