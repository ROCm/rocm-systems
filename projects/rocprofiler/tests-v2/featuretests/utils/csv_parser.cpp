/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "csv_parser.h"

namespace rocprofiler {
namespace tests {
namespace utility {

// Tokenize a comma separated string and saves result in vector
int CSVParser::GetTockenizedString(std::string str, std::vector<std::string>& stringVec,
                                   char delim) {
  char* token = strtok(const_cast<char*>(str.c_str()), &delim);
  while (token) {
    std::string temp = token;
    stringVec.push_back(temp);
    token = strtok(NULL, &delim);
  }

  return stringVec.size();
}

// Get map for collected counters
countermap& CSVParser::GetCounterMap() { return counter_map_; }

// Read counter value based on row-column
std::string* CSVParser::ReadCounter(uint32_t row, uint32_t col) {
  auto itr = counter_map_.find(row);

  if (itr != counter_map_.end()) {
    std::map<uint32_t, std::string>& rowmap = itr->second;
    auto it = rowmap.find(col);
    if (it != rowmap.end()) {
      return &(it->second);
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

// Parses CSV file and saves result in a map
void CSVParser::ParseCSV(const char* path) {
  FILE* pFile = fopen(path, "r");

  if (pFile) {
    // Relocate the file pointer on the stream.
    fseek(pFile, 0, SEEK_END);
    // returns the current file position of the specified stream
    // with respect to the starting of the file
    uint32_t uSize = ftell(pFile);
    // Make the position pointer of the file pFile point
    // to the beginning of the file
    rewind(pFile);

    char* fileBuffer = new char[uSize];
    size_t size = fread(fileBuffer, 1, uSize, pFile);
    if (size < 0) std::cerr << "Incorrect File!" << std::endl;

    std::map<uint32_t, std::string> rowmap;
    uint32_t uiIndex = 1;

    char* pBegin = fileBuffer;
    char* pEnd = strchr(pBegin, '\n');

    // The beginning of the second line, discarding the first line
    pBegin = pEnd + 1;
    pEnd = strchr(pBegin, '\n');

    while (pEnd) {
      std::string tmp;
      tmp.insert(0, pBegin, pEnd - pBegin);
      assert(!tmp.empty());
      // Store the string of each line in the map,
      // the key is the sequence number,
      // and the value is the string
      rowmap[uiIndex++] = tmp;

      pBegin = pEnd + 1;
      pEnd = strchr(pBegin, '\n');
    }
    // clear buffers
    delete[] fileBuffer;
    fileBuffer = nullptr;
    pBegin = nullptr;
    pEnd = nullptr;

    auto itr = rowmap.begin();
    for (; itr != rowmap.end(); ++itr) {
      std::vector<std::string> countervec;
      std::map<uint32_t, std::string> rowmap_tmp;
      assert(GetTockenizedString(itr->second, countervec) > 0);

      std::vector<std::string>::size_type idx = 0;
      for (; idx != countervec.size(); ++idx) {
        rowmap_tmp[idx + 1] = countervec[idx];
      }
      counter_map_[itr->first] = rowmap_tmp;
    }
  }

  fclose(pFile);
}

//  Parses profiler output buffer and saves result in a map
void CSVParser::ParseCSV(std::string buffer) {
  std::vector<char> buff(buffer.begin(), buffer.end());
  char* pBegin = &buff[0];
  char* pEnd = strchr(pBegin, '\n');

  // The beginning of the second line, discarding the first line
  pBegin = pEnd + 1;
  pEnd = strchr(pBegin, '\n');
  std::map<uint32_t, std::string> rowmap;
  uint32_t uiIndex = 1;

  while (pEnd) {
    std::string tmp;
    tmp.insert(0, pBegin, pEnd - pBegin);
    // Store the string of each line in the map,
    // the key is the sequence number,
    // and the value is the string
    rowmap[uiIndex++] = tmp;

    pBegin = pEnd + 1;
    pEnd = strchr(pBegin, '\n');
  }

  // clear buffers
  buff.clear();
  pBegin = nullptr;
  pEnd = nullptr;

  auto itr = rowmap.begin();
  for (; itr != rowmap.end(); ++itr) {
    std::vector<std::string> countervec;
    std::map<uint32_t, std::string> rowmap_tmp;
    GetTockenizedString(itr->second, countervec);

    std::vector<std::string>::size_type idx = 0;
    for (; idx != countervec.size(); ++idx) {
      rowmap_tmp[idx + 1] = countervec[idx];
    }
    counter_map_[itr->first] = rowmap_tmp;
  }
}
}  // namespace utility
}  // namespace tests
}  // namespace rocprofiler
