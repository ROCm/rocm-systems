
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
#ifndef TESTS_FEATURETESTS_PROFILER_UTILS_CSV_PARSER_H_
#define TESTS_FEATURETESTS_PROFILER_UTILS_CSV_PARSER_H_

#include <assert.h>
#include <stdio.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis Implementation of a CSV Parser class for Profiler output
 * when running to collect counters.Subsequent counter collection tests
 * can use this to parse different applications.
 */
/* --------------------------------------------------------------------------*/
namespace rocprofiler {
namespace tests {
namespace utility {
using countermap = std::map<uint32_t, std::map<uint32_t, std::string>>;

class CSVParser {
 public:
  CSVParser() {}

  /**
   * Parses CSV file and saves result in a map
   *
   * Stores csv data as a 2-D array in row-column format
   * Skips first line of input csv as it only contains field names
   *
   * @param[in] path Pointer to the csv path.
   *
   * @return Returns 0 on success and -1 on error.
   */
  void ParseCSV(const char* path);

  /**
   * Parses profiler output buffer and saves result in a map
   *
   * Stores csv data as a 2-D array in row-column format
   * Skips first line of input csv as it only contains field names
   *
   * @param[in] path Pointer to the csv path.
   *
   * @return Returns 0 on success and -1 on error.
   */
  void ParseCSV(std::string buffer);

  /**
   * Read counter value based on row-column
   *
   * @param[in] row row number to be read in csv.
   *
   * @param[in] col column to be read in csv.(i.e: counter value)
   *
   * @return If found, returns counter value as a string pointer, nullptr
   * otherwise.
   */
  std::string* ReadCounter(uint32_t row, uint32_t col);

  /**
   * Tokenize a comma separated string and saves result in vector
   *
   * @param[in] str input string to be tokenized
   *
   * @param[in] csvtable vector to store tokenized values
   *
   * @param[in] delim delimitre used for tokenizing
   *
   * @return returns vector size of delimitd values
   */
  int GetTockenizedString(std::string str, std::vector<std::string>& csvtable, char delim = ',');

  /**
   * A getter for a map of collected counters
   * *
   * @return Returns a map of collected counters.
   */
  countermap& GetCounterMap();

 private:
  // map for counter collection
  countermap counter_map_;
};
}  // namespace utility
}  // namespace tests
}  // namespace rocprofiler

using rocprofiler::tests::utility::countermap;
using rocprofiler::tests::utility::CSVParser;
#endif  // TESTS_FEATURETESTS_PROFILER_UTILS_CSV_PARSER_H_
