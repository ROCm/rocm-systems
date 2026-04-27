/*
Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef UTILS_H_
#define UTILS_H_

#include <initializer_list>
#include <string>

// Get command-line option value (single option name)
char* getCmdOption(char ** begin, char ** end, const std::string & option);

// Get command-line option value (multiple option names)
char* getCmdOption(char** begin, char** end, std::initializer_list<const char*> options);

// Check if command-line option exists (single option name)
bool cmdOptionExists(char** begin, char** end, const std::string& option);

// Check if any of the given options exist
bool cmdOptionExists(char** begin, char** end, std::initializer_list<const char*> options);

// Extract architecture and number of GPUs from model description
std::string extractArchAndGpus(const char* desc, int* numGpus = nullptr);

// Check if model description matches GPU architecture
bool matchesArch(const char* desc, const std::string& gpuArch);

// Print help/usage message
void print_help();

// Print available models
void print_available_models(const std::string& gpuArch = "");

#endif // UTILS_H_
