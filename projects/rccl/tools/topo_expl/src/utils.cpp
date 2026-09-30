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

#include "utils.h"
#include "model_descs.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

char* getCmdOption(char ** begin, char ** end, const std::string & option) {
    const size_t opt_len = option.length();
    for (char** itr = begin; itr != end; ++itr) {
        if (std::strcmp(*itr, option.c_str()) == 0) {
            if (itr + 1 != end)
                return *(itr + 1);
            return nullptr;
        }
        if (opt_len < std::strlen(*itr) &&
            std::strncmp(*itr, option.c_str(), opt_len) == 0 && (*itr)[opt_len] == '=') {
            return *itr + opt_len + 1;
        }
    }
    return nullptr;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    const size_t opt_len = option.length();
    for (char** it = begin; it != end; ++it) {
        if (std::strcmp(*it, option.c_str()) == 0)
            return true;
        if (opt_len < std::strlen(*it) &&
            std::strncmp(*it, option.c_str(), opt_len) == 0 && (*it)[opt_len] == '=')
            return true;
    }
    return false;
}

char* getCmdOption(char** begin, char** end, std::initializer_list<const char*> options) {
    for (const char* opt : options) {
        char* val = getCmdOption(begin, end, std::string(opt));
        if (val) return val;
    }
    return nullptr;
}

bool cmdOptionExists(char** begin, char** end, std::initializer_list<const char*> options) {
    for (const char* opt : options) {
        if (cmdOptionExists(begin, end, std::string(opt))) return true;
    }
    return false;
}

// Extract architecture and number of GPUs from model description
std::string extractArchAndGpus(const char* desc, int* numGpus) {
    std::string descStr(desc);
    if (numGpus) *numGpus = 0;
    size_t start = 0;
    while (start < descStr.length() &&
           std::isspace(static_cast<unsigned char>(descStr[start]))) {
        start++;
    }
    
    size_t gfxPos = descStr.find("gfx", start);
    if (gfxPos != std::string::npos) {
        if (numGpus && gfxPos > start) {
            std::string numStr = descStr.substr(start, gfxPos - start);
            while (!numStr.empty() &&
                   std::isspace(static_cast<unsigned char>(numStr.back()))) {
                numStr.pop_back();
            }
            if (!numStr.empty()) {
                *numGpus = std::atoi(numStr.c_str());
            }
        }
        size_t end = gfxPos + 3;
        while (end < descStr.length() &&
               std::isdigit(static_cast<unsigned char>(descStr[end]))) {
            end++;
        }
        return descStr.substr(gfxPos, end - gfxPos);
    }
    return "";
}

bool matchesArch(const char* desc, const std::string& gpuArch) {
    if (gpuArch.empty()) return true;
    std::string modelArch = extractArchAndGpus(desc, nullptr);
    return modelArch == gpuArch;
}

void print_help() {
    printf("Usage: ./topo_expl [-m|--model model_id] [-n|--nodes numNodes=1] [-a|--arch gpu_arch] [-t] [-h]\n");
    printf("  -m, --model id    : Model ID (required to run a model)\n");
    printf("  -n, --nodes N     : Number of nodes (default: 1)\n");
    printf("  -a, --arch arch   : Show available models by GPU architecture (e.g., gfx906, gfx908, gfx910, gfx942, gfx950)\n");
    printf("  -t, --test        : Run test suite (all models with 1,2,4,8,16 nodes by default)\n");
    printf("  -h, --help        : Display this help message and all available models\n");
    printf("\n");
    printf("Usage examples:\n");
    printf("  ./topo_expl                                    # View all available models\n");
    printf("  ./topo_expl -a gfx942                          # View available models for gfx942\n");
    printf("  NCCL_DEBUG=VERSION ./topo_expl -m 5            # Run model 5 with 1 node\n");
    printf("  ./topo_expl --model 55 --nodes 2               # Run model 55 with 2 nodes\n");
    printf("  ./topo_expl -m 10 -n 4                         # Run model 10 with 4 nodes\n");
    printf("\n");
    printf("Test Usage: ./topo_expl -t [--include-models=ids] [--include-nodes=counts]\n");
    printf("                           [--exclude-models=ids] [--exclude-nodes=counts]\n");
    printf("  --include-models  : Comma-separated model IDs to test (e.g., 0,1,2,4,8)\n");
    printf("  --include-nodes   : Comma-separated node counts to test (e.g., 1,4,8)\n");
    printf("  --exclude-models  : Comma-separated model IDs to exclude from test (e.g., 57,59)\n");
    printf("  --exclude-nodes   : Comma-separated node counts to exclude from test (e.g., 16)\n");
    printf("\n");
    printf("Test examples:\n");
    printf("  ./topo_expl -t                                 # Test all models with 1,2,4,8,16 nodes\n");
    printf("  ./topo_expl -t --include-models=0,1,2          # Test only models 0,1,2\n");
    printf("  ./topo_expl -t --include-nodes=1,4             # Test all models with 1 and 4 nodes\n");
    printf("  ./topo_expl -t --exclude-models=57,59          # Test all except models 57,59\n");
    printf("  ./topo_expl -t --exclude-nodes=16              # Test all models except 16-node configs\n");
    printf("\n");
    printf("For additional information and detailed documentation, see README.md\n");
    printf("\n");
}

void print_available_models(const std::string& gpuArch) {
    if (!gpuArch.empty()) {
        printf("Available models for %s:\n", gpuArch.c_str());
    } else {
        printf("All available models:\n");
    }
    printf("┌──────┬─────────────────────────────────────┬────────────────────────────────────────────┐\n");
    printf("│  ID  │ Model Description                   │ Filename                                   │\n");
    printf("├──────┼─────────────────────────────────────┼────────────────────────────────────────────┤\n");
    
    bool foundAny = false;
    for (int i = 0; i < num_models; i++) {
        if (matchesArch(model_descs[i].description, gpuArch)) {
            printf("│ %4d │ %-35s │ %-42s │\n", 
                   i, model_descs[i].description, model_descs[i].filename);
            foundAny = true;
        }
    }

    printf("└──────┴─────────────────────────────────────┴────────────────────────────────────────────┘\n");
    
    if (!foundAny && !gpuArch.empty()) {
        printf("\nNo models found for GPU architecture '%s'\n", gpuArch.c_str());
        printf("Available GPU architectures: gfx906, gfx908, gfx910, gfx942, gfx950\n");
    } else if (gpuArch.empty()) {
        printf("\nTo view available models by GPU architecture, use: ./topo_expl -a <gpu_arch> or --arch <gpu_arch>\n");
        printf("Example: ./topo_expl -a gfx942\n");
    }
}
