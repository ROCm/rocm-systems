// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "generateCSV.hpp"
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>

namespace rocprofiler
{
namespace tool
{

void
generate_csv(tmp_file& file)
{
    std::string csv_filename = "/home/amdtest/abhinab/iteration_multiplexing/projects/rocprofiler-compute/proof_of_concept/tmp/counters.csv";
    std::cerr << "Generating CSV file: " << csv_filename << std::endl;
    file.filename = std::string("/home/amdtest/abhinab/iteration_multiplexing/projects/rocprofiler-compute/proof_of_concept/tmp/counters.tmp");

    std::streampos pos = std::streampos(0);
    std::vector<std::pair<rocprofiler_counter_id_t, double>> data = file.read<std::pair<rocprofiler_counter_id_t, double>>(pos);
    std::cerr << "Read " << data.size() << " records from tmp file\n";
    for(const auto& record : data)
    {
        std::cerr << record.first.handle << "," << record.second << std::endl;
    }

    // std::string filename = "/home/amdtest/abhinab/iteration_multiplexing/projects/rocprofiler-compute/proof_of_concept/tmp/counters.tmp";
    // std::cerr << "Generating CSV file from tmp file: " << filename << std::endl;

    
}
}  // namespace tool
}  // namespace rocprofiler