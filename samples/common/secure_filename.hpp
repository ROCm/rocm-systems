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

#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace common
{
namespace secure
{
/**
 * @brief Safely get and validate a filename from an environment variable
 * 
 * This function retrieves a filename from an environment variable and validates it
 * to prevent path traversal attacks and other security vulnerabilities.
 * 
 * @param env_variable The name of the environment variable to read
 * @param default_filename The default filename to use if env variable is not set or invalid
 * @return A safe filename that has been validated
 */
inline std::string
get_safe_filename(const char* env_variable, const std::string& default_filename)
{
    auto* env_value = getenv(env_variable);
    if(!env_value)
    {
        return default_filename;
    }

    std::string env_filename = env_value;
    
    // Allow special values for stdout/stderr
    if(env_filename == "stdout" || env_filename == "stderr")
    {
        return env_filename;
    }
    
    // Validate and sanitize the filename
    if(env_filename.find("..") != std::string::npos ||        // Path traversal
       env_filename.find("/") == 0 ||                         // Absolute path (Unix)
       env_filename.find("\\") != std::string::npos ||        // Windows path separators
       env_filename.find(":") != std::string::npos ||         // Windows drive letters
       env_filename.empty() ||                                // Empty filename
       env_filename.length() > 255 ||                         // Reasonable length limit
       env_filename.find_first_of("<>:\"|?*") != std::string::npos)  // Windows forbidden chars
    {
        std::cerr << "Warning: Invalid or potentially unsafe filename '" << env_filename 
                 << "' from " << env_variable << ". Using default filename '" 
                 << default_filename << "'.\n";
        return default_filename;
    }
    
    return env_filename;
}

/**
 * @brief Get a safe output filename from the standard ROCPROFILER_SAMPLE_OUTPUT_FILE environment variable
 * 
 * @param default_filename The default filename to use if env variable is not set or invalid
 * @return A safe filename that has been validated
 */
inline std::string
get_safe_output_filename(const std::string& default_filename)
{
    return get_safe_filename("ROCPROFILER_SAMPLE_OUTPUT_FILE", default_filename);
}

}  // namespace secure
}  // namespace common
