/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <fstream>
#include <fcntl.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#define dup _dup
#define dup2 _dup2
#define fd_close _close
#define unlink _unlink
#define STDERR_FD _fileno(stderr)
#define OPEN_FLAGS (_O_WRONLY | _O_CREAT | _O_TRUNC)
#define OPEN_MODE (_S_IREAD | _S_IWRITE)
#define open _open
#else
#include <unistd.h>
#define fd_close close
#define STDERR_FD STDERR_FILENO
#define OPEN_FLAGS (O_WRONLY | O_CREAT | O_TRUNC)
#define OPEN_MODE 0644
#endif

// Class to capture all stderr output (HIP logging uses stderr)
class OutCapture {
private:
    std::stringstream captured_stream_;
    std::streambuf* cerr_backup_;
    int stderr_backup_;
    std::string temp_file_;

    static std::string getTempFilePath() {
#ifdef _WIN32
        char temp_path[MAX_PATH];
        if (GetTempPathA(MAX_PATH, temp_path)) {
            return std::string(temp_path) + "hip_stderr_capture.txt";
        }
        // Fallback to current directory
        return "hip_stderr_capture.txt";
#else
        return "/tmp/hip_stderr_capture.txt";
#endif
    }

public:
    OutCapture() : temp_file_(getTempFilePath()) {
        // Backup original cerr stream buffer (HIP logging uses stderr)
        cerr_backup_ = std::cerr.rdbuf();

        // Backup original stderr file descriptor
        stderr_backup_ = dup(STDERR_FD);
    }

    void startCapture() {
        // Clear any previous content
        captured_stream_.str("");
        captured_stream_.clear();

        // Redirect std::cerr to our stringstream
        std::cerr.rdbuf(captured_stream_.rdbuf());

        // Redirect stderr file descriptor to temp file (for fprintf to stderr)
        int temp_fd = open(temp_file_.c_str(), OPEN_FLAGS, OPEN_MODE);
        if (temp_fd != -1) {
            dup2(temp_fd, STDERR_FD);
            fd_close(temp_fd);
        }
    }

    std::string stopCapture() {
        // Restore original cerr stream
        std::cerr.rdbuf(cerr_backup_);

        // Restore original stderr file descriptor
        dup2(stderr_backup_, STDERR_FD);

        // Read from temp file (captures fprintf(stderr) output from HIP logging)
        std::ifstream temp_file(temp_file_);
        std::string file_content;
        if (temp_file.is_open()) {
            std::string line;
            while (std::getline(temp_file, line)) {
                file_content += line + "\n";
            }
            temp_file.close();
        }

        // Combine both captures: C++ streams and file descriptor output
        std::string stream_content = captured_stream_.str();
        std::string total_output = stream_content + file_content;

        // Clean up temp file
        unlink(temp_file_.c_str());

        return total_output;
    }

    ~OutCapture() {
        // Ensure everything is restored
        std::cerr.rdbuf(cerr_backup_);
        dup2(stderr_backup_, STDERR_FD);
        fd_close(stderr_backup_);
        unlink(temp_file_.c_str());
    }
};
