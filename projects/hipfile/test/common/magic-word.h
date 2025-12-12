/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

static const char *message = "Ah ah ah, you didn't say the magic word!\n"
                             "                    __\n"
                             "                   |_ |\n"
                             "                   |_ |\n"
                             "           __ __ __|_ |\n"
                             "          |  |  |__|__|\n"
                             "          |__|__(___   \\\n"
                             "          |         \\   \\\n"
                             "          |             /\n"
                             "           \\           /\n"
                             "\n"
                             "hipFile tests should be run using ctest. ctest runs each test in an\n"
                             "isolated process, which prevents side effects from affecting subsequent\n"
                             "tests. Running the test binaries directly may result in test errors if\n"
                             "running multiple tests.\n";

class MagicWord {
public:
    MagicWord() : print_message{false}
    {
        pid_t             parent = getppid();
        std::stringstream parent_comm_path;
        parent_comm_path << "/proc/" << parent << "/comm";

        std::ifstream is{parent_comm_path.str()};
        std::string   parent_comm;
        std::getline(is, parent_comm);

        print_message =
            parent_comm.find("cmake") == std::string::npos && parent_comm.find("ctest") == std::string::npos;

        if (print_message) {
            std::cerr << message << "\n";
        }
    }

    ~MagicWord()
    {
        if (print_message) {
            std::cerr << "\n" << message;
        }
    }

private:
    bool print_message;
};

extern MagicWord mw;
