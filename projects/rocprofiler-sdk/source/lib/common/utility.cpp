// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "lib/common/utility.hpp"

#include <unistd.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace common
{
std::vector<std::string>
read_command_line(pid_t _pid)
{
    auto _cmdline = std::vector<std::string>{};
    auto fcmdline = std::stringstream{};
    fcmdline << "/proc/" << _pid << "/cmdline";
    auto ifs = std::ifstream{fcmdline.str().c_str()};
    if(ifs)
    {
        char        cstr;
        std::string sarg;
        while(!ifs.eof())
        {
            ifs >> cstr;
            if(!ifs.eof())
            {
                if(cstr != '\0')
                {
                    sarg += cstr;
                }
                else
                {
                    _cmdline.push_back(sarg);
                    sarg = "";
                }
            }
        }
        ifs.close();
    }

    return _cmdline;
}
}  // namespace common
}  // namespace rocprofiler
