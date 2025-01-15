// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <pthread.h>
#include <regex>
#include <signal.h>
#include <sstream>
#include <string_view>
#include <thread>
#include <unistd.h>

int
main(int argc, char** argv)
{
    static const char* _warning = R"warning(

    WWWWWWWW                           WWWWWWWW                                                      iiii                                        !!!
    W::::::W                           W::::::W                                                     i::::i                                      !!:!!
    W::::::W                           W::::::W                                                      iiii                                       !:::!
    W::::::W                           W::::::W                                                                                                 !:::!
     W:::::W           WWWWW           W:::::Waaaaaaaaaaaaa  rrrrr   rrrrrrrrr   nnnn  nnnnnnnn    iiiiiiinnnn  nnnnnnnn       ggggggggg   ggggg!:::!
      W:::::W         W:::::W         W:::::W a::::::::::::a r::::rrr:::::::::r  n:::nn::::::::nn  i:::::in:::nn::::::::nn    g:::::::::ggg::::g!:::!
       W:::::W       W:::::::W       W:::::W  aaaaaaaaa:::::ar:::::::::::::::::r n::::::::::::::nn  i::::in::::::::::::::nn  g:::::::::::::::::g!:::!
        W:::::W     W:::::::::W     W:::::W            a::::arr::::::rrrrr::::::rnn:::::::::::::::n i::::inn:::::::::::::::ng::::::ggggg::::::gg!:::!
         W:::::W   W:::::W:::::W   W:::::W      aaaaaaa:::::a r:::::r     r:::::r  n:::::nnnn:::::n i::::i  n:::::nnnn:::::ng:::::g     g:::::g !:::!
          W:::::W W:::::W W:::::W W:::::W     aa::::::::::::a r:::::r     rrrrrrr  n::::n    n::::n i::::i  n::::n    n::::ng:::::g     g:::::g !:::!
           W:::::W:::::W   W:::::W:::::W     a::::aaaa::::::a r:::::r              n::::n    n::::n i::::i  n::::n    n::::ng:::::g     g:::::g !!:!!
            W:::::::::W     W:::::::::W     a::::a    a:::::a r:::::r              n::::n    n::::n i::::i  n::::n    n::::ng::::::g    g:::::g  !!!
             W:::::::W       W:::::::W      a::::a    a:::::a r:::::r              n::::n    n::::ni::::::i n::::n    n::::ng:::::::ggggg:::::g
              W:::::W         W:::::W       a:::::aaaa::::::a r:::::r              n::::n    n::::ni::::::i n::::n    n::::n g::::::::::::::::g  !!!
               W:::W           W:::W         a::::::::::aa:::ar:::::r              n::::n    n::::ni::::::i n::::n    n::::n  gg::::::::::::::g !!:!!
                WWW             WWW           aaaaaaaaaa  aaaarrrrrrr              nnnnnn    nnnnnniiiiiiii nnnnnn    nnnnnn    gggggggg::::::g  !!!
                                                                                                                                        g:::::g
                                                                                                                            gggggg      g:::::g
                                                                                                                            g:::::gg   gg:::::g
                                                                                                                             g::::::ggg:::::::g
                                                                                                                              gg:::::::::::::g
                                                                                                                                ggg::::::ggg
                                                                                                                                   gggggg


    ROCm Systems Profiler has renamed the "rocprof-sys" executable to "rocprof-sys-instrument".

    This executable only exists to provide this deprecation warning and maintain backwards compatibility for a few releases.
    This executable will soon invoke "rocprof-sys-instrument" with the arguments you just provided after we've given you
    a chance to read this message.

    If you are running this job interactively, please acknowledge that you've read this message and whether you want to continue.
    If you are running this job non-interactively, we will resume executing after ~1 minute unless CI or ROCPROFSYS_CI is defined
    in the environment, in which case, we will throw an error.

    Thanks for using ROCm Systems Profiler and happy optimizing!
    )warning";

    auto _completed    = std::promise<void>{};
    bool _acknowledged = false;
    auto _emit_warning = []() {
        // emit warning
        std::cerr << _warning << std::endl;
    };
    auto _get_env = [](const char* _var) {
        auto* _val = getenv(_var);
        if(_val == nullptr) return false;
        return !std::regex_match(
            _val, std::regex{ "0|false|off|no", std::regex_constants::icase });
    };
    auto _env_failure = [_emit_warning, argv](std::string_view _env_var) {
        // emit warning
        _emit_warning();
        std::cerr
            << "[" << argv[0] << "] Detected " << _env_var
            << " environment variable. Exiting to prevent consuming CI resources. "
               "Use \"rocprof-sys-instrument\" executable instead of \"rocprof-sys\" "
               "to prevent this error."
            << std::endl;
        std::exit(EXIT_FAILURE);
    };
    auto _wait_for_input = [&_completed, &_acknowledged, _emit_warning]() {
        // emit warning and wait for input
        _emit_warning();

        std::cerr << "Do you want to continue? [Y/n] " << std::flush;
        auto _input = char{};
        std::cin.get(_input);

        _input = tolower(_input);
        if(_input == 'n') std::exit(EXIT_SUCCESS);

        _acknowledged = true;
        _completed.set_value();
    };

    for(const auto* itr : { "CI", "ROCPROFSYS_CI" })
    {
        if(_get_env(itr)) _env_failure(itr);
    }

    {
        auto _thr = std::thread{ _wait_for_input };

        _completed.get_future().wait_for(std::chrono::seconds{ 60 });

        if(!_acknowledged)
        {
            std::cerr << "[" << argv[0]
                      << "] No acknowledgement after 1 minute. Continuing..."
                      << std::endl;
            _thr.detach();
        }
        else
            _thr.join();
    }

    // generate the new command
    auto _argv = std::vector<char*>{};
    _argv.emplace_back(
        strdup(std::string{ std::string{ argv[0] } + "-instrument" }.c_str()));
    for(int i = 1; i < argc; ++i)
        _argv.emplace_back(argv[i]);

    // echo the new command for diagnostic purposes
    auto _cmdss = std::stringstream{};
    for(const auto& itr : _argv)
        if(itr) _cmdss << " " << itr;

    auto _cmd = _cmdss.str();
    if(!_cmd.empty())
        std::cerr << "[" << argv[0] << "] Executing: \"" << _cmd.substr(1) << "\"...\n"
                  << std::endl;

    // make sure command ends with nullptr
    _argv.emplace_back(nullptr);

    return execvp(_argv.front(), _argv.data());
}
