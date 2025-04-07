
#include <chrono>
#include <ratio>
#include <sstream>
#include <thread>

using clock_type = std::chrono::steady_clock;

int
main(int argc, char** argv)
{
    double _val = 0.0;
    if(argc > 0)
    {
        auto _ss = std::stringstream{};
        _ss << argv[1];
        _ss >> _val;
    }

    intmax_t _nsec = _val * std::nano::den;
    auto     _end  = clock_type::now() + std::chrono::nanoseconds{ _nsec };
    while(clock_type::now() < _end)
    {
        std::this_thread::sleep_for(std::chrono::nanoseconds{ _nsec / 10 });
    }

    return 0;
}
