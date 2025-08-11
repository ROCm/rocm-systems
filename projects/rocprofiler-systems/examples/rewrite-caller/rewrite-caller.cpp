#include <cstdio>
#include <cstdlib>
#include <string>

#define NOINLINE __attribute__((noinline))

int num_calls;

void NOINLINE
inner()
{
    ++num_calls;
}

void NOINLINE
outer()
{
    inner();
}

int
main(int argc, char** argv)
{
    int ncalls = 10;
    if(argc > 1) ncalls = atol(argv[1]);

    std::string _name = argv[0];
    auto        _pos  = _name.find_last_of('/');
    if(_pos != std::string::npos) _name = _name.substr(_pos + 1);

    for(int i = 0; i < ncalls; ++i)
    {
        outer();
    }

    printf("[%s] number of calls made = %d\n", _name.c_str(), num_calls);
}
