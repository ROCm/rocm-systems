// Fixture library for test_stripped_symbol_size.cpp, stripped to .dynsym only.
// Linked before the hidden-visibility object so the hidden function sits in
// the gap after the last known .dynsym symbol.

#include <cstdint>

extern "C" std::int64_t stripped_fixture_static_fn(std::int64_t x);

extern "C" std::int64_t stripped_fixture_fn_a(std::int64_t x)
{
    return x + 1;
}

extern "C" std::int64_t stripped_fixture_fn_b(std::int64_t x)
{
    return stripped_fixture_static_fn(x) + 2;
}

extern "C" void* stripped_fixture_get_static_fn_addr()
{
    return reinterpret_cast<void*>(&stripped_fixture_static_fn);
}
