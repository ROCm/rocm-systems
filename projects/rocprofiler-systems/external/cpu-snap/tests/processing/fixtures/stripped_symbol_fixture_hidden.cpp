// Hidden-visibility companion to stripped_symbol_fixture.cpp.
//
// external linkage (needed to be callable from the other translation
// unit) but hidden visibility, so it never gets a .dynsym entry in the
// shared library -- it is not exported.

#include <cstdint>

extern "C" [[gnu::visibility("hidden")]] [[gnu::noinline]] std::int64_t
stripped_fixture_static_fn(std::int64_t x)
{
    auto acc = static_cast<std::uint64_t>(x);
    for(int i = 0; i < 64; ++i)
    {
        acc = (acc * 3U) + static_cast<std::uint64_t>(i);
        acc ^= (acc >> 7U);
    }
    return static_cast<std::int64_t>(acc);
}
