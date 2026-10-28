#pragma once

/**
 * @file symbol_name.hpp
 * @brief Display-name cleanup for symbol names: demangling and noise removal.
 */

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

#include <cxxabi.h>

namespace cpu_snap::processing::symbols
{

class symbol_name
{
public:
    /** Keeps the parameter list intact (callers want signatures); only
     *  strips clone tags and "(anonymous namespace)::" noise. */
    static void clean(std::string& fn)
    {
        if(const auto pos = fn.find(" [clone "); pos != std::string::npos)
        {
            fn.erase(pos);
        }

        const std::string anon = "(anonymous namespace)::";
        for(std::size_t pos = fn.find(anon); pos != std::string::npos;
            pos             = fn.find(anon, pos))
        {
            fn.erase(pos, anon.size());
        }
    }

    /** Itanium demangle via abi::__cxa_demangle; returns @p mangled
     *  unchanged if demangling fails. */
    [[nodiscard]] static std::string demangle(std::string_view mangled)
    {
        struct free_deleter
        {
            // Frees the malloc'd buffer __cxa_demangle allocates; no
            // alternative to std::free for that buffer.
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            void operator()(char* ptr) const noexcept { std::free(ptr); }
        };

        int                                       status = 0;
        const std::unique_ptr<char, free_deleter> demangled{ abi::__cxa_demangle(
            std::string{ mangled }.c_str(), nullptr, nullptr, &status) };
        if(status == 0 && demangled != nullptr)
        {
            return demangled.get();
        }
        return std::string{ mangled };
    }
};

}  // namespace cpu_snap::processing::symbols
