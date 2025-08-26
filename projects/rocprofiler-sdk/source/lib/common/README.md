# ROCProfiler SDK Common API Library

## Custom Regex Engine

### Why We Have Our Own Regex Implementation

This directory contains a custom regex engine implementation designed explicitly for ROCm profiling tools. The primary motivation for implementing our own regex engine instead of using `std::regex` is to avoid the **dual ABI compatibility issues** that plague `std::regex` in the GNU libstdc++ library.

#### The Dual ABI Problem

The GNU libstdc++ library introduced a dual ABI (Application Binary Interface) system starting with GCC 5.1 to maintain backward compatibility while introducing C++11 improvements. This dual ABI system affects `std::string` and other standard library components, including `std::regex`.

##### Technical Background

The dual ABI allows two different implementations to coexist:
- **Old ABI (pre-C++11)**: Uses Copy-on-Write (COW) strings
- **New ABI (C++11+)**: Uses Short String Optimization (SSO)

The ABI is controlled by the `_GLIBCXX_USE_CXX11_ABI` macro:
- `_GLIBCXX_USE_CXX11_ABI=0`: Old ABI (default for GCC < 5.1)
- `_GLIBCXX_USE_CXX11_ABI=1`: New ABI (default for GCC >= 5.1)

##### The std::regex Problem

`std::regex` is particularly problematic because:

1. **ABI Sensitivity**: The `std::regex` implementation is tightly coupled to the string ABI being used
2. **Symbol Conflicts**: Different ABI versions create incompatible symbols that cannot be mixed
3. **Runtime Failures**: Applications linking against libraries compiled with different ABI settings experience runtime failures
4. **Distribution Issues**: Different Linux distributions and package managers may use different ABI settings

##### Real-World Impact

As explained in the [Stack Overflow discussion](https://stackoverflow.com/questions/51382355/stdregex-and-dual-abi), this creates several problematic scenarios:

- Applications compiled with GCC 4.x linking against libraries compiled with GCC 5+
- Mixing libraries compiled with different `_GLIBCXX_USE_CXX11_ABI` settings
- Distribution packages that assume different ABI defaults
- Cross-compilation scenarios where ABI settings don't match

Example error scenarios:
```cpp
// Library A compiled with _GLIBCXX_USE_CXX11_ABI=0
// Library B compiled with _GLIBCXX_USE_CXX11_ABI=1
// Both use std::regex -> Runtime failures or linking errors
```

### Our Solution

To avoid these compatibility issues entirely, we implemented a custom regex engine with the following benefits:

#### 1. **ABI Independence**
- No dependency on `std::regex` or dual ABI settings
- Consistent behavior across all GCC versions and distributions
- Eliminates linking and runtime compatibility issues

#### 2. **Controlled Dependencies**
- Uses only basic standard library components (`std::string_view`, `std::vector`, etc.)
- Minimizes external dependencies that could introduce ABI conflicts
- Self-contained implementation

#### 3. **Targeted Feature Set**
Our implementation focuses on the regex features actually needed by ROCm profiling tools:

##### Supported Features
- **Literals and Escapes**: `\n`, `\t`, `\\`, etc.
- **Anchors**: `^` (beginning), `$` (end)
- **Character Classes**: `[abc]`, `[a-z]`, `[^0-9]`
- **Shortcuts**: `\d`, `\D`, `\w`, `\W`, `\s`, `\S`
- **Quantifiers**: `*`, `+`, `?`, `{m}`, `{m,}`, `{m,n}`
- **Lazy Quantifiers**: `*?`, `+?`, `??`, `{m,n}?`
- **Groups and Alternation**: `()`, `|`
- **Dot Metacharacter**: `.`

##### API Compatibility
The API is designed to be familiar to users of `std::regex`:

```cpp
namespace rocprofiler::common::regex {
    bool regex_match(std::string_view text, std::string_view pattern);
    bool regex_search(std::string_view text, std::string_view pattern);
    bool regex_search(std::string_view text, std::string_view pattern,
                     size_t& begin, size_t& end);
    std::string regex_replace(std::string_view text, std::string_view pattern,
                             std::string_view replacement);
}
```

#### 4. **Replacement Token Support**
Full support for replacement tokens in `regex_replace`:
- `$0` or `$&`: Whole match
- `$1` to `$99`: Capture groups
- `$``: Prefix (text before match)
- `$'`: Suffix (text after match)

### Implementation Architecture

#### 1. **Parser** (`struct Parser`)
- Converts regex pattern strings into an Abstract Syntax Tree (AST)
- Handles escape sequences, character classes, and quantifiers
- Validates pattern syntax and reports errors

#### 2. **AST Nodes** (`struct Node`)
- Represents different regex components (literals, classes, quantifiers, etc.)
- Supports recursive structure for complex patterns
- Memory-efficient representation

#### 3. **Matchers**
- **FastMatcher**: Optimized for simple matching without capture groups
- **CaptureMatcher**: Full-featured matcher with capture group support
- Memoization for performance optimization

#### 4. **Algorithm Features**
- **Backtracking**: Supports complex patterns with alternatives
- **Greedy/Lazy Quantifiers**: Proper implementation of both modes
- **Zero-length Guards**: Prevents infinite loops in edge cases
- **Capture Group Tracking**: Maintains group boundaries during matching

### Usage Examples

```cpp
#include "lib/common/regex.hpp"

using namespace rocprofiler::common::regex;

// Basic matching
bool matches = regex_match("hello123", "hello\\d+");

// Search with position
size_t begin, end;
if (regex_search("prefix_hello123_suffix", "hello\\d+", begin, end)) {
    // Found match at positions [begin, end)
}

// Replace with captures
std::string result = regex_replace(
    "file_v1.2.3.txt",
    "v(\\d+)\\.(\\d+)\\.(\\d+)",
    "version_$1_$2_$3"
);
// result: "file_version_1_2_3.txt"
```

### Testing and Validation

The implementation includes comprehensive tests that verify compatibility with ECMAScript regex semantics:

- **Parity Tests**: Compare behavior against `std::regex` where possible
- **Edge Cases**: Handle corner cases like zero-length matches, nested captures
- **Compatibility Tests**: Verify consistent behavior across different string types and usage patterns

### Maintenance Notes

- The implementation prioritizes correctness and ABI independence over maximum performance
- Features are added based on actual requirements from ROCm profiling tools
- Regular testing ensures compatibility with target environments
- Documentation is maintained to explain design decisions and limitations

This custom implementation provides a robust, ABI-independent regex solution that eliminates the compatibility issues that would otherwise plague ROCm profiling tools when deployed across diverse environments.

### Notes on ABI Independence Testing

The current test suite includes "compatibility tests" that verify consistent behavior across different string types and usage patterns. However, **true ABI independence testing** would require:

1. **Cross-compilation builds**: Building test applications with different `_GLIBCXX_USE_CXX11_ABI` settings (0 and 1)
2. **Binary compatibility verification**: Ensuring object files compiled with different ABI settings can link together
3. **Runtime validation**: Testing that regex functionality works consistently regardless of how dependent libraries were compiled

Such comprehensive ABI testing would require:

```bash
# Build with old ABI
g++ -D_GLIBCXX_USE_CXX11_ABI=0 -c test_old_abi.cpp

# Build with new ABI
g++ -D_GLIBCXX_USE_CXX11_ABI=1 -c test_new_abi.cpp

# Link together and verify functionality
g++ test_old_abi.o test_new_abi.o -o cross_abi_test
```

The current implementation achieves ABI independence by avoiding `std::regex` entirely, relying instead on minimal standard library components and custom string processing that remains stable across ABI versions.
