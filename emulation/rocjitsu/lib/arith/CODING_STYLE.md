# rj-arith coding style

Follow [Rocjitsu's style guide](../../docs/style.md) and parent `.clang-format`,
with the following differences for `lib/arith/`.

## Integer types

Follow the Google C++ Style Guide's [Integer Types](https://google.github.io/styleguide/cppguide.html#Integer_Types)
guidance: use signed integers for ordinary indices and counts, including lanes,
registers, and matrix coordinates. `int` suffices for these bounded dimensions.
Use fixed-width unsigned types for register bits, masks, and arithmetic requiring
defined unsigned wrapping. A value being nonnegative is not by itself a reason
to make its type unsigned.

## Language and API conventions

Implementation code uses Rocjitsu's C++ standard; public headers support C23 and C++17.
The C API uses `rj_arith_` function names, `RJ_ARITH_` constants, integer status
returns, C headers such as `<stdint.h>`, and traditional include guards.
Public integer constants use typed `constexpr` declarations.
Internal C++ code lives in `rocjitsu::arith`, not a top-level `arith` namespace.

Short arithmetic names such as `a`, `b`, `c`, and `d` are appropriate when they
match instruction operands or a nearby mathematical expression.

## Control flow

Always use braces around control-flow bodies, including single-statement
`if`, `else`, `for`, `while`, and `do` bodies. The local `.clang-format`
inherits Rocjitsu’s formatting and enables brace insertion.

## Comments

Prefer short, precise comments over verbose descriptions. File comments explain
scope; function comments explain purpose and contract. Comments within a function
explain nontrivial arithmetic or layout decisions. Plain `//` comments suffice;
Doxygen markup is not required.

## CPU-specific implementation symbols

CMake selects `RJ_ARITH_IMPL_NAMESPACE` together with each source's CPU options.
Shared implementation definitions belong to that namespace, so the linker cannot
substitute a definition compiled for a different ISA. Backend entry points carry
their ISA in their name; cross-backend headers contain declarations only.
File-local helpers may instead have internal linkage.

Implementation code avoids runtime C++ standard-library functions, including
implicit calls through containers and initializer lists. Use the small internal
bit helpers, compiler intrinsics, fixed arrays, or C library functions as needed.
Compile-time traits are fine. Tests and benchmarks are not subject to this rule.

The baseline resets the CPU model with `-march=x86-64`. Enclosing builds must
not enable stronger explicit ISA flags globally: a trailing `-march` does not
cancel options such as `-mavx2`. The archive audit checks symbol ownership,
not the instruction set of each function. Debug CI exercises emitted inline
definitions that optimized builds may remove.
