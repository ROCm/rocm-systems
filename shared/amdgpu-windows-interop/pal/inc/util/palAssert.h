/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
 ***********************************************************************************************************************
 * @file  palAssert.h
 * @brief PAL utility collection assert macros.
 ***********************************************************************************************************************
 */

#pragma once

#include "palDbgPrint.h"
#include "palUtil.h"

#if defined(_WIN32)
#include <intrin.h>
/// OS-independent macro to force a break into the debugger.
#define PAL_DEBUG_BREAK() [[unlikely]] __debugbreak()
#else
#include <signal.h>
/// OS-independent macro to force a break into the debugger.
#define PAL_DEBUG_BREAK() [[unlikely]] raise(SIGTRAP);
#endif

#if defined(_MSC_VER) && !defined(__clang__)
/// Macro to direct static code analysis to assume the specified expression will always be true.
/// Purpose is to suppress warnings from MSVC's /analysis setting.
/// Only pertains to static code analysis. Does not impact compile optimization. Not the same as C++23's [[assume]].
# define PAL_ANALYSIS_ASSUME(_expr) __analysis_assume(_expr)
#else
/// This macro is only useful on MSVC builds. It has no meaning for other builds.
# define PAL_ANALYSIS_ASSUME(_expr) ((void)0)
#endif

namespace Util
{

/// A helper function to check the size-in-bits of a 'reserved' member in a bitfield.
/// This is intended for use with static_asserts to ensure things don't go out-of-sync.
///
/// @param [in] expectedTotalBitWidth  Number of bits expected in the whole type
/// @param [in] expectedReservedBits   Number of bits in the 'reserved' field
///
/// @return true if the bit lengths of the type T match the values in the args.
///         true if the compiler lacks support to do this at compile time.
///
/// @note This may not work properly with old compilers, but this is meant for linting anyhow.
template <typename T>
constexpr bool CheckReservedBits(
    uint32 expectedTotalBitWidth,
    uint32 expectedReservedBits)
{
    bool match = false;

    // Fail if the whole size is different
    if (sizeof(T) * 8 == expectedTotalBitWidth)
    {
        // Get the width of the reserved field by detecting when it stops filling bits
        T      sample       = {};
        uint64 mask         = 0;
        uint32 reservedBits = 0;
        do
        {
            sample = {};
            mask   = (mask << 1) | 1;
            reservedBits++;
            sample.reserved = mask;
        } while ((sample.reserved == mask) && (reservedBits < sizeof(T) * 8));
        // when the loop terminates, it's one past the size of the field.
        match = (reservedBits - 1) == expectedReservedBits;
    }
    return match;
}

/// A helper function to check that a series of static numeric values are sequential.
/// This is intended for use with static_asserts to ensure things don't go out-of-sync.
///
/// @param [in] args     Array of numeric values to check
/// @param [in] interval Expected interval between each (default 1, 4 is also common for field offsets)
///
/// @return true if all the values are sequential
///         true if the compiler lacks support to do this at compile time.
///
/// @note This may not work properly with old compilers, but this is meant for linting anyhow.
template <typename T, size_t N>
constexpr bool CheckSequential(
    const T (&args)[N],
    T       interval = 1)
{
    bool isSequential = true;
    for (int i = 0; i < (N - 1); i++)
    {
        if ((args[i] + interval) != args[i + 1])
        {
            isSequential = false;
            break;
        }
    }
    return isSequential;
}

#if (PAL_ENABLE_PRINTS_ASSERTS || PAL_ENABLE_LOGGING)

/// Specifies how severe an triggered assert (or alert) is.
///
/// Both asserts and alerts can print out a debug string and break into the debugger.  Asserts are to be used to verify
/// the known, assumed state of the program at any time.  Alerts are to be used to notify the developer of a _possible_,
/// but unexpected condition such as memory allocation failure, an OS call failure, or an application behavior that is
/// known to be slow.
enum AssertCategory : uint32
{
    AssertCatAssert = 0,
    AssertCatAlert,
    AssertCatCount
};

/// Enables/disables the specified assert category.
///
/// Probably controlled by a setting and set during initialization.
///
/// @param [in] category  Assert category to enable/disable (asserts or alerts).
/// @param [in] enable    True to enable the specified assert category, false to disable it.
extern void EnableAssertMode(
    AssertCategory category,
    bool           enable);

/// Returns true if the specified assert category is enabled and false otherwise.
///
/// @param [in] category  Assert category to check
extern bool IsAssertCategoryEnabled(
    AssertCategory category);

#endif

} // namespace Util

#if (PAL_ENABLE_PRINTS_ASSERTS || PAL_ENABLE_LOGGING)
/// Prints an error message with the specified reason via the debug print system. A debug break will also be triggered
/// if they're currently enabled for asserts.
///
/// @note This version of assert inlines an 'int 3' every time it is used so that each occurrence can be zapped
///       independently.  This macro cannot be used in assignment operations.
#define PAL_TRIGGER_ASSERT(_pFormat, ...) [[unlikely]]              \
do {                                                                \
    PAL_DPERROR(_pFormat, ##__VA_ARGS__);                           \
    if (::Util::IsAssertCategoryEnabled(::Util::AssertCatAssert))   \
    {                                                               \
        PAL_DEBUG_BREAK();                                          \
    }                                                               \
} while (false)

/// If the expression evaluates to false, then it calls the PAL_TRIGGER_ASSERT macro with an error message with the
/// specified reason.
///
/// @note This assert should not be used in constant evaluated contexts (e.g., constexpr functions).
//
// This previously said:
//    if (_expr_eval == false) [[unlikely]]
//    {
//        PAL_TRIGGER_ASSERT(...);
//    }
// However there is a bug in the initial gcc implementation of [[unlikely]] that means you cannot
// attach it to a compound statement. So:
// 1. we ignore PAL coding standards and don't use a compound statement;
// 2. we don't use [[unlikely]] as the expansion of PAL_TRIGGER_ASSERT already has one.
#define PAL_ASSERT_MSG(_expr, _pReasonFmt, ...)                                                   \
do {                                                                                              \
    const bool _expr_eval = static_cast<bool>(_expr);                                             \
    if (_expr_eval == false)                                                                      \
        PAL_TRIGGER_ASSERT("Assertion failed: %s | Reason: " _pReasonFmt, #_expr, ##__VA_ARGS__); \
    PAL_ANALYSIS_ASSUME(_expr_eval);                                                              \
} while (false)

#if !defined(__clang__) && !defined(_MSC_VER) && (__GNUC__ < 6)

// Function to circumvent gcc 5.x inability to use lambdas in unevaluated constant expression contexts.
constexpr void PalTriggerAssertImpl(
    const char* pFormat,
    const char* pExpr,
    const char* pFile,
    int         line,
    const char* pFunc)
{
    // pExpr is always not nullptr, as it's supposed to be a preprocessor string, but it does convince gcc
    // to compile PalTriggerAssertImpl() as potentially constexpr
    pExpr != nullptr ?
        [&]
        {
            Util::DbgPrintf(
                Util::DbgPrintCatErrorMsg,
                Util::DbgPrintStyleDefault,
                pFormat,
                pExpr,
                pFile,
                line,
                pFunc);
            if (Util::IsAssertCategoryEnabled(Util::AssertCatAssert))
            {
                PAL_DEBUG_BREAK();
            }
            return 0;
        }()
        : 0;
}

// gcc 5.4 implementation of PAL_CONSTEXPR_ASSERT_MSG that ignores the additional reason for the assertion
//
// This previously said:
//    if (_expr_eval == false) [[unlikely]]
//    {
//        PalTriggerAssertImpl(...);
//    }
// However there is a bug in the initial gcc implementation of [[unlikely]] that means you cannot
// attach it to a compound statement. So we ignore PAL coding standards and don't use a compound statement.
#define PAL_CONSTEXPR_ASSERT_MSG(_expr, _pReasonFmt, ...)                                               \
do {                                                                                                    \
    const bool _expr_eval = static_cast<bool>(_expr);                                                   \
    if (_expr_eval == false) [[unlikely]]                                                               \
        PalTriggerAssertImpl("Assertion failed: %s (%s:%d:%s)", #_expr,  __FILE__, __LINE__, __func__); \
    PAL_ANALYSIS_ASSUME(_expr_eval);                                                                    \
} while (false)

#else

/// If the expression evaluates to false, then it calls the PAL_TRIGGER_ASSERT macro with an error message with the
/// specified reason.
///
/// @note This assert should be used in constant evaluated contexts (e.g., constexpr functions).
/// @note This assert uses an immediately-invoked function expression in the form of an internal lambda to signal a
///       failed assert. Since PAL_TRIGGER_ASSERT is not constexpr, an _expr that evaluates to false will fail to
///       compile the function operator of the lambda.
//
// This previously said:
//    if (_expr_eval == false) [[unlikely]]
//    {
//        [&] { PAL_TRIGGER_ASSERT(...); }();
//    }
// However there is a bug in the initial gcc implementation of [[unlikely]] that means you cannot
// attach it to a compound statement. So we ignore PAL coding standards and don't use a compound statement.
#define PAL_CONSTEXPR_ASSERT_MSG(_expr, _pReasonFmt, ...)                                                    \
do {                                                                                                         \
    const bool _expr_eval = static_cast<bool>(_expr);                                                        \
    if (_expr_eval == false) [[unlikely]]                                                                    \
        [&] { PAL_TRIGGER_ASSERT("Assertion failed: %s | Reason: " _pReasonFmt, #_expr, ##__VA_ARGS__); }(); \
    PAL_ANALYSIS_ASSUME(_expr_eval);                                                                         \
} while (false)

#endif

/// Calls the PAL_ASSERT_MSG macro with a generic reason string
#define PAL_ASSERT(_expr) PAL_ASSERT_MSG(_expr, "%s", "Unknown")

/// Calls the PAL_CONSTEXPR_ASSERT_MSG macro with a generic reason string
#define PAL_CONSTEXPR_ASSERT(_expr) PAL_CONSTEXPR_ASSERT_MSG(_expr, "%s", "Unknown")

#if DEBUG
/// Debug build only PAL assert, the typical usage is when make an assertion on a debug-only variables.
/// The only difference than PAL assert is it's empty in release mode.
#define PAL_DEBUG_BUILD_ONLY_ASSERT(_expr) \
do {                                       \
    PAL_ASSERT(_expr);                     \
} while (false)
#else
#define PAL_DEBUG_BUILD_ONLY_ASSERT(_expr) ((void)0)
#endif

/// If the expression evaluates to true, then a warning message with the specified reason will be printed via the
/// debug print system. A debug break will also be triggered if they're currently enabled for alerts.
///
/// @note This is the opposite polarity of asserts.  The assert macro _asserts_ that the specified condition is true.
///       While the alert macro _alerts_ the developer if the specified condition is true.
///
/// This macro should be used in places where an assert is inappropriate because an error condition is _possible_, but
/// not typically expected.  For example, asserting that an OS call succeeded should be avoided since there cannot be an
/// assumption that it will succeed.  Nonetheless, a developer may want to be alerted immediately and dropped into the
/// debugger when such a failure occurs.
#define PAL_TRIGGER_ALERT(_pFormat, ...) [[unlikely]]               \
do {                                                                \
    PAL_DPWARN(_pFormat, ##__VA_ARGS__);                            \
    if (::Util::IsAssertCategoryEnabled(::Util::AssertCatAlert))    \
    {                                                               \
        PAL_DEBUG_BREAK();                                          \
    }                                                               \
} while (false)

//
// This previously said:
//    if (_expr) [[unlikely]]
//    {
//        PAL_TRIGGER_ASSERT(...);
//    }
// However there is a bug in the initial gcc implementation of [[unlikely]] that means you cannot
// attach it to a compound statement. So:
// 1. we ignore PAL coding standards and don't use a compound statement;
// 2. we don't use [[unlikely]] as the expansion of PAL_TRIGGER_ASSERT already has one.
#define PAL_ALERT_MSG(_expr, _pReasonFmt, ...)                                                  \
do {                                                                                            \
    if (_expr)                                                                                  \
        PAL_TRIGGER_ALERT("Alert triggered: %s | Reason: " _pReasonFmt, #_expr, ##__VA_ARGS__); \
} while (false)

/// Calls the PAL_ALERT_MSG macro with a generic reason string
#define PAL_ALERT(_expr) PAL_ALERT_MSG(_expr, "%s", "Unknown")

/// Convenience macro that asserts if something has never been tested.
#define PAL_NOT_TESTED_MSG(_pReasonFmt, ...) PAL_TRIGGER_ASSERT("Code Not Tested! | Reason: " _pReasonFmt, ##__VA_ARGS__)
#define PAL_NOT_TESTED() PAL_NOT_TESTED_MSG("%s", "Unknown")

/// Convenience macro that asserts if something has not been implemented.
#define PAL_NOT_IMPLEMENTED_MSG(_pReasonFmt, ...) PAL_TRIGGER_ASSERT("Not Implemented! | Reason: " _pReasonFmt, ##__VA_ARGS__)
#define PAL_NOT_IMPLEMENTED() PAL_NOT_IMPLEMENTED_MSG("%s", "Unknown")

/// Convenience macro that asserts if an area of code that shouldn't be executed is reached.
#define PAL_NEVER_CALLED_MSG(_pReasonFmt, ...) PAL_TRIGGER_ASSERT("Code should never be called! | Reason: " _pReasonFmt, ##__VA_ARGS__)
#define PAL_NEVER_CALLED() PAL_NEVER_CALLED_MSG("%s", "Unknown")

/// Convenience macro that always asserts.  Expect this to be used instead of PAL_ASSERT(false).
#define PAL_ASSERT_ALWAYS_MSG(_pReasonFmt, ...) PAL_TRIGGER_ASSERT("Unconditional Assert | Reason: " _pReasonFmt, ##__VA_ARGS__)
#define PAL_ASSERT_ALWAYS() PAL_ASSERT_ALWAYS_MSG("%s", "Unknown")

/// Convenience macro that always alerts.  Expect this to be used instead of PAL_ALERT(true).
#define PAL_ALERT_ALWAYS_MSG(_pReasonFmt, ...) PAL_TRIGGER_ALERT("Unconditional Alert | Reason: " _pReasonFmt, ##__VA_ARGS__)
#define PAL_ALERT_ALWAYS() PAL_ALERT_ALWAYS_MSG("%s", "Unknown")

#else

#define PAL_ASSERT(_expr)                    PAL_ANALYSIS_ASSUME(_expr)
#define PAL_CONSTEXPR_ASSERT(_expr)          PAL_ANALYSIS_ASSUME(_expr)
#define PAL_ASSERT_MSG(_expr, ...)           PAL_ANALYSIS_ASSUME(_expr)
#define PAL_CONSTEXPR_ASSERT_MSG(_expr, ...) PAL_ANALYSIS_ASSUME(_expr)
#define PAL_DEBUG_BUILD_ONLY_ASSERT(_expr)   ((void)0)
#define PAL_ALERT(_expr)                     ((void)0)
#define PAL_ALERT_MSG(_expr, ...)            ((void)0)
#define PAL_NOT_TESTED()                     [[unlikely]] ((void)0)
#define PAL_NOT_TESTED_MSG(...)              [[unlikely]] ((void)0)
#define PAL_NOT_IMPLEMENTED()                [[unlikely]] ((void)0)
#define PAL_NOT_IMPLEMENTED_MSG(...)         [[unlikely]] ((void)0)
#define PAL_NEVER_CALLED()                   [[unlikely]] ((void)0)
#define PAL_NEVER_CALLED_MSG(...)            [[unlikely]] ((void)0)
#define PAL_ASSERT_ALWAYS()                  [[unlikely]] ((void)0)
#define PAL_ASSERT_ALWAYS_MSG(...)           [[unlikely]] ((void)0)
#define PAL_ALERT_ALWAYS()                   [[unlikely]] ((void)0)
#define PAL_ALERT_ALWAYS_MSG(...)            [[unlikely]] ((void)0)

#endif

