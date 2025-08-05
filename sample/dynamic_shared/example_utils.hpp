// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_EXAMPLE_UTILS_HPP
#define COMMON_EXAMPLE_UTILS_HPP

// Compiling HIP on Windows includes windows.h, and this triggers many silly warnings.
#include <cstdint>
#if defined(_WIN32) && defined(__NVCC__)
    #pragma nv_diag_suppress 108 // signed bit field of length 1
    #pragma nv_diag_suppress 174 // expression has no effect
    #pragma nv_diag_suppress 1835 // attribute "dllimport" does not apply here
#endif

// rocPRIM adds a #warning about printf on NAVI.
#ifdef __clang__
    #pragma clang diagnostic ignored "-W#warnings"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>

constexpr int error_exit_code = -1;

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }

/// \brief Formats a range of elements to a pretty string.
/// \tparam BidirectionalIterator - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to
/// \p std::ostream.
template<class BidirectionalIterator>
inline std::string format_range(const BidirectionalIterator begin, const BidirectionalIterator end)
{
    std::stringstream sstream;
    sstream << "[ ";
    for(auto it = begin; it != end; ++it)
    {
        sstream << *it;
        if(it != std::prev(end))
        {
            sstream << ", ";
        }
    }
    sstream << " ]";
    return sstream.str();
}

/// \brief Formats a range of pairs to a pretty string. The length of the two ranges must match.
/// \tparam BidirectionalIteratorT - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to \p std::ostream.
/// \tparam BidirectionalIteratorU - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to \p std::ostream.
template<class BidirectionalIteratorT, typename BidirectionalIteratorU>
inline std::string format_pairs(const BidirectionalIteratorT begin_a,
                                const BidirectionalIteratorT end_a,
                                const BidirectionalIteratorU begin_b,
                                const BidirectionalIteratorU end_b)
{
    (void)end_b;
    assert(std::distance(begin_a, end_a) == std::distance(begin_b, end_b));

    std::stringstream sstream;
    sstream << "[ ";
    auto it_a = begin_a;
    auto it_b = begin_b;
    for(; it_a < end_a; ++it_a, ++it_b)
    {
        sstream << "(" << *it_a << ", " << *it_b << ")";

        if(it_a != std::prev(end_a))
        {
            sstream << ", ";
        }
    }
    sstream << " ]";
    return sstream.str();
}

/// \brief A function to parse a string for an int. If the string is a valid integer then return true
/// else if it has non-numeric character then return false.
inline bool parse_int_string(const std::string& str, int& out)
{
    try
    {
        size_t end;
        int    value = std::stoi(str, &end);
        if(end == str.size())
        {
            out = value;
            return true;
        }
        return false;
    }
    catch(const std::exception&)
    {
        return false;
    }
}

/// \brief A class to measures time between intervals
class HostClock
{
private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::duration   elapsed_time;

public:
    HostClock()
    {
        this->reset_timer();
    }

    inline void reset_timer()
    {
        this->elapsed_time = std::chrono::steady_clock::duration(0);
    }

    inline void start_timer()
    {
        this->start_time = std::chrono::steady_clock::now();
    }

    inline void stop_timer()
    {
        const auto end_time = std::chrono::steady_clock::now();
        this->elapsed_time += end_time - this->start_time;
    }

    /// @brief Returns time elapsed in Seconds
    /// @return type double that contains the elapsed time in Seconds
    inline double get_elapsed_time() const
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(this->elapsed_time)
            .count();
    }
};

/// \brief Returns <tt>ceil(dividend / divisor)</tt>, where \p dividend is an integer and
/// \p divisor is an unsigned integer.
template<typename T,
         typename U,
         std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
__host__ __device__ constexpr auto ceiling_div(const T& dividend, const U& divisor)
{
    return (dividend + divisor - 1) / divisor;
}

/// \brief Report validation results.
inline int report_validation_result(int errors)
{
    if(errors)
    {
        std::cout << "Validation failed. Errors: " << errors << std::endl;
        return error_exit_code;
    }

    std::cout << "Validation passed." << std::endl;
    return 0;
}

/// \brief Generate an identity matrix.
/// The identity matrix is a $m \times n$ matrix with ones in the main diagonal and zeros elsewhere.
template<typename T>
void generate_identity_matrix(T* A, int m, int n, size_t lda)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            A[i + j * lda] = T(i == j);
        }
    }
}

/// \brief Multiply an $A$ matrix ($m \times k$) with a $B$ matrix ($k \times n$) as:
/// $C := \alpha \cdot A \cdot B + \beta \cdot C$
template<typename T>
void multiply_matrices(T        alpha,
                       T        beta,
                       int      m,
                       int      n,
                       int      k,
                       const T* A,
                       int      stride1_a,
                       int      stride2_a,
                       const T* B,
                       int      stride1_b,
                       int      stride2_b,
                       T*       C,
                       int      stride_c)
{
    for(int i1 = 0; i1 < m; ++i1)
    {
        for(int i2 = 0; i2 < n; ++i2)
        {
            T t = T(0.0);
            for(int i3 = 0; i3 < k; ++i3)
            {
                t += A[i1 * stride1_a + i3 * stride2_a] * B[i3 * stride1_b + i2 * stride2_b];
            }
            C[i1 + i2 * stride_c] = beta * C[i1 + i2 * stride_c] + alpha * t;
        }
    }
}

/// \brief Prints an {1,2,3}-dimensional array. The last dimension (fastest-index) specified in
/// \p n will be printed horizontally.
///
/// By default a row-major layout of the data is assumed. When printing data in column-major
/// layout, the \p column_major parameter must be set to \p true for a correct interpretation
/// of the dimensions' sizes.
template<class Tdata, class Tsize>
void print_nd_data(const std::vector<Tdata>& data,
                   std::vector<Tsize>        np,
                   const int                 column_width = 4,
                   const bool                column_major = false)
{
    if(column_major)
    {
        std::reverse(np.begin(), np.end());
    }
    const std::vector<Tsize> n(np);
    // Note: we want to print the last dimension horizontally (on the x-axis)!
    int size_x = n[n.size() - 1];
    int size_y = n.size() > 1 ? n[n.size() - 2] : 1;
    int size_z = n.size() > 2 ? n[n.size() - 3] : 1;
    for(int z = 0; z < size_z; ++z)
    {
        for(int y = 0; y < size_y; ++y)
        {
            for(int x = 0; x < size_x; ++x)
            {
                auto index = (z * size_y + y) * size_x + x;
                std::cout << std::setfill(' ') << std::setw(column_width) << data[index] << " ";
            }
            std::cout << "\n";
        }
        if(z != size_z - 1)
        {
            std::cout << "\n";
        }
    }
    std::cout << std::flush;
}

/// \brief Returns a string from the double \p value with specified \p precision .
inline std::string
    double_precision(const double value, const int precision, const bool fixed = false)
{
    std::stringstream ss;
    if(fixed)
    {
        ss << std::fixed;
    }
    ss << std::setprecision(precision) << value;
    return ss.str();
}

#endif // COMMON_EXAMPLE_UTILS_HPP
