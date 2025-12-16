#pragma once
#include <array>
#include <cstddef>
#include <vector>

namespace rocprofsys
{
inline namespace common
{

template <typename T>
struct span
{
    using value_type = T;

    constexpr span(T* data, size_t size) noexcept
    : m_data(data)
    , m_size(size)
    {}

    constexpr span(std::vector<T>& vec) noexcept
    : m_data(vec.data())
    , m_size(vec.size())
    {}

    template <size_t N>
    constexpr span(std::array<T, N>& arr) noexcept
    : m_data(arr.data())
    , m_size(arr.size())
    {}

    constexpr inline __attribute__((always_inline)) T* data() const noexcept
    {
        return m_data;
    }
    constexpr inline __attribute__((always_inline)) T* data() noexcept { return m_data; }
    constexpr inline __attribute__((always_inline)) T* begin() noexcept { return m_data; }
    constexpr inline __attribute__((always_inline)) T* end() noexcept
    {
        return m_data + m_size;
    }
    constexpr inline __attribute__((always_inline)) size_t size() const noexcept
    {
        return m_size;
    }
    constexpr inline __attribute__((always_inline)) bool empty() const noexcept
    {
        return m_size == 0;
    }
    constexpr inline __attribute__((always_inline)) T& operator[](size_t index) noexcept
    {
        return m_data[index];
    }
    constexpr inline __attribute__((always_inline)) const T& operator[](
        size_t index) const noexcept
    {
        return m_data[index];
    }

private:
    T*     m_data = nullptr;
    size_t m_size = 0;
};

}  // namespace common
}  // namespace rocprofsys
