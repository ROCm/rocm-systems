#pragma once
#include <array>
#include <cstddef>
#include <vector>

namespace rocprofsys
{
namespace common
{

template <typename T>
struct span
{
    span(T* data, size_t size)
    : m_data(data)
    , m_size(size)
    {}

    span(std::vector<T>& vec)
    : m_data(vec.data())
    , m_size(vec.size())
    {}

    template <size_t N>
    span(const std::array<T, N>& arr)
    : m_data(arr.data())
    , m_size(arr.size())
    {}

    T*       data() const { return m_data; }
    T*       data() { return m_data; }
    T*       begin() { return m_data; }
    T*       end() { return m_data + m_size; }
    size_t   size() const { return m_size; }
    bool     empty() const { return m_size == 0; }
    T&       operator[](size_t index) { return m_data[index]; }
    const T& operator[](size_t index) const { return m_data[index]; }

private:
    T*     m_data;
    size_t m_size;
};

}  // namespace common
}  // namespace rocprofsys
