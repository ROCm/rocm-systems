// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_INLINE_VECTOR_H_
#define UTIL_INLINE_VECTOR_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#ifdef UTIL_INLINE_VECTOR_ENABLE_HISTOGRAM
#include <fstream>
#include <map>
#include <mutex>
#include <ostream>
#include <source_location>
#include <string>
#include <tuple>
#include <unordered_map>
#endif

namespace util {

namespace detail {

#ifdef UTIL_INLINE_VECTOR_ENABLE_HISTOGRAM
using inline_vector_source_location = std::source_location;

struct InlineVectorSourceKey {
  std::string file;
  std::string function;
  uint32_t line = 0;
  uint32_t column = 0;

  friend bool operator<(const InlineVectorSourceKey &lhs,
                        const InlineVectorSourceKey &rhs) noexcept {
    return std::tie(lhs.file, lhs.line, lhs.column, lhs.function) <
           std::tie(rhs.file, rhs.line, rhs.column, rhs.function);
  }
};

class inline_vector_histogram_registry {
public:
  static inline_vector_histogram_registry &instance() noexcept {
    static inline_vector_histogram_registry *registry = create();
    return *registry;
  }

  void register_vector(const void *ptr, inline_vector_source_location loc) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    active_[ptr] = InlineVectorSourceKey{
        .file = loc.file_name(),
        .function = loc.function_name(),
        .line = loc.line(),
        .column = loc.column(),
    };
  }

  void record_destruction(const void *ptr, uint32_t size) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.find(ptr);
    if (it == active_.end()) {
      return;
    }
    ++histograms_[it->second][size];
    active_.erase(it);
  }

  void write_jsonl(std::ostream &os) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &[source, histogram] : histograms_) {
      for (const auto &[size, count] : histogram) {
        os << "{\"file\":";
        write_json_string(os, source.file);
        os << ",\"line\":" << source.line << ",\"column\":" << source.column << ",\"function\":";
        write_json_string(os, source.function);
        os << ",\"size\":" << size << ",\"count\":" << count << "}\n";
      }
    }
  }

  void reset_for_testing() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    active_.clear();
    histograms_.clear();
  }

private:
  using Histogram = std::map<uint32_t, uint64_t>;

  static inline_vector_histogram_registry *create() noexcept {
    auto *registry = new inline_vector_histogram_registry();
    std::atexit(dump_at_exit);
    return registry;
  }

  static void dump_at_exit() noexcept {
    const char *path = std::getenv("UTIL_INLINE_VECTOR_HISTOGRAM_FILE");
    if (path == nullptr || path[0] == '\0') {
      return;
    }

    std::ofstream os(path);
    if (!os) {
      return;
    }
    instance().write_jsonl(os);
  }

  static void write_json_string(std::ostream &os, const std::string &str) noexcept {
    os << '"';
    for (char c : str) {
      switch (c) {
      case '\\':
        os << "\\\\";
        break;
      case '"':
        os << "\\\"";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        os << c;
        break;
      }
    }
    os << '"';
  }

  std::mutex mutex_;
  std::unordered_map<const void *, InlineVectorSourceKey> active_;
  std::map<InlineVectorSourceKey, Histogram> histograms_;
};

#else
struct inline_vector_source_location {
  static constexpr inline_vector_source_location current() noexcept { return {}; }
};
#endif

template <typename T, uint32_t InlineN> struct InlineVectorStorage {
  alignas(T) std::byte storage_[sizeof(T) * InlineN];

  T *data() noexcept { return std::launder(reinterpret_cast<T *>(storage_)); }
  const T *data() const noexcept { return std::launder(reinterpret_cast<const T *>(storage_)); }
};

template <typename T> struct InlineVectorStorage<T, 0> {
  T *data() noexcept { return nullptr; }
  const T *data() const noexcept { return nullptr; }
};

template <typename T> struct DefaultInlineVectorCapacity {
  static constexpr uint32_t metadata_bytes = sizeof(void *) + 2 * sizeof(uint32_t);
  static constexpr uint32_t target_bytes = 64;
  static constexpr uint32_t available_bytes =
      target_bytes > metadata_bytes ? target_bytes - metadata_bytes : 0;
  static constexpr uint32_t by_size =
      sizeof(T) <= available_bytes ? available_bytes / sizeof(T) : 1;
  static constexpr uint32_t value = by_size == 0 ? 1 : by_size;
};

[[noreturn]] inline void inline_vector_abort() noexcept { std::abort(); }

inline uint32_t checked_inline_vector_size(std::size_t size) noexcept {
  if (size > std::numeric_limits<uint32_t>::max()) {
    inline_vector_abort();
  }
  return static_cast<uint32_t>(size);
}

} // namespace detail

/// @brief Vector-like container with inline storage for the small case.
///
/// @details The container is intended for RocJITsu hot paths where allocation
/// failures are not recovered with C++ exceptions. Methods are marked noexcept;
/// allocation failure terminates the process instead of throwing. The
/// `InlineN == 0` specialization has no inline storage and keeps the object to
/// pointer + uint32_t size + uint32_t capacity for normally aligned element
/// types.
template <typename T, uint32_t InlineN = detail::DefaultInlineVectorCapacity<T>::value>
class inline_vector {
  static_assert(!std::is_const_v<T>, "inline_vector<T> requires non-const T");
  static_assert(!std::is_void_v<T>, "inline_vector<T> requires object T");

public:
  using value_type = T;
  using size_type = uint32_t;
  using difference_type = std::ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;

  static constexpr size_type inline_capacity = InlineN;

  inline_vector(detail::inline_vector_source_location source_location =
                    detail::inline_vector_source_location::current()) noexcept
      : data_(inline_data()), capacity_(InlineN) {
    register_construction(source_location);
  }

  explicit inline_vector(size_type count,
                         detail::inline_vector_source_location source_location =
                             detail::inline_vector_source_location::current()) noexcept
      : inline_vector(source_location) {
    resize(count);
  }

  inline_vector(size_type count, const T &value,
                detail::inline_vector_source_location source_location =
                    detail::inline_vector_source_location::current()) noexcept
      : inline_vector(source_location) {
    resize(count, value);
  }

  inline_vector(std::initializer_list<T> init,
                detail::inline_vector_source_location source_location =
                    detail::inline_vector_source_location::current()) noexcept
      : inline_vector(source_location) {
    reserve(detail::checked_inline_vector_size(init.size()));
    for (const T &value : init) {
      emplace_back(value);
    }
  }

  template <typename InputIt>
    requires(!std::is_integral_v<InputIt>)
  inline_vector(InputIt first, InputIt last,
                detail::inline_vector_source_location source_location =
                    detail::inline_vector_source_location::current()) noexcept
      : inline_vector(source_location) {
    assign(first, last);
  }

  inline_vector(const inline_vector &other,
                detail::inline_vector_source_location source_location =
                    detail::inline_vector_source_location::current()) noexcept
      : inline_vector(source_location) {
    copy_from(other);
  }

  inline_vector(inline_vector &&other,
                detail::inline_vector_source_location source_location =
                    detail::inline_vector_source_location::current()) noexcept
      : inline_vector(source_location) {
    move_from(std::move(other));
  }

  ~inline_vector() noexcept {
    record_destruction();
    clear();
    deallocate_dynamic();
  }

  inline_vector &operator=(const inline_vector &other) noexcept {
    if (this == &other) {
      return *this;
    }
    clear();
    reserve(other.size_);
    copy_construct_from(other.data_, other.data_ + other.size_);
    return *this;
  }

  inline_vector &operator=(inline_vector &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    clear();
    deallocate_dynamic();
    reset_to_inline();
    move_from(std::move(other));
    return *this;
  }

  inline_vector &operator=(std::initializer_list<T> init) noexcept {
    clear();
    reserve(detail::checked_inline_vector_size(init.size()));
    for (const T &value : init) {
      emplace_back(value);
    }
    return *this;
  }

  void assign(size_type count, const T &value) noexcept {
    clear();
    reserve(count);
    while (size_ < count) {
      construct_at(data_ + size_, value);
      ++size_;
    }
  }

  template <typename InputIt>
    requires(!std::is_integral_v<InputIt>)
  void assign(InputIt first, InputIt last) noexcept {
    clear();
    reserve_for_range(first, last);
    for (; first != last; ++first) {
      emplace_back(*first);
    }
  }

  void assign(std::initializer_list<T> init) noexcept { assign(init.begin(), init.end()); }

  reference operator[](size_type index) noexcept {
    assert(index < size_);
    return data_[index];
  }

  const_reference operator[](size_type index) const noexcept {
    assert(index < size_);
    return data_[index];
  }

  reference front() noexcept {
    assert(size_ > 0);
    return data_[0];
  }

  const_reference front() const noexcept {
    assert(size_ > 0);
    return data_[0];
  }

  reference back() noexcept {
    assert(size_ > 0);
    return data_[size_ - 1];
  }

  const_reference back() const noexcept {
    assert(size_ > 0);
    return data_[size_ - 1];
  }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return data_; }

  iterator end() noexcept { return data_ + size_; }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cend() const noexcept { return data_ + size_; }

  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] size_type size() const noexcept { return size_; }
  [[nodiscard]] size_type capacity() const noexcept { return capacity_; }
  [[nodiscard]] bool is_inline() const noexcept { return data_ == inline_data(); }

  void clear() noexcept {
    destroy_range(data_, data_ + size_);
    size_ = 0;
  }

  void reserve(size_type new_capacity) noexcept {
    if (new_capacity <= capacity_) {
      return;
    }
    grow_to(new_capacity);
  }

  void resize(size_type new_size) noexcept {
    if (new_size < size_) {
      destroy_range(data_ + new_size, data_ + size_);
      size_ = new_size;
      return;
    }
    reserve(new_size);
    while (size_ < new_size) {
      construct_at(data_ + size_);
      ++size_;
    }
  }

  void resize(size_type new_size, const T &value) noexcept {
    if (new_size < size_) {
      destroy_range(data_ + new_size, data_ + size_);
      size_ = new_size;
      return;
    }
    reserve(new_size);
    while (size_ < new_size) {
      construct_at(data_ + size_, value);
      ++size_;
    }
  }

  template <typename... Args> reference emplace_back(Args &&...args) noexcept {
    if (size_ == capacity_) {
      if (size_ == std::numeric_limits<size_type>::max()) {
        detail::inline_vector_abort();
      }
      grow_for_size(static_cast<size_type>(size_ + 1));
    }
    construct_at(data_ + size_, std::forward<Args>(args)...);
    ++size_;
    return back();
  }

  void push_back(const T &value) noexcept { emplace_back(value); }
  void push_back(T &&value) noexcept { emplace_back(std::move(value)); }

  void pop_back() noexcept {
    assert(size_ > 0);
    --size_;
    destroy_at(data_ + size_);
  }

  iterator insert(const_iterator pos, const T &value) noexcept { return emplace(pos, value); }

  iterator insert(const_iterator pos, T &&value) noexcept { return emplace(pos, std::move(value)); }

  iterator insert(const_iterator pos, size_type count, const T &value) noexcept {
    const size_type start_index = checked_index(pos);
    for (size_type i = 0; i < count; ++i) {
      emplace(data_ + start_index + i, value);
    }
    return data_ + start_index;
  }

  template <typename InputIt>
    requires(!std::is_integral_v<InputIt>)
  iterator insert(const_iterator pos, InputIt first, InputIt last) noexcept {
    const size_type start_index = checked_index(pos);
    size_type insert_index = start_index;
    for (; first != last; ++first, ++insert_index) {
      emplace(data_ + insert_index, *first);
    }
    return data_ + start_index;
  }

  iterator insert(const_iterator pos, std::initializer_list<T> init) noexcept {
    return insert(pos, init.begin(), init.end());
  }

  template <typename... Args> iterator emplace(const_iterator pos, Args &&...args) noexcept {
    const size_type index = checked_index(pos);
    if (index == size_) {
      emplace_back(std::forward<Args>(args)...);
      return data_ + index;
    }

    if (size_ == capacity_) {
      if (size_ == std::numeric_limits<size_type>::max()) {
        detail::inline_vector_abort();
      }
      grow_for_size(static_cast<size_type>(size_ + 1));
    }

    construct_at(data_ + size_, std::move(data_[size_ - 1]));
    for (size_type i = size_ - 1; i > index; --i) {
      data_[i] = std::move(data_[i - 1]);
    }
    data_[index] = T(std::forward<Args>(args)...);
    ++size_;
    return data_ + index;
  }

  iterator erase(const_iterator pos) noexcept {
    assert(pos >= cbegin() && pos < cend());
    return erase(pos, pos + 1);
  }

  iterator erase(const_iterator first, const_iterator last) noexcept {
    assert(first >= cbegin() && first <= cend());
    assert(last >= first && last <= cend());

    const size_type first_index = static_cast<size_type>(first - cbegin());
    const size_type last_index = static_cast<size_type>(last - cbegin());
    const size_type erase_count = last_index - first_index;
    if (erase_count == 0) {
      return data_ + first_index;
    }

    T *erase_begin = data_ + first_index;
    T *erase_end = data_ + last_index;
    T *old_end = data_ + size_;

    if constexpr (std::is_move_assignable_v<T>) {
      std::move(erase_end, old_end, erase_begin);
      destroy_range(old_end - erase_count, old_end);
    } else {
      T *out = erase_begin;
      for (T *it = erase_end; it != old_end; ++it, ++out) {
        destroy_at(out);
        construct_at(out, std::move(*it));
      }
      destroy_range(out, old_end);
    }

    size_ -= erase_count;
    return data_ + first_index;
  }

private:
  size_type checked_index(const_iterator pos) const noexcept {
    assert(pos >= cbegin() && pos <= cend());
    return static_cast<size_type>(pos - cbegin());
  }

  template <typename InputIt> void reserve_for_range(InputIt first, InputIt last) noexcept {
    if constexpr (std::forward_iterator<InputIt>) {
      reserve(
          detail::checked_inline_vector_size(static_cast<std::size_t>(std::distance(first, last))));
    }
  }

  void register_construction(detail::inline_vector_source_location source_location) noexcept {
#ifdef UTIL_INLINE_VECTOR_ENABLE_HISTOGRAM
    detail::inline_vector_histogram_registry::instance().register_vector(this, source_location);
#else
    (void)source_location;
#endif
  }

  void record_destruction() noexcept {
#ifdef UTIL_INLINE_VECTOR_ENABLE_HISTOGRAM
    detail::inline_vector_histogram_registry::instance().record_destruction(this, size_);
#endif
  }

  template <typename... Args> static void construct_at(T *ptr, Args &&...args) noexcept {
    ::new (static_cast<void *>(ptr)) T(std::forward<Args>(args)...);
  }

  static void destroy_at(T *ptr) noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      ptr->~T();
    }
  }

  static void destroy_range(T *first, T *last) noexcept {
    while (last != first) {
      --last;
      destroy_at(last);
    }
  }

  static T *allocate(size_type capacity) noexcept {
    if (capacity == 0) {
      return nullptr;
    }
    if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      detail::inline_vector_abort();
    }
    void *ptr = ::operator new(sizeof(T) * static_cast<std::size_t>(capacity), std::nothrow);
    if (ptr == nullptr) {
      detail::inline_vector_abort();
    }
    return static_cast<T *>(ptr);
  }

  static void deallocate(T *ptr) noexcept { ::operator delete(ptr); }

  T *inline_data() noexcept { return inline_storage_.data(); }
  const T *inline_data() const noexcept { return inline_storage_.data(); }

  void reset_to_inline() noexcept {
    data_ = inline_data();
    size_ = 0;
    capacity_ = InlineN;
  }

  void deallocate_dynamic() noexcept {
    if (!is_inline()) {
      deallocate(data_);
    }
  }

  void copy_construct_from(const T *first, const T *last) noexcept {
    for (const T *it = first; it != last; ++it) {
      construct_at(data_ + size_, *it);
      ++size_;
    }
  }

  void copy_from(const inline_vector &other) noexcept {
    reserve(other.size_);
    copy_construct_from(other.data_, other.data_ + other.size_);
  }

  void move_from(inline_vector &&other) noexcept {
    if (!other.is_inline()) {
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.reset_to_inline();
      return;
    }

    reserve(other.size_);
    for (T &value : other) {
      construct_at(data_ + size_, std::move(value));
      ++size_;
    }
    other.clear();
  }

  void grow_for_size(size_type min_capacity) noexcept {
    uint64_t next = capacity_ == 0 ? 1 : static_cast<uint64_t>(capacity_) * 2;
    next = std::max<uint64_t>(next, min_capacity);
    if (next > std::numeric_limits<size_type>::max()) {
      detail::inline_vector_abort();
    }
    grow_to(static_cast<size_type>(next));
  }

  void grow_to(size_type new_capacity) noexcept {
    T *new_data = allocate(new_capacity);
    const size_type old_size = size_;
    for (size_type i = 0; i < old_size; ++i) {
      construct_at(new_data + i, std::move(data_[i]));
    }

    destroy_range(data_, data_ + old_size);
    deallocate_dynamic();
    data_ = new_data;
    size_ = old_size;
    capacity_ = new_capacity;
  }

  [[no_unique_address]] detail::InlineVectorStorage<T, InlineN> inline_storage_;
  T *data_;
  size_type size_ = 0;
  size_type capacity_ = 0;
};

#ifdef UTIL_INLINE_VECTOR_ENABLE_HISTOGRAM
inline void dump_inline_vector_histograms(std::ostream &os) noexcept {
  detail::inline_vector_histogram_registry::instance().write_jsonl(os);
}

inline void reset_inline_vector_histograms_for_testing() noexcept {
  detail::inline_vector_histogram_registry::instance().reset_for_testing();
}
#endif

} // namespace util

#endif // UTIL_INLINE_VECTOR_H_
