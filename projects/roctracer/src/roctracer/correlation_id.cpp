/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "correlation_id.h"
#include "roctracer.h"

#include <atomic>
#include <stack>
#include <vector>

namespace {

// A stack that can be used for TLS variables. TLS destructors are invoked before global destructors
// which is a problem if operations invoked by global destructors use TLS variables. If the TLS
// stack is destructed, it still has well defined behavior by always returning a dummy element.
template <typename T> class Stack : std::stack<T, std::vector<T>> {
  using parent_type = typename std::stack<T, std::vector<T>>;

 public:
  Stack() { valid_.store(true, std::memory_order_relaxed); }
  ~Stack() { valid_.store(false, std::memory_order_relaxed); }

  template <class... Args> auto& emplace(Args&&... args) {
    return is_valid() ? parent_type::emplace(std::forward<Args>(args)...)
                      : dummy_element_ = T(std::forward<Args>(args)...);
  }
  void push(const T& v) {
    if (is_valid()) parent_type::push(v);
  }
  void push(T&& v) {
    if (is_valid()) parent_type::push(std::move(v));
  }
  void pop() {
    if (is_valid()) parent_type::pop();
  }
  const auto& top() const { return is_valid() ? parent_type::top() : dummy_element_; }
  auto& top() { return is_valid() ? parent_type::top() : (dummy_element_ = {}); }

  bool is_valid() const { return valid_.load(std::memory_order_relaxed); }
  size_t size() const { return is_valid() ? parent_type::size() : 0; }
  bool empty() const { return size() == 0; }

 private:
  std::atomic<bool> valid_{false};
  T dummy_element_;  // Dummy element used when the stack is not valid.
};

thread_local Stack<activity_correlation_id_t> correlation_id_stack{};
thread_local Stack<activity_correlation_id_t> external_id_stack{};

}  // namespace

namespace roctracer {

activity_correlation_id_t CorrelationIdPush() {
  static std::atomic<uint64_t> counter{1};
  return correlation_id_stack.emplace(counter.fetch_add(1, std::memory_order_relaxed));
}

void CorrelationIdPop() { correlation_id_stack.pop(); }

activity_correlation_id_t CorrelationId() {
  return correlation_id_stack.empty() ? 0 : correlation_id_stack.top();
}

void ExternalCorrelationIdPush(activity_correlation_id_t external_id) {
  external_id_stack.push(external_id);
}

std::optional<activity_correlation_id_t> ExternalCorrelationIdPop() {
  if (external_id_stack.empty()) return std::nullopt;

  auto external_id = external_id_stack.top();
  external_id_stack.pop();
  return std::make_optional(external_id);
}

std::optional<activity_correlation_id_t> ExternalCorrelationId() {
  return external_id_stack.empty() ? std::nullopt : std::make_optional(external_id_stack.top());
}

}  // namespace roctracer