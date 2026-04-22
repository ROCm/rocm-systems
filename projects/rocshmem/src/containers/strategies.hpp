/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_CONTAINERS_STRATEGIES_HPP_
#define LIBRARY_SRC_CONTAINERS_STRATEGIES_HPP_

#include "index_strategy.hpp"
#include "share_strategy.hpp"

namespace rocshmem {

struct ObjectStrategy {
  ShareStrategy share_strategy{};
  IndexStrategy index_strategy_one{};
  IndexStrategy index_strategy_two{};
  IndexStrategy index_strategy_three{};
  IndexStrategy index_strategy_four{};
};

class DefaultObjectStrategy {
 private:
  static DefaultObjectStrategy* _instance;

  DefaultObjectStrategy() {
    _os.share_strategy = ShareStrategy(ShareStrategyEnum::PRIVATE);
    _os.index_strategy_one = IndexStrategy(IndexStrategyEnum::TDBD);
    _os.index_strategy_two = IndexStrategy(IndexStrategyEnum::TDBD);
    _os.index_strategy_three = IndexStrategy(IndexStrategyEnum::TDBD);
    _os.index_strategy_four = IndexStrategy(IndexStrategyEnum::TDBD);
  }

  ObjectStrategy _os{};

 public:
  static DefaultObjectStrategy* instance();

  const ObjectStrategy* get() { return &_os; }
};

struct Strategies {};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTAINERS_STRATEGIES_HPP_
