#ifndef RCCL_TEST_HOST_RAS_MESSAGE_TEST_SUPPORT_H_
#define RCCL_TEST_HOST_RAS_MESSAGE_TEST_SUPPORT_H_

#include <memory>
#include <gtest/gtest.h>
#include "ras/ras_internal.h"

namespace ras_test {
class OwnedMsg {
 public:
  OwnedMsg(size_t length, ncclResult_t (*allocate)(rasMsg**, size_t), void (*release)(rasMsg*))
      : message_(nullptr, release) {
    rasMsg* message = nullptr;
    const auto result = allocate(&message, length);
    message_.reset(message);
    EXPECT_EQ(ncclSuccess, result);
  }
  rasMsg* get() const { return message_.get(); }
  rasMsg* operator->() const { return get(); }
  rasMsg* release() { return message_.release(); }

 private:
  std::unique_ptr<rasMsg, void (*)(rasMsg*)> message_;
};
}

#endif
