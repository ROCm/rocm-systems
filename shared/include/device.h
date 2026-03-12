#ifndef _WSL_INC_WDDM_DEVICE_H_
#define _WSL_INC_WDDM_DEVICE_H_

#include <memory>
#include <vector>

#include "d3dkmt_types.h"
#include "gpu_info.h"
#include "status.h"
#include "thunk_proxy/thunk_proxy.h"

namespace wsl {
namespace thunk {

constexpr gpusize PageSize = 0x1000u;

class Platform;
class LdaChain;

class Device {
public:
  static ErrorCode Create(Platform *platform, LdaChain *ldaChain,
                          u32 deviceIndex, u32 chainIndex,
                          const thunk_proxy::DeviceInfo &deviceInfo,
                          Device **deviceOut);

  LdaChain *GetLdaChain()   const { return lda_chain_; }
  u32       GetChainIndex() const { return chain_index_; }

  ErrorCode QueryVramUsage(VramUsage *usage) const;

  ErrorCode QueryVramInfo(VramInfo *info) const;

  ErrorCode Escape(void *pData, size_t dataSize,
                   bool hardwareAccess = false) const;

private:
  Device(Platform *platform, LdaChain *lda_chain, u32 chainIndex,
         std::unique_ptr<thunk_proxy::DeviceContext> device_ctx);

  std::unique_ptr<thunk_proxy::DeviceContext> device_ctx_;
  LdaChain *const lda_chain_ = nullptr;
  const u32       chain_index_ = 0;
};

} // namespace thunk
} // namespace wsl

#endif
