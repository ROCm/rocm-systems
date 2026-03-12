#include "thunks.h"
#include "device.h"
#include "lda_chain.h"
#include "thunk_proxy/thunk_proxy.h"
#include <memory>

using namespace std;

namespace wsl {
namespace thunk {

Device::Device(Platform *platform, LdaChain *lda_chain, u32 chainIndex,
               std::unique_ptr<thunk_proxy::DeviceContext> device_ctx)
    : device_ctx_(std::move(device_ctx)),
      lda_chain_(lda_chain),
      chain_index_(chainIndex) {
  (void)platform;
}

ErrorCode Device::Create(Platform *platform, LdaChain *ldaChain,
                         u32 deviceIndex, u32 chainIndex,
                         const thunk_proxy::DeviceInfo &deviceInfo,
                         Device **deviceOut) {
  (void)deviceIndex;
  (void)deviceInfo;

  auto dctx = std::unique_ptr<thunk_proxy::DeviceContext>(
      ldaChain->GetChainContext()->CreateDevice(
          ldaChain->DeviceHandle(), chainIndex));

  if (!dctx)
    return ErrorCode::InitializationFailed;

  if (!dctx->IsWddm2Supported())
    return ErrorCode::InitializationFailed;

  *deviceOut = new Device(platform, ldaChain, chainIndex, std::move(dctx));
  return ErrorCode::Success;
}

ErrorCode Device::QueryVramInfo(VramInfo *info) const {
  return device_ctx_->QueryVramInfo(info);
}

ErrorCode Device::QueryVramUsage(VramUsage *usage) const {
  return device_ctx_->QueryVramUsage(usage);
}

ErrorCode Device::QueryRasFeature(RasFeature *info) const {
  return device_ctx_->QueryRasFeature(info);
}

ErrorCode Device::QueryBdfInfo(BdfInfo *info) const {
  return device_ctx_->QueryBdfInfo(info);
}

ErrorCode Device::QueryAsicInfo(AsicInfo *info) const {
  return device_ctx_->QueryAsicInfo(info);
}

ErrorCode Device::QueryVBiosInfo(VBiosInfo *info) const {
  return device_ctx_->QueryVBiosInfo(info);
}

ErrorCode Device::Escape(void *pData, size_t dataSize, bool hardwareAccess) const {
  return device_ctx_->Escape(pData, dataSize, hardwareAccess);
}

} // namespace thunk
} // namespace wsl
