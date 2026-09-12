/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/devrpc.hpp"

#if __has_include("shared/rpc.h")

#include "utils/debug.hpp"
#include "top.hpp"

#include "device/devhcmessages.hpp"

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "device/devsanitizer.hpp"
#endif
#endif

#include <cstring>

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"
#include "shared/rpc_server.h"

namespace amd {

bool rpcAvailable() { return true; }

size_t getRpcBufferSize(uint32_t num_lanes, uint32_t num_ports) {
  return rpc::Server::allocation_size(num_lanes, num_ports);
}

static rpc::RPCStatus handleClrOpcodes(rpc::Server::Port& port, const amd::Device& dev,
                                       MessageHandler& messages) {
  switch (port.get_opcode()) {
    case LIBC_MALLOC: {
      port.recv_and_send([&](rpc::Buffer* Buffer, uint32_t) {
        amd::Context& ctx = dev.context();
        uint64_t size = Buffer->data[0];
        constexpr size_t kLargePageAlign = 2 * Mi;
        amd::Buffer* buf =
            new (ctx) amd::Buffer(ctx, CL_MEM_READ_WRITE, size, nullptr, kLargePageAlign);
        uintptr_t va = 0;
        if (buf && buf->create()) {
          va = buf->getDeviceMemory(dev)->virtualAddress();
          amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(va), buf);
          const_cast<amd::Device*>(&dev)->TrackHostcallMemory(buf);
        } else if (buf) {
          buf->release();
        }
        Buffer->data[0] = va;
      });
      break;
    }
    case LIBC_FREE: {
      port.recv([&](rpc::Buffer* Buffer, uint32_t) {
        void* Ptr = reinterpret_cast<void*>(Buffer->data[0]);
        if (Ptr) {
          amd::Memory* mem = amd::MemObjMap::FindMemObj(Ptr);
          if (mem) {
            const_cast<amd::Device*>(&dev)->RemoveHostcallMemory(mem);
            amd::MemObjMap::RemoveMemObj(Ptr);
            mem->release();
          }
        }
      });
      break;
    }
    case ROCM_HOSTCALL_FUNCTION: {
      port.recv_and_send([&](rpc::Buffer* Buffer, uint32_t) {
        typedef void (*HostcallFn)(uint64_t* output, const uint64_t* input);
        auto fptr = reinterpret_cast<HostcallFn>(Buffer->data[0]);
        uint64_t output[2] = {};
        fptr(output, &Buffer->data[1]);
        Buffer->data[0] = output[0];
        Buffer->data[1] = output[1];
      });
      break;
    }
    case ROCM_HOSTCALL_DEVMEM: {
      port.recv_and_send([&](rpc::Buffer* Buffer, uint32_t) {
        uint64_t addr = Buffer->data[0];
        uint64_t size = Buffer->data[1];
        guarantee(addr != 0 || size != 0, "ROCM_HOSTCALL_DEVMEM: both addr and size are zero");
        if (addr) {
          amd::Memory* mem = amd::MemObjMap::FindMemObj(reinterpret_cast<void*>(addr));
          if (mem) {
            const_cast<amd::Device*>(&dev)->RemoveHostcallMemory(mem);
            amd::MemObjMap::RemoveMemObj(reinterpret_cast<void*>(addr));
            mem->release();
          } else {
            ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "ROCM_HOSTCALL_DEVMEM: unknown pointer %p",
                    reinterpret_cast<void*>(addr));
          }
          Buffer->data[0] = 0;
        } else {
          amd::Context& ctx = dev.context();
          amd::Buffer* buf = new (ctx)
              amd::Buffer(ctx, CL_MEM_READ_WRITE, size, nullptr, (size == 2 * Mi) ? 2 * Mi : 0);
          uint64_t va = 0;
          if (buf) {
            if (buf->create()) {
              device::Memory* dm = buf->getDeviceMemory(dev);
              va = dm->virtualAddress();
              amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(va), buf);
              const_cast<amd::Device*>(&dev)->TrackHostcallMemory(buf);
            } else {
              buf->release();
            }
          }
          Buffer->data[0] = va;
        }
      });
      break;
    }
    case ROCM_HOSTCALL_PRINTF: {
      port.recv_and_send([&](rpc::Buffer* Buffer, uint32_t) {
        if (!messages.handlePayload(SERVICE_PRINTF, Buffer->data)) {
          ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "ROCM_HOSTCALL_PRINTF: invalid message payload");
        }
      });
      break;
    }
#if defined(__clang__)
#if __has_feature(address_sanitizer)
    case ROCM_HOSTCALL_SANITIZER: {
      constexpr uint32_t kMaxLanes = 64;
      uint64_t addrs[kMaxLanes] = {};
      uint64_t callstack[1] = {};
      uint64_t entity_id[kMaxLanes + 4] = {};
      uint64_t access_info = 0, acc_size = 0;
      uint32_t n_lanes = 0;
      entity_id[0] = dev.index();

      port.recv([&](rpc::Buffer* Buffer, uint32_t) {
        addrs[n_lanes] = Buffer->data[0];
        if (n_lanes == 0) {
          callstack[0] = Buffer->data[1];
          entity_id[1] = Buffer->data[2];
          entity_id[2] = Buffer->data[3];
          entity_id[3] = Buffer->data[4];
        }
        entity_id[4 + n_lanes] = Buffer->data[5];
        access_info = Buffer->data[6];
        acc_size = Buffer->data[7];
        n_lanes++;
      });

      bool is_write = (access_info & 1) != 0;
      bool is_abort = (access_info & 0xFFFFFFFF00000000ULL) == 0;
      auto uri_fd = amd::Os::FDescInit();
      __asan_report_nonself_error(callstack, 1, addrs, n_lanes, entity_id, n_lanes + 4, is_write,
                                  acc_size, is_abort, "amdgpu", 0, uri_fd, 0, 0);
      port.send([](rpc::Buffer*, uint32_t) {});
      break;
    }
#endif
#endif
    default:
      return rpc::RPC_UNHANDLED_OPCODE;
  }
  return rpc::RPC_SUCCESS;
}

static void servicePort(RpcBufferInfo* info, rpc::Server::Port& port) {
  rpc::RPCStatus status = handleClrOpcodes(port, *info->device, *info->messages);

  if (status == rpc::RPC_UNHANDLED_OPCODE)
    status = rpc::handle_libc_opcodes(port, info->num_lanes);

  if (status != rpc::RPC_SUCCESS && status != rpc::RPC_UNHANDLED_OPCODE) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "RPC: Unhandled or invalid opcode");
  }
}

bool processRpcBuffer(RpcBufferInfo* info) {
  rpc::Server server(info->port_count, info->buffer);

  // Drain every port that is ready this pass.
  bool serviced = false;
  for (uint32_t n = 0; n < info->port_count; ++n) {
    auto port = server.try_open(info->num_lanes);
    if (!port) break;
    servicePort(info, *port);
    serviced = true;
  }
  return serviced;
}

void flushRpcBuffer(RpcBufferInfo* info) {
  rpc::Server server(info->port_count, info->buffer);

  while (auto port = server.try_open(info->num_lanes)) {
    servicePort(info, *port);
  }
}

size_t getRpcClientSize() { return sizeof(rpc::Client); }

void initRpcClient(void* staging, void* buffer, uint32_t num_ports) {
  rpc::Client client(num_ports, buffer);
  std::memcpy(staging, &client, sizeof(rpc::Client));
}

void initRpcDoorbell(void* buffer, void* signal_handle) {
  // Must match the prefix of amd_signal_t in ROCR's core/inc/amd_signal.h.
  // The KFD event fields are not exposed through the public HSA API.
  struct AMDSignal {
    int64_t kind;
    int64_t value;
    uint64_t event_mailbox_ptr;
    uint32_t event_id;
  };
  auto* sig = reinterpret_cast<AMDSignal*>(signal_handle);
  auto* doorbell =
      reinterpret_cast<rpc::Doorbell*>(static_cast<char*>(buffer) + rpc::Server::doorbell_offset());
  doorbell->value = reinterpret_cast<uint64_t*>(&sig->value);
  doorbell->mailbox = reinterpret_cast<uint64_t*>(sig->event_mailbox_ptr);
  doorbell->event_id = sig->event_id;
}

}  // namespace amd

#else  // !__has_include("shared/rpc.h")

namespace amd {

bool rpcAvailable() { return false; }
size_t getRpcBufferSize(uint32_t, uint32_t) { return 0; }
bool processRpcBuffer(RpcBufferInfo*) { return false; }
void flushRpcBuffer(RpcBufferInfo*) {}
size_t getRpcClientSize() { return 0; }
void initRpcClient(void*, void*, uint32_t) {}
void initRpcDoorbell(void*, void*) {}

}  // namespace amd

#endif
