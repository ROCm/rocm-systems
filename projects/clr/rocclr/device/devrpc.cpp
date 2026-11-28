/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/devrpc.hpp"

#if defined(ROCCLR_ENABLE_RPC)

#include "utils/debug.hpp"
#include "top.hpp"

#include <cstring>

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"
#include "shared/rpc_server.h"

namespace amd {

bool rpcAvailable() { return true; }

size_t getRpcBufferSize(uint32_t num_lanes, uint32_t num_ports) {
  return rpc::Server::allocation_size(num_lanes, num_ports);
}

static uintptr_t allocateMemory(amd::Device& dev, size_t size, size_t alignment) {
  amd::Context& ctx = dev.context();
  amd::Buffer* buffer = new (ctx) amd::Buffer(ctx, CL_MEM_READ_WRITE, size, nullptr, alignment);
  if (buffer == nullptr || !buffer->create()) {
    if (buffer != nullptr) {
      buffer->release();
    }
    return 0;
  }

  device::Memory* memory = buffer->getDeviceMemory(dev);
  uintptr_t address = memory != nullptr ? memory->virtualAddress() : 0;
  if (address == 0) {
    buffer->release();
    return 0;
  }

  amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(address), buffer);
  dev.TrackHostcallMemory(buffer);
  return address;
}

static bool releaseMemory(amd::Device& dev, void* ptr) {
  amd::Memory* memory = amd::MemObjMap::FindAndRemoveMemObj(ptr);
  if (memory == nullptr) {
    return false;
  }
  dev.RemoveHostcallMemory(memory);
  memory->release();
  return true;
}

static rpc::RPCStatus handleClrOpcodes(rpc::Server::Port& port, RpcBufferInfo* info) {
  amd::Device& dev = *info->device;
  switch (port.get_opcode()) {
    case LIBC_MALLOC: {
      port.recv_and_send([&](rpc::Buffer* buffer, uint32_t) {
        uint64_t size = buffer->data[0];
        constexpr size_t kLargePageAlign = 2 * Mi;
        buffer->data[0] = allocateMemory(dev, size, kLargePageAlign);
      });
      break;
    }
    case LIBC_FREE: {
      port.recv([&](rpc::Buffer* buffer, uint32_t) {
        void* ptr = reinterpret_cast<void*>(buffer->data[0]);
        if (ptr && !releaseMemory(dev, ptr)) {
          ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "RPC: unknown pointer %p in free", ptr);
        }
      });
      break;
    }
    default:
      return rpc::RPC_UNHANDLED_OPCODE;
  }
  return rpc::RPC_SUCCESS;
}

static void servicePort(RpcBufferInfo* info, rpc::Server::Port& port) {
  rpc::RPCStatus status = handleClrOpcodes(port, info);

  if (status == rpc::RPC_UNHANDLED_OPCODE) status = rpc::handle_libc_opcodes(port, info->num_lanes);

  if (status != rpc::RPC_SUCCESS) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "RPC: opcode 0x%x failed with status 0x%x",
            port.get_opcode(), status);
  }
}

bool processRpcBuffer(RpcBufferInfo* info) {
  rpc::Server server(info->port_count, info->buffer);

  // Drain every port that is ready this pass.
  bool serviced = false;
  for (uint32_t n = 0; n < info->port_count; ++n) {
    {
      auto port = server.try_open(info->num_lanes);
      if (!port) break;
      servicePort(info, *port);
      serviced = true;
    }
  }
  return serviced;
}

void flushRpcBuffer(RpcBufferInfo* info) { processRpcBuffer(info); }

size_t getRpcClientSize() { return sizeof(rpc::Client); }

void initRpcClient(void* staging, void* buffer, uint32_t num_ports) {
  rpc::Client client(num_ports, buffer);
  std::memcpy(staging, &client, sizeof(rpc::Client));
}

void initRpcDoorbell(void* buffer, device::Signal& signal, device::Signal* interrupt_signal) {
  auto* doorbell =
      reinterpret_cast<rpc::Doorbell*>(static_cast<char*>(buffer) + rpc::Server::doorbell_offset());
  auto counter = signal.getDoorbell();
  auto interrupt = interrupt_signal != nullptr ? interrupt_signal->getDoorbell() : counter;
  doorbell->value = counter.value;
  doorbell->mailbox = interrupt.mailbox;
  doorbell->event_id = interrupt.eventId;
}

}  // namespace amd

#else  // !ROCCLR_ENABLE_RPC

namespace amd {

bool rpcAvailable() { return false; }
size_t getRpcBufferSize(uint32_t, uint32_t) { return 0; }
bool processRpcBuffer(RpcBufferInfo*) { return false; }
void flushRpcBuffer(RpcBufferInfo*) {}
size_t getRpcClientSize() { return 0; }
void initRpcClient(void*, void*, uint32_t) {}
void initRpcDoorbell(void*, device::Signal&, device::Signal*) {}

}  // namespace amd

#endif
