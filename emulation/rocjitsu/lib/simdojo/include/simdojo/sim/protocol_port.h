// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "simdojo/sim/component.h"

#include <concepts>
#include <string_view>
#include <typeinfo>

namespace simdojo {

/// The protocol name describes the modeled traffic. The C++ type determines
/// which messages the input handler can consume in this simulation.
class ProtocolPort : public Port {
public:
  [[nodiscard]] virtual std::string_view protocol_name() const noexcept = 0;
  [[nodiscard]] virtual const std::type_info &message_type() const noexcept = 0;

protected:
  ProtocolPort(std::string name, PortID id, Component *owner, PortDirection direction)
      : Port(std::move(name), id, owner, direction, PortProtocol::SPECIALIZED) {}
};

template <std::derived_from<Message> Packet, PortDirection Direction>
class TypedPort final : public ProtocolPort {
public:
  TypedPort(std::string name, PortID id, Component *owner)
      : ProtocolPort(std::move(name), id, owner, Direction) {
    static_assert(!std::string_view(Packet::protocol_name).empty());
  }

  std::string_view protocol_name() const noexcept override { return Packet::protocol_name; }
  const std::type_info &message_type() const noexcept override { return typeid(Packet); }

  void send(std::unique_ptr<Packet> packet)
    requires(Direction == PortDirection::OUT)
  {
    Port::send(std::move(packet));
  }

  template <class Handler>
  void set_handler(Handler handler)
    requires(Direction == PortDirection::IN)
  {
    Port::set_handler([handler = std::move(handler)](Tick tick, Message *message) {
      if (auto *packet = dynamic_cast<Packet *>(message)) {
        handler(tick, *packet);
      }
    });
  }
};

} // namespace simdojo
