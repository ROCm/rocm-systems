// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integration test: verifies that find_package(simdojo) works end-to-end by
// building a minimal simulation against an installed simdojo package.

#include <simdojo/sim/simulation.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace simdojo;

class Producer : public Component {
public:
  Producer() : Component("producer") {
    out_ = add_port(std::make_unique<Port>(
        "out", 0, this, PortDirection::OUT, PortProtocol::UNTYPED));
    ev_.set_handler([this](Tick, Message *) {
      out_->send(std::make_unique<Message>());
    });
  }
  void startup() override { schedule_event(&ev_, 1); }
  Port *out_port() { return out_; }

private:
  Port *out_ = nullptr;
  Event ev_{this, EventType::TIMER_CALLBACK};
};

class Consumer : public Component {
public:
  Consumer() : Component("consumer") {
    in_ = add_port(std::make_unique<Port>(
        "in", 0, this, PortDirection::IN, PortProtocol::UNTYPED));
    in_->set_handler([this](Tick ts, Message *) { ticks.push_back(ts); });
  }
  Port *in_port() { return in_; }
  std::vector<Tick> ticks;

private:
  Port *in_ = nullptr;
};

int main() {
  SimulationEngine engine({.max_ticks = 1000, .num_threads = 1});
  auto root = std::make_unique<CompositeComponent>("root");

  auto *p = root->add_child(std::make_unique<Producer>());
  auto *c = root->add_child(std::make_unique<Consumer>());
  auto *prod = static_cast<Producer *>(p);
  auto *cons = static_cast<Consumer *>(c);

  engine.topology().set_root(std::move(root));
  engine.topology().add_link(prod->out_port(), cons->in_port(), 42);
  engine.build();
  auto exit = engine.run();

  assert(exit.reason == ExitReason::COMPLETED);
  assert(cons->ticks.size() == 1);
  assert(cons->ticks[0] == 43);

  std::printf("PASS: message arrived at tick %lu (expected 43)\n",
              static_cast<unsigned long>(cons->ticks[0]));
  return 0;
}
