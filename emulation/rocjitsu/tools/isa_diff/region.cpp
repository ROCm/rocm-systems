// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "region.h"

#include "artifact.h"

namespace rocjitsu::isa_diff {
namespace {

Json side_to_json(const RegionSide &side) {
  Json j = Json::object();
  j.set("begin_index", static_cast<uint64_t>(side.begin));
  j.set("end_index", static_cast<uint64_t>(side.end));
  Json insts = Json::array();
  for (const auto &inst : side.insts) {
    Json ij = Json::object();
    ij.set("pc", hex64(inst.pc));
    Json words = Json::array();
    for (uint32_t w : inst.words)
      words.push(hex32(w));
    ij.set("words", std::move(words));
    ij.set("asm", inst.text);
    insts.push(std::move(ij));
  }
  j.set("insts", std::move(insts));
  return j;
}

bool side_from_json(const Json &j, RegionSide &side, std::string &error) {
  side.begin = j["begin_index"].as_uint();
  side.end = j["end_index"].as_uint();
  for (const Json &ij : j["insts"].items()) {
    RegionInst inst;
    auto pc = json_u64(ij["pc"]);
    if (!pc) {
      error = "region instruction without a pc";
      return false;
    }
    inst.pc = *pc;
    for (const Json &w : ij["words"].items()) {
      auto v = json_u64(w);
      if (!v || *v > 0xffffffffu) {
        error = "bad instruction word";
        return false;
      }
      inst.words.push_back(static_cast<uint32_t>(*v));
    }
    if (inst.words.empty()) {
      error = "region instruction without words";
      return false;
    }
    inst.text = ij["asm"].as_string();
    side.insts.push_back(std::move(inst));
  }
  return true;
}

} // namespace

const char *region_status_name(RegionStatus s) {
  switch (s) {
  case RegionStatus::Ready:
    return "READY";
  case RegionStatus::Unsupported:
    return "UNSUPPORTED";
  case RegionStatus::Inconclusive:
    return "INCONCLUSIVE";
  }
  return "?";
}

Json RegionPair::to_json() const {
  Json j = Json::object();
  j.set("id", id);
  j.set("status", region_status_name(status));
  j.set("reason", reason);
  j.set("synthetic", synthetic);
  if (!parent.empty()) {
    j.set("parent", parent);
    j.set("context_before", context_before);
    j.set("context_after", context_after);
  }
  Json hj = Json::array();
  for (size_t h : hunks)
    hj.push(static_cast<uint64_t>(h));
  j.set("hunks", std::move(hj));
  j.set("old", side_to_json(old_side));
  j.set("new", side_to_json(new_side));
  j.set("contract", contract.to_json());
  j.set("prepared_key", prepared_key);
  return j;
}

std::optional<RegionPair> RegionPair::from_json(const Json &j, std::string &error) {
  RegionPair r;
  r.id = j["id"].as_string();
  const std::string &status = j["status"].as_string();
  if (status == "READY")
    r.status = RegionStatus::Ready;
  else if (status == "INCONCLUSIVE")
    r.status = RegionStatus::Inconclusive;
  else
    r.status = RegionStatus::Unsupported;
  r.reason = j["reason"].as_string();
  r.synthetic = j["synthetic"].as_bool();
  r.parent = j["parent"].as_string();
  r.context_before = static_cast<uint32_t>(j["context_before"].as_uint());
  r.context_after = static_cast<uint32_t>(j["context_after"].as_uint());
  for (const Json &h : j["hunks"].items())
    r.hunks.push_back(h.as_uint());
  if (!side_from_json(j["old"], r.old_side, error) || !side_from_json(j["new"], r.new_side, error))
    return std::nullopt;
  auto contract = RegionContract::from_json(j["contract"], error);
  if (!contract)
    return std::nullopt;
  r.contract = std::move(*contract);
  r.prepared_key = j["prepared_key"].as_string();
  return r;
}

std::string compute_prepared_key(const RegionPair &region) {
  // Bind every input of a prepared region: both instruction streams with their PCs,
  // the complete contract (so a liveness change invalidates), and the execution mode.
  Json j = Json::object();
  j.set("schema", "rj_isa_diff.prepared_region.v1");
  j.set("old", side_to_json(region.old_side)["insts"]);
  j.set("new", side_to_json(region.new_side)["insts"]);
  Json contract = region.contract.to_json();
  j.set("contract", std::move(contract));
  return sha256_hex(j.dump(-1));
}

} // namespace rocjitsu::isa_diff
