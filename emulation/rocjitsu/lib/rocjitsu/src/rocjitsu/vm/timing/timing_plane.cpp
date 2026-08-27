// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/timing_plane.h"

#include "rocjitsu/vm/timing/dispatch_des.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <unistd.h>

namespace rocjitsu::timing {
namespace {

std::uint64_t frequency_hz(double megahertz) {
  const double hertz = megahertz * 1'000'000.0;
  return hertz > 0.0 ? static_cast<std::uint64_t>(hertz) : 1'000'000'000ULL;
}

std::uint64_t packed_key(std::uint32_t dispatch_id, std::uint32_t queue_id) {
  // Both halves, never the dispatch id alone. Ids are allocated per command
  // processor and a multi-die part has one per die, so they collide across
  // dies and a bare id would close some other die's live dispatch.
  return (static_cast<std::uint64_t>(queue_id) << 32) | dispatch_id;
}

} // namespace

TimingPlane::TimingPlane(Tuning tuning)
    : tuning_(std::move(tuning)), shader_("shader", frequency_hz(tuning_.clock_mhz)),
      fabric_("fabric", frequency_hz(tuning_.clock_mhz)),
      memory_("memory", frequency_hz(tuning_.clock_mhz)) {
  // Opened once, appended per dispatch, closed by the destructor. Line
  // buffered rather than block buffered: a run that is killed for taking too
  // long is exactly the run whose trace is worth having.
  if (const char *path = std::getenv("ROCJITSU_TIMING_TRACE"); path != nullptr && *path != '\0') {
    // Appended, not truncated, and every line carries the process that wrote
    // it. A guest that forks -- which any torch run with a compiler in it does
    // -- gives each child its own plane, and each child would otherwise open
    // this path for writing and throw away everything the parent had written.
    trace_ = std::fopen(path, "ae");
    if (trace_ != nullptr)
      std::setvbuf(trace_, nullptr, _IOLBF, 0);
  }

  if (!tuning_.enabled)
    return;

  const std::uint32_t units =
      static_cast<std::uint32_t>(std::max<std::uint64_t>(1, tuning_.compute_units));
  const std::uint32_t dies = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, tuning_.xcds));
  const std::uint32_t channels =
      static_cast<std::uint32_t>(std::max<std::uint64_t>(1, tuning_.memory_channels));

  // The line pool is reserved once and never reallocated while a request is in
  // flight: a request refers to its lines by an index into it, and a
  // reallocation under an in-flight request would leave that index pointing at
  // freed storage.
  line_pool_.reserve(1 << 20);

  // Memory side first, so each level exists before the level above is told
  // where to forward to.
  for (std::uint32_t channel = 0; channel < channels; ++channel) {
    dram_.push_back(std::make_unique<ChannelDes>(
        "dram" + std::to_string(channel), memory_, engine_, tuning_.dram_bytes_per_cycle / channels,
        tuning_.dram_latency_cycles, tuning_.dram_row_miss_cycles));
    mall_.push_back(std::make_unique<CacheDes>("mall" + std::to_string(channel), memory_, engine_,
                                               tuning_.mall, &line_pool_));
    fabric_channels_.push_back(
        std::make_unique<ChannelDes>("fabric" + std::to_string(channel), fabric_, engine_,
                                     tuning_.fabric_bytes_per_cycle / channels, 0));
  }
  for (std::uint32_t die = 0; die < dies; ++die)
    l2_.push_back(std::make_unique<CacheDes>("l2_" + std::to_string(die), shader_, engine_,
                                             tuning_.l2, &line_pool_));
  for (std::uint32_t unit = 0; unit < units; ++unit) {
    l1_vector_.push_back(std::make_unique<CacheDes>("l1v" + std::to_string(unit), shader_, engine_,
                                                    tuning_.l1_vector, &line_pool_));
    l1_scalar_.push_back(std::make_unique<CacheDes>("l1s" + std::to_string(unit), shader_, engine_,
                                                    tuning_.l1_scalar, &line_pool_));
    l1_instruction_.push_back(std::make_unique<CacheDes>(
        "l1i" + std::to_string(unit), shader_, engine_, tuning_.l1_instruction, &line_pool_));
    compute_units_.push_back(std::make_unique<ComputeUnitDes>(
        "cu_des" + std::to_string(unit), shader_, engine_, tuning_, *this, unit));
  }

  cu_base_.assign(units, 0);
  cu_dispatch_.assign(units, 0);

  // Wire it. A completion anywhere records the tick against the request that is
  // being waited on; a miss travels one level further out.
  // An access is answered when its LAST part lands, not its first. A request
  // that crossed several memory channels is several requests from here on, and
  // taking the first completion would report the latency of whichever part
  // happened to be quickest.
  const auto complete = [this](const MemoryRequest &request, std::uint64_t tick) {
    if (request.origin.sequence != awaited_ || answered_)
      return;
    completion_tick_ = std::max(completion_tick_, tick);
    if (awaited_outstanding_ > 0 && --awaited_outstanding_ == 0)
      answered_ = true;
  };
  for (std::uint32_t channel = 0; channel < channels; ++channel) {
    dram_[channel]->set_completion(complete);
    mall_[channel]->set_completion(complete);
    // A memory-side cache miss is served by DRAM on the same channel: the
    // channel count is shared between them, which is what makes one accountant
    // array cover both.
    mall_[channel]->set_downstream(
        [this, channel](const MemoryRequest &request) { dram_[channel]->deliver(request); });
    fabric_channels_[channel]->set_completion(complete);
    fabric_channels_[channel]->set_downstream(
        [this, channel](const MemoryRequest &request) { mall_[channel]->deliver(request); });
  }
  for (std::uint32_t die = 0; die < dies; ++die) {
    l2_[die]->set_completion(complete);
    // Fine-grained interleave: consecutive fabric requests land on consecutive
    // channels, which is what spreads a linear stream over the whole interface
    // instead of parking it on one.
    l2_[die]->set_downstream(
        [this](const MemoryRequest &request) { forward_to_channels(request); });
  }
  const std::uint64_t per_die = std::max<std::uint64_t>(1, (units + dies - 1) / dies);
  for (std::uint32_t unit = 0; unit < units; ++unit) {
    const std::size_t die = std::min<std::size_t>(dies - 1, unit / per_die);
    for (CacheDes *first :
         {l1_vector_[unit].get(), l1_scalar_[unit].get(), l1_instruction_[unit].get()}) {
      first->set_completion(complete);
      first->set_downstream(
          [this, die](const MemoryRequest &request) { l2_[die]->deliver(request); });
    }
  }
}

TimingPlane::~TimingPlane() {
  if (trace_ != nullptr)
    std::fclose(trace_);
}

void TimingPlane::trace_dispatch(const DispatchDes &dispatch) {
  if (trace_ == nullptr)
    return;
  const DispatchDes::Terms &terms = dispatch.terms();
  const DispatchShape &shape = dispatch.shape();
  const BandwidthLedger &ledger = dispatch.ledger();
  // Written by hand rather than through a JSON library: this file is in the
  // dynamic-binary-translation tool's link, where pulling in a serializer for
  // a diagnostic would cost every functional run the dependency.
  std::string line = "{";
  const auto number = [&line](const char *key, unsigned long long value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "\"%s\":%llu,", key, value);
    line += buffer;
  };
  number("pid", static_cast<unsigned long long>(::getpid()));
  number("device_cycles", static_cast<unsigned long long>(current_cycles()));
  number("cycles", static_cast<unsigned long long>(dispatch.cycles()));
  number("fixed", static_cast<unsigned long long>(terms.fixed));
  number("issue", static_cast<unsigned long long>(terms.issue));
  number("bandwidth", static_cast<unsigned long long>(terms.bandwidth));
  number("placement", static_cast<unsigned long long>(terms.placement));
  number("latency", static_cast<unsigned long long>(terms.latency));
  number("filling", static_cast<unsigned long long>(terms.filling));
  number("critical_path", static_cast<unsigned long long>(terms.critical_path));
  number("worst_stall", static_cast<unsigned long long>(terms.worst_stall));
  number("fill", static_cast<unsigned long long>(terms.fill));
  number("straggler", static_cast<unsigned long long>(terms.straggler));
  number("rounds", static_cast<unsigned long long>(terms.rounds));
  number("resident", static_cast<unsigned long long>(terms.resident));
  number("unit_issue", static_cast<unsigned long long>(terms.unit_issue));
  number("unit_critical", static_cast<unsigned long long>(terms.unit_critical));
  number("unit_worst_stall", static_cast<unsigned long long>(terms.unit_worst_stall));
  number("unit_waves", static_cast<unsigned long long>(terms.unit_waves));
  number("latency_issue", static_cast<unsigned long long>(terms.latency_issue));
  number("latency_critical", static_cast<unsigned long long>(terms.latency_critical));
  number("latency_worst_stall", static_cast<unsigned long long>(terms.latency_worst_stall));
  number("latency_waves", static_cast<unsigned long long>(terms.latency_waves));
  number("l1_cycles", static_cast<unsigned long long>(terms.l1_cycles));
  number("l2_cycles", static_cast<unsigned long long>(terms.l2_cycles));
  number("fabric_cycles", static_cast<unsigned long long>(terms.fabric_cycles));
  number("mall_cycles", static_cast<unsigned long long>(terms.mall_cycles));
  number("dram_cycles", static_cast<unsigned long long>(terms.dram_cycles));
  number("waves", static_cast<unsigned long long>(dispatch.waves()));
  number("instructions", static_cast<unsigned long long>(dispatch.instructions()));
  number("workgroups", shape.workgroup_count);
  number("waves_per_workgroup", shape.waves_per_workgroup);
  number("vgprs", shape.vector_registers_per_wave);
  number("sgprs", shape.scalar_registers_per_wave);
  number("lds_bytes", shape.lds_bytes_per_workgroup);
  number("l1_bytes", static_cast<unsigned long long>(ledger.l1_bytes()));
  number("l2_bytes", static_cast<unsigned long long>(ledger.l2_bytes()));
  number("fabric_bytes", static_cast<unsigned long long>(ledger.fabric_bytes()));
  number("mall_bytes", static_cast<unsigned long long>(ledger.mall_bytes()));
  number("dram_bytes", static_cast<unsigned long long>(ledger.dram_bytes()));
  line += "\"unit_cycles\":{";
  for (std::size_t unit = 0; unit < kNumFunctionalUnits; ++unit) {
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer), "%s\"%s\":%llu", unit == 0 ? "" : ",",
                  functional_unit_name(static_cast<FunctionalUnit>(unit)),
                  static_cast<unsigned long long>(terms.unit_cycles[unit]));
    line += buffer;
  }
  line += "},\"latency_unit_cycles\":{";
  for (std::size_t unit = 0; unit < kNumFunctionalUnits; ++unit) {
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer), "%s\"%s\":%llu", unit == 0 ? "" : ",",
                  functional_unit_name(static_cast<FunctionalUnit>(unit)),
                  static_cast<unsigned long long>(terms.latency_unit_cycles[unit]));
    line += buffer;
  }
  line += "},\"class_counts\":{";
  for (std::size_t cls = 0; cls < kNumInstClasses; ++cls) {
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer), "%s\"%s\":%llu", cls == 0 ? "" : ",",
                  inst_class_name(static_cast<InstClass>(cls)),
                  static_cast<unsigned long long>(terms.class_counts[cls]));
    line += buffer;
  }
  line += "},\"bound\":\"";
  line += dispatch.bound_by();
  line += "\",\"kernel\":\"";
  for (char character : shape.kernel_name) {
    if (character == '"' || character == '\\')
      line += '\\';
    line += character;
  }
  line += "\"}\n";
  std::fwrite(line.data(), 1, line.size(), trace_);
}

ComputeUnitDes &TimingPlane::compute_unit(std::uint32_t index) {
  return *compute_units_[index % compute_units_.size()];
}

std::uint64_t TimingPlane::issue_memory(const MemoryRequest &request,
                                        const std::vector<std::uint64_t> &lines) {
  if (compute_units_.empty() || lines.empty())
    return request.issued_tick;

  const std::size_t unit = request.origin.compute_unit % compute_units_.size();
  MemoryRequest routed = request;
  routed.line_base = static_cast<std::uint32_t>(line_pool_.size());
  routed.line_count = static_cast<std::uint32_t>(lines.size());
  line_pool_.insert(line_pool_.end(), lines.begin(), lines.end());

  // A wavefront's clock starts at zero when the wavefront starts, so a request
  // arrives on the wavefront's own axis. Offsetting by the tick this compute
  // unit's dispatch began puts every wavefront of a dispatch on one axis, which
  // is what lets the components represent contention between them at all. Two
  // dispatches running at once share the engine's axis and therefore contend,
  // which is the intent.
  const std::uint64_t base = cu_base_[unit];
  routed.issued_tick = base + request.issued_tick;
  routed.origin.sequence = ++sequence_;

  awaited_ = routed.origin.sequence;
  awaited_outstanding_ = 1;
  answered_ = false;
  completion_tick_ = 0;

  // What the wavefront needs back is how much *longer* this access takes, and
  // that has to be measured from the moment the hierarchy actually begins
  // serving it. A wavefront's clock is its own and starts at zero, while the
  // engine's clock only moves forward across every wavefront and dispatch that
  // has already run, so by the time the second access of a run arrives its
  // wavefront-relative issue tick is far behind the engine. Measuring from the
  // issue tick then reports the whole distance between the two clocks as this
  // one access's latency: a sixteen-workgroup vector add came out at 409
  // microseconds, growing with everything that had run before it.
  const std::uint64_t served_from = std::max(routed.issued_tick, engine_.now());
  // Re-stamp the request onto the engine's axis before handing it over. Each
  // level completes no earlier than issued_tick plus its hit latency, and a
  // wavefront-relative stamp puts that floor in the past on every access after
  // the first, so the unloaded latency of every level silently stopped
  // applying and only the bandwidth term survived. Latency-bound kernels --
  // which is most of a realistic corpus -- then read uniformly fast.
  routed.issued_tick = served_from;

  CacheDes *entry = request.instruction_fetch              ? l1_instruction_[unit].get()
                    : request.space == MemorySpace::Scalar ? l1_scalar_[unit].get()
                                                           : l1_vector_[unit].get();
  entry->set_allocates(!request.non_temporal);
  entry->deliver(routed);
  engine_.run_until(answered_);
  entry->set_allocates(true);

  // Trim the pool back when nothing is in flight, so a long run does not grow
  // it without bound. While requests are outstanding the indices must stand.
  if (line_pool_.size() > (1u << 19) && answered_)
    line_pool_.clear();

  if (!answered_ || completion_tick_ <= served_from)
    return request.issued_tick;
  return request.issued_tick + (completion_tick_ - served_from);
}

void TimingPlane::dispatch_begin(const DispatchShape &shape) {
  if (!tuning_.enabled)
    return;
  std::vector<ComputeUnitDes *> units;
  units.reserve(compute_units_.size());
  for (const auto &unit : compute_units_)
    units.push_back(unit.get());
  auto dispatch =
      std::make_unique<DispatchDes>("dispatch", shader_, engine_, tuning_, std::move(units));
  dispatch->begin(shape);
  const std::uint64_t key = packed_key(shape.dispatch_id, shape.queue_id);
  std::unique_ptr<DispatchDes> displaced;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = open_.find(key);
    if (found != open_.end())
      displaced = std::move(found->second);
    open_[key] = std::move(dispatch);
  }
  // A dispatch id arriving twice means the first one is over: ids are allocated
  // per command processor and are reused. Closing it here rather than dropping
  // it keeps its work attributed, and keeps the clock advancing by it.
  if (displaced) {
    charge_bandwidth(*displaced, key);
    displaced->end();
    publish(cycles_.load(std::memory_order_relaxed) + displaced->cycles());
  }
}

void TimingPlane::wave_begin(std::uint32_t compute_unit, std::uint32_t slot,
                             std::uint32_t dispatch_id, std::uint32_t queue_id) {
  if (!tuning_.enabled || compute_units_.empty())
    return;
  const std::size_t unit = compute_unit % compute_units_.size();
  const std::uint64_t key = packed_key(dispatch_id, queue_id);
  DispatchDes *dispatch = nullptr;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = open_.find(key);
    if (found != open_.end())
      dispatch = found->second.get();
  }
  if (dispatch != nullptr && dispatch->claim_acquire()) {
    // The launch acquire, applied at the moment the dispatch first executes.
    dispatch->clear_units();
    for (auto &cache : l1_vector_)
      cache->invalidate();
    for (auto &cache : l1_scalar_)
      cache->invalidate();
    for (auto &cache : l1_instruction_)
      cache->invalidate();
    for (auto &cache : l2_)
      cache->invalidate();
    snapshot_bandwidth(key);
  }
  if (cu_dispatch_[unit] != key) {
    cu_dispatch_[unit] = key;
    cu_base_[unit] = engine_.now();
  }
  compute_units_[unit]->wave_begin(slot);
}

void TimingPlane::instruction(std::uint32_t compute_unit, std::uint32_t slot,
                              const RetiredInstruction &retired) {
  if (!tuning_.enabled || compute_units_.empty())
    return;
  compute_units_[compute_unit % compute_units_.size()]->instruction(slot, retired);
}

void TimingPlane::wave_end(std::uint32_t compute_unit, std::uint32_t slot) {
  if (!tuning_.enabled || compute_units_.empty())
    return;
  compute_units_[compute_unit % compute_units_.size()]->wave_end(slot);
}

void TimingPlane::forward_to_channels(const MemoryRequest &request) {
  const std::uint64_t channels = std::max<std::uint64_t>(1, fabric_channels_.size());
  const std::uint64_t stride = std::max<std::uint64_t>(1, tuning_.fabric_request_bytes);
  if (request.line_count == 0 || request.line_base + request.line_count > line_pool_.size()) {
    fabric_channels_[0]->deliver(request);
    return;
  }

  // Group the request's lines by the channel their addresses select and send
  // one sub-request per channel. Routing the whole request by its first line
  // instead concentrates it: with sixty-four byte interleave, the lines of one
  // contiguous multi-line access step the channel index by two apiece, so a
  // kilobyte access advances it by sixteen and a hundred and twenty-eight
  // channel part is driven as though it had eight. That read as a sixteenfold
  // bandwidth shortfall on a streaming copy -- 171 microseconds against a
  // measured 11 -- and it looked like a bandwidth figure being wrong rather
  // than an address being routed wrong, which is what made it worth finding
  // once and writing down.
  // Beyond the second level the unit of traffic is the fabric request, which is
  // smaller than a cache line, so one line is several requests and they land on
  // consecutive channels. Grouping whole lines instead leaves every second
  // channel idle on a part whose interleave is half its line size, which is a
  // further factor of two on top of the concentration above: the same streaming
  // copy read 22 microseconds against a measured 11 until the split moved down
  // to this granularity.
  const std::uint32_t count = request.line_count;
  const std::uint64_t per_line = std::max<std::uint64_t>(1, request.line_bytes / stride);
  request_scratch_.clear();
  for (std::uint32_t index = 0; index < count; ++index) {
    const std::uint64_t line = line_pool_[request.line_base + index];
    for (std::uint64_t part = 0; part < per_line; ++part)
      request_scratch_.push_back(line + part * stride);
  }

  channel_scratch_.assign(static_cast<std::size_t>(channels), 0);
  for (std::uint64_t address : request_scratch_)
    ++channel_scratch_[static_cast<std::size_t>((address / stride) % channels)];

  std::uint32_t distinct = 0;
  for (std::uint64_t occupancy : channel_scratch_)
    if (occupancy != 0)
      ++distinct;
  if (distinct == 0) {
    fabric_channels_[0]->deliver(request);
    return;
  }

  // Every part but the first is an extra completion the waiting access must
  // see before it is answered.
  awaited_outstanding_ += distinct - 1;
  for (std::size_t channel = 0; channel < channel_scratch_.size(); ++channel) {
    if (channel_scratch_[channel] == 0)
      continue;
    MemoryRequest part = request;
    part.line_bytes = static_cast<std::uint32_t>(stride);
    part.line_base = static_cast<std::uint32_t>(line_pool_.size());
    part.line_count = 0;
    part.row_misses = 0;
    row_scratch_.clear();
    const std::uint64_t row_bytes = std::max<std::uint64_t>(1, tuning_.dram_row_bytes);
    for (std::uint64_t address : request_scratch_) {
      if ((address / stride) % channels != channel)
        continue;
      // Activations are counted as the number of DISTINCT rows this one
      // instruction's traffic touches on this channel, not as transitions
      // against a persistently open row. Two reasons, and the second is the one
      // that matters.
      //
      // A persistent open row is not observable here: 256 compute units
      // interleave onto 128 channels in whatever order the emulator happened to
      // run them, so the row a channel "last opened" is unrelated to the row
      // the next request wants. Charged that way, every request is an
      // activation for every kernel, which is a uniform slowdown rather than a
      // discriminator -- it made the streaming case worse and left the
      // scattered one where it was.
      //
      // And a real controller reorders. Requests from one instruction arrive
      // together and are scheduled row-first, so what that instruction costs is
      // the number of rows it has to open, not the order they arrived in. A
      // wave64 access that walks one row opens one; one whose lanes are
      // scattered opens as many rows as it has lanes. That is precisely the
      // difference between the streaming kernels the model already fits and the
      // strided and reduction kernels it does not.
      //
      // The row index is taken in the channel's own address space: with
      // fine-grained interleave, a row of row_bytes on one channel covers
      // row_bytes times the channel count of device address space.
      const std::uint64_t row = address / (row_bytes * channels);
      if (row_scratch_.insert(row))
        ++part.row_misses;
      line_pool_.push_back(address);
      ++part.line_count;
    }
    fabric_channels_[channel]->deliver(part);
  }
}

void TimingPlane::snapshot_bandwidth(std::uint64_t key) {
  // One snapshot per dispatch, not one shared by all of them. A single shared
  // baseline is destroyed by the next dispatch to start: kernels launched back
  // to back on one stream overlap at this seam, so the second one's snapshot
  // becomes the first one's baseline and the first one's traffic is measured
  // from most of the way through itself. It reads as a kernel that moved almost
  // no bytes, which is indistinguishable from a kernel that was not bandwidth
  // bound.
  std::vector<std::uint64_t> &base = bandwidth_base_[key];
  base.clear();
  for (const auto &cache : l1_vector_)
    base.push_back(cache->bytes());
  for (const auto &cache : l2_)
    base.push_back(cache->bytes());
  for (const auto &channel : fabric_channels_)
    base.push_back(channel->bytes());
  for (const auto &cache : mall_)
    base.push_back(cache->bytes());
  for (const auto &channel : dram_)
    base.push_back(channel->bytes());
  std::vector<std::uint64_t> &activations = activation_base_[key];
  activations.clear();
  for (const auto &channel : dram_)
    activations.push_back(channel->activations());
}

void TimingPlane::charge_bandwidth(DispatchDes &dispatch, std::uint64_t key) {
  const auto found = bandwidth_base_.find(key);
  if (found == bandwidth_base_.end())
    return;
  const std::vector<std::uint64_t> &base = found->second;
  // Per dispatch, taken as the difference in what each component moved. Each
  // component is one instance of its level -- one compute unit's cache, one
  // channel -- so the difference is already per instance, which is what the
  // ledger needs to say that a kernel landing all its traffic on one channel
  // does not get the whole part's bandwidth.
  if (base.size() !=
      l1_vector_.size() + l2_.size() + fabric_channels_.size() + mall_.size() + dram_.size())
    return;
  std::size_t index = 0;
  const auto delta = [&base, &index](std::uint64_t now) {
    const std::uint64_t before = base[index++];
    return now > before ? now - before : 0;
  };
  for (std::size_t unit = 0; unit < l1_vector_.size(); ++unit)
    dispatch.ledger().charge_l1(unit, delta(l1_vector_[unit]->bytes()));
  for (std::size_t die = 0; die < l2_.size(); ++die)
    dispatch.ledger().charge_l2(die, delta(l2_[die]->bytes()));
  for (std::size_t channel = 0; channel < fabric_channels_.size(); ++channel)
    dispatch.ledger().charge_fabric(channel, delta(fabric_channels_[channel]->bytes()));
  for (std::size_t channel = 0; channel < mall_.size(); ++channel)
    dispatch.ledger().charge_mall(channel, delta(mall_[channel]->bytes()));
  for (std::size_t channel = 0; channel < dram_.size(); ++channel)
    dispatch.ledger().charge_dram(channel, delta(dram_[channel]->bytes()));
  // Row activations, taken the same way: the difference in what each channel
  // performed while this dispatch ran.
  if (activation_base_.count(key) != 0) {
    const std::vector<std::uint64_t> &before = activation_base_.at(key);
    if (before.size() == dram_.size()) {
      for (std::size_t channel = 0; channel < dram_.size(); ++channel) {
        const std::uint64_t now = dram_[channel]->activations();
        const std::uint64_t performed = now > before[channel] ? now - before[channel] : 0;
        dispatch.ledger().charge_dram_cycles(channel, performed * tuning_.dram_row_miss_cycles);
      }
    }
  }
}

void TimingPlane::dispatch_end(std::uint32_t dispatch_id, std::uint32_t queue_id) {
  if (!tuning_.enabled)
    return;
  std::unique_ptr<DispatchDes> dispatch;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = open_.find(packed_key(dispatch_id, queue_id));
    if (found == open_.end())
      return;
    dispatch = std::move(found->second);
    open_.erase(found);
  }
  const std::uint64_t key = packed_key(dispatch_id, queue_id);
  charge_bandwidth(*dispatch, key);
  dispatch->end();
  trace_dispatch(*dispatch);
  {
    std::lock_guard<std::mutex> guard(mutex_);
    bandwidth_base_.erase(key);
    activation_base_.erase(key);
  }

  char line[512];
  std::snprintf(line, sizeof(line),
                "  %llu cycles (%.3f us) %-9s wgs=%u waves=%llu insts=%llu fabric=%lluKiB  ",
                static_cast<unsigned long long>(dispatch->cycles()),
                static_cast<double>(dispatch->cycles()) / (tuning_.clock_mhz / 1000.0) / 1000.0,
                dispatch->bound_by(), dispatch->shape().workgroup_count,
                static_cast<unsigned long long>(dispatch->waves()),
                static_cast<unsigned long long>(dispatch->instructions()),
                static_cast<unsigned long long>(dispatch->ledger().fabric_bytes() / 1024));
  {
    std::lock_guard<std::mutex> guard(mutex_);
    completed_.push_back(std::string(line) + dispatch->shape().kernel_name);
  }
  // The device clock advances by what the dispatch cost, and does not advance
  // between dispatches: simulated time is produced by modelled work, and an
  // idle device produces none. That is what keeps a host-side event bracket
  // measuring the kernels inside it rather than the host's enqueue latency.
  publish(cycles_.load(std::memory_order_relaxed) + dispatch->cycles());
}

void TimingPlane::acquire_once(std::uint32_t dispatch_id, std::uint32_t queue_id) {
  (void)dispatch_id;
  (void)queue_id;
}

void TimingPlane::publish(std::uint64_t cycles) {
  std::uint64_t seen = cycles_.load(std::memory_order_relaxed);
  while (cycles > seen && !cycles_.compare_exchange_weak(seen, cycles, std::memory_order_relaxed))
    ;
}

void TimingPlane::write_report(std::string &out) const {
  std::lock_guard<std::mutex> guard(mutex_);
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer), "timing plane: %zu dispatches, %llu events\n",
                completed_.size(), static_cast<unsigned long long>(engine_.events()));
  out += buffer;
  for (const std::string &entry : completed_)
    out += entry + "\n";
  std::uint64_t l1_hits = 0, l1_misses = 0, l2_hits = 0, l2_misses = 0, dram_bytes = 0,
                fabric_bytes = 0;
  for (const auto &cache : l1_vector_) {
    l1_hits += cache->hits();
    l1_misses += cache->misses();
  }
  for (const auto &cache : l2_) {
    l2_hits += cache->hits();
    l2_misses += cache->misses();
  }
  for (const auto &channel : dram_)
    dram_bytes += channel->bytes();
  for (const auto &channel : fabric_channels_)
    fabric_bytes += channel->bytes();
  // The spread across channels, not just the total. A kernel whose traffic
  // lands on a fraction of the channels is bandwidth limited at that fraction
  // of the part's rate, and the total alone cannot say so: 67 MB spread evenly
  // and 67 MB on eight channels are the same number and a sixteenfold
  // difference in duration.
  std::uint64_t busiest = 0;
  std::uint64_t used = 0;
  for (const auto &channel : fabric_channels_) {
    busiest = std::max(busiest, channel->bytes());
    if (channel->bytes() != 0)
      ++used;
  }
  const double mean = fabric_channels_.empty() ? 0.0
                                               : static_cast<double>(fabric_bytes) /
                                                     static_cast<double>(fabric_channels_.size());
  std::snprintf(buffer, sizeof(buffer),
                "  channels used %llu of %zu, busiest %llu bytes, mean %.0f (%.2fx)\n",
                static_cast<unsigned long long>(used), fabric_channels_.size(),
                static_cast<unsigned long long>(busiest), mean,
                mean > 0.0 ? static_cast<double>(busiest) / mean : 0.0);
  out += buffer;

  const auto rate = [](std::uint64_t hits, std::uint64_t misses) {
    const std::uint64_t total = hits + misses;
    return total ? 100.0 * static_cast<double>(hits) / static_cast<double>(total) : 0.0;
  };
  std::snprintf(buffer, sizeof(buffer),
                "memory: first level hit %.1f%% of %llu, second level hit %.1f%% of %llu\n"
                "  fabric %llu bytes, dram %llu bytes\n",
                rate(l1_hits, l1_misses), static_cast<unsigned long long>(l1_hits + l1_misses),
                rate(l2_hits, l2_misses), static_cast<unsigned long long>(l2_hits + l2_misses),
                static_cast<unsigned long long>(fabric_bytes),
                static_cast<unsigned long long>(dram_bytes));
  out += buffer;
  tuning_.write_report(out);
}

} // namespace rocjitsu::timing
