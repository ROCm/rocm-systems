/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "amdsmi_wrap.h"

#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace RcclUnitTesting
{

// ---------------------------------------------------------------------------
// Fabric telemetry sample reduction
//
// RCCL_FABRIC_TELEMETRY_ENABLE samples every amd_smi telemetry category on a
// period and logs what moved. The counters are cumulative since firmware boot
// and the firmware republishes whole datasets under a generation count, so the
// reportable quantity is a difference between samples -- which means the
// reduction carries the state that can be wrong: a stale baseline, a misaligned
// flat index, or a wrapped delta after a firmware restart.
//
// These tests drive amdSmiFabricTelemetryDiff() and its helpers with synthetic
// samples. They need no fabric, no amd_smi and no GPU, so they run on any host
// and in both Release and Debug.
// ---------------------------------------------------------------------------

namespace
{

// Owns every array a synthetic amdsmi_fabric_telemetry_t points at. The sample
// structs hold raw pointers, so the backing storage has to outlive the diff call
// and must not move while categories are still being added.
class FakeSample
{
public:
    // Adds one category. Each entry of `instanceValues` becomes an instance, and
    // its contents become that instance's counters. Counter IDs restart at 1 in
    // every instance, as the firmware's do: instances of the same kind report the
    // same set of counters, so an ID is only unique within one. FakeTelemName()
    // names them.
    void AddCategory(unsigned category, uint64_t generation,
                     const std::vector<std::vector<uint64_t>>& instanceValues,
                     const char* labelPrefix = "inst")
    {
        auto instances = std::make_unique<std::vector<amdsmi_fabric_telemetry_instance_t>>();

        for(size_t i = 0; i < instanceValues.size(); i++)
        {
            auto     items  = std::make_unique<std::vector<amdsmi_fabric_telemetry_item_t>>();
            uint64_t nextId = 1;
            for(uint64_t value : instanceValues[i])
                items->push_back({nextId++, value});

            amdsmi_fabric_telemetry_instance_t inst{};
            snprintf(inst.name.text, sizeof(inst.name.text), "%s%zu", labelPrefix, i);
            inst.logical_idx = static_cast<unsigned>(i);
            inst.item_count  = static_cast<unsigned>(items->size());
            inst.items       = items->empty() ? nullptr : items->data();
            instances->push_back(inst);

            itemArrays_.push_back(std::move(items));
        }

        auto dataset = std::make_unique<amdsmi_fabric_telemetry_dataset_t>();
        dataset->category         = static_cast<amdsmi_fabric_telemetry_category_t>(category);
        dataset->generation_count = generation;
        dataset->instance_count   = static_cast<unsigned>(instances->size());
        dataset->instances        = instances->data();

        telemetry_.datasets[category] = dataset.get();
        datasets_.push_back(std::move(dataset));
        instanceArrays_.push_back(std::move(instances));
    }

    // Replaces an instance's item array with a null pointer, as amd_smi does for a
    // category it enumerated but could not populate.
    void ClearInstanceItems(unsigned category, size_t instance)
    {
        amdsmi_fabric_telemetry_instance_t* inst = &telemetry_.datasets[category]->instances[instance];
        inst->items                              = nullptr;
    }

    // Drops the instance array while leaving instance_count set, the shape amd_smi
    // leaves behind for a category it counted but could not populate.
    void ClearInstances(unsigned category)
    {
        telemetry_.datasets[category]->instances = nullptr;
    }

    // Blanks an instance label, as amd_smi leaves it when the firmware supplies no
    // name for the instance.
    void ClearLabel(unsigned category, size_t instance)
    {
        amdsmi_fabric_telemetry_instance_t* inst = &telemetry_.datasets[category]->instances[instance];
        memset(inst->name.text, 0, sizeof(inst->name.text));
    }

    void SetLogicalIdx(unsigned category, size_t instance, unsigned logicalIdx)
    {
        telemetry_.datasets[category]->instances[instance].logical_idx = logicalIdx;
    }

    void SetLabel(unsigned category, size_t instance, const char* text, size_t length)
    {
        memcpy(telemetry_.datasets[category]->instances[instance].name.text, text, length);
    }

    void SetValue(unsigned category, size_t instance, size_t item, uint64_t value)
    {
        telemetry_.datasets[category]->instances[instance].items[item].value = value;
    }

    uint64_t ItemId(unsigned category, size_t instance, size_t item) const
    {
        return telemetry_.datasets[category]->instances[instance].items[item].id;
    }

    void SetGeneration(unsigned category, uint64_t generation)
    {
        telemetry_.datasets[category]->generation_count = generation;
    }

    amdsmi_fabric_telemetry_t* Get() { return &telemetry_; }

private:
    amdsmi_fabric_telemetry_t                                                  telemetry_{};
    std::vector<std::unique_ptr<amdsmi_fabric_telemetry_dataset_t>>            datasets_;
    std::vector<std::unique_ptr<std::vector<amdsmi_fabric_telemetry_instance_t>>> instanceArrays_;
    std::vector<std::unique_ptr<std::vector<amdsmi_fabric_telemetry_item_t>>>  itemArrays_;
};

// Deterministic counter names, so a test can assert which counter was reported
// without a loaded amd_smi.
const char* FakeTelemName(uint64_t telemId)
{
    switch(telemId)
    {
    case 1: return "COUNTER_ONE";
    case 2: return "COUNTER_TWO";
    case 3: return "COUNTER_THREE";
    case 4: return "COUNTER_FOUR";
    case 5: return "COUNTER_FIVE";
    default: return "COUNTER_OTHER";
    }
}

using Reports = amdsmiFabricTelemetryCategoryReport[AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX];

} // namespace

// ---------------------------------------------------------------------------
// Category coverage
//
// The feature is specified to sample every category, so the mask it passes to
// amdsmi_alloc_fabric_telemetry has to select all of them. It is derived from
// the enum rather than OR-ed by name, which is only correct while the categories
// stay a dense range -- that is what this pins.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricTelemetryCategories, AllCategoriesMaskSelectsEveryCategory)
{
    for(unsigned cat = 0; cat < AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX; cat++)
        EXPECT_NE(kAmdSmiFabricTelemetryAllCategories & (1u << cat), 0u) << "category " << cat << " not selected";

    // Nothing above the defined range, so the mask cannot ask for a category the
    // library does not know about.
    EXPECT_EQ(kAmdSmiFabricTelemetryAllCategories >> AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX, 0u);
}

TEST(AmdSmiFabricTelemetryCategories, EveryCategoryHasADistinctName)
{
    std::set<std::string> names;
    for(unsigned cat = 0; cat < AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX; cat++)
    {
        const std::string name = amdSmiFabricTelemetryCategoryName(cat);
        EXPECT_NE(name, "Unknown") << "category " << cat << " has no name";
        names.insert(name);
    }
    EXPECT_EQ(names.size(), static_cast<size_t>(AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX));
}

TEST(AmdSmiFabricTelemetryCategories, UnrecognizedCategoryIsNamedUnknown)
{
    EXPECT_STREQ(amdSmiFabricTelemetryCategoryName(AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX), "Unknown");
    EXPECT_STREQ(amdSmiFabricTelemetryCategoryName(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UNKNOWN), "Unknown");
}

// ---------------------------------------------------------------------------
// Sampling period
//
// A period of zero would spin the sampler thread, and one sample reads every
// counter of every device over a single character device, so a too-small period
// lets the diagnostic disturb the fabric it is measuring.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricTelemetryInterval, NonPositivePeriodDisablesSampling)
{
    EXPECT_EQ(amdSmiFabricTelemetryResolveIntervalMs(0), 0);
    EXPECT_EQ(amdSmiFabricTelemetryResolveIntervalMs(-1), 0);
    EXPECT_EQ(amdSmiFabricTelemetryResolveIntervalMs(-5000), 0);
}

TEST(AmdSmiFabricTelemetryInterval, PeriodBelowFloorIsClampedToFloor)
{
    EXPECT_EQ(kAmdSmiFabricTelemetryMinIntervalMs, 100) << "env-variables.rst documents a 100ms floor";
    EXPECT_EQ(amdSmiFabricTelemetryResolveIntervalMs(1), kAmdSmiFabricTelemetryMinIntervalMs);
    EXPECT_EQ(amdSmiFabricTelemetryResolveIntervalMs(kAmdSmiFabricTelemetryMinIntervalMs - 1),
              kAmdSmiFabricTelemetryMinIntervalMs);
}

TEST(AmdSmiFabricTelemetryInterval, PeriodAtOrAboveFloorIsKept)
{
    EXPECT_EQ(amdSmiFabricTelemetryResolveIntervalMs(kAmdSmiFabricTelemetryMinIntervalMs),
              kAmdSmiFabricTelemetryMinIntervalMs);
    EXPECT_EQ(amdSmiFabricTelemetryResolveIntervalMs(1000), 1000);
}

// ---------------------------------------------------------------------------
// Baseline and deltas
// ---------------------------------------------------------------------------

// Nothing can be reported from a single sample of a cumulative counter, so the
// first one only records values. Reporting it would present a counter's
// since-boot total as if it were traffic from one interval.
TEST(AmdSmiFabricTelemetryDiff, FirstSampleOnlyEstablishesBaseline)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{10, 20, 30}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;

    EXPECT_FALSE(baseline.established);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    EXPECT_EQ(reports[0].changedCount, 0);
    EXPECT_EQ(reports[0].moverCount, 0);
    EXPECT_EQ(reports[0].itemCount, 3);
    EXPECT_FALSE(reports[0].stale);
    EXPECT_TRUE(baseline.established);
    EXPECT_EQ(baseline.values.size(), 3u);
}

TEST(AmdSmiFabricTelemetryDiff, SecondSampleReportsOnlyCountersThatMoved)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 100, {{10, 20, 30}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    // Counter 0 gains 5, counter 2 gains 70, counter 1 stands still.
    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, 0, 15);
    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, 2, 100);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 101);

    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);
    EXPECT_EQ(reports[0].changedCount, 2);
    EXPECT_EQ(reports[0].itemCount, 3);
    EXPECT_EQ(reports[0].generation, 101u);
    EXPECT_FALSE(reports[0].stale);

    // Largest mover first, and it is the counter that actually gained the most.
    ASSERT_EQ(reports[0].moverCount, 2);
    EXPECT_EQ(reports[0].movers[0].delta, 70u);
    EXPECT_STREQ(reports[0].movers[0].name, "COUNTER_THREE");
    EXPECT_EQ(reports[0].movers[1].delta, 5u);
    EXPECT_STREQ(reports[0].movers[1].name, "COUNTER_ONE");
}

TEST(AmdSmiFabricTelemetryDiff, IdenticalSampleReportsNothing)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_PFC, 100, {{10, 20}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_PFC, 101);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);
    EXPECT_EQ(reports[0].changedCount, 0);
    EXPECT_EQ(reports[0].moverCount, 0);
}

// An unchanged generation count means the firmware has not republished the
// dataset, so any apparent movement is a torn or repeated read rather than
// traffic. Reporting it would invent deltas the fabric never saw.
TEST(AmdSmiFabricTelemetryDiff, StaleGenerationSuppressesDeltas)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_CRYPTO, 100, {{10, 20}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    // Values move but the generation count does not.
    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_CRYPTO, 0, 0, 999);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    EXPECT_TRUE(reports[0].stale);
    EXPECT_EQ(reports[0].changedCount, 0);

    // The baseline still advanced, so once the generation moves the delta is
    // measured from the value actually last seen, not from the pre-stale one.
    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_CRYPTO, 0, 0, 1000);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_CRYPTO, 101);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);
    EXPECT_FALSE(reports[0].stale);
    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_EQ(reports[0].movers[0].delta, 1u);
}

// How the closing report at communicator destroy is built: a second reference
// point is taken from the same first sample and then left alone while the periodic
// ticks advance their own. Diffing it once at the end covers the whole span, which
// is what makes a run shorter than the firmware's publication interval still
// report something.
TEST(AmdSmiFabricTelemetryDiff, SeparateBaselineMeasuresTheWholeSpan)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{10}});

    amdsmiFabricTelemetryBaseline tick{};
    amdsmiFabricTelemetryBaseline span{};
    Reports                       reports;

    // Both reference points are established from the same first sample.
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &tick, FakeTelemName, reports), 1);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &span, FakeTelemName, reports), 1);
    EXPECT_EQ(reports[0].changedCount, 0);

    // Two periodic ticks, which only advance their own reference point.
    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0, 0, 15);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 101);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &tick, FakeTelemName, reports), 1);
    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_EQ(reports[0].movers[0].delta, 5u);

    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0, 0, 23);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 102);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &tick, FakeTelemName, reports), 1);
    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_EQ(reports[0].movers[0].delta, 8u);

    // The untouched reference point reports both ticks together.
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &span, FakeTelemName, reports), 1);
    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_EQ(reports[0].movers[0].delta, 13u);
}

// The case the closing report exists for: the generation advances once, after the
// only periodic tick has already run. The tick sees a stale dataset and stays
// quiet, but the span still has something to say.
TEST(AmdSmiFabricTelemetryDiff, SpanReportsWhenEveryTickFoundTheDatasetStale)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 100, {{700}});

    amdsmiFabricTelemetryBaseline tick{};
    amdsmiFabricTelemetryBaseline span{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &tick, FakeTelemName, reports), 1);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &span, FakeTelemName, reports), 1);

    // The one tick that fits in the run lands before the firmware republishes.
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &tick, FakeTelemName, reports), 1);
    EXPECT_TRUE(reports[0].stale);
    EXPECT_EQ(reports[0].changedCount, 0);

    // Destroy happens just after the republish.
    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, 0, 900);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 101);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &span, FakeTelemName, reports), 1);
    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_EQ(reports[0].movers[0].delta, 200u);
}

// A counter that goes backwards means the firmware restarted. Subtracting would
// underflow to an enormous delta, so the sample becomes the new reference point.
TEST(AmdSmiFabricTelemetryDiff, CounterGoingBackwardsReportsNoWrappedDelta)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{5000}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0, 0, 7);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 101);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);
    EXPECT_EQ(reports[0].changedCount, 0);
    EXPECT_EQ(reports[0].moverCount, 0);

    // Counting resumes from the post-restart value.
    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0, 0, 9);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 102);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);
    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_EQ(reports[0].movers[0].delta, 2u);
}

// The baseline is a flat array indexed by traversal order, so a category
// appearing or disappearing shifts every index after it. Continuing to diff
// across that shift would compare unrelated counters.
TEST(AmdSmiFabricTelemetryDiff, SampleLayoutChangeInvalidatesBaseline)
{
    FakeSample first;
    first.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{10, 20}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(first.Get(), &baseline, FakeTelemName, reports), 1);
    ASSERT_TRUE(baseline.established);
    ASSERT_EQ(baseline.values.size(), 2u);

    // A second category appears, so the sample is now a different shape.
    FakeSample grown;
    grown.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 101, {{10, 20}});
    grown.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 101, {{1, 2, 3}});

    ASSERT_EQ(amdSmiFabricTelemetryDiff(grown.Get(), &baseline, FakeTelemName, reports), 2);
    EXPECT_FALSE(baseline.established) << "a layout change must drop the stale reference point";
    EXPECT_EQ(baseline.values.size(), 5u);
    EXPECT_EQ(reports[0].changedCount, 0) << "deltas against a shifted baseline must not be reported";
    EXPECT_EQ(reports[1].changedCount, 0);

    // The next sample of the new shape re-establishes, still reporting nothing.
    ASSERT_EQ(amdSmiFabricTelemetryDiff(grown.Get(), &baseline, FakeTelemName, reports), 2);
    EXPECT_TRUE(baseline.established);
    EXPECT_EQ(reports[0].changedCount, 0);
    EXPECT_EQ(reports[1].changedCount, 0);

    // And the one after that diffs normally against it.
    grown.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, 1, 42);
    grown.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 102);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(grown.Get(), &baseline, FakeTelemName, reports), 2);
    EXPECT_EQ(reports[1].changedCount, 1);
    ASSERT_EQ(reports[1].moverCount, 1);
    EXPECT_EQ(reports[1].movers[0].delta, 40u);
}

// The dangerous shape of a layout change: the new category sorts ahead of the
// existing one, so the surviving counters land on flat indices that used to hold
// something else entirely. Diffing those would publish large fabricated deltas.
TEST(AmdSmiFabricTelemetryDiff, CategoryAppearingAheadOfExistingOneReportsNoFabricatedDeltas)
{
    FakeSample first;
    first.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 100, {{10, 20}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(first.Get(), &baseline, FakeTelemName, reports), 1);
    ASSERT_TRUE(baseline.established);

    // UALOE is category 0, so it is walked before NETPORT and takes over flat slots
    // 0 and 1 -- the slots holding NETPORT's 10 and 20.
    FakeSample grown;
    grown.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 101, {{1000, 2000}});
    grown.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 101, {{10, 20}});

    ASSERT_EQ(amdSmiFabricTelemetryDiff(grown.Get(), &baseline, FakeTelemName, reports), 2);
    EXPECT_FALSE(baseline.established);
    EXPECT_EQ(reports[0].changedCount, 0) << "1000-10 and 2000-20 are not real deltas";
    EXPECT_EQ(reports[1].changedCount, 0);

    // Once re-established, the counters diff against their own history again.
    ASSERT_EQ(amdSmiFabricTelemetryDiff(grown.Get(), &baseline, FakeTelemName, reports), 2);
    ASSERT_TRUE(baseline.established);
    grown.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0, 0, 1001);
    grown.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 102);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(grown.Get(), &baseline, FakeTelemName, reports), 2);
    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_EQ(reports[0].movers[0].delta, 1u);
}

// A category that shrinks has to invalidate too: the stored baseline is longer
// than the sample, so the trailing entries describe counters no longer present.
TEST(AmdSmiFabricTelemetryDiff, ShrinkingSampleInvalidatesBaseline)
{
    FakeSample wide;
    wide.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{10, 20, 30, 40}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(wide.Get(), &baseline, FakeTelemName, reports), 1);
    ASSERT_EQ(baseline.values.size(), 4u);

    FakeSample narrow;
    narrow.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 101, {{10, 20}});
    ASSERT_EQ(amdSmiFabricTelemetryDiff(narrow.Get(), &baseline, FakeTelemName, reports), 1);

    EXPECT_FALSE(baseline.established);
    EXPECT_EQ(baseline.values.size(), 2u);
}

// The shape changed but the counter total did not, so a size comparison alone sees
// nothing wrong. One category gave up a counter and another took one on, which slides
// every slot after the shrink down by one and lines each survivor up against its
// neighbour's history.
TEST(AmdSmiFabricTelemetryDiff, ShiftThatKeepsTheTotalCountReportsNoFabricatedDeltas)
{
    FakeSample first;
    first.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{10, 20}});
    first.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 100, {{1, 2}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(first.Get(), &baseline, FakeTelemName, reports), 2);
    ASSERT_TRUE(baseline.established);
    ASSERT_EQ(baseline.values.size(), 4u);

    // Four counters before and four after, but NetPort's now start a slot earlier:
    // its 2 lands where its 1 was and its 3 where its 2 was, each a fabricated +1.
    FakeSample shifted;
    shifted.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 101, {{10}});
    shifted.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 101, {{1, 2, 3}});

    ASSERT_EQ(amdSmiFabricTelemetryDiff(shifted.Get(), &baseline, FakeTelemName, reports), 2);
    EXPECT_EQ(baseline.values.size(), 4u) << "the shift is invisible to a count comparison";
    EXPECT_FALSE(baseline.established);
    EXPECT_EQ(reports[0].changedCount, 0);
    EXPECT_EQ(reports[1].changedCount, 0) << "neighbouring counters must not be diffed against each other";

    // Re-established on the new shape, and diffing its own history again.
    ASSERT_EQ(amdSmiFabricTelemetryDiff(shifted.Get(), &baseline, FakeTelemName, reports), 2);
    ASSERT_TRUE(baseline.established);
    shifted.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, 2, 9);
    shifted.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 102);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(shifted.Get(), &baseline, FakeTelemName, reports), 2);
    ASSERT_EQ(reports[1].moverCount, 1);
    EXPECT_EQ(reports[1].movers[0].delta, 6u);
}

// Counter IDs repeat across instances of the same kind, so a slot's identity has to
// include which instance it came from. Here the sample keeps its counts and its IDs
// and only re-labels an instance, as a device would after re-enumerating a link.
TEST(AmdSmiFabricTelemetryDiff, InstanceBeingRelabelledReportsNoFabricatedDeltas)
{
    FakeSample first;
    first.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 100, {{10}, {20}}, "netport");

    // The premise: the two ports report the same counter, so the label is all that
    // separates them.
    ASSERT_EQ(first.ItemId(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, 0),
              first.ItemId(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 1, 0));

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(first.Get(), &baseline, FakeTelemName, reports), 1);
    ASSERT_TRUE(baseline.established);

    // Same two instances by position and by counter ID, but the first is a different
    // port, so its 30 is not 20 counts on from the 10 recorded for netport0.
    FakeSample relabelled;
    relabelled.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 101, {{30}, {20}}, "netport");
    relabelled.SetLabel(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, "netport7", sizeof("netport7"));

    ASSERT_EQ(amdSmiFabricTelemetryDiff(relabelled.Get(), &baseline, FakeTelemName, reports), 1);
    EXPECT_FALSE(baseline.established);
    EXPECT_EQ(reports[0].changedCount, 0) << "a counter from another instance is not this one's history";
}

// The label is not guaranteed to be populated, and an unset one is the same empty
// string for every instance, so it cannot be the only thing identifying them. Here
// two unnamed instances with the same counter IDs swap positions, which leaves the
// counter total unchanged and so is invisible to a count comparison too.
TEST(AmdSmiFabricTelemetryDiff, UnnamedInstancesSwappingPositionsReportNoFabricatedDeltas)
{
    FakeSample first;
    first.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 100, {{10}, {20}});
    first.ClearLabel(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0);
    first.ClearLabel(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 1);
    first.SetLogicalIdx(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, 6);
    first.SetLogicalIdx(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 1, 7);

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(first.Get(), &baseline, FakeTelemName, reports), 1);
    ASSERT_TRUE(baseline.established);

    // The same two ports, reported in the other order. Port 7's 20 now sits in the
    // slot holding port 6's 10, which would read as +10.
    FakeSample swapped;
    swapped.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 101, {{20}, {10}});
    swapped.ClearLabel(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0);
    swapped.ClearLabel(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 1);
    swapped.SetLogicalIdx(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 0, 7);
    swapped.SetLogicalIdx(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 1, 6);

    ASSERT_EQ(amdSmiFabricTelemetryDiff(swapped.Get(), &baseline, FakeTelemName, reports), 1);
    EXPECT_FALSE(baseline.established);
    EXPECT_EQ(reports[0].changedCount, 0) << "the logical index is all that separates unnamed instances";
}

TEST(AmdSmiFabricTelemetryDiff, OnlyPresentCategoriesAreReported)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_SWITCH, 7, {{1}});
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_DERIVED_NETPORT, 9, {{2}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 2);

    // Reported in category order, each tagged with the category it came from, so
    // the caller's log line cannot attribute counters to the wrong category.
    EXPECT_EQ(reports[0].category, static_cast<unsigned>(AMDSMI_FABRIC_TELEMETRY_CATEGORY_SWITCH));
    EXPECT_EQ(reports[0].generation, 7u);
    EXPECT_EQ(reports[1].category, static_cast<unsigned>(AMDSMI_FABRIC_TELEMETRY_CATEGORY_DERIVED_NETPORT));
    EXPECT_EQ(reports[1].generation, 9u);
}

TEST(AmdSmiFabricTelemetryDiff, CountersAreSummedAcrossEveryInstance)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 100, {{1, 2}, {3, 4}, {5, 6}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    EXPECT_EQ(reports[0].itemCount, 6);
    EXPECT_EQ(baseline.values.size(), 6u);

    // A mover is attributed to the instance that owns it, not the first one.
    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 2, 1, 106);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 101);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);
    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_STREQ(reports[0].movers[0].instance, "inst2");
    EXPECT_EQ(reports[0].movers[0].delta, 100u);
}

// amd_smi enumerates instances it cannot populate with a null item array. A walk
// that dereferenced it would fault inside a sampler thread.
TEST(AmdSmiFabricTelemetryDiff, InstanceWithNoItemsIsSkipped)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{1, 2}, {3, 4}});
    sample.ClearInstanceItems(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0);

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    // Only the populated instance contributes.
    EXPECT_EQ(reports[0].itemCount, 2);
    EXPECT_EQ(baseline.values.size(), 2u);
}

// A non-zero instance_count with no instance array would otherwise be walked as if
// the pointer were valid.
TEST(AmdSmiFabricTelemetryDiff, CategoryWithNoInstanceArrayIsReportedEmpty)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{1, 2}});
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT, 100, {{3, 4}});
    sample.ClearInstances(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE);

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 2);

    EXPECT_EQ(reports[0].itemCount, 0) << "the category is still reported, just with nothing in it";
    EXPECT_EQ(reports[1].itemCount, 2) << "the category after it must still be walked";
    EXPECT_EQ(baseline.values.size(), 2u);
}

TEST(AmdSmiFabricTelemetryDiff, EmptySampleReportsNoCategories)
{
    FakeSample                    sample; // every dataset pointer left null
    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;

    EXPECT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 0);
    EXPECT_TRUE(baseline.values.empty());
}

// ---------------------------------------------------------------------------
// Top-mover selection
//
// A category carries thousands of counters and an active fabric moves hundreds
// per interval, so a report keeps only the largest. Dropping the wrong ones
// would quietly hide the signal the feature exists to surface.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricTelemetryMovers, LargestDeltasAreKeptInDescendingOrder)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{0, 0, 0, 0, 0}});

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    // Five counters move by different amounts; only the top three may be reported.
    const uint64_t deltas[] = {7, 400, 30, 9000, 50};
    for(size_t i = 0; i < 5; i++)
        sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0, i, deltas[i]);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 101);

    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);
    EXPECT_EQ(reports[0].changedCount, 5) << "every mover is still counted";
    ASSERT_EQ(reports[0].moverCount, kAmdSmiFabricTelemetryTopMovers);
    EXPECT_EQ(reports[0].movers[0].delta, 9000u);
    EXPECT_EQ(reports[0].movers[1].delta, 400u);
    EXPECT_EQ(reports[0].movers[2].delta, 50u);
}

TEST(AmdSmiFabricTelemetryMovers, CandidateSmallerThanEveryTrackedMoverIsRejected)
{
    amdsmiFabricTelemetryMover top[kAmdSmiFabricTelemetryTopMovers] = {};

    for(uint64_t delta : {100u, 90u, 80u})
    {
        amdsmiFabricTelemetryMover mover = {"big", {}, delta};
        amdSmiFabricTelemetryRecordMover(top, kAmdSmiFabricTelemetryTopMovers, mover);
    }

    amdsmiFabricTelemetryMover small = {"small", {}, 1};
    amdSmiFabricTelemetryRecordMover(top, kAmdSmiFabricTelemetryTopMovers, small);

    EXPECT_EQ(top[0].delta, 100u);
    EXPECT_EQ(top[1].delta, 90u);
    EXPECT_EQ(top[2].delta, 80u);
    EXPECT_STRNE(top[2].name, "small");
}

TEST(AmdSmiFabricTelemetryMovers, CandidateDisplacesTheSmallestTrackedMover)
{
    amdsmiFabricTelemetryMover top[kAmdSmiFabricTelemetryTopMovers] = {};

    for(uint64_t delta : {100u, 90u, 80u})
    {
        amdsmiFabricTelemetryMover mover = {"old", {}, delta};
        amdSmiFabricTelemetryRecordMover(top, kAmdSmiFabricTelemetryTopMovers, mover);
    }

    amdsmiFabricTelemetryMover middle = {"new", {}, 95};
    amdSmiFabricTelemetryRecordMover(top, kAmdSmiFabricTelemetryTopMovers, middle);

    EXPECT_EQ(top[0].delta, 100u);
    EXPECT_EQ(top[1].delta, 95u);
    EXPECT_STREQ(top[1].name, "new");
    EXPECT_EQ(top[2].delta, 90u) << "the smallest tracked mover is the one dropped";
}

// ---------------------------------------------------------------------------
// Instance labels
//
// amd_smi instance labels are a fixed 32-byte array that need not be
// NUL-terminated, so handing one straight to a %s would read past the field.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricTelemetryLabel, FullyOccupiedLabelIsTerminated)
{
    char raw[kAmdSmiFabricTelemetryLabelSize];
    memset(raw, 'x', sizeof(raw)); // no NUL anywhere in the field

    char copy[kAmdSmiFabricTelemetryLabelSize + 1];
    amdSmiFabricTelemetryCopyLabel(copy, raw);

    EXPECT_EQ(strlen(copy), kAmdSmiFabricTelemetryLabelSize);
    EXPECT_EQ(copy[kAmdSmiFabricTelemetryLabelSize], '\0');
}

TEST(AmdSmiFabricTelemetryLabel, UnterminatedInstanceLabelIsSafeToReport)
{
    FakeSample sample;
    sample.AddCategory(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 100, {{0}});

    char raw[kAmdSmiFabricTelemetryLabelSize];
    memset(raw, 'n', sizeof(raw));
    sample.SetLabel(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0, raw, sizeof(raw));

    amdsmiFabricTelemetryBaseline baseline{};
    Reports                       reports;
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    sample.SetValue(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 0, 0, 5);
    sample.SetGeneration(AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE, 101);
    ASSERT_EQ(amdSmiFabricTelemetryDiff(sample.Get(), &baseline, FakeTelemName, reports), 1);

    ASSERT_EQ(reports[0].moverCount, 1);
    EXPECT_EQ(strlen(reports[0].movers[0].instance), kAmdSmiFabricTelemetryLabelSize);
}

} // namespace RcclUnitTesting
