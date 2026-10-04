// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/mca/venice.hpp"

#include <optional>
#include <string_view>

namespace addc::mca::base::venice
{

namespace
{

struct AfidRange
{
    uint32_t corrected;
    uint32_t non_fatal;
    uint32_t fatal;
};

std::optional<AfidRange> afidRangeFromIpid(uint16_t hw_id,
                                           uint16_t mca_type) noexcept
{
    struct Entry
    {
        uint16_t hw_id;
        uint16_t mca_type;
        AfidRange range;
    };
    static constexpr Entry kTable[] = {
        {0xb0, 0x0, {20513, 22242, 23971}},
        {0xb0, 0x1, {20257, 21986, 23715}},
        {0xb0, 0x2, {20385, 22114, 23843}},
        {0xb0, 0x3, {20065, 21794, 23523}},
        {0xb0, 0x5, {20129, 21858, 23587}},
        {0xb0, 0x6, {20193, 21922, 23651}},
        {0xb0, 0x7, {20449, 22178, 23907}},
        {0x01, 0x1, {21538, 23267, 24996}},
        {0x01, 0x2, {21090, 22819, 24548}},
        {0x2e, 0x1, {21410, 23139, 24868}},
        {0x2e, 0x2, {20001, 21730, 23459}},
        {0x1e1, 0x0, {21282, 23011, 24740}},
        {0x5c, 0x0, {20770, 22499, 24228}},
        {0xbe, 0x0, {20577, 22306, 24035}},
        {0x96, 0x0, {21602, 23331, 25060}},
        {0x50, 0x0, {21346, 23075, 24804}},
        {0x18, 0x0, {21218, 22947, 24676}},
        {0x6c, 0x0, {21154, 22883, 24612}},
        {0xaa, 0x0, {21666, 23395, 25124}},
        {0x80, 0x0, {21474, 23203, 24932}},
        {0xfd, 0x0, {20834, 22563, 24292}},
        {0xf9, 0x0, {20642, 22371, 24100}},
        {0x259, 0x0, {20321, 22050, 23779}},
        {0xff, 0x2, {20706, 22435, 24164}},
        {0x1e0, 0x0, {20898, 22627, 24356}},
        {0x164, 0x0, {20962, 22691, 24420}},
        {0x157, 0x0, {21026, 22755, 24484}},
        {0x12,  0x0, {25765, 25829, 25893}},
        {0x46,  0x1, {25957, 26021, 26085}},
    };
    for (const auto& e : kTable)
    {
        if (e.hw_id == hw_id && e.mca_type == mca_type)
        {
            return e.range;
        }
    }
    return std::nullopt;
}

} // namespace

uint32_t decode_afid(McaRegisters regs)
{
    const auto range =
        afidRangeFromIpid(regs.ipid.hardware_id, regs.ipid.mca_type);
    if (!range)
    {
        return kAfidSentinel;
    }

    const auto ext = regs.status.error_code_ext;
    if (ext > 63)
    {
        return kAfidSentinel;
    }

    const auto sev = addc::mca::decode_severity(regs);
    uint32_t base = range->corrected;
    if (sev == "Fatal")
    {
        base = range->fatal;
    }
    else if (sev == "Non-Fatal")
    {
        base = range->non_fatal;
    }
    return base + ext;
}

McaResult decode(const BankCapture& bank)
{
    return decode_common(bank, decode_afid(bank.regs));
}

} // namespace addc::mca::base::venice
