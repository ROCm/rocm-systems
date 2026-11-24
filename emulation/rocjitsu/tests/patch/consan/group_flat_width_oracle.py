# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Expected retained sites and cross-wave findings for the Group-FLAT width fixtures."""

WIDTHS = {'half': 2, 'b64': 8, 'b128': 16}

FLAT_WIDTH_CASES = tuple(
    f'flat_{form}_{scenario}'
    for form in WIDTHS
    for scenario in ('values', 'overlap', 'adjacent')
)


def flat_width_pairs(case):
    if not case.endswith('_overlap'):
        return 0
    return 6 if case.startswith('flat_half_') else 2


def assert_flat_width_observations(decoded, case, wave_size, completed_status):
    _, form, scenario = case.split('_')
    values = scenario == 'values'
    half = form == 'half'
    width = WIDTHS[form]
    if half:
        # Native dword accesses initialize/check the unaffected neighboring bytes.
        half_roles = [(2, 2), (1, 2), (1, 2), (1, 2), (1, 2), (2, 2)]
        if values:
            roles = [(2, 4), half_roles[0], (1, 4), *half_roles[1:], (1, 4)]
        else:
            roles = [*half_roles, (2, 2)]
        flat_sites = {1, 3, 4, 5, 6, 7} if values else set(range(6))
    elif values:
        roles = [(2, width), (1, width), (2, width), (1, width)]
        flat_sites = {1, 2}
    else:
        roles = [(1, width), (2, width), (2, 4)]
        flat_sites = {0, 1}
    sites = decoded['sites']
    assert [(s['access'], s['width']) for s in sites] == roles
    assert {s['id'] for s in sites if s.get('origin') == 'group_flat'} == flat_sites
    assert all(sites[i]['confidence'] == 'exact' for i in flat_sites)

    (execution,) = decoded['executions']
    assert execution['grid_x'] == execution['workgroup_size_x'] == 2 * wave_size
    rows = execution['rows']
    assert len(rows) == 2
    for wave, row in enumerate(rows):
        assert row['wave'] == wave and row['serial'] == execution['serial'] == 1
        assert (
            row['status'] == completed_status and row['loss'] == row['final_phase'] == 0
        )
        assert not row['completions']
        if half and values:
            dword = 1012 + 8 * wave
            expected = [
                (i, dword + 2 if i in flat_sites else dword) for i in range(len(roles))
            ]
        elif values:
            expected = [(i, 1024 - 2 * width + wave * width) for i in range(len(roles))]
        elif wave == 0:
            offset = 1022 if half else 1024 - 2 * width
            expected = [(i, offset) for i in sorted(flat_sites)]
        elif half:
            expected = [(6, 1020 if scenario == 'adjacent' else 1022)]
        else:
            expected = [(2, 1024 - width - (0 if scenario == 'adjacent' else 4))]
        assert len(row['records']) == row['final_sequence'] == len(expected)
        for sequence, ((site, offset), record) in enumerate(
            zip(expected, row['records']), 1
        ):
            assert record['site'] == site and record['sequence'] == sequence
            assert (
                record['commit'] == 0x43534143
                and record['flags'] == 1
                and record['phase'] == 0
            )
            assert (record['offset'], record['width'], record['mask']) == (
                offset,
                roles[site][1],
                1 << (wave_size - 1),
            )


def assert_flat_width_findings(raw_sites, analyzed, case, wave_size):
    assert analyzed['ordered_pairs'] == analyzed['ordering_unresolved_pairs'] == 0
    assert (
        analyzed['completed_barrier_waves'] == analyzed['required_barrier_waves'] == 0
    )
    findings = analyzed['cross_wave_examples']
    if not case.endswith('_overlap'):
        assert not findings
        return
    form = case.split('_')[1]
    width = WIDTHS[form]
    count = 6 if form == 'half' else 2
    assert len(findings) == count
    expected_ranges = (
        (2, 1022, 2, 1022)
        if form == 'half'
        else (width, 1024 - 2 * width, 4, 1020 - width)
    )
    observed = set()
    for finding in findings:
        assert finding['ordering'] == 'unordered' and finding['lane_pairs'] == 1
        flat, peer = sorted(
            (finding['first'], finding['second']), key=lambda endpoint: endpoint['wave']
        )
        observed.add(flat['site'])
        assert flat['origin'] == 'group_flat' and flat['confidence'] == 'exact'
        assert (
            flat['width'],
            flat['offset'],
            peer['width'],
            peer['offset'],
        ) == expected_ranges
        assert peer['site'] == count and peer['access'] == 'write'
        for wave, endpoint in enumerate((flat, peer)):
            assert endpoint['wave'] == wave and endpoint['mask'] == 1 << (wave_size - 1)
            site = raw_sites[endpoint['site']]
            assert endpoint['original_offset'] == site['original_offset']
            assert endpoint['access'] == {1: 'read', 2: 'write'}[site['access']]
    assert observed == set(range(count))
