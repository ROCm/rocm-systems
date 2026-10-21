# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run required ConSan coverage, rejecting absent registrations and skipped cases."""

import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

MODES = (
    ('gfx942', 64),
    ('gfx950', 64),
    ('gfx1100', 32),
    ('gfx1100', 64),
    ('gfx1201', 32),
    ('gfx1201', 64),
    ('gfx1250', 32),
)


def required_names():
    names = {
        'ConSanRuntime.SymbolBoundary',
        'ConSanSimProductionE2E.gfx1250.wave32.assembly.missing-statistics',
        'ConSanContract.TargetWaveFactsRoundTripAndRowsFollowActualGeometry',
        'ConSanAnalysis.Wave64HighLanesAndFullMasksRetainExactLanePairCounts',
        'ConSanAnalysis.InvalidUtf8EntryNamesAreAnalysisInputErrorsAfterBinaryRoundTrip',
        'ConSanAnalysis.Utf8ScalarBoundariesRetainTheirBytesInJson',
        'ConSanHsaPluginDeathTest.FinishDuringFreezePreventsWorkerStartupAndUnloadsNormally',
        'ConSanHsaPluginDeathTest.DestroyDuringFreezeInvalidatesTheAdmittedModule',
        'ConSanHsaPlugin.RejectedBatchNotifiesEveryDiscardedDispatch',
        'ConSanHsaPlugin.RejectedBatchPreservesSignalsForAlreadyPendingWork',
        'ConSanHsaPlugin.RejectedBatchDoesNotTreatBarrierCompletionAsDispatchCompletion',
        'ConSanHsaPlugin.RejectedNonDispatchPacketDoesNotModifyATrackedSignal',
        'ConSanHsaPlugin.IndividualBarriersStillPassThroughWithoutDispatchAdmission',
    }
    for gpu, wave in MODES:
        suffix = f'{gpu}Wave{wave}'
        for case in (
            'EquivalentMetadataEnvironmentsPreserveTargetAndAbiValidation',
            'AssemblyLoweringPreservesConcreteIdentityAndLoadableImages',
            'PreparedCdnaDescriptorMustAllocateItsDeclaredOrdinarySgprs',
            'ProbeScalarDependenciesHaveTargetRequiredGuards',
            'HipWorkloadsMustContainQualifiedNativeLdsSites',
            'GdsAndAccumulatorAliasingAreRejectedAtTheirActualBoundaries',
        ):
            names.add(f'ConSanTargets/ConSanTargetStatic.{case}/{suffix}')
        for case in (
            'AssemblyAliasPreservesAllGuestRegistersAndPublishesBothWaves',
            'HighLaneMasksAndEmptyExecHaveRealPublicationEvidence',
            'ThreeWavesAndPartialTailUseDistinctRows',
        ):
            names.add(f'ConSanTargets/ConSanTargetSim.{case}/{suffix}')
        names.add(f'ConSanSimE2E.{gpu}.wave{wave}.assembly.high_lane')
        kind = 'assembly' if gpu == 'gfx1250' else 'hip'
        names.add(f'ConSanSimProductionE2E.{gpu}.wave{wave}.{kind}.disjoint')
        for case in ('race', 'disjoint', 'read_read'):
            names.add(f'ConSanSimE2E.{gpu}.wave{wave}.{kind}.{case}')
        if kind == 'hip':
            names.add(
                f'ConSanTargets/ConSanHipSim.CompiledHipRaceDisjointAndReadReadExecuteTheirNativeSites/{suffix}'
            )
    return names


def validate_selected(tests):
    missing = required_names() - {test['name'] for test in tests}
    if missing:
        raise ValueError(
            'Required ConSan registrations missing:\n' + '\n'.join(sorted(missing))
        )


def validate_results(root):
    cases = root.findall('.//testcase')
    missing = required_names() - {case.attrib['name'] for case in cases}
    skipped = [
        case.attrib['name'] for case in cases if case.find('skipped') is not None
    ]
    if missing or skipped:
        raise ValueError(
            f'Required coverage missing: {sorted(missing)}; skipped: {skipped}'
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = [
        'ctest',
        '--test-dir',
        str(args.build),
        '-R',
        '^ConSan',
        '-LE',
        'physical-gpu',
        '--no-tests=error',
    ]
    listed = subprocess.run(
        [*selection, '--show-only=json-v1'], capture_output=True, text=True, check=True
    )
    (args.output / 'selection.json').write_text(listed.stdout)
    validate_selected(json.loads(listed.stdout)['tests'])
    junit = (args.output / 'results.xml').resolve()
    result = subprocess.run(
        [
            *selection,
            '--output-on-failure',
            '-j',
            str(args.jobs),
            '--output-junit',
            str(junit),
        ],
        check=False,
    )
    if result.returncode:
        return result.returncode
    validate_results(ET.parse(junit).getroot())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
