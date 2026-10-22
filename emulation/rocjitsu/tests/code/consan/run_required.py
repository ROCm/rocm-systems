# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check physical registrations and run required ConSan coverage without skips."""

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
        'ConSanPrepare.EntryNamesMustBeUsableByTheShippedJsonAnalyzer',
        'ConSanInput.MissingEntryAndAdditionalKernelHaveDistinctDiagnostics',
        'ConSanSimProductionE2E.gfx1250.wave32.assembly.batch-barrier-and',
        'ConSanSimProductionE2E.gfx1250.wave32.assembly.batch-barrier-or',
        'ConSanHsaPlugin.RejectedBatchNotifiesBarrierWaitersOfFailure',
        'ConSanHsaPlugin.BarrierOnlyBatchPreservesOrderAndDoesNotFailCapture',
        'ConSanHsaPlugin.RejectedUnsupportedPacketNotifiesItsNeighboringBarrierOfFailure',
        'ConSanHsaPlugin.BarrierBatchWithoutCompletionSignalsStillPassesThrough',
        'ConSanHsaPlugin.RejectedBatchPreservesPendingWorkWhenNotifyingBarrierFailure',
        'ConSanHsaPlugin.RejectedBarrierCannotOverwritePendingDispatchCompletion',
        'ConSanHsaPlugin.BarriersAfterADispatchRejectionReceiveFailureNotifications',
        'ConSanSimProductionE2E.gfx1250.wave32.assembly.late-barrier-and',
        'ConSanSimProductionE2E.gfx1250.wave32.assembly.late-barrier-or',
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


def physical_names():
    names = set()
    for gpu, wave in MODES:
        kind = 'assembly' if gpu == 'gfx1250' else 'hip'
        prefix = f'{gpu}.wave{wave}'
        for case in ('race', 'disjoint', 'read_read', 'partial'):
            names.add(f'ConSanPhysicalE2E.{prefix}.{kind}.{case}')
        names.add(f'ConSanPhysicalE2E.{prefix}.assembly.high_lane')
        for case in (
            'disjoint',
            'batch-barrier-and',
            'batch-barrier-or',
            'late-barrier-and',
            'late-barrier-or',
        ):
            names.add(f'ConSanPhysicalProductionE2E.{prefix}.{kind}.{case}')
    return names


def validate_physical_registrations(tests):
    available = {test['name']: test for test in tests}
    expected = physical_names()
    missing = expected - available.keys()
    if missing:
        raise ValueError(
            'Physical ConSan registrations missing:\n' + '\n'.join(sorted(missing))
        )
    for name in sorted(expected):
        _, gpu, wave, kind, scenario = name.split('.')
        wave_size = int(wave.removeprefix('wave'))
        test = available[name]
        properties = {
            item['name']: item['value'] for item in test.get('properties', [])
        }
        labels = properties.get('LABELS', [])
        if (
            'physical-gpu' not in labels
            or 'simulator' in labels
            or properties.get('DISABLED', False)
            or properties.get('SKIP_RETURN_CODE') != 77
            or properties.get('RESOURCE_LOCK') != [f'consan-{gpu}']
        ):
            raise ValueError(f'{name}: invalid physical execution properties')
        command = test.get('command', [])
        if '--launcher' in command or '--require-agent' in command:
            raise ValueError(f'{name}: physical test requests simulator execution')
        options = {
            '--target': gpu,
            '--wave-size': str(wave_size),
            '--fixture-kind': kind,
            '--threads': str(
                3 * wave_size + 1 if scenario == 'partial' else 2 * wave_size
            ),
            '--case': 'disjoint' if scenario == 'partial' else scenario,
        }
        for option, value in options.items():
            if (
                option not in command
                or command.index(option) + 1 >= len(command)
                or command[command.index(option) + 1] != value
            ):
                raise ValueError(f'{name}: expected {option} {value}')


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
    registrations = subprocess.run(
        [
            'ctest',
            '--test-dir',
            str(args.build),
            '-R',
            '^ConSan',
            '--show-only=json-v1',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    (args.output / 'registrations.json').write_text(registrations.stdout)
    validate_physical_registrations(json.loads(registrations.stdout)['tests'])
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
