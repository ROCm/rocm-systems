# HRR per-API playback matrix

Every HIP API HRR knows about has a declared replay verdict here and something
that checks it. The Catch2 tests in `../../tests/integration` do the checking;
this directory holds the declarations they compile against.

Each API's class is derived from HRR's own generator rather than transcribed,
so "faithfully replayed" is not a number maintained by hand: an upstream
reclassification surfaces as a hard failure with a diff instead of a silently
shrinking test matrix.

## The two stages

```
gen_hrr_api_args.py  ->  derive_manifest.py  ->  api_classes.json
hrr_playback.cpp     ->                              |
                                                     v
api_matrix.yaml      ->  check_matrix.py --emit-cxx  ->  hrr_api_matrix_expectations.h
```

`api_classes.json` is **derived**: one row per API with its capture class,
replay class and mechanically-detected payload-loss shape, read straight out of
the generator's `MANUAL_PLAYBACK_APIS` / `NOOP_PLAYBACK_APIS` /
`ERROR_STUB_PLAYBACK_APIS` sets plus `is_special()` in `hrr_playback.cpp`.

`api_matrix.yaml` is **authored**: the tier definitions, coverage floors,
per-tier workload lists, and the rationale for every API the matrix admits it
cannot reach.

## Regenerating the header

`hrr_api_matrix_expectations.h` is checked in, so the build never runs these
scripts. `api_classes.json` is not checked in, so derive it once per clone
(see below), then after changing `api_matrix.yaml`:

```sh
./check_matrix.py --emit-cxx      # writes the header in tests/integration
./check_matrix.py --summary       # tier table, runs nothing
```

Adding a workload to a tier means adding its Catch2 test name to that tier's
list in `api_matrix.yaml` and regenerating. Skipping the regeneration leaves
the tests compiling against the old tier lists, and the new coverage simply
does not count.

That is worth guarding, because the failure is quiet: the workload still
compiles and still passes when invoked by name, and only the coverage count
moves. This reads the two sides against each other, with no GPU and no derived
manifest:

```sh
./check_matrix.py --check-workloads
```

It fails when a tier names a workload nothing defines (a rename or a typo), and
when `hrr_api_matrix_workload_test.cc` or `hrr_spt_workload_test.cc` defines a
workload no tier names. Workloads in the other test files are driven by their
own roundtrip tests and are not expected to appear in a tier.

## Deriving the manifest

`api_classes.json` is deliberately not committed: a second copy of the
classification would be free to go stale against the generator it came from.
Derive it once per clone, and again whenever the generator or the playback
driver changes:

```sh
./derive_manifest.py              # writes api_classes.json
```

It asserts the counts in `api_classes_baseline.json` and **exits non-zero when
they drift**. That failure is the feature. Read the delta before refreshing it,
because the matrix was authored against the old classification and an API may
now have an expected verdict that no longer matches reality:

```sh
./derive_manifest.py --update-baseline    # only after reviewing the delta
```

## Dependencies

Python 3.8+ and PyYAML. No ROCm, no GPU and no build are required; these run on
the host. PyYAML is needed only by the two scripts above, which are developer
tools rather than build steps, so it is not a build dependency of HRR.

## Running the matrix against hardware

Generating the header is the host-side half. Actually capturing and replaying
each tier is a GPU job driven by the Catch2 tests
(`Unit_HRR_ApiMatrix_T*_Roundtrip`), which write per-tier observations to
`$HRR_MATRIX_RESULTS_DIR`. Turn those into a coverage report with:

```sh
./check_matrix.py --results DIR
```

The containerised sweep that builds, runs every tier and collects the report
lives in aim-labs under `scenarios/prep/hrr-api-matrix/`.
