# CLI flag prototype results

The prototype implements the [accepted architecture](cli-flags-rfc.md) in
[`lib/cli`](../lib/cli), across all six native command-line tools. It supports
instance-owned typed values, generic string reads and updates, aliases, nested
commands, grouped automatic help, idempotent registration by flag address, and
existing result/diagnostic conventions.
It remains local experimental code, including the main launcher migration; it
does not define a stable production plugin ABI.

## Try it

From the `rocm-systems` repository root, with CMake, Ninja, a C++20 compiler,
GoogleTest, and Python available:

```sh
cmake -S emulation/rocjitsu/lib/cli -B ../build-cli -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build ../build-cli
ctest --test-dir ../build-cli -j4 --output-on-failure

../build-cli/tests/rj_cli_probe run -l 128 input.bin
../build-cli/tests/rj_cli_probe run --help
../build-cli/tests/rj_cli_probe --plugin \
  ../build-cli/tests/librocjitsu_cli_plugin_fixture.so \
  plugin inspect -f=json input.bin
```

The plugin example uses a required leading `--plugin PATH` bootstrap prefix.
The host loads all requested modules before the single normal parse. Discovery
does not guess the arity of unknown plugin flags. This is an example integration
policy, not a requirement of the parser library. Plugin tests require Linux.

The ordinary RocJITsu CMake build also exposes `rj_decode_benchmark`,
`rocjitsu_cli_test`, `rocjitsu_benchmark_cli_test`, `rocjitsu_cli_plugin_test`, and
`rj_cli_probe`. Run their focused tests with:

```sh
cmake --build ../build --target rj_decode_benchmark rocjitsu_cli_test \
  rocjitsu_benchmark_cli_test rocjitsu_cli_plugin_test rj_cli_probe
ctest --test-dir ../build -j4 --output-on-failure -R '^cli\.'
```

No ROCm SDK or GPU is needed for these targets. The standalone build avoids ISA
generation entirely. The full build's process test creates a minimal AMDGPU ELF
and runs the actual decoder, so it needs no external code-object fixture.

## What validation covers

- Core tests exercise scoped lookup, alias collisions, negative/empty/overflowing
  values, separators, positional ordering, repeated parses, independent trees,
  owned strings, stable registration handles, destruction, and failures.
- Custom typed flags enforce constraints through both typed setters and
  generic updates. The benchmark uses a typed decoder-target flag and constrained
  count flags rather than converting a string map after parsing.
- A loaded plugin registers nested commands and a custom enum flag. Its flag
  remains readable and writable after the loader's handle and the entire command
  tree are destroyed. The final flag reference releases the library owner.
- The executable probe exercises help, nested plugin flags, aliases, positional
  forwarding, registration failures, and process exit codes.
- The benchmark process test covers old argument spellings, aliases, equals
  syntax, actual decoding, help, malformed input, and target mismatches.
- A deterministic 4,000-case test checks integer round trips, arbitrary string
  storage, and malformed argument sequences.

The current migration passes 37 focused CTest entries in the native build,
including the DBT/pretranslation smoke suite, real local and daemon launcher
handoffs, fuzz replay/seeds, IP discovery, and plugin loading/unloading. Standalone
GCC and Clang AddressSanitizer/UndefinedBehaviorSanitizer builds each pass 28 tests,
with leak detection enabled for the latter. The sanitizer run covers the shared
library and its standalone consumers; the full launcher/translator runs are native.
VFIO transport was disabled in this build, and an AFL-instrumented fuzz loop was
not exercised. These are local correctness checks, not performance measurements
or hosted CI.

## Broader native CLI migration

| Consumer | Coverage and design pressure |
| --- | --- |
| `rj_decode_benchmark` | Typed target/counts, aliases, actual decoding |
| `rj_dbt_translate` / `rocjitsu_translate_cli` | Typed target/revision/output choices, decimal/hex bounds, required and omitted values, existing translation smoke suite |
| `rj_pretranslate` | Store paths, switches, multiple files, real store write/reuse/force, rejected inputs and I/O failures |
| `rj-ip-discovery` | Required positional arguments, generated table written to disk, leading-dash output path |
| `rj_decode_fuzz` | Mutually exclusive modes, target aliases, replay JSON and generated seeds, preserved model-only dependency boundary |
| `rocjitsu` | Optional JSON overrides, last-value semantics, mandatory separator, exact application argv forwarding, local and daemon config handoff |

All native flag-scanning loops now use the shared library. Shell/Python wrappers
and test-framework argument processing retain their existing language-specific
parsers. Application dispatch, backend operations, and cross-option checks stay
with their consumers. The prototype plugin bootstrap prefix remains an explicit
example of loading before registration and parsing.

Build and run the broader checks from the repository root:

```sh
cmake --build ../build --target rocjitsu_shared rocjitsu_bin rj_pretranslate \
  rj_ip_discovery rj_decode_fuzz rj_dbt_translate_smoke_test \
  rocjitsu_launch_flags_test rocjitsu_cli_test rocjitsu_benchmark_cli_test \
  rocjitsu_cli_plugin_test rj_cli_probe
ctest --test-dir ../build -j4 --output-on-failure \
  -R '^(cli\.|RjDbtTranslate.Smoke|DecodeFuzzCliTest|DecodeFuzzModelOnly.SymbolBoundary|RocjitsuWaitCheckingCliTest)'
```

The launcher tests need the matching `rocjitsu_shared` runtime built, but these
focused tests need no GPU or ROCm SDK. The ELF fixture for pretranslation contains
a kernel descriptor; a minimal descriptorless ELF only exercises domain rejection.

## Findings and implementation choices

**One shared flag replaces the wrapper and binding.** Metadata, conversion, and
the typed value live together. Callers register `shared_ptr<Flag>` and retain a
concrete typed handle; lookup returns an owning handle too. Metadata survives the
tree and cannot be reassigned. Sharing an object across commands shares its
spellings and value intentionally. Assignments apply in input order, including repeated aliases and shared objects
in parent and child scopes.

**Registration identity is distinct from argument repetition.** Re-registering
the same address before parsing succeeds without changing its group or adding an
entry. A different object claiming the same canonical name or alias emits a
diagnostic, including across groups or inside a batch. A batch validates all new
entries before insertion. Repeated scalar arguments are assignments, not
registrations, and use the last successful value.

**Groups supply automatic help without another namespace.** `add_flags` takes a
common description and a batch of shared flags. The description becomes a section
heading; every flag's spellings and individual description come from its metadata.
The benchmark uses a `Decoder settings` section, the probe uses `Run settings`,
and the loaded plugin contributes `Output formatting`. Empty and entirely redundant
batches do not add headings. Adjacent unnamed registrations share `Options`;
section order follows registration order.

**Keep values live through execution.** The benchmark no longer has a separate
`DecodeBenchmarkOptions` snapshot or parse adapter. Its entry point registers,
parses, handles help/positionals, and executes using the original typed flags.
Small integer conversion functions also replace temporary numeric flag objects
inside custom validators. These simplifications are implemented, not just proposed.


**Plugin lifetime needs more than a shared flag pointer.** A plugin-defined
destructor and a plugin-created shared-pointer control block can still execute
code from the module. The fixture uses a host-created deleter that retains the
library until deletion returns. Putting the last library reference inside the
plugin object itself can unload code while destruction is still executing.
[`plugin_host.h`](../lib/cli/tests/plugin_host.h) demonstrates the lifetime guard;
it is not a proposed public loader ABI.

The stronger unload assertion caught a fixture problem: GCC-emitted GNU-unique
symbols kept the module loaded after `dlclose`, masking the lifetime test.
The fixture now suppresses those symbols in its own code and hides static-archive
exports, matching the simulator's archive-export policy. The test verifies actual
unload with `RTLD_NOLOAD` after the final flag reference is released.

**Freeze definitions, not values.** The first parse closes registration throughout
the supplied tree. Callbacks cannot add commands while parsing. Metadata stays
read-only, while typed and generic value updates continue to work. Commands and
flags retain stable addresses. Lookup and enumeration return shared flag handles;
section enumeration borrows read-only metadata until the next registration.

**Diagnostics need context.** `Flag::parse_and_set` prefixes a flag's conversion diagnostic
with its canonical spelling, and CLI parsing adds the command path. An erroneous
custom flag that returns failure without a message still receives a useful
fallback diagnostic. There is no recovery or rollback of earlier assignments.

**Help is application policy.** Each application explicitly registers a normal
boolean help flag and formats the selected command after a successful parse.
An unselected parent command is a successful parse result, letting the caller
choose help or a missing-command error. Unknown flags still fail even when help
is present. The benchmark now succeeds for `INPUT --help`; its previous caller
returned failure unless `--help` was the first argument.

**A required value may resemble a flag.** `--text --help` assigns the string
`--help`. The separator itself is reserved, so `--text --` reports a missing
value and `--text=--` assigns the literal separator. This supports negative numeric
values and paths beginning with dashes without guessing intent.

**Repeated scalars should use the last value.** Migration exposed existing launcher
coverage that deliberately repeats `--wait-checking`; the other handwritten
parsers also accepted repeated assignments. The initial prototype's rejection
would break these callers. Removing occurrence tracking preserves the behavior
and makes the parser smaller. A malformed later value still fails immediately.

**Omission cannot become a zero or false default.** The launcher stores optional
CPU-budget and wait-checking values, so absent overrides leave JSON untouched.
DBT's optional VGPR limit also stays absent; assigning zero unconditionally would
change the translator request. Regression tests distinguish omission, explicit
zero, bounds, and last-value updates. Optional flags display `unset` for inspection;
typed setters can clear them. Generic clearing remains a separate design question.

**Forwarded arguments need a separator policy.** `PositionalMode::AfterSeparator`
rejects bare launcher application arguments before `--`. `saw_separator` preserves
the distinction between no separator and an empty tail for mode validation. Owned
positionals are converted to a null-terminated argv at execution, preserving empty
strings, repeated separators, and flags that belong to the child process.

**Keep domain conversions local.** DBT preserves hexadecimal integers in its
bounded numeric adapter while the core numeric flags remain decimal. Its small
`ChoiceFlag<T>` owns an optional typed value and a spelling table. The launcher
uses its existing wait-checking enum. These concrete integrations provide evidence
for future helpers without broadening the core into a configuration framework.

**Complete parsing precedes help/version dispatch.** Invalid trailing arguments
now fail even after a help/version flag. Equals syntax, explicit boolean values,
and `--` are consistently available. IP discovery accepts a leading-dash output
path after `--`; launcher paths reject explicit empty strings. Generated help and
diagnostic wording change, while dispatch exit-code classes remain per-tool policy.

## Remaining adoption decisions

- Generic optional-value serialization and clearing still need an explicit contract.
  The migrated launcher and DBT already preserve omitted values using typed optionals;
  the core has no generic provenance or occurrence history.
- Define plugin discovery, binary compatibility, and serialization of settings
  across launcher/daemon process boundaries before migrating production plugins.
- The parser does not provide synchronization or automatically reconfigure a
  running subsystem when a stored value changes. Consumers choose when to read
  updated values and coordinate concurrent access.
- Richer help layout, command aliases, executable-name dispatch, and list-valued
  flags remain policy or follow-up decisions. Nested
  command dispatch itself is implemented and tested.
