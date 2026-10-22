# RocJITsu CLI flags RFC

Status: architecture implemented in the local prototype and used by all six
native command-line tools; publication and production plugin ABI remain separate decisions. See
[prototype results](cli-flags-prototype.md) for validation and integration limits.

Use an explicitly constructed command tree and a small inheritance interface.
Each flag is one shared object containing immutable metadata and a strongly typed
value. Components register their flags in batches, with descriptions that become
sections in generated help.

## Requirements

- Explicit registration coordinated by the executable. Components and plugins
  define their own flags; no global/static flag state or automatic registration.
- Plugins register before parsing and use the same parser as built-in components.
- Recursive subcommands with their own flag namespaces and aliases.
- Strongly typed access plus generic lookup, enumeration, formatting, and string
  updates. A string update converts one value and may fail; it does not reparse argv.
- Existing `Result`, `FailureOr`, and `DiagnosticEmitter`; stop at the first
  failure, with no exception-based CLI control flow or recovery machinery.
- Parsing returns the selected command and owned positional arguments in order.
  The application controls dispatch, output, and exit codes.
- Automatic per-command help from metadata, including batches with a common
  section description. Groups affect presentation, not parsing scope.
- Different objects claiming the same name or alias produce a diagnostic.
  Registering the same object address again in that command is a successful no-op.
- Independent instances and unit-testable registration, parsing, updates, help,
  diagnostics, and lifetimes. Prefer simple classes and functions over a framework.

## Flag object and ownership

The following interface excerpts omit implementations and namespace qualifiers:

```cpp
struct FlagInfo {
  std::string name;
  std::vector<std::string> aliases;
  std::string description;
  ValueMode value_mode = ValueMode::Required; // Or Switch.
};

class Flag {
public:
  explicit Flag(FlagInfo info);
  virtual ~Flag() = default;
  Flag(const Flag&) = delete;
  Flag& operator=(const Flag&) = delete;
  const FlagInfo& info() const;
  virtual std::string format_value() const = 0;
  Result parse_and_set(std::string_view text, DiagnosticEmitter diagnostic);

protected:
  virtual Result parse_value(std::string_view text,
                             DiagnosticEmitter diagnostic) = 0;
};

class IntFlag final : public Flag {
public:
  explicit IntFlag(FlagInfo info, int64_t initial = 0);
  int64_t value() const;
  Result set_value(int64_t value, DiagnosticEmitter diagnostic = {});
  std::string format_value() const override;

private:
  Result parse_value(std::string_view text, DiagnosticEmitter diagnostic) override;
  int64_t value_;
};
```

Provide `BoolFlag`, `IntFlag`, `UIntFlag`, and `StringFlag`. Custom types derive
from `Flag`, implement conversion/formatting, and expose their own typed getter
and setter. Both update paths enforce the same applicable constraints. Conversion
checks the whole input before assignment; strings own their contents. Plain
`parse_int` and `parse_uint` helpers let custom numeric flags reuse conversion
without constructing temporary flags.

The nonvirtual `parse_and_set` adds the canonical name to conversion diagnostics,
including a fallback if a custom implementation fails silently. CLI parsing adds
the command path. An invalid value leaves that flag unchanged; earlier successful
assignments remain. Typed setters do not require a string conversion.

Commands and callers share the **whole flag**, eliminating the separate binding
object. A component keeps `shared_ptr<IntFlag>`; generic lookup returns
`shared_ptr<Flag>` to the same object. Metadata cannot be renamed or reassigned,
so lookup remains valid. A retained handle keeps metadata and value alive after
the tree is destroyed. Aliases refer to that same object.

Commands own child commands uniquely at stable addresses. The parse result's
command pointer borrows the tree. Sharing a flag across commands intentionally
shares its metadata and value; creating a new flag gives independent state.

## Registration, groups, and entry point

```cpp
struct FlagGroup {
  std::string description;
  std::vector<std::shared_ptr<Flag>> flags;
};

// Command interface excerpts:
FailureOr<Command*> add_command(std::string name, DiagnosticEmitter diagnostic);
Result add_flag(std::shared_ptr<Flag> flag, DiagnosticEmitter diagnostic);
Result add_flags(std::string description, std::vector<std::shared_ptr<Flag>> flags,
                 DiagnosticEmitter diagnostic);
std::shared_ptr<Flag> find_flag(std::string_view spelling) const;
std::vector<std::shared_ptr<Flag>> flags() const;
std::span<const FlagGroup> groups() const;

struct ParsedArguments {
  Command* command;
  std::vector<std::string> positional;
  bool saw_separator = false;
};
FailureOr<ParsedArguments> parse(Command& root,
                                std::span<const std::string_view> arguments,
                                DiagnosticEmitter diagnostic);
std::string format_help(const Command& command);
```

Each nonempty named batch starts a help section in registration order. Adjacent
unnamed additions share an `Options` section. A group is not a namespace: all
names and aliases are checked against every group in the command and the new
batch. Re-registering an existing object neither duplicates it nor moves its
section. Empty or entirely redundant batches add no section. A batch validates
before inserting its new entries. Duplicate command names still fail.

`flags()` enumerates each registered object once using owning handles. `groups()`
borrows read-only section metadata and may be invalidated by further registration.
The first parse seals registration throughout the tree, including no-op attempts;
values remain mutable afterward. Arguments exclude argv[0]. Parsing does not reset
values to defaults or execute a selected command.

```cpp
struct CodeFlags {
  std::shared_ptr<IntFlag> limit = std::make_shared<IntFlag>(
      FlagInfo{"--limit", {"-l", "--max-items"}, "Maximum item count"}, 64);
  std::shared_ptr<StringFlag> output = std::make_shared<StringFlag>(
      FlagInfo{"--output", {"-o"}, "Output path"}, "out.txt");
};

Result register_code_flags(Command& code, const CodeFlags& flags,
                           DiagnosticEmitter diagnostic) {
  return code.add_flags("Code inspection", {flags.limit, flags.output}, diagnostic);
}

Result run_cli(std::span<const std::string_view> arguments,
               DiagnosticEmitter diagnostic) {
  Command root{"rocjitsu"};
  CodeFlags flags;
  auto inspect = root.add_command("inspect", diagnostic);
  if (inspect.failed()) return Result::failure();
  auto code = inspect.value()->add_command("code", diagnostic);
  if (code.failed()) return Result::failure();
  if (register_code_flags(*code.value(), flags, diagnostic).failed())
    return Result::failure();
  // Register other components and loaded plugins here.
  auto parsed = parse(root, arguments, diagnostic);
  if (parsed.failed()) return Result::failure();
  return execute(*parsed.value().command, parsed.value().positional,
                 flags, diagnostic); // Application-defined dispatch.
}
```

For `rocjitsu inspect code -l 128 input.bin`, parsing selects `code`, stores `128`,
and returns `{"input.bin"}`. Formatting that command's help produces:

```text
Usage: rocjitsu inspect code [options] [--] [arguments...]

Code inspection:
  --limit, -l, --max-items VALUE
      Maximum item count
  --output, -o VALUE
      Output path
```

The application can register an ordinary boolean `--help` flag, then print
`format_help(*parsed.value().command)` after successful parsing. Missing required
positionals may be bypassed for help; unknown flags still fail. The formatter
performs no output or process control itself.

After parsing, use the original typed handles or generic lookup:

```cpp
if (flags.limit->set_value(256, diagnostic).failed()) return Result::failure();
auto flag = parsed.value().command->find_flag("--max-items"); // Known registration.
std::string text = flag->format_value(); // "256"
if (flag->parse_and_set("512", diagnostic).failed()) return Result::failure();
// flags.limit->value() is now 512.
```

Keep these live objects through execution instead of copying them into a parallel
options struct. Runtime consumers choose when to read updates; assignments do not
automatically reconfigure subsystems that previously copied a value. All access
requires external synchronization when used concurrently.

## Prototype syntax and boundaries

- Exact `--name=value` / `--name value` and explicitly registered short aliases;
  no abbreviations or bundled short options.
- Parent flags precede child selection; after descent only that command's flags
  are recognized. Leaf commands allow interleaved flags and positionals.
- `--` ends both flag and command recognition. `Command` can opt into
  `PositionalMode::AfterSeparator` for launcher interfaces that require it; the
  result records whether a separator was present, including an empty tail. Required values consume one token,
  including a dash-prefixed token, except the reserved separator itself;
  `--text=--` passes that literal value.
- Switches supply `true` without consuming another token; explicit `=false` works.
  Integers are decimal without a leading plus; booleans are `true`/`false`.
- Repeated scalar arguments apply in order, including aliases: the last successful
  value wins. This preserves existing tool behavior and removes occurrence tracking.
  A failed later assignment still stops parsing. This is separate from duplicate
  registration, which continues to diagnose distinct objects claiming a spelling.
- A parent with no child selection is a successful parse result for the application
  to handle. Command aliases, argv[0] dispatch, and list-valued flags are deferred.
- The host must load plugins before parsing and keep their code loaded until all
  plugin objects and destruction code finish. A shared flag alone cannot pin a
  dynamically loaded module. The tested host-side deleter retains the library
  through destruction; this is not a stable public plugin ABI.

## Justification and further simplification

The shared object removes the metadata wrapper/binding split and borrowed lookup
handles while retaining strong types and custom validation. One small inheritance
interface keeps state, conversion, and formatting together without paired
`std::function` callbacks or erased context pointers. Allocation and lookup speed
are not design constraints here.

Batch registration replaces repetitive per-flag error checks and supplies help
sections from the same definitions. Removing the benchmark's parsed-options
snapshot keeps a single source of values through execution. Numeric conversion
helpers reduce custom-flag boilerplate without adding a validator framework.

Keep the canonical name and aliases explicit for diagnostics. Broader migration
validated application-local adapters for bounded integers, optional configuration
overrides, and typed choices. DBT uses a small table-driven `ChoiceFlag<T>` and a
hex-capable bounded integer flag; the launcher's optional values preserve omitted
JSON overrides. These remain outside the core while their contracts differ.
Absent optional values format as `unset` for inspection and can be cleared through
typed setters; `unset` is not an accepted CLI token. Generic clearing/serialization
would require an explicit contract rather than overloading empty strings.

The core remains 468 lines including comments and blank lines (165 header, 303
implementation). Supporting the remaining tools added six lines overall: a
positional policy and separator result bit replaced duplicate-argument tracking.

The applications retain policy checks for required inputs, incompatible modes,
required target revisions, and configuration handoff. DBT assembles its existing
translator request from typed values at dispatch; this domain request is distinct
from a redundant parser-owned options cache. Numeric conversion and enum checking
remain testable independently of translation or process launch.

Intentional syntax differences from handwritten parsers: help and version follow
a successful complete parse, so unknown flags no longer disappear after `--help`;
all tools accept equals syntax and explicit boolean values; leading-dash positional
paths use `--`. The launcher still requires `--` before application arguments.
Its config and VFIO path flags reject explicit empty strings. Human-readable help
and diagnostics are generated consistently rather than preserving old wording.
Production plugin discovery, binary compatibility, generic absence handling, and
settings transport across launcher/daemon boundaries remain separate concerns;
the existing launcher override handoff is exercised without redesigning it.
