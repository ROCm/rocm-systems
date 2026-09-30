# CLI Options and Arguments

> Rules for adding or changing command-line options in
> [`src/argparser.py`](../../src/argparser.py) and any other user-facing CLI.

---

## Filtering

Filter options match with **glob patterns** by default (`fnmatch`), not regex.
Glob covers most filtering cases and is much easier for users to write.

```console
rocprof-compute --select-kernel '*my-kernel*'
```

Regex may be offered **in addition** to glob where it is genuinely needed, never
as the replacement.

## Naming

An option that is only valid together with another option must be named with
that option as a prefix.

```console
--roofline --roofline-bench-only
```

Frequently used options get a short alias: a single lowercase letter with a
single dash, such as `-v`.

User-facing options are named for what they enable, not what they disable: use
`--roofline`, not `--no-roofline`. Debug and developer options may use disabling
names since customers rarely touch them. `--no-roof` and `--no-native-tool` are
such exceptions; mark them as "(advanced)" in their help.

## Renaming and Deprecation

A renamed option must keep its old name working for now.

To do this, add the old name with `add_deprecated_alias`:

```python
new_option = group.add_argument("--roofline-device", dest="device", ...)
add_deprecated_alias(group, "--device", new_option)
```

The old name then:

- still works the same as the new name
- does not show up in `--help`
- prints a warning that tells the user to use the new name

Keep the same `dest` as before, so the rest of the code does not need to change.

Every rename also needs these updates:

- add the new name to `CHANGELOG.md` under "Changed"
- add the old name to `CHANGELOG.md` under "Upcoming changes", since it will be removed later
- use the new name in the docs, tests, and `skills/`

## Arguments

A list of values is passed as one comma-separated argument. Use
`action=CommaListAction` (with `item_type` / `item_choices` for per-value
conversion and validation); it also accepts space-separated values.

```console
--roofline-data-types FP16,FP32,FP64
```

## Help Messages

Always set `metavar` so the help shows what the option takes. Never leave it
empty (`metavar=""`).

- If the value is required, use `metavar="<arg>"`.
- If the value is optional (`nargs="?"` or `nargs="*"`), use `metavar="arg"`.
  The help adds the `[]` automatically.

`CliHelpFormatter` adds `...` to options that take a list, so `<args>` shows as
`<args>...` and `args` shows as `[args]...`.

For an option that takes an argument, the help message follows these rules:

| Case | Notation |
|------|----------|
| Required argument | `<arg>` |
| Optional argument | `[arg]` |
| Array of required arguments | `<args>...` |
| Array of optional arguments | `[args]...` |

An argument with a fixed set of accepted values lists them on a line below the
option description, starting with `Values: `. An argument with a default states
it at the end of the description as `(Default: value)`.

```text
--roofline-data-types <types>...   Selects data <type>s to present in roofline (Default: FP32).
                                    Values: FP4, FP6, FP8, FP16, BF16, FP32, FP64, I8, I32, I64.
--list-metrics [arch]              List available metrics for specified GPU [arch] (Default: current GPU arch).
                                    Values: gfx908, gfx90a, gfx940, gfx941, gfx942, gfx950, gfx1150, gfx1151, gfx1152, gfx1153, gfx1250
```
