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

```text
--pc-sampling --pc-sampling-method stochastic
```

Options that belong to one feature share its name as a prefix even when they can
be used on their own, such as `--roofline-device` and `--roofline-sort`.

Frequently used options get a short alias: a single lowercase letter with a
single dash, such as `-v`.

Options are named for what they enable, not what they disable: use
`--roofline`, not `--no-roofline`.

## Renaming and Deprecation

A renamed option must keep its old name working for now. The old name:

- works the same as the new name, so keep the same `dest`
- is listed in the help as a deprecated alias of the new name
- prints a warning that names the new option whenever it is used

An option deprecated without a rename starts its help with `DEPRECATED:` and what
to do instead, and also prints a warning when used.

Every rename also needs these updates:

- add the new name to `CHANGELOG.md` under "Changed"
- add the old name to `CHANGELOG.md` under "Upcoming changes", since it will be removed later
- use the new name in the docs, tests, and `skills/`
- check the output of `rocprof-compute profile --help` and `rocprof-compute analyze --help`

## Experimental Options

Options behind `--experimental` follow the same rules as other options. While an
option is experimental, it may be renamed or removed without keeping the old name
working, but the change still needs a `CHANGELOG.md` entry. When the feature
becomes generally available, follow
[Adding Experimental Features](../../CONTRIBUTING.md#adding-experimental-features).

## Arguments

To pass several values, list them after the flag, separated by spaces:

```text
--roofline-data-types FP16 FP32 FP64
```

Don't repeat the flag: `-b 2 -b 5` keeps only `5`. Use `-b 2 5` instead.

The exception is analyze mode with several workloads. Repeat `-p` once per
workload, and repeat `-k` or `-d` to pick kernels or dispatches for each workload
in the same order:

```text
analyze -p run1 -k 0 1 -p run2 -k 3
```

This picks kernels 0 and 1 from `run1` and kernel 3 from `run2`. A single `-k` or
`-d` applies to every workload.

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
--option <arg>   Description of the option (Default: value).
                 Values: <value1>, <value2>, ..., <valueN>
```
