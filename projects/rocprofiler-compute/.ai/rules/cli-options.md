# CLI Options and Arguments

> Rules for adding or changing command-line options in
> [`src/argparser.py`](../../src/argparser.py) and any other user-facing CLI.

---

## Filtering

Filter options match with **glob patterns** by default (`fnmatch`), not regex.
Glob covers most filtering cases and is much easier for users to write.

```console
rocprof-compute --select-kernel '*my-kernel*'

Regex may be offered **in addition** to glob where it is genuinely needed, never
as the replacement.

## Naming

An option that is only valid together with another option must be named with
that option as a prefix.

```console
--roofline --roofline-bench-only
```

If an option already exists with the same meaning in another mode, keep its
name. Don't add a prefix. For example, `-b/--block` filters by block or metric
id in both `profile` and `analyze`.

Frequently used options get a short alias: a single lowercase letter with a
single dash, such as `-v`.

User-facing options are named for what they enable, not what they disable: use
`--roofline`, not `--no-roofline`. Debug and developer options may use disabling
names since customers rarely touch them.

## Arguments

A list of values is passed as one comma-separated argument.

```console
--example-ids 0,1,2
```

## Help Messages

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
--list-metrics [ids]...            List the blocks of the current GPU, or the metrics of the given block [ids].
```
