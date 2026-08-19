# CLI guidelines
## Arguments naming rules
- If option has argument then help message shall follow the following guidelines:
	- If argument is required it shall be reflected by argument name surrounded by `<>` in the help message.
	- If argument is optional it shall be reflected by argument name surrounded by `[]` in the help message.
	- If option accepts an array of arguments, then it shall be plural and have trailing `...`, such as `<arg>...` and `[arg]...`.
	- If argument has a range of values this range shall be listed below option description preceding by `Values: `.
	- If argument has a default value it shall be specified at the end of option description such as `(Default: value)`
```bash
--roofline-data-types <types>...   Selects data <type>s to present in roofline (Default: FP32).
									Values: FP4, FP6, FP8, FP16, BF16, FP32, FP64, I8, I32, I64.
--list-blocks [arch]               List all available blocks for analysis on specified GPU <arch> (Default: current GPU arch).
									Values: gfx908, gfx90a, gfx940, gfx941, gfx942, gfx950, gfx1151
```
- If option has a list of arguments, then this list shall be passed concatenated by comma.
- Frequently used commands shall have a short single lowercase letter alias with a single dash, ex `-v`.
- User-facing options shall have "enabling", not "disabling" notation. Ex. `--roofling` instead of `--no-roof`. Exception could be made for debug and developer options as they are expected to be rarely used by customers.
- If option `A` is a nested setting of option `B` (meaning setting `A` without `B` is invalid), then name of `B` shall be a prefix of `A`'s name.
```bash
# Example
tool --roofline --roofline-bench-only
```

## Data filtering rules
- Filtering shall be done by "glob pattern" by default. Regex while more powerful is also much more complicated and therefore is overkill for most of the scenarios. Regex support where applicable could be enabled in addition to usage of "glob pattern".
```bash
# Example
tool --select-kernel kernel-{debug,lts,rt}[0-9]*
```