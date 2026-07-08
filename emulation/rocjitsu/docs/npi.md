# Adding

This guide is not complete.

## Marking NPI tasks

Mark tasks that need to be done when introducing a new product with `\NPI` comment.

## Find npi tasks

```bash
grep -Hn '\NPI' $(git ls-files 'emulation/')
```

## An incomplete list of tasks that aren't marked inline.

- sync shared/machine-readable-isa using download.py
- update codegen
- add tests
- profit
