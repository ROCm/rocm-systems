---
name: code-quality
description: "Keeps new and modified code free of the smells in catalog.md. Use while writing or editing code, before the change is finished. Check every Smell index row and shape the code with the catalog technique so the smell is never introduced. Do not produce a smell report."
---

# Code Quality

Use this while writing the code. A scan-and-report after the code exists is
too late. `catalog.md` is the constraint, not a findings list.

## While writing

1. Read the **Smell index** in `catalog.md`.
2. Write the change so that it does not match any Part 1 detection criteria.
   Check every index row, from Long Method through Incomplete Library Class.
   Do not skip a row because it looks unlikely.
3. When the code you are about to write would match a row, stop and open that
   smell's Part 1 section, then the first technique in its **Fixes** list.
   If that name has no `####` heading, use the short description under the
   smell index. If the technique section says to leave the code alone and
   that exception applies, use the next technique in the row.
4. Write the code by following every numbered **Steps** entry of the chosen
   technique. The before/after example only shows the shape. Do not write the
   smelly version first and refactor it afterward.
5. Before finishing, check the written code against every Smell index row
   again. The check is incomplete if any row was not considered. If one still
   matches, rewrite that part with the next technique in the row.

Do not print a smell table, a found/not-found list, or a refactoring report.
The finished code is the result. Mention a catalog exception only when it is
the reason a listed technique was not used.

## Project constraints

- C++20. Follow `cpp-conventions.md` while writing.
- Keep linkage unchanged unless the chosen technique requires a new type in
  another target; then follow `cmake-conventions.md`.
- Run the affected unit tests when a build already exists.
