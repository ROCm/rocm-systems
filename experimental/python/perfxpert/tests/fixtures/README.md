# PerfXpert Fixture Policy

PerfXpert keeps only compact SQLite fixtures in git. Large captured traces
should be stored as release artifacts, Git LFS objects, or regenerated locally
for integration testing.

The integration tests already skip optional fixtures that are absent from
`tests/fixtures`, so heavyweight traces can be restored locally without making
normal source checkouts and source distributions carry the binary payload.

Current budget:

- No tracked `.db` fixture may exceed 10 MiB.
- The total tracked `.db` fixture budget is 32 MiB.
- New `.db` fixtures are marked for Git LFS by this directory's
  `.gitattributes`.
